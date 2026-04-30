import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    pass 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import matplotlib.lines as mlines
import heapq
import math
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, translate

# --- CONFIGURATION (PRODUCTION READY) ---
MAP_SIZE = 200

# 1. Controller Safety Settings
SAFE_MARGIN = 2.0  

# 2. Alignment Settings
XY_TOLERANCE = 0.1          
YAW_TOLERANCE = np.deg2rad(0.5) 

# Car Specs
CAR_L, CAR_W = 25.0, 18.0
WB = CAR_L * 0.75
MAX_STEER = np.deg2rad(25.0) 

SLOT_W, SLOT_D = 25.0, 40.0  

# Pathfinding Weights
SPEED = 5.0
COST_STEER = 2.0            
COST_GEAR_SWITCH = 10.0   
COST_STEER_CHANGE = 5.0   
H_WEIGHT = 4.0              

# Reeds-Shepp Constants
ANALYTIC_SHOT_DIST = 40.0   
RS_STEP_SIZE = 0.5          
MIN_TURN_RADIUS = WB / math.tan(MAX_STEER) * 1.05

MOTION_PRIMITIVES = [
    (SPEED, 0), 
    (SPEED, MAX_STEER), (SPEED, -MAX_STEER),             
    (SPEED, MAX_STEER*0.5), (SPEED, -MAX_STEER*0.5),     
    (-SPEED, 0), 
    (-SPEED, MAX_STEER), (-SPEED, -MAX_STEER),
    (-SPEED, MAX_STEER*0.5), (-SPEED, -MAX_STEER*0.5),   
]

# --- UTILS ---
def normalize_angle(angle): 
    return (angle + np.pi) % (2 * np.pi) - np.pi

def M(theta): return normalize_angle(theta)

def add_exit_btn(fig):
    ax_exit = plt.axes([0.89, 0.01, 0.1, 0.05])
    btn = Button(ax_exit, 'EXIT', color='#8b0000', hovercolor='#ff4d4d')
    btn.label.set_color('white')
    btn.label.set_fontsize(9)
    def on_exit(event):
        plt.close('all')
        sys.exit(0)
    btn.on_clicked(on_exit)
    return btn

def save_path_json(path, start_pose, target_slot_id, target_pose, filename="planned_path.json"):
    """
    Save planned path to a JSON file for the Raspberry Pi controller.
    Path points are stored in cm/rad with direction (+1 forward / -1 reverse).
    """
    data = {
        "version": 1,
        "generated_at_unix_s": time.time(),
        "map_size_cm": MAP_SIZE,
        "start": {"x_cm": float(start_pose[0]), "y_cm": float(start_pose[1]), "yaw_rad": float(start_pose[2])},
        "target": {
            "slot_id": int(target_slot_id),
            "x_cm": float(target_pose[0]),
            "y_cm": float(target_pose[1]),
            "yaw_rad": float(target_pose[2]),
        },
        "points": [
            {
                "x_cm": float(x),
                "y_cm": float(y),
                "yaw_rad": float(yaw),
                "direction": int(d),
                "steer_rad": float(steer),
            }
            for (x, y, yaw, d, steer) in path
        ],
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- REEDS-SHEPP SOLVER ---
class ReedsSheppPlanner:
    def __init__(self, step_size, min_r):
        self.step_size = step_size
        self.min_r = min_r

    def get_optimal_path(self, sx, sy, syaw, gx, gy, gyaw):
        dx, dy = gx - sx, gy - sy
        d = math.hypot(dx, dy)
        theta = M(math.atan2(dy, dx) - syaw)
        x_norm = d * math.cos(theta) / self.min_r
        y_norm = d * math.sin(theta) / self.min_r
        phi = M(gyaw - syaw)
        paths = self.generate_paths(x_norm, y_norm, phi)
        if not paths: return None
        best_path = min(paths, key=lambda p: abs(p.L))
        return self.generate_trajectory(best_path, sx, sy, syaw)

    class Path:
        def __init__(self, lengths, types, L):
            self.lengths = lengths; self.types = types; self.L = L

    def generate_paths(self, x, y, phi):
        paths = []
        paths.extend(self.csc(x, y, phi))
        paths.extend(self.ccc(x, y, phi))
        return paths

    def csc(self, x, y, phi):
        paths = []
        u, t = self.R(x - math.sin(phi), y - 1.0 + math.cos(phi))
        v = M(phi - t)
        if t >= 0 and u >= 0 and v >= 0: paths.append(self.Path([t, u, v], ['L','S','L'], t+u+v))
        u1, t1 = self.R(x + math.sin(phi), y - 1.0 - math.cos(phi))
        if u1**2 >= 4.0:
            u = math.sqrt(u1**2 - 4.0)
            theta = math.atan2(2.0, u)
            t = M(t1 + theta); v = M(t - phi)
            if t >= 0 and u >= 0 and v >= 0: paths.append(self.Path([t, u, v], ['L','S','R'], t+u+v))
        return paths

    def ccc(self, x, y, phi):
        paths = []
        u1, t1 = self.R(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if u1 <= 4.0:
            u = -2.0 * math.asin(0.25 * u1)
            t = M(t1 + 0.5 * u + np.pi)
            v = M(phi - t + u)
            if t >= 0 and u <= 0: paths.append(self.Path([t, u, v], ['L','R','L'], abs(t)+abs(u)+abs(v)))
        return paths

    def R(self, x, y): return math.hypot(x, y), math.atan2(y, x)

    def generate_trajectory(self, path, sx, sy, syaw):
        px, py, pyaw, pd, ps = [sx], [sy], [syaw], [1], [0.0]
        step = self.step_size
        steer_map = {'L': MAX_STEER, 'R': -MAX_STEER, 'S': 0.0}
        for i in range(3):
            dist = path.lengths[i] * self.min_r
            steer = steer_map[path.types[i]]
            d = 1 if dist >= 0 else -1
            n_steps = int(abs(dist) / step) + 1
            step_d = (abs(dist) / n_steps) * d
            for _ in range(n_steps):
                cx, cy, cyaw = px[-1], py[-1], pyaw[-1]
                nx = cx + step_d * math.cos(cyaw)
                ny = cy + step_d * math.sin(cyaw)
                nyaw = cyaw + (step_d / WB) * math.tan(steer)
                px.append(nx); py.append(ny); pyaw.append(M(nyaw))
                pd.append(d); ps.append(steer)
        return px, py, pyaw, pd, ps

def get_car_polygon(x, y, yaw, padding=0.0):
    l, w = CAR_L + (2 * padding), CAR_W + (2 * padding)
    outline = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = outline.dot(rot.T)
    rotated[:, 0] += x
    rotated[:, 1] += y
    return Polygon(rotated)

class Node:
    def __init__(self, x, y, yaw, g, h, parent=None, direction=1, steering=0.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.g, self.h, self.f = g, h, g + h
        self.parent = parent
        self.direction, self.steering = direction, steering
    def __lt__(self, other): return self.f < other.f

# --- UI (SIMPLIFIED): only target slot selection ---
class TargetSlotSelector:
    def __init__(self, slots):
        self.slots = slots
        self.target_id = None
        self.done = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("Select Target Slot")
        plt.subplots_adjust(bottom=0.12)
        self.exit_btn = add_exit_btn(self.fig)
        self._setup_plot()

        self.ax.set_title("Select Target Slot (Green)\nClick a slot to choose", fontsize=10)
        self.slot_patches = {}
        for s in self.slots:
            p = translate(
                rotate(
                    box(-SLOT_D / 2, -SLOT_W / 2, SLOT_D / 2, SLOT_W / 2),
                    s["yaw"],
                    use_radians=True,
                ),
                s["cx"],
                s["cy"],
            )
            self.slot_patches[s["id"]] = self.ax.fill(
                *p.exterior.xy, color="white", edgecolor="#333333", alpha=0.9
            )[0]
            self.ax.text(s["cx"], s["cy"], str(s["id"]), ha="center", va="center", fontweight="bold")
            s["poly_geom"] = p

        ax_btn = plt.axes([0.75, 0.02, 0.2, 0.06])
        self.btn = Button(ax_btn, "CONFIRM", color="#e0e0e0", hovercolor="#c0c0c0")
        self.btn.on_clicked(self._on_confirm)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE)
        self.ax.set_ylim(0, MAP_SIZE)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.6)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        pt = Point(event.xdata, event.ydata)
        s = next((s for s in self.slots if s["poly_geom"].contains(pt)), None)
        if not s:
            return

        # reset colors
        for sid, patch in self.slot_patches.items():
            patch.set_facecolor("white")
        self.target_id = s["id"]
        self.slot_patches[self.target_id].set_facecolor("#66ff66")
        self.fig.canvas.draw()

    def _on_confirm(self, event):
        if self.target_id is None:
            return
        self.done = True
        plt.close(self.fig)

# --- HYBRID A* (REALISTIC + SAFE) ---
def hybrid_a_star(start, goal, obstacles):
    gx, gy, gyaw = goal
    grid_res, yaw_res = 2.0, np.deg2rad(5.0)
    rs_planner = ReedsSheppPlanner(step_size=RS_STEP_SIZE, min_r=MIN_TURN_RADIUS)
    
    h_start = (np.hypot(start[0]-gx, start[1]-gy) + abs(normalize_angle(start[2]-gyaw)) * 10.0) * H_WEIGHT
    start_node = Node(*start, 0, h_start)
    
    open_l = [start_node]; closed = {}
    iter_count = 0
    MAX_ITER = 80000 
    
    print(f"Computing Path (Safety Margin: {SAFE_MARGIN})...")
    
    while open_l and iter_count < MAX_ITER:
        iter_count += 1
        curr = heapq.heappop(open_l)
        
        key = (int(curr.x/grid_res), int(curr.y/grid_res), int(curr.yaw/yaw_res))
        if key in closed and closed[key] <= curr.g: continue
        closed[key] = curr.g

        dist_to_goal = np.hypot(curr.x-gx, curr.y-gy)
        
        # Analytic Shot
        if dist_to_goal < ANALYTIC_SHOT_DIST:
            rs_path = rs_planner.get_optimal_path(curr.x, curr.y, curr.yaw, gx, gy, gyaw)
            if rs_path:
                rx, ry, ryaw, rd, rs = rs_path
                
                # --- SMOOTHING (Interpolation) ---
                if len(rx) > 1:
                    err_x = gx - rx[-1]
                    err_y = gy - ry[-1]
                    err_yaw = normalize_angle(gyaw - ryaw[-1])
                    
                    if np.hypot(err_x, err_y) < 1.0:
                        path_len = len(rx)
                        for i in range(path_len):
                            # Interpolate error correction
                            ratio = i / (path_len - 1)
                            rx[i] += err_x * ratio
                            ry[i] += err_y * ratio
                            ryaw[i] = normalize_angle(ryaw[i] + err_yaw * ratio)

                # --- STRICT COLLISION CHECK (Post-Smoothing) ---
                if np.hypot(rx[-1] - gx, ry[-1] - gy) < 0.1:
                    collision = False
                    for i in range(len(rx)):
                        if not (0 < rx[i] < MAP_SIZE and 0 < ry[i] < MAP_SIZE): collision = True; break
                        cp = get_car_polygon(rx[i], ry[i], ryaw[i], SAFE_MARGIN)
                        if any(cp.intersects(o) for o in obstacles): collision = True; break
                    
                    if not collision:
                        # EMOJI REMOVED HERE
                        print(f"[OK] Analytic Shot Success at step {iter_count}!")
                        path = []
                        temp = curr
                        while temp:
                            path.append((temp.x, temp.y, temp.yaw, temp.direction, temp.steering))
                            temp = temp.parent
                        path = path[::-1]
                        for i in range(len(rx)):
                            path.append((rx[i], ry[i], ryaw[i], rd[i], rs[i]))
                        return path, True

        for v, delta in MOTION_PRIMITIVES:
            nx, ny, nyaw = curr.x, curr.y, curr.yaw
            safe = True
            sub_steps = 2
            dt = 1.0 / sub_steps
            for _ in range(sub_steps): 
                nx += (v * dt) * np.cos(nyaw)
                ny += (v * dt) * np.sin(nyaw)
                nyaw += ((v * dt)/WB) * math.tan(delta)
                if not (0 < nx < MAP_SIZE and 0 < ny < MAP_SIZE): safe=False; break
                if any(get_car_polygon(nx, ny, nyaw, SAFE_MARGIN).intersects(o) for o in obstacles): safe=False; break
            
            nyaw = normalize_angle(nyaw)
            if safe:
                ndir = 1 if v > 0 else -1
                cost_move = abs(v)
                cost_steer = 0 if delta == 0 else abs(delta) * COST_STEER
                cost_gear = COST_GEAR_SWITCH if ndir != curr.direction else 0.0 
                cost_smooth = abs(delta - curr.steering) * COST_STEER_CHANGE
                new_g = curr.g + cost_move + cost_steer + cost_gear + cost_smooth
                angle_err = abs(normalize_angle(nyaw - gyaw))
                h = (np.hypot(nx-gx, ny-gy) + angle_err * 10.0) * H_WEIGHT
                heapq.heappush(open_l, Node(nx, ny, nyaw, new_g, h, curr, ndir, delta))
    
    # EMOJI REMOVED HERE
    print("[X] Timeout / No Path Found.")
    return [], False

def create_slots():
    # Slot midpoints (cx, cy) in cm as provided by user.
    # Slots are 25x40 cm (SLOT_W x SLOT_D).
    #
    # Yaw convention in this file:
    #   0 rad = +X (east), CCW positive.
    # Wall-based slot orientation:
    #   - Top wall (y high): face down  => -pi/2
    #   - Bottom wall (y low): face up => +pi/2
    #   - Left wall (x low): face right => 0
    #   - Right wall (x high): face left => pi
    slots = [
        {"id": 1, "cx": 112.5, "cy": 180.0, "yaw": -np.pi / 2},
        {"id": 2, "cx": 180.0, "cy": 137.5, "yaw": np.pi},
        {"id": 3, "cx": 180.0, "cy": 112.5, "yaw": np.pi},
        {"id": 4, "cx": 180.0, "cy": 87.5, "yaw": np.pi},
        {"id": 5, "cx": 180.0, "cy": 62.5, "yaw": np.pi},
        {"id": 6, "cx": 112.5, "cy": 20.0, "yaw": np.pi / 2},
        {"id": 7, "cx": 87.5, "cy": 20.0, "yaw": np.pi / 2},
        {"id": 8, "cx": 20.0, "cy": 62.5, "yaw": 0.0},
        {"id": 9, "cx": 20.0, "cy": 87.5, "yaw": 0.0},
        {"id": 10, "cx": 20.0, "cy": 112.5, "yaw": 0.0},
        {"id": 11, "cx": 20.0, "cy": 137.5, "yaw": 0.0},
        {"id": 12, "cx": 87.5, "cy": 180.0, "yaw": -np.pi / 2},
    ]
    return slots

# --- SIMULATION LOGIC ---
def run_simulation():
    slots = create_slots()

    # Only keep slot selection UI
    sel = TargetSlotSelector(slots)
    plt.show()
    if not sel.done or sel.target_id is None:
        return

    # Defaults requested:
    # - obstacle count = 0 (no custom obstacles)
    # - occupied slots count = 0 (no occupied slots)
    obs_list = [
        box(0, 0, MAP_SIZE, 0),
        box(0, MAP_SIZE, MAP_SIZE, MAP_SIZE),
        box(0, 0, 0, MAP_SIZE),
        box(MAP_SIZE, 0, MAP_SIZE, MAP_SIZE),
    ]

    # Start pose comes from AprilTag map output
    try:
        from apriltag_pose import get_latest_map_pose
    except Exception as e:
        print(f"[X] Could not import apriltag_pose.get_latest_map_pose: {e}")
        return

    pose = get_latest_map_pose(timeout_s=8.0, target_fps=10.0)
    if pose is None:
        print("[X] No AprilTag detected. Start pose unavailable.")
        return

    # AprilTag map yaw definition (per your spec):
    # 0° points from (100,100) -> (100,200) (North / +Y), CCW positive.
    # Planner yaw expects: 0 rad = +X (East), CCW positive.
    start_x = pose["x_cm"]
    start_y = pose["y_cm"]
    start_yaw = normalize_angle(np.deg2rad(pose["yaw_deg"]) + (np.pi / 2.0))
    start_pose = (start_x, start_y, start_yaw)

    target = next(s for s in slots if s["id"] == sel.target_id)

    # Try forward first; if it fails, try reverse automatically (no popups).
    yaw_reverse = target["yaw"]
    yaw_forward = normalize_angle(target["yaw"] + np.pi)

    print(
        f"\n[POSE] start=({start_x:.1f},{start_y:.1f},{np.rad2deg(start_yaw):.1f}deg std) "
        f"| from tag {pose['tag_id']} ({pose['camera']}) dist={pose['distance_cm']:.1f}cm"
    )
    print(f"[PLAN] target slot={sel.target_id} at ({target['cx']:.1f},{target['cy']:.1f})")

    path, success = hybrid_a_star(start_pose, (target["cx"], target["cy"], yaw_forward), obs_list)
    final_mode_str = "FORWARD"

    if not success:
        print("[!] Forward plan failed. Trying reverse...")
        path, success = hybrid_a_star(start_pose, (target["cx"], target["cy"], yaw_reverse), obs_list)
        final_mode_str = "REVERSE" if success else "NONE"

    if not success:
        return

    # Write planned path for the controller (Raspberry Pi)
    target_pose = (target["cx"], target["cy"], yaw_forward if final_mode_str == "FORWARD" else yaw_reverse)
    try:
        save_path_json(path, start_pose, sel.target_id, target_pose, filename="planned_path.json")
        print("[OK] Wrote planned path to planned_path.json")
    except Exception as e:
        print(f"[!] Could not write planned_path.json: {e}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE); ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    exit_btn = add_exit_btn(fig) 

    title_text = f"SUCCESS ({final_mode_str})"
    title_color = "#006400"
    ax.set_title(title_text, color=title_color, fontweight='bold', fontsize=12)

    for o in obs_list: ax.fill(*o.exterior.xy, color='#2f4f4f') 
    p_target = translate(rotate(box(-SLOT_D/2,-SLOT_W/2,SLOT_D/2,SLOT_W/2), target['yaw'], use_radians=True), target['cx'], target['cy'])
    ax.fill(*p_target.exterior.xy, color='#98fb98', alpha=0.5) 
    ax.plot(target['cx'], target['cy'], 'rx', markersize=8)

    if path and len(path) > 1:
        seg_x, seg_y, curr_dir = [path[0][0]], [path[0][1]], path[0][3]
        for i in range(1, len(path)):
            x, y, _, d, _ = path[i]
            if d == curr_dir:
                seg_x.append(x); seg_y.append(y)
            else:
                color = 'blue' if curr_dir == 1 else 'red'
                style = '-' if curr_dir == 1 else '--'
                ax.plot(seg_x, seg_y, color=color, linestyle=style, lw=2, alpha=0.6)
                seg_x, seg_y, curr_dir = [path[i-1][0], x], [path[i-1][1], y], d
        color = 'blue' if curr_dir == 1 else 'red'
        style = '-' if curr_dir == 1 else '--'
        ax.plot(seg_x, seg_y, color=color, linestyle=style, lw=2, alpha=0.6)

    fwd_line = mlines.Line2D([], [], color='blue', linestyle='-', lw=2, label='Forward Gear')
    rev_line = mlines.Line2D([], [], color='red', linestyle='--', lw=2, label='Reverse Gear')
    ax.legend(handles=[fwd_line, rev_line], loc='upper right')

    start_x, start_y, start_yaw, _, _ = path[0]
    cp = patches.Polygon(np.array(get_car_polygon(start_x, start_y, start_yaw).exterior.coords), fc='#00ced1', ec='black')
    ax.add_patch(cp)
    arrow_line, = ax.plot([], [], 'r-', linewidth=2, zorder=10)
    
    class AnimControl:
        def __init__(self, anim): self.anim = anim
        def stop(self, event): self.anim.event_source.stop()

    def upd(i):
        x, y, yaw, _, _ = path[i]
        cp.set_xy(list(get_car_polygon(x, y, yaw).exterior.coords))
        front_x = x + (CAR_L/2) * np.cos(yaw)
        front_y = y + (CAR_L/2) * np.sin(yaw)
        arrow_line.set_data([x, front_x], [y, front_y])
        return cp, arrow_line
    
    anim = FuncAnimation(fig, upd, frames=len(path), interval=30, repeat=False, blit=False)
    ctrl = AnimControl(anim)
    
    ax_stop = plt.axes([0.60, 0.01, 0.1, 0.05])
    btn_stop = Button(ax_stop, 'STOP', color='#ff4d4d', hovercolor='#ff3333')
    btn_stop.on_clicked(ctrl.stop)
    
    # --- ADDED RESTART BUTTON ---
    ax_restart = plt.axes([0.72, 0.01, 0.12, 0.05])
    btn_restart = Button(ax_restart, 'RESTART', color='#2E8B57', hovercolor='#3CB371')
    btn_restart.label.set_color('white')
    
    def on_restart(event):
        plt.close(fig) # Close current window
        run_simulation() # Restart the whole process
        
    btn_restart.on_clicked(on_restart)
    
    plt.show()

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    run_simulation()
