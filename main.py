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
import math
import sys
import json
import time
import tkinter as tk
from tkinter import messagebox

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, translate

# New, more robust planner implementation (algorithm changes live here)
from planner_v2 import hybrid_a_star_v2

# --- CONFIGURATION (PRODUCTION READY) ---
MAP_SIZE = 200

# 1. Controller Safety Settings
SAFE_MARGIN = 2.0

# 2. Target Tolerance Settings (AYARLANABİLİR TOLERANSLAR)
FINAL_XY_TOLERANCE = 1
FINAL_YAW_TOLERANCE = np.deg2rad(5.0)

# Car Specs
CAR_L, CAR_W = 25.0, 18.0
WB = CAR_L * 0.75
MAX_STEER = np.deg2rad(25.0)

SLOT_W, SLOT_D = 25.0, 40.0

# Pathfinding Weights
COST_STEER = 0.5
COST_GEAR_SWITCH = 15.0  # Dengelendi: Zorunlu geri vites manevralarına izin verir
COST_STEER_CHANGE = 1.0
H_WEIGHT = 2.5  # Artırıldı: Hedefe daha agresif (hızlı) yönelir

# Reeds-Shepp Constants
ANALYTIC_SHOT_DIST = 150.0  # Artırıldı: Uzaktan kavisli yolları daha erken dener
RS_STEP_SIZE = 0.5
MIN_TURN_RADIUS = WB / math.tan(MAX_STEER) * 1.05


# --- UTILS ---
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_wall_obstacles(map_size: float, wall_thickness: float = 1.0) -> list[Polygon]:
    """
    Create thin rectangular boundary obstacles around the map.

    Note: The planner also has explicit (0 < x < MAP_SIZE, 0 < y < MAP_SIZE)
    checks, so these walls are mainly for UI/consistency.
    """
    t = float(wall_thickness)
    s = float(map_size)
    return [
        box(0.0, -t, s, 0.0),  # bottom
        box(0.0, s, s, s + t),  # top
        box(-t, 0.0, 0.0, s),  # left
        box(s, 0.0, s + t, s),  # right
    ]


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
    points = [
        {
            "x_cm": float(x),
            "y_cm": float(y),
            "yaw_rad": float(yaw),
            "direction": int(d),
            "steer_rad": float(steer),
        }
        for (x, y, yaw, d, steer) in path
    ]

    if len(points) >= 2:
        points[0]["direction"] = int(points[1]["direction"])
        points[0]["steer_rad"] = float(points[1]["steer_rad"])

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
        "points": points,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def trim_path(full_path, min_tail_dist=15.0):
    if len(full_path) < 2: return full_path

    tail_dist = 0.0
    last_dir = full_path[-1][3]
    cutoff_idx = 0

    for i in range(len(full_path) - 2, -1, -1):
        if full_path[i][3] != last_dir:
            cutoff_idx = i
            break
        tail_dist += np.hypot(full_path[i][0] - full_path[i + 1][0], full_path[i][1] - full_path[i + 1][1])

    if cutoff_idx > 0 and tail_dist < min_tail_dist:
        print(f"[*] Gerçeklik Filtresi: Fiziksel limit altı ({tail_dist:.1f}cm) son manevra kırpıldı.")
        return full_path[:cutoff_idx + 1]

    return full_path


def get_car_polygon(x, y, yaw, padding=0.0):
    front_overhang = 4.0 + padding
    rear_overhang = (CAR_L - 4.0) + padding
    w = CAR_W + (2 * padding)
    outline = np.array([
        [-rear_overhang, -w / 2],
        [front_overhang, -w / 2],
        [front_overhang, w / 2],
        [-rear_overhang, w / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = outline.dot(rot.T)
    rotated[:, 0] += x
    rotated[:, 1] += y
    return Polygon(rotated)


# --- UI CLASSES ---
class ObstacleEditor:
    def __init__(self, slots):
        self.slots, self.custom_obstacles = slots, []
        self.temp_point, self.preview_patch = None, None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('Stage 1: Environment')
        plt.subplots_adjust(bottom=0.15)
        self.exit_btn = add_exit_btn(self.fig)
        self.setup_plot()
        self.ax.set_title("STAGE 1: Draw Obstacles\nClick Start -> Click End -> Save", fontsize=10)
        for s in self.slots:
            p = translate(rotate(box(-SLOT_D / 2, -SLOT_W / 2, SLOT_D / 2, SLOT_W / 2), s['yaw'], use_radians=True),
                          s['cx'], s['cy'])
            self.ax.plot(*p.exterior.xy, color='gray', alpha=0.4, linestyle='--')
        ax_btn = plt.axes([0.7, 0.02, 0.15, 0.06])
        self.btn = Button(ax_btn, 'NEXT', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn.on_clicked(self.close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def close(self, event):
        plt.close(self.fig)

    def setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE);
        self.ax.set_ylim(0, MAP_SIZE)
        self.ax.set_aspect('equal');
        self.ax.grid(True, linestyle=':', alpha=0.6)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.temp_point is None:
            self.temp_point = (event.xdata, event.ydata)
        else:
            yaw = math.atan2(event.ydata - self.temp_point[1], event.xdata - self.temp_point[0])
            self.custom_obstacles.append((self.temp_point[0], self.temp_point[1], yaw))
            p = translate(rotate(box(-5, -5, 5, 5), yaw, use_radians=True), self.temp_point[0], self.temp_point[1])
            self.ax.fill(*p.exterior.xy, color='#333333');
            self.temp_point = None;
            self.fig.canvas.draw()

    def on_move(self, event):
        if self.temp_point is None or event.inaxes != self.ax: return
        if self.preview_patch: self.preview_patch.remove()
        yaw = math.atan2(event.ydata - self.temp_point[1], event.xdata - self.temp_point[0])
        p = translate(rotate(box(-5, -5, 5, 5), yaw, use_radians=True), self.temp_point[0], self.temp_point[1])
        self.preview_patch = self.ax.fill(*p.exterior.xy, color='gray', alpha=0.5)[0];
        self.fig.canvas.draw()


class MapSelector:
    def __init__(self, slots, custom_obs):
        self.slots, self.custom_obs = slots, custom_obs
        self.occupied_ids, self.target_id = set(), None
        self.done, self.selection_step = False, 0
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title('Stage 2: Setup')
        plt.subplots_adjust(bottom=0.15)
        self.exit_btn = add_exit_btn(self.fig)
        self.setup_plot()
        self.ax.set_title("STAGE 2: Select Occupied Slots (Red)\nWhite = Free", fontsize=10)
        self.slot_patches = {}
        for obs in self.custom_obs:
            p = translate(rotate(box(-5, -5, 5, 5), obs[2], use_radians=True), obs[0], obs[1])
            self.ax.fill(*p.exterior.xy, color='#333333')
        for s in self.slots:
            p = translate(rotate(box(-SLOT_D / 2, -SLOT_W / 2, SLOT_D / 2, SLOT_W / 2), s['yaw'], use_radians=True),
                          s['cx'], s['cy'])
            self.slot_patches[s['id']] = self.ax.fill(*p.exterior.xy, color='white', edgecolor='#333333', alpha=0.9)[0]
            self.ax.text(s['cx'], s['cy'], str(s['id']), ha='center', va='center', fontweight='bold')
            s['poly_geom'] = p
        ax_btn = plt.axes([0.7, 0.02, 0.15, 0.06])
        self.btn = Button(ax_btn, 'CONFIRM', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn.on_clicked(self.on_confirm)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE);
        self.ax.set_ylim(0, MAP_SIZE);
        self.ax.set_aspect('equal');
        self.ax.grid(True, linestyle=':', alpha=0.6)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        pt = Point(event.xdata, event.ydata)
        s = next((s for s in self.slots if s['poly_geom'].contains(pt)), None)
        if s:
            sid = s['id']
            if self.selection_step == 0:
                if sid in self.occupied_ids:
                    self.occupied_ids.remove(sid); self.slot_patches[sid].set_facecolor('white')
                else:
                    self.occupied_ids.add(sid); self.slot_patches[sid].set_facecolor('#ff6666')
            else:
                self.target_id = sid;
                self.slot_patches[sid].set_facecolor('#66ff66')
                self.fig.canvas.draw();
                plt.pause(0.2);
                self.done = True;
                plt.close(self.fig)
            self.fig.canvas.draw()

    def on_confirm(self, event):
        if self.selection_step == 0:
            self.selection_step = 1
            self.ax.set_title("STAGE 3: Select Target Slot (Green)", fontweight='bold')
            self.btn.label.set_text("SELECTING...")


class DirectionAndPoseSelector:
    def __init__(self, slots, occupied, custom_obs, target_id):
        self.is_reverse_park = True
        self.start_pose, self.done = None, False
        self.target_slot = next(s for s in slots if s['id'] == target_id)

        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.canvas.manager.set_window_title('Stage 3: Pose')
        plt.subplots_adjust(bottom=0.20)
        self.exit_btn = add_exit_btn(self.fig)
        self.setup_plot()
        self.ax.set_title("STAGE 4: Position Car (Drag to Orient)\nRed Arrow = Front of Car", fontsize=10)

        self.obs_list = make_wall_obstacles(MAP_SIZE)
        for o in custom_obs: self.obs_list.append(
            translate(rotate(box(-5, -5, 5, 5), o[2], use_radians=True), o[0], o[1]))
        for s in slots:
            p = translate(rotate(box(-SLOT_D / 2, -SLOT_W / 2, SLOT_D / 2, SLOT_W / 2), s['yaw'], use_radians=True),
                          s['cx'], s['cy'])
            if s['id'] == target_id:
                self.ax.fill(*p.exterior.xy, color='#66ff66', alpha=0.4)
            elif s['id'] in occupied:
                self.obs_list.append(p); self.ax.fill(*p.exterior.xy, color='#ff6666', alpha=0.4)
            else:
                self.ax.plot(*p.exterior.xy, color='#333333', linestyle=':')
        for o in self.obs_list:
            if isinstance(o, Polygon): self.ax.fill(*o.exterior.xy, color='#333333')

        ax_rev = plt.axes([0.15, 0.05, 0.3, 0.07]);
        ax_fwd = plt.axes([0.55, 0.05, 0.3, 0.07])
        self.btn_rev = Button(ax_rev, 'Reverse', color='lightblue');
        self.btn_fwd = Button(ax_fwd, 'Forward', color='white')
        self.btn_rev.on_clicked(self.set_reverse);
        self.btn_fwd.on_clicked(self.set_forward)

        self.start_point = None;
        self.point, = self.ax.plot([], [], 'b.', ms=10)
        self.ghost_patch, self.ghost_arrow, self.car_preview_patch, self.car_preview_arrow = None, None, None, None

        self.info_box = self.ax.text(0.02, 0.98, "Select Start Position...", transform=self.ax.transAxes,
                                     fontsize=10, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self.collision_text = self.ax.text(MAP_SIZE / 2, MAP_SIZE - 20, "", ha='center', color='red', fontsize=12,
                                           fontweight='bold')
        self.update_ghost_car()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE);
        self.ax.set_ylim(0, MAP_SIZE);
        self.ax.set_aspect('equal');
        self.ax.grid(True, linestyle=':', alpha=0.6)

    def get_target_yaw(self):
        return self.target_slot['yaw'] if self.is_reverse_park else normalize_angle(self.target_slot['yaw'] + np.pi)

    def update_ghost_car(self):
        if self.ghost_patch: self.ghost_patch.remove(); self.ghost_patch = None
        if self.ghost_arrow: self.ghost_arrow.remove(); self.ghost_arrow = None
        yaw = self.get_target_yaw()
        cx, cy = self.target_slot['cx'], self.target_slot['cy']
        poly = get_car_polygon(cx, cy, yaw)
        self.ghost_patch = patches.Polygon(np.array(poly.exterior.coords), fc='none', ec='black', lw=2, linestyle='--',
                                           alpha=0.7)
        self.ax.add_patch(self.ghost_patch)
        arrow_len = CAR_L / 2
        self.ghost_arrow = self.ax.arrow(cx, cy, arrow_len * np.cos(yaw), arrow_len * np.sin(yaw), head_width=4,
                                         head_length=4, fc='black', ec='black', alpha=0.7)
        self.fig.canvas.draw()

    def set_reverse(self, event):
        self.is_reverse_park = True; self.btn_rev.color = 'lightblue'; self.btn_fwd.color = 'white'; self.update_ghost_car()

    def set_forward(self, event):
        self.is_reverse_park = False; self.btn_fwd.color = 'lightblue'; self.btn_rev.color = 'white'; self.update_ghost_car()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.start_point is None:
            self.start_point = (event.xdata, event.ydata)
            self.point.set_data([event.xdata], [event.ydata])
            self.collision_text.set_text("Drag to set orientation...")
            self.info_box.set_text("Position selected. Drag for Angle.")
            self.fig.canvas.draw()
        else:
            yaw = math.atan2(event.ydata - self.start_point[1], event.xdata - self.start_point[0])
            poly = get_car_polygon(self.start_point[0], self.start_point[1], yaw, padding=0.0)
            if any(poly.intersects(obs) for obs in self.obs_list):
                self.collision_text.set_text("[!] COLLISION! Select new position.")
                self.start_point = None;
                self.point.set_data([], [])
                if self.car_preview_patch: self.car_preview_patch.remove(); self.car_preview_patch = None
                if self.car_preview_arrow: self.car_preview_arrow.remove(); self.car_preview_arrow = None
                self.fig.canvas.draw()
            else:
                self.start_pose = (self.start_point[0], self.start_point[1], yaw)
                self.done = True;
                plt.close(self.fig)

    def on_move(self, event):
        if self.start_point is None or event.inaxes != self.ax: return
        yaw = math.atan2(event.ydata - self.start_point[1], event.xdata - self.start_point[0])
        self.info_box.set_text(
            f"Pos: ({self.start_point[0]:.1f}, {self.start_point[1]:.1f})\nAngle: {np.rad2deg(normalize_angle(yaw)):.1f}°")
        if self.car_preview_patch: self.car_preview_patch.remove(); self.car_preview_patch = None
        if self.car_preview_arrow: self.car_preview_arrow.remove(); self.car_preview_arrow = None
        poly = get_car_polygon(self.start_point[0], self.start_point[1], yaw)
        self.car_preview_patch = patches.Polygon(np.array(poly.exterior.coords), fc='cyan', ec='black', alpha=0.6)
        self.ax.add_patch(self.car_preview_patch)
        arrow_len = CAR_L / 2
        self.car_preview_arrow = self.ax.arrow(self.start_point[0], self.start_point[1], arrow_len * np.cos(yaw),
                                               arrow_len * np.sin(yaw), head_width=3, head_length=3, fc='red', ec='red',
                                               zorder=10)
        self.fig.canvas.draw()


def create_slots():
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
    # Initialize TKinter root hidden for message boxes
    root = tk.Tk()
    root.withdraw()

    slots = create_slots()

    # UI Loop from kodSon
    ed = ObstacleEditor(slots);
    plt.show()
    sel = MapSelector(slots, ed.custom_obstacles);
    plt.show()
    if not sel.done or sel.target_id is None: return
    dp_sel = DirectionAndPoseSelector(slots, sel.occupied_ids, ed.custom_obstacles, sel.target_id);
    plt.show()
    if not dp_sel.done: return
    plt.close('all')

    obs_list = make_wall_obstacles(MAP_SIZE)
    for o in ed.custom_obstacles: obs_list.append(
        translate(rotate(box(-5, -5, 5, 5), o[2], use_radians=True), o[0], o[1]))

    target = next(s for s in slots if s['id'] == sel.target_id)
    for s in slots:
        if s['id'] in sel.occupied_ids:
            p = translate(rotate(box(-SLOT_D / 2, -SLOT_W / 2, SLOT_D / 2, SLOT_W / 2), s['yaw'], use_radians=True),
                          s['cx'], s['cy'])
            obs_list.append(p)

    start_pose = dp_sel.start_pose

    PARK_INSET_CM = 10.0
    wall_dx = -np.cos(target["yaw"])
    wall_dy = -np.sin(target["yaw"])
    goal_in_x = float(target["cx"] + PARK_INSET_CM * wall_dx)
    goal_in_y = float(target["cy"] + PARK_INSET_CM * wall_dy)
    goal_out_x = float(target["cx"] - PARK_INSET_CM * wall_dx)
    goal_out_y = float(target["cy"] - PARK_INSET_CM * wall_dy)

    yaw_reverse = target["yaw"]
    yaw_forward = normalize_angle(target["yaw"] + np.pi)

    if dp_sel.is_reverse_park:
        primary_yaw, fallback_yaw = yaw_reverse, yaw_forward
        primary_x, primary_y = goal_out_x, goal_out_y
        fallback_x, fallback_y = goal_in_x, goal_in_y
        mode_str, fallback_str = "REVERSE", "FORWARD"
    else:
        primary_yaw, fallback_yaw = yaw_forward, yaw_reverse
        primary_x, primary_y = goal_in_x, goal_in_y
        fallback_x, fallback_y = goal_out_x, goal_out_y
        mode_str, fallback_str = "FORWARD", "REVERSE"

    print(f"\n[CAR] Attempting {mode_str} parking (User Selection)...")
    path, success = hybrid_a_star_v2(
        start_pose,
        (primary_x, primary_y, primary_yaw),
        obs_list,
        map_size=MAP_SIZE,
        wb=WB,
        max_steer=MAX_STEER,
        min_turn_radius=MIN_TURN_RADIUS,
        car_l=CAR_L,
        car_w=CAR_W,
        safe_margin=SAFE_MARGIN,
        final_xy_tolerance=FINAL_XY_TOLERANCE,
        final_yaw_tolerance=FINAL_YAW_TOLERANCE,
        cost_gear_switch=COST_GEAR_SWITCH,
        cost_steer=COST_STEER,
        cost_steer_change=COST_STEER_CHANGE,
        h_weight=H_WEIGHT,
        analytic_shot_dist=ANALYTIC_SHOT_DIST,
        rs_step_size=RS_STEP_SIZE,
    )

    switched_mode = False
    final_mode_str = mode_str
    chosen_goal_x, chosen_goal_y = primary_x, primary_y

    if not success:
        print(f"\n[!]  {mode_str} PARKING FAILED.")

        user_wants_fallback = messagebox.askyesno(
            title="Parking Failed",
            message=f"{mode_str} parking is impossible from this position.\n\nDo you want to attempt {fallback_str} parking instead?"
        )

        if user_wants_fallback:
            print(f"\n[...]  Switching strategy: Attempting {fallback_str} parking...")
            path, success = hybrid_a_star_v2(
                start_pose,
                (fallback_x, fallback_y, fallback_yaw),
                obs_list,
                map_size=MAP_SIZE,
                wb=WB,
                max_steer=MAX_STEER,
                min_turn_radius=MIN_TURN_RADIUS,
                car_l=CAR_L,
                car_w=CAR_W,
                safe_margin=SAFE_MARGIN,
                final_xy_tolerance=FINAL_XY_TOLERANCE,
                final_yaw_tolerance=FINAL_YAW_TOLERANCE,
                cost_gear_switch=COST_GEAR_SWITCH,
                cost_steer=COST_STEER,
                cost_steer_change=COST_STEER_CHANGE,
                h_weight=H_WEIGHT,
                analytic_shot_dist=ANALYTIC_SHOT_DIST,
                rs_step_size=RS_STEP_SIZE,
            )

            if success:
                switched_mode = True
                final_mode_str = fallback_str
                chosen_goal_x, chosen_goal_y = fallback_x, fallback_y
                print(f"[OK]  FALLBACK SUCCESS: Found a valid {fallback_str} parking path!")
            else:
                messagebox.showerror("Error", f"Both Reverse and Forward parking are impossible from this position.")
                print(f"[X]  CRITICAL FAILURE: {fallback_str} parking is ALSO impossible.")
        else:
            print("\n[STOP]  Parking aborted by user.")
            return

    if not success:
        return

    target_pose = (chosen_goal_x, chosen_goal_y, fallback_yaw if switched_mode else primary_yaw)
    try:
        save_path_json(path, start_pose, sel.target_id, target_pose, filename="planned_path.json")
        print("[OK] Wrote planned path to planned_path.json")
    except Exception as e:
        print(f"[!] Could not write planned_path.json: {e}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_xlim(0, MAP_SIZE);
    ax.set_ylim(0, MAP_SIZE);
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    exit_btn = add_exit_btn(fig)

    if switched_mode:
        title_text = f"SUCCESS ({final_mode_str} - User Approved Switch)"
        title_color = '#d4af37'
    else:
        title_text = f"SUCCESS ({final_mode_str})"
        title_color = '#006400'

    ax.set_title(title_text, color=title_color, fontweight='bold', fontsize=12)

    for o in obs_list: ax.fill(*o.exterior.xy, color='#2f4f4f')
    p_target = translate(rotate(box(-SLOT_D / 2, -SLOT_W / 2, SLOT_D / 2, SLOT_W / 2), target['yaw'], use_radians=True),
                         target['cx'], target['cy'])
    ax.fill(*p_target.exterior.xy, color='#98fb98', alpha=0.5)
    ax.plot(chosen_goal_x, chosen_goal_y, 'rx', markersize=8)

    if path and len(path) > 1:
        seg_x, seg_y, curr_dir = [path[0][0]], [path[0][1]], path[0][3]
        for i in range(1, len(path)):
            x, y, _, d, _ = path[i]
            if d == curr_dir:
                seg_x.append(x);
                seg_y.append(y)
            else:
                color = 'blue' if curr_dir == 1 else 'red'
                style = '-' if curr_dir == 1 else '--'
                ax.plot(seg_x, seg_y, color=color, linestyle=style, lw=2, alpha=0.6)
                seg_x, seg_y, curr_dir = [path[i - 1][0], x], [path[i - 1][1], y], d
        color = 'blue' if curr_dir == 1 else 'red'
        style = '-' if curr_dir == 1 else '--'
        ax.plot(seg_x, seg_y, color=color, linestyle=style, lw=2, alpha=0.6)

    fwd_line = mlines.Line2D([], [], color='blue', linestyle='-', lw=2, label='Forward Gear')
    rev_line = mlines.Line2D([], [], color='red', linestyle='--', lw=2, label='Reverse Gear')
    ax.legend(handles=[fwd_line, rev_line], loc='upper right')

    start_x, start_y, start_yaw, _, _ = path[0]
    cp = patches.Polygon(np.array(get_car_polygon(start_x, start_y, start_yaw).exterior.coords), fc='#00ced1',
                         ec='black')
    ax.add_patch(cp)
    arrow_line, = ax.plot([], [], 'r-', linewidth=2, zorder=10)

    class AnimControl:
        def __init__(self, anim): self.anim = anim

        def stop(self, event): self.anim.event_source.stop()

    def upd(i):
        x, y, yaw, _, _ = path[i]
        cp.set_xy(list(get_car_polygon(x, y, yaw).exterior.coords))
        front_x = x + (CAR_L / 2) * np.cos(yaw)
        front_y = y + (CAR_L / 2) * np.sin(yaw)
        arrow_line.set_data([x, front_x], [y, front_y])
        return cp, arrow_line

    anim = FuncAnimation(fig, upd, frames=len(path), interval=30, repeat=False, blit=False)
    ctrl = AnimControl(anim)

    ax_stop = plt.axes([0.60, 0.01, 0.1, 0.05])
    btn_stop = Button(ax_stop, 'STOP', color='#ff4d4d', hovercolor='#ff3333')
    btn_stop.on_clicked(ctrl.stop)

    ax_restart = plt.axes([0.72, 0.01, 0.12, 0.05])
    btn_restart = Button(ax_restart, 'RESTART', color='#2E8B57', hovercolor='#3CB371')
    btn_restart.label.set_color('white')

    def on_restart(event):
        plt.close(fig)
        run_simulation()

    btn_restart.on_clicked(on_restart)

    plt.show()


# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    run_simulation()