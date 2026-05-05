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
import tkinter as tk
from tkinter import messagebox

sys.stdout.reconfigure(encoding='utf-8')
from shapely.geometry import Polygon, box, Point
from shapely.affinity import rotate, translate

# --- CONFIGURATION (PRODUCTION READY) ---
MAP_SIZE = 200

# 1. Controller Safety Settings
SAFE_MARGIN = 2.0  

# 2. Target Tolerance Settings (AYARLANABİLİR TOLERANSLAR)
FINAL_XY_TOLERANCE = 2.0                
FINAL_YAW_TOLERANCE = np.deg2rad(5.0)  

# Car Specs
CAR_L, CAR_W = 25.0, 18.0
WB = CAR_L * 0.75
MAX_STEER = np.deg2rad(25.0) 

SLOT_W, SLOT_D = 25.0, 40.0  

# Pathfinding Weights
SPEED = 5.0
COST_STEER = 0.5            
COST_GEAR_SWITCH = 40.0   
COST_STEER_CHANGE = 2.0   
H_WEIGHT = 2.5            

# Reeds-Shepp Constants
ANALYTIC_SHOT_DIST = 90.0 
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
    
    for i in range(len(full_path)-2, -1, -1):
        if full_path[i][3] != last_dir:
            cutoff_idx = i
            break
        tail_dist += np.hypot(full_path[i][0] - full_path[i+1][0], full_path[i][1] - full_path[i+1][1])
        
    if cutoff_idx > 0 and tail_dist < min_tail_dist:
        print(f"[*] Gerçeklik Filtresi: Fiziksel limit altı ({tail_dist:.1f}cm) son manevra kırpıldı.")
        return full_path[:cutoff_idx+1]
        
    return full_path

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
        
        def path_cost(p):
            cost = abs(p.L) * self.min_r
            for i in range(len(p.lengths)):
                dist = abs(p.lengths[i]) * self.min_r
                if 0.1 < dist < 20.0:  
                    cost += 80.0 + (15.0 - dist) * 5.0
                if i > 0 and (p.lengths[i] * p.lengths[i-1] < 0):
                    cost += 200.0  
            return cost

        best_path = min(paths, key=path_cost)
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
    front_overhang = 4.0 + padding
    rear_overhang = (CAR_L - 4.0) + padding
    w = CAR_W + (2 * padding)
    outline = np.array([
        [-rear_overhang, -w / 2],
        [front_overhang, -w / 2],
        [front_overhang,  w / 2],
        [-rear_overhang,  w / 2],
    ])
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
            p = translate(rotate(box(-SLOT_D/2, -SLOT_W/2, SLOT_D/2, SLOT_W/2), s['yaw'], use_radians=True), s['cx'], s['cy'])
            self.ax.plot(*p.exterior.xy, color='gray', alpha=0.4, linestyle='--')
        ax_btn = plt.axes([0.7, 0.02, 0.15, 0.06])
        self.btn = Button(ax_btn, 'NEXT', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn.on_clicked(self.close)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
    
    def close(self, event): plt.close(self.fig)
    def setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE); self.ax.set_ylim(0, MAP_SIZE)
        self.ax.set_aspect('equal'); self.ax.grid(True, linestyle=':', alpha=0.6)
    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.temp_point is None: self.temp_point = (event.xdata, event.ydata)
        else:
            yaw = math.atan2(event.ydata - self.temp_point[1], event.xdata - self.temp_point[0])
            self.custom_obstacles.append((self.temp_point[0], self.temp_point[1], yaw))
            p = translate(rotate(box(-5,-5,5,5), yaw, use_radians=True), self.temp_point[0], self.temp_point[1])
            self.ax.fill(*p.exterior.xy, color='#333333'); self.temp_point = None; self.fig.canvas.draw()
    def on_move(self, event):
        if self.temp_point is None or event.inaxes != self.ax: return
        if self.preview_patch: self.preview_patch.remove()
        yaw = math.atan2(event.ydata - self.temp_point[1], event.xdata - self.temp_point[0])
        p = translate(rotate(box(-5,-5,5,5), yaw, use_radians=True), self.temp_point[0], self.temp_point[1])
        self.preview_patch = self.ax.fill(*p.exterior.xy, color='gray', alpha=0.5)[0]; self.fig.canvas.draw()

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
            p = translate(rotate(box(-5,-5,5,5), obs[2], use_radians=True), obs[0], obs[1])
            self.ax.fill(*p.exterior.xy, color='#333333')
        for s in self.slots:
            p = translate(rotate(box(-SLOT_D/2, -SLOT_W/2, SLOT_D/2, SLOT_W/2), s['yaw'], use_radians=True), s['cx'], s['cy'])
            self.slot_patches[s['id']] = self.ax.fill(*p.exterior.xy, color='white', edgecolor='#333333', alpha=0.9)[0]
            self.ax.text(s['cx'], s['cy'], str(s['id']), ha='center', va='center', fontweight='bold')
            s['poly_geom'] = p
        ax_btn = plt.axes([0.7, 0.02, 0.15, 0.06])
        self.btn = Button(ax_btn, 'CONFIRM', color='#e0e0e0', hovercolor='#c0c0c0')
        self.btn.on_clicked(self.on_confirm)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE); self.ax.set_ylim(0, MAP_SIZE); self.ax.set_aspect('equal'); self.ax.grid(True, linestyle=':', alpha=0.6)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        pt = Point(event.xdata, event.ydata)
        s = next((s for s in self.slots if s['poly_geom'].contains(pt)), None)
        if s:
            sid = s['id']
            if self.selection_step == 0:
                if sid in self.occupied_ids: self.occupied_ids.remove(sid); self.slot_patches[sid].set_facecolor('white')
                else: self.occupied_ids.add(sid); self.slot_patches[sid].set_facecolor('#ff6666') 
            else:
                self.target_id = sid; self.slot_patches[sid].set_facecolor('#66ff66') 
                self.fig.canvas.draw(); plt.pause(0.2); self.done = True; plt.close(self.fig)
            self.fig.canvas.draw()
            
    def on_confirm(self, event):
        if self.selection_step == 0:
            self.selection_step = 1
            self.ax.set_title("STAGE 3: Select Target Slot (Green)", fontweight='bold')
            self.btn.label.set_text("SELECTING...")

class DirectionAndPoseSelector:
    def __init__(self, slots, occupied, custom_obs, target_id):
        self.start_pose, self.done = None, False
        self.target_slot = next(s for s in slots if s['id'] == target_id)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.canvas.manager.set_window_title('Stage 3: Pose')
        plt.subplots_adjust(bottom=0.20)
        self.exit_btn = add_exit_btn(self.fig) 
        self.setup_plot()
        self.ax.set_title("STAGE 4: Position Car (Drag to Orient)\nFORWARD PARKING ONLY", fontsize=10, fontweight='bold')
        
        self.obs_list = [box(0,0,MAP_SIZE,0), box(0,MAP_SIZE,MAP_SIZE,MAP_SIZE), box(0,0,0,MAP_SIZE), box(MAP_SIZE,0,MAP_SIZE,MAP_SIZE)]
        for o in custom_obs: self.obs_list.append(translate(rotate(box(-5,-5,5,5), o[2], use_radians=True), o[0], o[1]))
        for s in slots:
            p = translate(rotate(box(-SLOT_D/2,-SLOT_W/2,SLOT_D/2,SLOT_W/2), s['yaw'], use_radians=True), s['cx'], s['cy'])
            if s['id'] == target_id: self.ax.fill(*p.exterior.xy, color='#66ff66', alpha=0.4)
            elif s['id'] in occupied: self.obs_list.append(p); self.ax.fill(*p.exterior.xy, color='#ff6666', alpha=0.4)
            else: self.ax.plot(*p.exterior.xy, color='#333333', linestyle=':')
        for o in self.obs_list: 
            if isinstance(o, Polygon): self.ax.fill(*o.exterior.xy, color='#333333')

        self.start_point = None; self.point, = self.ax.plot([], [], 'b.', ms=10)
        self.ghost_patch, self.ghost_arrow, self.car_preview_patch, self.car_preview_arrow = None, None, None, None
        
        self.info_box = self.ax.text(0.02, 0.98, "Select Start Position...", transform=self.ax.transAxes, 
                                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self.collision_text = self.ax.text(MAP_SIZE/2, MAP_SIZE - 20, "", ha='center', color='red', fontsize=12, fontweight='bold')
        self.update_ghost_car() 
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def setup_plot(self):
        self.ax.set_xlim(0, MAP_SIZE); self.ax.set_ylim(0, MAP_SIZE); self.ax.set_aspect('equal'); self.ax.grid(True, linestyle=':', alpha=0.6)
        
    def get_target_yaw(self): 
        return normalize_angle(self.target_slot['yaw'] + np.pi)
        
    def update_ghost_car(self):
        if self.ghost_patch: self.ghost_patch.remove(); self.ghost_patch = None
        if self.ghost_arrow: self.ghost_arrow.remove(); self.ghost_arrow = None
        yaw = self.get_target_yaw()
        cx, cy = self.target_slot['cx'], self.target_slot['cy']
        poly = get_car_polygon(cx, cy, yaw)
        self.ghost_patch = patches.Polygon(np.array(poly.exterior.coords), fc='none', ec='black', lw=2, linestyle='--', alpha=0.7)
        self.ax.add_patch(self.ghost_patch)
        arrow_len = CAR_L / 2
        self.ghost_arrow = self.ax.arrow(cx, cy, arrow_len*np.cos(yaw), arrow_len*np.sin(yaw), head_width=4, head_length=4, fc='black', ec='black', alpha=0.7)
        self.fig.canvas.draw()

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
            if any(poly.intersects(obs) for obs in self.obs_list if isinstance(obs, Polygon)):
                self.collision_text.set_text("[!] COLLISION! Select new position.")
                self.start_point = None; self.point.set_data([], [])
                if self.car_preview_patch: self.car_preview_patch.remove(); self.car_preview_patch = None
                if self.car_preview_arrow: self.car_preview_arrow.remove(); self.car_preview_arrow = None
                self.fig.canvas.draw()
            else:
                self.start_pose = (self.start_point[0], self.start_point[1], yaw)
                self.done = True; plt.close(self.fig)

    def on_move(self, event):
        if self.start_point is None or event.inaxes != self.ax: return
        yaw = math.atan2(event.ydata - self.start_point[1], event.xdata - self.start_point[0])
        self.info_box.set_text(f"Pos: ({self.start_point[0]:.1f}, {self.start_point[1]:.1f})\nAngle: {np.rad2deg(normalize_angle(yaw)):.1f}°")
        if self.car_preview_patch: self.car_preview_patch.remove(); self.car_preview_patch = None
        if self.car_preview_arrow: self.car_preview_arrow.remove(); self.car_preview_arrow = None
        poly = get_car_polygon(self.start_point[0], self.start_point[1], yaw)
        self.car_preview_patch = patches.Polygon(np.array(poly.exterior.coords), fc='cyan', ec='black', alpha=0.6)
        self.ax.add_patch(self.car_preview_patch)
        arrow_len = CAR_L / 2
        self.car_preview_arrow = self.ax.arrow(self.start_point[0], self.start_point[1], arrow_len * np.cos(yaw), arrow_len * np.sin(yaw), head_width=3, head_length=3, fc='red', ec='red', zorder=10)
        self.fig.canvas.draw()

class SummonSelector:
    def __init__(self, start_pose, obs_list):
        self.start_pose, self.done = start_pose, False
        self.obs_list = obs_list
        self.target_pose = None
        
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.canvas.manager.set_window_title('Summon Target')
        plt.subplots_adjust(bottom=0.15)
        self.exit_btn = add_exit_btn(self.fig) 
        
        self.ax.set_xlim(0, MAP_SIZE); self.ax.set_ylim(0, MAP_SIZE); self.ax.set_aspect('equal'); self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.set_title("SUMMON MODE: Click and Drag to set destination pose\nRed Arrow = Front of Car", fontsize=10)
        
        # Render Environment and Parked Car
        for o in self.obs_list:
            if isinstance(o, Polygon): self.ax.fill(*o.exterior.xy, color='#333333')
            
        current_poly = get_car_polygon(self.start_pose[0], self.start_pose[1], self.start_pose[2])
        self.ax.fill(*current_poly.exterior.xy, color='#00ced1', alpha=0.4)
        arrow_len = CAR_L / 2
        self.ax.arrow(self.start_pose[0], self.start_pose[1], arrow_len * np.cos(self.start_pose[2]), arrow_len * np.sin(self.start_pose[2]), head_width=3, head_length=3, fc='black', ec='black')
        
        self.start_point = None; self.point, = self.ax.plot([], [], 'b.', ms=10)
        self.car_preview_patch, self.car_preview_arrow = None, None
        
        self.info_box = self.ax.text(0.02, 0.98, "Select Summon Destination...", transform=self.ax.transAxes, 
                                     fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self.collision_text = self.ax.text(MAP_SIZE/2, MAP_SIZE - 20, "", ha='center', color='red', fontsize=12, fontweight='bold')
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.start_point is None: 
            self.start_point = (event.xdata, event.ydata)
            self.point.set_data([event.xdata], [event.ydata])
            self.collision_text.set_text("Drag to set heading...")
            self.info_box.set_text("Destination selected. Drag for Angle.")
            self.fig.canvas.draw()
        else: 
            yaw = math.atan2(event.ydata - self.start_point[1], event.xdata - self.start_point[0])
            poly = get_car_polygon(self.start_point[0], self.start_point[1], yaw, padding=0.0)
            if any(poly.intersects(obs) for obs in self.obs_list if isinstance(obs, Polygon)):
                self.collision_text.set_text("[!] COLLISION! Select new destination.")
                self.start_point = None; self.point.set_data([], [])
                if self.car_preview_patch: self.car_preview_patch.remove(); self.car_preview_patch = None
                if self.car_preview_arrow: self.car_preview_arrow.remove(); self.car_preview_arrow = None
                self.fig.canvas.draw()
            else:
                self.target_pose = (self.start_point[0], self.start_point[1], yaw)
                self.done = True; plt.close(self.fig)

    def on_move(self, event):
        if self.start_point is None or event.inaxes != self.ax: return
        yaw = math.atan2(event.ydata - self.start_point[1], event.xdata - self.start_point[0])
        self.info_box.set_text(f"Target: ({self.start_point[0]:.1f}, {self.start_point[1]:.1f})\nAngle: {np.rad2deg(normalize_angle(yaw)):.1f}°")
        if self.car_preview_patch: self.car_preview_patch.remove(); self.car_preview_patch = None
        if self.car_preview_arrow: self.car_preview_arrow.remove(); self.car_preview_arrow = None
        poly = get_car_polygon(self.start_point[0], self.start_point[1], yaw)
        self.car_preview_patch = patches.Polygon(np.array(poly.exterior.coords), fc='#98fb98', ec='black', alpha=0.6)
        self.ax.add_patch(self.car_preview_patch)
        arrow_len = CAR_L / 2
        self.car_preview_arrow = self.ax.arrow(self.start_point[0], self.start_point[1], arrow_len * np.cos(yaw), arrow_len * np.sin(yaw), head_width=3, head_length=3, fc='magenta', ec='magenta', zorder=10)
        self.fig.canvas.draw()


# --- HYBRID A* (REALISTIC + SAFE) ---
def hybrid_a_star(start, goal, obstacles, mode="FORWARD"):
    gx, gy, gyaw = goal
    grid_res, yaw_res = 2.0, np.deg2rad(5.0)
    rs_planner = ReedsSheppPlanner(step_size=RS_STEP_SIZE, min_r=MIN_TURN_RADIUS)

    # --- DİNAMİK TOLERANSLAR VE HIZLANDIRICILAR ---
    xy_tol = 15.0 if mode == "SUMMON" else FINAL_XY_TOLERANCE
    yaw_tol = np.deg2rad(15.0) if mode == "SUMMON" else FINAL_YAW_TOLERANCE
    hw = 4.0 if mode == "SUMMON" else H_WEIGHT
    analytic_dist = 50.0 if mode == "SUMMON" else ANALYTIC_SHOT_DIST

    # Merkezi Maliyet (Heuristic) Fonksiyonu
    def calc_heuristic(nx, ny, nyaw):
        dist_to_goal = math.hypot(nx - gx, ny - gy)
        angle_err = abs(normalize_angle(nyaw - gyaw))
        
        if mode == "SUMMON":
            h_angle = angle_err * 10.0 if angle_err > yaw_tol else 0.0
            return (dist_to_goal + h_angle) * hw
            
        # Aracın park çizgisine olan dik (lateral) uzaklığı
        cross_track_err = abs((nx - gx) * math.sin(gyaw) - (ny - gy) * math.cos(gyaw))
        proximity = math.exp(-dist_to_goal / 40.0)
        alignment_cost = cross_track_err * proximity * 2.0
        angle_cost = angle_err * (10.0 + 15.0 * proximity)
        
        return (dist_to_goal + angle_cost + alignment_cost) * hw

    h_start = calc_heuristic(start[0], start[1], start[2])
    start_node = Node(*start, 0, h_start)
    
    open_l = [start_node]; closed = {}
    iter_count = 0
    MAX_ITER = 80000 
    
    print(f"Computing Path [{mode}] (Safety Margin: {SAFE_MARGIN})...")
    
    while open_l and iter_count < MAX_ITER:
        iter_count += 1
        curr = heapq.heappop(open_l)
        
        key = (int(curr.x/grid_res), int(curr.y/grid_res), int(curr.yaw/yaw_res))
        if key in closed and closed[key] <= curr.g: continue
        closed[key] = curr.g

        dist_to_goal = np.hypot(curr.x-gx, curr.y-gy)
        
        # 1. ERKEN DURMA (Dinamik Tolerans Kullanımı)
        if dist_to_goal <= xy_tol and abs(normalize_angle(curr.yaw - gyaw)) <= yaw_tol:
            ang_err = np.rad2deg(abs(normalize_angle(curr.yaw - gyaw)))
            print(f"[OK] Hedefe ulaşıldı (A* Tolerans içi: {ang_err:.1f}° sapma)! Adım: {iter_count}")
            path = []
            temp = curr
            while temp:
                path.append((temp.x, temp.y, temp.yaw, temp.direction, temp.steering))
                temp = temp.parent
            path = path[::-1]
            path = trim_path(path, min_tail_dist=15.0)
            return path, True
        
        # Reeds-Shepp Atış Güvenlik Kilidi
        cross_track_err = abs((curr.x - gx) * math.sin(gyaw) - (curr.y - gy) * math.cos(gyaw))
        curr_angle_err = abs(normalize_angle(curr.yaw - gyaw))
        
        is_aligned = (cross_track_err < 10.0 and curr_angle_err < np.deg2rad(20.0))
        can_shoot = (mode == "SUMMON") or (dist_to_goal < 80.0 and is_aligned) or (dist_to_goal < 15.0)
        
        # 2. ANALİTİK ATIŞ
        if dist_to_goal < analytic_dist and can_shoot:
            if mode == "SUMMON":
                # Summon için sadece 1-2 açı test et (Hızlandırma)
                test_yaws = [gyaw]
                if curr_angle_err <= yaw_tol:
                    test_yaws.append(curr.yaw)
            else:
                test_yaws = [
                    gyaw,
                    normalize_angle(gyaw + yaw_tol * 0.5),
                    normalize_angle(gyaw - yaw_tol * 0.5),
                    normalize_angle(gyaw + yaw_tol),
                    normalize_angle(gyaw - yaw_tol)
                ]
                if curr_angle_err <= yaw_tol:
                    test_yaws.append(curr.yaw)
            
            best_rs_cost = float('inf')
            best_rs_data = None
            
            for tyaw in test_yaws:
                rs_path = rs_planner.get_optimal_path(curr.x, curr.y, curr.yaw, gx, gy, tyaw)
                if rs_path:
                    rx, ry, ryaw, rd, rs = rs_path
                    
                    if len(rx) > 1:
                        err_x = gx - rx[-1]
                        err_y = gy - ry[-1]
                        err_yaw = normalize_angle(tyaw - ryaw[-1])
                        
                        if np.hypot(err_x, err_y) < 1.0:
                            path_len = len(rx)
                            for i in range(path_len):
                                ratio = i / (path_len - 1)
                                rx[i] += err_x * ratio
                                ry[i] += err_y * ratio
                                ryaw[i] = normalize_angle(ryaw[i] + err_yaw * ratio)

                    if np.hypot(rx[-1] - gx, ry[-1] - gy) < 0.1:
                        collision = False
                        for i in range(len(rx)):
                            if not (0 < rx[i] < MAP_SIZE and 0 < ry[i] < MAP_SIZE): 
                                collision = True; break
                            cp = get_car_polygon(rx[i], ry[i], ryaw[i], SAFE_MARGIN)
                            if any(cp.intersects(o) for o in obstacles): 
                                collision = True; break
                        
                        if not collision:
                            path_cost = 0.0
                            switches = 0
                            for i in range(1, len(rx)):
                                path_cost += np.hypot(rx[i]-rx[i-1], ry[i]-ry[i-1])
                                if rd[i] != rd[i-1]:
                                    path_cost += COST_GEAR_SWITCH
                                    switches += 1
                            
                            if mode == "SUMMON" and switches > 0:
                                continue 
                            
                            ang_err_rad = abs(normalize_angle(tyaw - gyaw))
                            path_cost += ang_err_rad * 20.0 
                            
                            if path_cost < best_rs_cost:
                                best_rs_cost = path_cost
                                best_rs_data = (rx, ry, ryaw, rd, rs, tyaw, switches)
            
            if best_rs_data:
                rx, ry, ryaw, rd, rs, tyaw, switches = best_rs_data
                ang_err_deg = np.rad2deg(abs(normalize_angle(tyaw - gyaw)))
                print(f"[OK] Analytic Shot Success! Seçilen Sapma: {ang_err_deg:.1f}°, Vites: {switches}, Maliyet: {best_rs_cost:.1f}")
                
                path = []
                temp = curr
                while temp:
                    path.append((temp.x, temp.y, temp.yaw, temp.direction, temp.steering))
                    temp = temp.parent
                path = path[::-1]
                for i in range(len(rx)):
                    path.append((rx[i], ry[i], ryaw[i], rd[i], rs[i]))
                    
                path = trim_path(path, min_tail_dist=15.0)
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
                 
                MIN_MANEUVER_DIST = 10.0 

                if curr.parent is not None:
                    step_dist_step = np.hypot(nx - curr.x, ny - curr.y)
                    if ndir != curr.direction and step_dist_step < MIN_MANEUVER_DIST:
                        cost_gear += 25.0

                cost_smooth = abs(delta - curr.steering) * COST_STEER_CHANGE
                new_g = curr.g + cost_move + cost_steer + cost_gear + cost_smooth
                
                h = calc_heuristic(nx, ny, nyaw)
                heapq.heappush(open_l, Node(nx, ny, nyaw, new_g, h, curr, ndir, delta))
    
    print("[X] Timeout / No Path Found.")
    return [], False

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
    root = tk.Tk()
    root.withdraw() 

    while True: # Main application loop allowing full restarts
        slots = create_slots()
        
        ed = ObstacleEditor(slots); plt.show()
        if not hasattr(ed, 'custom_obstacles'): return # Exited
        
        sel = MapSelector(slots, ed.custom_obstacles); plt.show()
        if not sel.done or sel.target_id is None: return
        dp_sel = DirectionAndPoseSelector(slots, sel.occupied_ids, ed.custom_obstacles, sel.target_id); plt.show()
        if not dp_sel.done: return
        plt.close('all')

        obs_list = [box(0,0,MAP_SIZE,0), box(0,MAP_SIZE,MAP_SIZE,MAP_SIZE), box(0,0,0,MAP_SIZE), box(MAP_SIZE,0,MAP_SIZE,MAP_SIZE)]
        for o in ed.custom_obstacles: obs_list.append(translate(rotate(box(-5,-5,5,5), o[2], use_radians=True), o[0], o[1]))
        
        target = next(s for s in slots if s['id'] == sel.target_id)
        for s in slots:
            if s['id'] in sel.occupied_ids:
                p = translate(rotate(box(-SLOT_D/2,-SLOT_W/2,SLOT_D/2,SLOT_W/2), s['yaw'], use_radians=True), s['cx'], s['cy'])
                obs_list.append(p)

        PARK_INSET_CM = 10.0
        wall_dx = -np.cos(target["yaw"])
        wall_dy = -np.sin(target["yaw"])
        goal_in_x = float(target["cx"] + PARK_INSET_CM * wall_dx)
        goal_in_y = float(target["cy"] + PARK_INSET_CM * wall_dy)

        # Sadece FORWARD park hedefini zorunlu kıldık
        primary_yaw = normalize_angle(target["yaw"] + np.pi)
        primary_x = goal_in_x
        primary_y = goal_in_y
        mode_str = "FORWARD"

        current_start = dp_sel.start_pose
        current_target = (primary_x, primary_y, primary_yaw)
        is_summon = False

        while True: 
            if not is_summon:
                print(f"\n[CAR] Attempting {mode_str} parking (User Selection)...")
                path, success = hybrid_a_star(current_start, current_target, obs_list, mode=mode_str)
                
                final_mode_str = mode_str
                chosen_goal_x, chosen_goal_y, final_yaw = current_target

                if not success:
                    print(f"\n[!]  {mode_str} PARKING FAILED.")
                    messagebox.showerror("Error", f"{mode_str} parking is impossible from this position.")
                    break # Ana döngüye, yani harita seçimine geri dön
            else:
                print(f"\n[CAR] Attempting SUMMON to selected target...")
                path, success = hybrid_a_star(current_start, current_target, obs_list, mode="SUMMON")
                
                final_mode_str = "SUMMON"
                chosen_goal_x, chosen_goal_y, final_yaw = current_target
                
                if not success:
                    messagebox.showerror("Error", "Could not find a valid path to the summon target!")
                    sum_sel = SummonSelector(current_start, obs_list)
                    plt.show()
                    if sum_sel.done:
                        current_target = sum_sel.target_pose
                        continue
                    else:
                        return

            target_pose = (chosen_goal_x, chosen_goal_y, final_yaw)
            try:
                save_path_json(path, current_start, sel.target_id if not is_summon else -1, target_pose, filename="planned_path.json")
                print("[OK] Wrote planned path to planned_path.json")
            except Exception as e:
                print(f"[!] Could not write planned_path.json: {e}")

            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(9, 10))
            ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE); ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)
            exit_btn = add_exit_btn(fig) 

            if is_summon:
                title_text = "SUCCESS (SUMMON)"
                title_color = '#8A2BE2'
            else:
                title_text = f"SUCCESS ({final_mode_str})"
                title_color = '#006400' 

            ax.set_title(title_text, color=title_color, fontweight='bold', fontsize=12)

            for o in obs_list: 
                if isinstance(o, Polygon):
                    ax.fill(*o.exterior.xy, color='#2f4f4f') 
                    
            p_target = translate(rotate(box(-SLOT_D/2,-SLOT_W/2,SLOT_D/2,SLOT_W/2), target['yaw'], use_radians=True), target['cx'], target['cy'])
            ax.fill(*p_target.exterior.xy, color='#98fb98', alpha=0.5) 
            
            if is_summon:
                ax.plot(chosen_goal_x, chosen_goal_y, 'mx', markersize=10, markeredgewidth=2)
            else:
                ax.plot(chosen_goal_x, chosen_goal_y, 'rx', markersize=8)

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
            
            action_state = {"action": "exit"}

            ax_stop = plt.axes([0.46, 0.01, 0.1, 0.05])
            btn_stop = Button(ax_stop, 'STOP', color='#ff4d4d', hovercolor='#ff3333')
            btn_stop.on_clicked(ctrl.stop)
            
            ax_summon = plt.axes([0.58, 0.01, 0.12, 0.05])
            btn_summon = Button(ax_summon, 'SUMMON', color='#8A2BE2', hovercolor='#9370DB')
            btn_summon.label.set_color('white')
            
            def on_summon_click(event):
                action_state["action"] = "summon"
                plt.close(fig)
            btn_summon.on_clicked(on_summon_click)

            ax_restart = plt.axes([0.72, 0.01, 0.14, 0.05])
            btn_restart = Button(ax_restart, 'RESTART', color='#2E8B57', hovercolor='#3CB371')
            btn_restart.label.set_color('white')
            
            def on_restart_click(event):
                action_state["action"] = "restart"
                plt.close(fig)
            btn_restart.on_clicked(on_restart_click)
            
            plt.show() 

            if action_state["action"] == "restart":
                break 
            elif action_state["action"] == "summon":
                current_start = path[-1][:3] 
                
                sum_sel = SummonSelector(current_start, obs_list)
                plt.show()
                if sum_sel.done:
                    current_target = sum_sel.target_pose
                    is_summon = True
                else:
                    return 
            else:
                return 

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    run_simulation()