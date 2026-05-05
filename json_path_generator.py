import json
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from apriltag_pose import get_latest_map_pose


MAP_SIZE = 200.0  # cm
AXLE_FRONT_TO_REAR_CM = 16.0  # cm (front axle to rear axle)

# Slot geometry (same as pathcreation.py)
SLOT_W, SLOT_D = 25.0, 40.0  # cm


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def create_slots():
    # Slot midpoints (cx, cy) in cm (same as your current pathcreation.py).
    # Yaw convention: 0 rad = +X (east), CCW positive.
    return [
        {"id": 1, "cx": 112.5, "cy": 180.0, "yaw": -math.pi / 2},
        {"id": 2, "cx": 180.0, "cy": 137.5, "yaw": math.pi},
        {"id": 3, "cx": 180.0, "cy": 112.5, "yaw": math.pi},
        {"id": 4, "cx": 180.0, "cy": 87.5, "yaw": math.pi},
        {"id": 5, "cx": 180.0, "cy": 62.5, "yaw": math.pi},
        {"id": 6, "cx": 112.5, "cy": 20.0, "yaw": math.pi / 2},
        {"id": 7, "cx": 87.5, "cy": 20.0, "yaw": math.pi / 2},
        {"id": 8, "cx": 20.0, "cy": 62.5, "yaw": 0.0},
        {"id": 9, "cx": 20.0, "cy": 87.5, "yaw": 0.0},
        {"id": 10, "cx": 20.0, "cy": 112.5, "yaw": 0.0},
        {"id": 11, "cx": 20.0, "cy": 137.5, "yaw": 0.0},
        {"id": 12, "cx": 87.5, "cy": 180.0, "yaw": -math.pi / 2},
    ]


def draw_slot_rect(ax, cx: float, cy: float, yaw: float, color="white", alpha=0.9, edge="#333333"):
    # Rectangle centered at (cx,cy) with SLOT_D along local X and SLOT_W along local Y.
    # Same as pathcreation: it uses box(-D/2,-W/2,D/2,W/2) rotated by yaw.
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = SLOT_D / 2.0, SLOT_W / 2.0
    # Local corners
    corners = np.array(
        [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]], dtype=float
    )
    R = np.array([[c, -s], [s, c]], dtype=float)
    rc = corners @ R.T
    rc[:, 0] += cx
    rc[:, 1] += cy
    ax.fill(rc[:, 0], rc[:, 1], color=color, alpha=alpha, edgecolor=edge, linewidth=1.5)


@dataclass
class ControlPoint:
    x: float
    y: float
    direction: int  # +1 forward, -1 reverse (for the segment starting at this point)


def resample_polyline(points: List[ControlPoint], step_cm: float) -> List[Tuple[float, float, float, int, float]]:
    """
    Convert control points into evenly spaced path points.

    Output tuples: (x_cm, y_cm, yaw_rad, direction, steer_rad)
    - yaw_rad is vehicle heading.
      For forward: heading aligns with segment tangent.
      For reverse: heading points opposite of motion (tangent + pi).
    - steer_rad is set to 0 (Stanley controller doesn't require it).
    """
    if len(points) < 2:
        return []

    out = []
    step_cm = max(0.5, float(step_cm))

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            continue

        tangent = math.atan2(dy, dx)  # motion direction if driving forward along the segment
        direction = int(p0.direction)

        # Vehicle heading (front direction)
        yaw = tangent if direction > 0 else normalize_angle(tangent + math.pi)

        n = max(1, int(math.floor(seg_len / step_cm)))
        for k in range(n):
            t = (k * step_cm) / seg_len
            x_front = p0.x + t * dx
            y_front = p0.y + t * dy
            if direction < 0:
                # Follow rear axle in reverse, keep yaw unchanged.
                x = x_front - AXLE_FRONT_TO_REAR_CM * math.cos(yaw)
                y = y_front - AXLE_FRONT_TO_REAR_CM * math.sin(yaw)
            else:
                x, y = x_front, y_front
            out.append((float(x), float(y), float(yaw), direction, 0.0))

    # Add final point with last segment's yaw/direction
    last_dir = int(points[-2].direction)
    dx = points[-1].x - points[-2].x
    dy = points[-1].y - points[-2].y
    tangent = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-6 else 0.0
    last_yaw = tangent if last_dir > 0 else normalize_angle(tangent + math.pi)
    end_x_front = float(points[-1].x)
    end_y_front = float(points[-1].y)
    if last_dir < 0:
        end_x = end_x_front - AXLE_FRONT_TO_REAR_CM * math.cos(last_yaw)
        end_y = end_y_front - AXLE_FRONT_TO_REAR_CM * math.sin(last_yaw)
    else:
        end_x, end_y = end_x_front, end_y_front
    out.append((float(end_x), float(end_y), float(last_yaw), last_dir, 0.0))

    # Make sure the very first point direction matches first motion (like pathcreation fix)
    if len(out) >= 2:
        x0, y0, yaw0, _, _ = out[0]
        _, _, _, d1, _ = out[1]
        out[0] = (x0, y0, yaw0, int(d1), 0.0)

    return out


class PathGeneratorUI:
    def __init__(self):
        self.step_cm = 2.0
        self.current_direction = 1
        self.control_points: List[ControlPoint] = []
        self.start_pose: Optional[Tuple[float, float, float]] = None  # (x,y,yaw) in std rad

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("JSON Path Generator (test)")
        plt.subplots_adjust(bottom=0.14)
        self._setup_map()

        # UI buttons
        ax_fwd = plt.axes([0.02, 0.02, 0.14, 0.06])
        ax_rev = plt.axes([0.17, 0.02, 0.14, 0.06])
        ax_clear = plt.axes([0.32, 0.02, 0.14, 0.06])
        ax_export = plt.axes([0.47, 0.02, 0.18, 0.06])
        ax_step_minus = plt.axes([0.68, 0.02, 0.06, 0.06])
        ax_step_plus = plt.axes([0.75, 0.02, 0.06, 0.06])
        ax_exit = plt.axes([0.83, 0.02, 0.14, 0.06])

        self.btn_fwd = Button(ax_fwd, "FWD", color="#e0e0e0", hovercolor="#c0c0c0")
        self.btn_rev = Button(ax_rev, "REV", color="white", hovercolor="#c0c0c0")
        self.btn_clear = Button(ax_clear, "CLEAR", color="white", hovercolor="#c0c0c0")
        self.btn_export = Button(ax_export, "EXPORT JSON", color="#2E8B57", hovercolor="#3CB371")
        self.btn_step_minus = Button(ax_step_minus, "-", color="#e0e0e0", hovercolor="#c0c0c0")
        self.btn_step_plus = Button(ax_step_plus, "+", color="#e0e0e0", hovercolor="#c0c0c0")
        self.btn_exit = Button(ax_exit, "EXIT", color="#8b0000", hovercolor="#ff4d4d")

        self.btn_fwd.on_clicked(lambda event: self._set_dir(1))
        self.btn_rev.on_clicked(lambda event: self._set_dir(-1))
        self.btn_clear.on_clicked(self._clear)
        self.btn_export.on_clicked(self._export)
        self.btn_step_minus.on_clicked(lambda event: self._change_step(-0.5))
        self.btn_step_plus.on_clicked(lambda event: self._change_step(+0.5))
        self.btn_exit.on_clicked(lambda event: plt.close(self.fig))

        self.info = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        # Drawing artists
        (self.cp_line,) = self.ax.plot([], [], "k--", lw=1.5, alpha=0.6, label="control polyline")
        (self.path_line,) = self.ax.plot([], [], "b-", lw=2.0, alpha=0.8, label="resampled path")
        (self.car_dot,) = self.ax.plot([], [], "go", ms=8, label="car start")
        (self.car_arrow,) = self.ax.plot([], [], "g-", lw=2)
        self.ax.legend(loc="upper right")

        # Initialize start pose from AprilTag
        self._set_start_from_apriltag()

        self._set_dir(self.current_direction)
        self._update_plot()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _setup_map(self):
        self.ax.set_xlim(0, MAP_SIZE)
        self.ax.set_ylim(0, MAP_SIZE)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle=":", alpha=0.5)
        self.ax.set_title("Click to add points. Right click to undo.\nToggle gear while drawing. Export creates planned_path.json")

        # boundary
        self.ax.plot([0, MAP_SIZE, MAP_SIZE, 0, 0], [0, 0, MAP_SIZE, MAP_SIZE, 0], "k-", lw=2)

        # slots
        for s in create_slots():
            draw_slot_rect(self.ax, s["cx"], s["cy"], s["yaw"])
            self.ax.text(s["cx"], s["cy"], str(s["id"]), ha="center", va="center", fontweight="bold")

    def _set_dir(self, d: int):
        self.current_direction = 1 if d >= 0 else -1
        self.btn_fwd.color = "#e0e0e0" if self.current_direction > 0 else "white"
        self.btn_rev.color = "#e0e0e0" if self.current_direction < 0 else "white"
        # Direction is stored on the START of the next segment (the last control point).
        # When switching gear, apply it to the last control point so the next drawn segment
        # uses the selected direction.
        if self.control_points:
            self.control_points[-1].direction = self.current_direction
        self.fig.canvas.draw_idle()
        self._update_plot()

    def _change_step(self, delta: float):
        self.step_cm = max(0.5, round(self.step_cm + delta, 2))
        self._update_plot()

    def _clear(self, event=None):
        # Keep the start point (index 0) if it exists
        if self.start_pose is not None:
            x, y, _ = self.start_pose
            self.control_points = [ControlPoint(x=x, y=y, direction=self.current_direction)]
        else:
            self.control_points = []
        self._update_plot()

    def _set_start_from_apriltag(self):
        """
        Reads a single pose from apriltag_pose and sets it as the first control point.
        yaw conversion matches pathcreation:
          apriltag yaw: 0°=+Y, CCW+
          planner yaw:  0 rad=+X, CCW+   => yaw_rad = deg2rad(yaw_deg) + pi/2
        """
        p = get_latest_map_pose(timeout_s=8.0, target_fps=10.0)
        if p is None:
            print("[X] No AprilTag detected. Start pose not set.")
            return
        x = float(p["x_cm"])
        y = float(p["y_cm"])
        yaw = normalize_angle(math.radians(float(p["yaw_deg"])) + (math.pi / 2.0))
        self.start_pose = (x, y, yaw)
        # Set/overwrite first control point as start
        if self.control_points:
            self.control_points[0] = ControlPoint(x=x, y=y, direction=self.current_direction)
        else:
            self.control_points.append(ControlPoint(x=x, y=y, direction=self.current_direction))
        # Ensure start point direction matches current gear for the first segment.
        self.control_points[-1].direction = self.current_direction

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:  # right click undo
            if self.control_points:
                self.control_points.pop()
                self._update_plot()
            return
        if event.button != 1:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        if not (0.0 <= x <= MAP_SIZE and 0.0 <= y <= MAP_SIZE):
            return
        # Segment direction is stored on the previous point. Make sure it matches current gear.
        if self.control_points:
            self.control_points[-1].direction = self.current_direction
        self.control_points.append(ControlPoint(x=x, y=y, direction=self.current_direction))
        self._update_plot()

    def _update_plot(self):
        cxs = [p.x for p in self.control_points]
        cys = [p.y for p in self.control_points]
        self.cp_line.set_data(cxs, cys)

        path = resample_polyline(self.control_points, self.step_cm)
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        self.path_line.set_data(xs, ys)

        # Draw car start pose as dot + arrow
        if self.start_pose is not None:
            sx, sy, syaw = self.start_pose
            self.car_dot.set_data([sx], [sy])
            L = 12.0
            self.car_arrow.set_data([sx, sx + L * math.cos(syaw)], [sy, sy + L * math.sin(syaw)])
        else:
            self.car_dot.set_data([], [])
            self.car_arrow.set_data([], [])

        self.info.set_text(
            f"gear={'FWD' if self.current_direction>0 else 'REV'} | step={self.step_cm:.2f}cm\n"
            f"control_points={len(self.control_points)} | json_points={len(path)}"
        )
        self.fig.canvas.draw_idle()

    def _export(self, event=None):
        path = resample_polyline(self.control_points, self.step_cm)
        if len(path) < 2:
            print("[X] Need at least 2 points.")
            return

        # Use AprilTag start pose for JSON start if available.
        # If the first segment is reverse, convert start (front axle) -> rear axle for consistency.
        if self.start_pose is not None:
            sx, sy, syaw = self.start_pose
            first_dir = int(path[0][3])
            if first_dir < 0:
                sx = sx - AXLE_FRONT_TO_REAR_CM * math.cos(syaw)
                sy = sy - AXLE_FRONT_TO_REAR_CM * math.sin(syaw)
            start_pose = (sx, sy, syaw)
        else:
            start_pose = (path[0][0], path[0][1], path[0][2])
        target_pose = (path[-1][0], path[-1][1], path[-1][2])

        data = {
            "version": 1,
            "generated_at_unix_s": time.time(),
            "map_size_cm": MAP_SIZE,
            "start": {"x_cm": float(start_pose[0]), "y_cm": float(start_pose[1]), "yaw_rad": float(start_pose[2])},
            "target": {
                "slot_id": 0,  # test generator (no real slot selection)
                "x_cm": float(target_pose[0]),
                "y_cm": float(target_pose[1]),
                "yaw_rad": float(target_pose[2]),
            },
            "points": [
                {"x_cm": float(x), "y_cm": float(y), "yaw_rad": float(yaw), "direction": int(d), "steer_rad": float(steer)}
                for (x, y, yaw, d, steer) in path
            ],
        }

        filename = "planned_path.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote {filename} with {len(path)} points (step={self.step_cm}cm)")


def main():
    PathGeneratorUI()
    plt.show()


if __name__ == "__main__":
    main()

