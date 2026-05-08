"""Pygame UI for planning a path from the AprilTag-detected start pose.

The UI is modelled on ``main.py``: same 200 x 200 cm canvas, same slot
table, same path rendering. The differences:

* The **start pose** is acquired live from
  ``apriltag_pose.AprilTagMapPoseTracker`` rather than placed by the
  user. The AprilTag returns the front-axle position, which is shifted
  by ``-WHEELBASE/2`` along the body yaw to produce the centre-frame
  pose the planner expects.
* The **goal** is picked by clicking a parking slot.
* "Plan" runs Hybrid A* from the (live) start pose to an *approach*
  pose 40 cm before the slot centre, then appends a 40 cm dead-straight
  forward segment ending at the slot centre.
* "Save JSON" writes ``planned_path.json`` (centre-frame points,
  ``{x_cm, y_cm, yaw_rad, direction, steer_rad}``) to be consumed by
  the controller.

Add/erase obstacles works exactly as in ``main.py`` for testing.

If the AprilTag tracker can't be opened (e.g. running on a dev machine
without the camera stack), the UI degrades gracefully: it shows a
"[no camera]" indicator and lets you click & drag on the canvas to
place the start pose manually, just like ``main.py``.

A headless CLI mode is still available with ``--cli --slot N``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import pygame

# Re-use everything renderable from the existing UI so the look is
# identical and the parking-slot table is single-sourced.
from main import (
    ALPHA_POSE_BODY_OVER_PATH,
    ALPHA_POSE_LINES_OVER_PATH,
    ALPHA_SLIDE_BODY,
    ALPHA_SLIDE_LINES,
    Button,
    CANVAS_PX,
    C_ANIM_CAR,
    C_ANIM_CAR_BORDER,
    C_AXIS,
    C_BG,
    C_BORDER,
    C_BUTTON,
    C_BUTTON_ACCENT,
    C_BUTTON_ACTIVE,
    C_BUTTON_ACTIVE_TEXT,
    C_BUTTON_HOVER,
    C_DRAG,
    C_GOAL,
    C_GRID,
    C_INVALID,
    C_OBSTACLE,
    C_OBSTACLE_BORDER,
    C_PANEL,
    C_PATH_FWD,
    C_PATH_REV,
    C_SLOT_FILL,
    C_SLOT_HOVER,
    C_SLOT_HOVER_OUTLINE,
    C_SLOT_OUTLINE,
    C_SLOT_TEXT,
    C_START,
    C_TEXT,
    C_TEXT_DIM,
    DEFAULT_STEP,
    PARKING_SLOTS,
    SCALE,
    SIDEBAR_PX,
    SLOT_LENGTH,
    SLOT_WIDTH,
    STEP_INCR,
    STEP_MAX,
    STEP_MIN,
    WINDOW_H,
    WINDOW_W,
    clamp,
    draw_car,
    draw_car_alpha,
    draw_polygon_world,
    screen_to_world,
    slot_at,
    slot_corners,
    world_to_screen,
)
from collision import (
    AREA_SIZE,
    OBSTACLE_HALF,
    OBSTACLE_SIZE,
    obstacle_at,
    state_is_valid,
)
from planner import PathSegment, PlanResult, plan
from vehicle import WHEELBASE, normalize_angle


# ----- planning constants -------------------------------------------------
APPROACH_OFFSET_CM = 40.0      # distance before slot centre we aim for first
STRAIGHT_DENSITY_CM = 1.0      # spacing of samples in the final straight 40 cm
PLANNER_STEP_SIZE = DEFAULT_STEP

# How often we ask the AprilTag tracker for a fresh fix in the UI.  The
# tracker grabs frames from two cameras, which is heavy, so we throttle.
POSE_POLL_PERIOD_S = 0.15

# Stale pose: warn the user when the most recent fix is older than this.
POSE_STALE_S = 1.0


# ----- frame conversions --------------------------------------------------

def front_axle_to_centre(x_front: float, y_front: float, yaw_rad: float) -> Tuple[float, float]:
    """Shift front-axle pose back to the car centre (planner frame)."""
    half_wb = WHEELBASE / 2.0
    return (
        x_front - half_wb * math.cos(yaw_rad),
        y_front - half_wb * math.sin(yaw_rad),
    )


def yaw_deg_to_planner_rad(yaw_deg: float) -> float:
    """Map AprilTag's map-frame yaw_deg to the planner's yaw_rad.

    Matches ``benjiereverseimprovement.py``: ``yaw_rad = yaw_deg + 90°``.
    """
    return normalize_angle(math.radians(float(yaw_deg)) + math.pi / 2.0)


# ----- path-point assembly ------------------------------------------------

def _segment_directions(poses: Sequence[Tuple[float, float, float]],
                        segments: Sequence[PathSegment]) -> List[int]:
    """Per-pose direction array aligned with ``poses``."""
    dirs = [1] * len(poses)
    if not segments:
        return dirs
    dirs[0] = int(segments[0].direction)
    idx = 1
    for seg in segments:
        n_new = len(seg.poses) - 1
        d = int(seg.direction)
        for _ in range(n_new):
            if idx < len(dirs):
                dirs[idx] = d
                idx += 1
    return dirs


def _approx_steer_rad(prev: Tuple[float, float, float],
                      curr: Tuple[float, float, float],
                      direction: int) -> float:
    """Approximate the bicycle-model steer angle that connects two poses."""
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    chord = math.hypot(dx, dy)
    if chord < 1e-9:
        return 0.0
    signed = chord * (1.0 if direction >= 0 else -1.0)
    dtheta = normalize_angle(curr[2] - prev[2])
    return math.atan2(dtheta * WHEELBASE, signed)


def build_path_points(plan_result: PlanResult,
                      approach_pose: Tuple[float, float, float],
                      goal_pose: Tuple[float, float, float]) -> List[dict]:
    """Convert a planner result + 40 cm straight tail into JSON-ready points."""
    poses = list(plan_result.poses)
    dirs = _segment_directions(poses, plan_result.segments)

    points: List[dict] = []
    for i, pose in enumerate(poses):
        x, y, th = pose
        if i == 0:
            steer = 0.0
        else:
            steer = _approx_steer_rad(poses[i - 1], pose, dirs[i])
        points.append({
            "x_cm": float(x),
            "y_cm": float(y),
            "yaw_rad": float(normalize_angle(th)),
            "direction": int(dirs[i]),
            "steer_rad": float(steer),
        })

    ax, ay, _ = approach_pose
    gx, gy, gyaw = goal_pose
    n_straight = max(2, int(math.ceil(APPROACH_OFFSET_CM / STRAIGHT_DENSITY_CM)))
    for k in range(1, n_straight + 1):
        u = k / n_straight
        points.append({
            "x_cm": float(ax + u * (gx - ax)),
            "y_cm": float(ay + u * (gy - ay)),
            "yaw_rad": float(normalize_angle(gyaw)),
            "direction": 1,
            "steer_rad": 0.0,
        })
    return points


def approach_pose_for_slot(slot: Tuple[int, float, float, float]) -> Tuple[float, float, float]:
    _, gx, gy, gyaw = slot
    return (
        gx - APPROACH_OFFSET_CM * math.cos(gyaw),
        gy - APPROACH_OFFSET_CM * math.sin(gyaw),
        gyaw,
    )


def get_slot(slot_id: int) -> Tuple[int, float, float, float]:
    for s in PARKING_SLOTS:
        if s[0] == slot_id:
            return s
    raise ValueError(f"Unknown parking slot id {slot_id}.")


def write_path_json(out_path: str,
                    points: List[dict],
                    *,
                    slot: Tuple[int, float, float, float],
                    start_pose_centre: Tuple[float, float, float],
                    obstacles: Sequence[Tuple[float, float]],
                    step_size: float,
                    relaxed: bool) -> None:
    sid, gx, gy, gyaw = slot
    approach = approach_pose_for_slot(slot)
    out_obj = {
        "metadata": {
            "frame": "centre",
            "slot_id": sid,
            "start_pose_centre": list(start_pose_centre),
            "approach_pose_centre": list(approach),
            "goal_pose_centre": [gx, gy, gyaw],
            "approach_offset_cm": APPROACH_OFFSET_CM,
            "wheelbase_cm": WHEELBASE,
            "planner_step_cm": step_size,
            "obstacles_cm": [list(o) for o in obstacles],
            "n_points": len(points),
            "relaxed": bool(relaxed),
        },
        "points": points,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)


# ----- AprilTag pose source ----------------------------------------------

class _PoseSource:
    """Wraps ``AprilTagMapPoseTracker`` so the UI can poll without crashing
    when the camera stack isn't available (e.g. on a dev laptop).
    """

    def __init__(self) -> None:
        self.tracker = None
        self.status = "tracker not started"
        self.last_raw: Optional[dict] = None
        self.last_pose_centre: Optional[Tuple[float, float, float]] = None
        self.last_t: float = 0.0
        try:
            from apriltag_pose import AprilTagMapPoseTracker  # noqa: WPS433
            self.tracker = AprilTagMapPoseTracker()
            self.status = "ok"
        except Exception as e:  # pragma: no cover - hardware dependent
            self.tracker = None
            self.status = f"unavailable ({e.__class__.__name__})"

    def poll(self) -> None:
        if self.tracker is None:
            return
        try:
            p = self.tracker.update()
        except Exception as e:  # pragma: no cover
            self.status = f"update error: {e.__class__.__name__}"
            return
        if p is None:
            return
        yaw = yaw_deg_to_planner_rad(float(p["yaw_deg"]))
        xc, yc = front_axle_to_centre(float(p["x_cm"]), float(p["y_cm"]), yaw)
        self.last_raw = p
        self.last_pose_centre = (xc, yc, yaw)
        self.last_t = time.perf_counter()
        self.status = "ok"

    def close(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception:
                pass
            self.tracker = None


# ----- UI app -------------------------------------------------------------

class App:
    """Pygame UI that pairs an AprilTag start pose with the Hybrid A* planner."""

    def __init__(self, out_path: str = "planned_path.json") -> None:
        pygame.init()
        pygame.display.set_caption("Path Planner — AprilTag start, save JSON")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.canvas = pygame.Surface((CANVAS_PX, CANVAS_PX))
        self.font = pygame.font.SysFont("Arial", 13)
        self.font_b = pygame.font.SysFont("Arial", 13, bold=True)
        self.font_t = pygame.font.SysFont("Arial", 16, bold=True)

        self.out_path = out_path

        # Live pose
        self.pose_source = _PoseSource()
        self._next_pose_poll_t = 0.0

        # Manual fallback start (only used if the tracker isn't available)
        self.manual_start: Optional[Tuple[float, float, float]] = None
        self.drag_anchor_world: Optional[Tuple[float, float]] = None
        self.drag_current_world: Optional[Tuple[float, float]] = None
        self.dragging_start = False

        # Goal & path
        self.selected_slot: Optional[Tuple[int, float, float, float]] = None
        self.hover_slot: Optional[Tuple[int, float, float, float]] = None
        self.obstacles: List[Tuple[float, float]] = []
        self.path_segments: List[PathSegment] = []
        self.path_poses: List[Tuple[float, float, float]] = []
        self.last_result: Optional[PlanResult] = None

        # Modes & status
        self.mode = "goal"   # "goal" | "obstacle" | "erase" | "start" (manual only)
        self.step_size = float(DEFAULT_STEP)
        self.is_planning = False
        self.status = "Click a parking slot to set the goal, then press Plan."

        self.buttons: List[Button] = []
        self._step_label_rect = pygame.Rect(0, 0, 0, 0)
        self._build_buttons()

        self.clock = pygame.time.Clock()
        self.running = True

    # ---------- buttons --------------------------------------------------
    def _build_buttons(self) -> None:
        x = CANVAS_PX + 18
        w = SIDEBAR_PX - 36
        y = 60
        h = 30

        for key, label in (
            ("goal", "Set Goal slot  (G)"),
            ("obstacle", "Add Obstacle  (O)"),
            ("erase", "Erase Obstacle  (E)"),
            ("start", "Manual Start  (S)"),
        ):
            self.buttons.append(Button(pygame.Rect(x, y, w, h), label, f"mode:{key}"))
            y += h + 6

        # Step size +/-
        y += 22
        bw_step = 36
        self.buttons.append(Button(pygame.Rect(x, y, bw_step, h), "−", "step:dec"))
        self.buttons.append(
            Button(pygame.Rect(x + w - bw_step, y, bw_step, h), "+", "step:inc")
        )
        self._step_label_rect = pygame.Rect(
            x + bw_step + 4, y, w - 2 * bw_step - 8, h
        )
        y += h + 18

        # Action buttons
        self.buttons.append(Button(pygame.Rect(x, y, w, h + 4), "Plan path  (Enter)", "plan"))
        y += h + 12
        self.buttons.append(Button(pygame.Rect(x, y, w, h + 4), "Save JSON  (W)", "save_json"))
        y += h + 12
        self.buttons.append(Button(pygame.Rect(x, y, w, h), "Refresh pose  (R)", "refresh_pose"))
        y += h + 8
        self.buttons.append(Button(pygame.Rect(x, y, w, h), "Clear path  (C)", "clear_path"))
        y += h + 6
        self.buttons.append(Button(pygame.Rect(x, y, w, h), "Clear all  (X)", "clear_all"))

    # ---------- pose handling -------------------------------------------
    def _maybe_poll_pose(self) -> None:
        now = time.perf_counter()
        if now < self._next_pose_poll_t:
            return
        self.pose_source.poll()
        self._next_pose_poll_t = now + POSE_POLL_PERIOD_S

    def _current_start_pose(self) -> Optional[Tuple[float, float, float]]:
        if self.pose_source.last_pose_centre is not None:
            return self.pose_source.last_pose_centre
        return self.manual_start

    def _pose_age_s(self) -> Optional[float]:
        if self.pose_source.last_t == 0.0:
            return None
        return time.perf_counter() - self.pose_source.last_t

    # ---------- main loop -----------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                self._maybe_poll_pose()
                for ev in pygame.event.get():
                    self._handle_event(ev)
                self._render()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            self.pose_source.close()
            pygame.quit()

    def _handle_event(self, ev: pygame.event.Event) -> None:
        if ev.type == pygame.QUIT:
            self.running = False
            return

        if ev.type == pygame.MOUSEMOTION:
            mx, my = ev.pos
            for b in self.buttons:
                b.hover = b.rect.collidepoint(mx, my)
            if self.dragging_start and 0 <= mx < CANVAS_PX:
                self.drag_current_world = screen_to_world(mx, my)
            if self.mode == "goal" and 0 <= mx < CANVAS_PX:
                wx, wy = screen_to_world(mx, my)
                self.hover_slot = slot_at(wx, wy)
            else:
                self.hover_slot = None
            return

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            if mx >= CANVAS_PX:
                self._click_panel(mx, my)
            else:
                self._click_canvas_down(mx, my)
            return

        if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            mx, my = ev.pos
            if self.dragging_start:
                self._click_canvas_up(mx, my)
            return

        if ev.type == pygame.KEYDOWN:
            self._handle_key(ev)

    def _handle_key(self, ev: pygame.event.Event) -> None:
        if ev.key == pygame.K_g:
            self._set_mode("goal")
        elif ev.key == pygame.K_o:
            self._set_mode("obstacle")
        elif ev.key == pygame.K_e:
            self._set_mode("erase")
        elif ev.key == pygame.K_s:
            self._set_mode("start")
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_p):
            self._do_plan()
        elif ev.key == pygame.K_w:
            self._do_save_json()
        elif ev.key == pygame.K_r:
            self._refresh_pose()
        elif ev.key == pygame.K_c:
            self._clear_path("Path cleared.")
        elif ev.key == pygame.K_x:
            self._clear_all()
        elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._adjust_step(+STEP_INCR)
        elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._adjust_step(-STEP_INCR)
        elif ev.key == pygame.K_ESCAPE:
            self.running = False

    # ---------- canvas interaction --------------------------------------
    def _click_canvas_down(self, mx: int, my: int) -> None:
        wx, wy = screen_to_world(mx, my)
        wx = clamp(wx, 0, AREA_SIZE)
        wy = clamp(wy, 0, AREA_SIZE)
        if self.mode == "goal":
            slot = slot_at(wx, wy)
            if slot is None:
                self.status = "Click a parking-slot rectangle to set the goal."
                return
            self._select_slot(slot)
        elif self.mode == "obstacle":
            self._add_obstacle(wx, wy)
        elif self.mode == "erase":
            self._erase_obstacle(wx, wy)
        elif self.mode == "start":
            # Manual fallback: drag to set a start pose if no AprilTag is available.
            self.dragging_start = True
            self.drag_anchor_world = (wx, wy)
            self.drag_current_world = (wx, wy)
            self.status = "Drag to set the manual start heading, release to commit."

    def _click_canvas_up(self, mx: int, my: int) -> None:
        if self.drag_anchor_world is None:
            self.dragging_start = False
            return
        wx, wy = screen_to_world(mx, my)
        ax, ay = self.drag_anchor_world
        dx, dy = wx - ax, wy - ay
        if math.hypot(dx, dy) < 1.0:
            theta = self.manual_start[2] if self.manual_start is not None else 0.0
        else:
            theta = math.atan2(dy, dx)
        pose = (clamp(ax, 0, AREA_SIZE), clamp(ay, 0, AREA_SIZE), theta)
        if not state_is_valid(pose[0], pose[1], pose[2], self.obstacles):
            self.status = "Manual start collides; pick another spot."
        else:
            self.manual_start = pose
            self.status = (
                f"Manual start: ({pose[0]:.1f}, {pose[1]:.1f}) cm, "
                f"{math.degrees(pose[2]):.1f}°."
            )
            self._invalidate_path()
        self.dragging_start = False
        self.drag_anchor_world = None
        self.drag_current_world = None

    def _add_obstacle(self, wx: float, wy: float) -> None:
        wx = round(wx)
        wy = round(wy)
        wx = clamp(wx, OBSTACLE_HALF, AREA_SIZE - OBSTACLE_HALF)
        wy = clamp(wy, OBSTACLE_HALF, AREA_SIZE - OBSTACLE_HALF)
        self.obstacles.append((wx, wy))
        self._invalidate_path()
        self.status = f"Obstacle added at ({wx:.0f}, {wy:.0f}) cm."

    def _erase_obstacle(self, wx: float, wy: float) -> None:
        idx = obstacle_at(wx, wy, self.obstacles)
        if idx >= 0:
            self.obstacles.pop(idx)
            self._invalidate_path()
            self.status = "Obstacle removed."
        else:
            self.status = "Click directly on an obstacle to erase it."

    # ---------- panel actions -------------------------------------------
    def _click_panel(self, mx: int, my: int) -> None:
        for b in self.buttons:
            if b.rect.collidepoint(mx, my):
                self._handle_button(b.key)
                return

    def _handle_button(self, key: str) -> None:
        if key.startswith("mode:"):
            self._set_mode(key.split(":", 1)[1])
        elif key == "step:inc":
            self._adjust_step(+STEP_INCR)
        elif key == "step:dec":
            self._adjust_step(-STEP_INCR)
        elif key == "plan":
            self._do_plan()
        elif key == "save_json":
            self._do_save_json()
        elif key == "refresh_pose":
            self._refresh_pose()
        elif key == "clear_path":
            self._clear_path("Path cleared.")
        elif key == "clear_all":
            self._clear_all()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.dragging_start = False
        if mode != "goal":
            self.hover_slot = None
        msgs = {
            "goal": "Click a parking slot to choose the goal.",
            "obstacle": "Click anywhere on the canvas to drop a 5×5 cm obstacle.",
            "erase": "Click an obstacle to remove it.",
            "start": "Manual fallback: click & drag on the canvas to place start.",
        }
        self.status = msgs.get(mode, "")

    def _select_slot(self, slot: Tuple[int, float, float, float]) -> None:
        sid, cx, cy, yaw = slot
        if not state_is_valid(cx, cy, yaw, self.obstacles):
            self.status = f"Slot {sid} is currently blocked by an obstacle."
            return
        self.selected_slot = slot
        self._invalidate_path()
        self.status = (
            f"Goal = slot {sid}  ({cx:.1f}, {cy:.1f}) cm, "
            f"{math.degrees(yaw):.0f}°. Press Plan."
        )

    def _adjust_step(self, delta: float) -> None:
        new = round((self.step_size + delta) * 2) / 2
        self.step_size = clamp(new, STEP_MIN, STEP_MAX)
        self.status = f"Step size = {self.step_size:.1f} cm."

    def _refresh_pose(self) -> None:
        if self.pose_source.tracker is None:
            # Try to (re-)open the tracker.
            self.pose_source.close()
            self.pose_source = _PoseSource()
            if self.pose_source.tracker is None:
                self.status = f"AprilTag tracker {self.pose_source.status}"
                return
        # Force an immediate poll on the next frame.
        self._next_pose_poll_t = 0.0
        self.status = "Polling AprilTag pose..."

    def _clear_path(self, msg: str) -> None:
        self.path_segments = []
        self.path_poses = []
        self.last_result = None
        self.status = msg

    def _invalidate_path(self) -> None:
        if self.path_segments or self.path_poses:
            self._clear_path("Path invalidated by edits.")

    def _clear_all(self) -> None:
        self.selected_slot = None
        self.obstacles = []
        self.manual_start = None
        self._clear_path("Cleared all. Pick a goal slot to begin.")

    # ---------- plan / save ---------------------------------------------
    def _do_plan(self) -> None:
        start = self._current_start_pose()
        if start is None:
            self.status = "No start pose yet (waiting on AprilTag, or set manual start)."
            return
        if self.selected_slot is None:
            self.status = "Pick a goal slot first."
            return
        if not state_is_valid(start[0], start[1], start[2], self.obstacles):
            self.status = "Start pose collides with an obstacle or boundary."
            return

        approach = approach_pose_for_slot(self.selected_slot)
        if not state_is_valid(approach[0], approach[1], approach[2], self.obstacles):
            self.status = "Approach point (40 cm before slot) is in collision."
            return

        self.is_planning = True
        self.status = "Planning..."
        self._render()
        pygame.display.flip()

        result = plan(start, approach, self.obstacles, self.step_size)
        self.last_result = result
        self.is_planning = False

        if not result.success:
            self.path_segments = []
            self.path_poses = []
            self.status = (
                f"Planner failed: {result.message}  "
                f"iter={result.iterations}, time={result.elapsed*1000:.0f} ms"
            )
            return

        # Draw the planner's curved path + a synthetic straight tail so the
        # operator sees the full route, including the final 40 cm to slot.
        self.path_segments = list(result.segments)
        self.path_poses = list(result.poses)
        gx, gy, gyaw = self.selected_slot[1], self.selected_slot[2], self.selected_slot[3]
        ax, ay = approach[0], approach[1]
        n_straight = max(2, int(math.ceil(APPROACH_OFFSET_CM / STRAIGHT_DENSITY_CM)))
        tail_poses: List[Tuple[float, float, float]] = []
        for k in range(1, n_straight + 1):
            u = k / n_straight
            tail_poses.append((ax + u * (gx - ax), ay + u * (gy - ay), gyaw))
        self.path_poses.extend(tail_poses)
        self.path_segments.append(
            PathSegment(poses=[approach] + tail_poses, direction=1)
        )

        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(self.path_poses, self.path_poses[1:])
        )
        relaxed_tag = "  [relaxed]" if getattr(result, "relaxed", False) else ""
        self.status = (
            f"Planned ({len(self.path_poses)} poses, ≈{length:.1f} cm){relaxed_tag}. "
            f"Press Save JSON to write {self.out_path}."
        )

    def _do_save_json(self) -> None:
        if self.last_result is None or not self.last_result.success:
            self.status = "Plan first — there's nothing to save."
            return
        if self.selected_slot is None:
            self.status = "Cannot save: no goal slot selected."
            return
        start = self._current_start_pose()
        if start is None:
            self.status = "Cannot save: no start pose."
            return
        approach = approach_pose_for_slot(self.selected_slot)
        gx, gy, gyaw = self.selected_slot[1], self.selected_slot[2], self.selected_slot[3]
        points = build_path_points(self.last_result, approach, (gx, gy, gyaw))
        try:
            write_path_json(
                self.out_path,
                points,
                slot=self.selected_slot,
                start_pose_centre=start,
                obstacles=self.obstacles,
                step_size=self.step_size,
                relaxed=bool(self.last_result.relaxed),
            )
        except OSError as e:
            self.status = f"Save failed: {e}"
            return
        self.status = f"Saved {len(points)} points to {self.out_path}."

    # ---------- rendering ------------------------------------------------
    def _render(self) -> None:
        self.screen.fill(C_BG)
        self._render_canvas()
        self.screen.blit(self.canvas, (0, 0))
        self._render_panel()

    def _render_canvas(self) -> None:
        self.canvas.fill((255, 255, 255))

        # Grid
        for cm in range(0, int(AREA_SIZE) + 1, 10):
            color = C_AXIS if cm % 50 == 0 else C_GRID
            wd = 2 if cm % 50 == 0 else 1
            sx = int(cm * SCALE)
            pygame.draw.line(self.canvas, color, (sx, 0), (sx, CANVAS_PX), wd)
            sy = int(CANVAS_PX - cm * SCALE)
            pygame.draw.line(self.canvas, color, (0, sy), (CANVAS_PX, sy), wd)
        pygame.draw.rect(
            self.canvas, C_BORDER, pygame.Rect(0, 0, CANVAS_PX, CANVAS_PX), 2
        )

        # Parking slots
        for slot in PARKING_SLOTS:
            sid, cx, cy, yaw = slot
            corners = slot_corners(cx, cy, yaw)
            is_hover = self.hover_slot is not None and self.hover_slot[0] == sid
            is_sel = self.selected_slot is not None and self.selected_slot[0] == sid
            fill = C_SLOT_HOVER if (is_hover or is_sel) else C_SLOT_FILL
            outline = (
                C_SLOT_HOVER_OUTLINE if (is_hover or is_sel) else C_SLOT_OUTLINE
            )
            draw_polygon_world(self.canvas, fill, corners)
            draw_polygon_world(self.canvas, outline, corners, width=2)

            ahead_len = SLOT_LENGTH * 0.32
            cosy, siny = math.cos(yaw), math.sin(yaw)
            tip = (cx + ahead_len * cosy, cy + ahead_len * siny)
            tail = (cx - ahead_len * 0.35 * cosy, cy - ahead_len * 0.35 * siny)
            pygame.draw.line(
                self.canvas, outline,
                world_to_screen(*tail), world_to_screen(*tip), 2,
            )
            head_w = 3.0
            left = (
                tip[0] - 4.0 * cosy + head_w * siny,
                tip[1] - 4.0 * siny - head_w * cosy,
            )
            right = (
                tip[0] - 4.0 * cosy - head_w * siny,
                tip[1] - 4.0 * siny + head_w * cosy,
            )
            pygame.draw.polygon(
                self.canvas, outline,
                [world_to_screen(*tip), world_to_screen(*left), world_to_screen(*right)],
            )
            label = self.font_b.render(str(sid), True, C_SLOT_TEXT)
            label_pos_world = (
                cx - ahead_len * 0.55 * cosy,
                cy - ahead_len * 0.55 * siny,
            )
            self.canvas.blit(label, label.get_rect(center=world_to_screen(*label_pos_world)))

        # Obstacles
        for ox, oy in self.obstacles:
            r = pygame.Rect(0, 0, int(OBSTACLE_SIZE * SCALE), int(OBSTACLE_SIZE * SCALE))
            tlx, tly = world_to_screen(ox - OBSTACLE_HALF, oy + OBSTACLE_HALF)
            r.topleft = (tlx, tly)
            pygame.draw.rect(self.canvas, C_OBSTACLE, r)
            pygame.draw.rect(self.canvas, C_OBSTACLE_BORDER, r, 1)

        # Path (curved part + appended straight tail)
        for seg in self.path_segments:
            color = C_PATH_FWD if seg.direction >= 0 else C_PATH_REV
            pts = [world_to_screen(p[0], p[1]) for p in seg.poses]
            if len(pts) >= 2:
                pygame.draw.lines(self.canvas, color, False, pts, 4)

        # Manual-start drag preview
        if (
            self.dragging_start
            and self.drag_anchor_world is not None
            and self.drag_current_world is not None
        ):
            ax, ay = self.drag_anchor_world
            cx, cy = self.drag_current_world
            theta = (
                math.atan2(cy - ay, cx - ax)
                if math.hypot(cx - ax, cy - ay) > 0.5
                else 0.0
            )
            valid = state_is_valid(ax, ay, theta, self.obstacles)
            preview = (200, 220, 255) if valid else (255, 220, 220)
            border = C_INVALID if not valid else C_START
            draw_car(self.canvas, ax, ay, theta, preview, border, width=2)
            pygame.draw.line(
                self.canvas, C_DRAG,
                world_to_screen(ax, ay), world_to_screen(cx, cy), 2,
            )

        # Live start car (centre-frame) at the AprilTag-detected (or manual) pose
        start = self._current_start_pose()
        if start is not None:
            x, y, t = start
            has_path = bool(self.path_segments)
            if has_path:
                draw_car_alpha(
                    self.canvas, x, y, t,
                    (210, 225, 250), C_START, C_START,
                    ALPHA_POSE_BODY_OVER_PATH,
                    ALPHA_POSE_LINES_OVER_PATH,
                    outline_width=2,
                )
            else:
                draw_car(self.canvas, x, y, t, (210, 225, 250), C_START,
                         front_color=C_START, width=2)

        # Goal car at slot centre
        if self.selected_slot is not None:
            _, gx, gy, gyaw = self.selected_slot
            has_path = bool(self.path_segments)
            if has_path:
                draw_car_alpha(
                    self.canvas, gx, gy, gyaw,
                    (250, 215, 220), C_GOAL, C_GOAL,
                    ALPHA_POSE_BODY_OVER_PATH,
                    ALPHA_POSE_LINES_OVER_PATH,
                    outline_width=2,
                )
            else:
                draw_car(self.canvas, gx, gy, gyaw, (250, 215, 220), C_GOAL,
                         front_color=C_GOAL, width=2)

            # Approach-point marker (faded car silhouette, no border highlight).
            ax, ay, ath = approach_pose_for_slot(self.selected_slot)
            draw_car_alpha(
                self.canvas, ax, ay, ath,
                C_ANIM_CAR, C_ANIM_CAR_BORDER, C_ANIM_CAR_BORDER,
                ALPHA_SLIDE_BODY, ALPHA_SLIDE_LINES, outline_width=2,
            )

    def _render_panel(self) -> None:
        panel_rect = pygame.Rect(CANVAS_PX, 0, SIDEBAR_PX, WINDOW_H)
        pygame.draw.rect(self.screen, C_PANEL, panel_rect)
        pygame.draw.line(
            self.screen, C_BORDER, (CANVAS_PX, 0), (CANVAS_PX, WINDOW_H), 1,
        )
        title = self.font_t.render("Path planner", True, C_TEXT)
        self.screen.blit(title, (CANVAS_PX + 18, 18))
        sub = self.font.render("AprilTag start → JSON", True, C_TEXT_DIM)
        self.screen.blit(sub, (CANVAS_PX + 18, 38))

        # Mode-button highlight
        active_mode_key = f"mode:{self.mode}"
        for b in self.buttons:
            accent_keys = {"plan", "save_json"}
            active = b.key == active_mode_key
            accent = b.key in accent_keys and not active
            b.draw(self.screen, self.font_b, active=active, accent=accent)

        # Step size header
        step_buttons = [b for b in self.buttons if b.key.startswith("step:")]
        if step_buttons:
            top = step_buttons[0].rect.top
            label = self.font_b.render("Step size", True, C_TEXT)
            self.screen.blit(label, (CANVAS_PX + 18, top - 22))
            value_txt = self.font_b.render(f"{self.step_size:.1f} cm", True, C_TEXT)
            self.screen.blit(
                value_txt, value_txt.get_rect(center=self._step_label_rect.center)
            )

        # Live pose / goal info, anchored under the last button.
        last_button_bottom = max(b.rect.bottom for b in self.buttons)
        info_y = last_button_bottom + 24

        start = self._current_start_pose()
        if start is not None:
            sx, sy, sth = start
            start_lines = [
                ("Start (cm)", f"({sx:.1f}, {sy:.1f})"),
                ("Start yaw", f"{math.degrees(sth):.1f}°"),
            ]
        else:
            start_lines = [("Start", "—")]

        # Status of the AprilTag pose source.
        tracker_label = "AprilTag"
        if self.pose_source.tracker is None:
            tracker_value = self.pose_source.status
        else:
            age = self._pose_age_s()
            if age is None:
                tracker_value = "waiting…"
            elif age > POSE_STALE_S:
                tracker_value = f"stale ({age:.1f}s)"
            else:
                tracker_value = f"ok ({age:.2f}s)"
            raw = self.pose_source.last_raw
            if raw is not None:
                tracker_value += f"  cam={raw.get('camera','?')} tag={raw.get('tag_id','?')}"

        if self.selected_slot is not None:
            sid, gx, gy, gyaw = self.selected_slot
            ax, ay, _ = approach_pose_for_slot(self.selected_slot)
            goal_lines = [
                ("Goal slot", f"#{sid}"),
                ("Goal (cm)", f"({gx:.1f}, {gy:.1f})"),
                ("Goal yaw", f"{math.degrees(gyaw):.0f}°"),
                ("Approach (cm)", f"({ax:.1f}, {ay:.1f})"),
            ]
        else:
            goal_lines = [("Goal slot", "—")]

        info_lines: List[Tuple[str, str]] = []
        info_lines.append((tracker_label, tracker_value))
        info_lines.extend(start_lines)
        info_lines.append(("", ""))
        info_lines.extend(goal_lines)
        info_lines.append(("", ""))
        info_lines.append(("Output JSON", self.out_path))
        info_lines.append(("Obstacles", f"{len(self.obstacles)}"))
        if self.last_result is not None and self.last_result.success:
            info_lines.append(("Path poses", f"{len(self.path_poses)}"))
        info_lines.append(("Approach offset", f"{APPROACH_OFFSET_CM:.0f} cm"))
        info_lines.append(("Wheelbase", f"{WHEELBASE:.0f} cm"))

        y = info_y
        for k, v in info_lines:
            if not k and not v:
                y += 6
                continue
            color = C_TEXT if k.strip() and not k.startswith("  ") else C_TEXT_DIM
            label = self.font.render(k, True, color)
            self.screen.blit(label, (CANVAS_PX + 18, y))
            if v:
                vsurf = self.font.render(v, True, C_TEXT)
                self.screen.blit(
                    vsurf,
                    (CANVAS_PX + SIDEBAR_PX - 18 - vsurf.get_width(), y),
                )
            y += 17

        # Status bar
        status_h = 26
        status_rect = pygame.Rect(0, WINDOW_H - status_h, CANVAS_PX, status_h)
        pygame.draw.rect(self.screen, (35, 40, 55), status_rect)
        prefix = "[planning] " if self.is_planning else ""
        msg = self.font_b.render(prefix + self.status, True, (240, 245, 255))
        self.screen.blit(msg, (8, WINDOW_H - status_h + 6))


# ----- headless CLI mode --------------------------------------------------

def acquire_start_pose_centre(timeout_s: float = 10.0) -> Tuple[float, float, float]:
    """CLI helper: open the tracker, wait for one fix, return centre-frame pose."""
    from apriltag_pose import AprilTagMapPoseTracker
    tracker = AprilTagMapPoseTracker()
    deadline = time.perf_counter() + float(timeout_s)
    try:
        while time.perf_counter() < deadline:
            p = tracker.update()
            if p is None:
                time.sleep(0.05)
                continue
            yaw = yaw_deg_to_planner_rad(float(p["yaw_deg"]))
            xc, yc = front_axle_to_centre(float(p["x_cm"]), float(p["y_cm"]), yaw)
            return (xc, yc, yaw)
    finally:
        tracker.close()
    raise RuntimeError(f"AprilTag pose not detected within {timeout_s:.1f}s.")


def parse_obstacles(spec: str) -> List[Tuple[float, float]]:
    if not spec:
        return []
    raw = json.loads(spec)
    return [(float(o[0]), float(o[1])) for o in raw]


def plan_to_slot_cli(slot_id: int,
                     obstacles: Iterable[Tuple[float, float]] = (),
                     step_size: float = PLANNER_STEP_SIZE,
                     out_path: str = "planned_path.json") -> dict:
    slot = get_slot(slot_id)
    sid, gx, gy, gyaw = slot
    approach = approach_pose_for_slot(slot)
    print("Acquiring AprilTag pose...")
    start = acquire_start_pose_centre()
    print(
        f"Start (centre): x={start[0]:.2f}, y={start[1]:.2f}, "
        f"yaw={math.degrees(start[2]):.1f}°"
    )
    obs = list(obstacles)
    print("Planning path to approach point...")
    result = plan(start, approach, obs, step_size)
    if not result.success:
        raise RuntimeError(f"Planner failed: {result.message}")
    print(f"Planner OK: {len(result.poses)} poses, "
          f"iter={result.iterations}, time={result.elapsed*1000:.0f} ms"
          + (" [relaxed]" if result.relaxed else ""))
    points = build_path_points(result, approach, (gx, gy, gyaw))
    write_path_json(
        out_path, points,
        slot=slot, start_pose_centre=start, obstacles=obs,
        step_size=step_size, relaxed=bool(result.relaxed),
    )
    print(f"Wrote {len(points)} points to {out_path}.")
    return {"out_path": out_path, "n_points": len(points)}


# ----- entry point --------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Path planner UI / CLI.")
    parser.add_argument("--out", type=str, default="planned_path.json", help="Output JSON path.")
    parser.add_argument("--cli", action="store_true", help="Headless mode (no UI).")
    parser.add_argument("--slot", type=int, default=None, help="(CLI) Parking slot ID.")
    parser.add_argument("--step", type=float, default=PLANNER_STEP_SIZE, help="Planner step size (cm).")
    parser.add_argument(
        "--obstacles",
        type=str,
        default="",
        help='JSON list of [x_cm, y_cm] obstacle centres.',
    )
    args = parser.parse_args()

    if args.cli:
        if args.slot is None:
            parser.error("--cli requires --slot")
        plan_to_slot_cli(
            slot_id=args.slot,
            obstacles=parse_obstacles(args.obstacles),
            step_size=args.step,
            out_path=args.out,
        )
        return

    App(out_path=args.out).run()


if __name__ == "__main__":
    main()
