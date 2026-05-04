import json
import math
import time

import serial

from apriltag_pose import AprilTagMapPoseTracker


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def dist_cm(p1, p2) -> float:
    dx = float(p1["x_cm"]) - float(p2["x_cm"])
    dy = float(p1["y_cm"]) - float(p2["y_cm"])
    return math.hypot(dx, dy)

def angle_diff_rad(a: float, b: float) -> float:
    return wrap_pi(a - b)

def rate_limit(value: float, prev: float, max_delta: float) -> float:
    """
    Limit how much 'value' can change from 'prev' by max_delta.
    """
    dv = value - prev
    if dv > max_delta:
        return prev + max_delta
    if dv < -max_delta:
        return prev - max_delta
    return value


def load_path(path_json_path: str):
    with open(path_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data["points"]
    return data, pts


def is_past_path_end(pose, points, past_long_cm: float = 3.0) -> bool:
    """
    True if the car has passed beyond the final path point along the commanded
    travel direction (handles overshoot where Euclidean distance grows again).

    Uses the last waypoint's yaw and direction (+1 forward / -1 reverse) to
    define travel direction in map frame.
    """
    if len(points) < 1:
        return False
    last = points[-1]
    gx = float(last["x_cm"])
    gy = float(last["y_cm"])
    yaw = float(last["yaw_rad"])
    direction = int(last.get("direction", 1))
    cx = float(pose["x_cm"])
    cy = float(pose["y_cm"])
    mx = math.cos(yaw)
    my = math.sin(yaw)
    # Body-forward in map frame; multiply by gear to get instantaneous travel direction.
    tx = float(direction) * mx
    ty = float(direction) * my
    longitudinal = (cx - gx) * tx + (cy - gy) * ty
    return longitudinal > float(past_long_cm)


def nearest_index(points, x_cm, y_cm, start_idx=0, window=400):
    """
    Find nearest point index in a sliding window for speed.
    """
    n = len(points)
    if n == 0:
        return 0
    i0 = max(0, int(start_idx))
    i1 = min(n, i0 + int(window))
    best_i = i0
    best_d2 = float("inf")
    for i in range(i0, i1):
        dx = points[i]["x_cm"] - x_cm
        dy = points[i]["y_cm"] - y_cm
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


# Planner wheelbase (cm): rear axle to front axle. Used as rear-offset from localized “front” in reverse CTE / index tracking.
STANLEY_WB_CM = 16


def path_tracking_point_cm(pose: dict, points: list, idx_near: int) -> tuple[float, float]:
    """
    Map position used to find nearest path index: same as localized pose in forward;
    in reverse, rear axle (matches reverse CTE reference).
    Uses direction on the path point at idx_near (clamped).
    """
    n = len(points)
    if n == 0:
        return float(pose["x_cm"]), float(pose["y_cm"])
    i = clamp(int(idx_near), 0, n - 1)
    seg_dir = int(points[i].get("direction", 1))
    xf = float(pose["x_cm"])
    yf = float(pose["y_cm"])
    psi = float(pose["yaw_rad"])
    if seg_dir < 0:
        return (
            xf - STANLEY_WB_CM * math.sin(psi),
            yf - STANLEY_WB_CM * math.cos(psi),
        )
    return xf, yf


def stanley_control(points, pose, idx_near, k_fwd=1.2, k_rev=1.2, softening_fwd=30.0, softening_rev=35.0):
    """
    Front-axle Stanley with planner feedforward; reverse uses rear-axle position for CTE only.
    """
    n = len(points)
    idx = clamp(idx_near, 0, n - 1)
    p = points[int(idx)]

    tx = float(p["x_cm"])
    ty = float(p["y_cm"])
    tyaw = float(p["yaw_rad"])
    direction = int(p.get("direction", 1))
    
    # Perfect kinematic steering angle from the Hybrid A* planner
    steer_ff = float(p.get("steer_rad", 0.0))

    x_front = float(pose["x_cm"])
    y_front = float(pose["y_cm"])
    yaw = float(pose["yaw_rad"])

    # 1. True Heading Error (unchanged: same body yaw as localization)
    heading_err = wrap_pi(tyaw - yaw)

    # 2. Cross-track reference: forward = localized point; reverse = rear axle from user formula
    if direction < 0:
        x_track = x_front - STANLEY_WB_CM * math.cos(yaw)
        y_track = y_front - STANLEY_WB_CM * math.sin(yaw)
    else:
        x_track = x_front
        y_track = y_front

    dx = x_track - tx
    dy = y_track - ty
    left_nx = -math.sin(tyaw)
    left_ny = math.cos(tyaw)
    cte = dx * left_nx + dy * left_ny  # +: vehicle is physically to the left of the target line

    # 3. Apply Gear-Specific Parameters & Reverse Inversion
    if direction >= 0:
        k = k_fwd
        softening = softening_fwd
        cte_direction = 1.0  # Normal steering correction
    else:
        k = k_rev
        softening = softening_rev
        # In reverse, to move the rear of the car right, we must steer the front wheels left. 
        # Inverting the CTE achieves this Kinematic flip cleanly.
        cte_direction = -1.0

    # 4. Control Law
    v = 20.0  # Nominal speed
    cte_term = math.atan2(k * cte * cte_direction, (v + softening))

    if direction == -1:
        steer = -steer_ff + heading_err - cte_term
    else:
        steer = -(-steer_ff + heading_err - cte_term)
    
    steer = wrap_pi(steer)
    
    return steer, int(idx), direction

def maybe_init_viz():
    """
    Lazy-import matplotlib so headless runs still work.
    Returns (plt, fig, ax, artists) or None if import fails.
    """
    try:
        import matplotlib

        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Stanley follower debug view")

    (path_line,) = ax.plot([], [], "k-", lw=2, alpha=0.5, label="planned path")
    (car_pt,) = ax.plot([], [], "go", ms=8, label="car")
    (car_head,) = ax.plot([], [], "g-", lw=2)
    (target_pt,) = ax.plot([], [], "rx", ms=10, label="target")
    (near_pt,) = ax.plot([], [], "bo", ms=6, label="nearest")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="upper right")

    return plt, fig, ax, {"path_line": path_line, "car_pt": car_pt, "car_head": car_head, "target_pt": target_pt, "near_pt": near_pt, "txt": txt}


def main():
    # --- Config ---
    PATH_FILE = "planned_path.json"
    SERIAL_PORT = "/dev/ttyUSB0"  # adjust if needed
    BAUD = 115200

    LOOP_HZ = 15.0
    POSE_TIMEOUT_S = 0.25
    CMD_TIMEOUT_S = 0.5

    # steering mapping (servo): straight ≈ 80°, mechanical limits 55°..105° (±25°)
    SERVO_CENTER_DEG = 80.0
    SERVO_MIN_DEG = 55.0
    SERVO_MAX_DEG = 105.0
    SERVO_MAX_DELTA_DEG = (SERVO_MAX_DEG - SERVO_MIN_DEG) / 2.0  # 25; scale full steer to limit range
    MAX_STEER_RAD_FOR_SERVO = math.radians(25.0)  # planner max; maps to ±SERVO_MAX_DELTA_DEG at servo
    SERVO_MAX_SPEED_DEG_S = 60.0  # limit how fast servo command changes

    # throttle mapping
    THROTTLE_FWD = 160  # 0..255
    THROTTLE_REV = 140  # 0..255 (often safer a bit lower)
    # End of route: stop if this far *past* the last point along travel direction (overshoot).
    # Only applies when tracked index is in the last N path points (avoids false positives mid-route).
    PAST_GOAL_LONG_CM = 3.0
    PAST_GOAL_CHECK_LAST_POINTS = 10
    # Stanley-style end: near last point in space and progressed along path
    END_DIST_CM = 3.0
    END_MIN_IDX_FROM_TAIL = 30
    GEAR_CHANGE_SERVO_SETTLE_S = 2.0  # pause motor on gear change; steer first
    TURN_PWM_BOOST_MAX = 15  # extra PWM at full steering
    THROTTLE_MIN_MOVING = 110  # don't go below this unless stopping

    # Localization jump filter
    LOCAL_JUMP_DIST_CM = 7.0
    LOCAL_PENDING_MATCH_CYCLES = 5
    LOCAL_PENDING_POS_TOL_CM = 6.0
    LOCAL_PENDING_YAW_TOL_DEG = 12.0

    # --- Load path ---
    meta, points = load_path(PATH_FILE)
    if not points:
        raise RuntimeError("Path file has no points.")

    print(f"[OK] Loaded {len(points)} path points from {PATH_FILE}")

    # --- Start tracker + serial ---
    tracker = AprilTagMapPoseTracker()
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.05)
    ser.write(b"STOP\n")

    last_pose_t = 0.0
    last_cmd_t = 0.0
    idx = 0
    last_pose = None
    last_servo_deg = SERVO_CENTER_DEG
    last_direction = None  # +1 forward / -1 reverse
    gear_change_until_t = 0.0
    startup_until_t = time.perf_counter() + GEAR_CHANGE_SERVO_SETTLE_S
    accepted_pose = None
    pending_pose = None
    pending_count = 0
    localization_hold = False

    # Optional debug visualization
    viz = maybe_init_viz()
    if viz is not None:
        plt, fig, ax, artists = viz
        xs = [p["x_cm"] for p in points]
        ys = [p["y_cm"] for p in points]
        artists["path_line"].set_data(xs, ys)
        artists["target_pt"].set_data([points[-1]["x_cm"]], [points[-1]["y_cm"]])
        fig.canvas.draw()
        fig.show()

    dt = 1.0 / LOOP_HZ
    try:
        while True:
            t0 = time.perf_counter()

            # Pose update
            p = tracker.update()
            if p is not None:
                # Convert apriltag yaw to controller yaw convention:
                # apriltag yaw: 0°=+Y, CCW+
                # controller yaw: 0 rad=+X, CCW+
                yaw_rad = wrap_pi(math.radians(float(p["yaw_deg"])) + (math.pi / 2.0))
                pose = {"x_cm": float(p["x_cm"]), "y_cm": float(p["y_cm"]), "yaw_rad": yaw_rad}
                # Jump filter: if pose jumps > 25cm, only accept after 5 consistent cycles.
                # While a jump is pending, we hold the motor stopped until localization is valid again.
                localization_hold = False
                if accepted_pose is None:
                    accepted_pose = pose
                    pending_pose = None
                    pending_count = 0
                else:
                    d = dist_cm(pose, accepted_pose)
                    if d <= LOCAL_JUMP_DIST_CM:
                        accepted_pose = pose
                        pending_pose = None
                        pending_count = 0
                    else:
                        # candidate jump; require several matching cycles
                        localization_hold = True
                        if pending_pose is None:
                            pending_pose = pose
                            pending_count = 1
                        else:
                            d_pend = dist_cm(pose, pending_pose)
                            yaw_ok = abs(math.degrees(angle_diff_rad(pose["yaw_rad"], pending_pose["yaw_rad"]))) <= LOCAL_PENDING_YAW_TOL_DEG
                            if d_pend <= LOCAL_PENDING_POS_TOL_CM and yaw_ok:
                                pending_count += 1
                            else:
                                pending_pose = pose
                                pending_count = 1
                        if pending_count >= LOCAL_PENDING_MATCH_CYCLES:
                            accepted_pose = pending_pose
                            pending_pose = None
                            pending_count = 0
                            localization_hold = False

                last_pose = accepted_pose
                last_pose_t = time.perf_counter()

            now = time.perf_counter()
            if last_pose is None or (now - last_pose_t) > POSE_TIMEOUT_S:
                # Fail-safe stop if pose is stale
                if (now - last_cmd_t) > CMD_TIMEOUT_S:
                    ser.write(b"STOP\n")
                    last_cmd_t = now
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
                continue

            # Progress along path: in reverse, snap index using rear-axle position (same as reverse CTE).
            tx, ty = path_tracking_point_cm(last_pose, points, idx)
            idx = nearest_index(points, tx, ty, start_idx=idx, window=5)

            # Distance-to-goal (used for STOP condition / debug / min throttle)
            gx = float(points[-1]["x_cm"])
            gy = float(points[-1]["y_cm"])
            dist_goal = math.hypot(last_pose["x_cm"] - gx, last_pose["y_cm"] - gy)

            # Overshoot: longitudinal past final point (dot check). Only near end of path by index.
            tail_gate = idx >= max(0, len(points) - PAST_GOAL_CHECK_LAST_POINTS)
            if tail_gate and is_past_path_end(last_pose, points, PAST_GOAL_LONG_CM):
                print(
                    f"[STOP] Past path end (>{PAST_GOAL_LONG_CM:.1f} cm along travel); "
                    f"dist_goal={dist_goal:.1f} cm idx={idx}"
                )
                ser.write(b"STOP\n")
                last_cmd_t = time.perf_counter()
                break

            steer_rad, idx, direction = stanley_control(points, last_pose, idx)
            scale = 1.0  # do not slow down while parking (per user request)

            # Detect gear change (direction flip) and pause motor to let steering settle first.
            if last_direction is None:
                last_direction = int(direction)
            if int(direction) != int(last_direction):
                gear_change_until_t = time.perf_counter() + GEAR_CHANGE_SERVO_SETTLE_S
                last_direction = int(direction)

            throttle = int((THROTTLE_FWD if direction > 0 else THROTTLE_REV) * scale)
            throttle = throttle if direction > 0 else -throttle
            if time.perf_counter() < gear_change_until_t or time.perf_counter() < startup_until_t:
                throttle = 0
            if localization_hold:
                throttle = 0

            # Map steering rad -> servo degrees with constraints
            steer_rad = clamp(steer_rad, -MAX_STEER_RAD_FOR_SERVO, MAX_STEER_RAD_FOR_SERVO)
            # Steering sign/mapping intentionally left as-is (per user request).
            servo_deg = SERVO_CENTER_DEG + (steer_rad / MAX_STEER_RAD_FOR_SERVO) * SERVO_MAX_DELTA_DEG
            servo_deg = clamp(servo_deg, SERVO_MIN_DEG, SERVO_MAX_DEG)
            # Rate-limit servo so it doesn't snap to large angles instantly
            max_delta = SERVO_MAX_SPEED_DEG_S * dt
            servo_deg = rate_limit(servo_deg, last_servo_deg, max_delta=max_delta)
            last_servo_deg = servo_deg

            # Turn-dependent PWM boost (linear): 0 at center, +15 at max steering
            if throttle != 0:
                turn_ratio = abs(servo_deg - SERVO_CENTER_DEG) / SERVO_MAX_DELTA_DEG
                turn_ratio = clamp(turn_ratio, 0.0, 1.0)
                boost = int(round(TURN_PWM_BOOST_MAX * turn_ratio))
                if throttle > 0:
                    throttle = min(255, throttle + boost)
                else:
                    throttle = max(-255, throttle - boost)

            # Enforce minimum moving throttle so we don't stop too early when slowing down.
            # If we are truly stopping (very close to goal), end condition below will handle STOP.
            if throttle != 0 and dist_goal > END_DIST_CM:
                if 0 < throttle < THROTTLE_MIN_MOVING:
                    throttle = THROTTLE_MIN_MOVING
                elif 0 > throttle > -THROTTLE_MIN_MOVING:
                    throttle = -THROTTLE_MIN_MOVING

            # Send command
            line = f"CMD {servo_deg:.1f} {throttle}\n".encode("ascii")
            ser.write(line)
            last_cmd_t = time.perf_counter()

            # Debug visualization update (non-blocking)
            if viz is not None:
                x = last_pose["x_cm"]
                y = last_pose["y_cm"]
                yaw = last_pose["yaw_rad"]
                head_len = 12.0
                artists["car_pt"].set_data([x], [y])
                artists["car_head"].set_data([x, x + head_len * math.cos(yaw)], [y, y + head_len * math.sin(yaw)])
                artists["near_pt"].set_data([points[idx]["x_cm"]], [points[idx]["y_cm"]])
                artists["txt"].set_text(
                    f"idx={idx}/{len(points)} dir={'F' if direction>0 else 'R'}\n"
                    f"steer={math.degrees(steer_rad):+.1f}deg servo={servo_deg:.1f} thr={throttle}\n"
                    f"goal_dist={dist_goal:.1f}cm"
                )
                fig.canvas.draw_idle()
                plt.pause(0.001)

            # End condition: reached neighborhood of goal near end of path index
            if dist_goal < END_DIST_CM and idx > len(points) - END_MIN_IDX_FROM_TAIL:
                print(f"[STOP] Goal reached (dist={dist_goal:.1f} cm, idx={idx})")
                ser.write(b"STOP\n")
                break

            # loop timing
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    finally:
        try:
            ser.write(b"STOP\n")
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        try:
            tracker.close()
        except Exception:
            pass
        if viz is not None:
            try:
                plt.close(fig)
            except Exception:
                pass


if __name__ == "__main__":
    main()
