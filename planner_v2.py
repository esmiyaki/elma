import heapq
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.prepared import prep


def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _M(theta: float) -> float:
    return normalize_angle(theta)


def prepare_obstacles(obstacles: Sequence[Polygon]):
    return [prep(o) for o in obstacles]


def collides(poly: Polygon, prepared_obstacles) -> bool:
    return any(o.intersects(poly) for o in prepared_obstacles)


def get_car_polygon(
    x: float,
    y: float,
    yaw: float,
    *,
    car_l: float,
    car_w: float,
    padding: float = 0.0,
) -> Polygon:
    """
    Vehicle footprint polygon. This matches the intent of oldpathcreation.py:
    - Reference point is near the rear axle (via front/rear overhang split)
    - `padding` is used for a safety margin
    """
    front_overhang = 4.0 + padding
    rear_overhang = (car_l - 4.0) + padding
    w = car_w + (2 * padding)
    outline = np.array(
        [
            [-rear_overhang, -w / 2],
            [front_overhang, -w / 2],
            [front_overhang, w / 2],
            [-rear_overhang, w / 2],
        ]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = outline.dot(rot.T)
    rotated[:, 0] += x
    rotated[:, 1] += y
    return Polygon(rotated)


# --- Reeds–Shepp (minimal) ---
class ReedsSheppPlanner:
    """
    Lightweight RS connector used as analytic expansion in Hybrid A*.

    It’s intentionally small (3-segment variants) to keep dependency surface low.
    """

    def __init__(self, *, step_size: float, min_r: float, wb: float, max_steer: float):
        self.step_size = float(step_size)
        self.min_r = float(min_r)
        self.wb = float(wb)
        self.max_steer = float(max_steer)

    class Path:
        def __init__(self, lengths, types, L):
            self.lengths = lengths
            self.types = types
            self.L = L

    def _R(self, x, y):
        return math.hypot(x, y), math.atan2(y, x)

    def _csc(self, x, y, phi):
        paths = []
        u, t = self._R(x - math.sin(phi), y - 1.0 + math.cos(phi))
        v = _M(phi - t)
        if t >= 0 and u >= 0 and v >= 0:
            paths.append(self.Path([t, u, v], ["L", "S", "L"], t + u + v))
        u1, t1 = self._R(x + math.sin(phi), y - 1.0 - math.cos(phi))
        if u1**2 >= 4.0:
            u = math.sqrt(u1**2 - 4.0)
            theta = math.atan2(2.0, u)
            t = _M(t1 + theta)
            v = _M(t - phi)
            if t >= 0 and u >= 0 and v >= 0:
                paths.append(self.Path([t, u, v], ["L", "S", "R"], t + u + v))
        return paths

    def _ccc(self, x, y, phi):
        paths = []
        u1, t1 = self._R(x - math.sin(phi), y - 1.0 + math.cos(phi))
        if u1 <= 4.0:
            u = -2.0 * math.asin(0.25 * u1)
            t = _M(t1 + 0.5 * u + np.pi)
            v = _M(phi - t + u)
            if t >= 0 and u <= 0:
                paths.append(self.Path([t, u, v], ["L", "R", "L"], abs(t) + abs(u) + abs(v)))
        return paths

    def _generate_paths(self, x, y, phi):
        paths = []
        paths.extend(self._csc(x, y, phi))
        paths.extend(self._ccc(x, y, phi))
        return paths

    def _rollout(self, path, sx, sy, syaw):
        px, py, pyaw, pd, ps = [sx], [sy], [syaw], [1], [0.0]
        step = self.step_size
        steer_map = {"L": self.max_steer, "R": -self.max_steer, "S": 0.0}
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
                nyaw = cyaw + (step_d / self.wb) * math.tan(steer)
                px.append(nx)
                py.append(ny)
                pyaw.append(_M(nyaw))
                pd.append(d)
                ps.append(steer)
        return px, py, pyaw, pd, ps

    def connect(self, sx, sy, syaw, gx, gy, gyaw):
        dx, dy = gx - sx, gy - sy
        d = math.hypot(dx, dy)
        theta = _M(math.atan2(dy, dx) - syaw)
        x_norm = d * math.cos(theta) / self.min_r
        y_norm = d * math.sin(theta) / self.min_r
        phi = _M(gyaw - syaw)

        paths = self._generate_paths(x_norm, y_norm, phi)
        if not paths:
            return None
        best = min(paths, key=lambda p: abs(p.L))
        return self._rollout(best, sx, sy, syaw)


@dataclass(order=True, slots=True)
class Node:
    f: float
    x: float
    y: float
    yaw: float
    g: float
    h: float
    direction: int
    steering: float
    parent: Optional["Node"] = None


def _goal_reached(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    xy_tol: float,
    yaw_tol: float,
) -> bool:
    return (math.hypot(x - gx, y - gy) <= xy_tol) and (abs(normalize_angle(yaw - gyaw)) <= yaw_tol)


def _heuristic(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    gyaw: float,
    min_turn_radius: float,
    h_weight: float,
) -> float:
    # Lower bound-ish: Euclidean + turn cost proxy.
    d = math.hypot(x - gx, y - gy)
    dyaw = abs(normalize_angle(yaw - gyaw))
    return (d + min_turn_radius * dyaw) * h_weight


def hybrid_a_star_v2(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    obstacles: Sequence[Polygon],
    *,
    map_size: float,
    # Kinematics (keep identical to caller)
    wb: float,
    max_steer: float,
    min_turn_radius: float,
    car_l: float,
    car_w: float,
    safe_margin: float,
    # Goal tolerance
    final_xy_tolerance: float,
    final_yaw_tolerance: float,
    # Costs (keep your “properties” but tune behavior)
    cost_gear_switch: float,
    cost_steer: float,
    cost_steer_change: float,
    h_weight: float,
    # Planner params
    grid_res: float = 2.0,
    yaw_res: float = math.radians(5.0),
    step_length: float = 5.0,
    n_substeps: int = 5,
    max_iter: int = 120_000,
    analytic_shot_dist: float = 200.0,
    rs_step_size: float = 1.0,
) -> tuple[list[tuple[float, float, float, int, float]], bool]:
    """
    More robust Hybrid A*:
    - richer steering set
    - longer primitives (step_length) with collision sampled along the arc
    - closed key includes direction to avoid pruning valid reverse maneuvers
    - frequent analytic expansion to goal using Reeds–Shepp connector
    """
    sx, sy, syaw = start
    gx, gy, gyaw = goal

    prepared = prepare_obstacles(obstacles)

    # More steering samples than the original code, still within the same MAX_STEER.
    steer_set = np.array(
        [
            -1.0,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
        dtype=float,
    ) * float(max_steer)
    dir_set = (1, -1)

    rs = ReedsSheppPlanner(step_size=rs_step_size, min_r=min_turn_radius, wb=wb, max_steer=max_steer)

    h0 = _heuristic(sx, sy, syaw, gx, gy, gyaw, min_turn_radius, h_weight)
    start_node = Node(f=h0, x=sx, y=sy, yaw=syaw, g=0.0, h=h0, direction=1, steering=0.0, parent=None)

    open_list: list[Node] = [start_node]
    closed: dict[tuple[int, int, int, int], float] = {}

    def state_key(x: float, y: float, yaw: float, direction: int) -> tuple[int, int, int, int]:
        return (
            int(x / grid_res),
            int(y / grid_res),
            int(normalize_angle(yaw) / yaw_res),
            int(direction),
        )

    def in_bounds(x: float, y: float) -> bool:
        return (0.0 < x < map_size) and (0.0 < y < map_size)

    def footprint_ok(x: float, y: float, yaw: float) -> bool:
        if not in_bounds(x, y):
            return False
        poly = get_car_polygon(x, y, yaw, car_l=car_l, car_w=car_w, padding=safe_margin)
        return not collides(poly, prepared)

    it = 0
    while open_list and it < max_iter:
        it += 1
        curr = heapq.heappop(open_list)

        k = state_key(curr.x, curr.y, curr.yaw, curr.direction)
        if k in closed and closed[k] <= curr.g:
            continue
        closed[k] = curr.g

        # Goal region check
        if _goal_reached(curr.x, curr.y, curr.yaw, gx, gy, gyaw, final_xy_tolerance, final_yaw_tolerance):
            path = []
            n: Optional[Node] = curr
            while n is not None:
                path.append((n.x, n.y, n.yaw, n.direction, n.steering))
                n = n.parent
            return path[::-1], True

        # Analytic expansion (RS) when close enough
        if math.hypot(curr.x - gx, curr.y - gy) < analytic_shot_dist:
            # Try several goal yaws inside tolerance to make it less brittle
            yaw_candidates = [
                gyaw,
                normalize_angle(gyaw + 0.5 * final_yaw_tolerance),
                normalize_angle(gyaw - 0.5 * final_yaw_tolerance),
                normalize_angle(gyaw + final_yaw_tolerance),
                normalize_angle(gyaw - final_yaw_tolerance),
            ]
            best = None
            best_cost = float("inf")

            for tyaw in yaw_candidates:
                rs_path = rs.connect(curr.x, curr.y, curr.yaw, gx, gy, tyaw)
                if not rs_path:
                    continue
                rx, ry, ryaw, rd, rs_steer = rs_path
                # RS connector in this lightweight implementation is not guaranteed
                # to exactly terminate at the goal pose; only accept if it lands
                # within the same tolerances as the lattice search.
                if not _goal_reached(
                    rx[-1],
                    ry[-1],
                    ryaw[-1],
                    gx,
                    gy,
                    tyaw,
                    final_xy_tolerance,
                    final_yaw_tolerance,
                ):
                    continue

                ok = True
                for i in range(len(rx)):
                    if not footprint_ok(rx[i], ry[i], ryaw[i]):
                        ok = False
                        break
                if not ok:
                    continue

                # Cost of RS connection added onto current g
                conn = 0.0
                switches = 0
                for i in range(1, len(rx)):
                    conn += math.hypot(rx[i] - rx[i - 1], ry[i] - ry[i - 1])
                    if rd[i] != rd[i - 1]:
                        conn += cost_gear_switch
                        switches += 1
                conn += abs(normalize_angle(tyaw - gyaw)) * 20.0

                if conn < best_cost:
                    best_cost = conn
                    best = (rx, ry, ryaw, rd, rs_steer)

            if best is not None:
                rx, ry, ryaw, rd, rs_steer = best
                # Reconstruct A* prefix
                prefix = []
                n: Optional[Node] = curr
                while n is not None:
                    prefix.append((n.x, n.y, n.yaw, n.direction, n.steering))
                    n = n.parent
                prefix.reverse()
                # Append RS segment
                for i in range(len(rx)):
                    prefix.append((rx[i], ry[i], ryaw[i], int(rd[i]), float(rs_steer[i])))
                return prefix, True

        # Expand lattice actions
        for direction in dir_set:
            for steer in steer_set:
                # integrate for fixed arc length
                v = direction * (step_length / n_substeps)
                x, y, yaw = curr.x, curr.y, curr.yaw

                ok = True
                for _ in range(n_substeps):
                    x += v * math.cos(yaw)
                    y += v * math.sin(yaw)
                    yaw += (v / wb) * math.tan(steer)
                    yaw = normalize_angle(yaw)
                    if not footprint_ok(x, y, yaw):
                        ok = False
                        break
                if not ok:
                    continue

                # Costs: distance, steer magnitude, gear switch, steer rate
                g = curr.g
                g += step_length
                if steer != 0.0:
                    g += abs(steer) * cost_steer
                if direction != curr.direction:
                    g += cost_gear_switch
                g += abs(steer - curr.steering) * cost_steer_change

                h = _heuristic(x, y, yaw, gx, gy, gyaw, min_turn_radius, h_weight)
                heapq.heappush(
                    open_list,
                    Node(
                        f=g + h,
                        x=x,
                        y=y,
                        yaw=yaw,
                        g=g,
                        h=h,
                        direction=int(direction),
                        steering=float(steer),
                        parent=curr,
                    ),
                )

    return [], False

