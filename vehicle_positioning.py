#!/usr/bin/env python3
"""
AprilTag-based 2D vehicle positioning for Raspberry Pi 5 with two
Pi Camera Module 3 units.

Map convention
--------------
    Origin (0, 0) is at the bottom-left of a 200 x 200 cm play area.
    +X axis points East (right), +Y axis points North (up).
    Yaw = 0 deg means the vehicle faces true North; yaw increases CLOCKWISE,
    so 90 deg = East, 180 deg = South, 270 deg = West.

Vehicle
-------
    Two cameras are mounted on the centerline:
        front camera : 12 cm IN FRONT of the vehicle center, lens forward
        rear  camera : 10 cm BEHIND the vehicle center, lens rearward

    The rear camera's heading is rotated by 180 deg in software so that
    both cameras report the vehicle's forward-facing pose.

Anchor tags (tag36h11, default 14.65 cm)
----------------------------------------
    ID 0 : (100, 200)   north wall, faces South
    ID 1 : (200, 125)   east  wall, faces West
    ID 2 : (200,  75)   east  wall, faces West
    ID 3 : (100,   0)   south wall, faces North
    ID 4 : (  0,  75)   west  wall, faces East
    ID 5 : (  0, 125)   west  wall, faces East

Configuration
-------------
    All parameters live in `vehicle_config.json` (auto-generated on first
    run). Each camera's intrinsics are loaded from a separate calibration
    file in the same schema as `camera_calibration.json`:
        { "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
          "distortion_coefficients": [[k1, k2, p1, p2, k3]] }

Dependencies
------------
    sudo apt install python3-opencv python3-picamera2
    pip install pyapriltags numpy

Quit with the 'q' key while the map window is focused.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from picamera2 import Picamera2
from pyapriltags import Detector


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

CONFIG_PATH = os.environ.get("VEHICLE_CONFIG", "vehicle_config.json")

DEFAULT_CONFIG = {
    "tag_family": "tag36h11",
    "tag_size_cm": 14.65,

    "front_camera_id": 0,
    "rear_camera_id": 1,
    "front_camera_offset_cm": 12.0,
    "rear_camera_offset_cm": 10.0,

    "front_calibration_path": "front_calibration.json",
    "rear_calibration_path": "rear_calibration.json",

    "image_width": 1280,
    "image_height": 720,

    "map_size_cm": 200.0,
    "map_pixels": 400,
    "grid_cm": 50,

    "anchors": {
        "0": {"x": 100.0, "y": 200.0, "facing_yaw": 180.0},
        "1": {"x": 200.0, "y": 125.0, "facing_yaw": 270.0},
        "2": {"x": 200.0, "y":  75.0, "facing_yaw": 270.0},
        "3": {"x": 100.0, "y":   0.0, "facing_yaw":   0.0},
        "4": {"x":   0.0, "y":  75.0, "facing_yaw":  90.0},
        "5": {"x":   0.0, "y": 125.0, "facing_yaw":  90.0},
    },
}


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[config] wrote default config to {path}")
    with open(path) as f:
        cfg = json.load(f)
    # Merge missing keys from defaults so old configs keep working.
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load camera_matrix / distortion_coefficients from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    D = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
    return K, D


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #

def tag_rotation_in_map(facing_yaw_deg: float) -> np.ndarray:
    """
    Rotation that maps a vector in the AprilTag's local frame
    (X = right when viewed from front, Y = down, Z = into the tag/wall)
    into map frame (X = East, Y = North, Z = Up).

    `facing_yaw_deg` is the compass bearing the tag's front face points to,
    measured clockwise from North (so 0 = N, 90 = E, 180 = S, 270 = W).
    """
    t = math.radians(facing_yaw_deg)
    c, s = math.cos(t), math.sin(t)
    # Columns are tag-frame basis vectors expressed in map frame:
    #   tag X -> ( -cos t,  sin t, 0 )       (left of viewer in horizontal plane)
    #   tag Y -> (  0,      0,    -1 )       (down)
    #   tag Z -> ( -sin t, -cos t, 0 )       (into the tag, opposite of facing)
    return np.array([
        [-c,   0.0, -s],
        [ s,   0.0, -c],
        [ 0.0, -1.0, 0.0],
    ])


def compass_yaw_from_forward(forward_xyz: np.ndarray) -> float:
    """Compass bearing (deg, CW from North) of a 3D vector projected to XY."""
    return math.degrees(math.atan2(forward_xyz[0], forward_xyz[1])) % 360.0


# --------------------------------------------------------------------------- #
# Camera worker                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class PoseSample:
    x: float
    y: float
    yaw: float
    tag_id: int
    timestamp: float


class CameraWorker(threading.Thread):
    """Captures frames from one Pi Camera, detects AprilTags, and writes the
    derived vehicle pose into shared state."""

    def __init__(
        self,
        name: str,
        camera_index: int,
        K: np.ndarray,
        D: np.ndarray,
        is_rear: bool,
        config: dict,
        shared_state: dict,
    ):
        super().__init__(daemon=True)
        self.name = name
        self.is_rear = is_rear
        self.shared_state = shared_state
        self.config = config

        self.tag_size = float(config["tag_size_cm"])
        self.offset = float(
            config["rear_camera_offset_cm"] if is_rear
            else config["front_camera_offset_cm"]
        )
        self.K = K
        self.D = D

        # Per-thread detector instance (the pyapriltags wrapper is not
        # guaranteed to be safe to share across threads).
        self.detector = Detector(
            families=config["tag_family"],
            nthreads=2,
            quad_decimate=1.0,
            refine_edges=1,
        )

        # Tag corners in the tag's local frame (units: cm). pyapriltags returns
        # corners in this order: bottom-left, bottom-right, top-right, top-left
        # (counter-clockwise when the tag is viewed upright from its front).
        h = self.tag_size / 2.0
        self.obj_pts = np.array([
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0],
        ], dtype=np.float64)

        self.picam = Picamera2(camera_index)
        cam_cfg = self.picam.create_video_configuration(
            main={"size": (config["image_width"], config["image_height"]),
                  "format": "RGB888"},
        )
        self.picam.configure(cam_cfg)
        self.picam.start()
        time.sleep(0.5)  # sensor warm-up

        self._running = True

    def stop(self) -> None:
        self._running = False
        try:
            self.picam.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------ #

    def run(self) -> None:
        while self._running:
            try:
                frame = self.picam.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                detections = self.detector.detect(gray)
                sample = self._best_pose(detections)
                with self.shared_state["lock"]:
                    self.shared_state[self.name] = sample
            except Exception as e:
                print(f"[{self.name}] worker error: {e}")
                time.sleep(0.1)

    def _best_pose(self, detections) -> Optional[PoseSample]:
        anchors = self.config["anchors"]
        candidates = []
        for d in detections:
            if str(d.tag_id) not in anchors:
                continue
            img_pts = np.asarray(d.corners, dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_pts, img_pts, self.K, self.D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue
            R_tc, _ = cv2.Rodrigues(rvec)
            t_tc = tvec.reshape(3)
            candidates.append((float(np.linalg.norm(t_tc)), d.tag_id, R_tc, t_tc))

        if not candidates:
            return None

        # Closest tag gives the most accurate pose.
        candidates.sort(key=lambda c: c[0])
        _, tag_id, R_tc, t_tc = candidates[0]
        return self._vehicle_pose(tag_id, R_tc, t_tc, anchors)

    def _vehicle_pose(self, tag_id, R_tc, t_tc, anchors) -> PoseSample:
        anchor = anchors[str(tag_id)]

        # Camera in tag frame.
        R_ct = R_tc.T
        t_ct = -R_ct @ t_tc

        # Tag in map frame.
        R_tm = tag_rotation_in_map(anchor["facing_yaw"])
        t_tm = np.array([anchor["x"], anchor["y"], 0.0])

        # Camera in map frame.
        R_cm = R_tm @ R_ct
        t_cm = t_tm + R_tm @ t_ct

        cam_yaw = compass_yaw_from_forward(R_cm[:, 2])

        # Reconcile with the vehicle's forward direction. The rear camera
        # looks backward, so the vehicle heads in the opposite direction,
        # and the rear camera sits BEHIND the vehicle center (so the vehicle
        # is in front of the camera by `offset` cm along the vehicle's
        # forward direction). The front camera is the mirror case.
        if self.is_rear:
            vehicle_yaw = (cam_yaw + 180.0) % 360.0
            offset_sign = +1.0
        else:
            vehicle_yaw = cam_yaw % 360.0
            offset_sign = -1.0

        yaw_rad = math.radians(vehicle_yaw)
        vx = t_cm[0] + offset_sign * self.offset * math.sin(yaw_rad)
        vy = t_cm[1] + offset_sign * self.offset * math.cos(yaw_rad)

        return PoseSample(
            x=float(vx), y=float(vy), yaw=float(vehicle_yaw),
            tag_id=int(tag_id), timestamp=time.time(),
        )


# --------------------------------------------------------------------------- #
# Pose fusion                                                                 #
# --------------------------------------------------------------------------- #

def fuse_poses(snapshot: dict, max_age_s: float = 0.5) -> Optional[PoseSample]:
    """Combine front+rear estimates into one unified pose."""
    now = time.time()
    fresh = []
    for key in ("front", "rear"):
        s = snapshot.get(key)
        if s is not None and (now - s.timestamp) <= max_age_s:
            fresh.append(s)

    if not fresh:
        return None
    if len(fresh) == 1:
        return fresh[0]

    xs = [s.x for s in fresh]
    ys = [s.y for s in fresh]
    sins = [math.sin(math.radians(s.yaw)) for s in fresh]
    coss = [math.cos(math.radians(s.yaw)) for s in fresh]
    yaw = math.degrees(math.atan2(
        sum(sins) / len(sins),
        sum(coss) / len(coss),
    )) % 360.0
    return PoseSample(
        x=sum(xs) / len(xs),
        y=sum(ys) / len(ys),
        yaw=yaw,
        tag_id=-1,  # fused
        timestamp=max(s.timestamp for s in fresh),
    )


# --------------------------------------------------------------------------- #
# Map rendering                                                               #
# --------------------------------------------------------------------------- #

GREEN = (0, 200, 0)        # in bounds  (BGR)
ORANGE = (0, 165, 255)     # out of bounds
ANCHOR_COLOR = (180, 60, 60)
GRID_COLOR = (215, 215, 215)
BORDER_COLOR = (90, 90, 90)
TEXT_COLOR = (30, 30, 30)


def render_map(pose: Optional[PoseSample], config: dict) -> np.ndarray:
    px = int(config["map_pixels"])
    cm = float(config["map_size_cm"])
    grid = float(config["grid_cm"])
    s = px / cm  # pixels per cm

    img = np.full((px, px, 3), 245, dtype=np.uint8)

    # --- grid lines -------------------------------------------------------- #
    g = grid
    while g <= cm + 1e-6:
        v = int(round(g * s))
        cv2.line(img, (v, 0), (v, px - 1), GRID_COLOR, 1)
        cv2.line(img, (0, px - v), (px - 1, px - v), GRID_COLOR, 1)
        cv2.putText(img, f"{int(g)}", (v + 2, px - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (140, 140, 140), 1)
        cv2.putText(img, f"{int(g)}", (2, px - v - 2),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (140, 140, 140), 1)
        g += grid
    cv2.rectangle(img, (0, 0), (px - 1, px - 1), BORDER_COLOR, 1)

    # --- anchors ----------------------------------------------------------- #
    for tid, a in config["anchors"].items():
        ax = max(0, min(px - 1, int(round(a["x"] * s))))
        ay = max(0, min(px - 1, int(round(px - a["y"] * s))))
        cv2.drawMarker(img, (ax, ay), ANCHOR_COLOR,
                       cv2.MARKER_SQUARE, 12, 2)
        cv2.putText(img, f"#{tid}", (ax + 7, ay - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ANCHOR_COLOR, 1)

    # --- axis labels ------------------------------------------------------- #
    cv2.putText(img, "N (Y+)", (px // 2 - 28, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
    cv2.putText(img, "E (X+)", (px - 62, px // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
    cv2.putText(img, "(0,0)", (4, px - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    cv2.putText(img, f"({int(cm)},{int(cm)})", (px - 70, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

    # --- vehicle ----------------------------------------------------------- #
    if pose is None:
        cv2.putText(img, "No tag in view", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
        cv2.putText(img, "Press 'q' to quit", (6, px - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
        return img

    in_bounds = (0.0 <= pose.x <= cm) and (0.0 <= pose.y <= cm)
    color = GREEN if in_bounds else ORANGE

    cx = int(round(pose.x * s))
    cy = int(round(px - pose.y * s))
    cx_d = max(0, min(px - 1, cx))
    cy_d = max(0, min(px - 1, cy))

    cv2.circle(img, (cx_d, cy_d), 7, color, -1)
    cv2.circle(img, (cx_d, cy_d), 7, (0, 0, 0), 1)

    # heading arrow: yaw = 0 -> +Y (up in image), yaw = 90 -> +X (right)
    L = 28
    yr = math.radians(pose.yaw)
    ex = int(round(cx_d + L * math.sin(yr)))
    ey = int(round(cy_d - L * math.cos(yr)))
    cv2.arrowedLine(img, (cx_d, cy_d), (ex, ey), color, 2, tipLength=0.3)

    info = f"X: {pose.x:7.2f} cm   Y: {pose.y:7.2f} cm   Yaw: {pose.yaw:6.1f} deg"
    status = "IN BOUNDS" if in_bounds else "OUT OF BOUNDS"
    cv2.putText(img, info, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
    cv2.putText(img, status, (6, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(img, "Press 'q' to quit", (6, px - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)
    return img


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    config = load_config(CONFIG_PATH)

    front_K, front_D = load_calibration(config["front_calibration_path"])
    rear_K, rear_D = load_calibration(config["rear_calibration_path"])

    shared_state: dict = {"lock": threading.Lock(), "front": None, "rear": None}

    front = CameraWorker(
        name="front",
        camera_index=int(config["front_camera_id"]),
        K=front_K, D=front_D,
        is_rear=False,
        config=config,
        shared_state=shared_state,
    )
    rear = CameraWorker(
        name="rear",
        camera_index=int(config["rear_camera_id"]),
        K=rear_K, D=rear_D,
        is_rear=True,
        config=config,
        shared_state=shared_state,
    )
    front.start()
    rear.start()

    cv2.namedWindow("Vehicle Map", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            with shared_state["lock"]:
                snap = {k: shared_state[k] for k in ("front", "rear")}
            pose = fuse_poses(snap)
            cv2.imshow("Vehicle Map", render_map(pose, config))
            if (cv2.waitKey(20) & 0xFF) == ord("q"):
                break
    finally:
        front.stop()
        rear.stop()
        # Give the worker threads a moment to exit cleanly.
        front.join(timeout=1.0)
        rear.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
