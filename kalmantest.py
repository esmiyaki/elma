"""
Standalone AprilTag pose visualization + Kalman smoothing (TEST ONLY).

Run:
  python3 kalmantest.py

Behavior:
  - Uses the same detection + map transform + visualizations as apriltag_pose.py
    (front/back camera windows + map window).
  - Applies a Kalman filter to smooth the final map pose (x_cm, y_cm, yaw_deg)
    before drawing on the map / printing.

This file is intentionally self-contained for testing: it imports apriltag_pose.py
but does not modify it and does not depend on rpi_stanley_controller.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

import apriltag_pose as base


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_rad(a: float, b: float) -> float:
    return wrap_pi(a - b)


@dataclass
class KalmanConfig:
    # Tuned for "smoothing but not lagging too much" in cm/deg scale.
    q_pos: float = 3.0  # cm^2 / s^2
    q_yaw: float = math.radians(20.0) ** 2  # rad^2 / s^2
    r_pos: float = 7.0 ** 2  # cm^2
    r_yaw: float = math.radians(12.0) ** 2  # rad^2
    dt_min: float = 1e-3
    dt_max: float = 0.5


class PoseKalmanCV:
    """
    Constant-velocity Kalman filter for [x, y, yaw] with velocity states.
    State: [x, y, yaw, vx, vy, vyaw]^T
    Measurement: [x, y, yaw]^T
    """

    def __init__(self, cfg: KalmanConfig):
        self.cfg = cfg
        self.x = np.zeros((6, 1), dtype=float)
        self.P = np.eye(6, dtype=float) * 1e6
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self._initialized = False
        self._t_last: Optional[float] = None

    def reset(self):
        self.x[:] = 0.0
        self.P[:] = np.eye(6) * 1e6
        self._initialized = False
        self._t_last = None

    def update(self, meas_x_cm: float, meas_y_cm: float, meas_yaw_rad: float, t: Optional[float] = None):
        if t is None:
            t = time.perf_counter()

        meas_yaw_rad = wrap_pi(float(meas_yaw_rad))

        if not self._initialized:
            self.x[0, 0] = float(meas_x_cm)
            self.x[1, 0] = float(meas_y_cm)
            self.x[2, 0] = meas_yaw_rad
            self.x[3:, 0] = 0.0
            self.P = np.eye(6, dtype=float)
            self.P[0, 0] = 60.0**2
            self.P[1, 1] = 60.0**2
            self.P[2, 2] = math.radians(60.0) ** 2
            self.P[3, 3] = 60.0**2
            self.P[4, 4] = 60.0**2
            self.P[5, 5] = math.radians(120.0) ** 2
            self._initialized = True
            self._t_last = t
            return

        dt = float(t - (self._t_last or t))
        self._t_last = t
        if dt < self.cfg.dt_min:
            dt = self.cfg.dt_min
        if dt > self.cfg.dt_max:
            self.reset()
            self.update(meas_x_cm, meas_y_cm, meas_yaw_rad, t=t)
            return

        F = np.eye(6, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        q = float(self.cfg.q_pos)
        qy = float(self.cfg.q_yaw)
        Q = np.zeros((6, 6), dtype=float)
        # x/vx
        Q[0, 0] = 0.25 * dt**4 * q
        Q[0, 3] = 0.5 * dt**3 * q
        Q[3, 0] = 0.5 * dt**3 * q
        Q[3, 3] = dt**2 * q
        # y/vy
        Q[1, 1] = 0.25 * dt**4 * q
        Q[1, 4] = 0.5 * dt**3 * q
        Q[4, 1] = 0.5 * dt**3 * q
        Q[4, 4] = dt**2 * q
        # yaw/vyaw
        Q[2, 2] = 0.25 * dt**4 * qy
        Q[2, 5] = 0.5 * dt**3 * qy
        Q[5, 2] = 0.5 * dt**3 * qy
        Q[5, 5] = dt**2 * qy

        # Predict
        self.x = F @ self.x
        self.x[2, 0] = wrap_pi(float(self.x[2, 0]))
        self.P = F @ self.P @ F.T + Q

        # Update (wrapped yaw innovation)
        z = np.array([[float(meas_x_cm)], [float(meas_y_cm)], [meas_yaw_rad]], dtype=float)
        z_pred = self.H @ self.x
        y = z - z_pred
        y[2, 0] = angle_diff_rad(float(z[2, 0]), float(z_pred[2, 0]))

        R = np.diag([self.cfg.r_pos, self.cfg.r_pos, self.cfg.r_yaw]).astype(float)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2, 0] = wrap_pi(float(self.x[2, 0]))
        I = np.eye(6, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get(self):
        return float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])


def main():
    print("=" * 60)
    print("AprilTag Pose + Kalman (TEST)")
    print("=" * 60)

    camera_matrix, dist_coeffs = base.load_calibration()
    if camera_matrix is None:
        print("Error: Camera calibration not found. Run camera_calibration.py first.")
        return

    # Detector (same as base)
    detector = None
    detector_type = None
    try:
        import pyapriltags

        detector = pyapriltags.Detector(families=base.TAG_FAMILY)
        detector_type = "pyapriltags"
        print("Using pyapriltags.")
    except ImportError:
        try:
            detector = cv2.aruco.AprilTagDetector()
            detector_type = "opencv"
            print("Using OpenCV AprilTag detector.")
        except AttributeError:
            print("No AprilTag detector available (install pyapriltags or OpenCV>=4.7).")
            return

    # Cameras (same indices + configuration as base)
    picam2_front = base.Picamera2(base.FRONT_CAMERA_INDEX)
    picam2_back = base.Picamera2(base.BACK_CAMERA_INDEX)

    config_front = picam2_front.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    config_back = picam2_back.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    picam2_front.configure(config_front)
    picam2_back.configure(config_back)
    picam2_front.start()
    picam2_back.start()

    kf = PoseKalmanCV(KalmanConfig())

    # Limit UI/update rate similarly to base main()
    target_fps = 5.0
    frame_period_s = 1.0 / target_fps
    next_frame_t = time.perf_counter()

    try:
        while True:
            frame_front = picam2_front.capture_array()
            frame_back = picam2_back.capture_array()

            display_front, tag_found_front, best_front = base.find_best_tag_in_frame(
                frame_front,
                detector,
                detector_type,
                camera_matrix,
                dist_coeffs,
                base.TAG_SIZE,
                pose_adjust_fn=None,
            )
            display_back, tag_found_back, best_back = base.find_best_tag_in_frame(
                frame_back,
                detector,
                detector_type,
                camera_matrix,
                dist_coeffs,
                base.TAG_SIZE,
                pose_adjust_fn=None,
            )

            chosen = None
            chosen_cam = None
            if best_front is not None:
                chosen = best_front
                chosen_cam = "front"
            if best_back is not None and (chosen is None or best_back["distance_m"] < chosen["distance_m"]):
                chosen = best_back
                chosen_cam = "back"

            # Add camera labels (same style as base)
            base.draw_text_with_background(display_front, "FRONT CAMERA", (10, 30), font_scale=0.9, thickness=2)
            base.draw_text_with_background(display_back, "BACK CAMERA", (10, 30), font_scale=0.7, thickness=2)

            if chosen is not None:
                display_frame = display_front if chosen_cam == "front" else display_back

                base.draw_tag_axes(
                    display_frame,
                    camera_matrix,
                    dist_coeffs,
                    chosen["rvec_cam"],
                    chosen["tvec_cam"],
                    base.TAG_SIZE,
                )

                pose_data = base.format_pose_info(chosen["tag_tvec"], chosen["tag_rvec"])
                base.draw_pose_info(display_frame, pose_data)

                tag_anchor_xy = base.TAG_MAP_POSITIONS_CM.get(chosen["tag_id"], (100, 200))
                if chosen_cam == "back":
                    x_map, y_map, yaw_map = base.transform_to_map_coordinates_back_camera(
                        chosen["raw_tag_tvec"],
                        chosen["raw_tag_rvec"],
                        chosen["tag_id"],
                        tag_map_xy_cm=tag_anchor_xy,
                    )
                else:
                    x_map, y_map, yaw_map = base.transform_to_map_coordinates(
                        chosen["tag_tvec"],
                        chosen["tag_rvec"],
                        chosen["tag_id"],
                        tag_map_xy_cm=tag_anchor_xy,
                    )

                # Kalman smoothing in map yaw space (deg -> rad)
                yaw_rad = wrap_pi(math.radians(float(yaw_map)))
                kf.update(float(x_map), float(y_map), yaw_rad)
                x_f, y_f, yaw_f = kf.get()
                yaw_f_deg = (math.degrees(yaw_f)) % 360.0

                map_img = base.create_map_visualization(x_f, y_f, yaw_f_deg)
                cv2.imshow("Map - Car Position", map_img)

                print(
                    "\r" + " " * 140,
                    end="",
                )
                print(
                    f"\rKalman | cam={chosen_cam} tag={chosen['tag_id']} "
                    f"raw=({x_map:6.1f},{y_map:6.1f},{yaw_map:6.1f}deg) "
                    f"filt=({x_f:6.1f},{y_f:6.1f},{yaw_f_deg:6.1f}deg) "
                    f"dist={chosen['distance_m']*100:5.1f}cm",
                    end="",
                )
            else:
                # Empty map when no tag detected
                map_img_empty = np.ones((base.MAP_SIZE * 2, base.MAP_SIZE * 2, 3), dtype=np.uint8) * 240
                grid_spacing = 50
                for i in range(0, base.MAP_SIZE + 1, grid_spacing):
                    x_pixel = i * 2
                    cv2.line(map_img_empty, (x_pixel, 0), (x_pixel, base.MAP_SIZE * 2), (200, 200, 200), 1)
                    cv2.line(map_img_empty, (0, x_pixel), (base.MAP_SIZE * 2, x_pixel), (200, 200, 200), 1)
                cv2.rectangle(map_img_empty, (0, 0), (base.MAP_SIZE * 2 - 1, base.MAP_SIZE * 2 - 1), (0, 0, 0), 2)
                cv2.putText(map_img_empty, "X (East)", (base.MAP_SIZE * 2 - 80, base.MAP_SIZE * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(map_img_empty, "Y (North)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow("Map - Car Position", map_img_empty)

            if not tag_found_front:
                cv2.putText(display_front, "Tag not detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not tag_found_back:
                cv2.putText(display_back, "Tag not detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            looking_for = f"Looking for tag IDs {sorted(base.TAG_MAP_POSITIONS_CM.keys())}"
            cv2.putText(display_front, looking_for, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_back, looking_for, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            def resize_if_needed(img, max_w=1280):
                if img.shape[1] <= max_w:
                    return img
                scale = max_w / img.shape[1]
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                return cv2.resize(img, (new_w, new_h))

            cv2.imshow("AprilTag - FRONT (Press 'q' to quit)", resize_if_needed(display_front))
            cv2.imshow("AprilTag - BACK (Press 'q' to quit)", resize_if_needed(display_back))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # FPS limiter (same idea as base)
            next_frame_t += frame_period_s
            sleep_s = next_frame_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_frame_t = time.perf_counter()

    finally:
        picam2_front.stop()
        picam2_back.stop()
        cv2.destroyAllWindows()
        try:
            cv2.destroyWindow("AprilTag - FRONT (Press 'q' to quit)")
            cv2.destroyWindow("AprilTag - BACK (Press 'q' to quit)")
            cv2.destroyWindow("Map - Car Position")
        except Exception:
            pass


if __name__ == "__main__":
    main()

