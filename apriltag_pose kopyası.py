"""
AprilTag Pose Estimation - WORLD COORDINATE SYSTEM
Tag is fixed at Origin (0,0,0).
Calculates Camera (Robot) Position relative to the Tag.
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import json
import os
from math import degrees, atan2, sqrt
from camera_calibration import load_calibration, CALIBRATION_FILE

# AprilTag parameters
TARGET_TAG_ID = 0  # Tag ID to detect
TAG_FAMILY = "tag36h11"  # AprilTag family
TAG_SIZE = 0.06  # Tag size in meters (6cm default - adjust based on your tag)

def get_tag_object_points(tag_size):
    """
    Get 3D object points for AprilTag corners (centered at 0,0,0).
    """
    half_size = tag_size / 2.0
    obj_points = np.array([
        [-half_size, -half_size, 0],  # Bottom-left
        [half_size, -half_size, 0],   # Bottom-right
        [half_size, half_size, 0],    # Top-right
        [-half_size, half_size, 0]    # Top-left
    ], dtype=np.float32)
    return obj_points


def detect_apriltag(image, detector, detector_type='pyapriltags'):
    if detector_type == 'pyapriltags':
        tags = detector.detect(image)
        return tags
    else:  # opencv
        corners, ids, rejected = detector.detect(image)
        return corners, ids, rejected


def draw_tag_axes(image, camera_matrix, dist_coeffs, rvec, tvec, tag_size):
    """
    Draw coordinate axes on the tag image.
    This visualizes the Tag's orientation relative to the camera.
    """
    axis_length = tag_size * 0.5
    axis_points = np.array([
        [0, 0, 0],           # Origin
        [axis_length, 0, 0], # X axis (red)
        [0, axis_length, 0], # Y axis (green)
        [0, 0, -axis_length] # Z axis (blue)
    ], dtype=np.float32)
    
    image_points, _ = cv2.projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    image_points = image_points.reshape(-1, 2).astype(int)
    origin = tuple(image_points[0])
    
    cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X - Red
    cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y - Green
    cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z - Blue


def rotation_vector_to_euler(rvec):
    """
    Convert rotation vector to Euler angles (degrees).
    """
    R, _ = cv2.Rodrigues(rvec)
    sy = sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = atan2(R[2, 1], R[2, 2])
        pitch = atan2(-R[2, 0], sy)
        yaw = atan2(R[1, 0], R[0, 0])
    else:
        roll = atan2(-R[1, 2], R[1, 1])
        pitch = atan2(-R[2, 0], sy)
        yaw = 0
    return degrees(roll), degrees(pitch), degrees(yaw)


# --- YENİ EKLENEN KRİTİK FONKSİYON: TERS DÖNÜŞÜM ---
def compute_camera_world_pose(rvec_tag, tvec_tag):
    """
    Calculates Camera's position in the World (Tag) Coordinate System.
    Formula: P_cam = -R^T * t
    """
    # 1. Rotation Vector -> Rotation Matrix
    R, _ = cv2.Rodrigues(rvec_tag)
    
    # 2. Transpose Matrix (Inverse Rotation)
    R_inv = np.transpose(R)
    
    # 3. Calculate Camera Position: -R_inv * t
    cam_pos = -np.dot(R_inv, tvec_tag)
    
    # 4. Calculate Camera Rotation in World Frame
    cam_rvec, _ = cv2.Rodrigues(R_inv)
    
    return cam_pos, cam_rvec
# ---------------------------------------------------


def solve_pnp_wrapper(tag_corners, camera_matrix, dist_coeffs, tag_size):
    """
    Standard PnP solver. 
    Returns Tag position relative to Camera (Raw Data).
    """
    obj_points = get_tag_object_points(tag_size)
    tag_corners = np.array(tag_corners, dtype=np.float32).reshape(4, 1, 2)
    
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        tag_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return success, rvec, tvec


def draw_text_with_background(img, text, position, font_scale=0.7, thickness=2, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    padding = 5
    bg_top_left = (x - padding, y - text_height - padding)
    bg_bottom_right = (x + text_width + padding, y + baseline + padding)
    overlay = img.copy()
    cv2.rectangle(overlay, bg_top_left, bg_bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)
    return text_height + baseline + padding * 2


def draw_pose_info_world(img, cam_pos, cam_rvec, start_x=10, start_y=30):
    """
    Displays CAMERA position relative to Tag Origin.
    """
    y = start_y
    font_scale = 0.8
    thickness = 2
    
    # Convert vectors to readable formats
    x, y_pos, z = cam_pos.flatten() * 100 # Convert to cm
    dist = sqrt(x**2 + y_pos**2 + z**2)
    roll, pitch, yaw = rotation_vector_to_euler(cam_rvec)
    
    # Draw Info
    y += draw_text_with_background(img, "=== REAL CAMERA POSITION ===", (start_x, y), 
                                  font_scale=1.0, text_color=(0, 255, 255))
    y += 10
    y += draw_text_with_background(img, "(Origin is the Tag)", (start_x, y), 
                                  font_scale=0.6, text_color=(200, 200, 200))
    y += 20
    
    y += draw_text_with_background(img, f"X (Right/Left): {x:>6.2f} cm", (start_x, y)) + 10
    y += draw_text_with_background(img, f"Y (Up/Down):    {y_pos:>6.2f} cm", (start_x, y)) + 10
    y += draw_text_with_background(img, f"Z (Fwd/Back):   {z:>6.2f} cm", (start_x, y)) + 10
    y += draw_text_with_background(img, f"Distance:       {dist:>6.2f} cm", (start_x, y), 
                                  text_color=(0, 255, 255)) + 20
    
    y += draw_text_with_background(img, f"Cam Yaw (Angle): {yaw:>6.2f} deg", (start_x, y))


def main():
    print("="*60)
    print("AprilTag Localization: Camera Position in World Frame")
    print("Tag is Origin (0,0,0)")
    print("="*60)
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_calibration()
    if camera_matrix is None:
        print("\nError: Camera calibration not found!")
        return
    
    # Initialize AprilTag detector
    detector = None
    detector_type = None
    try:
        import pyapriltags
        detector = pyapriltags.Detector(families=TAG_FAMILY)
        detector_type = 'pyapriltags'
        print("Using: pyapriltags")
    except ImportError:
        try:
            detector = cv2.aruco.AprilTagDetector()
            detector_type = 'opencv'
            print("Using: OpenCV ArUco")
        except AttributeError:
            print("Error: No detector found.")
            return
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    # --- CRITICAL: LOCK AUTOFOCUS ---
    try:
        # Focus at infinity (0.0) or specific distance
        picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
        print("Autofocus LOCKED (Infinite).")
    except Exception as e:
        print(f"Warning: Could not lock focus: {e}")
    # --------------------------------
    
    print("\nCamera ready. Press 'q' to quit")
    
    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect
            tags_result = detect_apriltag(gray, detector, detector_type)
            
            tag_found = False
            
            # Helper to unify detection formats
            detected_tags = []
            if detector_type == 'pyapriltags':
                detected_tags = [(t.tag_id, t.corners) for t in tags_result]
            else:
                corners, ids, _ = tags_result
                if ids is not None:
                    for i in range(len(ids)):
                        detected_tags.append((ids[i][0], corners[i]))

            # Process Tags
            for tag_id, corners in detected_tags:
                if tag_id == TARGET_TAG_ID:
                    tag_found = True
                    
                    # 1. Solve PnP (Get Tag relative to Camera)
                    success, rvec, tvec = solve_pnp_wrapper(corners, camera_matrix, dist_coeffs, TAG_SIZE)
                    
                    if success:
                        # 2. Draw Axes (Raw data is correct for image overlay)
                        draw_tag_axes(display_frame, camera_matrix, dist_coeffs, rvec, tvec, TAG_SIZE)
                        
                        # 3. COMPUTE REAL CAMERA POSITION (Inverse Transform)
                        # This is the magic step!
                        cam_pos, cam_rvec = compute_camera_world_pose(rvec, tvec)
                        
                        # 4. Draw Info (Using Transformed Data)
                        draw_pose_info_world(display_frame, cam_pos, cam_rvec)
                        
                        # Draw Box
                        pts = corners.astype(int).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

            if not tag_found:
                cv2.putText(display_frame, "Tag Not Detected", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Resize & Show
            if display_frame.shape[1] > 1280:
                scale = 1280 / display_frame.shape[1]
                display_frame = cv2.resize(display_frame, (int(display_frame.shape[1]*scale), int(display_frame.shape[0]*scale)))
            
            cv2.imshow("Real Camera Position", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()