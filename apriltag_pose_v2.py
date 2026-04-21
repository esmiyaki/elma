"""
AprilTag Pose Estimation for Raspberry Pi 5 Camera Module 3 (V2)
Detects AprilTag 36h11 tag ID 0 and computes tag position relative to camera
Coordinate system: Tag is the origin
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import json
import os
import time
from math import degrees, atan2, sqrt
from camera_calibration import load_calibration, CALIBRATION_FILE

# AprilTag parameters
TAG_FAMILY = "tag36h11"  # AprilTag family
TAG_SIZE = 0.06  # Tag size in meters (6cm default - adjust based on your tag)

# Multi-camera parameters (Raspberry Pi Camera Module 3 x2)
# Front camera faces vehicle forward; back camera is mounted 180° opposite (back-to-back).
FRONT_CAMERA_INDEX = 0
BACK_CAMERA_INDEX = 1

# Map coordinate system parameters
MAP_SIZE = 200  # 200x200 cm map

# Known AprilTag placements in map coordinates (cm).
# Map coordinate system (as used by create_map_visualization):
# - Origin at bottom-left (0, 0)
# - X increases to the right (East)
# - Y increases upward (North)
#
# Each tag defines an anchor point in the map; when that tag is observed,
# we estimate the camera/car position relative to that anchor.
TAG_MAP_POSITIONS_CM = {
    0: (100, 200),
    1: (200, 125),
    2: (200, 75),
    3: (100, 0),
    4: (0, 75),
    5: (0, 125),
}

# AprilTag 3D object points (in tag coordinate system)
# Tag center is at origin, tag lies in XY plane, Z points outward
# Corners are ordered: bottom-left, bottom-right, top-right, top-left
def get_tag_object_points(tag_size):
    """
    Get 3D object points for AprilTag corners in tag coordinate system.
    
    Args:
        tag_size: Size of the tag in meters (edge length)
    
    Returns:
        3D points array (4 corners, 3 coordinates each)
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
    """
    Detect AprilTag in image.
    
    Args:
        image: Input image (grayscale)
        detector: AprilTag detector object
        detector_type: Type of detector ('pyapriltags' or 'opencv')
    
    Returns:
        For pyapriltags: List of detected tags
        For opencv: (corners, ids, rejected)
    """
    if detector_type == 'pyapriltags':
        tags = detector.detect(image)
        return tags
    else:  # opencv
        corners, ids, rejected = detector.detect(image)
        return corners, ids, rejected


def invert_pose(rvec, tvec):
    """
    Invert pose transformation: from camera relative to tag -> tag relative to camera.
    
    If camera is at position C with rotation R relative to tag, then:
    - Tag position relative to camera: -R^T * C
    - Tag rotation relative to camera: R^T
    
    Args:
        rvec: Rotation vector (camera rotation relative to tag)
        tvec: Translation vector (camera position relative to tag)
    
    Returns:
        tag_rvec: Rotation vector (tag rotation relative to camera)
        tag_tvec: Translation vector (tag position relative to camera)
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Invert rotation: R_inv = R^T (transpose)
    R_inv = R.T
    
    # Convert back to rotation vector
    tag_rvec, _ = cv2.Rodrigues(R_inv)
    
    # Invert translation: tag_tvec = -R^T * tvec
    tag_tvec = -R_inv @ tvec
    
    return tag_rvec, tag_tvec


def draw_tag_axes(image, camera_matrix, dist_coeffs, rvec, tvec, tag_size):
    """
    Draw coordinate axes on the tag to visualize pose.
    Note: This function uses the original rvec/tvec (camera relative to tag)
    for drawing purposes.
    
    Args:
        image: Image to draw on
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector (camera rotation relative to tag)
        tvec: Translation vector (camera position relative to tag)
        tag_size: Tag size in meters
    """
    # Define axis points in tag coordinate system
    axis_length = tag_size * 0.5
    axis_points = np.array([
        [0, 0, 0],           # Origin
        [axis_length, 0, 0], # X axis (red)
        [0, axis_length, 0], # Y axis (green)
        [0, 0, -axis_length] # Z axis (blue, pointing toward camera)
    ], dtype=np.float32)
    
    # Project axis points to image
    image_points, _ = cv2.projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    image_points = image_points.reshape(-1, 2).astype(int)
    origin = tuple(image_points[0])
    
    # Draw axes
    cv2.line(image, origin, tuple(image_points[1]), (0, 0, 255), 3)  # X - Red
    cv2.line(image, origin, tuple(image_points[2]), (0, 255, 0), 3)  # Y - Green
    cv2.line(image, origin, tuple(image_points[3]), (255, 0, 0), 3)  # Z - Blue
    
    # Draw tag outline
    tag_corners = get_tag_object_points(tag_size)
    tag_corners_2d, _ = cv2.projectPoints(
        tag_corners, rvec, tvec, camera_matrix, dist_coeffs
    )
    tag_corners_2d = tag_corners_2d.reshape(-1, 2).astype(int)
    
    # Draw tag outline
    cv2.line(image, tuple(tag_corners_2d[0]), tuple(tag_corners_2d[1]), (255, 255, 0), 2)
    cv2.line(image, tuple(tag_corners_2d[1]), tuple(tag_corners_2d[2]), (255, 255, 0), 2)
    cv2.line(image, tuple(tag_corners_2d[2]), tuple(tag_corners_2d[3]), (255, 255, 0), 2)
    cv2.line(image, tuple(tag_corners_2d[3]), tuple(tag_corners_2d[0]), (255, 255, 0), 2)


def rotation_vector_to_euler(rvec):
    """
    Convert rotation vector to Euler angles (roll, pitch, yaw).
    
    Args:
        rvec: Rotation vector (3x1)
    
    Returns:
        (roll, pitch, yaw) in degrees
    """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles (ZYX convention)
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


def apply_yaw180_flip_to_pose(tag_rvec, tag_tvec):
    """
    Convert a pose from a back-facing camera frame into the front/vehicle frame.

    Assumes the back camera is rotated 180 degrees around the camera Y axis
    (so X and Z invert: (x, z) -> (-x, -z)). This keeps distances unchanged,
    but makes positions/orientations comparable to the front camera.

    Args:
        tag_rvec: Rotation vector (tag rotation relative to camera)
        tag_tvec: Translation vector (tag position relative to camera)

    Returns:
        (tag_rvec_flipped, tag_tvec_flipped)
    """
    # 180° rotation about camera Y axis.
    R_flip = np.array(
        [[-1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )

    R_tag, _ = cv2.Rodrigues(tag_rvec)
    R_tag_flipped = R_flip @ R_tag
    tag_rvec_flipped, _ = cv2.Rodrigues(R_tag_flipped)

    tag_tvec_flipped = R_flip @ tag_tvec
    return tag_rvec_flipped, tag_tvec_flipped


def find_best_tag_in_frame(
    frame_rgb,
    detector,
    detector_type,
    camera_matrix,
    dist_coeffs,
    tag_size,
    pose_adjust_fn=None,
):
    """
    Detect tags in a frame, estimate pose for each supported tag ID, and return the closest.

    Args:
        frame_rgb: RGB image from Picamera2
        detector: AprilTag detector object
        detector_type: 'pyapriltags' or 'opencv'
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        tag_size: Tag size in meters
        pose_adjust_fn: Optional function(tag_rvec, tag_tvec) -> (adj_rvec, adj_tvec)

    Returns:
        (display_frame_bgr, tag_found, best)
        where best includes both raw and adjusted poses:
          best['tag_rvec'], best['tag_tvec'] are adjusted if pose_adjust_fn provided,
          best['raw_tag_rvec'], best['raw_tag_tvec'] always store the raw pose.
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    tags_result = detect_apriltag(gray, detector, detector_type)

    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    tag_found = False
    best = None

    def consider_candidate(tag_id_int, tag_corners, center_xy):
        nonlocal best, tag_found
        tag_found = True

        # Draw tag outline + ID
        corners_int = tag_corners.astype(int)
        cv2.polylines(display_frame, [corners_int], True, (0, 255, 0), 2)
        cv2.putText(
            display_frame,
            f"ID: {tag_id_int}",
            (center_xy[0] - 30, center_xy[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Compute camera pose relative to tag (solvePnP result)
        success, rvec_cam, tvec_cam = compute_camera_pose(
            tag_corners, camera_matrix, dist_coeffs, tag_size
        )
        if not success:
            return

        # Invert pose: tag pose relative to camera (raw, camera's own frame)
        raw_tag_rvec, raw_tag_tvec = invert_pose(rvec_cam, tvec_cam)
        distance_m = float(sqrt(sum(raw_tag_tvec.flatten() ** 2)))

        tag_rvec, tag_tvec = raw_tag_rvec, raw_tag_tvec
        if pose_adjust_fn is not None:
            tag_rvec, tag_tvec = pose_adjust_fn(raw_tag_rvec, raw_tag_tvec)

        if best is None or distance_m < best["distance_m"]:
            best = {
                "tag_id": int(tag_id_int),
                "distance_m": distance_m,
                "raw_tag_rvec": raw_tag_rvec,
                "raw_tag_tvec": raw_tag_tvec,
                "tag_rvec": tag_rvec,
                "tag_tvec": tag_tvec,
                "rvec_cam": rvec_cam,
                "tvec_cam": tvec_cam,
                "tag_corners": tag_corners,
                "tag_center": (int(center_xy[0]), int(center_xy[1])),
            }

    if detector_type == "pyapriltags":
        for tag in tags_result:
            if tag.tag_id in TAG_MAP_POSITIONS_CM:
                tag_corners = tag.corners
                center = tag.center.astype(int)
                consider_candidate(int(tag.tag_id), tag_corners, (int(center[0]), int(center[1])))
    else:
        corners, ids, rejected = tags_result
        if ids is not None and len(ids) > 0:
            for i, tag_id in enumerate(ids.flatten()):
                tag_id_int = int(tag_id)
                if tag_id_int in TAG_MAP_POSITIONS_CM:
                    tag_corners = corners[i]
                    center = tag_corners.mean(axis=0).astype(int)
                    consider_candidate(tag_id_int, tag_corners, (int(center[0]), int(center[1])))

    return display_frame, tag_found, best


def compute_camera_pose(tag_corners, camera_matrix, dist_coeffs, tag_size):
    """
    Compute camera pose relative to AprilTag.
    
    Args:
        tag_corners: 2D image coordinates of tag corners (4 points)
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        tag_size: Tag size in meters
    
    Returns:
        success: Boolean indicating if pose was computed
        rvec: Rotation vector (camera rotation relative to tag)
        tvec: Translation vector (camera position relative to tag)
    """
    # Get 3D object points
    obj_points = get_tag_object_points(tag_size)
    
    # Ensure corners are in correct format
    tag_corners = np.array(tag_corners, dtype=np.float32).reshape(4, 1, 2)
    
    # Solve PnP to get pose
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
    """
    Draw text with a semi-transparent background for better readability.
    
    Args:
        img: Image to draw on
        text: Text to draw
        position: (x, y) position of text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        alpha: Background transparency (0.0 to 1.0)
    
    Returns:
        Height of the drawn text block
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle
    x, y = position
    padding = 5
    bg_top_left = (x - padding, y - text_height - padding)
    bg_bottom_right = (x + text_width + padding, y + baseline + padding)
    
    # Draw semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, bg_top_left, bg_bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text on top
    cv2.putText(img, text, position, font, font_scale, text_color, thickness)
    
    return text_height + baseline + padding * 2


def format_pose_info(tvec, rvec):
    """
    Format pose information as readable dictionary.
    Note: This now represents tag position relative to camera (tag is origin).
    
    Args:
        tvec: Translation vector (tag position relative to camera)
        rvec: Rotation vector (tag rotation relative to camera)
    
    Returns:
        Dictionary with formatted pose information
    """
    # Position (tag relative to camera)
    x, y, z = tvec.flatten()
    distance = sqrt(x**2 + y**2 + z**2)
    
    # Orientation (Euler angles)
    roll, pitch, yaw = rotation_vector_to_euler(rvec)
    
    return {
        'x': x * 100,  # cm
        'y': y * 100,  # cm
        'z': z * 100,  # cm
        'distance': distance * 100,  # cm
        'roll': roll,
        'pitch': pitch,
        'yaw': yaw
    }


def draw_pose_info(img, pose_data, start_x=10, start_y=30):
    """
    Draw pose information on image with readable formatting.
    
    Args:
        img: Image to draw on
        pose_data: Dictionary with pose information from format_pose_info
        start_x: Starting X position
        start_y: Starting Y position
    """
    y_offset = start_y
    font_scale = 0.8
    thickness = 2
    line_spacing = 35
    
    # Title
    y_offset += draw_text_with_background(
        img, "=== TAG POSE (Tag Origin) ===", (start_x, y_offset),
        font_scale=1.0, thickness=2, text_color=(0, 255, 255), bg_color=(0, 0, 0), alpha=0.8
    )
    y_offset += line_spacing
    
    # Position section
    y_offset += draw_text_with_background(
        img, "POSITION (cm):", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  X: {pose_data['x']:>7.2f} cm", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  Y: {pose_data['y']:>7.2f} cm", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  Z: {pose_data['z']:>7.2f} cm", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  Distance: {pose_data['distance']:>6.2f} cm", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(0, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing * 2
    
    # Orientation section
    y_offset += draw_text_with_background(
        img, "ORIENTATION (deg):", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  Roll:  {pose_data['roll']:>6.2f}°", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  Pitch: {pose_data['pitch']:>6.2f}°", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )
    y_offset += line_spacing
    
    y_offset += draw_text_with_background(
        img, f"  Yaw:   {pose_data['yaw']:>6.2f}°", (start_x, y_offset),
        font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
    )


def transform_to_map_coordinates(tvec, rvec, tag_id, tag_map_xy_cm=(100, 200)):
    """
    Transform tag position to map coordinate system.
    
    Map coordinate system:
    - Origin at bottom-left (0, 0)
    - 200x200 cm area
    - 0 degrees = facing north (top wall)
    - Angle increases clockwise
    
    Transformation varies by tag ID:
    - Tag 0: x_map = tag_map_x + x_tag, y_map = tag_map_y - z_tag
    - Tag 1 & 2: x_map = tag_map_x - z_tag, y_map = tag_map_y + x_tag
    - Tag 3: x_map = tag_map_x - x_tag, y_map = tag_map_y + z_tag
    - Tag 4 & 5: x_map = tag_map_x + z_tag, y_map = tag_map_y - x_tag
    
    Args:
        tvec: Translation vector (tag position relative to camera) in meters
        rvec: Rotation vector (tag rotation relative to camera)
        tag_id: Tag ID (0-5) to determine transformation
        tag_map_xy_cm: Tag's anchor position in map coordinates (cm)
    
    Returns:
        (x_map, y_map, yaw_map): Car position in map coordinates (cm, cm, degrees)
    """
    # Convert tvec from meters to cm
    x_tag, y_tag, z_tag = tvec.flatten() * 100  # Convert to cm
    
    tag_map_x, tag_map_y = tag_map_xy_cm

    # Apply tag-specific transformation
    if tag_id == 0:
        # Tag 0: x_map = tag_map_x + x_tag, y_map = tag_map_y - z_tag
        x_map = tag_map_x + x_tag
        y_map = tag_map_y - z_tag
    elif tag_id in [1, 2]:
        # Tag 1 & 2: x_map = tag_map_x - z_tag, y_map = tag_map_y + x_tag
        x_map = tag_map_x - z_tag
        y_map = tag_map_y + x_tag
    elif tag_id == 3:
        # Tag 3: x_map = tag_map_x - x_tag, y_map = tag_map_y + z_tag
        x_map = tag_map_x - x_tag
        y_map = tag_map_y + z_tag
    elif tag_id in [4, 5]:
        # Tag 4 & 5: x_map = tag_map_x + z_tag, y_map = tag_map_y - x_tag
        x_map = tag_map_x + z_tag
        y_map = tag_map_y - x_tag
    else:
        # Fallback to tag 0 transformation for unknown tags
        x_map = tag_map_x + x_tag
        y_map = tag_map_y - z_tag
    
    # Calculate pitch angle from rotation vector
    R_tag, _ = cv2.Rodrigues(rvec)
    # Extract pitch angle (rotation around Y axis)
    pitch_rad = -atan2(R_tag[2, 0], sqrt(R_tag[2, 1]**2 + R_tag[2, 2]**2))
    pitch_deg = degrees(pitch_rad)
    
    # Apply tag-specific yaw offset based on tag orientation
    if tag_id == 0:
        # Tag 0: no offset
        yaw_map = pitch_deg
    elif tag_id in [1, 2]:
        # Tags 1 & 2: -90 degrees
        yaw_map = pitch_deg - 90
    elif tag_id == 3:
        # Tag 3: +180 degrees
        yaw_map = pitch_deg + 180
    elif tag_id in [4, 5]:
        # Tags 4 & 5: +90 degrees
        yaw_map = pitch_deg + 90
    else:
        # Fallback to tag 0 for unknown tags
        yaw_map = pitch_deg
    
    return x_map, y_map, yaw_map


def create_map_visualization(x_map, y_map, yaw_map):
    """
    Create a visualization of the map with car position.
    
    Args:
        x_map: Car X position in map coordinates (cm)
        y_map: Car Y position in map coordinates (cm)
        yaw_map: Car yaw angle in map coordinates (degrees, 0 = north, clockwise positive)
    
    Returns:
        Visualization image
    """
    # Create image (scale: 2 pixels per cm, so 400x400 pixels for 200x200 cm)
    scale = 2  # Scale factor for better visibility
    img_size = MAP_SIZE * scale
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw grid
    grid_spacing = 50  # 50 cm grid
    for i in range(0, MAP_SIZE + 1, grid_spacing):
        x_pixel = i * scale
        cv2.line(img, (x_pixel, 0), (x_pixel, img_size), (200, 200, 200), 1)
        cv2.line(img, (0, x_pixel), (img_size, x_pixel), (200, 200, 200), 1)
    
    # Draw map boundary
    cv2.rectangle(img, (0, 0), (img_size - 1, img_size - 1), (0, 0, 0), 2)
    
    # Draw car position
    car_x_pixel = x_map * scale
    car_y_pixel = (MAP_SIZE - y_map) * scale  # Flip Y axis (image Y increases downward)
    
    # Check if car is within map bounds
    if 0 <= x_map <= MAP_SIZE and 0 <= y_map <= MAP_SIZE:
        # Draw car as a circle
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 255, 0), -1)  # Green circle
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 0, 0), 2)  # Black border
        
        # Draw car orientation arrow
        # In map coordinates: 0 degrees = north (up), clockwise positive
        # For drawing in image coordinates: 0 degrees = -90 degrees in standard math (upward)
        arrow_length = 12 * scale
        # Convert map yaw to drawing angle: map 0° (north) = -90° in standard coords
        # Map increases clockwise, standard increases counterclockwise, so negate
        yaw_rad = np.radians(-yaw_map - 90)
        car_arrow_end_x = car_x_pixel + arrow_length * np.cos(yaw_rad)
        car_arrow_end_y = car_y_pixel + arrow_length * np.sin(yaw_rad)  # Image Y increases downward
        cv2.arrowedLine(img, (int(car_x_pixel), int(car_y_pixel)),
                       (int(car_arrow_end_x), int(car_arrow_end_y)),
                       (0, 255, 0), 3, tipLength=0.3)
        
        # Draw coordinates text near car
        coord_text = f"({x_map:.1f}, {y_map:.1f})"
        yaw_text = f"{yaw_map:.1f}°"
        cv2.putText(img, coord_text, (int(car_x_pixel) + 10, int(car_y_pixel) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, yaw_text, (int(car_x_pixel) + 10, int(car_y_pixel) + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    else:
        # Car is outside map - draw with different color
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 165, 255), -1)  # Orange
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 0, 0), 2)
    
    # Add labels (removed "MAP (200x200 cm)" title)
    cv2.putText(img, f"Car Position: ({x_map:.1f}, {y_map:.1f}) cm", (10, img_size - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, f"Car Yaw: {yaw_map:.1f} deg (0°=North, CW+)", (10, img_size - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add axis labels
    cv2.putText(img, "X (East)", (img_size - 80, img_size - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Y (North)", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def main():
    """
    Main function to detect AprilTag and compute tag pose relative to camera.
    """
    print("="*60)
    print("AprilTag Pose Estimation V2 (Tag as Origin)")
    print("="*60)
    print(f"Tag Family: {TAG_FAMILY}")
    print(f"Tag Size: {TAG_SIZE*100}cm")
    print("Coordinate System: Tag is the origin")
    print(f"Using {len(TAG_MAP_POSITIONS_CM)} tag anchors: {sorted(TAG_MAP_POSITIONS_CM.keys())}")
    print("="*60)
    
    # Load camera calibration
    camera_matrix, dist_coeffs = load_calibration()
    if camera_matrix is None:
        print("\nError: Camera calibration not found!")
        print("Please run camera_calibration.py first.")
        return
    
    # Initialize AprilTag detector - try pyapriltags first (most reliable)
    detector = None
    detector_type = None
    
    # Try pyapriltags (recommended for Raspberry Pi)
    try:
        import pyapriltags
        detector = pyapriltags.Detector(families=TAG_FAMILY)
        detector_type = 'pyapriltags'
        print(f"\npyapriltags detector initialized (family: {TAG_FAMILY})")
    except ImportError:
        # Try OpenCV's AprilTag detector (if available)
        try:
            detector = cv2.aruco.AprilTagDetector()
            detector_type = 'opencv'
            print(f"\nOpenCV AprilTag detector initialized (family: {TAG_FAMILY})")
        except AttributeError:
            print("\nError: No AprilTag detector available!")
            print("Please install one of the following:")
            print("  1. pyapriltags (recommended): pip install pyapriltags")
            print("  2. Or ensure OpenCV >= 4.7.0 with contrib modules")
            return
    
    # Initialize cameras
    print("Initializing cameras...")
    picam2_front = Picamera2(FRONT_CAMERA_INDEX)
    picam2_back = Picamera2(BACK_CAMERA_INDEX)

    config_front = picam2_front.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    config_back = picam2_back.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )

    picam2_front.configure(config_front)
    picam2_back.configure(config_back)

    picam2_front.start()
    picam2_back.start()
    
    print("\nCameras ready. Looking for AprilTag...")
    print("Press 'q' to quit")
    print("="*60)
    
    try:
        # Cap processing rate (pose estimation + UI updates)
        target_fps = 5.0
        frame_period_s = 1.0 / target_fps
        next_frame_t = time.perf_counter()

        while True:
            # Capture frames from both cameras
            frame_front = picam2_front.capture_array()
            frame_back = picam2_back.capture_array()

            # Process each frame independently. For the back camera, flip pose into front/vehicle frame.
            display_front, tag_found_front, best_front = find_best_tag_in_frame(
                frame_front,
                detector,
                detector_type,
                camera_matrix,
                dist_coeffs,
                TAG_SIZE,
                pose_adjust_fn=None,
            )
            display_back, tag_found_back, best_back = find_best_tag_in_frame(
                frame_back,
                detector,
                detector_type,
                camera_matrix,
                dist_coeffs,
                TAG_SIZE,
                pose_adjust_fn=apply_yaw180_flip_to_pose,
            )

            # Choose the closest tag across both cameras (distance is based on the raw pose).
            chosen = None
            chosen_cam = None  # 'front' or 'back'
            if best_front is not None:
                chosen = best_front
                chosen_cam = "front"
            if best_back is not None and (chosen is None or best_back["distance_m"] < chosen["distance_m"]):
                chosen = best_back
                chosen_cam = "back"

            # Add camera labels
            draw_text_with_background(
                display_front,
                "FRONT CAMERA",
                (10, 30),
                font_scale=0.9,
                thickness=2,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                alpha=0.6,
            )
            draw_text_with_background(
                display_back,
                "BACK CAMERA (pose flipped 180° for mapping)",
                (10, 30),
                font_scale=0.7,
                thickness=2,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                alpha=0.6,
            )

            # If we have at least one valid tag, use the closest one for pose + map.
            if chosen is not None:
                display_frame = display_front if chosen_cam == "front" else display_back

                # Draw coordinate axes for the chosen (closest) tag
                draw_tag_axes(
                    display_frame,
                    camera_matrix,
                    dist_coeffs,
                    chosen["rvec_cam"],
                    chosen["tvec_cam"],
                    TAG_SIZE
                )

                # Get pose information (tag pose relative to vehicle/front frame; back camera is flipped)
                pose_data = format_pose_info(chosen["tag_tvec"], chosen["tag_rvec"])

                # Display pose info on image with readable formatting
                draw_pose_info(display_frame, pose_data)

                # Transform to map coordinates using this tag's known anchor point
                tag_anchor_xy = TAG_MAP_POSITIONS_CM.get(chosen["tag_id"], (100, 200))
                x_map, y_map, yaw_map = transform_to_map_coordinates(
                    chosen["tag_tvec"],
                    chosen["tag_rvec"],
                    chosen["tag_id"],
                    tag_map_xy_cm=tag_anchor_xy
                )

                # Create and display map visualization
                map_img = create_map_visualization(x_map, y_map, yaw_map)
                cv2.imshow("Map - Car Position", map_img)
            else:
                # Show empty map without car representation when tag not detected by either camera
                map_img_empty = np.ones((MAP_SIZE * 2, MAP_SIZE * 2, 3), dtype=np.uint8) * 240
                grid_spacing = 50
                for i in range(0, MAP_SIZE + 1, grid_spacing):
                    x_pixel = i * 2
                    cv2.line(map_img_empty, (x_pixel, 0), (x_pixel, MAP_SIZE * 2), (200, 200, 200), 1)
                    cv2.line(map_img_empty, (0, x_pixel), (MAP_SIZE * 2, x_pixel), (200, 200, 200), 1)
                cv2.rectangle(map_img_empty, (0, 0), (MAP_SIZE * 2 - 1, MAP_SIZE * 2 - 1), (0, 0, 0), 2)
                cv2.putText(
                    map_img_empty,
                    "X (East)",
                    (MAP_SIZE * 2 - 80, MAP_SIZE * 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                cv2.putText(
                    map_img_empty,
                    "Y (North)",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                cv2.imshow("Map - Car Position", map_img_empty)
            
            # Display status on each camera view
            if not tag_found_front:
                cv2.putText(
                    display_front,
                    "Tag not detected",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            if not tag_found_back:
                cv2.putText(
                    display_back,
                    "Tag not detected",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            looking_for = f"Looking for tag IDs {sorted(TAG_MAP_POSITIONS_CM.keys())}"
            cv2.putText(
                display_front,
                looking_for,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_back,
                looking_for,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            
            # Resize for display if too large
            def resize_if_needed(img, max_w=1280):
                if img.shape[1] <= max_w:
                    return img
                scale = max_w / img.shape[1]
                new_w = int(img.shape[1] * scale)
                new_h = int(img.shape[0] * scale)
                return cv2.resize(img, (new_w, new_h))

            display_front_resized = resize_if_needed(display_front)
            display_back_resized = resize_if_needed(display_back)

            cv2.imshow("AprilTag - FRONT (Press 'q' to quit)", display_front_resized)
            cv2.imshow("AprilTag - BACK (Press 'q' to quit)", display_back_resized)
            
            # Print pose info to console
            if chosen is not None:
                print("\r" + " "*120, end="")  # Clear line
                print(
                    f"\rUsing {chosen_cam} cam | closest tag ID {chosen['tag_id']} @ anchor {TAG_MAP_POSITIONS_CM[chosen['tag_id']]} "
                    f"| Distance: {chosen['distance_m']*100:.2f} cm",
                    end=""
                )
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # FPS limiter (sleep the remainder of the frame period).
            next_frame_t += frame_period_s
            sleep_s = next_frame_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # If we're slower than target FPS, reset to avoid drift.
                next_frame_t = time.perf_counter()
        
        print("\n\nExiting...")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        picam2_front.stop()
        picam2_back.stop()
        cv2.destroyAllWindows()
        # Close both windows
        try:
            cv2.destroyWindow("AprilTag - FRONT (Press 'q' to quit)")
            cv2.destroyWindow("AprilTag - BACK (Press 'q' to quit)")
            cv2.destroyWindow("Map - Car Position")
        except:
            pass


if __name__ == "__main__":
    # Allow tag size to be specified via command line or environment variable
    import sys
    if len(sys.argv) > 1:
        try:
            TAG_SIZE = float(sys.argv[1]) / 100.0  # Convert cm to meters
            print(f"Using tag size: {TAG_SIZE*100}cm (from command line)")
        except ValueError:
            print(f"Invalid tag size: {sys.argv[1]}. Using default: {TAG_SIZE*100}cm")
    
    main()

