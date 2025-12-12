"""
AprilTag Pose Estimation for Raspberry Pi 5 Camera Module 3
Detects AprilTag 36h11 tag ID 0 and computes camera position relative to the tag
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

# Parking area coordinate system parameters
AREA_SIZE = 200  # 200x200 cm parking area
TAG_POSITION_AREA = (100, 200)  # Tag position in area coordinates (cm) - middle of upper wall

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


def draw_tag_axes(image, camera_matrix, dist_coeffs, rvec, tvec, tag_size):
    """
    Draw coordinate axes on the tag to visualize pose.
    
    Args:
        image: Image to draw on
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector
        tvec: Translation vector
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
    
    Args:
        tvec: Translation vector (camera position relative to tag)
        rvec: Rotation vector (camera rotation relative to tag)
    
    Returns:
        Dictionary with formatted pose information
    """
    # Position (camera relative to tag)
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
        img, "=== CAMERA POSE ===", (start_x, y_offset),
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


def transform_to_area_coordinates(tvec, rvec):
    """
    Transform camera position from tag coordinate system to parking area coordinate system.
    
    Transformation rules:
    - new_x = 100 - old_z (where old_z is tag Z coordinate in cm)
    - new_y = 200 - old_x (where old_x is tag X coordinate in cm)
    - new_orientation = 90 + old_pitch_angle (in degrees)
    - Origin at bottom-left, 0 degrees = facing east (right wall)
    
    Args:
        tvec: Translation vector (camera position relative to tag) in meters
        rvec: Rotation vector (camera rotation relative to tag)
    
    Returns:
        (x_area, y_area, yaw_area): Car position in area coordinates (cm, cm, degrees)
    """
    # Convert tvec from meters to cm
    x_tag, y_tag, z_tag = tvec.flatten() * 100  # Convert to cm
    
    # Apply transformation: new_x = 100 - old_z, new_y = 200 - old_x
    x_area = 100 - z_tag
    y_area = 200 - x_tag
    
    # Calculate pitch angle from rotation vector
    R_tag, _ = cv2.Rodrigues(rvec)
    # Extract pitch angle (rotation around Y axis)
    # Pitch is the angle between the camera's forward direction and the horizontal plane
    # We can get it from the rotation matrix
    pitch_rad = -atan2(R_tag[2, 0], sqrt(R_tag[2, 1]**2 + R_tag[2, 2]**2))
    pitch_deg = degrees(pitch_rad)
    
    # Apply transformation: new_orientation = 90 + old_pitch_angle
    yaw_area = 90 + pitch_deg
    
    return x_area, y_area, yaw_area


def create_area_visualization(x_area, y_area, yaw_area):
    """
    Create a visualization of the parking area with car position.
    
    Args:
        x_area: Car X position in area coordinates (cm)
        y_area: Car Y position in area coordinates (cm)
        yaw_area: Car yaw angle in area coordinates (degrees)
    
    Returns:
        Visualization image
    """
    # Create image (scale: 2 pixels per cm, so 400x400 pixels for 200x200 cm)
    scale = 2  # Scale factor for better visibility
    img_size = AREA_SIZE * scale
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw grid
    grid_spacing = 50  # 50 cm grid
    for i in range(0, AREA_SIZE + 1, grid_spacing):
        x_pixel = i * scale
        cv2.line(img, (x_pixel, 0), (x_pixel, img_size), (200, 200, 200), 1)
        cv2.line(img, (0, x_pixel), (img_size, x_pixel), (200, 200, 200), 1)
    
    # Draw area boundary
    cv2.rectangle(img, (0, 0), (img_size - 1, img_size - 1), (0, 0, 0), 2)
    
    # Draw tag position
    tag_x_pixel = TAG_POSITION_AREA[0] * scale
    tag_y_pixel = (AREA_SIZE - TAG_POSITION_AREA[1]) * scale  # Flip Y axis (image Y increases downward)
    cv2.circle(img, (tag_x_pixel, tag_y_pixel), 8, (0, 0, 255), -1)  # Red circle
    cv2.circle(img, (tag_x_pixel, tag_y_pixel), 8, (0, 0, 0), 2)  # Black border
    cv2.putText(img, "TAG", (tag_x_pixel - 15, tag_y_pixel - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw tag orientation arrow (facing south)
    arrow_length = 15 * scale
    tag_arrow_end_x = tag_x_pixel
    tag_arrow_end_y = tag_y_pixel + arrow_length  # South is downward in image
    cv2.arrowedLine(img, (tag_x_pixel, tag_y_pixel), 
                   (tag_arrow_end_x, tag_arrow_end_y),
                   (0, 0, 255), 3, tipLength=0.3)
    
    # Draw car position
    car_x_pixel = x_area * scale
    car_y_pixel = (AREA_SIZE - y_area) * scale  # Flip Y axis
    
    # Check if car is within area bounds
    if 0 <= x_area <= AREA_SIZE and 0 <= y_area <= AREA_SIZE:
        # Draw car as a circle
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 255, 0), -1)  # Green circle
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 0, 0), 2)  # Black border
        
        # Draw car orientation arrow
        arrow_length = 12 * scale
        yaw_rad = np.radians(yaw_area)
        car_arrow_end_x = car_x_pixel + arrow_length * np.cos(yaw_rad)
        car_arrow_end_y = car_y_pixel - arrow_length * np.sin(yaw_rad)  # Negative because Y is flipped
        cv2.arrowedLine(img, (int(car_x_pixel), int(car_y_pixel)),
                       (int(car_arrow_end_x), int(car_arrow_end_y)),
                       (0, 255, 0), 3, tipLength=0.3)
        
        # Draw coordinates text near car
        coord_text = f"({x_area:.1f}, {y_area:.1f})"
        yaw_text = f"{yaw_area:.1f}°"
        cv2.putText(img, coord_text, (int(car_x_pixel) + 10, int(car_y_pixel) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, yaw_text, (int(car_x_pixel) + 10, int(car_y_pixel) + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    else:
        # Car is outside area - draw with different color
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 165, 255), -1)  # Orange
        cv2.circle(img, (int(car_x_pixel), int(car_y_pixel)), 6, (0, 0, 0), 2)
    
    # Add title and labels
    cv2.putText(img, "PARKING AREA (200x200 cm)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"Car Position: ({x_area:.1f}, {y_area:.1f}) cm", (10, img_size - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, f"Car Yaw: {yaw_area:.1f} deg", (10, img_size - 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add axis labels
    cv2.putText(img, "X (East)", (img_size - 80, img_size - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Y (North)", (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def main():
    """
    Main function to detect AprilTag and compute camera pose.
    """
    print("="*60)
    print("AprilTag Pose Estimation")
    print("="*60)
    print(f"Target Tag: {TAG_FAMILY} ID {TARGET_TAG_ID}")
    print(f"Tag Size: {TAG_SIZE*100}cm")
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
    
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    print("\nCamera ready. Looking for AprilTag...")
    print("Press 'q' to quit")
    print("="*60)
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect AprilTags
            tags_result = detect_apriltag(gray, detector, detector_type)
            
            # Convert to BGR for OpenCV drawing
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            tag_found = False
            pose_info = ""
            
            # Process detected tags based on detector type
            if detector_type == 'pyapriltags':
                # pyapriltags returns a list of tag objects
                for tag in tags_result:
                    if tag.tag_id == TARGET_TAG_ID:
                        tag_found = True
                        
                        # Get tag corners (in image coordinates)
                        # pyapriltags returns corners as array of 4 points
                        tag_corners = tag.corners
                        
                        # Draw tag outline
                        corners_int = tag_corners.astype(int)
                        cv2.polylines(display_frame, [corners_int], True, (0, 255, 0), 2)
                        
                        # Draw tag ID
                        center = tag.center.astype(int)
                        cv2.putText(display_frame, f"ID: {tag.tag_id}", 
                                   (center[0] - 30, center[1] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Compute camera pose
                        success, rvec, tvec = compute_camera_pose(
                            tag_corners, camera_matrix, dist_coeffs, TAG_SIZE
                        )
                        
                        if success:
                            # Draw coordinate axes
                            draw_tag_axes(display_frame, camera_matrix, dist_coeffs, 
                                        rvec, tvec, TAG_SIZE)
                            
                            # Get pose information
                            pose_data = format_pose_info(tvec, rvec)
                            
                            # Display pose info on image with readable formatting
                            draw_pose_info(display_frame, pose_data)
                            
                            # Transform to area coordinates
                            x_area, y_area, yaw_area = transform_to_area_coordinates(tvec, rvec)
                            
                            # Create and display area visualization
                            area_img = create_area_visualization(x_area, y_area, yaw_area)
                            cv2.imshow("Parking Area - Car Position", area_img)
            else:  # opencv
                # OpenCV returns (corners, ids, rejected)
                corners, ids, rejected = tags_result
                if ids is not None and len(ids) > 0:
                    for i, tag_id in enumerate(ids.flatten()):
                        if tag_id == TARGET_TAG_ID:
                            tag_found = True
                            
                            # Get tag corners (in image coordinates)
                            # OpenCV returns corners as (1, 4, 2) array
                            tag_corners = corners[i]
                            
                            # Draw tag outline
                            corners_int = tag_corners.astype(int)
                            cv2.polylines(display_frame, [corners_int], True, (0, 255, 0), 2)
                            
                            # Calculate center for drawing ID
                            center = tag_corners.mean(axis=0).astype(int)
                            cv2.putText(display_frame, f"ID: {tag_id}", 
                                       (center[0] - 30, center[1] - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Compute camera pose
                            success, rvec, tvec = compute_camera_pose(
                                tag_corners, camera_matrix, dist_coeffs, TAG_SIZE
                            )
                            
                            if success:
                                # Draw coordinate axes
                                draw_tag_axes(display_frame, camera_matrix, dist_coeffs, 
                                            rvec, tvec, TAG_SIZE)
                                
                                # Get pose information
                                pose_data = format_pose_info(tvec, rvec)
                                
                                # Display pose info on image with readable formatting
                                draw_pose_info(display_frame, pose_data)
                                
                                # Transform to area coordinates
                                x_area, y_area, yaw_area = transform_to_area_coordinates(tvec, rvec)
                                
                                # Create and display area visualization
                                area_img = create_area_visualization(x_area, y_area, yaw_area)
                                cv2.imshow("Parking Area - Car Position", area_img)
            
            # Display status
            if not tag_found:
                cv2.putText(display_frame, "Tag not detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Looking for tag ID {TARGET_TAG_ID}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show empty area visualization when tag not found
                area_img_empty = np.ones((AREA_SIZE * 2, AREA_SIZE * 2, 3), dtype=np.uint8) * 240
                # Draw grid
                grid_spacing = 50
                for i in range(0, AREA_SIZE + 1, grid_spacing):
                    x_pixel = i * 2
                    cv2.line(area_img_empty, (x_pixel, 0), (x_pixel, AREA_SIZE * 2), (200, 200, 200), 1)
                    cv2.line(area_img_empty, (0, x_pixel), (AREA_SIZE * 2, x_pixel), (200, 200, 200), 1)
                cv2.rectangle(area_img_empty, (0, 0), (AREA_SIZE * 2 - 1, AREA_SIZE * 2 - 1), (0, 0, 0), 2)
                # Draw tag
                tag_x_pixel = TAG_POSITION_AREA[0] * 2
                tag_y_pixel = (AREA_SIZE - TAG_POSITION_AREA[1]) * 2
                cv2.circle(area_img_empty, (tag_x_pixel, tag_y_pixel), 8, (0, 0, 255), -1)
                cv2.circle(area_img_empty, (tag_x_pixel, tag_y_pixel), 8, (0, 0, 0), 2)
                cv2.putText(area_img_empty, "TAG", (tag_x_pixel - 15, tag_y_pixel - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(area_img_empty, "PARKING AREA (200x200 cm)", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(area_img_empty, "Waiting for tag detection...", (10, AREA_SIZE * 2 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("Parking Area - Car Position", area_img_empty)
            
            # Resize for display if too large
            if display_frame.shape[1] > 1280:
                scale = 1280 / display_frame.shape[1]
                new_width = int(display_frame.shape[1] * scale)
                new_height = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            cv2.imshow("AprilTag Pose Estimation - Press 'q' to quit", display_frame)
            
            # Print pose info to console
            if tag_found and pose_info:
                # Extract distance from pose_info or compute it
                if detector_type == 'pyapriltags':
                    for tag in tags_result:
                        if tag.tag_id == TARGET_TAG_ID:
                            success, rvec, tvec = compute_camera_pose(
                                tag.corners, camera_matrix, dist_coeffs, TAG_SIZE
                            )
                            if success:
                                distance = sqrt(sum(tvec.flatten()**2))
                                print("\r" + " "*80, end="")  # Clear line
                                print(f"\rTag detected! Distance: {distance*100:.2f} cm", end="")
                                break
                else:  # opencv
                    if ids is not None:
                        for i, tag_id in enumerate(ids.flatten()):
                            if tag_id == TARGET_TAG_ID:
                                success, rvec, tvec = compute_camera_pose(
                                    corners[i], camera_matrix, dist_coeffs, TAG_SIZE
                                )
                                if success:
                                    distance = sqrt(sum(tvec.flatten()**2))
                                    print("\r" + " "*80, end="")  # Clear line
                                    print(f"\rTag detected! Distance: {distance*100:.2f} cm", end="")
                                    break
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n\nExiting...")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        # Close both windows
        try:
            cv2.destroyWindow("AprilTag Pose Estimation - Press 'q' to quit")
            cv2.destroyWindow("Parking Area - Car Position")
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

