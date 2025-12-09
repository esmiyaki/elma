"""
Camera Calibration for Raspberry Pi 5 with Camera Module 3
Uses a 7x7 checkerboard pattern (6x6 inner corners) with 2.5cm squares
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
from picamera2 import Picamera2

# Checkerboard parameters
CHECKERBOARD_SIZE = (6, 6)  # Inner corners (7x7 squares = 6x6 corners)
SQUARE_SIZE = 0.025  # 2.5cm in meters

# Calibration parameters
MIN_IMAGES = 10  # Minimum number of images needed for calibration
CALIBRATION_IMAGES_DIR = "calibration_images"
CALIBRATION_FILE = "camera_calibration.json"


def prepare_object_points():
    """
    Prepare 3D object points for checkerboard corners.
    Returns: numpy array of shape (N, 3) where N = rows * cols
    """
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale by square size
    return objp


def detect_checkerboard(image):
    """
    Detect checkerboard corners in an image.
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        ret: Boolean indicating if corners were found
        corners: Detected corner points (if found)
        image_with_corners: Image with corners drawn (for visualization)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + 
        cv2.CALIB_CB_FAST_CHECK + 
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    image_with_corners = image.copy()
    
    if ret:
        # Refine corner positions for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        
        # Draw corners
        cv2.drawChessboardCorners(image_with_corners, CHECKERBOARD_SIZE, corners_refined, ret)
        return True, corners_refined, image_with_corners
    
    return False, None, image_with_corners


def capture_calibration_images():
    """
    Interactive function to capture calibration images from camera.
    Press 's' to save image, 'q' to quit.
    """
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    # Create directory for calibration images
    Path(CALIBRATION_IMAGES_DIR).mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Camera Calibration Image Capture")
    print("="*60)
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE*100}cm")
    print("\nInstructions:")
    print("  - Move the checkerboard to different positions and angles")
    print("  - Make sure the entire checkerboard is visible")
    print("  - Press 's' to save image when checkerboard is detected")
    print("  - Press 'q' to quit and start calibration")
    print(f"  - Aim for at least {MIN_IMAGES} good images")
    print("="*60 + "\n")
    
    saved_count = 0
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Detect checkerboard
            ret, corners, frame_with_corners = detect_checkerboard(frame)
            
            # Display status
            status_text = f"Images saved: {saved_count}"
            if ret:
                status_text += " | CHECKERBOARD DETECTED - Press 's' to save"
                cv2.putText(frame_with_corners, "CHECKERBOARD DETECTED!", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                status_text += " | No checkerboard detected"
                cv2.putText(frame_with_corners, "No checkerboard detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(frame_with_corners, status_text, 
                       (10, frame_with_corners.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_corners, "Press 's' to save, 'q' to quit", 
                       (10, frame_with_corners.shape[0] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display image
            # Resize for display if too large
            display_frame = frame_with_corners.copy()
            if display_frame.shape[1] > 1280:
                scale = 1280 / display_frame.shape[1]
                new_width = int(display_frame.shape[1] * scale)
                new_height = int(display_frame.shape[0] * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            cv2.imshow("Camera Calibration - Press 's' to save, 'q' to quit", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if ret:
                    # Save image
                    filename = os.path.join(CALIBRATION_IMAGES_DIR, f"calibration_{saved_count:03d}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    print(f"Saved image {saved_count}: {filename}")
                else:
                    print("Cannot save: Checkerboard not detected!")
            
            elif key == ord('q'):
                break
        
        print(f"\nTotal images saved: {saved_count}")
        
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
    
    return saved_count


def calibrate_camera():
    """
    Perform camera calibration using saved calibration images.
    
    Returns:
        ret: Calibration success flag
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    print("\n" + "="*60)
    print("Starting Camera Calibration")
    print("="*60)
    
    # Prepare object points
    objp = prepare_object_points()
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    image_files = sorted(Path(CALIBRATION_IMAGES_DIR).glob("*.jpg"))
    
    if len(image_files) < MIN_IMAGES:
        print(f"Error: Need at least {MIN_IMAGES} images, found {len(image_files)}")
        return False, None, None, None, None
    
    print(f"Found {len(image_files)} calibration images")
    print("Detecting checkerboard corners in images...")
    
    successful_detections = 0
    
    for i, img_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {img_path.name}", end=" ... ")
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("Failed to load image")
            continue
        
        ret, corners, _ = detect_checkerboard(img)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful_detections += 1
            print("OK")
        else:
            print("Failed to detect corners")
    
    print(f"\nSuccessfully detected corners in {successful_detections} images")
    
    if successful_detections < MIN_IMAGES:
        print(f"Error: Need at least {MIN_IMAGES} successful detections, got {successful_detections}")
        return False, None, None, None, None
    
    # Get image dimensions from first image
    img = cv2.imread(str(image_files[0]))
    img_size = (img.shape[1], img.shape[0])  # (width, height)
    
    print(f"\nImage size: {img_size[0]}x{img_size[1]}")
    print("Performing calibration...")
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,
        None
    )
    
    if ret:
        print("\nCalibration successful!")
        print("\nCamera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                             camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(objpoints)
        print(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
        print("(Lower is better, typically < 0.5 pixels is good)")
        
    else:
        print("Calibration failed!")
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def save_calibration(camera_matrix, dist_coeffs):
    """
    Save calibration parameters to JSON file.
    
    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "image_size": None,  # Will be set when loading
        "checkerboard_size": CHECKERBOARD_SIZE,
        "square_size": SQUARE_SIZE
    }
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"\nCalibration saved to: {CALIBRATION_FILE}")


def load_calibration():
    """
    Load calibration parameters from JSON file.
    
    Returns:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file '{CALIBRATION_FILE}' not found")
        return None, None
    
    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)
    
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"])
    
    print(f"Calibration loaded from: {CALIBRATION_FILE}")
    return camera_matrix, dist_coeffs


def test_undistortion():
    """
    Test undistortion by showing original and undistorted images side by side.
    """
    camera_matrix, dist_coeffs = load_calibration()
    
    if camera_matrix is None:
        print("Cannot test: No calibration data found")
        return
    
    print("\nTesting undistortion...")
    print("Press 'q' to quit")
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    try:
        while True:
            frame = picam2.capture_array()
            
            # Undistort image
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, 
                                       None, new_camera_matrix)
            
            # Combine original and undistorted side by side
            combined = np.hstack([frame, undistorted])
            
            # Resize for display if too large
            if combined.shape[1] > 1920:
                scale = 1920 / combined.shape[1]
                new_width = int(combined.shape[1] * scale)
                new_height = int(combined.shape[0] * scale)
                combined = cv2.resize(combined, (new_width, new_height))
            
            cv2.putText(combined, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Undistorted", (combined.shape[1]//2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Undistortion Test - Press 'q' to quit", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


def main():
    """
    Main function to run camera calibration workflow.
    """
    print("="*60)
    print("Raspberry Pi 5 Camera Module 3 Calibration")
    print("="*60)
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE*100}cm ({SQUARE_SIZE}m)")
    print("="*60)
    
    # Check if calibration already exists
    if os.path.exists(CALIBRATION_FILE):
        print(f"\nExisting calibration found: {CALIBRATION_FILE}")
        response = input("Do you want to:\n  1) Capture new images and recalibrate\n  2) Test existing calibration\n  3) Exit\nChoice (1/2/3): ")
        
        if response == "1":
            # Capture new images
            capture_calibration_images()
            # Calibrate
            ret, camera_matrix, dist_coeffs, _, _ = calibrate_camera()
            if ret:
                save_calibration(camera_matrix, dist_coeffs)
                test_undistortion()
        elif response == "2":
            test_undistortion()
        else:
            print("Exiting...")
            return
    else:
        # First time calibration
        print("\nNo existing calibration found. Starting calibration process...")
        capture_calibration_images()
        
        # Calibrate
        ret, camera_matrix, dist_coeffs, _, _ = calibrate_camera()
        
        if ret:
            save_calibration(camera_matrix, dist_coeffs)
            
            # Ask if user wants to test
            response = input("\nDo you want to test the undistortion? (y/n): ")
            if response.lower() == 'y':
                test_undistortion()


if __name__ == "__main__":
    main()

