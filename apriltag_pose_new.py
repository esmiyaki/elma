"""
AprilTag Pose Estimation for Raspberry Pi 5 Camera Module 3 (V2)

Detects AprilTag 36h11 tag IDs 0..5 and reports the car pose in map
coordinates. Two cameras (front + back) feed into one tracker.

Pose pipeline per frame:
  1. Detect tags with pyapriltags (or OpenCV ArUco fallback).
  2. Gate on `decision_margin` to drop low-confidence detections.
  3. Run cv2.SOLVEPNP_IPPE_SQUARE for each accepted tag, pick the lower
     reprojection-error solution out of the two planar-pose branches,
     and refine with cv2.solvePnPRefineLM.
  4. Convert each tag pose into a candidate map-frame pose (x,y,yaw)
     using the per-tag config table.
  5. Fuse all candidates from both cameras with inverse-variance
     weighting (decision_margin / (distance_cm**2 + 1)).
  6. Smooth the fused pose through an EMA inside the tracker, with a
     jump-reset that lets large step changes pass through.

The dict shape returned by AprilTagMapPoseTracker.update() is
backward-compatible:
    {x_cm, y_cm, yaw_deg, tag_id, camera, distance_cm}
plus a few new optional fields (n_tags, fused, reproj_error_px).
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2
import json
import os
import time
import math
from math import degrees, atan2, sqrt, cos, sin
from camera_calibration import load_calibration, CALIBRATION_FILE

# AprilTag parameters
TAG_FAMILY = "tag36h11"  # AprilTag family
TAG_SIZE = 0.1465  # Tag edge length in meters

# Multi-camera parameters (Raspberry Pi Camera Module 3 x2)
# Front camera faces vehicle forward; back camera is mounted 180 deg opposite (back-to-back).
FRONT_CAMERA_INDEX = 0
BACK_CAMERA_INDEX = 1

# Camera-mount offsets along the body-forward direction (cm). The pose we
# report is the car's reference point (front axle); the cameras live some
# distance ahead/behind it. Subtracting (camera_offset * forward_in_map)
# lands us back at the reference point.
FRONT_CAMERA_OFFSET_CM = 4.0
BACK_CAMERA_OFFSET_CM = 18.0

# pyapriltags' decision_margin is the difference (in normalised units)
# between the most-likely and second-most-likely tag code. Higher = more
# confident. 30 is conservative for tag36h11 in good light; tune down
# for darker scenes if detections drop too often.
MIN_DECISION_MARGIN = 30.0

# IPPE_SQUARE returns up to 2 candidate poses for a planar marker. If
# the better candidate's reprojection error exceeds this we drop the
# detection as untrustworthy (tag too small / too tilted / partially
# occluded).
MAX_REPROJ_ERROR_PX = 4.0

# When fusing several tags we drop any whose yaw differs from the first
# (highest-weight) candidate by more than this -- that's the "ambiguity
# flip" failure mode of planar-pose estimation.
FUSION_YAW_OUTLIER_DEG = 30.0

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


# ----- per-tag map config -------------------------------------------------
#
# Each tag is a planar marker on a wall. The tag's local frame (after
# invert_pose) has X to the right, Y up, Z out of the face. To express
# the camera position in the map we rotate the tag-frame translation
# (x_tag, z_tag) by a 2x2 R that depends only on which wall the tag is
# on, then add the tag's anchor point in the map.
#
# Columns of R are (tag_x_in_map, tag_z_in_map):
#   north wall:  tag_x = +X (east)   tag_z = -Y (south)   R = [[ 1, 0],[ 0,-1]]
#   east wall:   tag_x = -Y (south)  tag_z = -X (west)    R = [[ 0,-1],[-1, 0]]
#   south wall:  tag_x = -X (west)   tag_z = +Y (north)   R = [[-1, 0],[ 0, 1]]
#   west wall:   tag_x = +Y (north)  tag_z = +X (east)    R = [[ 0, 1],[ 1, 0]]
#
# yaw_offset_deg is added to (-pitch_deg) to produce the map-frame yaw.
# It encodes how the tag is rotated about the up axis relative to the
# canonical "north wall facing south" tag.

@dataclass(frozen=True)
class TagConfig:
    anchor_xy_cm: Tuple[float, float]
    R_tag_to_map: np.ndarray         # 2x2; columns are tag_x and tag_z in map
    yaw_offset_deg: float


def _R(a, b, c, d):
    return np.array([[a, b], [c, d]], dtype=np.float64)


TAG_CONFIGS = {
    0: TagConfig((100, 200), _R( 1,  0,  0, -1),    0),
    1: TagConfig((200, 125), _R( 0, -1, -1,  0),  -90),
    2: TagConfig((200,  75), _R( 0, -1, -1,  0),  -90),
    3: TagConfig((100,   0), _R(-1,  0,  0,  1),  180),
    4: TagConfig((  0,  75), _R( 0,  1,  1,  0),   90),
    5: TagConfig((  0, 125), _R( 0,  1,  1,  0),   90),
}


# ----- detector factory ---------------------------------------------------

def make_apriltag_detector():
    """Construct a tuned pyapriltags detector (or fall back to OpenCV).

    Tuning rationale (Pi 5, Camera Module 3, daylight):
    - nthreads=4: the Pi 5 has 4 cores otherwise idle during pose work.
    - quad_decimate=1.0: decimating the image speeds detection but
      destroys corner accuracy; we need the precision more than the FPS.
    - quad_sigma=0.0: the IMX708 imager is sharp enough that the
      Gaussian smoothing pass tends to *hurt* detection range.
    - refine_edges=1: sub-pixel corner refinement, free win for PnP.
    - decode_sharpening=0.25: helps decode when motion blur softens
      the bit pattern.

    Returns (detector, detector_type) where detector_type is
    'pyapriltags' or 'opencv'.
    """
    try:
        import pyapriltags
        detector = pyapriltags.Detector(
            families=TAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )
        return detector, "pyapriltags"
    except ImportError:
        pass
    try:
        detector = cv2.aruco.AprilTagDetector()
        return detector, "opencv"
    except AttributeError as e:
        raise RuntimeError(
            "No AprilTag detector available. Install pyapriltags or use OpenCV >= 4.7."
        ) from e


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
    """Convert a rotation vector to Euler angles (roll, pitch, yaw) in degrees.

    Note: ``pitch`` here is the rotation around the camera Y axis without
    the ``sqrt`` clamp that the textbook decomposition uses, so it covers
    a full 360 deg unambiguously (no gimbal-lock confusion at +/-90 deg).
    Roll and yaw are the conventional Tait-Bryan extractions.
    """
    R, _ = cv2.Rodrigues(rvec)
    pitch = atan2(-R[2, 0], -R[2, 2])
    roll = atan2(R[2, 1], R[2, 2])
    yaw = atan2(R[1, 0], R[0, 0])
    return degrees(roll), degrees(pitch), degrees(yaw)

def find_tags_in_frame(
    frame_rgb,
    detector,
    detector_type,
    camera_matrix,
    dist_coeffs,
    tag_size,
    *,
    prior_poses=None,
    draw=True,
):
    """Detect all accepted tags in a frame and estimate pose for each.

    Detections are gated by ``MIN_DECISION_MARGIN`` (pyapriltags only;
    OpenCV's detector doesn't expose a margin) and by
    ``MAX_REPROJ_ERROR_PX`` after the IPPE_SQUARE solve.

    Args:
        frame_rgb: RGB image from Picamera2.
        detector / detector_type: from ``make_apriltag_detector()``.
        camera_matrix, dist_coeffs: from ``load_calibration()``.
        tag_size: tag edge length in meters.
        prior_poses: optional ``{tag_id: (rvec, tvec)}`` map of cached
            poses from the previous frame, used as warm starts in
            ``compute_camera_pose`` (useExtrinsicGuess=True path).
        draw: if True, annotate the BGR display frame in place with
            tag outlines and IDs.

    Returns:
        (display_frame_bgr, detections) where ``detections`` is a list
        of dicts (possibly empty). Each dict has:
            tag_id, distance_m, decision_margin, reproj_error_px,
            raw_tag_rvec, raw_tag_tvec,
            rvec_cam, tvec_cam,
            tag_corners, tag_center
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    tags_result = detect_apriltag(gray, detector, detector_type)
    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    detections: List[dict] = []

    def _process_candidate(tag_id_int, tag_corners, center_xy, decision_margin):
        if decision_margin is not None and decision_margin < MIN_DECISION_MARGIN:
            return
        if draw:
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

        prior = prior_poses.get(int(tag_id_int)) if prior_poses else None
        rvec_init, tvec_init = (prior if prior is not None else (None, None))
        success, rvec_cam, tvec_cam, reproj_err = compute_camera_pose(
            tag_corners, camera_matrix, dist_coeffs, tag_size,
            rvec_init=rvec_init, tvec_init=tvec_init,
        )
        if not success:
            return
        raw_tag_rvec, raw_tag_tvec = invert_pose(rvec_cam, tvec_cam)
        distance_m = float(sqrt(sum(raw_tag_tvec.flatten() ** 2)))
        detections.append({
            "tag_id": int(tag_id_int),
            "distance_m": distance_m,
            "decision_margin": float(decision_margin) if decision_margin is not None else float("nan"),
            "reproj_error_px": float(reproj_err),
            "raw_tag_rvec": raw_tag_rvec,
            "raw_tag_tvec": raw_tag_tvec,
            # ``tag_rvec`` / ``tag_tvec`` are kept as aliases for legacy callers
            # that referenced them when ``pose_adjust_fn`` was a thing.
            "tag_rvec": raw_tag_rvec,
            "tag_tvec": raw_tag_tvec,
            "rvec_cam": rvec_cam,
            "tvec_cam": tvec_cam,
            "tag_corners": tag_corners,
            "tag_center": (int(center_xy[0]), int(center_xy[1])),
        })

    if detector_type == "pyapriltags":
        for tag in tags_result:
            if tag.tag_id in TAG_MAP_POSITIONS_CM:
                center = tag.center.astype(int)
                _process_candidate(
                    int(tag.tag_id),
                    tag.corners,
                    (int(center[0]), int(center[1])),
                    float(getattr(tag, "decision_margin", float("nan"))),
                )
    else:
        corners, ids, rejected = tags_result
        if ids is not None and len(ids) > 0:
            for i, tag_id in enumerate(ids.flatten()):
                tag_id_int = int(tag_id)
                if tag_id_int in TAG_MAP_POSITIONS_CM:
                    tag_corners = corners[i]
                    center = tag_corners.mean(axis=0).astype(int)
                    # OpenCV ArUco detector doesn't expose decision_margin;
                    # passing None disables that gate (we still gate on
                    # reprojection error inside compute_camera_pose).
                    _process_candidate(
                        tag_id_int, tag_corners,
                        (int(center[0]), int(center[1])),
                        None,
                    )

    return display_frame, detections


def find_best_tag_in_frame(
    frame_rgb,
    detector,
    detector_type,
    camera_matrix,
    dist_coeffs,
    tag_size,
    pose_adjust_fn=None,
):
    """Back-compat shim: returns the closest accepted detection (or None).

    New code should call ``find_tags_in_frame`` directly so it can fuse
    multiple tags. Note: ``pose_adjust_fn`` is no longer applied — the
    only existing user passed ``None`` and the back-camera flip is now
    handled inside ``transform_to_map``.
    """
    display_frame, detections = find_tags_in_frame(
        frame_rgb, detector, detector_type, camera_matrix, dist_coeffs, tag_size,
    )
    if not detections:
        return display_frame, False, None
    best = min(detections, key=lambda d: d["distance_m"])
    return display_frame, True, best

def _reprojection_error_px(obj_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """Mean per-corner reprojection error in pixels."""
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    img = image_points.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img, axis=1)))


def compute_camera_pose(tag_corners, camera_matrix, dist_coeffs, tag_size,
                        rvec_init=None, tvec_init=None):
    """Compute camera pose relative to a planar AprilTag.

    Uses ``cv2.SOLVEPNP_IPPE_SQUARE`` (Collins-Bartoli closed form for square
    planar markers). IPPE_SQUARE has a hard requirement on the obj_points
    ordering -- it must be the canonical [TL, TR, BR, BL] sequence, NOT
    the [BL, BR, TR, TL] convention ``get_tag_object_points`` returns
    (which matches the AprilTag detector's corner output). Feeding the
    solver the wrong order silently produces wrong poses (the closed
    form's intermediate computations rely on which slot is which corner),
    so we build a separate canonical-ordered (obj, image) pair just for
    the solver call. Reprojection error and LM refinement are then run
    against the original convention -- ``rvec`` / ``tvec`` describe the
    same coordinate transformation either way (the obj origin and axes
    don't depend on point indexing), so downstream code is unchanged.

    The planar 4-point case is genuinely ambiguous: IPPE returns up to 2
    candidates and picking the wrong one is the dominant failure mode
    for square markers (yaw flips, position jumps tens of cm). The
    primary discriminator is reprojection error, but for nearly
    fronto-parallel views both candidates project almost equally well
    and reproj err picks essentially randomly -- that's the source of
    the frame-to-frame flips. So when a previous-frame pose is
    available AND the two candidates' errs are close, we break the tie
    by rotational closeness to the prior (locks the choice in to the
    same branch every frame). A clearly-better err always wins, so the
    prior cannot pin a wrong pose against the data.

    Args:
        tag_corners: 2D image coordinates of tag corners (4 points,
            same [BL, BR, TR, TL] ordering as ``get_tag_object_points``).
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        tag_size: Tag size in meters.
        rvec_init / tvec_init: optional pose from the previous frame, used
            both to disambiguate IPPE's 2-solution ambiguity and as the
            LM refinement starting point.

    Returns:
        (success, rvec, tvec, reproj_error_px). ``success=False`` either
        when the solver fails or when the chosen candidate's reprojection
        error exceeds ``MAX_REPROJ_ERROR_PX``.
    """
    half = float(tag_size) / 2.0
    # Canonical IPPE_SQUARE ordering: [TL, TR, BR, BL].
    ippe_obj_points = np.array([
        [-half,  half, 0.0],   # TL
        [ half,  half, 0.0],   # TR
        [ half, -half, 0.0],   # BR
        [-half, -half, 0.0],   # BL
    ], dtype=np.float32)
    # Detector emits corners in [BL, BR, TR, TL] -- the reverse of
    # IPPE_SQUARE's canonical order, so we just flip the array.
    image_pts_ippe = np.ascontiguousarray(
        np.array(tag_corners, dtype=np.float32).reshape(4, 2)[::-1]
    ).reshape(4, 1, 2)

    # Original-convention pair, used for reprojection error scoring and
    # LM refinement. (The (obj, image) correspondences here are also
    # consistent, just in the opposite order.)
    obj_points = get_tag_object_points(tag_size)
    image_points = np.array(tag_corners, dtype=np.float32).reshape(4, 1, 2)

    try:
        n_solutions, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            ippe_obj_points,
            image_pts_ippe,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        n_solutions = 0
        rvecs, tvecs = (), ()

    if not n_solutions:
        # IPPE_SQUARE can refuse on degenerate inputs; fall back to the
        # iterative solver so we at least get *something*.
        success, rvec, tvec = cv2.solvePnP(
            obj_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return False, None, None, float("nan")
        rvecs, tvecs = (rvec,), (tvec,)
        n_solutions = 1

    # Build candidate list with reprojection errors evaluated in our
    # original convention (the rvec/tvec are valid in both since they
    # describe the same coordinate transformation).
    candidates = []
    for k in range(int(n_solutions)):
        rv = np.array(rvecs[k], dtype=np.float64).copy()
        tv = np.array(tvecs[k], dtype=np.float64).copy()
        err = _reprojection_error_px(
            obj_points, image_points, rv, tv, camera_matrix, dist_coeffs
        )
        if math.isfinite(err):
            candidates.append({"rvec": rv, "tvec": tv, "err": err})

    if not candidates:
        return False, None, None, float("nan")

    candidates.sort(key=lambda c: c["err"])
    if rvec_init is not None and len(candidates) > 1:
        # Reprojection error reliably discriminates the two planar-pose
        # candidates only when one is clearly better -- a tilted tag
        # gives a 2x-5x error ratio. When the tag is nearly
        # fronto-parallel both candidates project almost equally well,
        # so reproj err picks essentially randomly and we get the
        # frame-to-frame flips. In that regime fall back to prior
        # closeness (which is what makes the choice stable). We only
        # do this when err is genuinely ambiguous so the prior can
        # never "lock in" a wrong pose against the data: a clearly
        # better err always wins.
        err_lo = candidates[0]["err"]
        err_hi = candidates[1]["err"]
        if err_hi < 1.5 * max(err_lo, 0.1):
            prior = np.asarray(rvec_init, dtype=np.float64).reshape(3, 1)
            R_prior, _ = cv2.Rodrigues(prior)

            def _angular_dist(rv):
                R, _ = cv2.Rodrigues(rv)
                cos_theta = (np.trace(R_prior.T @ R) - 1.0) * 0.5
                return float(math.acos(max(-1.0, min(1.0, cos_theta))))

            candidates.sort(key=lambda c: _angular_dist(c["rvec"]))

    chosen = candidates[0]
    best_rvec = chosen["rvec"]
    best_tvec = chosen["tvec"]
    best_err = chosen["err"]

    # LM refine starting from the chosen IPPE candidate (NOT from the
    # prior -- the prior may be stale and nudge LM into the wrong basin).
    rvec_refine = best_rvec.copy()
    tvec_refine = best_tvec.copy()
    try:
        cv2.solvePnPRefineLM(
            obj_points, image_points, camera_matrix, dist_coeffs,
            rvec_refine, tvec_refine,
        )
        refined_err = _reprojection_error_px(
            obj_points, image_points, rvec_refine, tvec_refine,
            camera_matrix, dist_coeffs,
        )
        if math.isfinite(refined_err) and refined_err <= best_err + 1e-3:
            best_rvec = rvec_refine
            best_tvec = tvec_refine
            best_err = refined_err
    except cv2.error:
        pass

    if best_err > MAX_REPROJ_ERROR_PX:
        return False, best_rvec, best_tvec, best_err
    return True, best_rvec, best_tvec, best_err

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

def compute_map_pose_from_detection(chosen, chosen_cam, tag_anchor_xy_cm=None):
    """Compute car pose in MAP coordinates from a chosen detection.

    The ``tag_anchor_xy_cm`` argument is kept for backward compatibility
    but is no longer used: the per-tag config table provides the anchor.

    Returns (x_cm, y_cm, yaw_deg).
    """
    is_back = (chosen_cam == "back")
    camera_offset = BACK_CAMERA_OFFSET_CM if is_back else FRONT_CAMERA_OFFSET_CM
    return transform_to_map(
        chosen["raw_tag_tvec"],
        chosen["raw_tag_rvec"],
        chosen["tag_id"],
        camera_offset_cm=camera_offset,
        is_back_camera=is_back,
    )

def format_map_pose_info(x_cm, y_cm, yaw_deg, tag_id=None, camera=None, distance_cm=None):
    """
    Format map pose information as a dictionary for display/logging.
    """
    out = {"x_cm": float(x_cm), "y_cm": float(y_cm), "yaw_deg": float(yaw_deg)}
    if tag_id is not None:
        out["tag_id"] = int(tag_id)
    if camera is not None:
        out["camera"] = str(camera)
    if distance_cm is not None:
        out["distance_cm"] = float(distance_cm)
    return out

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

    # Prefer MAP pose display (single source of truth for user-facing pose).
    if isinstance(pose_data, dict) and "x_cm" in pose_data and "y_cm" in pose_data and "yaw_deg" in pose_data:
        y_offset += draw_text_with_background(
            img, "=== MAP POSE (Car) ===", (start_x, y_offset),
            font_scale=1.0, thickness=2, text_color=(0, 255, 255), bg_color=(0, 0, 0), alpha=0.8
        )
        y_offset += line_spacing

        if "camera" in pose_data or "tag_id" in pose_data:
            cam = pose_data.get("camera", "?")
            tid = pose_data.get("tag_id", "?")
            y_offset += draw_text_with_background(
                img, f"Cam: {cam} | Tag: {tid}", (start_x, y_offset),
                font_scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
            )
            y_offset += line_spacing

        y_offset += draw_text_with_background(
            img, "POSITION (cm):", (start_x, y_offset),
            font_scale=font_scale, thickness=thickness, text_color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.7
        )
        y_offset += line_spacing

        y_offset += draw_text_with_background(
            img, f"  X: {pose_data['x_cm']:>7.2f} cm (East+)", (start_x, y_offset),
            font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
        )
        y_offset += line_spacing

        y_offset += draw_text_with_background(
            img, f"  Y: {pose_data['y_cm']:>7.2f} cm (North+)", (start_x, y_offset),
            font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
        )
        y_offset += line_spacing * 2

        y_offset += draw_text_with_background(
            img, "ORIENTATION (deg):", (start_x, y_offset),
            font_scale=font_scale, thickness=thickness, text_color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.7
        )
        y_offset += line_spacing

        y_offset += draw_text_with_background(
            img, f"  Yaw: {pose_data['yaw_deg']:>7.2f}° (0°=North, CW+)", (start_x, y_offset),
            font_scale=font_scale, thickness=thickness, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7
        )

        if "distance_cm" in pose_data:
            y_offset += line_spacing
            y_offset += draw_text_with_background(
                img, f"  Distance: {pose_data['distance_cm']:>6.2f} cm", (start_x, y_offset),
                font_scale=font_scale, thickness=thickness, text_color=(0, 255, 255), bg_color=(0, 0, 0), alpha=0.7
            )
        return
    
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

def normalize_angle_deg_0_360(angle_deg):
    """Normalize an angle to the range [0, 360)."""
    return angle_deg % 360.0


def transform_to_map(tvec, rvec, tag_id, *, camera_offset_cm, is_back_camera):
    """Convert a tag-frame pose into the car's pose in map coordinates.

    Replaces the previous per-tag-ID if/elif blocks (one per camera) with
    a single config-driven formula:

        delta = R_tag_to_map @ [x_tag, z_tag]
        yaw_map_deg = -pitch_deg + cfg.yaw_offset_deg + (180 if back else 0)
        offset_vec = sign * camera_offset_cm * body_forward(yaw_map_rad)
        (x_map, y_map) = anchor + delta + offset_vec

    where
      - sign = -1 for the front camera (lens is ahead of the reference
        point, so move backward to land on it)
      - sign = +1 for the back camera (lens is behind the reference
        point, so move forward to land on it)
      - body_forward(yaw_map_rad) = (-sin yaw, cos yaw), matching the
        existing yaw_map convention (0 = north).

    Args:
        tvec: Translation vector (tag position relative to camera) in meters.
        rvec: Rotation vector (tag rotation relative to camera).
        tag_id: Tag ID (0..5).
        camera_offset_cm: distance between camera lens and the reported
            reference point along body forward. Use FRONT_CAMERA_OFFSET_CM
            or BACK_CAMERA_OFFSET_CM.
        is_back_camera: True if the pose came from the back camera.

    Returns:
        (x_map, y_map, yaw_map_deg) — yaw normalized to [0, 360).
    """
    cfg = TAG_CONFIGS.get(int(tag_id))
    if cfg is None:
        cfg = TAG_CONFIGS[0]   # benign fallback; should never trigger

    x_tag_cm, _, z_tag_cm = (np.asarray(tvec).flatten() * 100.0)

    R_tag, _ = cv2.Rodrigues(rvec)
    # "pitch" here is the rotation of the tag around the camera's Y
    # axis. Because the tag is mounted vertically on a wall, this is
    # exactly the relative yaw between camera and tag (the variable
    # name is preserved for git-blame friendliness).
    pitch_rad = atan2(-R_tag[2, 0], -R_tag[2, 2])
    pitch_deg = degrees(pitch_rad)

    delta = cfg.R_tag_to_map @ np.array([x_tag_cm, z_tag_cm])

    yaw_map_deg = -pitch_deg + cfg.yaw_offset_deg + (180.0 if is_back_camera else 0.0)
    yaw_map_rad = math.radians(yaw_map_deg)
    bf_x = -math.sin(yaw_map_rad)
    bf_y = math.cos(yaw_map_rad)

    sign = +1.0 if is_back_camera else -1.0
    off_x = sign * camera_offset_cm * bf_x
    off_y = sign * camera_offset_cm * bf_y

    anchor_x, anchor_y = cfg.anchor_xy_cm
    x_map = anchor_x + delta[0] + off_x
    y_map = anchor_y + delta[1] + off_y

    return float(x_map), float(y_map), normalize_angle_deg_0_360(yaw_map_deg)


# Back-compat shims for any external/legacy callers. New code should call
# transform_to_map directly.
def transform_to_map_coordinates(tvec, rvec, tag_id, tag_map_xy_cm=(100, 200)):
    return transform_to_map(
        tvec, rvec, tag_id,
        camera_offset_cm=FRONT_CAMERA_OFFSET_CM,
        is_back_camera=False,
    )


def transform_to_map_coordinates_back_camera(tvec, rvec, tag_id, tag_map_xy_cm=(100, 200)):
    return transform_to_map(
        tvec, rvec, tag_id,
        camera_offset_cm=BACK_CAMERA_OFFSET_CM,
        is_back_camera=True,
    )

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

def _detection_to_map_candidate(detection, camera_name):
    """Map a single tag detection to a (x_cm, y_cm, yaw_deg, weight, ...) record."""
    is_back = (camera_name == "back")
    camera_offset = BACK_CAMERA_OFFSET_CM if is_back else FRONT_CAMERA_OFFSET_CM
    x_cm, y_cm, yaw_deg = transform_to_map(
        detection["raw_tag_tvec"],
        detection["raw_tag_rvec"],
        detection["tag_id"],
        camera_offset_cm=camera_offset,
        is_back_camera=is_back,
    )
    distance_cm = float(detection["distance_m"] * 100.0)
    margin = float(detection.get("decision_margin", 0.0))
    if not math.isfinite(margin):
        margin = MIN_DECISION_MARGIN  # OpenCV detector path: fall back to neutral weight
    # Inverse-variance weight: distant tags and low-margin tags contribute less.
    weight = max(margin, 1.0) / (distance_cm * distance_cm + 1.0)
    return {
        "tag_id": int(detection["tag_id"]),
        "camera": camera_name,
        "x_cm": x_cm,
        "y_cm": y_cm,
        "yaw_deg": yaw_deg,
        "distance_cm": distance_cm,
        "weight": float(weight),
        "decision_margin": margin,
        "reproj_error_px": float(detection.get("reproj_error_px", float("nan"))),
    }


def _fuse_candidates(candidates):
    """Inverse-variance fuse a list of map candidates into one pose.

    Position is a weighted average. Yaw is fused as a circular mean
    (atan2 of weighted sums of sin/cos) to handle wrap-around. Any
    candidate whose yaw differs from the highest-weight candidate by
    more than ``FUSION_YAW_OUTLIER_DEG`` is dropped — that's the
    planar-pose ambiguity flip failure mode.

    Returns a dict with the fused pose, or None if ``candidates`` is empty.
    """
    if not candidates:
        return None
    # Sort by descending weight so candidates[0] is the "anchor" for yaw outlier checks.
    candidates = sorted(candidates, key=lambda c: c["weight"], reverse=True)
    anchor_yaw_rad = math.radians(candidates[0]["yaw_deg"])
    inliers = []
    for c in candidates:
        d = abs(_wrap_pi(math.radians(c["yaw_deg"]) - anchor_yaw_rad))
        if math.degrees(d) <= FUSION_YAW_OUTLIER_DEG:
            inliers.append(c)
    if not inliers:
        inliers = candidates[:1]

    w_sum = sum(c["weight"] for c in inliers)
    if w_sum <= 0:
        # Degenerate: fall back to the single best-weight candidate.
        c = inliers[0]
        return {
            "x_cm": c["x_cm"], "y_cm": c["y_cm"], "yaw_deg": c["yaw_deg"],
            "tag_id": c["tag_id"], "camera": c["camera"],
            "distance_cm": c["distance_cm"], "n_tags": 1, "fused": False,
            "reproj_error_px": c["reproj_error_px"],
        }

    x = sum(c["weight"] * c["x_cm"] for c in inliers) / w_sum
    y = sum(c["weight"] * c["y_cm"] for c in inliers) / w_sum
    sin_acc = sum(c["weight"] * math.sin(math.radians(c["yaw_deg"])) for c in inliers)
    cos_acc = sum(c["weight"] * math.cos(math.radians(c["yaw_deg"])) for c in inliers)
    yaw_deg = normalize_angle_deg_0_360(math.degrees(math.atan2(sin_acc, cos_acc)))

    primary = inliers[0]
    return {
        "x_cm": float(x),
        "y_cm": float(y),
        "yaw_deg": float(yaw_deg),
        "tag_id": int(primary["tag_id"]),
        "camera": primary["camera"],
        "distance_cm": float(primary["distance_cm"]),
        "n_tags": int(len(inliers)),
        "fused": bool(len(inliers) > 1),
        "reproj_error_px": float(primary["reproj_error_px"]),
    }


def _process_two_cameras(frame_front, frame_back, *, detector, detector_type,
                         camera_matrix, dist_coeffs, prior_front=None, prior_back=None,
                         draw=True):
    """Detect + score tags from both cameras and return their map candidates.

    Returns ``(detections_front, detections_back, candidates)``. The
    ``detections_*`` lists are the per-camera raw detections (so callers
    can update warm-start caches); ``candidates`` is the fused-ready
    list of map-pose dicts from both cameras combined.
    """
    _, dets_front = find_tags_in_frame(
        frame_front, detector, detector_type, camera_matrix, dist_coeffs,
        TAG_SIZE, prior_poses=prior_front, draw=draw,
    )
    _, dets_back = find_tags_in_frame(
        frame_back, detector, detector_type, camera_matrix, dist_coeffs,
        TAG_SIZE, prior_poses=prior_back, draw=draw,
    )
    candidates = []
    for d in dets_front:
        candidates.append(_detection_to_map_candidate(d, "front"))
    for d in dets_back:
        candidates.append(_detection_to_map_candidate(d, "back"))
    return dets_front, dets_back, candidates


def get_latest_map_pose(timeout_s=5.0, target_fps=10.0, tag_size_m=None):
    """Grab frames until at least one accepted AprilTag is detected (or timeout),
    fuse all visible tags from both cameras, and return the car pose in
    map coordinates.

    Returns a dict (same shape as ``AprilTagMapPoseTracker.update()``)
    or ``None`` on timeout.
    """
    global TAG_SIZE
    if tag_size_m is not None:
        TAG_SIZE = float(tag_size_m)

    camera_matrix, dist_coeffs = load_calibration()
    if camera_matrix is None:
        raise RuntimeError("Camera calibration not found. Run camera_calibration.py first.")

    detector, detector_type = make_apriltag_detector()

    picam2_front = Picamera2(FRONT_CAMERA_INDEX)
    picam2_back = Picamera2(BACK_CAMERA_INDEX)
    for cam in (picam2_front, picam2_back):
        cam.configure(cam.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        ))
        cam.start()

    frame_period_s = 1.0 / float(target_fps) if target_fps else 0.0
    deadline = time.perf_counter() + float(timeout_s)
    next_frame_t = time.perf_counter()

    try:
        while time.perf_counter() < deadline:
            frame_front = picam2_front.capture_array()
            frame_back = picam2_back.capture_array()

            _, _, candidates = _process_two_cameras(
                frame_front, frame_back,
                detector=detector, detector_type=detector_type,
                camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
                draw=False,
            )
            fused = _fuse_candidates(candidates)
            if fused is not None:
                return fused

            if frame_period_s > 0:
                next_frame_t += frame_period_s
                sleep_s = next_frame_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_frame_t = time.perf_counter()

        return None
    finally:
        for cam in (picam2_front, picam2_back):
            try:
                cam.stop()
            except Exception:
                pass


class AprilTagMapPoseTracker:
    """Persistent AprilTag pose tracker (keeps cameras + detector open).

    Call ``update()`` in a loop to get the latest map pose. The tracker
    fuses every visible tag from both cameras and returns the fused pose
    directly (no temporal smoothing is applied -- the caller is responsible
    for any filtering they need).

    A small per-(camera, tag_id) pose cache is maintained to warm-start
    the LM refinement step in ``compute_camera_pose``.
    """

    def __init__(self, tag_size_m=None):
        global TAG_SIZE
        if tag_size_m is not None:
            TAG_SIZE = float(tag_size_m)

        self.camera_matrix, self.dist_coeffs = load_calibration()
        if self.camera_matrix is None:
            raise RuntimeError("Camera calibration not found. Run camera_calibration.py first.")

        self.detector, self.detector_type = make_apriltag_detector()

        self.picam2_front = Picamera2(FRONT_CAMERA_INDEX)
        self.picam2_back = Picamera2(BACK_CAMERA_INDEX)
        for cam in (self.picam2_front, self.picam2_back):
            cam.configure(cam.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            ))
            cam.start()

        # Warm-start cache: {tag_id: (rvec_cam, tvec_cam)} per camera.
        self._prior_front = {}
        self._prior_back = {}

    def close(self):
        for cam in (self.picam2_front, self.picam2_back):
            try:
                cam.stop()
            except Exception:
                pass

    def _update_priors(self, dets_front, dets_back):
        for d in dets_front:
            self._prior_front[int(d["tag_id"])] = (d["rvec_cam"], d["tvec_cam"])
        for d in dets_back:
            self._prior_back[int(d["tag_id"])] = (d["rvec_cam"], d["tvec_cam"])

    def update(self):
        """Capture both cameras, fuse all visible tags, and return.

        Return shape:
            {
              "x_cm": float, "y_cm": float, "yaw_deg": float,
              "tag_id": int (the highest-weight tag),
              "camera": "front" | "back" (highest-weight tag's camera),
              "distance_cm": float (highest-weight tag's distance),
              "n_tags": int (1 if single-tag, >1 if fused),
              "fused": bool,
              "reproj_error_px": float,
            }
        Returns ``None`` if no tag was accepted in this frame.
        """
        frame_front = self.picam2_front.capture_array()
        frame_back = self.picam2_back.capture_array()

        dets_front, dets_back, candidates = _process_two_cameras(
            frame_front, frame_back,
            detector=self.detector, detector_type=self.detector_type,
            camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs,
            prior_front=self._prior_front, prior_back=self._prior_back,
            draw=False,
        )
        # Refresh warm-start cache regardless of fusion outcome — even a
        # detection that got dropped by the yaw outlier filter is still a
        # good prior for next frame's PnP refine.
        self._update_priors(dets_front, dets_back)

        return _fuse_candidates(candidates)

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
    
    # Initialize AprilTag detector via the shared factory (tuned threads,
    # refine_edges, decode_sharpening) and report which backend won.
    try:
        detector, detector_type = make_apriltag_detector()
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("Please install pyapriltags (pip install pyapriltags)")
        print("or ensure OpenCV >= 4.7.0 with contrib modules.")
        return
    print(f"\n{detector_type} detector initialized (family: {TAG_FAMILY})")

    # Initialize cameras
    print("Initializing cameras...")
    picam2_front = Picamera2(FRONT_CAMERA_INDEX)
    picam2_back = Picamera2(BACK_CAMERA_INDEX)

    for cam in (picam2_front, picam2_back):
        cam.configure(cam.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        ))
        cam.start()
    
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
                pose_adjust_fn=None,
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
                "BACK CAMERA",
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

                # Transform to map coordinates using this tag's known anchor point
                tag_anchor_xy = TAG_MAP_POSITIONS_CM.get(chosen["tag_id"], (100, 200))
                x_map, y_map, yaw_map = compute_map_pose_from_detection(chosen, chosen_cam, tag_anchor_xy)

                # Display MAP pose info on image (x, y, yaw in map coordinates)
                pose_data = format_map_pose_info(
                    x_map,
                    y_map,
                    yaw_map,
                    tag_id=chosen["tag_id"],
                    camera=chosen_cam,
                    distance_cm=chosen["distance_m"] * 100.0,
                )
                draw_pose_info(display_frame, pose_data)

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
