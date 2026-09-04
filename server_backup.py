import base64
import io
import os
import tempfile
import time
import math
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportImage,
    PageBreak,
)


# ============================================================
# APP CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

MAX_DIMENSION = 2000
MAX_FEATURES = 10000


# ============================================================
# BASIC UTILITIES
# ============================================================

def allowed_file(filename):
    if not filename:
        return False

    extension = os.path.splitext(filename)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def encode_jpeg(image, quality=92):
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not ok:
        return ""

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image.")

    height, width = image.shape[:2]

    largest_dimension = max(height, width)

    if largest_dimension > MAX_DIMENSION:
        scale = MAX_DIMENSION / float(largest_dimension)

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Primary preprocessing.

    CLAHE improves local contrast while Gaussian smoothing
    suppresses small sensor/compression noise.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return enhanced


def alternate_preprocess(image):
    """
    Stronger illumination-normalized representation.
    """

    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(12, 12)
    )

    enhanced = clahe.apply(image)

    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.2
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.35,
        blur,
        -0.35,
        0
    )

    return sharpened


def gradient_preprocess(image):
    """
    Gradient representation.

    Useful when absolute brightness differs substantially
    between two lunar observations.
    """

    image_float = image.astype(np.float32)

    gx = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(gx, gy)

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(np.uint8)


def local_contrast_preprocess(image):
    """
    Local contrast representation.

    Helps preserve terrain boundaries when illumination changes.
    """

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        5
    )

    local = cv2.subtract(
        image,
        blurred
    )

    local = cv2.normalize(
        local,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return local.astype(np.uint8)


# ============================================================
# IMAGE QUALITY
# ============================================================

def image_quality(image):
    height, width = image.shape[:2]

    area = height * width

    resolution_score = min(
        100.0,
        (area / 1_000_000.0) * 100.0
    )

    contrast = float(np.std(image))

    contrast_score = min(
        100.0,
        contrast * 2.2
    )

    laplacian_variance = float(
        cv2.Laplacian(
            image,
            cv2.CV_64F
        ).var()
    )

    sharpness_score = min(
        100.0,
        laplacian_variance / 12.0
    )

    overall = (
        resolution_score * 0.25
        + contrast_score * 0.35
        + sharpness_score * 0.40
    )

    if overall >= 75:
        label = "Excellent"
    elif overall >= 55:
        label = "Good"
    elif overall >= 35:
        label = "Fair"
    else:
        label = "Limited"

    return {
        "resolution_score": round(resolution_score, 2),
        "contrast_score": round(contrast_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "overall_score": round(overall, 2),
        "label": label,
        "width": width,
        "height": height,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def create_sift():
    return cv2.SIFT_create(
        nfeatures=MAX_FEATURES,
        contrastThreshold=0.012,
        edgeThreshold=12,
        sigma=1.6
    )


def extract_sift(image):
    sift = create_sift()

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    if keypoints is None or descriptors is None:
        return [], None

    return keypoints, descriptors


def deduplicate_features(
    keypoints,
    descriptors,
    max_features=MAX_FEATURES
):
    """
    Removes duplicate/near-duplicate feature locations created
    by multiple image representations.

    Stronger features are retained.
    """

    if not keypoints or descriptors is None:
        return [], None

    candidates = []

    for index, kp in enumerate(keypoints):
        candidates.append(
            (
                float(kp.response),
                index,
                float(kp.pt[0]),
                float(kp.pt[1]),
                float(kp.size)
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []

    # Spatial suppression radius.
    # This prevents the same physical terrain point from
    # appearing repeatedly because of multiple preprocessing views.
    suppression_radius = 2.5

    radius_sq = suppression_radius ** 2

    for response, index, x, y, size in candidates:

        too_close = False

        for selected_index in selected:
            selected_kp = keypoints[selected_index]

            dx = x - selected_kp.pt[0]
            dy = y - selected_kp.pt[1]

            if (dx * dx + dy * dy) <= radius_sq:
                too_close = True
                break

        if not too_close:
            selected.append(index)

        if len(selected) >= max_features:
            break

    selected_keypoints = [
        keypoints[index]
        for index in selected
    ]

    selected_descriptors = descriptors[
        np.array(selected)
    ]

    return selected_keypoints, selected_descriptors


def extract_multiview_sift(image):
    """
    Extract SIFT features from several representations:

    1. Original
    2. CLAHE
    3. Alternate CLAHE/sharpened
    4. Gradient
    5. Local contrast

    Features are then spatially deduplicated.
    """

    representations = [
        image,
        preprocess_image(image),
        alternate_preprocess(image),
        gradient_preprocess(image),
        local_contrast_preprocess(image),
    ]

    all_keypoints = []
    all_descriptors = []

    for representation in representations:

        keypoints, descriptors = extract_sift(
            representation
        )

        if (
            keypoints
            and descriptors is not None
            and len(descriptors) > 0
        ):
            all_keypoints.extend(keypoints)
            all_descriptors.append(descriptors)

    if not all_descriptors:
        return [], None

    descriptors = np.vstack(
        all_descriptors
    )

    keypoints, descriptors = deduplicate_features(
        all_keypoints,
        descriptors,
        MAX_FEATURES
    )

    return keypoints, descriptors


# ============================================================
# FEATURE MATCHING
# ============================================================

def ratio_match(
    des1,
    des2,
    ratio=0.76
):
    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    index_params = {
        "algorithm": 1,
        "trees": 8
    }

    search_params = {
        "checks": 120
    }

    matcher = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    try:
        raw = matcher.knnMatch(
            des1,
            des2,
            k=2
        )
    except cv2.error:
        return []

    good = []

    for pair in raw:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def reciprocal_matches(
    des1,
    des2,
    forward_matches,
    ratio=0.78
):
    if not forward_matches:
        return []

    reverse = ratio_match(
        des2,
        des1,
        ratio
    )

    reverse_pairs = {
        (m.queryIdx, m.trainIdx)
        for m in reverse
    }

    reciprocal = []

    for match in forward_matches:

        reverse_key = (
            match.trainIdx,
            match.queryIdx
        )

        if reverse_key in reverse_pairs:
            reciprocal.append(match)

    return reciprocal


def merge_unique_matches(*match_sets):
    """
    Merge match sets while keeping only the strongest match
    for each query/train pair.
    """

    best = {}

    for matches in match_sets:

        for match in matches:

            key = (
                int(match.queryIdx),
                int(match.trainIdx)
            )

            distance = float(match.distance)

            if (
                key not in best
                or distance < best[key].distance
            ):
                best[key] = match

    merged = list(best.values())

    merged.sort(
        key=lambda match: match.distance
    )

    return merged


# ============================================================
# MATCH QUALITY
# ============================================================

def descriptor_quality(matches):
    if not matches:
        return 0.0

    distances = np.array(
        [float(m.distance) for m in matches],
        dtype=np.float32
    )

    median_distance = float(
        np.median(distances)
    )

    quality = 100.0 * (
        1.0 - min(
            median_distance / 300.0,
            1.0
        )
    )

    return float(
        max(
            0.0,
            min(
                100.0,
                quality
            )
        )
    )


def median_descriptor_distance(matches):
    if not matches:
        return None

    distances = [
        float(match.distance)
        for match in matches
    ]

    return float(
        np.median(distances)
    )


# ============================================================
# GEOMETRIC VERIFICATION
# ============================================================

def adaptive_ransac_threshold(
    image_a,
    image_b
):
    diagonal_a = math.sqrt(
        image_a.shape[0] ** 2
        + image_a.shape[1] ** 2
    )

    diagonal_b = math.sqrt(
        image_b.shape[0] ** 2
        + image_b.shape[1] ** 2
    )

    diagonal = min(
        diagonal_a,
        diagonal_b
    )

    threshold = diagonal * 0.004

    return float(
        np.clip(
            threshold,
            3.0,
            8.0
        )
    )


def verify_homography(
    kp1,
    kp2,
    matches,
    ransac_threshold
):
    if len(matches) < 4:
        return {
            "inliers": 0,
            "homography": None,
            "mask": None,
            "consistency": 0.0,
            "reprojection_error": None,
        }

    src = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    try:
        H, mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            ransac_threshold,
            maxIters=15000,
            confidence=0.995
        )
    except cv2.error:
        return {
            "inliers": 0,
            "homography": None,
            "mask": None,
            "consistency": 0.0,
            "reprojection_error": None,
        }

    if H is None or mask is None:
        return {
            "inliers": 0,
            "homography": None,
            "mask": None,
            "consistency": 0.0,
            "reprojection_error": None,
        }

    mask = mask.ravel().astype(bool)

    inlier_count = int(
        np.sum(mask)
    )

    consistency = (
        100.0
        * inlier_count
        / max(len(matches), 1)
    )

    reprojection_error = None

    if inlier_count >= 4:

        projected = cv2.perspectiveTransform(
            src,
            H
        ).reshape(-1, 2)

        destination = dst.reshape(-1, 2)

        errors = np.linalg.norm(
            projected - destination,
            axis=1
        )

        inlier_errors = errors[mask]

        if len(inlier_errors) > 0:
            reprojection_error = float(
                np.median(inlier_errors)
            )

    return {
        "inliers": inlier_count,
        "homography": H,
        "mask": mask,
        "consistency": float(
            min(
                100.0,
                consistency
            )
        ),
        "reprojection_error": reprojection_error,
    }


def verify_affine(
    kp1,
    kp2,
    matches,
    ransac_threshold
):
    if len(matches) < 3:
        return {
            "inliers": 0,
            "matrix": None,
            "mask": None,
            "consistency": 0.0,
            "reprojection_error": None,
        }

    src = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    dst = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    try:
        matrix, mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=15000,
            confidence=0.995,
            refineIters=30
        )
    except cv2.error:
        return {
            "inliers": 0,
            "matrix": None,
            "mask": None,
            "consistency": 0.0,
            "reprojection_error": None,
        }

    if matrix is None or mask is None:
        return {
            "inliers": 0,
            "matrix": None,
            "mask": None,
            "consistency": 0.0,
            "reprojection_error": None,
        }

    mask = mask.ravel().astype(bool)

    inlier_count = int(
        np.sum(mask)
    )

    consistency = (
        100.0
        * inlier_count
        / max(len(matches), 1)
    )

    reprojection_error = None

    if inlier_count >= 3:

        homogeneous = np.hstack([
            src,
            np.ones(
                (len(src), 1),
                dtype=np.float32
            )
        ])

        projected = (
            homogeneous @ matrix.T
        )

        errors = np.linalg.norm(
            projected - dst,
            axis=1
        )

        inlier_errors = errors[mask]

        if len(inlier_errors) > 0:
            reprojection_error = float(
                np.median(inlier_errors)
            )

    return {
        "inliers": inlier_count,
        "matrix": matrix,
        "mask": mask,
        "consistency": float(
            min(
                100.0,
                consistency
            )
        ),
        "reprojection_error": reprojection_error,
    }


def choose_geometry(
    kp1,
    kp2,
    matches,
    ransac_threshold
):
    homography = verify_homography(
        kp1,
        kp2,
        matches,
        ransac_threshold
    )

    affine = verify_affine(
        kp1,
        kp2,
        matches,
        ransac_threshold
    )

    if homography["inliers"] >= affine["inliers"]:
        return {
            "model": "Homography",
            "inliers": homography["inliers"],
            "mask": homography["mask"],
            "matrix": homography["homography"],
            "consistency": homography["consistency"],
            "reprojection_error": homography["reprojection_error"],
            "homography_verified": homography["inliers"],
            "affine_verified": affine["inliers"],
        }

    return {
        "model": "Affine",
        "inliers": affine["inliers"],
        "mask": affine["mask"],
        "matrix": affine["matrix"],
        "consistency": affine["consistency"],
        "reprojection_error": affine["reprojection_error"],
        "homography_verified": homography["inliers"],
        "affine_verified": affine["inliers"],
    }


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def spatial_coverage(
    keypoints,
    matches,
    mask,
    image_shape,
    grid_size=4
):
    if not matches or mask is None:
        return 0.0

    valid = []

    for index, match in enumerate(matches):

        if index >= len(mask):
            break

        if not mask[index]:
            continue

        x, y = keypoints[
            match.queryIdx
        ].pt

        valid.append(
            (x, y)
        )

    if not valid:
        return 0.0

    height, width = image_shape[:2]

    occupied = set()

    for x, y in valid:

        col = int(
            np.clip(
                x / max(width, 1)
                * grid_size,
                0,
                grid_size - 1
            )
        )

        row = int(
            np.clip(
                y / max(height, 1)
                * grid_size,
                0,
                grid_size - 1
            )
        )

        occupied.add(
            (row, col)
        )

    total_cells = grid_size * grid_size

    return float(
        100.0
        * len(occupied)
        / total_cells
    )


# ============================================================
# SCALE / ROTATION CONSISTENCY
# ============================================================

def scale_rotation_consistency(
    kp1,
    kp2,
    matches,
    mask
):
    if not matches or mask is None:
        return {
            "score": 0.0,
            "scale_ratio": None,
            "rotation_difference": None,
        }

    scales = []
    rotations = []

    for index, match in enumerate(matches):

        if index >= len(mask):
            break

        if not mask[index]:
            continue

        first = kp1[match.queryIdx]
        second = kp2[match.trainIdx]

        if first.size <= 0:
            continue

        scale_ratio = (
            float(second.size)
            / float(first.size)
        )

        scales.append(
            scale_ratio
        )

        angle_difference = (
            second.angle
            - first.angle
        )

        while angle_difference > 180:
            angle_difference -= 360

        while angle_difference < -180:
            angle_difference += 360

        rotations.append(
            abs(angle_difference)
        )

    if not scales:
        return {
            "score": 0.0,
            "scale_ratio": None,
            "rotation_difference": None,
        }

    median_scale = float(
        np.median(scales)
    )

    median_rotation = float(
        np.median(rotations)
    )

    log_scale = abs(
        math.log(
            max(median_scale, 1e-6)
        )
    )

    scale_score = max(
        0.0,
        100.0
        * math.exp(
            -log_scale * 2.5
        )
    )

    rotation_score = max(
        0.0,
        100.0
        * math.exp(
            -median_rotation / 35.0
        )
    )

    score = (
        scale_score * 0.55
        + rotation_score * 0.45
    )

    return {
        "score": float(
            min(
                100.0,
                score
            )
        ),
        "scale_ratio": round(
            median_scale,
            4
        ),
        "rotation_difference": round(
            median_rotation,
            2
        ),
    }


# ============================================================
# CORRESPONDENCE VISUALIZATION
# ============================================================

def draw_correspondence_map(
    image_a,
    image_b,
    kp1,
    kp2,
    matches,
    mask
):
    if (
        not matches
        or mask is None
        or len(mask) != len(matches)
    ):
        combined = cv2.hconcat([
            image_a,
            image_b
        ])

        cv2.putText(
            combined,
            "NO VERIFIED CORRESPONDENCES",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            255,
            2,
            cv2.LINE_AA
        )

        return combined

    verified = [
        match
        for index, match in enumerate(matches)
        if mask[index]
    ]

    verified.sort(
        key=lambda match: match.distance
    )

    verified = verified[:200]

    image_a_color = cv2.cvtColor(
        image_a,
        cv2.COLOR_GRAY2BGR
    )

    image_b_color = cv2.cvtColor(
        image_b,
        cv2.COLOR_GRAY2BGR
    )

    visualization = cv2.drawMatches(
        image_a_color,
        kp1,
        image_b_color,
        kp2,
        verified,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 255, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return visualization


def create_side_by_side(
    image_a,
    image_b
):
    height = min(
        image_a.shape[0],
        image_b.shape[0]
    )

    def resize_height(image):
        if image.shape[0] == height:
            return image

        scale = (
            height
            / image.shape[0]
        )

        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    a = resize_height(image_a)
    b = resize_height(image_b)

    return cv2.hconcat([
        a,
        b
    ])


# ============================================================
# EVIDENCE FUSION
# ============================================================

def calculate_correspondence_score(
    verified,
    candidate,
    feature_a,
    feature_b,
    geometry_score,
    descriptor_score,
    spatial_score,
    quality_score,
    scale_rotation_score,
    reprojection_error
):
    minimum_features = max(
        1,
        min(
            feature_a,
            feature_b
        )
    )

    coverage = min(
        100.0,
        verified
        / minimum_features
        * 100.0
    )

    absolute_strength = (
        100.0
        * (
            1.0
            - math.exp(
                -verified / 42.0
            )
        )
    )

    verification_precision = (
        verified
        / max(candidate, 1)
        * 100.0
    )

    verification_precision = min(
        100.0,
        verification_precision
    )

    if reprojection_error is None:
        reprojection_score = 0.0
    else:
        reprojection_score = (
            100.0
            * math.exp(
                -reprojection_error / 4.0
            )
        )

    reprojection_score = min(
        100.0,
        max(
            0.0,
            reprojection_score
        )
    )

    score = (
        coverage * 0.13
        + absolute_strength * 0.20
        + geometry_score * 0.22
        + descriptor_score * 0.11
        + spatial_score * 0.10
        + verification_precision * 0.08
        + quality_score * 0.06
        + scale_rotation_score * 0.06
        + reprojection_score * 0.04
    )

    return float(
        max(
            0.0,
            min(
                100.0,
                score
            )
        )
    )


def determine_confidence(
    verified,
    geometry,
    spatial,
    descriptor,
    score,
    scale_rotation
):
    if (
        verified >= 50
        and geometry >= 60
        and spatial >= 35
        and descriptor >= 55
        and scale_rotation >= 45
        and score >= 60
    ):
        return "High"

    if (
        verified >= 28
        and geometry >= 42
        and spatial >= 25
        and descriptor >= 40
        and score >= 34
    ):
        return "Medium"

    return "Low"


def determine_match(
    verified,
    geometry,
    spatial,
    score,
    candidate
):
    """
    Conservative decision boundary.

    This intentionally avoids declaring a match from a handful
    of coincidental SIFT points.
    """

    if (
        verified >= 15
        and geometry >= 28
        and spatial >= 12
        and score >= 20
        and candidate >= 20
    ):
        return True

    return False


# ============================================================
# MAIN MATCHING ENGINE
# ============================================================

def run_matching(
    path1,
    path2
):
    start_time = time.perf_counter()

    image_a = prepare_image(path1)
    image_b = prepare_image(path2)

    quality_a = image_quality(
        image_a
    )

    quality_b = image_quality(
        image_b
    )

    average_quality = (
        quality_a["overall_score"]
        + quality_b["overall_score"]
    ) / 2.0

    kp1, des1 = extract_multiview_sift(
        image_a
    )

    kp2, des2 = extract_multiview_sift(
        image_b
    )

    feature_count_a = len(kp1)
    feature_count_b = len(kp2)

    if (
        des1 is None
        or des2 is None
        or feature_count_a < 8
        or feature_count_b < 8
    ):
        return {
            "match_found": False,
            "match_percentage": 0.0,
            "corresponding_features": 0,
            "raw_matches": 0,
            "candidate_matches": 0,
            "verified_matches": 0,
            "feature_count_a": feature_count_a,
            "feature_count_b": feature_count_b,
            "inlier_ratio": 0.0,
            "geometric_consistency": 0.0,
            "descriptor_quality": 0.0,
            "spatial_coverage": 0.0,
            "scale_rotation_consistency": 0.0,
            "scale_ratio": None,
            "rotation_difference": None,
            "reprojection_error": None,
            "confidence": "Low",
            "quality": "Insufficient features",
            "image_quality_a": quality_a,
            "image_quality_b": quality_b,
            "analysis_stage": "Feature extraction",
            "homography_verified": 0,
            "affine_verified": 0,
            "selected_geometry_model": "None",
            "ransac_threshold": None,
            "matching_method": (
                "Multi-view SIFT + FLANN + "
                "reciprocal filtering"
            ),
            "preprocessing": (
                "CLAHE + alternate contrast + "
                "gradient + local contrast"
            ),
            "verification_method": (
                "RANSAC homography / affine "
                "geometric verification"
            ),
            "visualization": encode_jpeg(
                create_side_by_side(
                    image_a,
                    image_b
                )
            ),
            "engine_time": round(
                time.perf_counter()
                - start_time,
                4
            ),
        }

    # --------------------------------------------------------
    # MULTI-PASS MATCHING
    # --------------------------------------------------------

    strict = ratio_match(
        des1,
        des2,
        ratio=0.70
    )

    balanced = ratio_match(
        des1,
        des2,
        ratio=0.76
    )

    tolerant = ratio_match(
        des1,
        des2,
        ratio=0.82
    )

    reciprocal = reciprocal_matches(
        des1,
        des2,
        balanced,
        ratio=0.78
    )

    strict_reciprocal = reciprocal_matches(
        des1,
        des2,
        strict,
        ratio=0.74
    )

    raw_matches = len(
        balanced
    )

    candidate_matches = merge_unique_matches(
        strict,
        strict_reciprocal,
        reciprocal,
        balanced,
        tolerant
    )

    # Avoid allowing very weak matches to dominate geometry.
    candidate_matches = candidate_matches[:750]

    candidate_count = len(
        candidate_matches
    )

    # --------------------------------------------------------
    # GEOMETRIC VERIFICATION
    # --------------------------------------------------------

    threshold = adaptive_ransac_threshold(
        image_a,
        image_b
    )

    geometry = choose_geometry(
        kp1,
        kp2,
        candidate_matches,
        threshold
    )

    verified_matches = geometry[
        "inliers"
    ]

    mask = geometry[
        "mask"
    ]

    geometry_score = geometry[
        "consistency"
    ]

    reprojection_error = geometry[
        "reprojection_error"
    ]

    # --------------------------------------------------------
    # SECONDARY EVIDENCE
    # --------------------------------------------------------

    descriptor_score = descriptor_quality(
        [
            candidate_matches[index]
            for index in range(
                len(candidate_matches)
            )
            if (
                mask is not None
                and index < len(mask)
                and mask[index]
            )
        ]
    )

    spatial_score = spatial_coverage(
        kp1,
        candidate_matches,
        mask,
        image_a.shape
    )

    scale_rotation = scale_rotation_consistency(
        kp1,
        kp2,
        candidate_matches,
        mask
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = calculate_correspondence_score(
        verified=verified_matches,
        candidate=candidate_count,
        feature_a=feature_count_a,
        feature_b=feature_count_b,
        geometry_score=geometry_score,
        descriptor_score=descriptor_score,
        spatial_score=spatial_score,
        quality_score=average_quality,
        scale_rotation_score=scale_rotation["score"],
        reprojection_error=reprojection_error
    )

    confidence = determine_confidence(
        verified_matches,
        geometry_score,
        spatial_score,
        descriptor_score,
        score,
        scale_rotation["score"]
    )

    match_found = determine_match(
        verified_matches,
        geometry_score,
        spatial_score,
        score,
        candidate_count
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    visualization = draw_correspondence_map(
        image_a,
        image_b,
        kp1,
        kp2,
        candidate_matches,
        mask
    )

    engine_time = (
        time.perf_counter()
        - start_time
    )

    return {
        "match_found": bool(match_found),

        "match_percentage": round(
            score,
            2
        ),

        "corresponding_features": int(
            verified_matches
        ),

        "raw_matches": int(
            raw_matches
        ),

        "candidate_matches": int(
            candidate_count
        ),

        "verified_matches": int(
            verified_matches
        ),

        "feature_count_a": int(
            feature_count_a
        ),

        "feature_count_b": int(
            feature_count_b
        ),

        "inlier_ratio": round(
            geometry_score,
            2
        ),

        "geometric_consistency": round(
            geometry_score,
            2
        ),

        "descriptor_quality": round(
            descriptor_score,
            2
        ),

        "spatial_coverage": round(
            spatial_score,
            2
        ),

        "scale_rotation_consistency": round(
            scale_rotation["score"],
            2
        ),

        "scale_ratio": scale_rotation[
            "scale_ratio"
        ],

        "rotation_difference": scale_rotation[
            "rotation_difference"
        ],

        "reprojection_error": (
            round(
                reprojection_error,
                3
            )
            if reprojection_error is not None
            else None
        ),

        "median_descriptor_distance": (
            round(
                median_descriptor_distance(
                    candidate_matches
                ),
                3
            )
            if candidate_matches
            else None
        ),

        "confidence": confidence,

        "quality": (
            "Strong correspondence evidence"
            if match_found
            else "Insufficient correspondence evidence"
        ),

        "image_quality_a": quality_a,

        "image_quality_b": quality_b,

        "analysis_stage": (
            "Geometric verification"
            if candidate_count >= 4
            else "Feature matching"
        ),

        "homography_verified": int(
            geometry[
                "homography_verified"
            ]
        ),

        "affine_verified": int(
            geometry[
                "affine_verified"
            ]
        ),

        "selected_geometry_model": geometry[
            "model"
        ],

        "ransac_threshold": round(
            threshold,
            3
        ),

        "matching_method": (
            "Multi-view SIFT + FLANN + "
            "multi-pass ratio testing + "
            "reciprocal filtering"
        ),

        "preprocessing": (
            "CLAHE + alternate contrast + "
            "gradient representation + "
            "local contrast normalization"
        ),

        "verification_method": (
            "RANSAC homography / affine "
            "verification + reprojection analysis"
        ),

        "visualization": encode_jpeg(
            visualization
        ),

        "engine_time": round(
            engine_time,
            4
        ),
    }


# ============================================================
# INTERPRETATION
# ============================================================

def interpretation_text(result):
    verified = result.get(
        "verified_matches",
        0
    )

    score = result.get(
        "match_percentage",
        0
    )

    confidence = result.get(
        "confidence",
        "Low"
    )

    geometry = result.get(
        "geometric_consistency",
        0
    )

    spatial = result.get(
        "spatial_coverage",
        0
    )

    model = result.get(
        "selected_geometry_model",
        "None"
    )

    if result.get("match_found"):

        return (
            f"The analysis identified {verified} "
            f"geometrically verified feature correspondences. "
            f"The correspondence evidence score is {score:.1f}/100 "
            f"with {confidence.lower()} confidence. "
            f"Geometric consistency is {geometry:.1f}% and "
            f"spatial coverage is {spatial:.1f}%. "
            f"The selected geometric model was {model}. "
            f"Taken together, these measurements support "
            f"a consistent correspondence between the observed "
            f"surface regions."
        )

    return (
        f"The analysis produced {verified} verified "
        f"correspondences with an evidence score of "
        f"{score:.1f}/100. The available evidence does not "
        f"meet the conservative threshold required to confirm "
        f"a corresponding lunar surface region."
    )


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf_report(
    result,
    filename_a,
    filename_b
):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=27,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=7,
    )

    story = []

    story.append(
        Paragraph(
            "LUNARMATCH",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Lunar Image Correspondence Analysis Report",
            subtitle_style
        )
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    summary_data = [
        ["Parameter", "Result"],
        [
            "Correspondence",
            "CONFIRMED"
            if result.get("match_found")
            else "NOT CONFIRMED"
        ],
        [
            "Evidence Score",
            f'{result.get("match_percentage", 0):.2f}/100'
        ],
        [
            "Confidence",
            result.get("confidence", "Low")
        ],
        [
            "Verified Features",
            str(
                result.get(
                    "verified_matches",
                    0
                )
            )
        ],
        [
            "Geometry",
            result.get(
                "selected_geometry_model",
                "None"
            )
        ],
        [
            "Generated",
            timestamp
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            55 * mm,
            115 * mm
        ]
    )

    summary_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#16213e")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
        ])
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(
            1,
            12
        )
    )

    story.append(
        Paragraph(
            "Input Images",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Image A:</b> "
            f"{secure_filename(filename_a)}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Image B:</b> "
            f"{secure_filename(filename_b)}",
            body_style
        )
    )

    story.append(
        Paragraph(
            "Image Quality",
            heading_style
        )
    )

    quality_a = result.get(
        "image_quality_a",
        {}
    )

    quality_b = result.get(
        "image_quality_b",
        {}
    )

    quality_data = [
        [
            "Metric",
            "Image A",
            "Image B"
        ],
        [
            "Resolution",
            f'{quality_a.get("resolution_score", 0):.1f}',
            f'{quality_b.get("resolution_score", 0):.1f}',
        ],
        [
            "Contrast",
            f'{quality_a.get("contrast_score", 0):.1f}',
            f'{quality_b.get("contrast_score", 0):.1f}',
        ],
        [
            "Sharpness",
            f'{quality_a.get("sharpness_score", 0):.1f}',
            f'{quality_b.get("sharpness_score", 0):.1f}',
        ],
        [
            "Overall",
            f'{quality_a.get("overall_score", 0):.1f}',
            f'{quality_b.get("overall_score", 0):.1f}',
        ],
    ]

    quality_table = Table(
        quality_data,
        colWidths=[
            60 * mm,
            55 * mm,
            55 * mm
        ]
    )

    quality_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#16213e")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),
            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        quality_table
    )

    story.append(
        Paragraph(
            "Correspondence Metrics",
            heading_style
        )
    )

    metrics_data = [
        ["Metric", "Value"],
        [
            "Feature Count A",
            str(
                result.get(
                    "feature_count_a",
                    0
                )
            )
        ],
        [
            "Feature Count B",
            str(
                result.get(
                    "feature_count_b",
                    0
                )
            )
        ],
        [
            "Raw Matches",
            str(
                result.get(
                    "raw_matches",
                    0
                )
            )
        ],
        [
            "Candidate Matches",
            str(
                result.get(
                    "candidate_matches",
                    0
                )
            )
        ],
        [
            "Verified Matches",
            str(
                result.get(
                    "verified_matches",
                    0
                )
            )
        ],
        [
            "Geometric Consistency",
            f'{result.get("geometric_consistency", 0):.2f}%'
        ],
        [
            "Descriptor Quality",
            f'{result.get("descriptor_quality", 0):.2f}%'
        ],
        [
            "Spatial Coverage",
            f'{result.get("spatial_coverage", 0):.2f}%'
        ],
        [
            "Scale/Rotation Consistency",
            f'{result.get("scale_rotation_consistency", 0):.2f}%'
        ],
        [
            "Reprojection Error",
            str(
                result.get(
                    "reprojection_error",
                    "N/A"
                )
            )
        ],
    ]

    metrics_table = Table(
        metrics_data,
        colWidths=[
            85 * mm,
            85 * mm
        ]
    )

    metrics_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#16213e")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.lightgrey
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        metrics_table
    )

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Correspondence Map",
            heading_style
        )
    )

    visualization_data = result.get(
        "visualization"
    )

    if visualization_data:

        try:
            image_bytes = base64.b64decode(
                visualization_data
            )

            image_stream = io.BytesIO(
                image_bytes
            )

            report_image = ReportImage(
                image_stream,
                width=175 * mm,
                height=100 * mm
            )

            story.append(
                report_image
            )

        except Exception:
            story.append(
                Paragraph(
                    "Correspondence visualization unavailable.",
                    body_style
                )
            )

    story.append(
        Spacer(
            1,
            10
        )
    )

    story.append(
        Paragraph(
            "Methodology",
            heading_style
        )
    )

    methodology = (
        "LUNARMATCH uses a multi-view classical computer-vision "
        "pipeline designed to improve robustness to changes in "
        "illumination, contrast, scale and local terrain appearance. "
        "SIFT features are extracted from multiple image "
        "representations, followed by FLANN descriptor matching, "
        "multi-pass Lowe ratio testing and reciprocal matching. "
        "Candidate correspondences are subsequently verified using "
        "RANSAC-based homography and affine models. Additional "
        "evidence includes descriptor quality, spatial coverage, "
        "scale consistency, rotation consistency and reprojection "
        "error."
    )

    story.append(
        Paragraph(
            methodology,
            body_style
        )
    )

    story.append(
        Paragraph(
            "Interpretation",
            heading_style
        )
    )

    story.append(
        Paragraph(
            interpretation_text(result),
            body_style
        )
    )

    story.append(
        Paragraph(
            "Technical Note",
            heading_style
        )
    )

    technical_note = (
        "The correspondence score and confidence classification "
        "are engineering metrics derived from the available "
        "image evidence. They are not scientifically validated "
        "probabilities. Reliable scientific performance claims "
        "require evaluation against a representative labelled "
        "lunar-image benchmark containing both corresponding and "
        "non-corresponding image pairs."
    )

    story.append(
        Paragraph(
            technical_note,
            body_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return app.send_static_file(
        "index.html"
    )


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "LUNARMATCH",
        "engine": "Hybrid Multi-view SIFT Correspondence Engine",
        "version": "4.0"
    })


@app.route(
    "/api/match",
    methods=["POST"]
)
def api_match():

    if (
        "imageA" not in request.files
        or "imageB" not in request.files
    ):
        return jsonify({
            "error": (
                "Both imageA and imageB "
                "are required."
            )
        }), 400

    image_a = request.files[
        "imageA"
    ]

    image_b = request.files[
        "imageB"
    ]

    if (
        not image_a.filename
        or not image_b.filename
    ):
        return jsonify({
            "error": "Both images must have filenames."
        }), 400

    if (
        not allowed_file(image_a.filename)
        or not allowed_file(image_b.filename)
    ):
        return jsonify({
            "error": (
                "Unsupported file format. "
                "Use JPG, JPEG, PNG, TIFF or WEBP."
            )
        }), 400

    temp_a = None
    temp_b = None

    try:

        suffix_a = os.path.splitext(
            image_a.filename
        )[1].lower()

        suffix_b = os.path.splitext(
            image_b.filename
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix_a
        ) as file_a:

            image_a.save(
                file_a.name
            )

            temp_a = file_a.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix_b
        ) as file_b:

            image_b.save(
                file_b.name
            )

            temp_b = file_b.name

        result = run_matching(
            temp_a,
            temp_b
        )

        result["processing_time"] = result[
            "engine_time"
        ]

        result["filename_a"] = secure_filename(
            image_a.filename
        )

        result["filename_b"] = secure_filename(
            image_b.filename
        )

        result["generated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        return jsonify(result)

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500

    finally:

        if temp_a and os.path.exists(
            temp_a
        ):
            try:
                os.remove(temp_a)
            except OSError:
                pass

        if temp_b and os.path.exists(
            temp_b
        ):
            try:
                os.remove(temp_b)
            except OSError:
                pass


@app.route(
    "/api/report",
    methods=["POST"]
)
def api_report():

    if (
        "imageA" not in request.files
        or "imageB" not in request.files
    ):
        return jsonify({
            "error": (
                "Both imageA and imageB "
                "are required."
            )
        }), 400

    image_a = request.files[
        "imageA"
    ]

    image_b = request.files[
        "imageB"
    ]

    if (
        not image_a.filename
        or not image_b.filename
    ):
        return jsonify({
            "error": "Both images are required."
        }), 400

    if (
        not allowed_file(image_a.filename)
        or not allowed_file(image_b.filename)
    ):
        return jsonify({
            "error": "Unsupported image format."
        }), 400

    temp_a = None
    temp_b = None

    try:

        suffix_a = os.path.splitext(
            image_a.filename
        )[1].lower()

        suffix_b = os.path.splitext(
            image_b.filename
        )[1].lower()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix_a
        ) as file_a:

            image_a.save(
                file_a.name
            )

            temp_a = file_a.name

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix_b
        ) as file_b:

            image_b.save(
                file_b.name
            )

            temp_b = file_b.name

        result = run_matching(
            temp_a,
            temp_b
        )

        pdf_buffer = build_pdf_report(
            result,
            image_a.filename,
            image_b.filename
        )

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=(
                "LUNARMATCH_Analysis_Report.pdf"
            )
        )

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 500

    finally:

        if temp_a and os.path.exists(
            temp_a
        ):
            try:
                os.remove(temp_a)
            except OSError:
                pass

        if temp_b and os.path.exists(
            temp_b
        ):
            try:
                os.remove(temp_b)
            except OSError:
                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("=" * 60)
    print("LUNARMATCH v4.0")
    print("Multi-view Lunar Correspondence Engine")
    print("=" * 60)
    print()
    print(
        f"Server running at: http://localhost:{port}"
    )
    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )