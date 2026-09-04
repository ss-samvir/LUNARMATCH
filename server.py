import base64
import io
import os
import tempfile
import time
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
# LUNARMATCH
# Lunar Image Correspondence & Verification Engine
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MAX_DIMENSION = 1800


# ============================================================
# BASIC UTILITIES
# ============================================================

def encode_jpeg(image, quality=92):
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not ok:
        return ""

    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image.")

    h, w = image.shape

    scale = min(1.0, MAX_DIMENSION / max(h, w))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return image


def preprocess_image(image):
    """
    Main preprocessing pipeline.

    CLAHE improves local lunar surface contrast while preserving
    crater/ridge information.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    # Gentle denoising
    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return enhanced


def alternate_preprocess(image):
    """
    Second representation used to recover features that may be
    weakened by the main preprocessing pipeline.
    """

    clahe = cv2.createCLAHE(
        clipLimit=1.6,
        tileGridSize=(12, 12)
    )

    enhanced = clahe.apply(image)

    # Unsharp enhancement
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)

    enhanced = cv2.addWeighted(
        enhanced,
        1.35,
        blur,
        -0.35,
        0
    )

    return enhanced


# ============================================================
# IMAGE QUALITY
# ============================================================

def image_quality(image):
    h, w = image.shape
    area = h * w

    resolution_score = min(
        100.0,
        (area / 1_000_000.0) * 100.0
    )

    contrast = float(np.std(image))
    contrast_score = min(
        100.0,
        contrast * 2.2
    )

    sharpness = float(
        cv2.Laplacian(image, cv2.CV_64F).var()
    )

    sharpness_score = min(
        100.0,
        sharpness / 12.0
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
        "resolution": f"{w} × {h}",
        "width": int(w),
        "height": int(h),
        "contrast_score": round(contrast_score, 2),
        "sharpness_score": round(sharpness_score, 2),
        "quality_score": round(overall, 2),
        "label": label
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def create_sift():
    return cv2.SIFT_create(
        nfeatures=9000,
        contrastThreshold=0.014,
        edgeThreshold=12,
        sigma=1.6
    )


def extract_sift(image):
    sift = create_sift()

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    return keypoints, descriptors


def extract_multiview_sift(image):
    """
    Extract features from two complementary representations.

    This increases robustness against illumination and contrast
    differences while retaining SIFT descriptor compatibility.
    """

    main = preprocess_image(image)
    alternate = alternate_preprocess(image)

    kp1, des1 = extract_sift(main)
    kp2, des2 = extract_sift(alternate)

    all_kp = []
    all_des = []

    if kp1 and des1 is not None:
        all_kp.extend(kp1)
        all_des.append(des1)

    if kp2 and des2 is not None:
        all_kp.extend(kp2)
        all_des.append(des2)

    if not all_des:
        return [], None

    descriptors = np.vstack(all_des)

    return all_kp, descriptors


# ============================================================
# FEATURE MATCHING
# ============================================================

def ratio_match(des1, des2, ratio=0.76):
    if des1 is None or des2 is None:
        return []

    if len(des1) == 0 or len(des2) < 2:
        return []

    index_params = dict(
        algorithm=1,
        trees=8
    )

    search_params = dict(
        checks=100
    )

    matcher = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    try:
        matches = matcher.knnMatch(
            des1,
            des2,
            k=2
        )
    except cv2.error:
        return []

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good.append(m)

    return good


def reciprocal_matches(des1, des2, forward, ratio=0.78):

    reverse = ratio_match(
        des2,
        des1,
        ratio
    )

    reverse_map = set()

    for m in reverse:
        reverse_map.add(
            (m.queryIdx, m.trainIdx)
        )

    result = []

    for m in forward:

        if (
            m.trainIdx,
            m.queryIdx
        ) in reverse_map:
            result.append(m)

    return result


def merge_unique_matches(*match_sets):

    unique = {}

    for match_set in match_sets:

        for m in match_set:

            key = (
                int(m.queryIdx),
                int(m.trainIdx)
            )

            if (
                key not in unique
                or m.distance < unique[key].distance
            ):
                unique[key] = m

    result = list(unique.values())

    result.sort(
        key=lambda x: x.distance
    )

    return result


# ============================================================
# MATCH QUALITY
# ============================================================

def descriptor_quality(matches):
    if not matches:
        return 0.0

    distances = np.array(
        [m.distance for m in matches],
        dtype=np.float32
    )

    # SIFT L2 distance is lower when descriptors are more similar.
    # Normalize against a practical descriptor-distance range.
    median_distance = float(
        np.median(distances)
    )

    quality = 100.0 * (
        1.0 -
        min(
            1.0,
            median_distance / 300.0
        )
    )

    return float(
        max(0.0, min(100.0, quality))
    )


# ============================================================
# GEOMETRIC VERIFICATION
# ============================================================

def adaptive_ransac_threshold(image_a, image_b):

    h1, w1 = image_a.shape
    h2, w2 = image_b.shape

    diagonal_a = np.sqrt(
        w1 ** 2 + h1 ** 2
    )

    diagonal_b = np.sqrt(
        w2 ** 2 + h2 ** 2
    )

    diagonal = min(
        diagonal_a,
        diagonal_b
    )

    threshold = diagonal * 0.004

    return float(
        max(
            3.0,
            min(
                8.0,
                threshold
            )
        )
    )


def verify_homography(
    kp1,
    kp2,
    matches,
    ransac_threshold
):

    if len(matches) < 4:
        return [], None, 0.0

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    try:

        homography, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            ransac_threshold,
            maxIters=10000,
            confidence=0.995
        )

    except cv2.error:

        return [], None, 0.0

    if homography is None or mask is None:
        return [], None, 0.0

    mask = mask.ravel().astype(bool)

    inliers = [
        match
        for match, flag
        in zip(matches, mask)
        if flag
    ]

    consistency = (
        len(inliers) /
        max(1, len(matches))
    ) * 100.0

    return (
        inliers,
        homography,
        consistency
    )


def verify_affine(
    kp1,
    kp2,
    matches,
    ransac_threshold
):

    if len(matches) < 3:
        return [], None, 0.0

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    )

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    )

    try:

        matrix, mask = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=10000,
            confidence=0.995,
            refineIters=20
        )

    except cv2.error:

        return [], None, 0.0

    if matrix is None or mask is None:
        return [], None, 0.0

    mask = mask.ravel().astype(bool)

    inliers = [
        match
        for match, flag
        in zip(matches, mask)
        if flag
    ]

    consistency = (
        len(inliers) /
        max(1, len(matches))
    ) * 100.0

    return (
        inliers,
        matrix,
        consistency
    )


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def spatial_coverage(
    keypoints,
    matches,
    image_shape
):

    if not matches:
        return 0.0

    h, w = image_shape

    # 4 × 4 spatial grid.
    grid = np.zeros(
        (4, 4),
        dtype=np.uint8
    )

    for m in matches:

        x, y = keypoints[
            m.queryIdx
        ].pt

        gx = int(
            min(
                3,
                max(
                    0,
                    x / max(1, w) * 4
                )
            )
        )

        gy = int(
            min(
                3,
                max(
                    0,
                    y / max(1, h) * 4
                )
            )
        )

        grid[gy, gx] = 1

    occupied = int(
        np.sum(grid)
    )

    return (
        occupied /
        16.0
    ) * 100.0


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    feature_count_a,
    feature_count_b,
    candidate_count,
    verified_count,
    geometric_consistency,
    descriptor_score,
    spatial_score,
    quality_a,
    quality_b
):

    minimum_features = max(
        1,
        min(
            feature_count_a,
            feature_count_b
        )
    )

    # How much of the available feature population
    # is actually supported by verified correspondence.
    coverage = (
        verified_count /
        minimum_features
    ) * 100.0

    coverage = min(
        100.0,
        coverage
    )

    # Saturating absolute evidence score.
    absolute_strength = (
        100.0 *
        (1.0 - np.exp(-verified_count / 45.0))
    )

    # Candidate-to-verified efficiency.
    verification_precision = (
        verified_count /
        max(
            1,
            candidate_count
        )
    ) * 100.0

    verification_precision = min(
        100.0,
        verification_precision
    )

    image_quality_score = (
        quality_a +
        quality_b
    ) / 2.0

    score = (
        coverage * 0.16
        + absolute_strength * 0.22
        + geometric_consistency * 0.24
        + descriptor_score * 0.12
        + spatial_score * 0.12
        + verification_precision * 0.07
        + image_quality_score * 0.07
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


def confidence_label(
    verified,
    geometry,
    spatial,
    descriptor,
    score
):

    if (
        verified >= 45
        and geometry >= 55
        and spatial >= 35
        and descriptor >= 55
        and score >= 55
    ):
        return "High"

    if (
        verified >= 25
        and geometry >= 40
        and spatial >= 25
        and descriptor >= 40
        and score >= 30
    ):
        return "Medium"

    return "Low"


# ============================================================
# VISUALIZATION
# ============================================================

def draw_correspondence_map(
    image_a,
    kp1,
    image_b,
    kp2,
    matches
):

    if not matches:

        blank = np.hstack(
            [image_a, image_b]
        )

        blank = cv2.cvtColor(
            blank,
            cv2.COLOR_GRAY2BGR
        )

        cv2.putText(
            blank,
            "NO VERIFIED CORRESPONDENCES",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        return encode_jpeg(blank)

    ordered = sorted(
        matches,
        key=lambda m: m.distance
    )

    selected = ordered[:180]

    left = cv2.cvtColor(
        image_a,
        cv2.COLOR_GRAY2BGR
    )

    right = cv2.cvtColor(
        image_b,
        cv2.COLOR_GRAY2BGR
    )

    canvas = cv2.drawMatches(
        left,
        kp1,
        right,
        kp2,
        selected,
        None,
        matchColor=(40, 220, 120),
        singlePointColor=(255, 180, 40),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return encode_jpeg(
        canvas,
        quality=92
    )


def create_side_by_side(
    image_a,
    image_b
):

    h1, w1 = image_a.shape
    h2, w2 = image_b.shape

    target_h = min(
        700,
        max(
            h1,
            h2
        )
    )

    def resize_to_height(img, height):

        scale = height / img.shape[0]

        return cv2.resize(
            img,
            (
                int(img.shape[1] * scale),
                height
            ),
            interpolation=cv2.INTER_AREA
        )

    a = resize_to_height(
        image_a,
        target_h
    )

    b = resize_to_height(
        image_b,
        target_h
    )

    combined = np.hstack(
        [a, b]
    )

    return encode_jpeg(
        combined,
        quality=90
    )


# ============================================================
# MAIN MATCHING ENGINE
# ============================================================

def run_matching(path1, path2):

    start = time.perf_counter()

    image_a = prepare_image(path1)
    image_b = prepare_image(path2)

    quality_a = image_quality(
        image_a
    )

    quality_b = image_quality(
        image_b
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

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
        or feature_count_a < 4
        or feature_count_b < 4
    ):

        visualization = draw_correspondence_map(
            image_a,
            kp1,
            image_b,
            kp2,
            []
        )

        elapsed = time.perf_counter() - start

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

            "confidence": "Low",
            "quality": (
                "Limited"
            ),

            "image_quality_a": quality_a,
            "image_quality_b": quality_b,

            "analysis_stage": (
                "Insufficient feature evidence"
            ),

            "homography_verified": False,
            "affine_verified": False,

            "matching_method": (
                "Multi-view SIFT + FLANN"
            ),

            "preprocessing": (
                "CLAHE + Gaussian normalization "
                "+ alternate contrast representation"
            ),

            "verification_method": (
                "RANSAC homography / affine verification"
            ),

            "visualization": visualization,

            "engine_time": round(
                elapsed,
                3
            )
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

    # Strong candidates are given priority.
    candidate_matches = merge_unique_matches(
        strict,
        reciprocal,
        balanced
    )

    # Tolerant matches are considered only when they
    # do not duplicate stronger evidence.
    candidate_matches = merge_unique_matches(
        candidate_matches,
        tolerant
    )

    candidate_matches = candidate_matches[:650]

    raw_matches = len(
        balanced
    )

    candidate_count = len(
        candidate_matches
    )

    # --------------------------------------------------------
    # GEOMETRIC VERIFICATION
    # --------------------------------------------------------

    ransac_threshold = adaptive_ransac_threshold(
        image_a,
        image_b
    )

    homography_inliers, homography, homography_consistency = (
        verify_homography(
            kp1,
            kp2,
            candidate_matches,
            ransac_threshold
        )
    )

    affine_inliers, affine_matrix, affine_consistency = (
        verify_affine(
            kp1,
            kp2,
            candidate_matches,
            ransac_threshold
        )
    )

    # Select the model with stronger verified evidence.
    if len(affine_inliers) > len(
        homography_inliers
    ):
        verified = affine_inliers
        model = "Affine"
        model_object = affine_matrix
        geometric_consistency = affine_consistency
    else:
        verified = homography_inliers
        model = "Homography"
        model_object = homography
        geometric_consistency = homography_consistency

    verified_count = len(
        verified
    )

    # --------------------------------------------------------
    # EVIDENCE METRICS
    # --------------------------------------------------------

    descriptor_score = descriptor_quality(
        verified
    )

    spatial_score = spatial_coverage(
        kp1,
        verified,
        image_a.shape
    )

    score = calculate_score(
        feature_count_a,
        feature_count_b,
        candidate_count,
        verified_count,
        geometric_consistency,
        descriptor_score,
        spatial_score,
        quality_a["quality_score"],
        quality_b["quality_score"]
    )

    confidence = confidence_label(
        verified_count,
        geometric_consistency,
        spatial_score,
        descriptor_score,
        score
    )

    # More conservative actual match decision.
    match_found = (
        verified_count >= 12
        and geometric_consistency >= 25
        and spatial_score >= 12
        and score >= 18
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    visualization = draw_correspondence_map(
        image_a,
        kp1,
        image_b,
        kp2,
        verified
    )

    elapsed = (
        time.perf_counter() -
        start
    )

    avg_quality = (
        quality_a["quality_score"]
        + quality_b["quality_score"]
    ) / 2.0

    if avg_quality >= 75:
        quality_label = "Excellent"
    elif avg_quality >= 55:
        quality_label = "Good"
    elif avg_quality >= 35:
        quality_label = "Fair"
    else:
        quality_label = "Limited"

    return {
        "match_found": bool(
            match_found
        ),

        "match_percentage": round(
            score,
            2
        ),

        "corresponding_features": int(
            verified_count
        ),

        "raw_matches": int(
            raw_matches
        ),

        "candidate_matches": int(
            candidate_count
        ),

        "verified_matches": int(
            verified_count
        ),

        "feature_count_a": int(
            feature_count_a
        ),

        "feature_count_b": int(
            feature_count_b
        ),

        "inlier_ratio": round(
            geometric_consistency,
            2
        ),

        "geometric_consistency": round(
            geometric_consistency,
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

        "confidence": confidence,

        "quality": quality_label,

        "image_quality_a": quality_a,

        "image_quality_b": quality_b,

        "analysis_stage": (
            "Correspondence verified"
            if match_found
            else "Insufficient correspondence evidence"
        ),

        "homography_verified": (
            homography is not None
        ),

        "affine_verified": (
            affine_matrix is not None
        ),

        "selected_geometry_model": model,

        "ransac_threshold": round(
            ransac_threshold,
            2
        ),

        "matching_method": (
            "Multi-view SIFT + FLANN "
            "+ reciprocal filtering"
        ),

        "preprocessing": (
            "CLAHE + Gaussian normalization "
            "+ alternate contrast representation"
        ),

        "verification_method": (
            "RANSAC homography / "
            "affine geometric verification"
        ),

        "visualization": visualization,

        "engine_time": round(
            elapsed,
            3
        )
    }


# ============================================================
# REPORT INTERPRETATION
# ============================================================

def interpretation_text(result):

    score = result["match_percentage"]
    confidence = result["confidence"]
    verified = result["verified_matches"]
    geometry = result["geometric_consistency"]
    spatial = result["spatial_coverage"]

    if result["match_found"]:

        return (
            f"The correspondence engine identified {verified} "
            f"geometrically verified feature correspondences. "
            f"The final correspondence score is {score:.1f}/100 "
            f"with {confidence} confidence. "
            f"Geometric consistency is {geometry:.1f}% and "
            f"spatial coverage is {spatial:.1f}%. "
            f"These independent measurements provide supporting "
            f"evidence that the two images contain a consistent "
            f"set of corresponding surface features."
        )

    return (
        f"The engine identified {verified} geometrically "
        f"verified correspondences. The resulting score is "
        f"{score:.1f}/100 with {confidence} confidence. "
        f"The available evidence does not satisfy the prototype's "
        f"correspondence criteria strongly enough to classify the "
        f"images as a confirmed match."
    )


# ============================================================
# PDF REPORT
# ============================================================

def build_pdf_report(result, filename_a, filename_b):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=27,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555")
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=7,
        textColor=colors.HexColor("#1d3557")
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        spaceAfter=7
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#666666")
    )

    story = []

    # --------------------------------------------------------
    # COVER
    # --------------------------------------------------------

    story.append(
        Spacer(1, 20 * mm)
    )

    story.append(
        Paragraph(
            "LUNARMATCH",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Lunar Image Correspondence & Verification Report",
            subtitle_style
        )
    )

    story.append(
        Spacer(1, 10 * mm)
    )

    story.append(
        Paragraph(
            "CHANDRAYAAN-2 DATA INTERFACE",
            subtitle_style
        )
    )

    story.append(
        Spacer(1, 16 * mm)
    )

    summary_data = [
        ["Analysis Status", "MATCH VERIFIED" if result["match_found"] else "NO CONFIRMED MATCH"],
        ["Correspondence Score", f"{result['match_percentage']:.1f} / 100"],
        ["Confidence", result["confidence"]],
        ["Verified Features", str(result["verified_matches"])],
        ["Geometric Consistency", f"{result['geometric_consistency']:.1f}%"],
        ["Spatial Coverage", f"{result['spatial_coverage']:.1f}%"],
        ["Analysis Time", f"{result['engine_time']:.3f} s"],
        ["Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")]
    ]

    table = Table(
        summary_data,
        colWidths=[70 * mm, 95 * mm]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f8")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2df")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 10 * mm)
    )

    story.append(
        Paragraph(
            "Executive Interpretation",
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
        PageBreak()
    )

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. Input Data & Image Quality",
            heading_style
        )
    )

    input_data = [
        ["Parameter", "Image A", "Image B"],
        [
            "Filename",
            filename_a,
            filename_b
        ],
        [
            "Resolution",
            result["image_quality_a"]["resolution"],
            result["image_quality_b"]["resolution"]
        ],
        [
            "Contrast Score",
            f"{result['image_quality_a']['contrast_score']:.1f}",
            f"{result['image_quality_b']['contrast_score']:.1f}"
        ],
        [
            "Sharpness Score",
            f"{result['image_quality_a']['sharpness_score']:.1f}",
            f"{result['image_quality_b']['sharpness_score']:.1f}"
        ],
        [
            "Overall Quality",
            f"{result['image_quality_a']['quality_score']:.1f}",
            f"{result['image_quality_b']['quality_score']:.1f}"
        ],
        [
            "Quality Classification",
            result["image_quality_a"]["label"],
            result["image_quality_b"]["label"]
        ]
    ]

    table = Table(
        input_data,
        colWidths=[
            52 * mm,
            63 * mm,
            63 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2df")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(table)

    story.append(
        Paragraph(
            "The quality metrics are diagnostic indicators used to "
            "interpret matching reliability. They are not a measure "
            "of scientific image calibration or mission-grade data quality.",
            small_style
        )
    )

    story.append(
        Paragraph(
            "2. Feature & Correspondence Analysis",
            heading_style
        )
    )

    matching_data = [
        ["Metric", "Result"],
        ["Detected Features — Image A", str(result["feature_count_a"])],
        ["Detected Features — Image B", str(result["feature_count_b"])],
        ["Raw Descriptor Matches", str(result["raw_matches"])],
        ["Candidate Correspondences", str(result["candidate_matches"])],
        ["Verified Correspondences", str(result["verified_matches"])],
        ["Descriptor Quality", f"{result['descriptor_quality']:.1f}%"],
        ["Geometric Consistency", f"{result['geometric_consistency']:.1f}%"],
        ["Spatial Coverage", f"{result['spatial_coverage']:.1f}%"],
        ["RANSAC Threshold", f"{result['ransac_threshold']:.2f} px"],
        ["Selected Geometry Model", result["selected_geometry_model"]],
        ["Homography Verified", "YES" if result["homography_verified"] else "NO"],
        ["Affine Model Verified", "YES" if result["affine_verified"] else "NO"],
    ]

    table = Table(
        matching_data,
        colWidths=[
            100 * mm,
            65 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2df")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story.append(table)

    story.append(
        Spacer(1, 8 * mm)
    )

    # --------------------------------------------------------
    # CORRESPONDENCE MAP
    # --------------------------------------------------------

    if result.get("visualization"):

        try:

            image_bytes = base64.b64decode(
                result["visualization"]
            )

            report_image = ReportImage(
                io.BytesIO(image_bytes)
            )

            report_image._restrictSize(
                175 * mm,
                95 * mm
            )

            story.append(
                Paragraph(
                    "3. Verified Correspondence Map",
                    heading_style
                )
            )

            story.append(
                report_image
            )

            story.append(
                Spacer(1, 4 * mm)
            )

            story.append(
                Paragraph(
                    "Lines represent verified feature correspondences "
                    "selected after descriptor matching and geometric "
                    "consistency filtering. Only a limited number of "
                    "the strongest verified correspondences are rendered "
                    "for visual readability.",
                    small_style
                )
            )

        except Exception:
            pass

    story.append(
        PageBreak()
    )

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Analysis Methodology",
            heading_style
        )
    )

    methodology = [
        (
            "<b>1. Acquisition</b> — The uploaded images are decoded "
            "and normalized to a controlled processing resolution."
        ),
        (
            "<b>2. Preprocessing</b> — CLAHE-based local contrast "
            "enhancement and Gaussian normalization are applied."
        ),
        (
            "<b>3. Feature Extraction</b> — SIFT descriptors are "
            "generated from complementary image representations."
        ),
        (
            "<b>4. Descriptor Matching</b> — Multiple Lowe-ratio "
            "matching passes and reciprocal consistency filtering "
            "identify candidate correspondences."
        ),
        (
            "<b>5. Geometric Verification</b> — RANSAC-based "
            "homography and affine models reject geometrically "
            "inconsistent feature matches."
        ),
        (
            "<b>6. Evidence Scoring</b> — The final score combines "
            "verified feature coverage, absolute correspondence "
            "strength, geometric consistency, descriptor quality, "
            "spatial distribution and image quality."
        ),
        (
            "<b>7. Reporting</b> — The verified evidence is compiled "
            "into this technical analysis report."
        )
    ]

    for item in methodology:

        story.append(
            Paragraph(
                item,
                body_style
            )
        )

    story.append(
        Paragraph(
            "5. Interpretation",
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
            "6. Technical Limitations",
            heading_style
        )
    )

    limitations = (
        "LUNARMATCH is a computer-vision prototype. The reported "
        "correspondence score and confidence classification are "
        "engineering metrics derived from image features and "
        "geometric verification; they are not scientifically "
        "validated probabilities. Results can be influenced by "
        "image resolution, illumination, viewing geometry, "
        "compression, terrain appearance and the presence of "
        "repeated surface textures."
    )

    story.append(
        Paragraph(
            limitations,
            body_style
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    story.append(
        Paragraph(
            "LUNARMATCH • Lunar Image Correspondence System",
            subtitle_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# API ROUTES
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
        "engine": (
            "Multi-view SIFT + FLANN + "
            "RANSAC correspondence engine"
        ),
        "version": "3.0"
    })


@app.route("/api/match", methods=["POST"])
def api_match():

    start = time.perf_counter()

    if "imageA" not in request.files:
        return jsonify({
            "error": "Image A is required."
        }), 400

    if "imageB" not in request.files:
        return jsonify({
            "error": "Image B is required."
        }), 400

    file_a = request.files["imageA"]
    file_b = request.files["imageB"]

    ext_a = os.path.splitext(
        secure_filename(
            file_a.filename or ""
        )
    )[1].lower()

    ext_b = os.path.splitext(
        secure_filename(
            file_b.filename or ""
        )
    )[1].lower()

    if ext_a not in ALLOWED:
        return jsonify({
            "error": "Unsupported format for Image A."
        }), 400

    if ext_b not in ALLOWED:
        return jsonify({
            "error": "Unsupported format for Image B."
        }), 400

    temp_a = tempfile.NamedTemporaryFile(
        suffix=ext_a,
        delete=False
    )

    temp_b = tempfile.NamedTemporaryFile(
        suffix=ext_b,
        delete=False
    )

    temp_a.close()
    temp_b.close()

    try:

        file_a.save(
            temp_a.name
        )

        file_b.save(
            temp_b.name
        )

        result = run_matching(
            temp_a.name,
            temp_b.name
        )

        result["processing_time"] = round(
            time.perf_counter() -
            start,
            3
        )

        result["filename_a"] = (
            secure_filename(
                file_a.filename
            )
        )

        result["filename_b"] = (
            secure_filename(
                file_b.filename
            )
        )

        return jsonify(result)

    except Exception as exc:

        return jsonify({
            "error": (
                "Analysis failed: "
                + str(exc)
            )
        }), 500

    finally:

        try:
            os.unlink(
                temp_a.name
            )
        except Exception:
            pass

        try:
            os.unlink(
                temp_b.name
            )
        except Exception:
            pass


@app.route("/api/report", methods=["POST"])
def api_report():

    if "imageA" not in request.files:
        return jsonify({
            "error": "Image A is required."
        }), 400

    if "imageB" not in request.files:
        return jsonify({
            "error": "Image B is required."
        }), 400

    file_a = request.files["imageA"]
    file_b = request.files["imageB"]

    ext_a = os.path.splitext(
        secure_filename(
            file_a.filename or ""
        )
    )[1].lower()

    ext_b = os.path.splitext(
        secure_filename(
            file_b.filename or ""
        )
    )[1].lower()

    if ext_a not in ALLOWED or ext_b not in ALLOWED:
        return jsonify({
            "error": "Unsupported image format."
        }), 400

    temp_a = tempfile.NamedTemporaryFile(
        suffix=ext_a,
        delete=False
    )

    temp_b = tempfile.NamedTemporaryFile(
        suffix=ext_b,
        delete=False
    )

    temp_a.close()
    temp_b.close()

    try:

        file_a.save(
            temp_a.name
        )

        file_b.save(
            temp_b.name
        )

        result = run_matching(
            temp_a.name,
            temp_b.name
        )

        pdf = build_pdf_report(
            result,
            secure_filename(
                file_a.filename
            ),
            secure_filename(
                file_b.filename
            )
        )

        return send_file(
            pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=(
                "LUNARMATCH_Analysis_Report.pdf"
            )
        )

    except Exception as exc:

        return jsonify({
            "error": (
                "Report generation failed: "
                + str(exc)
            )
        }), 500

    finally:

        try:
            os.unlink(
                temp_a.name
            )
        except Exception:
            pass

        try:
            os.unlink(
                temp_b.name
            )
        except Exception:
            pass


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
