import base64
import io
import os
import tempfile
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
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


# =========================================================
# LUNARMATCH — PREMIUM LUNAR CORRESPONDENCE PLATFORM
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


ALLOWED = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp"
}

MAX_DIMENSION = 1800


# =========================================================
# BASIC UTILITIES
# =========================================================

def encode_jpeg(image, quality=90):
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    if not ok:
        raise RuntimeError("Could not encode visualization.")

    return base64.b64encode(buffer).decode("ascii")


def prepare_image(path):
    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Could not read one of the uploaded images."
        )

    h, w = image.shape[:2]

    longest = max(h, w)

    if longest > MAX_DIMENSION:

        scale = MAX_DIMENSION / float(longest)

        new_w = max(
            1,
            int(w * scale)
        )

        new_h = max(
            1,
            int(h * scale)
        )

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image):
    """
    Produces a normalized image for feature extraction.

    CLAHE helps when two lunar images have different
    illumination/contrast characteristics.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    # Very mild denoising.
    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    return enhanced


def image_quality(image):

    h, w = image.shape[:2]

    resolution_score = min(
        100.0,
        ((h * w) / (1000 * 1000)) * 100
    )

    contrast = float(
        np.std(image)
    )

    contrast_score = min(
        100.0,
        contrast * 2.2
    )

    sharpness = float(
        cv2.Laplacian(
            image,
            cv2.CV_64F
        ).var()
    )

    sharpness_score = min(
        100.0,
        sharpness / 12.0
    )

    overall = (
        resolution_score * 0.25 +
        contrast_score * 0.35 +
        sharpness_score * 0.40
    )

    if overall >= 70:
        label = "Excellent"
    elif overall >= 45:
        label = "Good"
    elif overall >= 25:
        label = "Fair"
    else:
        label = "Limited"

    return {
        "resolution": f"{w} × {h}",
        "width": w,
        "height": h,
        "contrast_score": round(
            contrast_score,
            1
        ),
        "sharpness_score": round(
            sharpness_score,
            1
        ),
        "quality_score": round(
            overall,
            1
        ),
        "label": label
    }


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_sift(image):

    sift = cv2.SIFT_create(
        nfeatures=7000,
        contrastThreshold=0.018,
        edgeThreshold=10,
        sigma=1.6
    )

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    return keypoints, descriptors


# =========================================================
# FLANN MATCHING
# =========================================================

def ratio_match(des1, des2, ratio=0.76):

    if des1 is None or des2 is None:
        return []

    index_params = dict(
        algorithm=1,
        trees=5
    )

    search_params = dict(
        checks=80
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


def reciprocal_matches(
    des1,
    des2,
    forward,
    ratio=0.76
):

    reverse = ratio_match(
        des2,
        des1,
        ratio
    )

    reverse_map = {}

    for m in reverse:

        reverse_map[
            (m.queryIdx, m.trainIdx)
        ] = True

    result = []

    for m in forward:

        if reverse_map.get(
            (m.trainIdx, m.queryIdx)
        ):
            result.append(m)

    return result


# =========================================================
# GEOMETRIC VERIFICATION
# =========================================================

def verify_geometry(
    kp1,
    kp2,
    matches
):

    if len(matches) < 4:

        return (
            [],
            None,
            0.0
        )

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    try:

        homography, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            4.5,
            maxIters=5000,
            confidence=0.995
        )

    except cv2.error:

        return (
            [],
            None,
            0.0
        )

    if mask is None:

        return (
            [],
            None,
            0.0
        )

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


# =========================================================
# MATCH SCORE
# =========================================================

def calculate_score(
    feature_count_a,
    feature_count_b,
    candidate_count,
    verified_count,
    geometric_consistency
):

    minimum_features = max(
        1,
        min(
            feature_count_a,
            feature_count_b
        )
    )

    # How much of the available feature set
    # was successfully verified.
    coverage = (
        verified_count /
        minimum_features
    ) * 100.0

    coverage = min(
        100.0,
        coverage
    )

    # Absolute correspondence strength.
    absolute_strength = min(
        100.0,
        (verified_count / 80.0) * 100.0
    )

    # Candidate-to-verified quality.
    verification_strength = (
        geometric_consistency
    )

    score = (
        coverage * 0.30 +
        absolute_strength * 0.30 +
        verification_strength * 0.40
    )

    return min(
        100.0,
        max(
            0.0,
            score
        )
    )


# =========================================================
# CONFIDENCE
# =========================================================

def confidence_label(
    verified,
    geometry,
    score
):

    if (
        verified >= 40
        and geometry >= 55
        and score >= 30
    ):

        return "High"

    if (
        verified >= 20
        and geometry >= 40
        and score >= 15
    ):

        return "Medium"

    return "Low"


# =========================================================
# VISUALIZATION
# =========================================================

def draw_correspondence_map(
    img1,
    kp1,
    img2,
    kp2,
    matches
):

    if not matches:

        return create_side_by_side(
            img1,
            img2
        )

    display_matches = sorted(
        matches,
        key=lambda m: m.distance
    )[:160]

    left = cv2.cvtColor(
        img1,
        cv2.COLOR_GRAY2BGR
    )

    right = cv2.cvtColor(
        img2,
        cv2.COLOR_GRAY2BGR
    )

    visualization = cv2.drawMatches(
        left,
        kp1,
        right,
        kp2,
        display_matches,
        None,
        matchColor=(0, 220, 120),
        singlePointColor=(255, 180, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return visualization


def create_side_by_side(
    img1,
    img2
):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_height = max(
        h1,
        h2
    )

    def resize_height(image):

        h, w = image.shape[:2]

        scale = (
            target_height /
            float(h)
        )

        return cv2.resize(
            image,
            (
                int(w * scale),
                target_height
            )
        )

    a = resize_height(img1)
    b = resize_height(img2)

    a = cv2.cvtColor(
        a,
        cv2.COLOR_GRAY2BGR
    )

    b = cv2.cvtColor(
        b,
        cv2.COLOR_GRAY2BGR
    )

    separator = np.zeros(
        (
            target_height,
            8,
            3
        ),
        dtype=np.uint8
    )

    return np.hstack([
        a,
        separator,
        b
    ])


# =========================================================
# PREMIUM MATCHING ENGINE
# =========================================================

def run_matching(
    path1,
    path2
):

    start_engine = time.perf_counter()

    # -----------------------------------------------------
    # ACQUIRE
    # -----------------------------------------------------

    img1 = prepare_image(path1)
    img2 = prepare_image(path2)

    quality1 = image_quality(img1)
    quality2 = image_quality(img2)

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    enhanced1 = preprocess_image(img1)
    enhanced2 = preprocess_image(img2)

    # -----------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------

    kp1, des1 = extract_sift(
        enhanced1
    )

    kp2, des2 = extract_sift(
        enhanced2
    )

    feature_count_a = len(kp1)
    feature_count_b = len(kp2)

    if des1 is None or des2 is None:

        vis = create_side_by_side(
            img1,
            img2
        )

        return {
            "match_found": False,
            "match_percentage": 0.0,
            "corresponding_features": 0,
            "raw_matches": 0,
            "verified_matches": 0,
            "candidate_matches": 0,
            "feature_count_a": feature_count_a,
            "feature_count_b": feature_count_b,
            "inlier_ratio": 0.0,
            "geometric_consistency": 0.0,
            "confidence": "Low",
            "quality": "Insufficient features",
            "image_quality_a": quality1,
            "image_quality_b": quality2,
            "analysis_stage": "Feature extraction",
            "homography_verified": False,
            "visualization": encode_jpeg(vis),
            "engine_time": round(
                time.perf_counter() -
                start_engine,
                3
            )
        }

    # -----------------------------------------------------
    # MULTI-PASS MATCHING
    # -----------------------------------------------------

    strict_matches = ratio_match(
        des1,
        des2,
        ratio=0.70
    )

    balanced_matches = ratio_match(
        des1,
        des2,
        ratio=0.76
    )

    tolerant_matches = ratio_match(
        des1,
        des2,
        ratio=0.82
    )

    # Prefer reciprocal matches where possible.
    reciprocal = reciprocal_matches(
        des1,
        des2,
        balanced_matches,
        ratio=0.76
    )

    # Combine candidate sets without duplicates.
    combined = {}

    for match in (
        strict_matches +
        reciprocal +
        tolerant_matches
    ):

        key = (
            match.queryIdx,
            match.trainIdx
        )

        if key not in combined:

            combined[key] = match

        else:

            if match.distance < combined[key].distance:
                combined[key] = match

    candidate_matches = list(
        combined.values()
    )

    # Sort by descriptor quality.
    candidate_matches.sort(
        key=lambda m: m.distance
    )

    # Keep the candidate set manageable.
    candidate_matches = candidate_matches[:500]

    # -----------------------------------------------------
    # GEOMETRIC VERIFICATION
    # -----------------------------------------------------

    (
        inlier_matches,
        homography,
        geometric_consistency
    ) = verify_geometry(
        kp1,
        kp2,
        candidate_matches
    )

    verified_count = len(
        inlier_matches
    )

    candidate_count = len(
        candidate_matches
    )

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    score = calculate_score(
        feature_count_a,
        feature_count_b,
        candidate_count,
        verified_count,
        geometric_consistency
    )

    confidence = confidence_label(
        verified_count,
        geometric_consistency,
        score
    )

    # -----------------------------------------------------
    # MATCH DECISION
    # -----------------------------------------------------

    match_found = (
        verified_count >= 10
        and geometric_consistency >= 22
        and score >= 8
    )

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    visualization = draw_correspondence_map(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches
    )

    engine_time = (
        time.perf_counter() -
        start_engine
    )

    return {

        "match_found":
            match_found,

        "match_percentage":
            round(score, 1),

        "corresponding_features":
            verified_count,

        "raw_matches":
            len(balanced_matches),

        "verified_matches":
            verified_count,

        "candidate_matches":
            candidate_count,

        "feature_count_a":
            feature_count_a,

        "feature_count_b":
            feature_count_b,

        "inlier_ratio":
            round(
                geometric_consistency,
                1
            ),

        "geometric_consistency":
            round(
                geometric_consistency,
                1
            ),

        "confidence":
            confidence,

        "quality":
            (
                "Excellent"
                if (
                    quality1["quality_score"] +
                    quality2["quality_score"]
                ) / 2 >= 70
                else
                "Good"
                if (
                    quality1["quality_score"] +
                    quality2["quality_score"]
                ) / 2 >= 45
                else
                "Fair"
                if (
                    quality1["quality_score"] +
                    quality2["quality_score"]
                ) / 2 >= 25
                else
                "Limited"
            ),

        "image_quality_a":
            quality1,

        "image_quality_b":
            quality2,

        "analysis_stage":
            "Complete",

        "homography_verified":
            homography is not None,

        "matching_method":
            "SIFT + FLANN + RANSAC",

        "preprocessing":
            "CLAHE + Gaussian normalization",

        "verification_method":
            "RANSAC homography",

        "visualization":
            encode_jpeg(
                visualization
            ),

        "engine_time":
            round(
                engine_time,
                3
            )
    }


# =========================================================
# PDF REPORT
# =========================================================

def build_pdf_report(
    result,
    filename_a,
    filename_b
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitlePremium",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        spaceAfter=8
    )

    heading_style = ParagraphStyle(
        "HeadingPremium",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=7
    )

    body_style = ParagraphStyle(
        "BodyPremium",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14
    )

    small_style = ParagraphStyle(
        "SmallPremium",
        parent=styles["BodyText"],
        fontSize=8,
        textColor=colors.grey
    )

    story = []

    story.append(
        Paragraph(
            "LUNAR<span color='#777777'>MATCH</span>",
            title_style
        )
    )

    story.append(
        Paragraph(
            "LUNAR IMAGE CORRESPONDENCE ANALYSIS REPORT",
            ParagraphStyle(
                "Subtitle",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=10
            )
        )
    )

    story.append(Spacer(1, 8))

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%d %B %Y · %H:%M UTC"
    )

    story.append(
        Paragraph(
            f"Analysis generated: {timestamp}",
            small_style
        )
    )

    story.append(
        Spacer(1, 12)
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "01 · ANALYSIS SUMMARY",
            heading_style
        )
    )

    summary_data = [

        [
            "MATCH STATUS",
            "MATCH SCORE",
            "CONFIDENCE"
        ],

        [
            (
                "MATCH FOUND"
                if result["match_found"]
                else
                "NO STRONG MATCH"
            ),
            f'{result["match_percentage"]:.1f}%',
            result["confidence"]
        ]
    ]

    table = Table(
        summary_data,
        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm
        ]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#cccccc")
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, 1),
                "Helvetica-Bold"
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
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(table)

    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "02 · INPUT IMAGERY",
            heading_style
        )
    )

    input_data = [
        ["Parameter", "IMAGE A", "IMAGE B"],

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
            "Quality",
            result["image_quality_a"]["label"],
            result["image_quality_b"]["label"]
        ],

        [
            "Contrast",
            str(
                result["image_quality_a"]
                ["contrast_score"]
            ),
            str(
                result["image_quality_b"]
                ["contrast_score"]
            )
        ],

        [
            "Sharpness",
            str(
                result["image_quality_a"]
                ["sharpness_score"]
            ),
            str(
                result["image_quality_b"]
                ["sharpness_score"]
            )
        ]
    ]

    input_table = Table(
        input_data,
        colWidths=[
            45 * mm,
            60 * mm,
            60 * mm
        ]
    )

    input_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
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
                0.4,
                colors.HexColor("#cccccc")
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            )
        ])
    )

    story.append(input_table)

    # -----------------------------------------------------
    # MATCHING METRICS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "03 · FEATURE CORRESPONDENCE",
            heading_style
        )
    )

    metrics = [
        ["Metric", "Value"],

        [
            "Features detected — Image A",
            str(result["feature_count_a"])
        ],

        [
            "Features detected — Image B",
            str(result["feature_count_b"])
        ],

        [
            "Candidate correspondences",
            str(result["candidate_matches"])
        ],

        [
            "Verified correspondences",
            str(result["verified_matches"])
        ],

        [
            "Geometric consistency",
            f'{result["geometric_consistency"]:.1f}%'
        ],

        [
            "Homography verified",
            (
                "YES"
                if result["homography_verified"]
                else
                "NO"
            )
        ],

        [
            "Processing time",
            f'{result["engine_time"]:.3f} sec'
        ]
    ]

    metric_table = Table(
        metrics,
        colWidths=[
            100 * mm,
            65 * mm
        ]
    )

    metric_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#111827")
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
                0.4,
                colors.HexColor("#cccccc")
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8.5
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            )
        ])
    )

    story.append(metric_table)

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "04 · FEATURE CORRESPONDENCE MAP",
            heading_style
        )
    )

    image_data = base64.b64decode(
        result["visualization"]
    )

    visualization_stream = io.BytesIO(
        image_data
    )

    report_image = ReportImage(
        visualization_stream,
        width=170 * mm,
        height=95 * mm
    )

    story.append(
        report_image
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "The correspondence map displays verified feature relationships "
            "that survived geometric RANSAC verification.",
            body_style
        )
    )

    # -----------------------------------------------------
    # METHODOLOGY
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "05 · ANALYSIS METHODOLOGY",
            heading_style
        )
    )

    methodology = (
        "<b>Preprocessing:</b> CLAHE contrast normalization and "
        "mild Gaussian smoothing.<br/>"
        "<b>Feature extraction:</b> SIFT local feature detection "
        "and descriptor generation.<br/>"
        "<b>Descriptor matching:</b> multi-pass FLANN nearest-neighbour "
        "matching with ratio filtering and reciprocal verification.<br/>"
        "<b>Geometric verification:</b> RANSAC homography estimation "
        "to reject geometrically inconsistent correspondences.<br/>"
        "<b>Scoring:</b> combination of verified feature coverage, "
        "absolute correspondence strength and geometric consistency."
    )

    story.append(
        Paragraph(
            methodology,
            body_style
        )
    )

    # -----------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "06 · SYSTEM INTERPRETATION",
            heading_style
        )
    )

    if result["match_found"]:

        interpretation = (
            f"The analysis identified "
            f"{result['verified_matches']} verified feature "
            f"correspondences with a geometric consistency of "
            f"{result['geometric_consistency']:.1f}%. "
            f"The resulting correspondence score was "
            f"{result['match_percentage']:.1f}%, classified as "
            f"{result['confidence']} confidence."
        )

    else:

        interpretation = (
            f"The analysis did not identify a sufficiently strong "
            f"geometrically consistent correspondence under the "
            f"current verification criteria. "
            f"{result['verified_matches']} correspondences were "
            f"verified from {result['candidate_matches']} candidates."
        )

    story.append(
        Paragraph(
            interpretation,
            body_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "LUNARMATCH is a computer-vision correspondence prototype. "
            "The match score represents the output of the implemented "
            "analysis pipeline and should not be interpreted as a "
            "scientifically validated probability of identical lunar "
            "locations.",
            small_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer


# =========================================================
# ERROR HANDLING
# =========================================================

@app.errorhandler(413)
def request_too_large(_error):

    return jsonify({
        "error":
            "Images are too large. "
            "Please use images smaller than "
            "25 MB total."
    }), 413


# =========================================================
# PAGES
# =========================================================

@app.get("/")
def index():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.get("/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "LUNARMATCH",
        "engine": "SIFT + FLANN + RANSAC"
    })


# =========================================================
# MATCH API
# =========================================================

@app.post("/api/match")
def api_match():

    start = time.perf_counter()

    f1 = request.files.get("image1")
    f2 = request.files.get("image2")

    if not f1 or not f2:

        return jsonify({
            "error":
                "Please provide Image A and Image B."
        }), 400

    suffix1 = os.path.splitext(
        secure_filename(
            f1.filename or ""
        )
    )[1].lower()

    suffix2 = os.path.splitext(
        secure_filename(
            f2.filename or ""
        )
    )[1].lower()

    if (
        suffix1 not in ALLOWED
        or suffix2 not in ALLOWED
    ):

        return jsonify({
            "error":
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, TIFF or WEBP."
        }), 400

    temp_paths = []

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix1
        ) as temp1:

            f1.save(
                temp1.name
            )

            temp_paths.append(
                temp1.name
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix2
        ) as temp2:

            f2.save(
                temp2.name
            )

            temp_paths.append(
                temp2.name
            )

        result = run_matching(
            temp_paths[0],
            temp_paths[1]
        )

        result["processing_time"] = round(
            time.perf_counter() - start,
            3
        )

        result["image_a_filename"] = (
            secure_filename(
                f1.filename or
                "image_a"
            )
        )

        result["image_b_filename"] = (
            secure_filename(
                f2.filename or
                "image_b"
            )
        )

        return jsonify(result)

    except Exception as exc:

        app.logger.exception(
            "Image matching failed"
        )

        return jsonify({
            "error":
                f"Image analysis failed: {exc}"
        }), 500

    finally:

        for path in temp_paths:

            try:
                os.remove(path)

            except OSError:
                pass


# =========================================================
# PDF REPORT API
# =========================================================

@app.post("/api/report")
def api_report():

    f1 = request.files.get("image1")
    f2 = request.files.get("image2")

    if not f1 or not f2:

        return jsonify({
            "error":
                "Both images are required to generate the report."
        }), 400

    suffix1 = os.path.splitext(
        secure_filename(
            f1.filename or ""
        )
    )[1].lower()

    suffix2 = os.path.splitext(
        secure_filename(
            f2.filename or ""
        )
    )[1].lower()

    if (
        suffix1 not in ALLOWED
        or suffix2 not in ALLOWED
    ):

        return jsonify({
            "error":
                "Unsupported image format."
        }), 400

    temp_paths = []

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix1
        ) as temp1:

            f1.save(
                temp1.name
            )

            temp_paths.append(
                temp1.name
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix2
        ) as temp2:

            f2.save(
                temp2.name
            )

            temp_paths.append(
                temp2.name
            )

        result = run_matching(
            temp_paths[0],
            temp_paths[1]
        )

        pdf = build_pdf_report(
            result,
            secure_filename(
                f1.filename or
                "image_a"
            ),
            secure_filename(
                f2.filename or
                "image_b"
            )
        )

        filename = (
            "LUNARMATCH_Analysis_Report.pdf"
        )

        return send_file(
            pdf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )

    except Exception as exc:

        app.logger.exception(
            "PDF generation failed"
        )

        return jsonify({
            "error":
                f"Could not generate PDF report: {exc}"
        }), 500

    finally:

        for path in temp_paths:

            try:
                os.remove(path)

            except OSError:
                pass


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
