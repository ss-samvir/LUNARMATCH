import base64
import os
import tempfile
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename


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

MAX_DIMENSION = 1600


# =========================================================
# IMAGE UTILITIES
# =========================================================

def encode_jpeg(image):
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 88]
    )

    if not ok:
        raise RuntimeError("Could not encode visualization.")

    return base64.b64encode(buffer).decode("ascii")


def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            "Could not read one of the uploaded images."
        )

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest > MAX_DIMENSION:

        scale = MAX_DIMENSION / float(longest)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        image = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    return image


def image_quality(image):
    """
    Basic image-quality estimate using:
    - resolution
    - contrast
    - Laplacian sharpness
    """

    h, w = image.shape[:2]

    resolution_score = min(
        100.0,
        ((h * w) / (1000 * 1000)) * 100
    )

    contrast = float(np.std(image))
    contrast_score = min(100.0, contrast * 2.2)

    sharpness = float(
        cv2.Laplacian(image, cv2.CV_64F).var()
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

    return {
        "resolution": f"{w} × {h}",
        "contrast_score": round(contrast_score, 1),
        "sharpness_score": round(sharpness_score, 1),
        "quality_score": round(overall, 1)
    }


# =========================================================
# MATCHING ENGINE
# =========================================================

def run_matching(path1, path2):

    start_engine = time.perf_counter()

    img1 = prepare_image(path1)
    img2 = prepare_image(path2)

    quality1 = image_quality(img1)
    quality2 = image_quality(img2)

    # -----------------------------------------------------
    # SIFT FEATURE EXTRACTION
    # -----------------------------------------------------

    sift = cv2.SIFT_create(
        nfeatures=5000,
        contrastThreshold=0.025,
        edgeThreshold=10,
        sigma=1.6
    )

    kp1, des1 = sift.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        img2,
        None
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
            "confidence": "Low",
            "quality": "Insufficient features",
            "feature_count_a": feature_count_a,
            "feature_count_b": feature_count_b,
            "inlier_ratio": 0.0,
            "geometric_consistency": 0.0,
            "image_quality_a": quality1,
            "image_quality_b": quality2,
            "analysis_stage": "Feature extraction",
            "visualization": encode_jpeg(vis),
            "engine_time": round(
                time.perf_counter() - start_engine,
                3
            )
        }

    # -----------------------------------------------------
    # DESCRIPTOR MATCHING
    # -----------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False
    )

    raw_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    # -----------------------------------------------------
    # LOWE RATIO TEST
    # -----------------------------------------------------

    ratio_matches = []

    for pair in raw_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.72 * n.distance:
            ratio_matches.append(m)

    # -----------------------------------------------------
    # CROSS-CHECK
    # -----------------------------------------------------

    reverse_matches = matcher.knnMatch(
        des2,
        des1,
        k=2
    )

    reverse_good = {}

    for pair in reverse_matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.72 * n.distance:

            reverse_good[m.queryIdx] = m.trainIdx

    mutual_matches = []

    for m in ratio_matches:

        reverse_train = reverse_good.get(
            m.trainIdx
        )

        if reverse_train == m.queryIdx:
            mutual_matches.append(m)

    # Use ratio matches if cross-check becomes too restrictive.
    if len(mutual_matches) >= 8:
        candidate_matches = mutual_matches
    else:
        candidate_matches = ratio_matches

    # -----------------------------------------------------
    # GEOMETRIC VERIFICATION USING RANSAC
    # -----------------------------------------------------

    inlier_matches = []
    homography = None
    geometric_consistency = 0.0

    if len(candidate_matches) >= 4:

        pts1 = np.float32([
            kp1[m.queryIdx].pt
            for m in candidate_matches
        ]).reshape(-1, 1, 2)

        pts2 = np.float32([
            kp2[m.trainIdx].pt
            for m in candidate_matches
        ]).reshape(-1, 1, 2)

        try:

            homography, mask = cv2.findHomography(
                pts1,
                pts2,
                cv2.RANSAC,
                5.0,
                maxIters=3000,
                confidence=0.995
            )

            if mask is not None:

                mask = mask.ravel().astype(bool)

                inlier_matches = [
                    m
                    for m, is_inlier
                    in zip(candidate_matches, mask)
                    if is_inlier
                ]

                if candidate_matches:
                    geometric_consistency = (
                        len(inlier_matches)
                        / len(candidate_matches)
                    ) * 100.0

        except cv2.error:

            homography = None

    # -----------------------------------------------------
    # FINAL MATCH SCORE
    # -----------------------------------------------------

    minimum_features = max(
        1,
        min(feature_count_a, feature_count_b)
    )

    verified_count = len(inlier_matches)
    candidate_count = len(candidate_matches)

    # Feature coverage.
    coverage = (
        verified_count /
        minimum_features
    ) * 100.0

    # Cap because this is a correspondence score,
    # not a probability.
    coverage = min(100.0, coverage)

    # Combine verified feature coverage and
    # geometric consistency.
    match_score = (
        coverage * 0.65 +
        geometric_consistency * 0.35
    )

    match_score = min(
        100.0,
        max(0.0, match_score)
    )

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if (
        verified_count >= 35
        and geometric_consistency >= 55
        and match_score >= 25
    ):

        confidence = "High"

    elif (
        verified_count >= 15
        and geometric_consistency >= 35
        and match_score >= 10
    ):

        confidence = "Medium"

    else:

        confidence = "Low"

    # -----------------------------------------------------
    # QUALITY LABEL
    # -----------------------------------------------------

    average_quality = (
        quality1["quality_score"] +
        quality2["quality_score"]
    ) / 2

    if average_quality >= 70:
        quality_label = "Excellent"
    elif average_quality >= 45:
        quality_label = "Good"
    elif average_quality >= 25:
        quality_label = "Fair"
    else:
        quality_label = "Limited"

    # -----------------------------------------------------
    # MATCH DECISION
    # -----------------------------------------------------

    match_found = (
        verified_count >= 10
        and geometric_consistency >= 25
        and match_score >= 8
    )

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    display_matches = inlier_matches[:120]

    if len(display_matches) >= 1:

        vis = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            display_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 180, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    else:

        vis = create_side_by_side(
            img1,
            img2
        )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {
        "match_found": match_found,

        "match_percentage": round(
            match_score,
            1
        ),

        "corresponding_features": verified_count,

        "raw_matches": len(ratio_matches),

        "verified_matches": verified_count,

        "candidate_matches": candidate_count,

        "feature_count_a": feature_count_a,

        "feature_count_b": feature_count_b,

        "inlier_ratio": round(
            geometric_consistency,
            1
        ),

        "geometric_consistency": round(
            geometric_consistency,
            1
        ),

        "confidence": confidence,

        "quality": quality_label,

        "image_quality_a": quality1,

        "image_quality_b": quality2,

        "analysis_stage": "Complete",

        "homography_verified": homography is not None,

        "visualization": encode_jpeg(vis),

        "engine_time": round(
            time.perf_counter() - start_engine,
            3
        )
    }


# =========================================================
# VISUALIZATION HELPERS
# =========================================================

def create_side_by_side(img1, img2):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_height = max(
        h1,
        h2
    )

    def resize_height(image):

        h, w = image.shape[:2]

        scale = target_height / float(h)

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
        "status": "ok"
    })


# =========================================================
# API
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

            f1.save(temp1.name)
            temp_paths.append(temp1.name)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix2
        ) as temp2:

            f2.save(temp2.name)
            temp_paths.append(temp2.name)

        result = run_matching(
            temp_paths[0],
            temp_paths[1]
        )

        result["processing_time"] = round(
            time.perf_counter() - start,
            3
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
