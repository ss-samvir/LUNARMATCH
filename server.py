import os
import cv2
import time
import uuid
import gc
import traceback
import numpy as np
import torch
import kornia.feature as KF

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename


# ============================================================
# LUNARMATCH V5.3
# Hybrid Learned + Classical Lunar Correspondence Engine
# Render Free / Low-Memory Optimized Edition
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

# ------------------------------------------------------------
# SERVER LIMITS
# ------------------------------------------------------------

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "webp"
}

# ------------------------------------------------------------
# DEVICE / CPU SETTINGS
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reduce CPU memory pressure on small Render instances.
if DEVICE == "cpu":
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ============================================================
# LOFTR INITIALIZATION
# ============================================================

LOFTR = None
LOFTR_STATUS = "not_loaded"

try:
    print("Initializing LoFTR...")

    LOFTR = KF.LoFTR(
        pretrained="outdoor"
    ).eval().to(DEVICE)

    LOFTR_STATUS = "ready"

    print("LoFTR initialized successfully.")
    print("Device:", DEVICE)

except Exception as e:
    LOFTR = None
    LOFTR_STATUS = "unavailable"

    print("LoFTR initialization failed.")
    print(str(e))


# ============================================================
# BASIC HELPERS
# ============================================================

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def set_safe_filename(filename):
    return secure_filename(filename)


# ============================================================
# IMAGE PROCESSING
# ============================================================

def resize_for_processing(image, max_dimension=1000):
    """
    Resize large images before processing.

    1000 px is a compromise between correspondence quality
    and memory usage on constrained servers.
    """

    if image is None:
        return None

    h, w = image.shape[:2]

    largest = max(h, w)

    if largest <= max_dimension:
        return image

    scale = max_dimension / float(largest)

    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def normalize_image(image):
    """
    Convert image to grayscale and improve local contrast.
    """

    if image is None:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# ============================================================
# IMAGE QUALITY
# ============================================================

def calculate_quality(image):
    if image is None:
        return {
            "width": 0,
            "height": 0,
            "resolution": 0,
            "contrast": 0,
            "sharpness": 0,
            "quality_score": 0,
            "label": "LOW"
        }

    h, w = image.shape[:2]

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    contrast = float(np.std(gray))

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(laplacian.var())

    resolution_score = min(
        100.0,
        ((w * h) / (1500 * 1500)) * 100.0
    )

    contrast_score = min(
        100.0,
        contrast * 2.0
    )

    sharpness_score = min(
        100.0,
        sharpness / 10.0
    )

    quality_score = (
        resolution_score * 0.30 +
        contrast_score * 0.30 +
        sharpness_score * 0.40
    )

    if quality_score >= 75:
        label = "EXCELLENT"
    elif quality_score >= 55:
        label = "GOOD"
    elif quality_score >= 35:
        label = "FAIR"
    else:
        label = "LOW"

    return {
        "width": int(w),
        "height": int(h),
        "resolution": int(w * h),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "quality_score": round(quality_score, 2),
        "label": label
    }


# ============================================================
# LOFTR IMAGE PREPARATION
# ============================================================

def prepare_loftr_image(gray):
    """
    Prepare image for LoFTR.

    IMPORTANT:
    The previous V5.3 implementation allowed up to 1200 px.
    This version uses 640 px to significantly reduce the
    transformer memory footprint on free Render instances.
    """

    if gray is None:
        return None

    max_dimension = 640

    h, w = gray.shape[:2]

    largest = max(h, w)

    scale = min(
        1.0,
        max_dimension / float(largest)
    )

    if scale < 1.0:
        new_w = max(
            8,
            int(w * scale) // 8 * 8
        )

        new_h = max(
            8,
            int(h * scale) // 8 * 8
        )

        gray = cv2.resize(
            gray,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    else:
        new_w = max(
            8,
            (w // 8) * 8
        )

        new_h = max(
            8,
            (h // 8) * 8
        )

        if new_w != w or new_h != h:
            gray = cv2.resize(
                gray,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

    gray = np.ascontiguousarray(
        gray,
        dtype=np.uint8
    )

    tensor = torch.from_numpy(
        gray
    ).float() / 255.0

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# LOFTR MATCHING
# ============================================================

def run_loftr(image_a, image_b):
    """
    Run LoFTR correspondence matching.

    Returns:
        points_a
        points_b
        confidences
    """

    if LOFTR is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32)
        )

    tensor_a = None
    tensor_b = None
    output = None

    try:
        tensor_a = prepare_loftr_image(image_a)
        tensor_b = prepare_loftr_image(image_b)

        if tensor_a is None or tensor_b is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        with torch.inference_mode():

            output = LOFTR({
                "image0": tensor_a,
                "image1": tensor_b
            })

        if output is None:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        if "keypoints0" not in output:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        points_a = (
            output["keypoints0"]
            .detach()
            .cpu()
            .numpy()
        )

        points_b = (
            output["keypoints1"]
            .detach()
            .cpu()
            .numpy()
        )

        confidence = (
            output["confidence"]
            .detach()
            .cpu()
            .numpy()
        )

        # Confidence filtering
        threshold = 0.20

        mask = confidence >= threshold

        points_a = points_a[mask]
        points_b = points_b[mask]
        confidence = confidence[mask]

        return (
            points_a.astype(np.float32),
            points_b.astype(np.float32),
            confidence.astype(np.float32)
        )

    except Exception as e:

        print("LoFTR inference error:")
        print(str(e))

        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32)
        )

    finally:

        del tensor_a
        del tensor_b
        del output

        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        gc.collect()


# ============================================================
# GEOMETRIC VERIFICATION
# ============================================================

def geometric_verification(points_a, points_b):
    """
    Verify correspondence using homography and affine geometry.
    """

    count = len(points_a)

    if count < 4:
        return {
            "verified": 0,
            "inlier_ratio": 0.0,
            "geometric_consistency": 0.0,
            "reprojection_error": 999.0,
            "model": "insufficient"
        }

    try:

        homography, mask_h = cv2.findHomography(
            points_a,
            points_b,
            cv2.RANSAC,
            5.0
        )

    except Exception:
        homography = None
        mask_h = None

    try:

        affine, mask_a = cv2.estimateAffinePartial2D(
            points_a,
            points_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

    except Exception:
        affine = None
        mask_a = None

    h_inliers = (
        int(mask_h.sum())
        if mask_h is not None
        else 0
    )

    a_inliers = (
        int(mask_a.sum())
        if mask_a is not None
        else 0
    )

    if h_inliers >= a_inliers and mask_h is not None:

        selected_mask = mask_h.ravel().astype(bool)

        model = "homography"

        selected_points_a = points_a[selected_mask]
        selected_points_b = points_b[selected_mask]

        if homography is not None and len(selected_points_a) > 0:

            projected = cv2.perspectiveTransform(
                selected_points_a.reshape(
                    -1,
                    1,
                    2
                ),
                homography
            ).reshape(-1, 2)

            errors = np.linalg.norm(
                projected - selected_points_b,
                axis=1
            )

            reprojection_error = float(
                np.mean(errors)
            )

    elif mask_a is not None:

        selected_mask = mask_a.ravel().astype(bool)

        model = "affine"

        selected_points_a = points_a[selected_mask]
        selected_points_b = points_b[selected_mask]

        if affine is not None and len(selected_points_a) > 0:

            transformed = cv2.transform(
                selected_points_a.reshape(
                    -1,
                    1,
                    2
                ),
                affine
            ).reshape(-1, 2)

            errors = np.linalg.norm(
                transformed - selected_points_b,
                axis=1
            )

            reprojection_error = float(
                np.mean(errors)
            )

    else:

        return {
            "verified": 0,
            "inlier_ratio": 0.0,
            "geometric_consistency": 0.0,
            "reprojection_error": 999.0,
            "model": "none"
        }

    verified = int(selected_mask.sum())

    inlier_ratio = (
        verified / float(count)
    ) if count else 0.0

    reprojection_error = locals().get(
        "reprojection_error",
        999.0
    )

    geometric_consistency = max(
        0.0,
        min(
            100.0,
            inlier_ratio * 100.0
        )
    )

    return {
        "verified": verified,
        "inlier_ratio": round(
            inlier_ratio * 100.0,
            2
        ),
        "geometric_consistency": round(
            geometric_consistency,
            2
        ),
        "reprojection_error": round(
            float(reprojection_error),
            2
        ),
        "model": model,
        "mask": selected_mask
    }


# ============================================================
# SPATIAL COVERAGE
# ============================================================

def calculate_spatial_coverage(points, image_shape):
    if points is None or len(points) == 0:
        return 0.0

    h, w = image_shape[:2]

    grid_size = 4

    occupied = set()

    for point in points:

        x = float(point[0])
        y = float(point[1])

        gx = int(
            np.clip(
                x / max(w, 1) * grid_size,
                0,
                grid_size - 1
            )
        )

        gy = int(
            np.clip(
                y / max(h, 1) * grid_size,
                0,
                grid_size - 1
            )
        )

        occupied.add(
            (gx, gy)
        )

    total_cells = grid_size * grid_size

    return (
        len(occupied) /
        float(total_cells)
    ) * 100.0


# ============================================================
# SIFT SUPPORT
# ============================================================

def run_sift_support(image_a, image_b):
    """
    Classical SIFT verification.

    Uses multiple image representations but keeps the feature
    count moderate to reduce memory and CPU usage.
    """

    try:

        sift = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.012,
            edgeThreshold=12,
            sigma=1.6
        )

        gray_a = cv2.cvtColor(
            image_a,
            cv2.COLOR_BGR2GRAY
        )

        gray_b = cv2.cvtColor(
            image_b,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        views_a = [
            gray_a,
            clahe.apply(gray_a)
        ]

        views_b = [
            gray_b,
            clahe.apply(gray_b)
        ]

        all_good = []

        for va, vb in zip(
            views_a,
            views_b
        ):

            kp_a, des_a = sift.detectAndCompute(
                va,
                None
            )

            kp_b, des_b = sift.detectAndCompute(
                vb,
                None
            )

            if des_a is None or des_b is None:
                continue

            if len(kp_a) < 2 or len(kp_b) < 2:
                continue

            matcher = cv2.BFMatcher(
                cv2.NORM_L2
            )

            matches = matcher.knnMatch(
                des_a,
                des_b,
                k=2
            )

            for pair in matches:

                if len(pair) < 2:
                    continue

                m, n = pair

                if m.distance < 0.78 * n.distance:

                    all_good.append(
                        (
                            kp_a[m.queryIdx].pt,
                            kp_b[m.trainIdx].pt,
                            m.distance
                        )
                    )

        # Spatial deduplication
        dedup = {}

        for point_a, point_b, distance in all_good:

            x, y = point_a

            key = (
                int(x / 6),
                int(y / 6)
            )

            if key not in dedup:
                dedup[key] = (
                    point_a,
                    point_b,
                    distance
                )

        candidates = list(
            dedup.values()
        )

        if len(candidates) < 4:

            return {
                "candidates": len(candidates),
                "verified": 0,
                "support": 0.0
            }

        pts_a = np.float32([
            x[0]
            for x in candidates
        ])

        pts_b = np.float32([
            x[1]
            for x in candidates
        ])

        _, mask = cv2.findHomography(
            pts_a,
            pts_b,
            cv2.RANSAC,
            5.0
        )

        if mask is None:

            return {
                "candidates": len(candidates),
                "verified": 0,
                "support": 0.0
            }

        verified = int(
            mask.sum()
        )

        support = (
            verified /
            float(len(candidates))
        ) * 100.0

        return {
            "candidates": len(candidates),
            "verified": verified,
            "support": round(
                support,
                2
            )
        }

    except Exception as e:

        print("SIFT error:")
        print(str(e))

        return {
            "candidates": 0,
            "verified": 0,
            "support": 0.0
        }


# ============================================================
# SCORE CALCULATION
# ============================================================

def calculate_score(
    verified_matches,
    confidence,
    inlier_ratio,
    spatial_coverage,
    reprojection_error,
    sift_support,
    quality_a,
    quality_b
):

    verification_strength = min(
        100.0,
        verified_matches * 2.0
    )

    confidence_strength = (
        confidence * 100.0
    )

    inlier_strength = min(
        100.0,
        inlier_ratio
    )

    spatial_strength = min(
        100.0,
        spatial_coverage
    )

    if reprojection_error >= 999:
        reprojection_strength = 0.0
    else:
        reprojection_strength = max(
            0.0,
            min(
                100.0,
                100.0 -
                reprojection_error * 5.0
            )
        )

    quality_strength = (
        quality_a["quality_score"] +
        quality_b["quality_score"]
    ) / 2.0

    score = (
        verification_strength * 0.25 +
        confidence_strength * 0.20 +
        inlier_strength * 0.20 +
        spatial_strength * 0.10 +
        reprojection_strength * 0.10 +
        sift_support * 0.10 +
        quality_strength * 0.05
    )

    return round(
        float(
            np.clip(
                score,
                0.0,
                100.0
            )
        ),
        2
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_match(
    score,
    verified,
    geometric_consistency,
    spatial_coverage
):

    if (
        score >= 65 and
        verified >= 25 and
        geometric_consistency >= 55 and
        spatial_coverage >= 25
    ):
        return "HIGH"

    if (
        score >= 45 and
        verified >= 12 and
        geometric_consistency >= 35 and
        spatial_coverage >= 15
    ):
        return "MEDIUM"

    return "LOW"


# ============================================================
# INTERPRETATION
# ============================================================

def generate_interpretation(
    decision,
    score,
    verified,
    confidence,
    geometric_consistency
):

    if decision == "HIGH":

        return (
            "Strong visual correspondence detected between "
            "the two lunar images. A substantial number of "
            "feature correspondences passed geometric "
            "verification, indicating a high-confidence match."
        )

    if decision == "MEDIUM":

        return (
            "Moderate visual correspondence detected. "
            "The images contain meaningful matching structures, "
            "but the available evidence is not strong enough "
            "for a high-confidence correspondence."
        )

    return (
        "Low correspondence detected. The available feature "
        "matches and geometric evidence are insufficient to "
        "establish a strong relationship between the images."
    )


# ============================================================
# VISUALIZATION
# ============================================================

def create_correspondence_visualization(
    image_a,
    image_b,
    points_a,
    points_b,
    mask
):

    try:

        height = max(
            image_a.shape[0],
            image_b.shape[0]
        )

        width_a = image_a.shape[1]
        width_b = image_b.shape[1]

        canvas = np.zeros(
            (
                height,
                width_a + width_b,
                3
            ),
            dtype=np.uint8
        )

        canvas[
            :image_a.shape[0],
            :image_a.shape[1]
        ] = image_a

        canvas[
            :image_b.shape[0],
            width_a:
        ] = image_b

        if mask is not None:

            verified_a = points_a[mask]
            verified_b = points_b[mask]

            max_draw = min(
                len(verified_a),
                120
            )

            for i in range(max_draw):

                pa = verified_a[i]
                pb = verified_b[i]

                x1 = int(pa[0])
                y1 = int(pa[1])

                x2 = int(pb[0]) + width_a
                y2 = int(pb[1])

                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 0),
                    1,
                    cv2.LINE_AA
                )

                cv2.circle(
                    canvas,
                    (x1, y1),
                    3,
                    (0, 255, 0),
                    -1
                )

                cv2.circle(
                    canvas,
                    (x2, y2),
                    3,
                    (0, 255, 0),
                    -1
                )

        filename = (
            "correspondence_" +
            str(uuid.uuid4()) +
            ".jpg"
        )

        output_path = os.path.join(
            RESULT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            canvas,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                88
            ]
        )

        return "/results/" + filename

    except Exception as e:

        print("Visualization error:")
        print(str(e))

        return None


# ============================================================
# PIPELINE
# ============================================================

def pipeline_state():

    return {
        "acquire": "complete",
        "preprocess": "complete",
        "extract": "complete",
        "match": "complete",
        "verify": "complete",
        "score": "complete",
        "report": "ready"
    }


# ============================================================
# MAIN MATCH ENGINE
# ============================================================

def perform_match(image_a, image_b):

    start_time = time.time()

    original_a = image_a.copy()
    original_b = image_b.copy()

    # --------------------------------------------------------
    # IMAGE QUALITY
    # --------------------------------------------------------

    quality_a = calculate_quality(
        original_a
    )

    quality_b = calculate_quality(
        original_b
    )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image_a = resize_for_processing(
        image_a,
        1000
    )

    image_b = resize_for_processing(
        image_b,
        1000
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    gray_a = normalize_image(
        image_a
    )

    gray_b = normalize_image(
        image_b
    )

    # --------------------------------------------------------
    # LOFTR
    # --------------------------------------------------------

    points_a, points_b, confidence_values = run_loftr(
        gray_a,
        gray_b
    )

    candidate_matches = len(
        points_a
    )

    if candidate_matches > 0:

        mean_confidence = float(
            np.mean(
                confidence_values
            )
        )

    else:

        mean_confidence = 0.0

    # --------------------------------------------------------
    # GEOMETRY
    # --------------------------------------------------------

    geometry = geometric_verification(
        points_a,
        points_b
    )

    verified_matches = geometry[
        "verified"
    ]

    inlier_ratio = geometry[
        "inlier_ratio"
    ]

    geometric_consistency = geometry[
        "geometric_consistency"
    ]

    reprojection_error = geometry[
        "reprojection_error"
    ]

    model = geometry[
        "model"
    ]

    geometry_mask = geometry.get(
        "mask",
        np.zeros(
            len(points_a),
            dtype=bool
        )
    )

    # --------------------------------------------------------
    # SPATIAL COVERAGE
    # --------------------------------------------------------

    spatial_coverage = calculate_spatial_coverage(
        points_a[geometry_mask]
        if len(points_a) == len(geometry_mask)
        else points_a,
        image_a.shape
    )

    # --------------------------------------------------------
    # SIFT
    # --------------------------------------------------------

    sift = run_sift_support(
        image_a,
        image_b
    )

    sift_support = sift[
        "support"
    ]

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = calculate_score(
        verified_matches=verified_matches,
        confidence=mean_confidence,
        inlier_ratio=inlier_ratio,
        spatial_coverage=spatial_coverage,
        reprojection_error=reprojection_error,
        sift_support=sift_support,
        quality_a=quality_a,
        quality_b=quality_b
    )

    decision = classify_match(
        score,
        verified_matches,
        geometric_consistency,
        spatial_coverage
    )

    interpretation = generate_interpretation(
        decision,
        score,
        verified_matches,
        mean_confidence,
        geometric_consistency
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    visualization = create_correspondence_visualization(
        image_a,
        image_b,
        points_a,
        points_b,
        geometry_mask
    )

    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    processing_time = (
        time.time() - start_time
    )

    return {
        "score": score,
        "confidence": round(
            mean_confidence * 100.0,
            2
        ),
        "decision": decision,

        "candidate_matches": candidate_matches,
        "verified_matches": verified_matches,

        "sift_candidates": sift[
            "candidates"
        ],

        "sift_verified": sift[
            "verified"
        ],

        "sift_support": sift_support,

        "inlier_ratio": inlier_ratio,

        "geometric_consistency":
            geometric_consistency,

        "spatial_coverage":
            round(
                spatial_coverage,
                2
            ),

        "reprojection_error":
            reprojection_error,

        "model": model,

        "quality_a": quality_a,
        "quality_b": quality_b,

        "visualization": visualization,

        "interpretation": interpretation,

        "pipeline": pipeline_state(),

        "processing_time":
            round(
                processing_time,
                2
            )
    }


# ============================================================
# API: MATCH
# ============================================================

@app.route(
    "/api/match",
    methods=["POST"]
)
def match_api():

    try:

        # ----------------------------------------------------
        # FILE A
        # ----------------------------------------------------

        file_a = (
            request.files.get("imageA")
            or request.files.get("image1")
            or request.files.get("fileA")
        )

        # ----------------------------------------------------
        # FILE B
        # ----------------------------------------------------

        file_b = (
            request.files.get("imageB")
            or request.files.get("image2")
            or request.files.get("fileB")
        )

        if file_a is None or file_b is None:

            return jsonify({
                "success": False,
                "error": "Both Image A and Image B are required."
            }), 400

        if not allowed_file(file_a.filename):

            return jsonify({
                "success": False,
                "error": "Unsupported format for Image A."
            }), 400

        if not allowed_file(file_b.filename):

            return jsonify({
                "success": False,
                "error": "Unsupported format for Image B."
            }), 400

        # ----------------------------------------------------
        # READ FILES
        # ----------------------------------------------------

        bytes_a = file_a.read()
        bytes_b = file_b.read()

        if not bytes_a or not bytes_b:

            return jsonify({
                "success": False,
                "error": "One or both uploaded files are empty."
            }), 400

        array_a = np.frombuffer(
            bytes_a,
            dtype=np.uint8
        )

        array_b = np.frombuffer(
            bytes_b,
            dtype=np.uint8
        )

        image_a = cv2.imdecode(
            array_a,
            cv2.IMREAD_COLOR
        )

        image_b = cv2.imdecode(
            array_b,
            cv2.IMREAD_COLOR
        )

        if image_a is None or image_b is None:

            return jsonify({
                "success": False,
                "error": "Unable to decode one or both images."
            }), 400

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        result = perform_match(
            image_a,
            image_b
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {
            "success": True,

            "version": "5.3",
            "engine": (
                "Hybrid Learned + Classical "
                "Lunar Correspondence Engine"
            ),

            "device": DEVICE,

            "loftr": LOFTR_STATUS,

            "score": result["score"],

            "confidence": result[
                "confidence"
            ],

            "decision": result[
                "decision"
            ],

            "candidate_matches":
                result[
                    "candidate_matches"
                ],

            "verified_matches":
                result[
                    "verified_matches"
                ],

            "sift_candidates":
                result[
                    "sift_candidates"
                ],

            "sift_verified":
                result[
                    "sift_verified"
                ],

            "sift_support":
                result[
                    "sift_support"
                ],

            "inlier_ratio":
                result[
                    "inlier_ratio"
                ],

            "geometric_consistency":
                result[
                    "geometric_consistency"
                ],

            "spatial_coverage":
                result[
                    "spatial_coverage"
                ],

            "reprojection_error":
                result[
                    "reprojection_error"
                ],

            "model":
                result[
                    "model"
                ],

            "quality_a":
                result[
                    "quality_a"
                ],

            "quality_b":
                result[
                    "quality_b"
                ],

            "visualization":
                result[
                    "visualization"
                ],

            "interpretation":
                result[
                    "interpretation"
                ],

            "pipeline":
                result[
                    "pipeline"
                ],

            "processing_time":
                result[
                    "processing_time"
                ]
        }

        # ----------------------------------------------------
        # MEMORY CLEANUP
        # ----------------------------------------------------

        del bytes_a
        del bytes_b
        del array_a
        del array_b
        del image_a
        del image_b

        gc.collect()

        return jsonify(response)

    except Exception as e:

        print("MATCH API ERROR")
        traceback.print_exc()

        gc.collect()

        return jsonify({
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }), 500


# ============================================================
# API: REPORT
# ============================================================

@app.route(
    "/api/report",
    methods=["POST"]
)
def report_api():

    try:

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        file_a = (
            request.files.get("imageA")
            or request.files.get("image1")
            or request.files.get("fileA")
        )

        file_b = (
            request.files.get("imageB")
            or request.files.get("image2")
            or request.files.get("fileB")
        )

        if file_a is None or file_b is None:

            return jsonify({
                "success": False,
                "error": "Both images are required."
            }), 400

        bytes_a = file_a.read()
        bytes_b = file_b.read()

        array_a = np.frombuffer(
            bytes_a,
            dtype=np.uint8
        )

        array_b = np.frombuffer(
            bytes_b,
            dtype=np.uint8
        )

        image_a = cv2.imdecode(
            array_a,
            cv2.IMREAD_COLOR
        )

        image_b = cv2.imdecode(
            array_b,
            cv2.IMREAD_COLOR
        )

        if image_a is None or image_b is None:

            return jsonify({
                "success": False,
                "error": "Unable to decode uploaded images."
            }), 400

        result = perform_match(
            image_a,
            image_b
        )

        report_id = str(
            uuid.uuid4()
        )

        pdf_filename = (
            "LUNARMATCH_Report_" +
            report_id +
            ".pdf"
        )

        pdf_path = os.path.join(
            RESULT_DIR,
            pdf_filename
        )

        pdf = canvas.Canvas(
            pdf_path,
            pagesize=A4
        )

        page_width, page_height = A4

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        pdf.setFont(
            "Helvetica-Bold",
            22
        )

        pdf.drawString(
            50,
            page_height - 60,
            "LUNARMATCH"
        )

        pdf.setFont(
            "Helvetica",
            10
        )

        pdf.drawString(
            50,
            page_height - 80,
            "Lunar Image Correspondence Analysis"
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        y = page_height - 125

        pdf.setFont(
            "Helvetica-Bold",
            13
        )

        pdf.drawString(
            50,
            y,
            "Analysis Summary"
        )

        y -= 28

        pdf.setFont(
            "Helvetica",
            10
        )

        summary = [
            f"Match Score: {result['score']}/100",
            f"Confidence: {result['confidence']}%",
            f"Decision: {result['decision']}",
            f"Candidate Matches: {result['candidate_matches']}",
            f"Verified Matches: {result['verified_matches']}",
            f"Geometric Consistency: {result['geometric_consistency']}%",
            f"Spatial Coverage: {result['spatial_coverage']}%",
            f"Inlier Ratio: {result['inlier_ratio']}%",
            f"Model: {result['model']}",
            f"Processing Time: {result['processing_time']} seconds"
        ]

        for line in summary:

            pdf.drawString(
                60,
                y,
                line
            )

            y -= 18

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        y -= 10

        pdf.setFont(
            "Helvetica-Bold",
            13
        )

        pdf.drawString(
            50,
            y,
            "Interpretation"
        )

        y -= 22

        pdf.setFont(
            "Helvetica",
            9
        )

        text = result[
            "interpretation"
        ]

        # Basic wrapping
        words = text.split()
        line = ""

        for word in words:

            test = (
                line + " " + word
            ).strip()

            if pdf.stringWidth(
                test,
                "Helvetica",
                9
            ) > 480:

                pdf.drawString(
                    60,
                    y,
                    line
                )

                y -= 14
                line = word

            else:

                line = test

        if line:

            pdf.drawString(
                60,
                y,
                line
            )

        y -= 35

        # ----------------------------------------------------
        # CORRESPONDENCE IMAGE
        # ----------------------------------------------------

        visualization = result[
            "visualization"
        ]

        if visualization:

            visualization_filename = (
                visualization
                .replace(
                    "/results/",
                    ""
                )
            )

            visualization_path = os.path.join(
                RESULT_DIR,
                visualization_filename
            )

            if os.path.exists(
                visualization_path
            ):

                pdf.setFont(
                    "Helvetica-Bold",
                    13
                )

                pdf.drawString(
                    50,
                    y,
                    "Correspondence Map"
                )

                y -= 20

                try:

                    img = cv2.imread(
                        visualization_path
                    )

                    if img is not None:

                        h, w = img.shape[:2]

                        max_w = 500
                        max_h = 280

                        scale = min(
                            max_w / w,
                            max_h / h
                        )

                        draw_w = w * scale
                        draw_h = h * scale

                        pdf.drawImage(
                            ImageReader(
                                visualization_path
                            ),
                            50,
                            max(
                                50,
                                y - draw_h
                            ),
                            width=draw_w,
                            height=draw_h,
                            preserveAspectRatio=True,
                            mask="auto"
                        )

                except Exception:
                    pass

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        pdf.setFont(
            "Helvetica",
            8
        )

        pdf.drawString(
            50,
            30,
            "Generated by LUNARMATCH V5.3"
        )

        pdf.save()

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return send_from_directory(
            RESULT_DIR,
            pdf_filename,
            as_attachment=True,
            download_name=(
                "LUNARMATCH_Analysis_Report.pdf"
            )
        )

    except Exception as e:

        print("REPORT API ERROR")
        traceback.print_exc()

        gc.collect()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RESULT FILES
# ============================================================

@app.route(
    "/results/<path:filename>"
)
def result_file(filename):

    return send_from_directory(
        RESULT_DIR,
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health"
)
def health():

    return jsonify({
        "status": "ok",
        "service": "LUNARMATCH",
        "version": "5.3",
        "engine": (
            "Hybrid Learned + Classical "
            "Lunar Correspondence Engine"
        ),
        "device": DEVICE,
        "loftr": LOFTR_STATUS
    })


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def index():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return jsonify({
        "success": False,
        "error": "Uploaded file is too large."
    }), 413


@app.errorhandler(
    500
)
def internal_error(error):

    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "Starting LUNARMATCH V5.3..."
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "LoFTR:",
        LOFTR_STATUS
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
