# ============================================================
# LUNARMATCH V5.3
# Hybrid Learned + Classical Lunar Correspondence Engine
# ============================================================

import os
import cv2
import time
import uuid
import traceback
import numpy as np
import torch
import kornia.feature as KF

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "tif", "tiff", "webp"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("LUNARMATCH V5.3")
print("Hybrid Learned + Classical Correspondence Engine")
print("=" * 65)
print("Device:", DEVICE)
print("Loading LoFTR...")

try:
    LOFTR = KF.LoFTR(pretrained="outdoor").eval().to(DEVICE)
    print("LoFTR loaded successfully.")
except Exception as e:
    print("LoFTR loading failed:", e)
    LOFTR = None

print("=" * 65)


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def resize_for_processing(image, max_dimension=1200):
    h, w = image.shape[:2]

    scale = min(1.0, max_dimension / max(h, w))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(32, int(w * scale)), max(32, int(h * scale))),
            interpolation=cv2.INTER_AREA
        )

    return image


def normalize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# ------------------------------------------------------------
# IMAGE QUALITY
# ------------------------------------------------------------

def calculate_image_quality(gray):

    h, w = gray.shape

    contrast = float(np.std(gray))

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    contrast_score = min(100.0, contrast / 65.0 * 100.0)
    sharpness_score = min(100.0, sharpness / 1500.0 * 100.0)

    resolution_score = min(
        100.0,
        ((w * h) / (1600 * 240)) * 100.0
    )

    quality = (
        0.40 * contrast_score +
        0.40 * sharpness_score +
        0.20 * resolution_score
    )

    if quality >= 90:
        label = "EXCELLENT"
    elif quality >= 75:
        label = "GOOD"
    elif quality >= 55:
        label = "FAIR"
    else:
        label = "LOW"

    return {
        "resolution": f"{w} × {h}",
        "width": w,
        "height": h,
        "contrast": round(contrast, 1),
        "sharpness": round(sharpness, 1),
        "quality_score": round(quality, 1),
        "quality_label": label
    }


# ------------------------------------------------------------
# LOFTR
# ------------------------------------------------------------

def prepare_loftr_image(gray):

    h, w = gray.shape

    scale = min(
        1.0,
        1200.0 / max(h, w)
    )

    if scale < 1.0:
        nw = max(32, int(w * scale))
        nh = max(32, int(h * scale))

        gray = cv2.resize(
            gray,
            (nw, nh),
            interpolation=cv2.INTER_AREA
        )

    h, w = gray.shape

    nh = max(32, (h // 8) * 8)
    nw = max(32, (w // 8) * 8)

    gray = cv2.resize(
        gray,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    return gray


def make_loftr_tensor(gray):

    tensor = torch.from_numpy(
        gray.astype(np.float32)
    ) / 255.0

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


def run_loftr(gray_a, gray_b):

    if LOFTR is None:
        return [], [], []

    original_a = gray_a.shape
    original_b = gray_b.shape

    img_a = prepare_loftr_image(gray_a)
    img_b = prepare_loftr_image(gray_b)

    tensor_a = make_loftr_tensor(img_a)
    tensor_b = make_loftr_tensor(img_b)

    with torch.inference_mode():

        output = LOFTR({
            "image0": tensor_a,
            "image1": tensor_b
        })

    k0 = output["keypoints0"].detach().cpu().numpy()
    k1 = output["keypoints1"].detach().cpu().numpy()
    confidence = output["confidence"].detach().cpu().numpy()

    if len(k0) == 0:
        return [], [], []

    # Confidence filtering
    keep = confidence >= 0.20

    k0 = k0[keep]
    k1 = k1[keep]
    confidence = confidence[keep]

    # Map coordinates back to original processing dimensions
    sy_a = original_a[0] / img_a.shape[0]
    sx_a = original_a[1] / img_a.shape[1]

    sy_b = original_b[0] / img_b.shape[0]
    sx_b = original_b[1] / img_b.shape[1]

    k0[:, 0] *= sx_a
    k0[:, 1] *= sy_a

    k1[:, 0] *= sx_b
    k1[:, 1] *= sy_b

    return k0, k1, confidence


# ------------------------------------------------------------
# GEOMETRIC VERIFICATION
# ------------------------------------------------------------

def verify_geometry(k0, k1):

    if len(k0) < 4:
        return {
            "verified": 0,
            "inlier_ratio": 0.0,
            "reprojection_error": None,
            "model": "INSUFFICIENT DATA",
            "mask": np.zeros(len(k0), dtype=bool)
        }

    best_mask = None
    best_model = "AFFINE / RANSAC"
    best_error = 9999.0

    # Homography
    if len(k0) >= 4:

        H, mask_h = cv2.findHomography(
            k0,
            k1,
            cv2.RANSAC,
            5.0
        )

        if mask_h is not None:

            mask_h = mask_h.ravel().astype(bool)

            if np.any(mask_h):

                src = k0[mask_h]
                dst = k1[mask_h]

                projected = cv2.perspectiveTransform(
                    src.reshape(-1, 1, 2).astype(np.float32),
                    H
                ).reshape(-1, 2)

                error = np.mean(
                    np.linalg.norm(
                        projected - dst,
                        axis=1
                    )
                )

                best_mask = mask_h
                best_error = error
                best_model = "HOMOGRAPHY / RANSAC"

    # Affine
    if len(k0) >= 3:

        M, mask_a = cv2.estimateAffinePartial2D(
            k0,
            k1,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        if mask_a is not None:

            mask_a = mask_a.ravel().astype(bool)

            if np.any(mask_a):

                src = k0[mask_a]
                dst = k1[mask_a]

                projected = cv2.transform(
                    src.reshape(-1, 1, 2).astype(np.float32),
                    M
                ).reshape(-1, 2)

                error = np.mean(
                    np.linalg.norm(
                        projected - dst,
                        axis=1
                    )
                )

                if (
                    best_mask is None or
                    mask_a.sum() > best_mask.sum()
                ):
                    best_mask = mask_a
                    best_error = error
                    best_model = "AFFINE / RANSAC"

    if best_mask is None:

        return {
            "verified": 0,
            "inlier_ratio": 0.0,
            "reprojection_error": None,
            "model": "NO ROBUST MODEL",
            "mask": np.zeros(len(k0), dtype=bool)
        }

    verified = int(best_mask.sum())

    ratio = (
        verified / len(k0) * 100.0
        if len(k0)
        else 0.0
    )

    return {
        "verified": verified,
        "inlier_ratio": ratio,
        "reprojection_error": float(best_error),
        "model": best_model,
        "mask": best_mask
    }


# ------------------------------------------------------------
# SPATIAL COVERAGE
# ------------------------------------------------------------

def calculate_spatial_coverage(points, width, height):

    if len(points) == 0:
        return 0.0

    occupied = set()

    for x, y in points:

        gx = min(3, max(0, int(x / width * 4)))
        gy = min(3, max(0, int(y / height * 4)))

        occupied.add((gx, gy))

    return len(occupied) / 16.0 * 100.0


# ------------------------------------------------------------
# SIFT SUPPORT
# ------------------------------------------------------------

def extract_sift(gray):

    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.012,
        edgeThreshold=12,
        sigma=1.6
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    ).apply(gray)

    sharpened = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        gray,
        1.5,
        sharpened,
        -0.5,
        0
    )

    views = [
        gray,
        clahe,
        sharpened
    ]

    all_kp = []
    all_des = []

    for view in views:

        kp, des = sift.detectAndCompute(
            view,
            None
        )

        if des is not None:
            all_kp.extend(kp)
            all_des.append(des)

    if not all_des:
        return [], None

    descriptors = np.vstack(all_des)

    # Spatial deduplication
    unique_kp = []
    unique_des = []
    cells = set()

    for kp, des in zip(all_kp, descriptors):

        x, y = kp.pt

        cell = (
            int(x / 6),
            int(y / 6)
        )

        if cell in cells:
            continue

        cells.add(cell)
        unique_kp.append(kp)
        unique_des.append(des)

    if not unique_des:
        return [], None

    return unique_kp, np.array(unique_des)


def run_sift(gray_a, gray_b):

    kp_a, des_a = extract_sift(gray_a)
    kp_b, des_b = extract_sift(gray_b)

    if des_a is None or des_b is None:
        return 0, 0, 0.0

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        des_a,
        des_b,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.78 * n.distance:
            good.append(m)

    if len(good) < 4:
        return len(good), 0, 0.0

    src = np.float32([
        kp_a[m.queryIdx].pt
        for m in good
    ])

    dst = np.float32([
        kp_b[m.trainIdx].pt
        for m in good
    ])

    _, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        5.0
    )

    verified = (
        int(mask.sum())
        if mask is not None
        else 0
    )

    support = (
        verified / len(good) * 100
        if good
        else 0
    )

    return len(good), verified, support


# ------------------------------------------------------------
# SCORE
# ------------------------------------------------------------

def calculate_scores(
    candidate_count,
    verified_count,
    mean_confidence,
    inlier_ratio,
    spatial_coverage,
    reprojection_error,
    sift_support,
    quality
):

    if candidate_count == 0:
        return 0.0, 0.0

    verification_strength = min(
        100.0,
        verified_count / 25.0 * 100.0
    )

    confidence_strength = (
        mean_confidence * 100.0
    )

    if reprojection_error is None:
        reprojection_score = 0.0
    else:
        reprojection_score = (
            np.exp(
                -reprojection_error / 8.0
            ) * 100.0
        )

    geometric_consistency = (
        0.70 * inlier_ratio +
        0.30 * reprojection_score
    )

    geometric_consistency = float(
        np.clip(
            geometric_consistency,
            0,
            100
        )
    )

    learned_score = (
        0.25 * verification_strength +
        0.20 * confidence_strength +
        0.25 * inlier_ratio +
        0.15 * spatial_coverage +
        0.15 * geometric_consistency
    )

    final_score = (
        0.55 * learned_score +
        0.15 * sift_support +
        0.15 * geometric_consistency +
        0.10 * spatial_coverage +
        0.05 * quality
    )

    if verified_count < 8:
        final_score *= 0.90

    if verified_count >= 15 and sift_support >= 25:
        final_score += 5

    final_score = float(
        np.clip(
            final_score,
            0,
            100
        )
    )

    return final_score, geometric_consistency


# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------

def classify_result(
    score,
    verified,
    geometry,
    spatial
):

    if (
        score >= 65 and
        verified >= 25 and
        geometry >= 55 and
        spatial >= 25
    ):
        return "HIGH", "RELIABLE CORRESPONDENCE"

    if (
        score >= 45 and
        verified >= 12 and
        geometry >= 35 and
        spatial >= 15
    ):
        return "MEDIUM", "MODERATE CORRESPONDENCE EVIDENCE"

    return "LOW", "NO RELIABLE CORRESPONDENCE"


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

def create_visualization(
    image_a,
    image_b,
    k0,
    k1,
    mask,
    model
):

    left = image_a.copy()
    right = image_b.copy()

    h = max(
        left.shape[0],
        right.shape[0]
    )

    def pad(img, target_h):

        if img.shape[0] >= target_h:
            return img

        return cv2.copyMakeBorder(
            img,
            0,
            target_h - img.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(20, 20, 20)
        )

    left = pad(left, h)
    right = pad(right, h)

    canvas = np.hstack([
        left,
        right
    ])

    offset = left.shape[1]

    for i, verified in enumerate(mask):

        if not verified:
            continue

        x1, y1 = k0[i]
        x2, y2 = k1[i]

        p1 = (
            int(x1),
            int(y1)
        )

        p2 = (
            int(x2 + offset),
            int(y2)
        )

        cv2.circle(
            canvas,
            p1,
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            canvas,
            p2,
            5,
            (0, 255, 0),
            -1
        )

        cv2.line(
            canvas,
            p1,
            p2,
            (0, 255, 0),
            2
        )

    cv2.putText(
        canvas,
        f"VERIFIED CORRESPONDENCES: {int(mask.sum())}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        model,
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    filename = (
        "correspondence_" +
        uuid.uuid4().hex +
        ".jpg"
    )

    path = os.path.join(
        RESULT_DIR,
        filename
    )

    cv2.imwrite(
        path,
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    return "/results/" + filename


# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------

def build_interpretation(
    score,
    confidence,
    candidates,
    verified,
    geometry,
    model
):

    if confidence == "HIGH":

        return (
            f"The engine identified strong correspondence evidence "
            f"between the two lunar observations. {candidates} learned "
            f"candidate correspondences were detected, with {verified} "
            f"surviving robust {model} verification. The resulting "
            f"evidence score is {score:.1f}/100. Multiple independent "
            f"signals support the correspondence, resulting in High "
            f"Confidence."
        )

    if confidence == "MEDIUM":

        return (
            f"The engine detected meaningful correspondence evidence "
            f"between the two lunar observations. {candidates} learned "
            f"candidate correspondences were identified, of which "
            f"{verified} survived robust {model} verification. The "
            f"measured evidence score is {score:.1f}/100. The result "
            f"supports a possible correspondence, but additional "
            f"validation would be required for reliable geographic "
            f"confirmation."
        )

    return (
        f"The engine detected candidate visual correspondences, but "
        f"the available geometric evidence is insufficient to establish "
        f"a reliable geographic correspondence. {candidates} learned "
        f"candidate correspondences were identified, with {verified} "
        f"surviving robust {model} verification. The measured evidence "
        f"score is {score:.1f}/100, therefore the system assigns Low "
        f"Confidence. This score represents correspondence evidence, "
        f"not a probability that the images depict the same geographic "
        f"location."
    )


# ------------------------------------------------------------
# MATCH API
# ------------------------------------------------------------

@app.route("/api/match", methods=["POST"])
def match():

    start = time.time()

    try:

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

        if not file_a or not file_b:
            return jsonify({
                "error": True,
                "message": "Both lunar images are required."
            }), 400

        if not allowed_file(file_a.filename):
            return jsonify({
                "error": True,
                "message": "Unsupported format for Image A."
            }), 400

        if not allowed_file(file_b.filename):
            return jsonify({
                "error": True,
                "message": "Unsupported format for Image B."
            }), 400

        data_a = np.frombuffer(
            file_a.read(),
            np.uint8
        )

        data_b = np.frombuffer(
            file_b.read(),
            np.uint8
        )

        image_a = cv2.imdecode(
            data_a,
            cv2.IMREAD_COLOR
        )

        image_b = cv2.imdecode(
            data_b,
            cv2.IMREAD_COLOR
        )

        if image_a is None or image_b is None:
            raise ValueError(
                "One or both images could not be decoded."
            )

        image_a = resize_for_processing(image_a)
        image_b = resize_for_processing(image_b)

        gray_a = normalize_image(image_a)
        gray_b = normalize_image(image_b)

        quality_a = calculate_image_quality(gray_a)
        quality_b = calculate_image_quality(gray_b)

        # Learned correspondence
        k0, k1, confidence = run_loftr(
            gray_a,
            gray_b
        )

        candidates = len(k0)

        mean_confidence = (
            float(np.mean(confidence))
            if candidates
            else 0.0
        )

        geometry_result = verify_geometry(
            np.asarray(k0),
            np.asarray(k1)
        )

        verified = geometry_result["verified"]
        inlier_ratio = geometry_result["inlier_ratio"]
        reprojection_error = geometry_result["reprojection_error"]
        model = geometry_result["model"]
        mask = geometry_result["mask"]

        h, w = gray_a.shape

        spatial = calculate_spatial_coverage(
            np.asarray(k0)[mask]
            if candidates and mask.any()
            else np.empty((0, 2)),
            w,
            h
        )

        # Classical supporting branch
        sift_candidates, sift_verified, sift_support = run_sift(
            gray_a,
            gray_b
        )

        overall_quality = (
            quality_a["quality_score"] +
            quality_b["quality_score"]
        ) / 2.0

        final_score, geometric_consistency = calculate_scores(
            candidates,
            verified,
            mean_confidence,
            inlier_ratio,
            spatial,
            reprojection_error,
            sift_support,
            overall_quality
        )

        confidence_level, decision = classify_result(
            final_score,
            verified,
            geometric_consistency,
            spatial
        )

        visualization = create_visualization(
            image_a,
            image_b,
            np.asarray(k0),
            np.asarray(k1),
            mask,
            model
        )

        interpretation = build_interpretation(
            final_score,
            confidence_level,
            candidates,
            verified,
            geometric_consistency,
            model
        )

        elapsed = round(
            time.time() - start,
            2
        )

        return jsonify({

            "success": True,

            "version": "5.3",

            "engine":
                "Hybrid LoFTR + Multi-view SIFT",

            "visualization_engine":
                "Hybrid LoFTR + Multi-view SIFT",

            "device": DEVICE,

            "match_percentage":
                round(final_score, 1),

            "evidence_score":
                round(final_score, 1),

            "confidence":
                confidence_level,

            "decision":
                decision,

            "candidate_matches":
                candidates,

            "loftr_candidate_matches":
                candidates,

            "verified_matches":
                verified,

            "loftr_verified_matches":
                verified,

            "sift_candidate_matches":
                sift_candidates,

            "sift_verified_matches":
                sift_verified,

            "sift_support_score":
                round(sift_support, 1),

            "learned_score":
                round(final_score, 1),

            "inlier_ratio":
                round(inlier_ratio, 1),

            "geometric_consistency":
                round(geometric_consistency, 1),

            "spatial_coverage":
                round(spatial, 1),

            "reprojection_error":
                (
                    round(reprojection_error, 2)
                    if reprojection_error is not None
                    else None
                ),

            "geometry_model":
                model,

            "image_quality_a":
                quality_a,

            "image_quality_b":
                quality_b,

            "feature_count_a":
                max(
                    sift_candidates,
                    candidates
                ),

            "feature_count_b":
                max(
                    sift_candidates,
                    candidates
                ),

            "visualization":
                visualization,

            "interpretation":
                interpretation,

            "pipeline": [
                {
                    "step": 1,
                    "name": "ACQUIRE",
                    "detail": "Lunar image input"
                },
                {
                    "step": 2,
                    "name": "NORMALIZE",
                    "detail": "Grayscale + contrast normalization"
                },
                {
                    "step": 3,
                    "name": "LEARNED MATCH",
                    "detail": "LoFTR correspondence extraction"
                },
                {
                    "step": 4,
                    "name": "GEOMETRIC VERIFY",
                    "detail": "Affine / Homography + RANSAC"
                },
                {
                    "step": 5,
                    "name": "CROSS-CHECK",
                    "detail": "Multi-view SIFT support"
                },
                {
                    "step": 6,
                    "name": "EVIDENCE FUSION",
                    "detail": "Multi-signal correspondence scoring"
                },
                {
                    "step": 7,
                    "name": "REPORT",
                    "detail": "Metrics + visualization"
                }
            ],

            "processing_time":
                elapsed,

            "process_time":
                elapsed

        })

    except Exception as e:

        print("\nLUNARMATCH ERROR")
        print(traceback.format_exc())

        return jsonify({
            "error": True,
            "message": str(e)
        }), 500


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(
        RESULT_DIR,
        filename
    )


@app.route("/health")
def health():

    return jsonify({
        "status": "online",
        "engine":
            "Hybrid LoFTR + Multi-view SIFT",
        "version": "5.3",
        "device": DEVICE,
        "loftr":
            "READY" if LOFTR is not None else "UNAVAILABLE"
    })


if __name__ == "__main__":

    print()
    print("=" * 65)
    print("LUNARMATCH SERVER")
    print("=" * 65)
    print("Engine  : Hybrid LoFTR + Multi-view SIFT")
    print("Version : 5.3")
    print("Device  :", DEVICE)
    print(
        "LoFTR   :",
        "READY" if LOFTR is not None else "UNAVAILABLE"
    )
    print("URL     : http://127.0.0.1:5000")
    print("=" * 65)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
    