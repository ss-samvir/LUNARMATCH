import base64
import os
import tempfile
import time

import cv2
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# Keep requests reasonable for a free hosting instance.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
MAX_DIMENSION = 1600


def encode_jpeg(image):
    ok, buffer = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not ok:
        raise RuntimeError("Could not encode visualization.")
    return base64.b64encode(buffer).decode("ascii")


def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read one of the uploaded images.")

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def run_matching(path1, path2):
    img1 = prepare_image(path1)
    img2 = prepare_image(path2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        return {
            "match_found": False,
            "match_percentage": 0.0,
            "corresponding_features": 0,
            "confidence": "Low",
            "quality": "Insufficient features",
            "visualization": encode_jpeg(vis),
        }

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    denominator = max(1, min(len(kp1), len(kp2)))
    score = min(100.0, (len(good) / denominator) * 100.0)

    if len(good) >= 30 and score >= 15:
        confidence = "High"
        quality = "Excellent"
    elif len(good) >= 10 and score >= 5:
        confidence = "Medium"
        quality = "Good"
    else:
        confidence = "Low"
        quality = "Limited"

    vis = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return {
        "match_found": len(good) >= 10,
        "match_percentage": score,
        "corresponding_features": len(good),
        "confidence": confidence,
        "quality": quality,
        "visualization": encode_jpeg(vis),
    }


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": "Images are too large. Please use images smaller than 25 MB total."
    }), 413


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/match")
def api_match():
    start = time.perf_counter()

    f1 = request.files.get("image1")
    f2 = request.files.get("image2")

    if not f1 or not f2:
        return jsonify({"error": "Please provide Image A and Image B."}), 400

    suffix1 = os.path.splitext(secure_filename(f1.filename or ""))[1].lower()
    suffix2 = os.path.splitext(secure_filename(f2.filename or ""))[1].lower()

    if suffix1 not in ALLOWED or suffix2 not in ALLOWED:
        return jsonify({
            "error": "Unsupported image format. Use JPG, JPEG, PNG, TIFF or WEBP."
        }), 400

    temp_paths = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as temp1:
            f1.save(temp1.name)
            temp_paths.append(temp1.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as temp2:
            f2.save(temp2.name)
            temp_paths.append(temp2.name)

        result = run_matching(temp_paths[0], temp_paths[1])
        result["processing_time"] = round(time.perf_counter() - start, 3)
        return jsonify(result)

    except Exception as exc:
        app.logger.exception("Image matching failed")
        return jsonify({"error": f"Image analysis failed: {exc}"}), 500

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    # Works both locally and on Render.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
