import base64
import os
import tempfile
import time

import cv2
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

def encode_jpeg(image):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Could not encode visualization.")
    return base64.b64encode(buffer).decode("ascii")

def run_matching(path1, path2):
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images.")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return {
            "match_found": False, "match_percentage": 0.0,
            "corresponding_features": 0, "confidence": "Low",
            "quality": "Insufficient features",
            "visualization": encode_jpeg(cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR))
        }

    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    # A transparent, deterministic score derived from the repository's
    # good-match count relative to the available keypoints.
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

    # Show actual SIFT correspondences returned by the backend.
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, good[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "match_found": len(good) >= 10,
        "match_percentage": score,
        "corresponding_features": len(good),
        "confidence": confidence,
        "quality": quality,
        "processing_time": 0.0,
        "visualization": encode_jpeg(vis)
    }

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.post("/api/match")
def api_match():
    start = time.perf_counter()
    f1 = request.files.get("image1")
    f2 = request.files.get("image2")
    if not f1 or not f2:
        return jsonify({"error": "Please provide Image A and Image B."}), 400

    suffix1 = os.path.splitext(secure_filename(f1.filename))[1].lower()
    suffix2 = os.path.splitext(secure_filename(f2.filename))[1].lower()
    if suffix1 not in ALLOWED or suffix2 not in ALLOWED:
        return jsonify({"error": "Unsupported image format."}), 400

    temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=suffix1)
    temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=suffix2)
    try:
        f1.save(temp1.name)
        f2.save(temp2.name)
        result = run_matching(temp1.name, temp2.name)
        result["processing_time"] = time.perf_counter() - start
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        temp1.close(); temp2.close()
        for p in (temp1.name, temp2.name):
            try: os.remove(p)
            except OSError: pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
