import os
from flask import Flask, send_file, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.route("/")
def home():
    return send_file(os.path.join(BASE_DIR, "index.html"))


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "service": "LUNARMATCH",
        "version": "5.3-browser-engine",
        "engine": "Browser OpenCV.js ORB + BFMatcher + RANSAC",
        "analysis_location": "client"
    })


@app.route("/<path:filename>")
def files(filename):
    file_path = os.path.join(BASE_DIR, filename)

    if os.path.isfile(file_path):
        return send_file(file_path)

    return jsonify({
        "error": "File not found"
    }), 404


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Uploaded file is too large. Maximum size is 25 MB."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
