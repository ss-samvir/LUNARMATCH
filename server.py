import os
from flask import Flask, send_from_directory, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    static_url_path=""
)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "online",
        "service": "LUNARMATCH",
        "version": "5.3-browser-engine",
        "engine": "Browser OpenCV.js ORB + BFMatcher + RANSAC",
        "analysis_location": "client"
    })


@app.route("/<path:path>")
def static_files(path):
    file_path = os.path.join(BASE_DIR, path)

    if os.path.isfile(file_path):
        return send_from_directory(BASE_DIR, path)

    return send_from_directory(BASE_DIR, "index.html")


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
