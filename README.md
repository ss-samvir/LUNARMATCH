# LUNARMATCH — Website + Real Python Backend

This package connects the LUNARMATCH frontend to a Flask API implementing the
same core correspondence approach as the supplied SIH repository:
OpenCV SIFT feature extraction + BFMatcher + Lowe's ratio test.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the server:
   `python server.py`
5. Open:
   `http://127.0.0.1:5000`

Upload two lunar images and press **COMPARE IMAGES**.

## Important

The percentage shown by this web version is a deterministic correspondence
score derived from the number of Lowe-ratio-test good matches relative to the
smaller keypoint set. It is NOT a trained-model probability.

The original repository's `src/matching.py` returns images, keypoints and
good matches but does not itself expose a web API or calculate a percentage.
The included server therefore adds the web/API layer needed for deployment.

For public hosting, deploy the Python server on a service that supports
Python/Flask. A static-only host such as GitHub Pages cannot execute this
Python backend.
