/* ============================================================
   LUNARMATCH
   Lunar Image Correspondence System
   Browser Analysis Engine — Professional Prototype
   ============================================================ */

(() => {
    "use strict";

    /* ---------------------------------------------------------
       GLOBAL STATE
    --------------------------------------------------------- */

    let imageAFile = null;
    let imageBFile = null;
    let imageAData = null;
    let imageBData = null;
    let lastAnalysis = null;

    let cvReadyPromise = null;
    let pdfReadyPromise = null;

    const MAX_IMAGE_DIMENSION = 1100;
    const ORB_FEATURES = 4000;
    const LOWE_RATIO = 0.78;
    const MAX_VISUAL_MATCHES = 80;

    /* ---------------------------------------------------------
       DOM HELPERS
    --------------------------------------------------------- */

    const $ = (id) => document.getElementById(id);

    function setText(id, value) {
        const el = $(id);
        if (el) el.textContent = value;
    }

    function show(id, visible = true) {
        const el = $(id);
        if (!el) return;

        if (visible) {
            el.style.display = "";
            el.removeAttribute("hidden");
        } else {
            el.style.display = "none";
        }
    }

    function setDisabled(id, disabled) {
        const el = $(id);
        if (el) el.disabled = disabled;
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function round(value, decimals = 1) {
        const p = Math.pow(10, decimals);
        return Math.round(value * p) / p;
    }

    function formatMs(ms) {
        if (!Number.isFinite(ms)) return "—";
        return `${round(ms / 1000, 2)} sec`;
    }

    /* ---------------------------------------------------------
       RESULT RESET
    --------------------------------------------------------- */

    function resetResults() {
        setText("status", "READY");
        setText("score", "—");
        setText("features", "—");
        setText("confidence", "—");
        setText("quality", "—");
        setText("time", "—");

        const ids = [
            "resolutionA",
            "keypointsA",
            "contrastA",
            "sharpnessA",
            "qualityScoreA",

            "resolutionB",
            "keypointsB",
            "contrastB",
            "sharpnessB",
            "qualityScoreB",

            "rawMatches",
            "candidateMatches",
            "verifiedMatches",
            "featureCoverage",
            "correspondenceStrength",

            "inlierRatio",
            "geometricConsistency",
            "homographyStatus",
            "verificationStatus"
        ];

        ids.forEach(id => setText(id, "—"));

        setText("interpretation",
            "Upload two lunar images to begin correspondence analysis."
        );

        const visual = $("correspondenceMap");

        if (visual) {
            visual.removeAttribute("src");
            visual.style.display = "none";
        }

        show("visualPlaceholder", true);

        const note = $("visualNote");
        if (note) {
            note.textContent =
                "Verified correspondences will appear here after analysis.";
        }

        resetPipeline();

        setDisabled("downloadReportBtn", true);

        lastAnalysis = null;
    }

    /* ---------------------------------------------------------
       PIPELINE
    --------------------------------------------------------- */

    const pipelineStages = [
        ["stageAcquire", "ACQUIRE"],
        ["stagePreprocess", "PREPROCESS"],
        ["stageExtract", "EXTRACT"],
        ["stageMatch", "MATCH"],
        ["stageVerify", "VERIFY"],
        ["stageScore", "SCORE"],
        ["stageReport", "REPORT"]
    ];

    function resetPipeline() {
        pipelineStages.forEach(([id]) => {
            const el = $(id);
            if (!el) return;

            el.classList.remove("active", "complete", "error");
        });
    }

    function pipelineActive(index) {
        pipelineStages.forEach(([id], i) => {
            const el = $(id);
            if (!el) return;

            el.classList.remove("active", "complete", "error");

            if (i < index) el.classList.add("complete");
            if (i === index) el.classList.add("active");
        });
    }

    function pipelineComplete() {
        pipelineStages.forEach(([id]) => {
            const el = $(id);
            if (!el) return;

            el.classList.remove("active", "error");
            el.classList.add("complete");
        });
    }

    function pipelineError(index) {
        const [id] = pipelineStages[index] || [];
        const el = $(id);

        if (el) {
            el.classList.remove("active");
            el.classList.add("error");
        }
    }

    /* ---------------------------------------------------------
       IMAGE VALIDATION
    --------------------------------------------------------- */

    function isImageFile(file) {
        if (!file) return false;

        if (file.type && file.type.startsWith("image/")) {
            return true;
        }

        const name = file.name.toLowerCase();

        return /\.(jpg|jpeg|png|webp|tif|tiff|bmp)$/i.test(name);
    }

    function validateFile(file) {
        if (!file) {
            throw new Error("No image selected.");
        }

        if (!isImageFile(file)) {
            throw new Error(
                "Unsupported image format. Use JPG, JPEG, PNG, WEBP, TIFF or BMP."
            );
        }

        if (file.size > 25 * 1024 * 1024) {
            throw new Error("Image exceeds the 25 MB upload limit.");
        }

        return true;
    }

    /* ---------------------------------------------------------
       IMAGE PREVIEW
    --------------------------------------------------------- */

    function showPreview(file, previewId) {
        const preview = $(previewId);
        if (!preview) return;

        const url = URL.createObjectURL(file);

        preview.src = url;
        preview.style.display = "block";

        preview.onload = () => {
            URL.revokeObjectURL(url);
        };
    }

    /* ---------------------------------------------------------
       FILE READER
    --------------------------------------------------------- */

    function fileToDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(
                new Error("Unable to read the selected image.")
            );

            reader.readAsDataURL(file);
        });
    }

    function loadImageElement(source) {
        return new Promise((resolve, reject) => {
            const img = new Image();

            img.onload = () => resolve(img);

            img.onerror = () => reject(
                new Error("The selected image could not be decoded.")
            );

            img.src = source;
        });
    }

    /* ---------------------------------------------------------
       CANVAS IMAGE NORMALIZATION
    --------------------------------------------------------- */

    async function imageToMat(file) {
        const dataURL = await fileToDataURL(file);
        const img = await loadImageElement(dataURL);

        let width = img.naturalWidth || img.width;
        let height = img.naturalHeight || img.height;

        const scale = Math.min(
            1,
            MAX_IMAGE_DIMENSION / Math.max(width, height)
        );

        width = Math.max(1, Math.round(width * scale));
        height = Math.max(1, Math.round(height * scale));

        const canvas = document.createElement("canvas");

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d", {
            willReadFrequently: true
        });

        ctx.drawImage(img, 0, 0, width, height);

        return {
            mat: cv.imread(canvas),
            width,
            height,
            originalWidth: img.naturalWidth || img.width,
            originalHeight: img.naturalHeight || img.height
        };
    }

    /* ---------------------------------------------------------
       OPENCV LOADER
    --------------------------------------------------------- */

    function waitForOpenCV() {
        if (window.cv && typeof window.cv.Mat === "function") {
            return Promise.resolve(window.cv);
        }

        if (cvReadyPromise) return cvReadyPromise;

        cvReadyPromise = new Promise((resolve, reject) => {
            const existing = document.querySelector(
                'script[data-lunarmatch-opencv="true"]'
            );

            if (existing) {
                const timer = setInterval(() => {
                    if (
                        window.cv &&
                        typeof window.cv.Mat === "function"
                    ) {
                        clearInterval(timer);
                        resolve(window.cv);
                    }
                }, 100);

                setTimeout(() => {
                    clearInterval(timer);
                    reject(
                        new Error(
                            "OpenCV.js failed to initialize."
                        )
                    );
                }, 30000);

                return;
            }

            const script = document.createElement("script");

            script.src = "https://docs.opencv.org/4.x/opencv.js";
            script.async = true;
            script.dataset.lunarmatchOpencv = "true";

            script.onload = () => {
                const timer = setInterval(() => {
                    if (
                        window.cv &&
                        typeof window.cv.Mat === "function"
                    ) {
                        clearInterval(timer);
                        resolve(window.cv);
                    }
                }, 100);

                setTimeout(() => {
                    clearInterval(timer);
                    reject(
                        new Error(
                            "OpenCV.js loaded but did not initialize."
                        )
                    );
                }, 30000);
            };

            script.onerror = () => {
                reject(
                    new Error(
                        "Unable to load OpenCV.js. Check the internet connection."
                    )
                );
            };

            document.head.appendChild(script);
        });

        return cvReadyPromise;
    }

    /* ---------------------------------------------------------
       GRAYSCALE
    --------------------------------------------------------- */

    function toGray(mat) {
        const gray = new cv.Mat();

        if (mat.channels() === 1) {
            mat.copyTo(gray);
        } else if (mat.channels() === 4) {
            cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
        } else {
            cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY);
        }

        return gray;
    }

    /* ---------------------------------------------------------
       CLAHE
    --------------------------------------------------------- */

    function enhanceGray(gray) {
        const enhanced = new cv.Mat();

        try {
            if (typeof cv.createCLAHE === "function") {
                const clahe = cv.createCLAHE(2.0, new cv.Size(8, 8));

                clahe.apply(gray, enhanced);

                if (clahe.delete) clahe.delete();

                return enhanced;
            }
        } catch (error) {
            // fallback below
        }

        try {
            cv.equalizeHist(gray, enhanced);
            return enhanced;
        } catch (error) {
            gray.copyTo(enhanced);
            return enhanced;
        }
    }

    /* ---------------------------------------------------------
       IMAGE QUALITY
    --------------------------------------------------------- */

    function calculateImageQuality(mat) {
        const gray = toGray(mat);

        const mean = new cv.Mat();
        const stddev = new cv.Mat();

        let contrast = 0;
        let sharpness = 0;

        try {
            cv.meanStdDev(gray, mean, stddev);

            const values = stddev.data64F || stddev.data32F;

            if (values && values.length) {
                contrast = Number(values[0]) || 0;
            }
        } catch (error) {
            contrast = 0;
        }

        try {
            const lap = new cv.Mat();

            cv.Laplacian(
                gray,
                lap,
                cv.CV_64F
            );

            const lapMean = new cv.Mat();
            const lapStd = new cv.Mat();

            cv.meanStdDev(
                lap,
                lapMean,
                lapStd
            );

            const values = lapStd.data64F || lapStd.data32F;

            if (values && values.length) {
                sharpness = Math.pow(
                    Number(values[0]) || 0,
                    2
                );
            }

            lap.delete();
            lapMean.delete();
            lapStd.delete();
        } catch (error) {
            sharpness = 0;
        }

        mean.delete();
        stddev.delete();
        gray.delete();

        const contrastScore = clamp(
            (contrast / 64) * 100,
            0,
            100
        );

        const sharpnessScore = clamp(
            (Math.log10(sharpness + 1) / 4.5) * 100,
            0,
            100
        );

        const qualityScore = clamp(
            contrastScore * 0.45 +
            sharpnessScore * 0.55,
            0,
            100
        );

        let label = "POOR";

        if (qualityScore >= 75) {
            label = "EXCELLENT";
        } else if (qualityScore >= 58) {
            label = "GOOD";
        } else if (qualityScore >= 40) {
            label = "FAIR";
        }

        return {
            contrast,
            sharpness,
            contrastScore,
            sharpnessScore,
            score: qualityScore,
            label
        };
    }

    /* ---------------------------------------------------------
       ORB CREATION
    --------------------------------------------------------- */

    function createORB() {
        if (typeof cv.ORB_create === "function") {
            return cv.ORB_create(
                ORB_FEATURES,
                1.2,
                8,
                31,
                0,
                2,
                0,
                31,
                20
            );
        }

        if (typeof cv.ORB === "function") {
            return new cv.ORB(
                ORB_FEATURES,
                1.2,
                8,
                31,
                0,
                2,
                0,
                31,
                20
            );
        }

        throw new Error("ORB is unavailable in this OpenCV.js build.");
    }

    /* ---------------------------------------------------------
       FEATURE EXTRACTION
    --------------------------------------------------------- */

    function extractFeatures(mat) {
        const gray = toGray(mat);
        const enhanced = enhanceGray(gray);

        const keypoints = new cv.KeyPointVector();
        const descriptors = new cv.Mat();

        const orb = createORB();

        try {
            orb.detectAndCompute(
                enhanced,
                new cv.Mat(),
                keypoints,
                descriptors,
                false
            );
        } catch (error) {
            if (orb.delete) orb.delete();
            gray.delete();
            enhanced.delete();
            keypoints.delete();
            descriptors.delete();

            throw error;
        }

        if (orb.delete) orb.delete();

        gray.delete();
        enhanced.delete();

        return {
            keypoints,
            descriptors,
            count: keypoints.size()
        };
    }

    /* ---------------------------------------------------------
       KEYPOINT CONVERSION
    --------------------------------------------------------- */

    function keypointXY(keypoint) {
        if (!keypoint) return { x: 0, y: 0 };

        if (typeof keypoint.pt !== "undefined") {
            return {
                x: Number(keypoint.pt.x),
                y: Number(keypoint.pt.y)
            };
        }

        if (
            typeof keypoint.pt_x !== "undefined" &&
            typeof keypoint.pt_y !== "undefined"
        ) {
            return {
                x: Number(keypoint.pt_x),
                y: Number(keypoint.pt_y)
            };
        }

        return {
            x: 0,
            y: 0
        };
    }

    /* ---------------------------------------------------------
       MATCHING
    --------------------------------------------------------- */

    function matchDescriptors(descA, descB) {
        if (
            !descA ||
            !descB ||
            descA.rows < 2 ||
            descB.rows < 2
        ) {
            return {
                rawMatches: 0,
                candidates: []
            };
        }

        const matcher = new cv.BFMatcher(
            cv.NORM_HAMMING,
            false
        );

        let knn = null;

        try {
            knn = new cv.DMatchVectorVector();

            matcher.knnMatch(
                descA,
                descB,
                knn,
                2
            );

            const candidates = [];

            for (let i = 0; i < knn.size(); i++) {
                const pair = knn.get(i);

                if (!pair || pair.size() < 2) {
                    if (pair && pair.delete) pair.delete();
                    continue;
                }

                const first = pair.get(0);
                const second = pair.get(1);

                const d1 = Number(first.distance);
                const d2 = Number(second.distance);

                if (
                    Number.isFinite(d1) &&
                    Number.isFinite(d2) &&
                    d2 > 0 &&
                    d1 < LOWE_RATIO * d2
                ) {
                    candidates.push({
                        queryIdx: Number(first.queryIdx),
                        trainIdx: Number(first.trainIdx),
                        distance: d1,
                        ratio: d1 / d2
                    });
                }

                if (pair.delete) pair.delete();
            }

            if (matcher.delete) matcher.delete();
            if (knn.delete) knn.delete();

            candidates.sort(
                (a, b) => a.distance - b.distance
            );

            return {
                rawMatches: Math.min(
                    descA.rows,
                    descB.rows
                ),
                candidates
            };
        } catch (error) {
            if (matcher.delete) matcher.delete();
            if (knn && knn.delete) knn.delete();

            throw error;
        }
    }

    /* ---------------------------------------------------------
       REMOVE DUPLICATE CORRESPONDENCES
    --------------------------------------------------------- */

    function deduplicateMatches(matches) {
        const usedA = new Set();
        const usedB = new Set();

        const result = [];

        for (const match of matches) {
            if (
                usedA.has(match.queryIdx) ||
                usedB.has(match.trainIdx)
            ) {
                continue;
            }

            usedA.add(match.queryIdx);
            usedB.add(match.trainIdx);

            result.push(match);
        }

        return result;
    }

    /* ---------------------------------------------------------
       GEOMETRIC VERIFICATION
    --------------------------------------------------------- */

    function maskValue(mask, index) {
        if (!mask) return 0;

        try {
            if (mask.rows === 1) {
                return mask.ucharAt(0, index);
            }

            return mask.ucharAt(index, 0);
        } catch (error) {
            return 0;
        }
    }

    function verifyGeometry(
        keypointsA,
        keypointsB,
        matches
    ) {
        if (matches.length < 4) {
            return {
                verified: [],
                inlierRatio: 0,
                geometricConsistency: 0,
                homographyFound: false
            };
        }

        const src = [];
        const dst = [];

        for (const match of matches) {
            const a = keypointXY(
                keypointsA.get(match.queryIdx)
            );

            const b = keypointXY(
                keypointsB.get(match.trainIdx)
            );

            src.push(a.x, a.y);
            dst.push(b.x, b.y);
        }

        const srcMat = cv.matFromArray(
            matches.length,
            2,
            cv.CV_32FC2,
            src
        );

        const dstMat = cv.matFromArray(
            matches.length,
            2,
            cv.CV_32FC2,
            dst
        );

        let H = null;
        let mask = null;

        try {
            H = cv.findHomography(
                srcMat,
                dstMat,
                cv.RANSAC,
                5.0
            );

            if (H && typeof H === "object" && H.mask) {
                mask = H.mask;
                H = H.homography || H.H || null;
            }

            if (!mask && H && H.rows === 3 && H.cols === 3) {
                // Some OpenCV.js builds return the mask separately
                // through the API internals; affine fallback below.
            }
        } catch (error) {
            H = null;
        }

        let verified = [];

        if (mask) {
            for (let i = 0; i < matches.length; i++) {
                if (maskValue(mask, i) !== 0) {
                    verified.push(matches[i]);
                }
            }
        }

        /* -----------------------------------------------------
           AFFINE FALLBACK
        ----------------------------------------------------- */

        if (
            verified.length < 4 &&
            typeof cv.estimateAffinePartial2D === "function"
        ) {
            try {
                const affineMask = new cv.Mat();

                const affine = cv.estimateAffinePartial2D(
                    srcMat,
                    dstMat,
                    affineMask,
                    cv.RANSAC,
                    5.0,
                    2000,
                    0.99,
                    10
                );

                if (affine && !affine.empty()) {
                    const affineVerified = [];

                    for (
                        let i = 0;
                        i < matches.length;
                        i++
                    ) {
                        if (
                            maskValue(
                                affineMask,
                                i
                            ) !== 0
                        ) {
                            affineVerified.push(
                                matches[i]
                            );
                        }
                    }

                    if (
                        affineVerified.length >
                        verified.length
                    ) {
                        verified = affineVerified;
                    }
                }

                if (affine && affine.delete) {
                    affine.delete();
                }

                affineMask.delete();
            } catch (error) {
                // Continue with homography result.
            }
        }

        const inlierRatio =
            matches.length > 0
                ? verified.length / matches.length
                : 0;

        const geometricConsistency = clamp(
            inlierRatio * 100,
            0,
            100
        );

        if (mask && mask.delete) mask.delete();
        if (H && H.delete) H.delete();

        srcMat.delete();
        dstMat.delete();

        return {
            verified,
            inlierRatio,
            geometricConsistency,
            homographyFound: verified.length >= 4
        };
    }

    /* ---------------------------------------------------------
       SPATIAL COVERAGE
    --------------------------------------------------------- */

    function calculateSpatialCoverage(
        keypoints,
        verifiedMatches,
        width,
        height
    ) {
        if (
            !verifiedMatches.length ||
            !width ||
            !height
        ) {
            return 0;
        }

        const occupied = new Set();

        for (const match of verifiedMatches) {
            const p = keypointXY(
                keypoints.get(match.queryIdx)
            );

            const x = clamp(
                Math.floor((p.x / width) * 4),
                0,
                3
            );

            const y = clamp(
                Math.floor((p.y / height) * 4),
                0,
                3
            );

            occupied.add(`${x}:${y}`);
        }

        return clamp(
            (occupied.size / 16) * 100,
            0,
            100
        );
    }

    /* ---------------------------------------------------------
       CORRESPONDENCE STRENGTH
    --------------------------------------------------------- */

    function calculateCorrespondenceStrength(
        matches,
        verifiedMatches
    ) {
        if (!matches.length) return 0;

        const averageDistance =
            matches.reduce(
                (sum, match) =>
                    sum + match.distance,
                0
            ) / matches.length;

        const distanceScore = clamp(
            100 - (averageDistance / 128) * 100,
            0,
            100
        );

        const verificationScore =
            verifiedMatches.length /
            matches.length *
            100;

        return clamp(
            distanceScore * 0.45 +
            verificationScore * 0.55,
            0,
            100
        );
    }

    /* ---------------------------------------------------------
       SCORE
    --------------------------------------------------------- */

    function calculateScore({
        candidates,
        verified,
        coverage,
        geometry,
        qualityA,
        qualityB
    }) {
        const verifiedCountScore = clamp(
            (verified.length / 80) * 100,
            0,
            100
        );

        const candidateScore = clamp(
            (candidates.length / 150) * 100,
            0,
            100
        );

        const qualityScore =
            (qualityA.score + qualityB.score) / 2;

        const correspondenceStrength =
            calculateCorrespondenceStrength(
                candidates,
                verified
            );

        const score =
            correspondenceStrength * 0.30 +
            verifiedCountScore * 0.20 +
            geometry * 0.25 +
            coverage * 0.10 +
            candidateScore * 0.10 +
            qualityScore * 0.05;

        return {
            score: clamp(score, 0, 100),
            correspondenceStrength
        };
    }

    /* ---------------------------------------------------------
       CLASSIFICATION
    --------------------------------------------------------- */

    function classifyMatch({
        score,
        verified,
        geometry,
        coverage
    }) {
        if (
            score >= 65 &&
            verified >= 25 &&
            geometry >= 55 &&
            coverage >= 25
        ) {
            return {
                decision: "HIGH",
                status: "MATCH FOUND",
                confidence: "HIGH"
            };
        }

        if (
            score >= 45 &&
            verified >= 12 &&
            geometry >= 35 &&
            coverage >= 15
        ) {
            return {
                decision: "MEDIUM",
                status: "POSSIBLE MATCH",
                confidence: "MEDIUM"
            };
        }

        return {
            decision: "LOW",
            status: "NO STRONG MATCH",
            confidence: "LOW"
        };
    }

    /* ---------------------------------------------------------
       INTERPRETATION
    --------------------------------------------------------- */

    function generateInterpretation(result) {
        const {
            classification,
            verifiedMatches,
            candidateMatches,
            coverage,
            geometry,
            quality
        } = result;

        const score = result.score;

        if (classification.decision === "HIGH") {
            return (
                `The correspondence analysis indicates a strong spatial ` +
                `relationship between Image A and Image B. ` +
                `${verifiedMatches} geometrically verified correspondences ` +
                `were retained from ${candidateMatches} candidate matches. ` +
                `The overall correspondence score is ${round(score, 1)}%, ` +
                `with ${round(geometry, 1)}% geometric consistency and ` +
                `${round(coverage, 1)}% spatial feature coverage. ` +
                `Both images provide sufficient visual information for ` +
                `a high-confidence correspondence assessment.`
            );
        }

        if (classification.decision === "MEDIUM") {
            return (
                `The analysis identified meaningful but incomplete ` +
                `correspondence between the supplied lunar images. ` +
                `${verifiedMatches} candidate correspondences passed ` +
                `geometric verification. The resulting score of ` +
                `${round(score, 1)}% suggests that the images may depict ` +
                `related terrain, but additional imagery or stronger ` +
                `feature coverage would improve confidence.`
            );
        }

        if (quality === "POOR") {
            return (
                `The analysis did not identify sufficient reliable ` +
                `correspondence. Image quality or feature availability ` +
                `may be limiting the result. Consider using images with ` +
                `higher resolution, stronger surface texture, or less ` +
                `illumination-related degradation.`
            );
        }

        return (
            `The analysis did not identify enough geometrically consistent ` +
            `correspondence to establish a strong match. The current result ` +
            `should be treated as a low-confidence correspondence assessment. ` +
            `Additional imagery may be required for a reliable conclusion.`
        );
    }

    /* ---------------------------------------------------------
       CORRESPONDENCE VISUALIZATION
    --------------------------------------------------------- */

    function createCorrespondenceVisualization(
        matA,
        matB,
        keypointsA,
        keypointsB,
        verifiedMatches
    ) {
        const widthA = matA.cols;
        const heightA = matA.rows;

        const widthB = matB.cols;
        const heightB = matB.rows;

        const height = Math.max(
            heightA,
            heightB
        );

        const width =
            widthA + widthB;

        const canvas = document.createElement("canvas");

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");

        ctx.fillStyle = "#05070c";
        ctx.fillRect(
            0,
            0,
            width,
            height
        );

        const canvasA = document.createElement("canvas");
        canvasA.width = widthA;
        canvasA.height = heightA;

        const canvasB = document.createElement("canvas");
        canvasB.width = widthB;
        canvasB.height = heightB;

        cv.imshow(canvasA, matA);
        cv.imshow(canvasB, matB);

        ctx.drawImage(
            canvasA,
            0,
            0
        );

        ctx.drawImage(
            canvasB,
            widthA,
            0
        );

        const selected =
            verifiedMatches
                .slice()
                .sort(
                    (a, b) =>
                        a.distance -
                        b.distance
                )
                .slice(
                    0,
                    MAX_VISUAL_MATCHES
                );

        ctx.lineWidth = 1;

        for (const match of selected) {
            const pA = keypointXY(
                keypointsA.get(match.queryIdx)
            );

            const pB = keypointXY(
                keypointsB.get(match.trainIdx)
            );

            const hue =
                (match.queryIdx * 37) % 360;

            ctx.strokeStyle =
                `hsl(${hue}, 90%, 68%)`;

            ctx.fillStyle =
                `hsl(${hue}, 90%, 75%)`;

            ctx.beginPath();

            ctx.moveTo(
                pA.x,
                pA.y
            );

            ctx.lineTo(
                widthA + pB.x,
                pB.y
            );

            ctx.stroke();

            ctx.beginPath();

            ctx.arc(
                pA.x,
                pA.y,
                3,
                0,
                Math.PI * 2
            );

            ctx.fill();

            ctx.beginPath();

            ctx.arc(
                widthA + pB.x,
                pB.y,
                3,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }

        return canvas.toDataURL(
            "image/jpeg",
            0.92
        );
    }

    /* ---------------------------------------------------------
       DISPLAY RESULTS
    --------------------------------------------------------- */

    function displayResults(result) {
        setText(
            "status",
            result.classification.status
        );

        setText(
            "score",
            `${round(result.score, 1)}%`
        );

        setText(
            "features",
            result.verifiedMatches
        );

        setText(
            "confidence",
            result.classification.confidence
        );

        setText(
            "quality",
            result.quality
        );

        setText(
            "time",
            formatMs(result.processingTime)
        );

        /* Image A */

        setText(
            "resolutionA",
            `${result.imageA.width} × ${result.imageA.height}`
        );

        setText(
            "keypointsA",
            result.imageA.keypoints
        );

        setText(
            "contrastA",
            round(result.imageA.contrast, 2)
        );

        setText(
            "sharpnessA",
            round(result.imageA.sharpness, 2)
        );

        setText(
            "qualityScoreA",
            `${round(result.imageA.qualityScore, 1)}%`
        );

        /* Image B */

        setText(
            "resolutionB",
            `${result.imageB.width} × ${result.imageB.height}`
        );

        setText(
            "keypointsB",
            result.imageB.keypoints
        );

        setText(
            "contrastB",
            round(result.imageB.contrast, 2)
        );

        setText(
            "sharpnessB",
            round(result.imageB.sharpness, 2)
        );

        setText(
            "qualityScoreB",
            `${round(result.imageB.qualityScore, 1)}%`
        );

        /* Correspondence */

        setText(
            "rawMatches",
            result.rawMatches
        );

        setText(
            "candidateMatches",
            result.candidateMatches
        );

        setText(
            "verifiedMatches",
            result.verifiedMatches
        );

        setText(
            "featureCoverage",
            `${round(result.coverage, 1)}%`
        );

        setText(
            "correspondenceStrength",
            `${round(
                result.correspondenceStrength,
                1
            )}%`
        );

        /* Geometry */

        setText(
            "inlierRatio",
            `${round(
                result.inlierRatio * 100,
                1
            )}%`
        );

        setText(
            "geometricConsistency",
            `${round(
                result.geometricConsistency,
                1
            )}%`
        );

        setText(
            "homographyStatus",
            result.homographyStatus
        );

        setText(
            "verificationStatus",
            result.verificationStatus
        );

        setText(
            "interpretation",
            result.interpretation
        );

        /* Visualization */

        const visual = $("correspondenceMap");

        if (visual && result.visualization) {
            visual.src = result.visualization;
            visual.style.display = "block";
        }

        show("visualPlaceholder", false);

        const note = $("visualNote");

        if (note) {
            note.textContent =
                `${Math.min(
                    result.verifiedMatches,
                    MAX_VISUAL_MATCHES
                )} verified correspondences visualized.`;
        }

        setDisabled(
            "downloadReportBtn",
            false
        );
    }

    /* ---------------------------------------------------------
       MAIN ANALYSIS
    --------------------------------------------------------- */

    async function analyzeImages() {
        if (!imageAFile || !imageBFile) {
            setText(
                "status",
                "UPLOAD BOTH IMAGES"
            );

            setText(
                "interpretation",
                "Please provide Image A and Image B before starting analysis."
            );

            return;
        }

        const startTime = performance.now();

        let matA = null;
        let matB = null;

        let grayA = null;
        let grayB = null;

        let featuresA = null;
        let featuresB = null;

        try {
            setDisabled("compareBtn", true);
            setDisabled("downloadReportBtn", true);

            resetPipeline();

            /* ACQUIRE */

            pipelineActive(0);

            setText(
                "status",
                "ACQUIRING"
            );

            await sleep(80);

            await waitForOpenCV();

            const loadedA =
                await imageToMat(imageAFile);

            const loadedB =
                await imageToMat(imageBFile);

            matA = loadedA.mat;
            matB = loadedB.mat;

            /* PREPROCESS */

            pipelineActive(1);

            setText(
                "status",
                "PREPROCESSING"
            );

            await sleep(80);

            const qualityA =
                calculateImageQuality(matA);

            const qualityB =
                calculateImageQuality(matB);

            /* EXTRACT */

            pipelineActive(2);

            setText(
                "status",
                "EXTRACTING FEATURES"
            );

            await sleep(100);

            featuresA =
                extractFeatures(matA);

            featuresB =
                extractFeatures(matB);

            /* MATCH */

            pipelineActive(3);

            setText(
                "status",
                "MATCHING FEATURES"
            );

            await sleep(100);

            const matchResult =
                matchDescriptors(
                    featuresA.descriptors,
                    featuresB.descriptors
                );

            const candidates =
                deduplicateMatches(
                    matchResult.candidates
                );

            /* VERIFY */

            pipelineActive(4);

            setText(
                "status",
                "VERIFYING GEOMETRY"
            );

            await sleep(100);

            const geometry =
                verifyGeometry(
                    featuresA.keypoints,
                    featuresB.keypoints,
                    candidates
                );

            const coverage =
                calculateSpatialCoverage(
                    featuresA.keypoints,
                    geometry.verified,
                    matA.cols,
                    matA.rows
                );

            /* SCORE */

            pipelineActive(5);

            setText(
                "status",
                "CALCULATING SCORE"
            );

            await sleep(100);

            const scoreResult =
                calculateScore({
                    candidates,
                    verified:
                        geometry.verified,
                    coverage,
                    geometry:
                        geometry.geometricConsistency,
                    qualityA,
                    qualityB
                });

            const classification =
                classifyMatch({
                    score: scoreResult.score,
                    verified:
                        geometry.verified.length,
                    geometry:
                        geometry.geometricConsistency,
                    coverage
                });

            /* VISUALIZATION */

            const visualization =
                createCorrespondenceVisualization(
                    matA,
                    matB,
                    featuresA.keypoints,
                    featuresB.keypoints,
                    geometry.verified
                );

            /* REPORT */

            pipelineActive(6);

            setText(
                "status",
                "GENERATING RESULTS"
            );

            await sleep(120);

            const processingTime =
                performance.now() -
                startTime;

            const quality =
                qualityA.label === "POOR" ||
                qualityB.label === "POOR"
                    ? "POOR"
                    : qualityA.label === "FAIR" ||
                      qualityB.label === "FAIR"
                        ? "FAIR"
                        : qualityA.label === "GOOD" ||
                          qualityB.label === "GOOD"
                            ? "GOOD"
                            : "EXCELLENT";

            const result = {
                success: true,

                engine:
                    "Browser OpenCV.js ORB + BFMatcher + RANSAC",

                version:
                    "LUNARMATCH 5.3 Browser Engine",

                score:
                    scoreResult.score,

                classification,

                quality,

                processingTime,

                rawMatches:
                    matchResult.rawMatches,

                candidateMatches:
                    candidates.length,

                verifiedMatches:
                    geometry.verified.length,

                coverage,

                correspondenceStrength:
                    scoreResult.correspondenceStrength,

                inlierRatio:
                    geometry.inlierRatio,

                geometricConsistency:
                    geometry.geometricConsistency,

                homographyStatus:
                    geometry.homographyFound
                        ? "DETECTED"
                        : "NOT ESTABLISHED",

                verificationStatus:
                    geometry.verified.length >= 4
                        ? "VERIFIED"
                        : "INSUFFICIENT",

                imageA: {
                    width: loadedA.width,
                    height: loadedA.height,
                    originalWidth:
                        loadedA.originalWidth,
                    originalHeight:
                        loadedA.originalHeight,
                    keypoints:
                        featuresA.count,
                    contrast:
                        qualityA.contrast,
                    sharpness:
                        qualityA.sharpness,
                    qualityScore:
                        qualityA.score,
                    qualityLabel:
                        qualityA.label
                },

                imageB: {
                    width: loadedB.width,
                    height: loadedB.height,
                    originalWidth:
                        loadedB.originalWidth,
                    originalHeight:
                        loadedB.originalHeight,
                    keypoints:
                        featuresB.count,
                    contrast:
                        qualityB.contrast,
                    sharpness:
                        qualityB.sharpness,
                    qualityScore:
                        qualityB.score,
                    qualityLabel:
                        qualityB.label
                },

                visualization,

                interpretation: ""
            };

            result.interpretation =
                generateInterpretation(result);

            lastAnalysis = result;

            displayResults(result);

            pipelineComplete();

            setText(
                "status",
                classification.status
            );

        } catch (error) {
            console.error(
                "LUNARMATCH analysis error:",
                error
            );

            pipelineError(5);

            setText(
                "status",
                "ANALYSIS ERROR"
            );

            setText(
                "interpretation",
                `Analysis could not be completed: ${
                    error.message ||
                    "Unknown processing error."
                }`
            );

            show(
                "visualPlaceholder",
                true
            );

        } finally {
            try {
                if (
                    featuresA &&
                    featuresA.keypoints
                ) {
                    featuresA.keypoints.delete();
                }

                if (
                    featuresA &&
                    featuresA.descriptors
                ) {
                    featuresA.descriptors.delete();
                }

                if (
                    featuresB &&
                    featuresB.keypoints
                ) {
                    featuresB.keypoints.delete();
                }

                if (
                    featuresB &&
                    featuresB.descriptors
                ) {
                    featuresB.descriptors.delete();
                }

                if (grayA) grayA.delete();
                if (grayB) grayB.delete();

                if (matA) matA.delete();
                if (matB) matB.delete();

            } catch (cleanupError) {
                console.warn(
                    "OpenCV cleanup warning:",
                    cleanupError
                );
            }

            setDisabled(
                "compareBtn",
                false
            );
        }
    }

    /* ---------------------------------------------------------
       PDF ENGINE
    --------------------------------------------------------- */

    function loadJsPDF() {
        if (
            window.jspdf &&
            window.jspdf.jsPDF
        ) {
            return Promise.resolve(
                window.jspdf.jsPDF
            );
        }

        if (pdfReadyPromise) {
            return pdfReadyPromise;
        }

        pdfReadyPromise = new Promise(
            (resolve, reject) => {
                const script =
                    document.createElement("script");

                script.src =
                    "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";

                script.onload = () => {
                    if (
                        window.jspdf &&
                        window.jspdf.jsPDF
                    ) {
                        resolve(
                            window.jspdf.jsPDF
                        );
                    } else {
                        reject(
                            new Error(
                                "PDF library initialized incorrectly."
                            )
                        );
                    }
                };

                script.onerror = () => {
                    reject(
                        new Error(
                            "Unable to load PDF generator."
                        )
                    );
                };

                document.head.appendChild(script);
            }
        );

        return pdfReadyPromise;
    }

    /* ---------------------------------------------------------
       PDF REPORT
    --------------------------------------------------------- */

    async function downloadReport() {
        if (!lastAnalysis) {
            alert(
                "Please complete an analysis before generating a report."
            );

            return;
        }

        const button =
            $("downloadReportBtn");

        if (button) {
            button.disabled = true;
            button.textContent =
                "GENERATING REPORT...";
        }

        try {
            const jsPDF =
                await loadJsPDF();

            const doc =
                new jsPDF({
                    orientation: "portrait",
                    unit: "mm",
                    format: "a4"
                });

            const result =
                lastAnalysis;

            const margin = 16;

            let y = 18;

            doc.setFont(
                "helvetica",
                "bold"
            );

            doc.setFontSize(22);

            doc.text(
                "LUNARMATCH",
                margin,
                y
            );

            y += 8;

            doc.setFontSize(10);

            doc.setFont(
                "helvetica",
                "normal"
            );

            doc.text(
                "Lunar Image Correspondence Analysis Report",
                margin,
                y
            );

            y += 10;

            doc.line(
                margin,
                y,
                194,
                y
            );

            y += 10;

            doc.setFont(
                "helvetica",
                "bold"
            );

            doc.setFontSize(13);

            doc.text(
                "ANALYSIS RESULT",
                margin,
                y
            );

            y += 8;

            doc.setFont(
                "helvetica",
                "normal"
            );

            doc.setFontSize(10);

            const resultLines = [
                `Decision: ${result.classification.status}`,
                `Correspondence Score: ${round(result.score, 1)}%`,
                `Confidence: ${result.classification.confidence}`,
                `Overall Image Quality: ${result.quality}`,
                `Processing Time: ${formatMs(result.processingTime)}`
            ];

            resultLines.forEach(line => {
                doc.text(
                    line,
                    margin,
                    y
                );

                y += 6;
            });

            y += 5;

            doc.setFont(
                "helvetica",
                "bold"
            );

            doc.text(
                "CORRESPONDENCE METRICS",
                margin,
                y
            );

            y += 8;

            doc.setFont(
                "helvetica",
                "normal"
            );

            const correspondenceLines = [
                `Detected Keypoints A: ${result.imageA.keypoints}`,
                `Detected Keypoints B: ${result.imageB.keypoints}`,
                `Candidate Matches: ${result.candidateMatches}`,
                `Verified Matches: ${result.verifiedMatches}`,
                `Spatial Coverage: ${round(result.coverage, 1)}%`,
                `Correspondence Strength: ${round(result.correspondenceStrength, 1)}%`,
                `Inlier Ratio: ${round(result.inlierRatio * 100, 1)}%`,
                `Geometric Consistency: ${round(result.geometricConsistency, 1)}%`,
                `Homography: ${result.homographyStatus}`,
                `Verification: ${result.verificationStatus}`
            ];

            correspondenceLines.forEach(line => {
                doc.text(
                    line,
                    margin,
                    y
                );

                y += 5.5;
            });

            y += 5;

            doc.setFont(
                "helvetica",
                "bold"
            );

            doc.text(
                "IMAGE QUALITY",
                margin,
                y
            );

            y += 8;

            doc.setFont(
                "helvetica",
                "normal"
            );

            const qualityLines = [
                `Image A: ${result.imageA.width} × ${result.imageA.height}`,
                `Image A Quality: ${round(result.imageA.qualityScore, 1)}% (${result.imageA.qualityLabel})`,
                `Image B: ${result.imageB.width} × ${result.imageB.height}`,
                `Image B Quality: ${round(result.imageB.qualityScore, 1)}% (${result.imageB.qualityLabel})`
            ];

            qualityLines.forEach(line => {
                doc.text(
                    line,
                    margin,
                    y
                );

                y += 5.5;
            });

            y += 5;

            doc.setFont(
                "helvetica",
                "bold"
            );

            doc.text(
                "INTERPRETATION",
                margin,
                y
            );

            y += 7;

            doc.setFont(
                "helvetica",
                "normal"
            );

            const interpretation =
                doc.splitTextToSize(
                    result.interpretation,
                    178
                );

            doc.text(
                interpretation,
                margin,
                y
            );

            y +=
                interpretation.length *
                5 +
                7;

            /* Visualization */

            if (
                result.visualization &&
                y < 230
            ) {
                doc.setFont(
                    "helvetica",
                    "bold"
                );

                doc.text(
                    "CORRESPONDENCE VISUALIZATION",
                    margin,
                    y
                );

                y += 6;

                try {
                    doc.addImage(
                        result.visualization,
                        "JPEG",
                        margin,
                        y,
                        178,
                        72
                    );
                } catch (imageError) {
                    console.warn(
                        "Could not embed visualization:",
                        imageError
                    );
                }
            }

            /* Footer */

            doc.setFontSize(8);
            doc.setFont(
                "helvetica",
                "normal"
            );

            doc.text(
                "LUNARMATCH • Lunar Intelligence Platform",
                margin,
                287
            );

            doc.text(
                new Date().toLocaleString(),
                194,
                287,
                {
                    align: "right"
                }
            );

            doc.save(
                "LUNARMATCH_Analysis_Report.pdf"
            );

        } catch (error) {
            console.error(
                "PDF generation error:",
                error
            );

            alert(
                "The PDF report could not be generated. Please try again."
            );

        } finally {
            if (button) {
                button.disabled =
                    !lastAnalysis;

                button.textContent =
                    "DOWNLOAD PDF REPORT";
            }
        }
    }

    /* ---------------------------------------------------------
       FILE INPUT SETUP
    --------------------------------------------------------- */

    function setupImageInput(
        inputId,
        previewId,
        slot
    ) {
        const input = $(inputId);

        if (!input) return;

        input.addEventListener(
            "change",
            async () => {
                const file =
                    input.files &&
                    input.files[0];

                if (!file) return;

                try {
                    validateFile(file);

                    if (slot === "A") {
                        imageAFile = file;
                        imageAData =
                            await fileToDataURL(file);
                    } else {
                        imageBFile = file;
                        imageBData =
                            await fileToDataURL(file);
                    }

                    showPreview(
                        file,
                        previewId
                    );

                    resetResults();

                    setText(
                        "status",
                        slot === "A"
                            ? "IMAGE A READY"
                            : "IMAGE B READY"
                    );

                } catch (error) {
                    alert(
                        error.message
                    );

                    input.value = "";

                    if (slot === "A") {
                        imageAFile = null;
                    } else {
                        imageBFile = null;
                    }
                }
            }
        );
    }

    /* ---------------------------------------------------------
       DRAG & DROP
    --------------------------------------------------------- */

    function setupDropZone(
        zone,
        input,
        previewId,
        slot
    ) {
        if (!zone || !input) return;

        [
            "dragenter",
            "dragover"
        ].forEach(eventName => {
            zone.addEventListener(
                eventName,
                event => {
                    event.preventDefault();
                    event.stopPropagation();

                    zone.classList.add(
                        "drag-active"
                    );
                }
            );
        });

        [
            "dragleave",
            "drop"
        ].forEach(eventName => {
            zone.addEventListener(
                eventName,
                event => {
                    event.preventDefault();
                    event.stopPropagation();

                    zone.classList.remove(
                        "drag-active"
                    );
                }
            );
        });

        zone.addEventListener(
            "drop",
            event => {
                const files =
                    event.dataTransfer.files;

                if (
                    !files ||
                    !files.length
                ) {
                    return;
                }

                const file = files[0];

                try {
                    validateFile(file);

                    const dataTransfer =
                        new DataTransfer();

                    dataTransfer.items.add(
                        file
                    );

                    input.files =
                        dataTransfer.files;

                    input.dispatchEvent(
                        new Event(
                            "change",
                            {
                                bubbles: true
                            }
                        )
                    );

                } catch (error) {
                    alert(
                        error.message
                    );
                }
            }
        );
    }

    /* ---------------------------------------------------------
       NAVIGATION
    --------------------------------------------------------- */

    function setupNavigation() {
        const links =
            document.querySelectorAll(
                'a[href^="#"]'
            );

        links.forEach(link => {
            link.addEventListener(
                "click",
                event => {
                    const targetId =
                        link.getAttribute(
                            "href"
                        );

                    if (
                        !targetId ||
                        targetId === "#"
                    ) {
                        return;
                    }

                    const target =
                        document.querySelector(
                            targetId
                        );

                    if (!target) return;

                    event.preventDefault();

                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }
            );
        });
    }

    /* ---------------------------------------------------------
       BUTTON EFFECTS
    --------------------------------------------------------- */

    function setupButtonEffects() {
        document
            .querySelectorAll(
                "button"
            )
            .forEach(button => {
                button.addEventListener(
                    "click",
                    () => {
                        button.classList.add(
                            "button-pulse"
                        );

                        setTimeout(
                            () => {
                                button.classList.remove(
                                    "button-pulse"
                                );
                            },
                            260
                        );
                    }
                );
            });
    }

    /* ---------------------------------------------------------
       REGISTRATION / CONTACT
    --------------------------------------------------------- */

    function setupPlaceholderActions() {
        const registration =
            $("registrationBtn");

        if (registration) {
            registration.addEventListener(
                "click",
                () => {
                    alert(
                        "LUNARMATCH account registration will be available in the next platform release."
                    );
                }
            );
        }

        const contact =
            $("contactBtn");

        if (contact) {
            contact.addEventListener(
                "click",
                () => {
                    const contactSection =
                        $("contact");

                    if (contactSection) {
                        contactSection.scrollIntoView({
                            behavior: "smooth"
                        });
                    }
                }
            );
        }
    }

    /* ---------------------------------------------------------
       SYSTEM STATUS
    --------------------------------------------------------- */

    async function initializeSystem() {
        try {
            setText(
                "status",
                "SYSTEM READY"
            );

            await waitForOpenCV();

            console.log(
                "LUNARMATCH: OpenCV.js initialized."
            );

        } catch (error) {
            console.warn(
                "OpenCV initialization delayed:",
                error
            );

            setText(
                "status",
                "READY"
            );
        }
    }

    /* ---------------------------------------------------------
       INITIALIZATION
    --------------------------------------------------------- */

    document.addEventListener(
        "DOMContentLoaded",
        () => {
            setupImageInput(
                "fileA",
                "previewA",
                "A"
            );

            setupImageInput(
                "fileB",
                "previewB",
                "B"
            );

            const inputA =
                $("fileA");

            const inputB =
                $("fileB");

            const drops =
                document.querySelectorAll(
                    ".drop-new"
                );

            setupDropZone(
                drops[0],
                inputA,
                "previewA",
                "A"
            );

            setupDropZone(
                drops[1],
                inputB,
                "previewB",
                "B"
            );

            const compare =
                $("compareBtn");

            if (compare) {
                compare.addEventListener(
                    "click",
                    analyzeImages
                );
            }

            const report =
                $("downloadReportBtn");

            if (report) {
                report.addEventListener(
                    "click",
                    downloadReport
                );
            }

            setupNavigation();
            setupButtonEffects();
            setupPlaceholderActions();

            resetResults();

            initializeSystem();
        }
    );

})();
