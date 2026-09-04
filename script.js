(() => {
    "use strict";

    /*
     * ============================================================
     * LUNARMATCH V7
     * Browser-side Lunar Correspondence Engine
     *
     * No backend dependency for image analysis.
     * Classical computer vision:
     * - Adaptive FAST/Harris-style feature detection
     * - Gradient histogram descriptors
     * - Ratio + mutual descriptor matching
     * - Relaxed fallback matching
     * - Affine RANSAC verification
     * - Evidence-based confidence scoring
     * ============================================================
     */

    const CONFIG = {
        MAX_IMAGE_DIMENSION: 1000,
        WORK_MAX_DIMENSION: 700,

        MAX_KEYPOINTS: 900,
        MAX_MATCH_FEATURES: 320,

        PATCH_RADIUS: 12,
        DESCRIPTOR_GRID: 4,
        ORIENTATION_BINS: 8,

        LOWE_RATIO: 0.92,
        RELAXED_RATIO: 0.96,

        MAX_DESCRIPTOR_DISTANCE: 0.95,
        RELAXED_DESCRIPTOR_DISTANCE: 1.20,

        RANSAC_ITERATIONS: 700,
        RANSAC_ERROR_PIXELS: 9,

        MIN_FEATURES_FOR_ANALYSIS: 8,
        MIN_CANDIDATES_FOR_GEOMETRY: 4,
        MIN_VERIFIED_MATCHES: 6,

        MAX_VISUAL_MATCHES: 80,

        GRID_ROWS: 6,
        GRID_COLS: 6,

        EARLY_RANSAC_INLIER_RATIO: 0.65
    };

    const state = {
        imageAFile: null,
        imageBFile: null,

        imageAData: null,
        imageBData: null,

        lastAnalysis: null
    };

    const PIPELINE = [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"
    ];

    /* ============================================================
       BASIC HELPERS
       ============================================================ */

    const $ = (id) => document.getElementById(id);

    function setText(id, value) {
        const el = $(id);
        if (el) el.textContent = value;
    }

    function setDisabled(id, disabled) {
        const el = $(id);
        if (el) el.disabled = disabled;
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function round(value, decimals = 1) {
        if (!Number.isFinite(value)) return 0;
        const p = Math.pow(10, decimals);
        return Math.round(value * p) / p;
    }

    function delay(ms = 0) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function nextFrame() {
        return new Promise(resolve => {
            requestAnimationFrame(() => resolve());
        });
    }

    function formatTime(ms) {
        if (ms < 1000) return `${Math.round(ms)} ms`;
        return `${(ms / 1000).toFixed(1)} sec`;
    }

    function distanceSquared(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        return dx * dx + dy * dy;
    }

    /* ============================================================
       PIPELINE UI
       ============================================================ */

    function resetPipeline() {
        PIPELINE.forEach(id => {
            const el = $(id);
            if (!el) return;

            el.classList.remove("active");
            el.classList.remove("complete");
            el.classList.remove("error");
        });
    }

    function setPipelineActive(id) {
        const el = $(id);
        if (!el) return;

        el.classList.add("active");
        el.classList.remove("complete");
        el.classList.remove("error");
    }

    function setPipelineComplete(id) {
        const el = $(id);
        if (!el) return;

        el.classList.remove("active");
        el.classList.remove("error");
        el.classList.add("complete");
    }

    function setPipelineError(id) {
        const el = $(id);
        if (!el) return;

        el.classList.remove("active");
        el.classList.remove("complete");
        el.classList.add("error");
    }

    /* ============================================================
       RESULT RESET
       ============================================================ */

    function resetResults() {
        const defaults = {
            score: "—",
            features: "—",
            confidence: "—",
            quality: "—",
            time: "—",

            resolutionA: "—",
            keypointsA: "—",
            contrastA: "—",
            sharpnessA: "—",
            qualityScoreA: "—",

            resolutionB: "—",
            keypointsB: "—",
            contrastB: "—",
            sharpnessB: "—",
            qualityScoreB: "—",

            rawMatches: "—",
            candidateMatches: "—",
            verifiedMatches: "—",
            featureCoverage: "—",
            correspondenceStrength: "—",

            inlierRatio: "—",
            geometricConsistency: "—",
            homographyStatus: "—",
            verificationStatus: "—"
        };

        Object.entries(defaults).forEach(([id, value]) => {
            setText(id, value);
        });

        setText(
            "status",
            "READY — Upload two lunar images to begin correspondence analysis."
        );

        const map = $("correspondenceMap");
        const placeholder = $("visualPlaceholder");

        if (map) {
            map.innerHTML = "";
            map.style.display = "none";
        }

        if (placeholder) {
            placeholder.style.display = "";
        }

        setText(
            "visualNote",
            "LOCAL FEATURES + RANSAC"
        );

        setText(
            "interpretation",
            "Upload two compatible lunar images and run the analysis engine."
        );

        setDisabled("downloadReportBtn", true);

        resetPipeline();
    }

    /* ============================================================
       FILE VALIDATION
       ============================================================ */

    function validateImageFile(file) {
        if (!file) {
            throw new Error("No image file selected.");
        }

        const allowed = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/tiff"
        ];

        const extension = file.name
            .split(".")
            .pop()
            .toLowerCase();

        const allowedExtensions = [
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
            "tif",
            "tiff"
        ];

        if (
            !allowed.includes(file.type) &&
            !allowedExtensions.includes(extension)
        ) {
            throw new Error(
                "Unsupported image format. Use JPG, PNG, WEBP, BMP or TIFF."
            );
        }

        if (file.size > 25 * 1024 * 1024) {
            throw new Error(
                "Image is too large. Maximum supported size is 25 MB."
            );
        }
    }

    /* ============================================================
       IMAGE LOADING
       ============================================================ */

    function readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => resolve(reader.result);

            reader.onerror = () => {
                reject(new Error("Unable to read the selected image."));
            };

            reader.readAsDataURL(file);
        });
    }

    function loadImage(dataURL) {
        return new Promise((resolve, reject) => {
            const img = new Image();

            img.onload = () => resolve(img);

            img.onerror = () => {
                reject(new Error("Unable to decode image."));
            };

            img.src = dataURL;
        });
    }

    async function imageToGray(file) {
        validateImageFile(file);

        const dataURL = await readFileAsDataURL(file);
        const img = await loadImage(dataURL);

        let width = img.naturalWidth;
        let height = img.naturalHeight;

        const scale = Math.min(
            1,
            CONFIG.MAX_IMAGE_DIMENSION / Math.max(width, height)
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

        const rgba = ctx.getImageData(
            0,
            0,
            width,
            height
        ).data;

        const gray = new Float32Array(width * height);

        for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
            gray[i] =
                0.299 * rgba[p] +
                0.587 * rgba[p + 1] +
                0.114 * rgba[p + 2];
        }

        return {
            data: gray,
            width,
            height,
            originalWidth: img.naturalWidth,
            originalHeight: img.naturalHeight,
            dataURL
        };
    }

    /* ============================================================
       RESIZING
       ============================================================ */

    function resizeGray(image, maxDimension) {
        if (
            Math.max(image.width, image.height) <= maxDimension
        ) {
            return image;
        }

        const scale =
            maxDimension /
            Math.max(image.width, image.height);

        const width = Math.max(1, Math.round(image.width * scale));
        const height = Math.max(1, Math.round(image.height * scale));

        const output = new Float32Array(width * height);

        for (let y = 0; y < height; y++) {
            const sy =
                (y / Math.max(1, height - 1)) *
                (image.height - 1);

            const y0 = Math.floor(sy);
            const y1 = Math.min(image.height - 1, y0 + 1);
            const fy = sy - y0;

            for (let x = 0; x < width; x++) {
                const sx =
                    (x / Math.max(1, width - 1)) *
                    (image.width - 1);

                const x0 = Math.floor(sx);
                const x1 = Math.min(image.width - 1, x0 + 1);
                const fx = sx - x0;

                const a =
                    image.data[y0 * image.width + x0];

                const b =
                    image.data[y0 * image.width + x1];

                const c =
                    image.data[y1 * image.width + x0];

                const d =
                    image.data[y1 * image.width + x1];

                const top = a + (b - a) * fx;
                const bottom = c + (d - c) * fx;

                output[y * width + x] =
                    top + (bottom - top) * fy;
            }
        }

        return {
            ...image,
            data: output,
            width,
            height
        };
    }

    /* ============================================================
       PERCENTILE
       ============================================================ */

    function percentile(values, q) {
        if (!values.length) return 0;

        const sampleLimit = 12000;

        let sample;

        if (values.length <= sampleLimit) {
            sample = Array.from(values);
        } else {
            sample = [];

            const step = values.length / sampleLimit;

            for (let i = 0; i < sampleLimit; i++) {
                sample.push(
                    values[Math.floor(i * step)]
                );
            }
        }

        sample.sort((a, b) => a - b);

        const index =
            (sample.length - 1) * q;

        const lower = Math.floor(index);
        const upper = Math.ceil(index);

        if (lower === upper) {
            return sample[lower];
        }

        return (
            sample[lower] +
            (sample[upper] - sample[lower]) *
            (index - lower)
        );
    }

    /* ============================================================
       LOCAL NORMALIZATION
       ============================================================ */

    function localNormalize(image) {
        image = resizeGray(
            image,
            CONFIG.WORK_MAX_DIMENSION
        );

        const width = image.width;
        const height = image.height;
        const n = width * height;

        const normalized = new Float32Array(n);

        const integral = new Float64Array(
            (width + 1) * (height + 1)
        );

        const integralSq = new Float64Array(
            (width + 1) * (height + 1)
        );

        for (let y = 1; y <= height; y++) {
            let rowSum = 0;
            let rowSq = 0;

            for (let x = 1; x <= width; x++) {
                const value =
                    image.data[
                        (y - 1) * width + (x - 1)
                    ];

                rowSum += value;
                rowSq += value * value;

                const index =
                    y * (width + 1) + x;

                integral[index] =
                    integral[index - (width + 1)] +
                    rowSum;

                integralSq[index] =
                    integralSq[index - (width + 1)] +
                    rowSq;
            }
        }

        const radius = 6;

        function regionSum(buffer, x0, y0, x1, y1) {
            return (
                buffer[y1 * (width + 1) + x1] -
                buffer[y0 * (width + 1) + x1] -
                buffer[y1 * (width + 1) + x0] +
                buffer[y0 * (width + 1) + x0]
            );
        }

        for (let y = 0; y < height; y++) {
            const y0 = Math.max(0, y - radius);
            const y1 = Math.min(height - 1, y + radius);

            for (let x = 0; x < width; x++) {
                const x0 = Math.max(0, x - radius);
                const x1 = Math.min(width - 1, x + radius);

                const rx0 = x0;
                const ry0 = y0;
                const rx1 = x1 + 1;
                const ry1 = y1 + 1;

                const count =
                    (rx1 - rx0) *
                    (ry1 - ry0);

                const sum = regionSum(
                    integral,
                    rx0,
                    ry0,
                    rx1,
                    ry1
                );

                const sq = regionSum(
                    integralSq,
                    rx0,
                    ry0,
                    rx1,
                    ry1
                );

                const mean = sum / count;

                const variance = Math.max(
                    4,
                    sq / count - mean * mean
                );

                const index =
                    y * width + x;

                normalized[index] =
                    (image.data[index] - mean) /
                    Math.sqrt(variance);
            }
        }

        /*
         * Mild smoothing removes single-pixel noise while
         * preserving crater/ridge boundaries.
         */
        const smoothed = new Float32Array(n);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const center =
                    normalized[y * width + x];

                const left =
                    normalized[
                        y * width +
                        Math.max(0, x - 1)
                    ];

                const right =
                    normalized[
                        y * width +
                        Math.min(width - 1, x + 1)
                    ];

                const up =
                    normalized[
                        Math.max(0, y - 1) * width + x
                    ];

                const down =
                    normalized[
                        Math.min(height - 1, y + 1) * width + x
                    ];

                smoothed[y * width + x] =
                    center * 0.50 +
                    (left + right + up + down) * 0.125;
            }
        }

        return {
            ...image,
            data: smoothed
        };
    }

    /* ============================================================
       IMAGE QUALITY
       ============================================================ */

    function imageQuality(image) {
        const data = image.data;
        const n = data.length;

        let sum = 0;
        let sumSq = 0;

        for (let i = 0; i < n; i++) {
            sum += data[i];
            sumSq += data[i] * data[i];
        }

        const mean = sum / n;
        const variance =
            Math.max(0, sumSq / n - mean * mean);

        const contrast = Math.sqrt(variance);

        let sharpnessSum = 0;
        let sharpnessCount = 0;

        for (let y = 1; y < image.height - 1; y++) {
            for (let x = 1; x < image.width - 1; x++) {
                const i =
                    y * image.width + x;

                const lap =
                    image.data[i - image.width] +
                    image.data[i - 1] +
                    image.data[i + 1] +
                    image.data[i + image.width] -
                    4 * image.data[i];

                sharpnessSum += lap * lap;
                sharpnessCount++;
            }
        }

        const sharpness =
            sharpnessCount
                ? sharpnessSum / sharpnessCount
                : 0;

        const p05 = percentile(data, 0.05);
        const p95 = percentile(data, 0.95);

        const dynamicRange = p95 - p05;

        const contrastScore =
            clamp((contrast / 55) * 100, 0, 100);

        const sharpnessScore =
            clamp(
                (Math.log1p(sharpness) /
                    Math.log1p(5000)) *
                100,
                0,
                100
            );

        const rangeScore =
            clamp((dynamicRange / 4.0) * 100, 0, 100);

        const qualityScore =
            contrastScore * 0.35 +
            sharpnessScore * 0.40 +
            rangeScore * 0.25;

        return {
            mean,
            variance,
            contrast,
            sharpness,
            dynamicRange,
            qualityScore: clamp(qualityScore, 0, 100)
        };
    }

    /* ============================================================
       GRADIENT
       ============================================================ */

    function gradientAt(image, x, y) {
        const width = image.width;
        const height = image.height;

        x = clamp(x, 1, width - 2);
        y = clamp(y, 1, height - 2);

        const left =
            image.data[y * width + x - 1];

        const right =
            image.data[y * width + x + 1];

        const up =
            image.data[(y - 1) * width + x];

        const down =
            image.data[(y + 1) * width + x];

        return {
            gx: (right - left) * 0.5,
            gy: (down - up) * 0.5
        };
    }

    /* ============================================================
       FEATURE DETECTION
       ============================================================ */

    function detectFeatures(image) {
        const width = image.width;
        const height = image.height;

        const candidates = [];

        /*
         * Estimate image contrast so the detector adapts
         * automatically between bright/dark lunar imagery.
         */
        let sum = 0;
        let sq = 0;

        for (let i = 0; i < image.data.length; i++) {
            const v = image.data[i];
            sum += v;
            sq += v * v;
        }

        const mean =
            sum / image.data.length;

        const variance =
            Math.max(
                1,
                sq / image.data.length -
                mean * mean
            );

        const sigma = Math.sqrt(variance);

        const threshold =
            clamp(sigma * 0.55, 7, 22);

        const radius = 3;

        const circle = [
            [0, -3],
            [1, -3],
            [2, -2],
            [3, -1],
            [3, 0],
            [3, 1],
            [2, 2],
            [1, 3],
            [0, 3],
            [-1, 3],
            [-2, 2],
            [-3, 1],
            [-3, 0],
            [-3, -1],
            [-2, -2],
            [-1, -3]
        ];

        for (
            let y = radius + 1;
            y < height - radius - 1;
            y++
        ) {
            for (
                let x = radius + 1;
                x < width - radius - 1;
                x++
            ) {
                const center =
                    image.data[y * width + x];

                let brighter = 0;
                let darker = 0;

                for (const [dx, dy] of circle) {
                    const value =
                        image.data[
                            (y + dy) * width +
                            (x + dx)
                        ];

                    if (value > center + threshold) {
                        brighter++;
                    }

                    if (value < center - threshold) {
                        darker++;
                    }
                }

                /*
                 * FAST-like contiguous arc approximation.
                 */
                if (
                    brighter < 5 &&
                    darker < 5
                ) {
                    continue;
                }

                let gx = 0;
                let gy = 0;

                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (dx === 0 && dy === 0) continue;

                        const weight =
                            dx === 0 || dy === 0
                                ? 1
                                : 0.707;

                        const current =
                            image.data[
                                (y + dy) * width +
                                (x + dx)
                            ];

                        gx +=
                            (current - center) *
                            dx *
                            weight;

                        gy +=
                            (current - center) *
                            dy *
                            weight;
                    }
                }

                const gradientStrength =
                    Math.sqrt(gx * gx + gy * gy);

                /*
                 * Harris-style corner response.
                 */
                let sxx = 0;
                let syy = 0;
                let sxy = 0;

                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        const g =
                            gradientAt(
                                image,
                                x + dx,
                                y + dy
                            );

                        sxx += g.gx * g.gx;
                        syy += g.gy * g.gy;
                        sxy += g.gx * g.gy;
                    }
                }

                const determinant =
                    sxx * syy - sxy * sxy;

                const trace =
                    sxx + syy + 1e-6;

                const harris =
                    determinant -
                    0.04 * trace * trace;

                const response =
                    Math.max(
                        0,
                        harris
                    ) +
                    gradientStrength * 4;

                if (response <= 5) continue;

                candidates.push({
                    x,
                    y,
                    response
                });
            }
        }

        candidates.sort(
            (a, b) => b.response - a.response
        );

        /*
         * Spatial quota:
         * Don't allow one crater/ridge to consume
         * the entire feature budget.
         */
        const cellWidth =
            width / CONFIG.GRID_COLS;

        const cellHeight =
            height / CONFIG.GRID_ROWS;

        const perCell =
            Math.ceil(
                CONFIG.MAX_KEYPOINTS /
                (CONFIG.GRID_ROWS * CONFIG.GRID_COLS)
            );

        const cellCounts =
            new Map();

        const selected = [];

        const minDistance =
            Math.max(
                8,
                Math.min(width, height) * 0.018
            );

        const minDistanceSq =
            minDistance * minDistance;

        for (const candidate of candidates) {
            const col =
                Math.min(
                    CONFIG.GRID_COLS - 1,
                    Math.floor(candidate.x / cellWidth)
                );

            const row =
                Math.min(
                    CONFIG.GRID_ROWS - 1,
                    Math.floor(candidate.y / cellHeight)
                );

            const cell =
                row * CONFIG.GRID_COLS + col;

            const count =
                cellCounts.get(cell) || 0;

            if (count >= perCell) {
                continue;
            }

            let tooClose = false;

            /*
             * Compare only against nearby selected points
             * by using a small spatial neighborhood.
             */
            for (let i = selected.length - 1; i >= 0; i--) {
                const point = selected[i];

                if (
                    Math.abs(point.x - candidate.x) >
                    minDistance * 1.5
                ) {
                    continue;
                }

                if (
                    Math.abs(point.y - candidate.y) >
                    minDistance * 1.5
                ) {
                    continue;
                }

                if (
                    distanceSquared(
                        point,
                        candidate
                    ) < minDistanceSq
                ) {
                    tooClose = true;
                    break;
                }
            }

            if (tooClose) continue;

            selected.push(candidate);

            cellCounts.set(
                cell,
                count + 1
            );

            if (
                selected.length >=
                CONFIG.MAX_KEYPOINTS
            ) {
                break;
            }
        }

        /*
         * If spatial quotas were too restrictive,
         * fill remaining capacity with strong points.
         */
        if (
            selected.length <
            Math.min(
                CONFIG.MAX_KEYPOINTS,
                candidates.length
            )
        ) {
            for (const candidate of candidates) {
                if (
                    selected.length >=
                    CONFIG.MAX_KEYPOINTS
                ) {
                    break;
                }

                if (
                    selected.includes(candidate)
                ) {
                    continue;
                }

                selected.push(candidate);
            }
        }

        return selected;
    }

    /* ============================================================
       ROBUST GRADIENT DESCRIPTOR
       ============================================================ */

    function describeFeature(image, feature) {
        const grid =
            CONFIG.DESCRIPTOR_GRID;

        const bins =
            CONFIG.ORIENTATION_BINS;

        const radius =
            CONFIG.PATCH_RADIUS;

        /*
         * Determine dominant orientation.
         */
        let dominantX = 0;
        let dominantY = 0;

        for (
            let y = -radius;
            y <= radius;
            y += 2
        ) {
            for (
                let x = -radius;
                x <= radius;
                x += 2
            ) {
                const g =
                    gradientAt(
                        image,
                        feature.x + x,
                        feature.y + y
                    );

                const magnitude =
                    Math.sqrt(
                        g.gx * g.gx +
                        g.gy * g.gy
                    );

                dominantX +=
                    g.gx * magnitude;

                dominantY +=
                    g.gy * magnitude;
            }
        }

        const dominantAngle =
            Math.atan2(
                dominantY,
                dominantX
            );

        const descriptor =
            new Float32Array(
                grid * grid * bins
            );

        /*
         * Rotate sampling coordinates according to
         * dominant local gradient orientation.
         */
        const cosA =
            Math.cos(-dominantAngle);

        const sinA =
            Math.sin(-dominantAngle);

        const cellSize =
            (radius * 2) / grid;

        for (
            let py = -radius;
            py < radius;
            py += 1
        ) {
            for (
                let px = -radius;
                px < radius;
                px += 1
            ) {
                const rx =
                    px * cosA -
                    py * sinA;

                const ry =
                    px * sinA +
                    py * cosA;

                const sx =
                    Math.round(
                        feature.x + rx
                    );

                const sy =
                    Math.round(
                        feature.y + ry
                    );

                if (
                    sx < 1 ||
                    sy < 1 ||
                    sx >= image.width - 1 ||
                    sy >= image.height - 1
                ) {
                    continue;
                }

                const g =
                    gradientAt(
                        image,
                        sx,
                        sy
                    );

                const magnitude =
                    Math.sqrt(
                        g.gx * g.gx +
                        g.gy * g.gy
                    );

                if (magnitude < 0.01) {
                    continue;
                }

                let angle =
                    Math.atan2(
                        g.gy,
                        g.gx
                    ) -
                    dominantAngle;

                while (angle < 0) {
                    angle += Math.PI * 2;
                }

                while (angle >= Math.PI * 2) {
                    angle -= Math.PI * 2;
                }

                const binFloat =
                    (angle /
                        (Math.PI * 2)) *
                    bins;

                const bin0 =
                    Math.floor(binFloat) % bins;

                const fraction =
                    binFloat - Math.floor(binFloat);

                const bin1 =
                    (bin0 + 1) % bins;

                const cellX =
                    clamp(
                        Math.floor(
                            (px + radius) /
                            cellSize
                        ),
                        0,
                        grid - 1
                    );

                const cellY =
                    clamp(
                        Math.floor(
                            (py + radius) /
                            cellSize
                        ),
                        0,
                        grid - 1
                    );

                const base =
                    (
                        cellY * grid +
                        cellX
                    ) * bins;

                /*
                 * Distance from feature center gives
                 * a gentle spatial weighting.
                 */
                const radial =
                    Math.sqrt(
                        px * px +
                        py * py
                    ) / radius;

                const weight =
                    Math.exp(
                        -0.5 *
                        radial *
                        radial
                    );

                descriptor[base + bin0] +=
                    magnitude *
                    weight *
                    (1 - fraction);

                descriptor[base + bin1] +=
                    magnitude *
                    weight *
                    fraction;
            }
        }

        /*
         * Root-style normalization.
         */
        let norm = 0;

        for (let i = 0; i < descriptor.length; i++) {
            norm +=
                descriptor[i] *
                descriptor[i];
        }

        norm =
            Math.sqrt(norm) + 1e-8;

        for (let i = 0; i < descriptor.length; i++) {
            descriptor[i] /= norm;
        }

        /*
         * Hellinger/root normalization.
         */
        for (let i = 0; i < descriptor.length; i++) {
            descriptor[i] =
                Math.sqrt(
                    Math.abs(descriptor[i])
                );
        }

        norm = 0;

        for (let i = 0; i < descriptor.length; i++) {
            norm +=
                descriptor[i] *
                descriptor[i];
        }

        norm =
            Math.sqrt(norm) + 1e-8;

        for (let i = 0; i < descriptor.length; i++) {
            descriptor[i] /= norm;
        }

        return descriptor;
    }

    async function extractFeatures(image) {
        const points =
            detectFeatures(image);

        const features = [];

        for (let i = 0; i < points.length; i++) {
            const point = points[i];

            const descriptor =
                describeFeature(
                    image,
                    point
                );

            features.push({
                x: point.x,
                y: point.y,
                response: point.response,
                descriptor
            });

            if (i % 80 === 0) {
                await delay(0);
            }
        }

        return features;
    }

    /* ============================================================
       DESCRIPTOR DISTANCE
       ============================================================ */

    function descriptorDistance(a, b) {
        let sum = 0;

        for (let i = 0; i < a.length; i++) {
            const d = a[i] - b[i];
            sum += d * d;
        }

        return Math.sqrt(sum);
    }

    /* ============================================================
       MATCHING
       ============================================================ */

    function nearestTwo(query, features) {
        let bestIndex = -1;
        let bestDistance = Infinity;

        let secondDistance = Infinity;

        for (let i = 0; i < features.length; i++) {
            const distance =
                descriptorDistance(
                    query.descriptor,
                    features[i].descriptor
                );

            if (distance < bestDistance) {
                secondDistance = bestDistance;
                bestDistance = distance;
                bestIndex = i;
            } else if (
                distance < secondDistance
            ) {
                secondDistance = distance;
            }
        }

        return {
            bestIndex,
            bestDistance,
            secondDistance
        };
    }

    function matchFeatures(
        featuresA,
        featuresB,
        options = {}
    ) {
        const ratio =
            options.ratio ||
            CONFIG.LOWE_RATIO;

        const maxDistance =
            options.maxDistance ||
            CONFIG.MAX_DESCRIPTOR_DISTANCE;

        const limitA =
            featuresA
                .slice()
                .sort(
                    (a, b) =>
                        b.response - a.response
                )
                .slice(
                    0,
                    CONFIG.MAX_MATCH_FEATURES
                );

        const limitB =
            featuresB
                .slice()
                .sort(
                    (a, b) =>
                        b.response - a.response
                )
                .slice(
                    0,
                    CONFIG.MAX_MATCH_FEATURES
                );

        const forward = [];

        for (let i = 0; i < limitA.length; i++) {
            const result =
                nearestTwo(
                    limitA[i],
                    limitB
                );

            if (result.bestIndex < 0) {
                continue;
            }

            const ratioValue =
                result.secondDistance === Infinity
                    ? 0
                    : result.bestDistance /
                      result.secondDistance;

            if (
                ratioValue <= ratio &&
                result.bestDistance <= maxDistance
            ) {
                forward.push({
                    a: limitA[i],
                    b: limitB[result.bestIndex],
                    distance: result.bestDistance,
                    ratio: ratioValue
                });
            }
        }

        /*
         * Reverse pass for mutual consistency.
         */
        const reverseMap =
            new Map();

        for (let i = 0; i < limitB.length; i++) {
            const result =
                nearestTwo(
                    limitB[i],
                    limitA
                );

            if (result.bestIndex < 0) {
                continue;
            }

            reverseMap.set(
                limitB[i],
                {
                    feature: limitA[result.bestIndex],
                    distance: result.bestDistance
                }
            );
        }

        const mutual = [];

        for (const match of forward) {
            const reverse =
                reverseMap.get(match.b);

            if (!reverse) continue;

            if (
                reverse.feature !== match.a
            ) {
                continue;
            }

            mutual.push(match);
        }

        /*
         * Sort strongest descriptor matches first.
         */
        mutual.sort(
            (a, b) =>
                a.distance - b.distance
        );

        return mutual;
    }

    async function robustMatch(
        featuresA,
        featuresB
    ) {
        let matches =
            matchFeatures(
                featuresA,
                featuresB,
                {
                    ratio: CONFIG.LOWE_RATIO,
                    maxDistance:
                        CONFIG.MAX_DESCRIPTOR_DISTANCE
                }
            );

        let mode = "STRICT";

        /*
         * If strict matching is too sparse,
         * perform a controlled relaxed pass.
         */
        if (matches.length < 8) {
            const relaxed =
                matchFeatures(
                    featuresA,
                    featuresB,
                    {
                        ratio:
                            CONFIG.RELAXED_RATIO,
                        maxDistance:
                            CONFIG.RELAXED_DESCRIPTOR_DISTANCE
                    }
                );

            if (
                relaxed.length >
                matches.length
            ) {
                matches = relaxed;
                mode = "RELAXED";
            }
        }

        /*
         * Deduplicate spatially similar matches.
         */
        const filtered = [];

        for (const match of matches) {
            let duplicate = false;

            for (const existing of filtered) {
                const da =
                    distanceSquared(
                        match.a,
                        existing.a
                    );

                const db =
                    distanceSquared(
                        match.b,
                        existing.b
                    );

                if (
                    da < 9 &&
                    db < 9
                ) {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate) {
                filtered.push(match);
            }
        }

        await delay(0);

        return {
            matches: filtered,
            mode
        };
    }

    /* ============================================================
       AFFINE MODEL
       ============================================================ */

    function solveAffine(
        p1,
        p2,
        p3,
        q1,
        q2,
        q3
    ) {
        const matrix = [
            [
                p1.x,
                p1.y,
                1,
                0,
                0,
                0
            ],
            [
                0,
                0,
                0,
                p1.x,
                p1.y,
                1
            ],
            [
                p2.x,
                p2.y,
                1,
                0,
                0,
                0
            ],
            [
                0,
                0,
                0,
                p2.x,
                p2.y,
                1
            ],
            [
                p3.x,
                p3.y,
                1,
                0,
                0,
                0
            ],
            [
                0,
                0,
                0,
                p3.x,
                p3.y,
                1
            ]
        ];

        const values = [
            q1.x,
            q1.y,
            q2.x,
            q2.y,
            q3.x,
            q3.y
        ];

        /*
         * Gaussian elimination.
         */
        for (let col = 0; col < 6; col++) {
            let pivot = col;

            for (
                let row = col + 1;
                row < 6;
                row++
            ) {
                if (
                    Math.abs(
                        matrix[row][col]
                    ) >
                    Math.abs(
                        matrix[pivot][col]
                    )
                ) {
                    pivot = row;
                }
            }

            if (
                Math.abs(
                    matrix[pivot][col]
                ) < 1e-9
            ) {
                return null;
            }

            [
                matrix[col],
                matrix[pivot]
            ] = [
                matrix[pivot],
                matrix[col]
            ];

            [
                values[col],
                values[pivot]
            ] = [
                values[pivot],
                values[col]
            ];

            const divisor =
                matrix[col][col];

            for (let j = col; j < 6; j++) {
                matrix[col][j] /= divisor;
            }

            values[col] /= divisor;

            for (let row = 0; row < 6; row++) {
                if (row === col) continue;

                const factor =
                    matrix[row][col];

                for (
                    let j = col;
                    j < 6;
                    j++
                ) {
                    matrix[row][j] -=
                        factor *
                        matrix[col][j];
                }

                values[row] -=
                    factor *
                    values[col];
            }
        }

        return {
            a: values[0],
            b: values[1],
            c: values[2],
            d: values[3],
            e: values[4],
            f: values[5]
        };
    }

    function transform(model, point) {
        return {
            x:
                model.a * point.x +
                model.b * point.y +
                model.c,

            y:
                model.d * point.x +
                model.e * point.y +
                model.f
        };
    }

    function triangleArea(a, b, c) {
        return Math.abs(
            (
                a.x * (b.y - c.y) +
                b.x * (c.y - a.y) +
                c.x * (a.y - b.y)
            ) * 0.5
        );
    }

    /* ============================================================
       GEOMETRIC VERIFICATION
       ============================================================ */

    function verifyGeometry(
        matches,
        imageA,
        imageB
    ) {
        if (
            matches.length <
            CONFIG.MIN_CANDIDATES_FOR_GEOMETRY
        ) {
            return {
                model: null,
                inliers: [],
                ratio: 0,
                meanError: Infinity,
                medianError: Infinity,
                consistency: 0
            };
        }

        /*
         * Threshold adapts to working image dimensions.
         */
        const scale =
            Math.max(
                imageA.width,
                imageA.height,
                imageB.width,
                imageB.height
            ) / 700;

        const threshold =
            clamp(
                CONFIG.RANSAC_ERROR_PIXELS * scale,
                7,
                14
            );

        let bestModel = null;
        let bestInliers = [];
        let bestError = Infinity;

        for (
            let iteration = 0;
            iteration < CONFIG.RANSAC_ITERATIONS;
            iteration++
        ) {
            const n = matches.length;

            const i1 =
                Math.floor(Math.random() * n);

            let i2 =
                Math.floor(Math.random() * n);

            let i3 =
                Math.floor(Math.random() * n);

            if (
                i1 === i2 ||
                i1 === i3 ||
                i2 === i3
            ) {
                continue;
            }

            const m1 = matches[i1];
            const m2 = matches[i2];
            const m3 = matches[i3];

            /*
             * Reject degenerate triangles.
             */
            const areaA =
                triangleArea(
                    m1.a,
                    m2.a,
                    m3.a
                );

            const areaB =
                triangleArea(
                    m1.b,
                    m2.b,
                    m3.b
                );

            if (
                areaA < 3 ||
                areaB < 3
            ) {
                continue;
            }

            const model =
                solveAffine(
                    m1.a,
                    m2.a,
                    m3.a,
                    m1.b,
                    m2.b,
                    m3.b
                );

            if (!model) continue;

            const inliers = [];
            let totalError = 0;

            for (const match of matches) {
                const predicted =
                    transform(
                        model,
                        match.a
                    );

                const dx =
                    predicted.x -
                    match.b.x;

                const dy =
                    predicted.y -
                    match.b.y;

                const error =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );

                if (error <= threshold) {
                    inliers.push({
                        ...match,
                        error
                    });

                    totalError += error;
                }
            }

            if (!inliers.length) continue;

            const meanError =
                totalError /
                inliers.length;

            const better =
                inliers.length >
                    bestInliers.length ||
                (
                    inliers.length ===
                        bestInliers.length &&
                    meanError < bestError
                );

            if (better) {
                bestModel = model;
                bestInliers = inliers;
                bestError = meanError;
            }

            const ratio =
                inliers.length /
                matches.length;

            /*
             * Early termination once we have a very
             * convincing model.
             */
            if (
                inliers.length >= 20 &&
                ratio >=
                    CONFIG.EARLY_RANSAC_INLIER_RATIO
            ) {
                break;
            }
        }

        if (
            !bestModel ||
            bestInliers.length < 3
        ) {
            return {
                model: bestModel,
                inliers: bestInliers,
                ratio:
                    matches.length
                        ? bestInliers.length /
                          matches.length
                        : 0,
                meanError: Infinity,
                medianError: Infinity,
                consistency: 0
            };
        }

        /*
         * Refit using all current inliers.
         * This stabilizes the affine model.
         */
        if (bestInliers.length >= 3) {
            const base =
                bestInliers;

            /*
             * Use several random triplets and choose the
             * least-error refined model.
             */
            let refinedModel =
                bestModel;

            if (base.length >= 6) {
                let bestRefinedError =
                    Infinity;

                for (
                    let k = 0;
                    k < Math.min(20, base.length * 2);
                    k++
                ) {
                    const i1 =
                        Math.floor(
                            Math.random() *
                            base.length
                        );

                    let i2 =
                        Math.floor(
                            Math.random() *
                            base.length
                        );

                    let i3 =
                        Math.floor(
                            Math.random() *
                            base.length
                        );

                    if (
                        i1 === i2 ||
                        i1 === i3 ||
                        i2 === i3
                    ) {
                        continue;
                    }

                    const candidate =
                        solveAffine(
                            base[i1].a,
                            base[i2].a,
                            base[i3].a,
                            base[i1].b,
                            base[i2].b,
                            base[i3].b
                        );

                    if (!candidate) continue;

                    let errorSum = 0;

                    for (const m of base) {
                        const p =
                            transform(
                                candidate,
                                m.a
                            );

                        const dx =
                            p.x - m.b.x;

                        const dy =
                            p.y - m.b.y;

                        errorSum +=
                            Math.sqrt(
                                dx * dx +
                                dy * dy
                            );
                    }

                    const average =
                        errorSum /
                        base.length;

                    if (
                        average <
                        bestRefinedError
                    ) {
                        bestRefinedError =
                            average;

                        refinedModel =
                            candidate;
                    }
                }
            }

            bestModel =
                refinedModel;
        }

        /*
         * Recalculate inliers using refined model.
         */
        const refinedInliers = [];
        const errors = [];

        for (const match of matches) {
            const predicted =
                transform(
                    bestModel,
                    match.a
                );

            const dx =
                predicted.x - match.b.x;

            const dy =
                predicted.y - match.b.y;

            const error =
                Math.sqrt(
                    dx * dx +
                    dy * dy
                );

            if (error <= threshold) {
                refinedInliers.push({
                    ...match,
                    error
                });

                errors.push(error);
            }
        }

        errors.sort((a, b) => a - b);

        const meanError =
            errors.length
                ? errors.reduce(
                    (a, b) => a + b,
                    0
                ) / errors.length
                : Infinity;

        const medianError =
            errors.length
                ? errors[
                    Math.floor(
                        errors.length / 2
                    )
                ]
                : Infinity;

        const ratio =
            matches.length
                ? refinedInliers.length /
                  matches.length
                : 0;

        const consistency =
            Number.isFinite(meanError)
                ? clamp(
                    100 -
                    meanError * 10 -
                    medianError * 4,
                    0,
                    100
                )
                : 0;

        return {
            model: bestModel,
            inliers: refinedInliers,
            ratio,
            meanError,
            medianError,
            consistency
        };
    }

    /* ============================================================
       COVERAGE
       ============================================================ */

    function calculateCoverage(
        inliers,
        image
    ) {
        if (!inliers.length) return 0;

        const occupied =
            new Set();

        const cols = 4;
        const rows = 4;

        for (const match of inliers) {
            const col =
                clamp(
                    Math.floor(
                        (match.a.x /
                            image.width) *
                        cols
                    ),
                    0,
                    cols - 1
                );

            const row =
                clamp(
                    Math.floor(
                        (match.a.y /
                            image.height) *
                        rows
                    ),
                    0,
                    rows - 1
                );

            occupied.add(
                row * cols + col
            );
        }

        return (
            occupied.size /
            (rows * cols)
        ) * 100;
    }

    /* ============================================================
       MATCH QUALITY
       ============================================================ */

    function calculateMatchQuality(matches) {
        if (!matches.length) return 0;

        let sum = 0;

        for (const match of matches) {
            const distanceScore =
                clamp(
                    1 -
                    match.distance /
                    CONFIG.RELAXED_DESCRIPTOR_DISTANCE,
                    0,
                    1
                );

            const ratioScore =
                clamp(
                    1 -
                    match.ratio,
                    0,
                    1
                );

            sum +=
                distanceScore * 0.65 +
                ratioScore * 0.35;
        }

        return (
            sum /
            matches.length
        ) * 100;
    }

    /* ============================================================
       SCORE
       ============================================================ */

    function calculateScore(
        featuresA,
        featuresB,
        matches,
        verification,
        qualityA,
        qualityB,
        matchingMode
    ) {
        const verified =
            verification.inliers.length;

        const candidates =
            matches.length;

        const featureBase =
            Math.max(
                1,
                Math.min(
                    featuresA.length,
                    featuresB.length
                )
            );

        const density =
            clamp(
                (verified /
                    featureBase) *
                500,
                0,
                100
            );

        const geometric =
            clamp(
                verification.ratio * 150,
                0,
                100
            );

        const consistency =
            verification.consistency;

        const coverage =
            calculateCoverage(
                verification.inliers,
                {
                    width: 700,
                    height: 700
                }
            );

        const matchQuality =
            calculateMatchQuality(
                verification.inliers.length
                    ? verification.inliers
                    : matches
            );

        const imageQuality =
            (
                qualityA.qualityScore +
                qualityB.qualityScore
            ) / 2;

        /*
         * Candidate abundance is useful, but it should
         * never overpower geometric evidence.
         */
        const candidateSupport =
            clamp(
                candidates / 25 * 100,
                0,
                100
            );

        let score =
            density * 0.27 +
            geometric * 0.25 +
            consistency * 0.18 +
            coverage * 0.10 +
            matchQuality * 0.12 +
            imageQuality * 0.05 +
            candidateSupport * 0.03;

        /*
         * Relaxed matching gets a small evidence penalty.
         * This prevents relaxed matches from being treated
         * as equally trustworthy as strict matches.
         */
        if (matchingMode === "RELAXED") {
            score *= 0.94;
        }

        /*
         * Strong evidence floor:
         * a result cannot become strong merely because
         * the images are high quality.
         */
        if (verified < 6) {
            score = Math.min(score, 44);
        } else if (verified < 10) {
            score = Math.min(score, 59);
        }

        const strong =
            verified >= 20 &&
            score >= 78 &&
            verification.ratio >= 0.35 &&
            verification.consistency >= 75;

        const found =
            verified >= 10 &&
            score >= 60 &&
            verification.ratio >= 0.25;

        const possible =
            verified >= 6 &&
            score >= 45 &&
            coverage >= 18;

        let status =
            "NO RELIABLE CORRESPONDENCE";

        let confidence =
            "LOW";

        if (strong) {
            status = "STRONG CORRESPONDENCE";
            confidence = "HIGH";
        } else if (found) {
            status = "CORRESPONDENCE FOUND";
            confidence = "MODERATE";
        } else if (possible) {
            status = "POSSIBLE CORRESPONDENCE";
            confidence = "LOW";
        }

        return {
            score: clamp(score, 0, 100),
            status,
            confidence,

            density,
            geometric,
            consistency,
            coverage,
            matchQuality,
            imageQuality,
            candidateSupport,

            strong,
            found,
            possible
        };
    }

    /* ============================================================
       INTERPRETATION
       ============================================================ */

    function buildInterpretation(
        scored,
        featuresA,
        featuresB,
        matches,
        verification,
        matchingMode
    ) {
        const verified =
            verification.inliers.length;

        const candidates =
            matches.length;

        if (scored.strong) {
            return (
                `Strong lunar correspondence established. ` +
                `${verified} feature correspondences remain geometrically consistent ` +
                `after affine RANSAC verification, with a ${(verification.ratio * 100).toFixed(1)}% ` +
                `inlier ratio. The correspondence is supported by spatial coverage and descriptor agreement.`
            );
        }

        if (scored.found) {
            return (
                `Reliable lunar correspondence detected. ` +
                `${verified} verified features support a consistent affine relationship ` +
                `between the two images. The evidence is sufficient for a moderate-confidence match.`
            );
        }

        if (scored.possible) {
            return (
                `Possible lunar correspondence detected. ` +
                `${verified} features survived geometric verification from ${candidates} candidate matches. ` +
                `The geometric evidence is promising but remains below the threshold required for a reliable match.`
            );
        }

        if (matchingMode === "RELAXED") {
            return (
                `The engine detected limited feature agreement using a controlled relaxed matching pass. ` +
                `${featuresA.length} and ${featuresB.length} features were available, but only ` +
                `${verified} correspondences survived geometric verification. ` +
                `More verified features are required before the correspondence can be considered reliable.`
            );
        }

        return (
            `The images contain sufficient visual structure for analysis, but the current ` +
            `correspondence evidence is too sparse to establish a reliable relationship. ` +
            `${featuresA.length} and ${featuresB.length} local features were detected, ` +
            `while ${candidates} candidate correspondences were generated and ` +
            `${verified} survived geometric verification.`
        );
    }

    /* ============================================================
       CORRESPONDENCE VISUALIZATION
       ============================================================ */

    function createCorrespondenceMap(
        sourceA,
        sourceB,
        workA,
        workB,
        inliers
    ) {
        const container =
            $("correspondenceMap");

        if (!container) return;

        container.innerHTML = "";

        const maxWidth = 1200;

        const totalWidth =
            Math.min(
                maxWidth,
                sourceA.width +
                sourceB.width
            );

        const leftWidth =
            totalWidth *
            (
                sourceA.width /
                (sourceA.width +
                    sourceB.width)
            );

        const rightWidth =
            totalWidth -
            leftWidth;

        const maxHeight =
            Math.max(
                sourceA.height,
                sourceB.height
            );

        const scale =
            Math.min(
                1,
                maxWidth /
                (
                    sourceA.width +
                    sourceB.width
                )
            );

        const canvas =
            document.createElement("canvas");

        canvas.width =
            Math.round(
                (
                    sourceA.width +
                    sourceB.width
                ) * scale
            );

        canvas.height =
            Math.round(
                maxHeight * scale
            );

        canvas.style.width = "100%";
        canvas.style.height = "auto";

        const ctx =
            canvas.getContext("2d");

        const canvasA =
            document.createElement("canvas");

        canvasA.width =
            sourceA.width;

        canvasA.height =
            sourceA.height;

        const ctxA =
            canvasA.getContext("2d");

        const canvasB =
            document.createElement("canvas");

        canvasB.width =
            sourceB.width;

        canvasB.height =
            sourceB.height;

        const ctxB =
            canvasB.getContext("2d");

        /*
         * Convert grayscale arrays to display images.
         */
        function drawGray(
            image,
            ctxTarget
        ) {
            const imageData =
                ctxTarget.createImageData(
                    image.width,
                    image.height
                );

            for (
                let i = 0, p = 0;
                i < image.data.length;
                i++, p += 4
            ) {
                const value =
                    clamp(
                        Math.round(
                            image.data[i]
                        ),
                        0,
                        255
                    );

                imageData.data[p] =
                    value;

                imageData.data[p + 1] =
                    value;

                imageData.data[p + 2] =
                    value;

                imageData.data[p + 3] =
                    255;
            }

            ctxTarget.putImageData(
                imageData,
                0,
                0
            );
        }

        drawGray(sourceA, ctxA);
        drawGray(sourceB, ctxB);

        ctx.drawImage(
            canvasA,
            0,
            0,
            sourceA.width * scale,
            sourceA.height * scale
        );

        const offsetX =
            sourceA.width * scale;

        ctx.drawImage(
            canvasB,
            offsetX,
            0,
            sourceB.width * scale,
            sourceB.height * scale
        );

        /*
         * Feature coordinates come from work images,
         * while the display uses source images.
         * Correct the coordinate scale here.
         */
        const scaleAX =
            sourceA.width /
            workA.width;

        const scaleAY =
            sourceA.height /
            workA.height;

        const scaleBX =
            sourceB.width /
            workB.width;

        const scaleBY =
            sourceB.height /
            workB.height;

        ctx.lineWidth = 1.2;

        const displayMatches =
            inliers
                .slice()
                .sort(
                    (a, b) =>
                        a.error - b.error
                )
                .slice(
                    0,
                    CONFIG.MAX_VISUAL_MATCHES
                );

        for (const match of displayMatches) {
            const ax =
                match.a.x *
                scaleAX *
                scale;

            const ay =
                match.a.y *
                scaleAY *
                scale;

            const bx =
                offsetX +
                match.b.x *
                scaleBX *
                scale;

            const by =
                match.b.y *
                scaleBY *
                scale;

            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(
                ax,
                ay,
                2.4,
                0,
                Math.PI * 2
            );
            ctx.fill();

            ctx.beginPath();
            ctx.arc(
                bx,
                by,
                2.4,
                0,
                Math.PI * 2
            );
            ctx.fill();
        }

        container.appendChild(canvas);
        container.style.display = "";
    }

    /* ============================================================
       DISPLAY RESULTS
       ============================================================ */

    function displayResults(result) {
        const {
            scored,
            processingMs,
            qualityA,
            qualityB,
            featuresA,
            featuresB,
            matches,
            verification,
            matchingMode
        } = result;

        setText(
            "status",
            scored.status
        );

        setText(
            "score",
            `${Math.round(scored.score)}%`
        );

        setText(
            "features",
            `${featuresA.length} / ${featuresB.length}`
        );

        setText(
            "confidence",
            scored.confidence
        );

        setText(
            "quality",
            `${Math.round(
                (
                    qualityA.qualityScore +
                    qualityB.qualityScore
                ) / 2
            )}/100`
        );

        setText(
            "time",
            formatTime(processingMs)
        );

        /* ---------------- IMAGE A ---------------- */

        setText(
            "resolutionA",
            `${result.imageA.width} × ${result.imageA.height}`
        );

        setText(
            "keypointsA",
            featuresA.length
        );

        setText(
            "contrastA",
            round(
                qualityA.contrast,
                2
            )
        );

        setText(
            "sharpnessA",
            round(
                qualityA.sharpness,
                2
            )
        );

        setText(
            "qualityScoreA",
            `${round(
                qualityA.qualityScore,
                1
            )}/100`
        );

        /* ---------------- IMAGE B ---------------- */

        setText(
            "resolutionB",
            `${result.imageB.width} × ${result.imageB.height}`
        );

        setText(
            "keypointsB",
            featuresB.length
        );

        setText(
            "contrastB",
            round(
                qualityB.contrast,
                2
            )
        );

        setText(
            "sharpnessB",
            round(
                qualityB.sharpness,
                2
            )
        );

        setText(
            "qualityScoreB",
            `${round(
                qualityB.qualityScore,
                1
            )}/100`
        );

        /* ---------------- MATCHING ---------------- */

        setText(
            "rawMatches",
            result.rawFeatureComparisons
        );

        setText(
            "candidateMatches",
            matches.length
        );

        setText(
            "verifiedMatches",
            verification.inliers.length
        );

        const coverage =
            calculateCoverage(
                verification.inliers,
                result.workA
            );

        setText(
            "featureCoverage",
            `${round(
                coverage,
                0
            )}%`
        );

        setText(
            "correspondenceStrength",
            `${round(
                (
                    verification.inliers.length /
                    Math.max(
                        1,
                        Math.min(
                            featuresA.length,
                            featuresB.length
                        )
                    )
                ) * 100,
                1
            )}%`
        );

        /* ---------------- GEOMETRY ---------------- */

        setText(
            "inlierRatio",
            `${round(
                verification.ratio * 100,
                1
            )}%`
        );

        setText(
            "geometricConsistency",
            `${round(
                verification.consistency,
                1
            )}%`
        );

        setText(
            "homographyStatus",
            verification.model
                ? "AFFINE MODEL VERIFIED"
                : "NOT ESTABLISHED"
        );

        setText(
            "verificationStatus",
            verification.inliers.length >= 6
                ? "RANSAC CONSISTENT"
                : "INSUFFICIENT INLIERS"
        );

        setText(
            "visualNote",
            `LOCAL FEATURES + RANSAC • ${matchingMode} MATCHING`
        );

        setText(
            "interpretation",
            buildInterpretation(
                scored,
                featuresA,
                featuresB,
                matches,
                verification,
                matchingMode
            )
        );

        createCorrespondenceMap(
            result.imageA,
            result.imageB,
            result.workA,
            result.workB,
            verification.inliers
        );

        const placeholder =
            $("visualPlaceholder");

        if (placeholder) {
            placeholder.style.display =
                verification.inliers.length
                    ? "none"
                    : "";
        }

        setDisabled(
            "downloadReportBtn",
            false
        );
    }

    /* ============================================================
       MAIN ANALYSIS
       ============================================================ */

    async function analyzeImages() {
        if (
            !state.imageAFile ||
            !state.imageBFile
        ) {
            setText(
                "status",
                "Please upload both Image A and Image B before analysis."
            );
            return;
        }

        const start =
            performance.now();

        resetPipeline();

        try {
            /* ---------------- ACQUIRE ---------------- */

            setPipelineActive(
                "stageAcquire"
            );

            setText(
                "status",
                "ACQUIRING IMAGE DATA..."
            );

            const [
                imageA,
                imageB
            ] = await Promise.all([
                imageToGray(
                    state.imageAFile
                ),
                imageToGray(
                    state.imageBFile
                )
            ]);

            setPipelineComplete(
                "stageAcquire"
            );

            await nextFrame();

            /* ---------------- PREPROCESS ---------------- */

            setPipelineActive(
                "stagePreprocess"
            );

            setText(
                "status",
                "NORMALIZING LUNAR SURFACE DATA..."
            );

            const workA =
                localNormalize(imageA);

            const workB =
                localNormalize(imageB);

            const qualityA =
                imageQuality(workA);

            const qualityB =
                imageQuality(workB);

            setPipelineComplete(
                "stagePreprocess"
            );

            await nextFrame();

            /* ---------------- EXTRACT ---------------- */

            setPipelineActive(
                "stageExtract"
            );

            setText(
                "status",
                "EXTRACTING LUNAR FEATURES..."
            );

            const featuresA =
                await extractFeatures(workA);

            const featuresB =
                await extractFeatures(workB);

            if (
                featuresA.length <
                CONFIG.MIN_FEATURES_FOR_ANALYSIS ||
                featuresB.length <
                CONFIG.MIN_FEATURES_FOR_ANALYSIS
            ) {
                throw new Error(
                    "Insufficient visual features detected in one or both images."
                );
            }

            setPipelineComplete(
                "stageExtract"
            );

            await nextFrame();

            /* ---------------- MATCH ---------------- */

            setPipelineActive(
                "stageMatch"
            );

            setText(
                "status",
                "SEARCHING FOR LUNAR FEATURE CORRESPONDENCES..."
            );

            const matchResult =
                await robustMatch(
                    featuresA,
                    featuresB
                );

            const matches =
                matchResult.matches;

            const rawFeatureComparisons =
                Math.min(
                    featuresA.length,
                    CONFIG.MAX_MATCH_FEATURES
                ) *
                Math.min(
                    featuresB.length,
                    CONFIG.MAX_MATCH_FEATURES
                );

            setPipelineComplete(
                "stageMatch"
            );

            await nextFrame();

            /* ---------------- VERIFY ---------------- */

            setPipelineActive(
                "stageVerify"
            );

            setText(
                "status",
                "VERIFYING GEOMETRIC CONSISTENCY..."
            );

            const verification =
                verifyGeometry(
                    matches,
                    workA,
                    workB
                );

            setPipelineComplete(
                "stageVerify"
            );

            await nextFrame();

            /* ---------------- SCORE ---------------- */

            setPipelineActive(
                "stageScore"
            );

            setText(
                "status",
                "CALCULATING EVIDENCE-BASED CONFIDENCE..."
            );

            const scored =
                calculateScore(
                    featuresA,
                    featuresB,
                    matches,
                    verification,
                    qualityA,
                    qualityB,
                    matchResult.mode
                );

            setPipelineComplete(
                "stageScore"
            );

            await nextFrame();

            /* ---------------- REPORT ---------------- */

            setPipelineActive(
                "stageReport"
            );

            const processingMs =
                performance.now() -
                start;

            const result = {
                scored,
                processingMs,

                imageA,
                imageB,

                workA,
                workB,

                qualityA,
                qualityB,

                featuresA,
                featuresB,

                matches,
                verification,

                matchingMode:
                    matchResult.mode,

                rawFeatureComparisons,

                generatedAt:
                    new Date().toISOString()
            };

            state.lastAnalysis =
                result;

            displayResults(
                result
            );

            setPipelineComplete(
                "stageReport"
            );

            setText(
                "status",
                scored.status
            );

        } catch (error) {
            console.error(
                "LUNARMATCH V7 ERROR:",
                error
            );

            const current =
                PIPELINE.find(id => {
                    const el = $(id);
                    return (
                        el &&
                        el.classList.contains(
                            "active"
                        )
                    );
                });

            if (current) {
                setPipelineError(
                    current
                );
            }

            setText(
                "status",
                `ANALYSIS ERROR — ${error.message}`
            );
        }
    }

    /* ============================================================
       IMAGE PREVIEW
       ============================================================ */

    function updatePreview(
        input,
        previewId
    ) {
        const file =
            input.files &&
            input.files[0];

        if (!file) return;

        const preview =
            $(previewId);

        if (!preview) return;

        const reader =
            new FileReader();

        reader.onload = event => {
            preview.src =
                event.target.result;

            preview.style.display =
                "";
        };

        reader.readAsDataURL(file);
    }

    /* ============================================================
       IMAGE HANDLING
       ============================================================ */

    function handleImage(
        file,
        side,
        input
    ) {
        try {
            validateImageFile(file);

            if (side === "A") {
                state.imageAFile = file;

                updatePreview(
                    input,
                    "previewA"
                );
            } else {
                state.imageBFile = file;

                updatePreview(
                    input,
                    "previewB"
                );
            }

            setText(
                "status",
                `${side === "A"
                    ? "IMAGE A"
                    : "IMAGE B"} READY — `
                +
                file.name
            );

            setDisabled(
                "downloadReportBtn",
                true
            );

        } catch (error) {
            setText(
                "status",
                `UPLOAD ERROR — ${error.message}`
            );
        }
    }

    /* ============================================================
       FILE INPUT SETUP
       ============================================================ */

    function setupImageInput(
        inputId,
        side
    ) {
        const input =
            $(inputId);

        if (!input) return;

        input.type = "file";

        input.accept =
            ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,image/*";

        input.addEventListener(
            "change",
            () => {
                const file =
                    input.files &&
                    input.files[0];

                if (file) {
                    handleImage(
                        file,
                        side,
                        input
                    );
                }
            }
        );
    }

    /* ============================================================
       DROP ZONE
       ============================================================ */

    function setupDropZone(
        zone,
        side
    ) {
        if (!zone) return;

        const input =
            side === "A"
                ? $("fileA")
                : $("fileB");

        if (!input) return;

        zone.addEventListener(
            "click",
            event => {
                if (
                    event.target.tagName ===
                    "INPUT"
                ) {
                    return;
                }

                input.click();
            }
        );

        [
            "dragenter",
            "dragover"
        ].forEach(eventName => {
            zone.addEventListener(
                eventName,
                event => {
                    event.preventDefault();

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
                    event.dataTransfer &&
                    event.dataTransfer.files;

                if (
                    files &&
                    files.length
                ) {
                    handleImage(
                        files[0],
                        side,
                        input
                    );
                }
            }
        );
    }

    /* ============================================================
       NAVIGATION
       ============================================================ */

    function setupNavigation() {
        document
            .querySelectorAll(
                'a[href^="#"]'
            )
            .forEach(link => {
                link.addEventListener(
                    "click",
                    event => {
                        const href =
                            link.getAttribute(
                                "href"
                            );

                        if (
                            !href ||
                            href === "#"
                        ) {
                            return;
                        }

                        const target =
                            document.querySelector(
                                href
                            );

                        if (!target) return;

                        event.preventDefault();

                        target.scrollIntoView({
                            behavior:
                                "smooth",
                            block:
                                "start"
                        });
                    }
                );
            });
    }

    /* ============================================================
       BUTTON EFFECTS
       ============================================================ */

    function setupButtonEffects() {
        document
            .querySelectorAll(
                "button"
            )
            .forEach(button => {
                button.addEventListener(
                    "pointerdown",
                    () => {
                        button.classList.add(
                            "button-pressed"
                        );
                    }
                );

                button.addEventListener(
                    "pointerup",
                    () => {
                        button.classList.remove(
                            "button-pressed"
                        );
                    }
                );

                button.addEventListener(
                    "pointerleave",
                    () => {
                        button.classList.remove(
                            "button-pressed"
                        );
                    }
                );
            });
    }

    /* ============================================================
       REPORT GENERATION
       ============================================================ */

    async function loadJsPDF() {
        if (
            window.jspdf &&
            window.jspdf.jsPDF
        ) {
            return window.jspdf.jsPDF;
        }

        await new Promise(
            (resolve, reject) => {
                const script =
                    document.createElement(
                        "script"
                    );

                script.src =
                    "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";

                script.onload =
                    resolve;

                script.onerror =
                    () => reject(
                        new Error(
                            "Unable to load PDF report engine."
                        )
                    );

                document.head.appendChild(
                    script
                );
            }
        );

        return window.jspdf.jsPDF;
    }

    async function downloadReport() {
        if (!state.lastAnalysis) {
            return;
        }

        try {
            const jsPDF =
                await loadJsPDF();

            const result =
                state.lastAnalysis;

            const doc =
                new jsPDF({
                    unit: "mm",
                    format: "a4"
                });

            const margin = 18;

            let y = 20;

            doc.setFontSize(20);

            doc.text(
                "LUNARMATCH",
                margin,
                y
            );

            y += 9;

            doc.setFontSize(10);

            doc.text(
                "Lunar Image Correspondence Analysis Report",
                margin,
                y
            );

            y += 12;

            doc.setFontSize(13);

            doc.text(
                "Analysis Summary",
                margin,
                y
            );

            y += 8;

            doc.setFontSize(10);

            const summary = [
                [
                    "Status",
                    result.scored.status
                ],
                [
                    "Overall Match",
                    `${Math.round(
                        result.scored.score
                    )}%`
                ],
                [
                    "Confidence",
                    result.scored.confidence
                ],
                [
                    "Features A",
                    `${result.featuresA.length}`
                ],
                [
                    "Features B",
                    `${result.featuresB.length}`
                ],
                [
                    "Candidate Matches",
                    `${result.matches.length}`
                ],
                [
                    "Verified Matches",
                    `${result.verification.inliers.length}`
                ],
                [
                    "Inlier Ratio",
                    `${round(
                        result.verification.ratio *
                        100,
                        1
                    )}%`
                ],
                [
                    "Geometric Consistency",
                    `${round(
                        result.verification.consistency,
                        1
                    )}%`
                ],
                [
                    "Matching Mode",
                    result.matchingMode
                ],
                [
                    "Processing Time",
                    formatTime(
                        result.processingMs
                    )
                ]
            ];

            for (const row of summary) {
                doc.text(
                    `${row[0]}: ${row[1]}`,
                    margin,
                    y
                );

                y += 6;

                if (y > 270) {
                    doc.addPage();
                    y = 20;
                }
            }

            y += 7;

            doc.setFontSize(13);

            doc.text(
                "Image A",
                margin,
                y
            );

            y += 7;

            doc.setFontSize(10);

            doc.text(
                `Resolution: ${result.imageA.width} × ${result.imageA.height}`,
                margin,
                y
            );

            y += 6;

            doc.text(
                `Quality Score: ${round(
                    result.qualityA.qualityScore,
                    1
                )}/100`,
                margin,
                y
            );

            y += 10;

            doc.setFontSize(13);

            doc.text(
                "Image B",
                margin,
                y
            );

            y += 7;

            doc.setFontSize(10);

            doc.text(
                `Resolution: ${result.imageB.width} × ${result.imageB.height}`,
                margin,
                y
            );

            y += 6;

            doc.text(
                `Quality Score: ${round(
                    result.qualityB.qualityScore,
                    1
                )}/100`,
                margin,
                y
            );

            y += 12;

            doc.setFontSize(13);

            doc.text(
                "Interpretation",
                margin,
                y
            );

            y += 7;

            doc.setFontSize(9);

            const interpretation =
                buildInterpretation(
                    result.scored,
                    result.featuresA,
                    result.featuresB,
                    result.matches,
                    result.verification,
                    result.matchingMode
                );

            const lines =
                doc.splitTextToSize(
                    interpretation,
                    174
                );

            doc.text(
                lines,
                margin,
                y
            );

            y +=
                lines.length * 4.5 +
                10;

            doc.setFontSize(8);

            doc.text(
                "Generated by LUNARMATCH browser-side correspondence engine.",
                margin,
                Math.min(y, 285)
            );

            doc.save(
                "LUNARMATCH_Analysis_Report.pdf"
            );

        } catch (error) {
            console.error(
                "PDF ERROR:",
                error
            );

            setText(
                "status",
                `REPORT ERROR — ${error.message}`
            );
        }
    }

    /* ============================================================
       REGISTRATION / CONTACT PLACEHOLDERS
       ============================================================ */

    function setupPlaceholders() {
        const registrationBtn =
            $("registrationBtn");

        if (registrationBtn) {
            registrationBtn.addEventListener(
                "click",
                () => {
                    setText(
                        "status",
                        "ACCOUNT FEATURES — COMING SOON"
                    );
                }
            );
        }

        const contactBtn =
            $("contactBtn");

        if (contactBtn) {
            contactBtn.addEventListener(
                "click",
                () => {
                    const contact =
                        $("contact");

                    if (contact) {
                        contact.scrollIntoView({
                            behavior:
                                "smooth"
                        });
                    }
                }
            );
        }
    }

    /* ============================================================
       INITIALIZATION
       ============================================================ */

    function init() {
        setupImageInput(
            "fileA",
            "A"
        );

        setupImageInput(
            "fileB",
            "B"
        );

        const dropZones =
            document.querySelectorAll(
                ".drop-new"
            );

        if (dropZones.length >= 1) {
            setupDropZone(
                dropZones[0],
                "A"
            );
        }

        if (dropZones.length >= 2) {
            setupDropZone(
                dropZones[1],
                "B"
            );
        }

        const compareBtn =
            $("compareBtn");

        if (compareBtn) {
            compareBtn.addEventListener(
                "click",
                analyzeImages
            );
        }

        const reportBtn =
            $("downloadReportBtn");

        if (reportBtn) {
            reportBtn.addEventListener(
                "click",
                downloadReport
            );
        }

        setupNavigation();
        setupButtonEffects();
        setupPlaceholders();

        resetResults();

        console.log(
            "LUNARMATCH V7 initialized."
        );

        console.log(
            `Feature budget: ${CONFIG.MAX_KEYPOINTS}`
        );

        console.log(
            `Match budget: ${CONFIG.MAX_MATCH_FEATURES}`
        );

        console.log(
            "Engine: Gradient descriptors + Ratio/Mutual Matching + Affine RANSAC"
        );
    }

    if (
        document.readyState ===
        "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            init
        );
    } else {
        init();
    }

})();
