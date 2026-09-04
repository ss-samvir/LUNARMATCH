/* ============================================================
   LUNARMATCH — LUNAR CORRESPONDENCE ENGINE V8
   Browser-based image correspondence + geometric verification
   ============================================================ */

(() => {
    "use strict";

    /* ----------------------------------------------------------
       CONFIGURATION
       ---------------------------------------------------------- */

    const CONFIG = {
        MAX_IMAGE_DIMENSION: 1000,
        WORK_MAX_DIMENSION: 700,

        MAX_KEYPOINTS: 900,
        MAX_MATCH_FEATURES: 320,

        PATCH_RADIUS: 12,
        DESCRIPTOR_GRID: 4,
        ORIENTATION_BINS: 8,

        LOWE_RATIO: 0.90,
        RELAXED_RATIO: 0.96,

        MAX_DESCRIPTOR_DISTANCE: 0.95,
        RELAXED_DESCRIPTOR_DISTANCE: 1.20,

        RANSAC_ITERATIONS: 1000,
        RANSAC_ERROR_PIXELS: 8,

        MIN_FEATURES_FOR_ANALYSIS: 8,
        MIN_CANDIDATES_FOR_GEOMETRY: 4,
        MIN_VERIFIED_MATCHES: 6,

        MAX_VISUAL_MATCHES: 80,

        GRID_ROWS: 6,
        GRID_COLS: 6,

        EARLY_RANSAC_INLIER_RATIO: 0.65
    };

    /* ----------------------------------------------------------
       STATE
       ---------------------------------------------------------- */

    const state = {
        imageAFile: null,
        imageBFile: null,
        imageAData: null,
        imageBData: null,
        lastAnalysis: null,
        initialized: false,
        objectUrls: []
    };

    /* ----------------------------------------------------------
       DOM HELPERS
       ---------------------------------------------------------- */

    const $ = id => document.getElementById(id);

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

    function round(value, digits = 1) {
        if (!Number.isFinite(value)) return 0;
        const p = Math.pow(10, digits);
        return Math.round(value * p) / p;
    }

    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function nextFrame() {
        return new Promise(resolve =>
            requestAnimationFrame(() => resolve())
        );
    }

    function distanceSquared(a, b) {
        let sum = 0;
        const n = Math.min(a.length, b.length);

        for (let i = 0; i < n; i++) {
            const d = a[i] - b[i];
            sum += d * d;
        }

        return sum;
    }

    function formatTime(ms) {
        if (!Number.isFinite(ms)) return "0.0 sec";
        return `${round(ms / 1000, 1)} sec`;
    }

    /* ----------------------------------------------------------
       PIPELINE UI
       ---------------------------------------------------------- */

    const PIPELINE = [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"
    ];

    function resetPipeline() {
        PIPELINE.forEach(id => {
            const el = $(id);
            if (!el) return;

            el.classList.remove("active", "complete", "error");
        });
    }

    function setPipelineActive(id) {
        resetPipeline();

        const index = PIPELINE.indexOf(id);

        PIPELINE.forEach((stage, i) => {
            const el = $(stage);
            if (!el) return;

            if (i < index) {
                el.classList.add("complete");
            }

            if (i === index) {
                el.classList.add("active");
            }
        });
    }

    function setPipelineComplete(id) {
        const el = $(id);
        if (el) {
            el.classList.remove("active", "error");
            el.classList.add("complete");
        }
    }

    function setPipelineError(id) {
        const el = $(id);
        if (el) {
            el.classList.remove("active");
            el.classList.add("error");
        }
    }

    /* ----------------------------------------------------------
       RESULT RESET
       ---------------------------------------------------------- */

    function resetResults() {
        const ids = [
            "overallScore",
            "confidence",
            "verifiedFeatures",
            "imageQuality",
            "processTime",
            "rawMatches",
            "candidateMatches",
            "featureCoverage",
            "correspondenceStrength",
            "inlierRatio",
            "geometricConsistency",
            "modelType",
            "verificationStatus"
        ];

        ids.forEach(id => setText(id, "—"));

        const map = $("correspondenceMap");
        const placeholder = $("correspondencePlaceholder");

        if (map) {
            map.innerHTML = "";
            map.style.display = "none";
        }

        if (placeholder) {
            placeholder.style.display = "";
        }

        const interpretation = $("interpretation");
        if (interpretation) {
            interpretation.textContent =
                "Upload two lunar images to begin correspondence analysis.";
        }

        setDisabled("downloadReportBtn", true);

        state.lastAnalysis = null;
    }

    /* ----------------------------------------------------------
       IMAGE VALIDATION
       ---------------------------------------------------------- */

    function validateImageFile(file) {
        if (!file) {
            throw new Error("No image selected.");
        }

        const allowed = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/tiff"
        ];

        const name = file.name.toLowerCase();

        const validExtension =
            /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/.test(name);

        if (!allowed.includes(file.type) && !validExtension) {
            throw new Error(
                "Please select a JPG, PNG, WEBP, BMP or TIFF image."
            );
        }

        if (file.size > 25 * 1024 * 1024) {
            throw new Error("Image must be smaller than 25 MB.");
        }

        return true;
    }

    /* ----------------------------------------------------------
       IMAGE DECODING
       ---------------------------------------------------------- */

    function readImageFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = () => {
                const img = new Image();

                img.onload = () => resolve(img);

                img.onerror = () =>
                    reject(new Error("The selected image could not be decoded."));

                img.src = reader.result;
            };

            reader.onerror = () =>
                reject(new Error("Could not read the selected image."));

            reader.readAsDataURL(file);
        });
    }

    async function imageToGray(file) {
        validateImageFile(file);

        const img = await readImageFile(file);

        let width = img.naturalWidth || img.width;
        let height = img.naturalHeight || img.height;

        const scale =
            Math.min(1, CONFIG.MAX_IMAGE_DIMENSION / Math.max(width, height));

        width = Math.max(1, Math.round(width * scale));
        height = Math.max(1, Math.round(height * scale));

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d", {
            willReadFrequently: true
        });

        ctx.drawImage(img, 0, 0, width, height);

        const data = ctx.getImageData(0, 0, width, height);

        const gray = new Float32Array(width * height);

        for (let i = 0, p = 0; i < data.data.length; i += 4, p++) {
            const r = data.data[i];
            const g = data.data[i + 1];
            const b = data.data[i + 2];

            gray[p] =
                0.299 * r +
                0.587 * g +
                0.114 * b;
        }

        return {
            width,
            height,
            data: gray
        };
    }

    function resizeGray(image, maxDimension) {
        const scale =
            Math.min(1, maxDimension / Math.max(image.width, image.height));

        if (scale >= 0.999) {
            return image;
        }

        const width = Math.max(1, Math.round(image.width * scale));
        const height = Math.max(1, Math.round(image.height * scale));

        const result = new Float32Array(width * height);

        for (let y = 0; y < height; y++) {
            const sy = Math.min(
                image.height - 1,
                Math.floor(y / scale)
            );

            for (let x = 0; x < width; x++) {
                const sx = Math.min(
                    image.width - 1,
                    Math.floor(x / scale)
                );

                result[y * width + x] =
                    image.data[sy * image.width + sx];
            }
        }

        return {
            width,
            height,
            data: result
        };
    }

    /* ----------------------------------------------------------
       IMAGE NORMALIZATION
       ---------------------------------------------------------- */

    function percentile(array, q) {
        if (!array.length) return 0;

        const sorted = Array.from(array).sort((a, b) => a - b);

        const index =
            clamp(q, 0, 1) * (sorted.length - 1);

        const lower = Math.floor(index);
        const upper = Math.ceil(index);

        if (lower === upper) {
            return sorted[lower];
        }

        const weight = index - lower;

        return (
            sorted[lower] * (1 - weight) +
            sorted[upper] * weight
        );
    }

    function localNormalize(image) {
        const data = new Float32Array(image.data.length);

        const p5 = percentile(image.data, 0.05);
        const p95 = percentile(image.data, 0.95);

        const range = Math.max(1, p95 - p5);

        for (let i = 0; i < image.data.length; i++) {
            data[i] = clamp(
                (image.data[i] - p5) / range,
                0,
                1
            );
        }

        return {
            width: image.width,
            height: image.height,
            data
        };
    }

    /* ----------------------------------------------------------
       IMAGE QUALITY
       ---------------------------------------------------------- */

    function imageQuality(image) {
        const data = image.data;

        if (!data.length) {
            return {
                contrast: 0,
                sharpness: 0,
                score: 0
            };
        }

        const sampleStep = Math.max(
            1,
            Math.floor(data.length / 50000)
        );

        const samples = [];

        for (let i = 0; i < data.length; i += sampleStep) {
            samples.push(data[i]);
        }

        let mean = 0;

        for (const v of samples) {
            mean += v;
        }

        mean /= samples.length;

        let variance = 0;

        for (const v of samples) {
            const d = v - mean;
            variance += d * d;
        }

        variance /= samples.length;

        let sharpness = 0;
        let count = 0;

        for (let y = 1; y < image.height - 1; y += 2) {
            for (let x = 1; x < image.width - 1; x += 2) {
                const i = y * image.width + x;

                const lap =
                    image.data[i - image.width] +
                    image.data[i + image.width] +
                    image.data[i - 1] +
                    image.data[i + 1] -
                    4 * image.data[i];

                sharpness += lap * lap;
                count++;
            }
        }

        sharpness = count ? sharpness / count : 0;

        const contrastScore =
            clamp(Math.sqrt(variance) * 3.0, 0, 1);

        const sharpnessScore =
            clamp(Math.log10(sharpness + 1) / 4.0, 0, 1);

        const score =
            100 *
            (
                0.55 * contrastScore +
                0.45 * sharpnessScore
            );

        return {
            contrast: Math.sqrt(variance) * 100,
            sharpness,
            score: clamp(score, 0, 100)
        };
    }

    /* ----------------------------------------------------------
       GRADIENT
       ---------------------------------------------------------- */

    function gradientAt(image, x, y) {
        const w = image.width;
        const h = image.height;

        x = clamp(Math.round(x), 1, w - 2);
        y = clamp(Math.round(y), 1, h - 2);

        const i = y * w + x;

        const gx =
            image.data[i + 1] -
            image.data[i - 1];

        const gy =
            image.data[i + w] -
            image.data[i - w];

        return {
            gx,
            gy,
            magnitude: Math.sqrt(gx * gx + gy * gy),
            angle: Math.atan2(gy, gx)
        };
    }

    /* ----------------------------------------------------------
       FEATURE DETECTION
       ---------------------------------------------------------- */

    function detectFeatures(image) {
        const candidates = [];

        const w = image.width;
        const h = image.height;

        const border = CONFIG.PATCH_RADIUS + 3;

        const step = Math.max(
            4,
            Math.floor(
                Math.min(w, h) / 100
            )
        );

        for (
            let y = border;
            y < h - border;
            y += step
        ) {
            for (
                let x = border;
                x < w - border;
                x += step
            ) {
                const g = gradientAt(image, x, y);

                if (g.magnitude < 0.015) {
                    continue;
                }

                let score = 0;

                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        if (!dx && !dy) continue;

                        const n = gradientAt(
                            image,
                            x + dx,
                            y + dy
                        );

                        score += Math.abs(
                            g.magnitude - n.magnitude
                        );
                    }
                }

                score *= g.magnitude;

                candidates.push({
                    x,
                    y,
                    score,
                    magnitude: g.magnitude,
                    angle: g.angle
                });
            }
        }

        candidates.sort(
            (a, b) => b.score - a.score
        );

        const selected = [];

        const minDistance =
            Math.max(6, Math.min(w, h) * 0.025);

        const minDistanceSquared =
            minDistance * minDistance;

        for (const feature of candidates) {
            let tooClose = false;

            for (const existing of selected) {
                if (
                    distanceSquared(
                        [feature.x, feature.y],
                        [existing.x, existing.y]
                    ) < minDistanceSquared
                ) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                selected.push(feature);
            }

            if (selected.length >= CONFIG.MAX_KEYPOINTS) {
                break;
            }
        }

        return selected;
    }

    /* ----------------------------------------------------------
       FEATURE DESCRIPTOR
       ---------------------------------------------------------- */

    function describeFeature(image, feature) {
        const radius = CONFIG.PATCH_RADIUS;
        const grid = CONFIG.DESCRIPTOR_GRID;
        const bins = CONFIG.ORIENTATION_BINS;

        const descriptor = [];

        const start =
            -radius;

        const cellSize =
            (radius * 2) / grid;

        for (let gy = 0; gy < grid; gy++) {
            for (let gx = 0; gx < grid; gx++) {

                const hist =
                    new Float32Array(bins);

                const x0 =
                    Math.floor(
                        feature.x -
                        radius +
                        gx * cellSize
                    );

                const y0 =
                    Math.floor(
                        feature.y -
                        radius +
                        gy * cellSize
                    );

                const x1 =
                    Math.floor(
                        feature.x -
                        radius +
                        (gx + 1) * cellSize
                    );

                const y1 =
                    Math.floor(
                        feature.y -
                        radius +
                        (gy + 1) * cellSize
                    );

                for (
                    let y = y0;
                    y < y1;
                    y++
                ) {
                    for (
                        let x = x0;
                        x < x1;
                        x++
                    ) {
                        if (
                            x < 1 ||
                            y < 1 ||
                            x >= image.width - 1 ||
                            y >= image.height - 1
                        ) {
                            continue;
                        }

                        const g =
                            gradientAt(
                                image,
                                x,
                                y
                            );

                        const magnitude =
                            g.magnitude;

                        let angle =
                            g.angle -
                            feature.angle;

                        while (angle < 0) {
                            angle += Math.PI * 2;
                        }

                        while (angle >= Math.PI * 2) {
                            angle -= Math.PI * 2;
                        }

                        const bin = Math.floor(
                            angle /
                            (Math.PI * 2 / bins)
                        ) % bins;

                        hist[bin] += magnitude;
                    }
                }

                for (let b = 0; b < bins; b++) {
                    descriptor.push(hist[b]);
                }
            }
        }

        let norm = 0;

        for (const v of descriptor) {
            norm += v * v;
        }

        norm = Math.sqrt(norm) || 1;

        for (let i = 0; i < descriptor.length; i++) {
            descriptor[i] =
                descriptor[i] / norm;
        }

        return descriptor;
    }

    function extractFeatures(image) {
        const features =
            detectFeatures(image);

        for (const feature of features) {
            feature.descriptor =
                describeFeature(
                    image,
                    feature
                );
        }

        return features;
    }

    /* ----------------------------------------------------------
       DESCRIPTOR MATCHING
       ---------------------------------------------------------- */

    function descriptorDistance(a, b) {
        let sum = 0;

        const n =
            Math.min(a.length, b.length);

        for (let i = 0; i < n; i++) {
            const d = a[i] - b[i];
            sum += d * d;
        }

        return Math.sqrt(sum / n);
    }

    function matchFeatures(featuresA, featuresB) {
        const matches = [];

        const reverseBest =
            new Map();

        /* First determine best B -> A matches. */

        for (let j = 0; j < featuresB.length; j++) {
            let bestIndex = -1;
            let bestDistance = Infinity;

            for (let i = 0; i < featuresA.length; i++) {
                const d =
                    descriptorDistance(
                        featuresB[j].descriptor,
                        featuresA[i].descriptor
                    );

                if (d < bestDistance) {
                    bestDistance = d;
                    bestIndex = i;
                }
            }

            if (bestIndex >= 0) {
                reverseBest.set(
                    j,
                    {
                        index: bestIndex,
                        distance: bestDistance
                    }
                );
            }
        }

        /* Then A -> B using Lowe ratio + mutual consistency. */

        for (let i = 0; i < featuresA.length; i++) {
            let best = null;
            let second = null;

            for (let j = 0; j < featuresB.length; j++) {
                const d =
                    descriptorDistance(
                        featuresA[i].descriptor,
                        featuresB[j].descriptor
                    );

                if (!best || d < best.distance) {
                    second = best;
                    best = {
                        index: j,
                        distance: d
                    };
                } else if (
                    !second ||
                    d < second.distance
                ) {
                    second = {
                        index: j,
                        distance: d
                    };
                }
            }

            if (!best) continue;

            const secondDistance =
                second
                    ? second.distance
                    : Infinity;

            const ratio =
                best.distance /
                Math.max(
                    secondDistance,
                    0.000001
                );

            const reverse =
                reverseBest.get(best.index);

            const mutual =
                reverse &&
                reverse.index === i;

            if (
                (
                    ratio <= CONFIG.LOWE_RATIO &&
                    best.distance <=
                        CONFIG.MAX_DESCRIPTOR_DISTANCE
                ) ||
                (
                    mutual &&
                    ratio <= CONFIG.RELAXED_RATIO &&
                    best.distance <=
                        CONFIG.RELAXED_DESCRIPTOR_DISTANCE
                )
            ) {
                matches.push({
                    indexA: i,
                    indexB: best.index,
                    distance: best.distance,
                    ratio,
                    mutual: !!mutual
                });
            }
        }

        matches.sort(
            (a, b) => {
                const scoreA =
                    a.distance +
                    a.ratio * 0.25;

                const scoreB =
                    b.distance +
                    b.ratio * 0.25;

                return scoreA - scoreB;
            }
        );

        return matches.slice(
            0,
            CONFIG.MAX_MATCH_FEATURES
        );
    }

    /* ----------------------------------------------------------
       ROBUST MATCHING
       ---------------------------------------------------------- */

    function robustMatch(featuresA, featuresB) {
        let matches =
            matchFeatures(
                featuresA,
                featuresB
            );

        /* If too few matches survive, try a relaxed pass. */

        if (matches.length < CONFIG.MIN_CANDIDATES_FOR_GEOMETRY) {
            const relaxed = [];

            for (let i = 0; i < featuresA.length; i++) {
                let best = null;
                let second = null;

                for (let j = 0; j < featuresB.length; j++) {
                    const d =
                        descriptorDistance(
                            featuresA[i].descriptor,
                            featuresB[j].descriptor
                        );

                    if (!best || d < best.distance) {
                        second = best;
                        best = {
                            index: j,
                            distance: d
                        };
                    } else if (
                        !second ||
                        d < second.distance
                    ) {
                        second = {
                            index: j,
                            distance: d
                        };
                    }
                }

                if (!best) continue;

                const ratio =
                    best.distance /
                    Math.max(
                        second
                            ? second.distance
                            : Infinity,
                        0.000001
                    );

                if (
                    ratio <= CONFIG.RELAXED_RATIO &&
                    best.distance <=
                        CONFIG.RELAXED_DESCRIPTOR_DISTANCE
                ) {
                    relaxed.push({
                        indexA: i,
                        indexB: best.index,
                        distance: best.distance,
                        ratio,
                        mutual: false
                    });
                }
            }

            relaxed.sort(
                (a, b) =>
                    a.distance - b.distance
            );

            matches = relaxed.slice(
                0,
                CONFIG.MAX_MATCH_FEATURES
            );
        }

        return matches;
    }

    /* ----------------------------------------------------------
       AFFINE MODEL
       ---------------------------------------------------------- */

    function solveAffine(points) {
        if (!points || points.length < 3) {
            return null;
        }

        const A = [];
        const B = [];

        for (const p of points) {
            const x = p.x;
            const y = p.y;
            const X = p.X;
            const Y = p.Y;

            A.push([
                x, y, 1, 0, 0, 0
            ]);

            B.push(X);

            A.push([
                0, 0, 0, x, y, 1
            ]);

            B.push(Y);
        }

        const ATA =
            Array.from(
                { length: 6 },
                () => Array(6).fill(0)
            );

        const ATB =
            Array(6).fill(0);

        for (let r = 0; r < A.length; r++) {
            for (let c = 0; c < 6; c++) {
                ATB[c] +=
                    A[r][c] * B[r];

                for (let k = 0; k < 6; k++) {
                    ATA[c][k] +=
                        A[r][c] * A[r][k];
                }
            }
        }

        /* Gaussian elimination */

        const M =
            ATA.map(
                (row, i) =>
                    row.concat(ATB[i])
            );

        for (let col = 0; col < 6; col++) {
            let pivot = col;

            for (
                let row = col + 1;
                row < 6;
                row++
            ) {
                if (
                    Math.abs(M[row][col]) >
                    Math.abs(M[pivot][col])
                ) {
                    pivot = row;
                }
            }

            if (
                Math.abs(M[pivot][col]) <
                1e-10
            ) {
                return null;
            }

            [M[col], M[pivot]] =
                [M[pivot], M[col]];

            const div =
                M[col][col];

            for (
                let j = col;
                j <= 6;
                j++
            ) {
                M[col][j] /= div;
            }

            for (
                let row = 0;
                row < 6;
                row++
            ) {
                if (row === col) continue;

                const factor =
                    M[row][col];

                for (
                    let j = col;
                    j <= 6;
                    j++
                ) {
                    M[row][j] -=
                        factor * M[col][j];
                }
            }
        }

        const x =
            M.map(row => row[6]);

        return {
            a: x[0],
            b: x[1],
            c: x[2],
            d: x[3],
            e: x[4],
            f: x[5]
        };
    }

    function transform(model, x, y) {
        return {
            x:
                model.a * x +
                model.b * y +
                model.c,

            y:
                model.d * x +
                model.e * y +
                model.f
        };
    }

    /* ----------------------------------------------------------
       GEOMETRY HELPERS
       ---------------------------------------------------------- */

    function triangleArea(a, b, c) {
        return Math.abs(
            (
                a.x * (b.y - c.y) +
                b.x * (c.y - a.y) +
                c.x * (a.y - b.y)
            ) / 2
        );
    }

    function verifyGeometry(
        featuresA,
        featuresB,
        matches
    ) {
        if (
            matches.length <
            CONFIG.MIN_CANDIDATES_FOR_GEOMETRY
        ) {
            return {
                model: null,
                inliers: [],
                inlierRatio: 0,
                consistency: 0,
                status: "INSUFFICIENT MATCHES"
            };
        }

        const usable =
            matches.map(m => ({
                x: featuresA[m.indexA].x,
                y: featuresA[m.indexA].y,
                X: featuresB[m.indexB].x,
                Y: featuresB[m.indexB].y,
                match: m
            }));

        let bestModel = null;
        let bestInliers = [];

        const iterations =
            Math.min(
                CONFIG.RANSAC_ITERATIONS,
                Math.max(
                    250,
                    usable.length * 25
                )
            );

        for (
            let iter = 0;
            iter < iterations;
            iter++
        ) {
            const chosen = [];

            while (chosen.length < 3) {
                const index =
                    Math.floor(
                        Math.random() *
                        usable.length
                    );

                if (!chosen.includes(index)) {
                    chosen.push(index);
                }
            }

            const points =
                chosen.map(i => usable[i]);

            if (
                triangleArea(
                    points[0],
                    points[1],
                    points[2]
                ) < 2
            ) {
                continue;
            }

            const model =
                solveAffine(points);

            if (!model) continue;

            const inliers = [];

            for (let i = 0; i < usable.length; i++) {
                const p = usable[i];

                const predicted =
                    transform(
                        model,
                        p.x,
                        p.y
                    );

                const dx =
                    predicted.x - p.X;

                const dy =
                    predicted.y - p.Y;

                const error =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );

                if (
                    error <=
                    CONFIG.RANSAC_ERROR_PIXELS
                ) {
                    inliers.push({
                        index: i,
                        error
                    });
                }
            }

            if (
                inliers.length >
                bestInliers.length
            ) {
                bestModel = model;
                bestInliers = inliers;

                const ratio =
                    bestInliers.length /
                    usable.length;

                if (
                    ratio >=
                    CONFIG.EARLY_RANSAC_INLIER_RATIO
                ) {
                    break;
                }
            }
        }

        if (!bestModel) {
            return {
                model: null,
                inliers: [],
                inlierRatio: 0,
                consistency: 0,
                status: "NO VALID GEOMETRY"
            };
        }

        /* Refine model using all inliers. */

        const refinedPoints =
            bestInliers.map(
                item => usable[item.index]
            );

        const refinedModel =
            refinedPoints.length >= 3
                ? solveAffine(refinedPoints)
                : bestModel;

        const model =
            refinedModel || bestModel;

        let finalInliers = [];

        for (let i = 0; i < usable.length; i++) {
            const p = usable[i];

            const predicted =
                transform(
                    model,
                    p.x,
                    p.y
                );

            const dx =
                predicted.x - p.X;

            const dy =
                predicted.y - p.Y;

            const error =
                Math.sqrt(
                    dx * dx +
                    dy * dy
                );

            if (
                error <=
                CONFIG.RANSAC_ERROR_PIXELS
            ) {
                finalInliers.push({
                    index: i,
                    error
                });
            }
        }

        const inlierRatio =
            finalInliers.length /
            Math.max(1, usable.length);

        const meanError =
            finalInliers.length
                ? finalInliers.reduce(
                      (sum, item) =>
                          sum + item.error,
                      0
                  ) /
                  finalInliers.length
                : Infinity;

        const consistency =
            clamp(
                inlierRatio *
                Math.exp(
                    -meanError /
                    Math.max(
                        1,
                        CONFIG.RANSAC_ERROR_PIXELS
                    )
                ),
                0,
                1
            );

        let status =
            "GEOMETRICALLY CONSISTENT";

        if (
            finalInliers.length <
            CONFIG.MIN_VERIFIED_MATCHES
        ) {
            status =
                "VERIFICATION INSUFFICIENT INLIERS";
        }

        return {
            model,
            inliers: finalInliers,
            inlierRatio,
            consistency,
            status,
            usable
        };
    }

    /* ----------------------------------------------------------
       COVERAGE
       ---------------------------------------------------------- */

    function calculateCoverage(
        featuresA,
        featuresB,
        verifiedMatches
    ) {
        if (!verifiedMatches.length) {
            return 0;
        }

        const cells = new Set();

        const rows = CONFIG.GRID_ROWS;
        const cols = CONFIG.GRID_COLS;

        for (const item of verifiedMatches) {
            const p = featuresA[item.match.indexA];

            const col = clamp(
                Math.floor(
                    p.x /
                    Math.max(
                        1,
                        featuresA.imageWidth || 1
                    ) *
                    cols
                ),
                0,
                cols - 1
            );

            const row = clamp(
                Math.floor(
                    p.y /
                    Math.max(
                        1,
                        featuresA.imageHeight || 1
                    ) *
                    rows
                ),
                0,
                rows - 1
            );

            cells.add(
                `${row}:${col}`
            );
        }

        return clamp(
            cells.size /
            (rows * cols),
            0,
            1
        );
    }

    /* ----------------------------------------------------------
       MATCH QUALITY
       ---------------------------------------------------------- */

    function calculateMatchQuality(
        matches,
        verified
    ) {
        if (!matches.length) {
            return 0;
        }

        const verifiedSet =
            new Set(
                verified.map(
                    item => item.index
                )
            );

        const good =
            matches.filter(
                (_, i) =>
                    verifiedSet.has(i)
            );

        if (!good.length) {
            return 0;
        }

        const averageDistance =
            good.reduce(
                (sum, item) =>
                    sum + item.distance,
                0
            ) / good.length;

        const descriptorQuality =
            clamp(
                1 -
                averageDistance /
                CONFIG.RELAXED_DESCRIPTOR_DISTANCE,
                0,
                1
            );

        const ratioQuality =
            good.reduce(
                (sum, item) =>
                    sum +
                    clamp(
                        1 -
                        item.ratio,
                        0,
                        1
                    ),
                0
            ) /
            good.length;

        return clamp(
            0.65 * descriptorQuality +
            0.35 * ratioQuality,
            0,
            1
        );
    }

    /* ----------------------------------------------------------
       SCORE
       ---------------------------------------------------------- */

    function calculateScore({
        qualityA,
        qualityB,
        matches,
        verifiedMatches,
        inlierRatio,
        geometryConsistency,
        coverage,
        matchQuality
    }) {
        const imageQualityScore =
            (
                qualityA +
                qualityB
            ) / 200;

        const evidenceScore =
            clamp(
                verifiedMatches /
                Math.max(
                    CONFIG.MIN_VERIFIED_MATCHES,
                    12
                ),
                0,
                1
            );

        const score =
            100 *
            (
                0.12 * imageQualityScore +
                0.30 * inlierRatio +
                0.25 * geometryConsistency +
                0.13 * coverage +
                0.15 * matchQuality +
                0.05 * evidenceScore
            );

        return clamp(
            Math.round(score),
            0,
            100
        );
    }

    /* ----------------------------------------------------------
       CONFIDENCE
       ---------------------------------------------------------- */

    function calculateConfidence({
        score,
        verifiedMatches,
        inlierRatio,
        geometryConsistency,
        coverage,
        matchQuality
    }) {
        /*
         * Confidence is evidence-based.
         * It is NOT artificially increased just to make
         * the result look better.
         */

        const evidence =
            clamp(
                verifiedMatches / 20,
                0,
                1
            );

        const confidence =
            100 *
            (
                0.24 * evidence +
                0.24 * inlierRatio +
                0.22 * geometryConsistency +
                0.15 * coverage +
                0.15 * matchQuality
            );

        const finalConfidence =
            clamp(
                0.75 * confidence +
                0.25 * score,
                0,
                100
            );

        let label = "LOW";

        if (finalConfidence >= 75) {
            label = "HIGH";
        } else if (finalConfidence >= 50) {
            label = "MEDIUM";
        }

        return {
            value: Math.round(finalConfidence),
            label
        };
    }

    /* ----------------------------------------------------------
       INTERPRETATION
       ---------------------------------------------------------- */

    function buildInterpretation(result) {
        const {
            score,
            confidence,
            verifiedMatches,
            inlierRatio,
            geometryConsistency,
            coverage
        } = result;

        if (
            verifiedMatches >= 10 &&
            inlierRatio >= 0.60 &&
            geometryConsistency >= 0.50
        ) {
            return (
                `STRONG CORRESPONDENCE DETECTED. ` +
                `${verifiedMatches} geometrically verified features ` +
                `support a consistent spatial relationship between the ` +
                `two lunar images. Overall correspondence score: ${score}%. ` +
                `Confidence: ${confidence.label} (${confidence.value}%).`
            );
        }

        if (
            verifiedMatches >= 6 &&
            inlierRatio >= 0.45
        ) {
            return (
                `PROMISING CORRESPONDENCE. ` +
                `The engine identified ${verifiedMatches} verified ` +
                `feature correspondences with measurable geometric ` +
                `consistency. Additional imagery or stronger overlap ` +
                `would increase confidence.`
            );
        }

        if (
            verifiedMatches >= 3 &&
            geometryConsistency >= 0.35
        ) {
            return (
                `WEAK BUT POSSIBLE CORRESPONDENCE. ` +
                `Some compatible image structure was detected, but the ` +
                `available evidence is not yet strong enough for a reliable ` +
                `identification.`
            );
        }

        return (
            `NO RELIABLE CORRESPONDENCE ESTABLISHED. ` +
            `The image quality may be sufficient, but the current feature ` +
            `evidence does not provide enough geometrically verified matches.`
        );
    }

    /* ----------------------------------------------------------
       CORRESPONDENCE MAP
       ---------------------------------------------------------- */

    function createCorrespondenceMap(
        imageA,
        imageB,
        featuresA,
        featuresB,
        verifiedMatches
    ) {
        const container =
            $("correspondenceMap");

        const placeholder =
            $("correspondencePlaceholder");

        if (!container) return;

        container.innerHTML = "";
        container.style.display = "block";

        if (placeholder) {
            placeholder.style.display = "none";
        }

        const canvas =
            document.createElement("canvas");

        canvas.width = 1000;
        canvas.height = 500;

        canvas.style.width = "100%";
        canvas.style.height = "auto";

        container.appendChild(canvas);

        const ctx =
            canvas.getContext("2d");

        const halfWidth =
            canvas.width / 2;

        const scaleA =
            Math.min(
                halfWidth / imageA.width,
                canvas.height / imageA.height
            );

        const scaleB =
            Math.min(
                halfWidth / imageB.width,
                canvas.height / imageB.height
            );

        const offsetAY =
            (canvas.height -
                imageA.height * scaleA) / 2;

        const offsetBY =
            (canvas.height -
                imageB.height * scaleB) / 2;

        function drawGray(
            image,
            xOffset,
            scale,
            yOffset
        ) {
            const temp =
                document.createElement("canvas");

            temp.width = image.width;
            temp.height = image.height;

            const tctx =
                temp.getContext("2d");

            const pixels =
                tctx.createImageData(
                    image.width,
                    image.height
                );

            for (
                let i = 0, p = 0;
                i < pixels.data.length;
                i += 4, p++
            ) {
                const value =
                    Math.round(
                        clamp(
                            image.data[p],
                            0,
                            255
                        )
                    );

                pixels.data[i] = value;
                pixels.data[i + 1] = value;
                pixels.data[i + 2] = value;
                pixels.data[i + 3] = 255;
            }

            tctx.putImageData(
                pixels,
                0,
                0
            );

            ctx.drawImage(
                temp,
                xOffset,
                yOffset,
                image.width * scale,
                image.height * scale
            );
        }

        drawGray(
            imageA,
            0,
            scaleA,
            offsetAY
        );

        drawGray(
            imageB,
            halfWidth,
            scaleB,
            offsetBY
        );

        for (
            let i = 0;
            i < verifiedMatches.length &&
            i < CONFIG.MAX_VISUAL_MATCHES;
            i++
        ) {
            const item =
                verifiedMatches[i];

            const fa =
                featuresA[item.match.indexA];

            const fb =
                featuresB[item.match.indexB];

            const ax =
                fa.x * scaleA;

            const ay =
                offsetAY +
                fa.y * scaleA;

            const bx =
                halfWidth +
                fb.x * scaleB;

            const by =
                offsetBY +
                fb.y * scaleB;

            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);

            ctx.globalAlpha =
                0.18 +
                0.55 *
                clamp(
                    1 -
                    item.error /
                    CONFIG.RANSAC_ERROR_PIXELS,
                    0,
                    1
                );

            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1;

            ctx.stroke();

            ctx.globalAlpha = 1;

            ctx.beginPath();
            ctx.arc(
                ax,
                ay,
                3,
                0,
                Math.PI * 2
            );

            ctx.fillStyle = "#ffffff";
            ctx.fill();

            ctx.beginPath();
            ctx.arc(
                bx,
                by,
                3,
                0,
                Math.PI * 2
            );

            ctx.fillStyle = "#ffffff";
            ctx.fill();
        }

        ctx.globalAlpha = 1;
    }

    /* ----------------------------------------------------------
       DISPLAY RESULTS
       ---------------------------------------------------------- */

    function displayResults(result) {
        setText(
            "overallScore",
            `${result.score}%`
        );

        setText(
            "confidence",
            `${result.confidence.label} (${result.confidence.value}%)`
        );

        setText(
            "verifiedFeatures",
            result.verifiedMatches
        );

        setText(
            "imageQuality",
            `${round(result.averageQuality, 1)}%`
        );

        setText(
            "processTime",
            formatTime(result.processTime)
        );

        setText(
            "rawMatches",
            result.rawMatches
        );

        setText(
            "candidateMatches",
            result.candidateMatches
        );

        setText(
            "featureCoverage",
            `${round(result.coverage * 100, 1)}%`
        );

        setText(
            "correspondenceStrength",
            `${round(result.correspondenceStrength * 100, 1)}%`
        );

        setText(
            "inlierRatio",
            `${round(result.inlierRatio * 100, 1)}%`
        );

        setText(
            "geometricConsistency",
            `${round(result.geometryConsistency * 100, 1)}%`
        );

        setText(
            "modelType",
            result.modelType
        );

        setText(
            "verificationStatus",
            result.verificationStatus
        );

        const interpretation =
            $("interpretation");

        if (interpretation) {
            interpretation.textContent =
                buildInterpretation(result);
        }

        createCorrespondenceMap(
            result.imageA,
            result.imageB,
            result.featuresA,
            result.featuresB,
            result.verified
        );
    }

    /* ----------------------------------------------------------
       MAIN ANALYSIS
       ---------------------------------------------------------- */

    async function analyzeImages() {
        if (
            !state.imageAFile ||
            !state.imageBFile
        ) {
            setText(
                "status",
                "PLEASE UPLOAD BOTH IMAGES."
            );

            return;
        }

        const start =
            performance.now();

        const analyzeButton =
            $("analyzeBtn");

        if (analyzeButton) {
            analyzeButton.disabled = true;
        }

        resetPipeline();
        resetResults();

        try {
            /* ACQUIRE */

            setPipelineActive("stageAcquire");

            setText(
                "status",
                "ACQUIRING IMAGE DATA..."
            );

            await nextFrame();

            const imageA =
                await imageToGray(
                    state.imageAFile
                );

            const imageB =
                await imageToGray(
                    state.imageBFile
                );

            setPipelineComplete(
                "stageAcquire"
            );

            /* PREPROCESS */

            setPipelineActive(
                "stagePreprocess"
            );

            setText(
                "status",
                "PREPROCESSING LUNAR IMAGERY..."
            );

            await delay(80);

            const processedA =
                localNormalize(
                    resizeGray(
                        imageA,
                        CONFIG.WORK_MAX_DIMENSION
                    )
                );

            const processedB =
                localNormalize(
                    resizeGray(
                        imageB,
                        CONFIG.WORK_MAX_DIMENSION
                    )
                );

            const qualityA =
                imageQuality(
                    processedA
                );

            const qualityB =
                imageQuality(
                    processedB
                );

            setPipelineComplete(
                "stagePreprocess"
            );

            /* FEATURE EXTRACTION */

            setPipelineActive(
                "stageExtract"
            );

            setText(
                "status",
                "EXTRACTING LUNAR SURFACE FEATURES..."
            );

            await nextFrame();

            const featuresA =
                extractFeatures(
                    processedA
                );

            const featuresB =
                extractFeatures(
                    processedB
                );

            featuresA.imageWidth =
                processedA.width;

            featuresA.imageHeight =
                processedA.height;

            featuresB.imageWidth =
                processedB.width;

            featuresB.imageHeight =
                processedB.height;

            setPipelineComplete(
                "stageExtract"
            );

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

            /* MATCH */

            setPipelineActive(
                "stageMatch"
            );

            setText(
                "status",
                "SEARCHING FOR FEATURE CORRESPONDENCES..."
            );

            await nextFrame();

            const allPossibleComparisons =
                Math.min(
                    featuresA.length,
                    CONFIG.MAX_MATCH_FEATURES
                ) *
                Math.min(
                    featuresB.length,
                    CONFIG.MAX_MATCH_FEATURES
                );

            const matches =
                robustMatch(
                    featuresA,
                    featuresB
                );

            setPipelineComplete(
                "stageMatch"
            );

            /* VERIFY */

            setPipelineActive(
                "stageVerify"
            );

            setText(
                "status",
                "VERIFYING SPATIAL CONSISTENCY..."
            );

            await nextFrame();

            const geometry =
                verifyGeometry(
                    featuresA,
                    featuresB,
                    matches
                );

            setPipelineComplete(
                "stageVerify"
            );

            const verified =
                geometry.inliers.map(
                    item =>
                        matches[
                            item.index
                        ]
                            ? {
                                  match:
                                      matches[
                                          item.index
                                      ],
                                  error:
                                      item.error
                              }
                            : null
                ).filter(Boolean);

            const verifiedMatches =
                verified.length;

            const coverage =
                calculateCoverage(
                    featuresA,
                    featuresB,
                    verified
                );

            const matchQuality =
                calculateMatchQuality(
                    matches,
                    geometry.inliers
                );

            const correspondenceStrength =
                clamp(
                    0.40 *
                        clamp(
                            verifiedMatches /
                            20,
                            0,
                            1
                        ) +
                    0.35 *
                        geometry.inlierRatio +
                    0.25 *
                        matchQuality,
                    0,
                    1
                );

            /* SCORE */

            setPipelineActive(
                "stageScore"
            );

            setText(
                "status",
                "CALCULATING CORRESPONDENCE SCORE..."
            );

            await nextFrame();

            const score =
                calculateScore({
                    qualityA:
                        qualityA.score,
                    qualityB:
                        qualityB.score,
                    matches,
                    verifiedMatches,
                    inlierRatio:
                        geometry.inlierRatio,
                    geometryConsistency:
                        geometry.consistency,
                    coverage,
                    matchQuality
                });

            const confidence =
                calculateConfidence({
                    score,
                    verifiedMatches,
                    inlierRatio:
                        geometry.inlierRatio,
                    geometryConsistency:
                        geometry.consistency,
                    coverage,
                    matchQuality
                });

            setPipelineComplete(
                "stageScore"
            );

            /* REPORT */

            setPipelineActive(
                "stageReport"
            );

            setText(
                "status",
                "GENERATING ANALYSIS REPORT..."
            );

            await delay(120);

            const processTime =
                performance.now() -
                start;

            const result = {
                score,

                confidence,

                verifiedMatches,

                averageQuality:
                    (
                        qualityA.score +
                        qualityB.score
                    ) / 2,

                processTime,

                rawMatches:
                    allPossibleComparisons,

                candidateMatches:
                    matches.length,

                coverage,

                correspondenceStrength,

                inlierRatio:
                    geometry.inlierRatio,

                geometryConsistency:
                    geometry.consistency,

                modelType:
                    geometry.model
                        ? "AFFINE MODEL VERIFIED"
                        : "NO MODEL",

                verificationStatus:
                    geometry.status,

                qualityA,
                qualityB,

                imageA:
                    processedA,

                imageB:
                    processedB,

                featuresA,
                featuresB,

                matches,
                verified,

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
                result.verifiedMatches >=
                    CONFIG.MIN_VERIFIED_MATCHES
                    ? "ANALYSIS COMPLETE — CORRESPONDENCE VERIFIED"
                    : "ANALYSIS COMPLETE — EVIDENCE INSUFFICIENT FOR RELIABLE CORRESPONDENCE"
            );

            setDisabled(
                "downloadReportBtn",
                false
            );

        } catch (error) {
            console.error(
                "LUNARMATCH analysis error:",
                error
            );

            setText(
                "status",
                `ANALYSIS ERROR — ${error.message}`
            );

            const active =
                PIPELINE.find(id => {
                    const el = $(id);
                    return (
                        el &&
                        el.classList.contains(
                            "active"
                        )
                    );
                });

            if (active) {
                setPipelineError(
                    active
                );
            }

        } finally {
            if (analyzeButton) {
                analyzeButton.disabled =
                    false;
            }
        }
    }

    /* ----------------------------------------------------------
       PREVIEW
       ---------------------------------------------------------- */

    function updatePreview(
        file,
        previewId
    ) {
        if (!file) return;

        const preview =
            $(previewId);

        if (!preview) return;

        const oldUrl =
            preview.dataset.objectUrl;

        if (oldUrl) {
            URL.revokeObjectURL(
                oldUrl
            );
        }

        const url =
            URL.createObjectURL(
                file
            );

        preview.dataset.objectUrl =
            url;

        preview.src = url;
        preview.style.display = "";
    }

    /* ----------------------------------------------------------
       IMAGE HANDLING
       ---------------------------------------------------------- */

    function handleImage(
        file,
        side,
        input
    ) {
        try {
            validateImageFile(
                file
            );

            if (side === "A") {
                state.imageAFile =
                    file;

                updatePreview(
                    file,
                    "previewA"
                );
            } else {
                state.imageBFile =
                    file;

                updatePreview(
                    file,
                    "previewB"
                );
            }

            state.lastAnalysis =
                null;

            setDisabled(
                "downloadReportBtn",
                true
            );

            setText(
                "status",
                `${side === "A" ? "IMAGE A" : "IMAGE B"} READY — ${file.name}`
            );

            if (input) {
                input.value = "";
            }

        } catch (error) {
            console.error(
                "Upload error:",
                error
            );

            setText(
                "status",
                `UPLOAD ERROR — ${error.message}`
            );

            if (input) {
                input.value = "";
            }
        }
    }

    /* ----------------------------------------------------------
       FILE INPUTS
       ---------------------------------------------------------- */

    function setupImageInput(
        inputId,
        side
    ) {
        const input =
            $(inputId);

        if (!input) {
            console.warn(
                `LUNARMATCH: ${inputId} not found.`
            );

            return;
        }

        input.type = "file";

        input.accept =
            ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,image/*";

        input.onchange = () => {
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
        };
    }

    /* ----------------------------------------------------------
       DROP ZONES
       IMPORTANT:
       No manual input.click() here.
       The HTML <label for="fileA/fileB"> already handles
       normal clicking and prevents double file dialogs.
       ---------------------------------------------------------- */

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

        [
            "dragenter",
            "dragover"
        ].forEach(
            eventName => {
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
            }
        );

        [
            "dragleave",
            "drop"
        ].forEach(
            eventName => {
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
            }
        );

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

    /* ----------------------------------------------------------
       PDF REPORT
       ---------------------------------------------------------- */

    function downloadReport() {
        if (!state.lastAnalysis) {
            return;
        }

        if (
            typeof window.jspdf ===
            "undefined"
        ) {
            alert(
                "PDF engine is unavailable. Please refresh the page and try again."
            );

            return;
        }

        const result =
            state.lastAnalysis;

        const {
            jsPDF
        } = window.jspdf;

        const doc =
            new jsPDF();

        const left = 18;

        let y = 20;

        doc.setFontSize(20);

        doc.text(
            "LUNARMATCH",
            left,
            y
        );

        y += 9;

        doc.setFontSize(11);

        doc.text(
            "Lunar Intelligence Platform",
            left,
            y
        );

        y += 14;

        doc.setFontSize(14);

        doc.text(
            "Lunar Image Correspondence Report",
            left,
            y
        );

        y += 12;

        doc.setFontSize(10);

        const lines = [
            `Generated: ${new Date().toLocaleString()}`,
            `Image A: ${state.imageAFile ? state.imageAFile.name : "—"}`,
            `Image B: ${state.imageBFile ? state.imageBFile.name : "—"}`,
            "",
            `Overall Match: ${result.score}%`,
            `Confidence: ${result.confidence.label} (${result.confidence.value}%)`,
            `Verified Features: ${result.verifiedMatches}`,
            `Candidate Matches: ${result.candidateMatches}`,
            `Feature Coverage: ${round(result.coverage * 100, 1)}%`,
            `Correspondence Strength: ${round(result.correspondenceStrength * 100, 1)}%`,
            `Inlier Ratio: ${round(result.inlierRatio * 100, 1)}%`,
            `Geometric Consistency: ${round(result.geometryConsistency * 100, 1)}%`,
            `Model: ${result.modelType}`,
            `Verification: ${result.verificationStatus}`,
            `Image Quality: ${round(result.averageQuality, 1)}%`,
            `Processing Time: ${formatTime(result.processTime)}`
        ];

        lines.forEach(line => {
            doc.text(
                line,
                left,
                y
            );

            y += 7;
        });

        y += 6;

        doc.setFontSize(12);

        doc.text(
            "Interpretation",
            left,
            y
        );

        y += 7;

        doc.setFontSize(10);

        const interpretation =
            buildInterpretation(
                result
            );

        const wrapped =
            doc.splitTextToSize(
                interpretation,
                170
            );

        doc.text(
            wrapped,
            left,
            y
        );

        y +=
            wrapped.length * 6 +
            10;

        doc.setFontSize(8);

        doc.text(
            "LUNARMATCH — Experimental browser-based lunar image correspondence analysis.",
            left,
            285
        );

        doc.save(
            "LUNARMATCH-Analysis-Report.pdf"
        );
    }

    /* ----------------------------------------------------------
       BUTTONS
       ---------------------------------------------------------- */

    function setupButtons() {
        const analyze =
            $("analyzeBtn");

        if (analyze) {
            analyze.onclick =
                analyzeImages;
        }

        const report =
            $("downloadReportBtn");

        if (report) {
            report.onclick =
                downloadReport;
        }
    }

    /* ----------------------------------------------------------
       NAVIGATION
       ---------------------------------------------------------- */

    function setupNavigation() {
        const links =
            document.querySelectorAll(
                "[data-section]"
            );

        links.forEach(link => {
            link.addEventListener(
                "click",
                event => {
                    const target =
                        link.dataset.section;

                    if (!target) return;

                    const section =
                        document.getElementById(
                            target
                        );

                    if (section) {
                        event.preventDefault();

                        section.scrollIntoView({
                            behavior: "smooth",
                            block: "start"
                        });
                    }
                }
            );
        });
    }

    /* ----------------------------------------------------------
       VISUAL EFFECTS
       ---------------------------------------------------------- */

    function setupEffects() {
        document.addEventListener(
            "mousemove",
            event => {
                document.documentElement
                    .style
                    .setProperty(
                        "--mouse-x",
                        `${event.clientX}px`
                    );

                document.documentElement
                    .style
                    .setProperty(
                        "--mouse-y",
                        `${event.clientY}px`
                    );
            }
        );
    }

    /* ----------------------------------------------------------
       GLOBAL ERROR REPORTING
       ---------------------------------------------------------- */

    window.addEventListener(
        "error",
        event => {
            console.error(
                "LUNARMATCH runtime error:",
                event.error || event.message
            );
        }
    );

    window.addEventListener(
        "unhandledrejection",
        event => {
            console.error(
                "LUNARMATCH promise error:",
                event.reason
            );
        }
    );

    /* ----------------------------------------------------------
       INITIALIZATION
       ---------------------------------------------------------- */

    function init() {
        if (
            window.__LUNARMATCH_INITIALIZED__
        ) {
            return;
        }

        window.__LUNARMATCH_INITIALIZED__ =
            true;

        state.initialized = true;

        setupImageInput(
            "fileA",
            "A"
        );

        setupImageInput(
            "fileB",
            "B"
        );

        setupDropZone(
            document.querySelector(
                'label[for="fileA"]'
            ),
            "A"
        );

        setupDropZone(
            document.querySelector(
                'label[for="fileB"]'
            ),
            "B"
        );

        setupButtons();
        setupNavigation();
        setupEffects();

        resetPipeline();
        resetResults();

        setText(
            "status",
            "SYSTEM READY — AWAITING LUNAR IMAGERY"
        );

        console.log(
            "LUNARMATCH Lunar Correspondence Engine initialized."
        );
    }

    if (
        document.readyState ===
        "loading"
    ) {
        document.addEventListener(
            "DOMContentLoaded",
            init,
            { once: true }
        );
    } else {
        init();
    }

})();
