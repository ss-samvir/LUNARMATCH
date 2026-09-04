(() => {
    "use strict";

    /* =========================================================
       LUNARMATCH
       Lunar Image Correspondence Engine
       Compatible with current index.html
    ========================================================= */

    const CONFIG = {
        maxImageDimension: 1200,
        maxFeatures: 500,
        patchRadius: 4,
        descriptorSize: 81,
        maxCandidateMatches: 180,
        ransacIterations: 250,
        ransacThreshold: 8,
        minimumVerifiedMatches: 4
    };

    const state = {
        imageAFile: null,
        imageBFile: null,
        imageA: null,
        imageB: null,
        lastAnalysis: null,
        initialized: false
    };

    /* =========================================================
       DOM HELPERS
    ========================================================= */

    const $ = id => document.getElementById(id);

    function setText(id, value) {
        const el = $(id);
        if (el) {
            el.textContent = value;
        }
    }

    function setDisabled(id, disabled) {
        const el = $(id);
        if (el) {
            el.disabled = disabled;
        }
    }

    function round(value, decimals = 1) {
        if (!Number.isFinite(value)) return 0;
        const p = Math.pow(10, decimals);
        return Math.round(value * p) / p;
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function formatTime(ms) {
        if (!Number.isFinite(ms)) return "—";

        if (ms < 1000) {
            return `${Math.round(ms)} ms`;
        }

        return `${round(ms / 1000, 2)} s`;
    }

    /* =========================================================
       STATUS / PIPELINE
    ========================================================= */

    function setStatus(message) {
        setText("status", message);
    }

    function resetPipeline() {
        const stages = [
            "stageAcquire",
            "stagePreprocess",
            "stageExtract",
            "stageMatch",
            "stageVerify",
            "stageScore",
            "stageReport"
        ];

        stages.forEach(id => {
            const el = $(id);
            if (!el) return;

            el.classList.remove(
                "active",
                "complete",
                "processing",
                "done"
            );
        });
    }

    function activateStage(id) {
        const el = $(id);
        if (!el) return;

        el.classList.add("active");
        el.classList.add("processing");
    }

    function completeStage(id) {
        const el = $(id);
        if (!el) return;

        el.classList.remove("processing");
        el.classList.remove("active");
        el.classList.add("complete");
        el.classList.add("done");
    }

    function runStage(id, message, delay = 80) {
        return new Promise(resolve => {
            activateStage(id);
            setStatus(message);

            setTimeout(() => {
                completeStage(id);
                resolve();
            }, delay);
        });
    }

    /* =========================================================
       RESET RESULTS
    ========================================================= */

    function resetResults() {
        const ids = [
            "score",
            "confidence",
            "features",
            "quality",
            "time",

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

        const map = $("correspondenceMap");
        const placeholder = $("visualPlaceholder");

        if (map) {
            map.removeAttribute("src");
            map.style.display = "none";
        }

        if (placeholder) {
            placeholder.style.display = "";
        }

        const note = $("visualNote");

        if (note) {
            note.textContent =
                "● Verified feature correspondences will appear here after analysis.";
        }

        const interpretation = $("interpretation");

        if (interpretation) {
            interpretation.textContent =
                "Upload two lunar images and run the correspondence engine to generate a detailed assessment.";
        }

        setDisabled("downloadReportBtn", true);

        state.lastAnalysis = null;
    }

    /* =========================================================
       IMAGE VALIDATION
    ========================================================= */

    function validateImageFile(file) {
        if (!file) {
            throw new Error("No image selected.");
        }

        if (!file.type.startsWith("image/")) {
            throw new Error("Please select a valid image file.");
        }

        const maxSize = 25 * 1024 * 1024;

        if (file.size > maxSize) {
            throw new Error("Image is too large. Maximum size is 25 MB.");
        }
    }

    /* =========================================================
       IMAGE PREVIEW
    ========================================================= */

    function updatePreview(file, previewId) {
        if (!file) return;

        const preview = $(previewId);

        if (!preview) return;

        const oldUrl = preview.dataset.objectUrl;

        if (oldUrl) {
            URL.revokeObjectURL(oldUrl);
        }

        const url = URL.createObjectURL(file);

        preview.dataset.objectUrl = url;
        preview.src = url;
        preview.style.display = "";
    }

    function handleImage(file, side, input) {
        try {
            validateImageFile(file);

            if (side === "A") {
                state.imageAFile = file;
                updatePreview(file, "previewA");
            } else {
                state.imageBFile = file;
                updatePreview(file, "previewB");
            }

            state.lastAnalysis = null;
            setDisabled("downloadReportBtn", true);

            setStatus(
                `${side === "A" ? "IMAGE A" : "IMAGE B"} READY — ${file.name}`
            );

        } catch (error) {
            console.error("LUNARMATCH upload error:", error);

            setStatus(
                `UPLOAD ERROR — ${error.message}`
            );
        }
    }

    /* =========================================================
       FILE INPUTS
    ========================================================= */

    function setupImageInput(inputId, side) {
        const input = $(inputId);

        if (!input) return;

        input.type = "file";
        input.accept =
            ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,image/*";

        input.addEventListener("change", event => {
            const file =
                event.target.files &&
                event.target.files[0];

            if (file) {
                handleImage(file, side, input);
            }
        });
    }

    /* =========================================================
       DROP ZONES
    ========================================================= */

    function setupDropZone(zone, side) {
        if (!zone) return;

        const input =
            side === "A"
                ? $("fileA")
                : $("fileB");

        if (!input) return;

        /*
         * Explicit click handler.
         * This makes the upload box work even if browser
         * label behaviour becomes inconsistent.
         */

        zone.addEventListener("click", event => {
            event.preventDefault();

            if (
                event.target &&
                event.target.tagName === "INPUT"
            ) {
                return;
            }

            input.click();
        });

        [
            "dragenter",
            "dragover"
        ].forEach(eventName => {

            zone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();
                    event.stopPropagation();

                    zone.classList.add("drag-active");
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

                    zone.classList.remove("drag-active");
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

    /* =========================================================
       IMAGE DECODING
    ========================================================= */

    function loadImage(file) {
        return new Promise((resolve, reject) => {

            const image = new Image();

            image.onload = () => {
                URL.revokeObjectURL(image.dataset.sourceUrl || "");
                resolve(image);
            };

            image.onerror = () => {
                URL.revokeObjectURL(image.dataset.sourceUrl || "");
                reject(
                    new Error(
                        "The selected image could not be decoded."
                    )
                );
            };

            const url = URL.createObjectURL(file);

            image.dataset.sourceUrl = url;
            image.src = url;
        });
    }

    function imageToCanvas(image) {

        const scale = Math.min(
            1,
            CONFIG.maxImageDimension /
            Math.max(
                image.naturalWidth || image.width,
                image.naturalHeight || image.height
            )
        );

        const width = Math.max(
            1,
            Math.round(
                (image.naturalWidth || image.width) *
                scale
            )
        );

        const height = Math.max(
            1,
            Math.round(
                (image.naturalHeight || image.height) *
                scale
            )
        );

        const canvas =
            document.createElement("canvas");

        canvas.width = width;
        canvas.height = height;

        const ctx =
            canvas.getContext("2d", {
                willReadFrequently: true
            });

        ctx.drawImage(
            image,
            0,
            0,
            width,
            height
        );

        return canvas;
    }

    function canvasToGray(canvas) {

        const ctx =
            canvas.getContext("2d", {
                willReadFrequently: true
            });

        const imageData =
            ctx.getImageData(
                0,
                0,
                canvas.width,
                canvas.height
            );

        const data = imageData.data;

        const gray =
            new Float32Array(
                canvas.width *
                canvas.height
            );

        for (
            let i = 0, p = 0;
            i < data.length;
            i += 4, p++
        ) {
            gray[p] =
                0.299 * data[i] +
                0.587 * data[i + 1] +
                0.114 * data[i + 2];
        }

        return {
            data: gray,
            width: canvas.width,
            height: canvas.height
        };
    }

    /* =========================================================
       IMAGE PREPROCESSING
    ========================================================= */

    function normalizeGray(gray) {

        let min = Infinity;
        let max = -Infinity;

        for (let i = 0; i < gray.length; i++) {
            if (gray[i] < min) min = gray[i];
            if (gray[i] > max) max = gray[i];
        }

        const range =
            Math.max(1, max - min);

        const normalized =
            new Float32Array(gray.length);

        for (let i = 0; i < gray.length; i++) {
            normalized[i] =
                ((gray[i] - min) / range) * 255;
        }

        return normalized;
    }

    function calculateContrast(gray) {

        if (!gray.length) return 0;

        let sum = 0;

        for (let i = 0; i < gray.length; i++) {
            sum += gray[i];
        }

        const mean =
            sum / gray.length;

        let variance = 0;

        for (let i = 0; i < gray.length; i++) {

            const diff =
                gray[i] - mean;

            variance += diff * diff;
        }

        variance /=
            gray.length;

        const standardDeviation =
            Math.sqrt(variance);

        return clamp(
            (standardDeviation / 64) * 100,
            0,
            100
        );
    }

    function calculateSharpness(
        gray,
        width,
        height
    ) {

        if (
            width < 3 ||
            height < 3
        ) {
            return 0;
        }

        let sum = 0;
        let count = 0;

        for (
            let y = 1;
            y < height - 1;
            y++
        ) {

            for (
                let x = 1;
                x < width - 1;
                x++
            ) {

                const i =
                    y * width + x;

                const lap =
                    gray[i - width] +
                    gray[i + width] +
                    gray[i - 1] +
                    gray[i + 1] -
                    4 * gray[i];

                sum += lap * lap;
                count++;
            }
        }

        const variance =
            count
                ? sum / count
                : 0;

        return clamp(
            (variance / 1500) * 100,
            0,
            100
        );
    }

    function calculateQuality(
        contrast,
        sharpness
    ) {

        return clamp(
            contrast * 0.45 +
            sharpness * 0.55,
            0,
            100
        );
    }

    /* =========================================================
       GRADIENT / FEATURE DETECTION
    ========================================================= */

    function calculateGradients(
        gray,
        width,
        height
    ) {

        const gx =
            new Float32Array(
                gray.length
            );

        const gy =
            new Float32Array(
                gray.length
            );

        const magnitude =
            new Float32Array(
                gray.length
            );

        for (
            let y = 1;
            y < height - 1;
            y++
        ) {

            for (
                let x = 1;
                x < width - 1;
                x++
            ) {

                const i =
                    y * width + x;

                const dx =
                    gray[i + 1] -
                    gray[i - 1];

                const dy =
                    gray[i + width] -
                    gray[i - width];

                gx[i] = dx;
                gy[i] = dy;

                magnitude[i] =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );
            }
        }

        return {
            gx,
            gy,
            magnitude
        };
    }

    function detectFeatures(
        gray,
        width,
        height
    ) {

        if (
            width < 20 ||
            height < 20
        ) {
            return [];
        }

        const gradients =
            calculateGradients(
                gray,
                width,
                height
            );

        const magnitude =
            gradients.magnitude;

        const candidates = [];

        const border = 8;

        for (
            let y = border;
            y < height - border;
            y += 2
        ) {

            for (
                let x = border;
                x < width - border;
                x += 2
            ) {

                const i =
                    y * width + x;

                const center =
                    magnitude[i];

                if (center < 12) {
                    continue;
                }

                let localMax = true;

                for (
                    let dy = -2;
                    dy <= 2 && localMax;
                    dy++
                ) {

                    for (
                        let dx = -2;
                        dx <= 2;
                        dx++
                    ) {

                        if (
                            dx === 0 &&
                            dy === 0
                        ) {
                            continue;
                        }

                        const ni =
                            (y + dy) *
                            width +
                            (x + dx);

                        if (
                            magnitude[ni] >
                            center
                        ) {
                            localMax = false;
                            break;
                        }
                    }
                }

                if (!localMax) {
                    continue;
                }

                /*
                 * Corner strength estimated from
                 * directional gradient energy.
                 */

                let xx = 0;
                let yy = 0;
                let xy = 0;

                for (
                    let dy = -3;
                    dy <= 3;
                    dy++
                ) {

                    for (
                        let dx = -3;
                        dx <= 3;
                        dx++
                    ) {

                        const px =
                            clamp(
                                x + dx,
                                1,
                                width - 2
                            );

                        const py =
                            clamp(
                                y + dy,
                                1,
                                height - 2
                            );

                        const pi =
                            py * width + px;

                        const gxx =
                            gradients.gx[pi];

                        const gyy =
                            gradients.gy[pi];

                        xx += gxx * gxx;
                        yy += gyy * gyy;
                        xy += gxx * gyy;
                    }
                }

                const trace =
                    xx + yy;

                const determinant =
                    xx * yy -
                    xy * xy;

                const response =
                    determinant -
                    0.04 *
                    trace *
                    trace;

                if (response > 1000) {

                    candidates.push({
                        x,
                        y,
                        score: response
                    });
                }
            }
        }

        candidates.sort(
            (a, b) =>
                b.score - a.score
        );

        const selected = [];

        const minDistance = 10;

        for (
            const candidate of candidates
        ) {

            let tooClose = false;

            for (
                const existing of selected
            ) {

                const dx =
                    candidate.x -
                    existing.x;

                const dy =
                    candidate.y -
                    existing.y;

                if (
                    dx * dx +
                    dy * dy <
                    minDistance *
                    minDistance
                ) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                selected.push(candidate);
            }

            if (
                selected.length >=
                CONFIG.maxFeatures
            ) {
                break;
            }
        }

        return selected;
    }

    /* =========================================================
       PATCH DESCRIPTORS
    ========================================================= */

    function descriptorAt(
        gray,
        width,
        height,
        x,
        y
    ) {

        const radius =
            CONFIG.patchRadius;

        const values = [];

        for (
            let dy = -radius;
            dy <= radius;
            dy++
        ) {

            for (
                let dx = -radius;
                dx <= radius;
                dx++
            ) {

                const px =
                    clamp(
                        Math.round(x + dx),
                        0,
                        width - 1
                    );

                const py =
                    clamp(
                        Math.round(y + dy),
                        0,
                        height - 1
                    );

                values.push(
                    gray[
                        py * width + px
                    ]
                );
            }
        }

        let mean = 0;

        for (
            const value of values
        ) {
            mean += value;
        }

        mean /=
            values.length;

        let variance = 0;

        for (
            const value of values
        ) {

            const diff =
                value - mean;

            variance +=
                diff * diff;
        }

        variance /=
            values.length;

        const std =
            Math.sqrt(
                variance
            ) || 1;

        return Float32Array.from(
            values.map(
                value =>
                    (value - mean) /
                    std
            )
        );
    }

    function buildDescriptors(
        gray,
        width,
        height,
        features
    ) {

        return features.map(
            feature => ({
                ...feature,
                descriptor:
                    descriptorAt(
                        gray,
                        width,
                        height,
                        feature.x,
                        feature.y
                    )
            })
        );
    }

    /* =========================================================
       DESCRIPTOR DISTANCE
    ========================================================= */

    function descriptorDistance(a, b) {

        let sum = 0;

        const length =
            Math.min(
                a.length,
                b.length
            );

        for (
            let i = 0;
            i < length;
            i++
        ) {

            const diff =
                a[i] - b[i];

            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    /* =========================================================
       FEATURE MATCHING
    ========================================================= */

    function matchFeatures(
        featuresA,
        featuresB
    ) {

        const forward = [];

        for (
            let i = 0;
            i < featuresA.length;
            i++
        ) {

            const featureA =
                featuresA[i];

            let best = null;
            let second = null;

            for (
                let j = 0;
                j < featuresB.length;
                j++
            ) {

                const featureB =
                    featuresB[j];

                const distance =
                    descriptorDistance(
                        featureA.descriptor,
                        featureB.descriptor
                    );

                if (
                    !best ||
                    distance < best.distance
                ) {

                    second = best;

                    best = {
                        a: i,
                        b: j,
                        distance
                    };

                } else if (
                    !second ||
                    distance <
                    second.distance
                ) {

                    second = {
                        a: i,
                        b: j,
                        distance
                    };
                }
            }

            if (!best) continue;

            const ratio =
                second
                    ? best.distance /
                      Math.max(
                          second.distance,
                          0.0001
                      )
                    : 1;

            if (
                ratio <= 0.90
            ) {

                forward.push({
                    ...best,
                    ratio
                });
            }
        }

        /*
         * Reciprocal matching.
         */

        const reverseBest =
            new Map();

        for (
            let j = 0;
            j < featuresB.length;
            j++
        ) {

            const featureB =
                featuresB[j];

            let best = null;

            for (
                let i = 0;
                i < featuresA.length;
                i++
            ) {

                const featureA =
                    featuresA[i];

                const distance =
                    descriptorDistance(
                        featureB.descriptor,
                        featureA.descriptor
                    );

                if (
                    !best ||
                    distance <
                    best.distance
                ) {

                    best = {
                        a: i,
                        b: j,
                        distance
                    };
                }
            }

            if (best) {
                reverseBest.set(
                    j,
                    best
                );
            }
        }

        const reciprocal = [];

        for (
            const match of forward
        ) {

            const reverse =
                reverseBest.get(
                    match.b
                );

            if (
                reverse &&
                reverse.a === match.a
            ) {

                reciprocal.push(
                    match
                );
            }
        }

        reciprocal.sort(
            (a, b) =>
                a.distance -
                b.distance
        );

        return {
            rawMatches:
                featuresA.length,

            candidates:
                forward,

            reciprocal:
                reciprocal.slice(
                    0,
                    CONFIG.maxCandidateMatches
                )
        };
    }

    /* =========================================================
       AFFINE MODEL
    ========================================================= */

    function solveAffine(
        pairs
    ) {

        if (pairs.length < 3) {
            return null;
        }

        let sx = 0;
        let sy = 0;
        let sxx = 0;
        let syy = 0;
        let sxy = 0;

        let tx = 0;
        let ty = 0;
        let stx = 0;
        let sty = 0;

        /*
         * Solve using normal equations.
         */

        const A = [];
        const bx = [];
        const by = [];

        for (
            const pair of pairs
        ) {

            const x = pair.a.x;
            const y = pair.a.y;

            const X = pair.b.x;
            const Y = pair.b.y;

            A.push([
                x,
                y,
                1
            ]);

            bx.push(X);
            by.push(Y);
        }

        function solve3(M, v) {

            const m =
                M.map(
                    row =>
                        row.slice()
                );

            const b =
                v.slice();

            for (
                let i = 0;
                i < 3;
                i++
            ) {

                let pivot = i;

                for (
                    let r = i + 1;
                    r < 3;
                    r++
                ) {

                    if (
                        Math.abs(
                            m[r][i]
                        ) >
                        Math.abs(
                            m[pivot][i]
                        )
                    ) {
                        pivot = r;
                    }
                }

                if (
                    Math.abs(
                        m[pivot][i]
                    ) < 1e-9
                ) {
                    return null;
                }

                [
                    m[i],
                    m[pivot]
                ] = [
                    m[pivot],
                    m[i]
                ];

                [
                    b[i],
                    b[pivot]
                ] = [
                    b[pivot],
                    b[i]
                ];

                const div =
                    m[i][i];

                for (
                    let c = i;
                    c < 3;
                    c++
                ) {
                    m[i][c] /= div;
                }

                b[i] /= div;

                for (
                    let r = 0;
                    r < 3;
                    r++
                ) {

                    if (r === i) {
                        continue;
                    }

                    const factor =
                        m[r][i];

                    for (
                        let c = i;
                        c < 3;
                        c++
                    ) {

                        m[r][c] -=
                            factor *
                            m[i][c];
                    }

                    b[r] -=
                        factor *
                        b[i];
                }
            }

            return b;
        }

        const normal =
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ];

        const nx =
            [0, 0, 0];

        const ny =
            [0, 0, 0];

        for (
            let i = 0;
            i < A.length;
            i++
        ) {

            const row =
                A[i];

            for (
                let r = 0;
                r < 3;
                r++
            ) {

                for (
                    let c = 0;
                    c < 3;
                    c++
                ) {

                    normal[r][c] +=
                        row[r] *
                        row[c];
                }

                nx[r] +=
                    row[r] *
                    bx[i];

                ny[r] +=
                    row[r] *
                    by[i];
            }
        }

        const cx =
            solve3(
                normal,
                nx
            );

        const cy =
            solve3(
                normal,
                ny
            );

        if (!cx || !cy) {
            return null;
        }

        return {
            a: cx[0],
            b: cx[1],
            c: cx[2],

            d: cy[0],
            e: cy[1],
            f: cy[2]
        };
    }

    function transformPoint(
        model,
        point
    ) {

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

    function pointDistance(a, b) {

        const dx =
            a.x - b.x;

        const dy =
            a.y - b.y;

        return Math.sqrt(
            dx * dx +
            dy * dy
        );
    }

    /* =========================================================
       RANSAC GEOMETRIC VERIFICATION
    ========================================================= */

    function verifyGeometry(
        matches,
        featuresA,
        featuresB
    ) {

        if (
            matches.length <
            CONFIG.minimumVerifiedMatches
        ) {

            return {
                model: null,
                inliers: [],
                ratio: 0,
                consistency: 0
            };
        }

        const pairs =
            matches.map(match => ({
                a: featuresA[match.a],
                b: featuresB[match.b],
                match
            }));

        let bestModel = null;
        let bestInliers = [];

        for (
            let iteration = 0;
            iteration <
            CONFIG.ransacIterations;
            iteration++
        ) {

            const shuffled =
                pairs
                    .slice()
                    .sort(
                        () =>
                            Math.random() -
                            0.5
                    );

            const sample =
                shuffled.slice(
                    0,
                    3
                );

            const model =
                solveAffine(
                    sample
                );

            if (!model) {
                continue;
            }

            const inliers = [];

            for (
                const pair of pairs
            ) {

                const projected =
                    transformPoint(
                        model,
                        pair.a
                    );

                const error =
                    pointDistance(
                        projected,
                        pair.b
                    );

                if (
                    error <=
                    CONFIG.ransacThreshold
                ) {

                    inliers.push({
                        ...pair,
                        error
                    });
                }
            }

            if (
                inliers.length >
                bestInliers.length
            ) {

                bestInliers =
                    inliers;

                bestModel =
                    model;
            }
        }

        /*
         * Refine the model using all inliers.
         */

        if (
            bestInliers.length >= 3
        ) {

            const refined =
                solveAffine(
                    bestInliers
                );

            if (refined) {
                bestModel =
                    refined;
            }
        }

        const ratio =
            matches.length
                ? bestInliers.length /
                  matches.length
                : 0;

        const meanError =
            bestInliers.length
                ? bestInliers.reduce(
                      (sum, item) =>
                          sum + item.error,
                      0
                  ) /
                  bestInliers.length
                : Infinity;

        const consistency =
            clamp(
                ratio *
                Math.max(
                    0,
                    1 -
                    meanError /
                    20
                ) *
                100,
                0,
                100
            );

        return {
            model: bestModel,
            inliers: bestInliers,
            ratio,
            consistency
        };
    }

    /* =========================================================
       FEATURE COVERAGE
    ========================================================= */

    function calculateCoverage(
        featuresA,
        verifiedMatches,
        widthA,
        heightA
    ) {

        if (
            !featuresA.length ||
            !verifiedMatches.length
        ) {
            return 0;
        }

        const cellsX = 8;
        const cellsY = 6;

        const occupied =
            new Set();

        for (
            const item of verifiedMatches
        ) {

            const x =
                clamp(
                    item.a.x /
                    widthA,
                    0,
                    0.9999
                );

            const y =
                clamp(
                    item.a.y /
                    heightA,
                    0,
                    0.9999
                );

            const cellX =
                Math.floor(
                    x * cellsX
                );

            const cellY =
                Math.floor(
                    y * cellsY
                );

            occupied.add(
                `${cellX}:${cellY}`
            );
        }

        return clamp(
            (
                occupied.size /
                (cellsX * cellsY)
            ) * 100,
            0,
            100
        );
    }

    /* =========================================================
       MATCH QUALITY / SCORE
    ========================================================= */

    function calculateScore(
        qualityA,
        qualityB,
        candidateCount,
        verifiedCount,
        inlierRatio,
        coverage,
        geometricConsistency
    ) {

        const imageQuality =
            (
                qualityA +
                qualityB
            ) / 2;

        const verification =
            clamp(
                inlierRatio * 100,
                0,
                100
            );

        const matchStrength =
            clamp(
                (
                    verifiedCount /
                    Math.max(
                        candidateCount,
                        1
                    )
                ) * 100,
                0,
                100
            );

        const score =
            imageQuality * 0.15 +
            verification * 0.35 +
            coverage * 0.15 +
            geometricConsistency * 0.25 +
            matchStrength * 0.10;

        return clamp(
            Math.round(score),
            0,
            100
        );
    }

    function getConfidence(score) {

        if (score >= 85) {
            return {
                label: "HIGH",
                value: score
            };
        }

        if (score >= 65) {
            return {
                label: "MODERATE",
                value: score
            };
        }

        if (score >= 40) {
            return {
                label: "LOW",
                value: score
            };
        }

        return {
            label: "VERY LOW",
            value: score
        };
    }

    /* =========================================================
       INTERPRETATION
    ========================================================= */

    function generateInterpretation(result) {

        const score =
            result.score;

        const verified =
            result.verifiedMatches;

        const ratio =
            result.inlierRatio;

        if (
            score >= 85 &&
            verified >= 15
        ) {

            return (
                "The correspondence engine identifies a strong " +
                "geometric relationship between the two lunar " +
                "observations. A substantial number of candidate " +
                "features remain consistent after robust geometric " +
                "verification, indicating a high-confidence visual " +
                "correspondence."
            );
        }

        if (
            score >= 65 &&
            verified >= 8
        ) {

            return (
                "The analysis indicates a meaningful correspondence " +
                "between the two lunar images. Several visual " +
                "features exhibit geometric consistency, although " +
                "the available evidence is not as strong as a " +
                "high-confidence correspondence."
            );
        }

        if (
            ratio >= 0.20 &&
            verified >= 4
        ) {

            return (
                "The system detected a limited set of geometrically " +
                "consistent feature correspondences. Additional " +
                "imagery or improved image quality may be required " +
                "for stronger verification."
            );
        }

        return (
            "The correspondence evidence is currently weak. " +
            "The images contain insufficient geometrically " +
            "consistent feature matches for a strong correspondence " +
            "assessment."
        );
    }

    /* =========================================================
       CORRESPONDENCE MAP
    ========================================================= */

    function createCorrespondenceMap(
        imageA,
        imageB,
        featuresA,
        featuresB,
        verifiedMatches
    ) {

        const output =
            $("correspondenceMap");

        const placeholder =
            $("visualPlaceholder");

        if (!output) {
            return;
        }

        const width = 1200;
        const height = 560;

        const canvas =
            document.createElement(
                "canvas"
            );

        canvas.width = width;
        canvas.height = height;

        const ctx =
            canvas.getContext("2d");

        ctx.fillStyle =
            "#05070b";

        ctx.fillRect(
            0,
            0,
            width,
            height
        );

        const gap = 12;

        const panelWidth =
            (width - gap) / 2;

        const panelHeight =
            height;

        function drawImageCover(
            image,
            x,
            y,
            w,
            h
        ) {

            const iw =
                image.naturalWidth ||
                image.width;

            const ih =
                image.naturalHeight ||
                image.height;

            const scale =
                Math.min(
                    w / iw,
                    h / ih
                );

            const dw =
                iw * scale;

            const dh =
                ih * scale;

            const dx =
                x +
                (w - dw) / 2;

            const dy =
                y +
                (h - dh) / 2;

            ctx.drawImage(
                image,
                dx,
                dy,
                dw,
                dh
            );

            return {
                x: dx,
                y: dy,
                width: dw,
                height: dh,
                scale
            };
        }

        const rectA =
            drawImageCover(
                imageA,
                0,
                0,
                panelWidth,
                panelHeight
            );

        const rectB =
            drawImageCover(
                imageB,
                panelWidth + gap,
                0,
                panelWidth,
                panelHeight
            );

        /*
         * Draw a subtle separator.
         */

        ctx.strokeStyle =
            "rgba(255,255,255,0.25)";

        ctx.lineWidth = 1;

        ctx.beginPath();

        ctx.moveTo(
            panelWidth + gap / 2,
            0
        );

        ctx.lineTo(
            panelWidth + gap / 2,
            height
        );

        ctx.stroke();

        /*
         * Draw correspondence lines.
         */

        for (
            let i = 0;
            i < verifiedMatches.length;
            i++
        ) {

            const item =
                verifiedMatches[i];

            const pointA =
                featuresA[item.match.a];

            const pointB =
                featuresB[item.match.b];

            if (!pointA || !pointB) {
                continue;
            }

            const ax =
                rectA.x +
                pointA.x *
                rectA.scale;

            const ay =
                rectA.y +
                pointA.y *
                rectA.scale;

            const bx =
                rectB.x +
                pointB.x *
                rectB.scale;

            const by =
                rectB.y +
                pointB.y *
                rectB.scale;

            const alpha =
                Math.max(
                    0.25,
                    1 -
                    item.error / 15
                );

            ctx.strokeStyle =
                `rgba(255,255,255,${alpha})`;

            ctx.lineWidth =
                i < 25
                    ? 1.5
                    : 0.8;

            ctx.beginPath();

            ctx.moveTo(
                ax,
                ay
            );

            ctx.lineTo(
                bx,
                by
            );

            ctx.stroke();

            ctx.fillStyle =
                "rgba(255,255,255,0.95)";

            ctx.beginPath();

            ctx.arc(
                ax,
                ay,
                3,
                0,
                Math.PI * 2
            );

            ctx.fill();

            ctx.beginPath();

            ctx.arc(
                bx,
                by,
                3,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }

        /*
         * Labels.
         */

        ctx.font =
            "600 13px Arial";

        ctx.fillStyle =
            "rgba(255,255,255,0.9)";

        ctx.fillText(
            "IMAGE A",
            18,
            28
        );

        ctx.fillText(
            "IMAGE B",
            panelWidth + gap + 18,
            28
        );

        output.src =
            canvas.toDataURL(
                "image/png"
            );

        output.style.display = "";

        if (placeholder) {
            placeholder.style.display =
                "none";
        }

        const note =
            $("visualNote");

        if (note) {
            note.textContent =
                `● ${verifiedMatches.length} verified feature correspondences visualized.`;
        }
    }

    /* =========================================================
       DISPLAY RESULTS
    ========================================================= */

    function displayResults(result) {

        setText(
            "score",
            `${result.score}%`
        );

        setText(
            "confidence",
            `${result.confidence.label} (${result.confidence.value}%)`
        );

        setText(
            "features",
            result.verifiedMatches
        );

        setText(
            "quality",
            `${round(
                result.averageQuality,
                1
            )}%`
        );

        setText(
            "time",
            formatTime(
                result.processTime
            )
        );

        /* IMAGE A */

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
            `${round(
                result.imageA.contrast,
                1
            )}%`
        );

        setText(
            "sharpnessA",
            `${round(
                result.imageA.sharpness,
                1
            )}%`
        );

        setText(
            "qualityScoreA",
            `${round(
                result.imageA.quality,
                1
            )}%`
        );

        /* IMAGE B */

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
            `${round(
                result.imageB.contrast,
                1
            )}%`
        );

        setText(
            "sharpnessB",
            `${round(
                result.imageB.sharpness,
                1
            )}%`
        );

        setText(
            "qualityScoreB",
            `${round(
                result.imageB.quality,
                1
            )}%`
        );

        /* MATCHING */

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
            `${round(
                result.featureCoverage,
                1
            )}%`
        );

        setText(
            "correspondenceStrength",
            `${round(
                result.correspondenceStrength,
                1
            )}%`
        );

        /* GEOMETRY */

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

        setStatus(
            "ANALYSIS COMPLETE — CORRESPONDENCE VERIFIED"
        );

        setDisabled(
            "downloadReportBtn",
            false
        );
    }

    /* =========================================================
       MAIN IMAGE ANALYSIS
    ========================================================= */

    async function analyzeSingleImage(
        file
    ) {

        const image =
            await loadImage(file);

        const canvas =
            imageToCanvas(image);

        const grayData =
            canvasToGray(canvas);

        const normalized =
            normalizeGray(
                grayData.data
            );

        const contrast =
            calculateContrast(
                normalized
            );

        const sharpness =
            calculateSharpness(
                normalized,
                grayData.width,
                grayData.height
            );

        const quality =
            calculateQuality(
                contrast,
                sharpness
            );

        const features =
            detectFeatures(
                normalized,
                grayData.width,
                grayData.height
            );

        const descriptors =
            buildDescriptors(
                normalized,
                grayData.width,
                grayData.height,
                features
            );

        return {
            image,
            width:
                grayData.width,
            height:
                grayData.height,
            gray:
                normalized,
            contrast,
            sharpness,
            quality,
            features:
                descriptors
        };
    }

    /* =========================================================
       COMPLETE ANALYSIS
    ========================================================= */

    async function analyzeImages() {

        const analyzeButton =
            $("compareBtn");

        if (!state.imageAFile ||
            !state.imageBFile) {

            setStatus(
                "ANALYSIS ERROR — SELECT BOTH IMAGES FIRST"
            );

            return;
        }

        if (analyzeButton) {
            analyzeButton.disabled = true;
        }

        resetPipeline();

        const startTime =
            performance.now();

        try {

            /* =============================================
               01 — ACQUIRE
            ============================================= */

            await runStage(
                "stageAcquire",
                "ACQUIRING LUNAR IMAGERY",
                120
            );

            /* =============================================
               02 — PREPROCESS
            ============================================= */

            await runStage(
                "stagePreprocess",
                "PREPROCESSING IMAGE DATA",
                120
            );

            const analysisA =
                await analyzeSingleImage(
                    state.imageAFile
                );

            const analysisB =
                await analyzeSingleImage(
                    state.imageBFile
                );

            state.imageA =
                analysisA;

            state.imageB =
                analysisB;

            /* =============================================
               03 — EXTRACT
            ============================================= */

            await runStage(
                "stageExtract",
                "EXTRACTING LUNAR FEATURES",
                150
            );

            /* =============================================
               04 — MATCH
            ============================================= */

            await runStage(
                "stageMatch",
                "MATCHING FEATURE DESCRIPTORS",
                150
            );

            const matching =
                matchFeatures(
                    analysisA.features,
                    analysisB.features
                );

            /* =============================================
               05 — VERIFY
            ============================================= */

            await runStage(
                "stageVerify",
                "RUNNING RANSAC GEOMETRIC VERIFICATION",
                180
            );

            const geometry =
                verifyGeometry(
                    matching.reciprocal,
                    analysisA.features,
                    analysisB.features
                );

            /* =============================================
               COVERAGE
            ============================================= */

            const coverage =
                calculateCoverage(
                    analysisA.features,
                    geometry.inliers,
                    analysisA.width,
                    analysisA.height
                );

            const candidateMatches =
                matching.reciprocal.length;

            const verifiedMatches =
                geometry.inliers.length;

            const correspondenceStrength =
                candidateMatches
                    ? clamp(
                          (
                              verifiedMatches /
                              candidateMatches
                          ) * 100,
                          0,
                          100
                      )
                    : 0;

            /* =============================================
               SCORE
            ============================================= */

            await runStage(
                "stageScore",
                "CALCULATING CORRESPONDENCE SCORE",
                140
            );

            const score =
                calculateScore(
                    analysisA.quality,
                    analysisB.quality,
                    candidateMatches,
                    verifiedMatches,
                    geometry.ratio,
                    coverage,
                    geometry.consistency
                );

            const confidence =
                getConfidence(score);

            const averageQuality =
                (
                    analysisA.quality +
                    analysisB.quality
                ) / 2;

            const homographyStatus =
                geometry.model
                    ? "VALID"
                    : "NOT ESTABLISHED";

            let verificationStatus;

            if (
                verifiedMatches >=
                    CONFIG.minimumVerifiedMatches &&
                geometry.ratio >= 0.25
            ) {

                verificationStatus =
                    "VERIFIED";

            } else if (
                verifiedMatches >= 3
            ) {

                verificationStatus =
                    "PARTIAL";

            } else {

                verificationStatus =
                    "INSUFFICIENT";
            }

            const processTime =
                performance.now() -
                startTime;

            const result = {
                score,
                confidence,

                verifiedMatches,

                averageQuality,

                processTime,

                imageA: {
                    width:
                        analysisA.width,
                    height:
                        analysisA.height,
                    keypoints:
                        analysisA.features.length,
                    contrast:
                        analysisA.contrast,
                    sharpness:
                        analysisA.sharpness,
                    quality:
                        analysisA.quality
                },

                imageB: {
                    width:
                        analysisB.width,
                    height:
                        analysisB.height,
                    keypoints:
                        analysisB.features.length,
                    contrast:
                        analysisB.contrast,
                    sharpness:
                        analysisB.sharpness,
                    quality:
                        analysisB.quality
                },

                rawMatches:
                    matching.rawMatches,

                candidateMatches,

                featureCoverage:
                    coverage,

                correspondenceStrength,

                inlierRatio:
                    geometry.ratio,

                geometricConsistency:
                    geometry.consistency,

                homographyStatus,

                verificationStatus,

                modelType:
                    "AFFINE / RANSAC",

                interpretation: ""
            };

            result.interpretation =
                generateInterpretation(
                    result
                );

            /* =============================================
               DRAW MAP
            ============================================= */

            createCorrespondenceMap(
                analysisA.image,
                analysisB.image,
                analysisA.features,
                analysisB.features,
                geometry.inliers
            );

            displayResults(result);

            state.lastAnalysis =
                result;

            /* =============================================
               07 — REPORT
            ============================================= */

            await runStage(
                "stageReport",
                "ANALYSIS REPORT READY",
                100
            );

            setStatus(
                "SYSTEM READY — ANALYSIS COMPLETE"
            );

        } catch (error) {

            console.error(
                "LUNARMATCH analysis error:",
                error
            );

            setStatus(
                `ANALYSIS ERROR — ${error.message || "Unable to complete analysis"}`
            );

        } finally {

            if (analyzeButton) {
                analyzeButton.disabled =
                    false;
            }
        }
    }

    /* =========================================================
       PDF REPORT
    ========================================================= */

    async function downloadReport() {

        if (!state.lastAnalysis) {
            return;
        }

        const result =
            state.lastAnalysis;

        setStatus(
            "GENERATING ANALYSIS REPORT..."
        );

        /*
         * First attempt the existing backend report
         * endpoint. If unavailable, use a printable
         * professional report window.
         */

        try {

            const response =
                await fetch(
                    "/api/report",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body:
                            JSON.stringify(
                                result
                            )
                    }
                );

            if (response.ok) {

                const blob =
                    await response.blob();

                if (
                    blob &&
                    blob.size > 0
                ) {

                    const url =
                        URL.createObjectURL(
                            blob
                        );

                    const link =
                        document.createElement(
                            "a"
                        );

                    link.href = url;

                    link.download =
                        "LUNARMATCH_Correspondence_Report.pdf";

                    document.body.appendChild(
                        link
                    );

                    link.click();

                    link.remove();

                    URL.revokeObjectURL(
                        url
                    );

                    setStatus(
                        "REPORT GENERATED SUCCESSFULLY"
                    );

                    return;
                }
            }

        } catch (error) {
            console.warn(
                "Backend PDF unavailable; using browser report.",
                error
            );
        }

        /*
         * Fallback printable report.
         */

        const reportWindow =
            window.open(
                "",
                "_blank"
            );

        if (!reportWindow) {

            setStatus(
                "REPORT ERROR — ALLOW POPUPS TO GENERATE REPORT"
            );

            return;
        }

        const map =
            $("correspondenceMap");

        const mapSrc =
            map &&
            map.src
                ? map.src
                : "";

        const html = `
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<title>LUNARMATCH Correspondence Report</title>

<style>

body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 40px;
    color: #111;
    background: #fff;
}

h1 {
    margin-bottom: 4px;
}

h2 {
    margin-top: 32px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 8px;
}

.subtitle {
    color: #666;
    margin-bottom: 30px;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(2, 1fr);
    gap: 12px;
}

.card {
    border: 1px solid #ddd;
    padding: 14px;
}

.label {
    display: block;
    color: #666;
    font-size: 11px;
    margin-bottom: 5px;
}

.value {
    font-size: 20px;
    font-weight: 700;
}

.map {
    width: 100%;
    max-width: 1000px;
    margin-top: 15px;
}

.interpretation {
    line-height: 1.7;
}

.footer {
    margin-top: 40px;
    padding-top: 15px;
    border-top: 1px solid #ddd;
    color: #777;
    font-size: 12px;
}

@media print {
    body {
        margin: 20px;
    }
}

</style>

</head>

<body>

<h1>LUNARMATCH</h1>

<div class="subtitle">
Lunar Image Correspondence Analysis Report
</div>

<h2>Match Overview</h2>

<div class="grid">

<div class="card">
<span class="label">OVERALL MATCH</span>
<div class="value">${result.score}%</div>
</div>

<div class="card">
<span class="label">CONFIDENCE</span>
<div class="value">
${result.confidence.label}
</div>
</div>

<div class="card">
<span class="label">VERIFIED FEATURES</span>
<div class="value">
${result.verifiedMatches}
</div>
</div>

<div class="card">
<span class="label">IMAGE QUALITY</span>
<div class="value">
${round(result.averageQuality, 1)}%
</div>
</div>

</div>

<h2>Image A</h2>

<div class="grid">

<div class="card">
<span class="label">RESOLUTION</span>
<div class="value">
${result.imageA.width} × ${result.imageA.height}
</div>
</div>

<div class="card">
<span class="label">KEYPOINTS</span>
<div class="value">
${result.imageA.keypoints}
</div>
</div>

<div class="card">
<span class="label">CONTRAST</span>
<div class="value">
${round(result.imageA.contrast, 1)}%
</div>
</div>

<div class="card">
<span class="label">SHARPNESS</span>
<div class="value">
${round(result.imageA.sharpness, 1)}%
</div>
</div>

</div>

<h2>Image B</h2>

<div class="grid">

<div class="card">
<span class="label">RESOLUTION</span>
<div class="value">
${result.imageB.width} × ${result.imageB.height}
</div>
</div>

<div class="card">
<span class="label">KEYPOINTS</span>
<div class="value">
${result.imageB.keypoints}
</div>
</div>

<div class="card">
<span class="label">CONTRAST</span>
<div class="value">
${round(result.imageB.contrast, 1)}%
</div>
</div>

<div class="card">
<span class="label">SHARPNESS</span>
<div class="value">
${round(result.imageB.sharpness, 1)}%
</div>
</div>

</div>

<h2>Feature Correspondence</h2>

<div class="grid">

<div class="card">
<span class="label">RAW MATCHES</span>
<div class="value">
${result.rawMatches}
</div>
</div>

<div class="card">
<span class="label">CANDIDATE MATCHES</span>
<div class="value">
${result.candidateMatches}
</div>
</div>

<div class="card">
<span class="label">VERIFIED MATCHES</span>
<div class="value">
${result.verifiedMatches}
</div>
</div>

<div class="card">
<span class="label">FEATURE COVERAGE</span>
<div class="value">
${round(result.featureCoverage, 1)}%
</div>
</div>

</div>

<h2>Geometric Verification</h2>

<div class="grid">

<div class="card">
<span class="label">INLIER RATIO</span>
<div class="value">
${round(result.inlierRatio * 100, 1)}%
</div>
</div>

<div class="card">
<span class="label">GEOMETRIC CONSISTENCY</span>
<div class="value">
${round(result.geometricConsistency, 1)}%
</div>
</div>

<div class="card">
<span class="label">HOMOGRAPHY</span>
<div class="value">
${result.homographyStatus}
</div>
</div>

<div class="card">
<span class="label">VERIFICATION</span>
<div class="value">
${result.verificationStatus}
</div>
</div>

</div>

<h2>Correspondence Visualization</h2>

${
    mapSrc
        ? `<img class="map" src="${mapSrc}" alt="Correspondence map">`
        : `<p>No correspondence visualization available.</p>`
}

<h2>Automated Interpretation</h2>

<p class="interpretation">
${result.interpretation}
</p>

<div class="footer">
LUNARMATCH · Lunar Intelligence Platform ·
Chandrayaan-2 Data Interface · 2026
</div>

<script>
window.onload = function () {
    setTimeout(function () {
        window.print();
    }, 500);
};
</script>

</body>
</html>
`;

        reportWindow.document.open();
        reportWindow.document.write(html);
        reportWindow.document.close();

        setStatus(
            "ANALYSIS REPORT READY"
        );
    }

    /* =========================================================
       BUTTONS
    ========================================================= */

    function setupButtons() {

        /*
         * IMPORTANT:
         * Current HTML uses compareBtn.
         */

        const analyze =
            $("compareBtn");

        if (analyze) {

            analyze.addEventListener(
                "click",
                event => {

                    event.preventDefault();

                    analyzeImages();
                }
            );
        }

        const report =
            $("downloadReportBtn");

        if (report) {

            report.addEventListener(
                "click",
                event => {

                    event.preventDefault();

                    downloadReport();
                }
            );
        }

        const registration =
            $("registrationBtn");

        if (registration) {

            registration.addEventListener(
                "click",
                () => {

                    setStatus(
                        "REGISTRATION PORTAL — COMING SOON"
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
                        document.querySelector(
                            "#contact"
                        );

                    if (contactSection) {
                        contactSection.scrollIntoView({
                            behavior:
                                "smooth"
                        });
                    }
                }
            );
        }
    }

    /* =========================================================
       NAVIGATION
    ========================================================= */

    function setupNavigation() {

        const navLinks =
            document.querySelectorAll(
                "nav a"
            );

        navLinks.forEach(link => {

            link.addEventListener(
                "click",
                () => {

                    navLinks.forEach(
                        item =>
                            item.classList.remove(
                                "active"
                            )
                    );

                    link.classList.add(
                        "active"
                    );
                }
            );
        });
    }

    /* =========================================================
       VISUAL EFFECTS
       Does NOT change the existing design.
    ========================================================= */

    function setupEffects() {

        const dropZones =
            document.querySelectorAll(
                ".drop-new"
            );

        dropZones.forEach(zone => {

            zone.addEventListener(
                "mouseenter",
                () => {
                    zone.classList.add(
                        "hover-active"
                    );
                }
            );

            zone.addEventListener(
                "mouseleave",
                () => {
                    zone.classList.remove(
                        "hover-active"
                    );
                }
            );
        });
    }

    /* =========================================================
       GLOBAL ERROR HANDLING
    ========================================================= */

    window.addEventListener(
        "error",
        event => {

            console.error(
                "LUNARMATCH runtime error:",
                event.error ||
                event.message
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

    /* =========================================================
       INITIALIZATION
    ========================================================= */

    function init() {

        if (
            window.__LUNARMATCH_INITIALIZED__
        ) {
            return;
        }

        window.__LUNARMATCH_INITIALIZED__ =
            true;

        state.initialized =
            true;

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

        setStatus(
            "SYSTEM READY — AWAITING LUNAR IMAGERY"
        );

        console.log(
            "LUNARMATCH Lunar Correspondence Engine initialized successfully."
        );
    }

    /* =========================================================
       START
    ========================================================= */

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
