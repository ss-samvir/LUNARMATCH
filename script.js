(() => {
    "use strict";

    /* =========================================================
       LUNARMATCH
       Lunar Image Correspondence Engine
       Compatible with current index.html
    ========================================================= */

    const $ = (id) => document.getElementById(id);

    const state = {
        imageAFile: null,
        imageBFile: null,
        imageA: null,
        imageB: null,
        lastAnalysis: null,
        initialized: false
    };

    const CONFIG = {
        maxImageSize: 1600,
        maxFeatures: 500,
        patchRadius: 5,
        matchRatio: 0.78,
        ransacIterations: 180,
        ransacThreshold: 10
    };


    /* =========================================================
       BASIC HELPERS
    ========================================================= */

    function setText(id, value) {
        const el = $(id);
        if (el) el.textContent = value;
    }

    function setDisabled(id, value) {
        const el = $(id);
        if (el) el.disabled = value;
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function round(value, digits = 1) {
        if (!Number.isFinite(value)) return 0;
        const p = Math.pow(10, digits);
        return Math.round(value * p) / p;
    }

    function percent(value) {
        return `${round(clamp(value, 0, 100), 1)}%`;
    }


    /* =========================================================
       STATUS / PIPELINE
    ========================================================= */

    const pipelineStages = [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"
    ];

    function resetPipeline() {
        pipelineStages.forEach(id => {
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

    function stageActive(id) {
        const el = $(id);
        if (!el) return;

        el.classList.remove("complete", "done");
        el.classList.add("active", "processing");
    }

    function stageDone(id) {
        const el = $(id);
        if (!el) return;

        el.classList.remove("active", "processing");
        el.classList.add("complete", "done");
    }

    function updateStatus(message) {
        setText("status", message);
    }


    /* =========================================================
       RESULT RESET
    ========================================================= */

    function resetResults() {

        const ids = [
            "score",
            "features",
            "confidence",
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

        setText(
            "interpretation",
            "Upload two lunar images and run the correspondence engine to generate a detailed assessment."
        );

        setText(
            "visualNote",
            "● Verified feature correspondences will appear here after analysis."
        );

        setDisabled("downloadReportBtn", true);

        state.lastAnalysis = null;
    }


    /* =========================================================
       FILE VALIDATION
    ========================================================= */

    function validateImageFile(file) {

        if (!file) {
            throw new Error("No image selected.");
        }

        const allowed = [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
            "image/bmp"
        ];

        const name = file.name.toLowerCase();

        const extensionAllowed =
            /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i.test(name);

        if (!allowed.includes(file.type) && !extensionAllowed) {
            throw new Error(
                "Please select a JPG, JPEG, PNG, WEBP, BMP or supported image file."
            );
        }

        if (file.size > 50 * 1024 * 1024) {
            throw new Error("Image is too large. Please use an image below 50 MB.");
        }
    }


    /* =========================================================
       IMAGE PREVIEW
    ========================================================= */

    function updatePreview(file, previewId) {

        const preview = $(previewId);

        if (!preview || !file) return;

        if (preview.dataset.objectUrl) {
            try {
                URL.revokeObjectURL(preview.dataset.objectUrl);
            } catch (_) {}
        }

        const url = URL.createObjectURL(file);

        preview.dataset.objectUrl = url;
        preview.src = url;

        preview.style.display = "block";

        const dropZone = preview.closest(".drop-new");

        if (dropZone) {
            dropZone.classList.add("has-image");
        }
    }


    /* =========================================================
       IMAGE LOADING
    ========================================================= */

    function loadImage(file) {

        return new Promise((resolve, reject) => {

            const img = new Image();

            img.onload = () => {
                resolve(img);
            };

            img.onerror = () => {
                reject(
                    new Error(
                        "The selected image could not be decoded by this browser."
                    )
                );
            };

            img.src = URL.createObjectURL(file);
        });
    }


    /* =========================================================
       FILE INPUT HANDLING
    ========================================================= */

    function handleImage(file, side, input) {

        try {

            validateImageFile(file);

            if (side === "A") {

                state.imageAFile = file;

                updatePreview(
                    file,
                    "previewA"
                );

            } else {

                state.imageBFile = file;

                updatePreview(
                    file,
                    "previewB"
                );
            }

            state.lastAnalysis = null;

            setDisabled(
                "downloadReportBtn",
                true
            );

            updateStatus(
                `${side === "A" ? "IMAGE A" : "IMAGE B"} READY — ${file.name}`
            );

            resetPipeline();

            if (input) {
                input.value = "";
            }

        } catch (error) {

            console.error("LUNARMATCH upload error:", error);

            updateStatus(
                `UPLOAD ERROR — ${error.message}`
            );

            if (input) {
                input.value = "";
            }
        }
    }


    function setupImageInput(inputId, side) {

        const input = $(inputId);

        if (!input) return;

        input.accept =
            "image/jpeg,image/png,image/webp,image/bmp,image/*";

        input.addEventListener("change", () => {

            const file =
                input.files &&
                input.files.length
                    ? input.files[0]
                    : null;

            if (file) {
                handleImage(
                    file,
                    side,
                    input
                );
            }
        });
    }


    /* =========================================================
       DROP ZONE
    ========================================================= */

    function setupDropZone(zone, side) {

        if (!zone) return;

        const input =
            side === "A"
                ? $("fileA")
                : $("fileB");

        if (!input) return;

        /* Prevent label/input weirdness and manually open picker */

        zone.addEventListener("click", event => {

            event.preventDefault();
            event.stopPropagation();

            input.click();
        });


        ["dragenter", "dragover"].forEach(eventName => {

            zone.addEventListener(eventName, event => {

                event.preventDefault();
                event.stopPropagation();

                zone.classList.add("drag-active");
            });
        });


        ["dragleave", "drop"].forEach(eventName => {

            zone.addEventListener(eventName, event => {

                event.preventDefault();
                event.stopPropagation();

                zone.classList.remove("drag-active");
            });
        });


        zone.addEventListener("drop", event => {

            const files =
                event.dataTransfer &&
                event.dataTransfer.files;

            if (!files || !files.length) return;

            handleImage(
                files[0],
                side,
                input
            );
        });
    }


    /* =========================================================
       CANVAS IMAGE DATA
    ========================================================= */

    function canvasFromImage(img) {

        const scale =
            Math.min(
                1,
                CONFIG.maxImageSize /
                Math.max(img.naturalWidth, img.naturalHeight)
            );

        const width =
            Math.max(
                1,
                Math.round(img.naturalWidth * scale)
            );

        const height =
            Math.max(
                1,
                Math.round(img.naturalHeight * scale)
            );

        const canvas =
            document.createElement("canvas");

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d", {
            willReadFrequently: true
        });

        ctx.drawImage(
            img,
            0,
            0,
            width,
            height
        );

        return {
            canvas,
            ctx,
            width,
            height,
            data: ctx.getImageData(
                0,
                0,
                width,
                height
            ).data
        };
    }


    function toGray(imageData) {

        const {
            data,
            width,
            height
        } = imageData;

        const gray =
            new Float32Array(
                width * height
            );

        for (let i = 0, p = 0; i < gray.length; i++, p += 4) {

            gray[i] =
                0.299 * data[p] +
                0.587 * data[p + 1] +
                0.114 * data[p + 2];
        }

        return {
            gray,
            width,
            height
        };
    }


    /* =========================================================
       IMAGE QUALITY
    ========================================================= */

    function calculateQuality(grayObj) {

        const {
            gray,
            width,
            height
        } = grayObj;

        let sum = 0;
        let sumSq = 0;
        let edges = 0;

        const count =
            width * height;

        for (let i = 0; i < count; i++) {

            const v = gray[i];

            sum += v;
            sumSq += v * v;
        }

        const mean =
            sum / count;

        const variance =
            Math.max(
                0,
                sumSq / count -
                mean * mean
            );

        const contrast =
            Math.sqrt(variance);

        for (let y = 1; y < height - 1; y++) {

            for (let x = 1; x < width - 1; x++) {

                const i =
                    y * width + x;

                const gx =
                    gray[i + 1] -
                    gray[i - 1];

                const gy =
                    gray[i + width] -
                    gray[i - width];

                edges +=
                    Math.sqrt(
                        gx * gx +
                        gy * gy
                    );
            }
        }

        const averageEdge =
            edges /
            Math.max(
                1,
                (width - 2) * (height - 2)
            );

        const contrastScore =
            clamp(
                contrast / 64 * 100,
                0,
                100
            );

        const sharpnessScore =
            clamp(
                averageEdge / 35 * 100,
                0,
                100
            );

        const quality =
            0.55 * contrastScore +
            0.45 * sharpnessScore;

        return {
            contrast: contrastScore,
            sharpness: sharpnessScore,
            quality: quality
        };
    }


    /* =========================================================
       FEATURE DETECTION
       Lightweight browser-based corner detector
    ========================================================= */

    function detectFeatures(grayObj) {

        const {
            gray,
            width,
            height
        } = grayObj;

        const candidates = [];

        const border = 8;
        const step =
            Math.max(
                2,
                Math.floor(
                    Math.min(width, height) / 180
                )
            );

        for (
            let y = border;
            y < height - border;
            y += step
        ) {

            for (
                let x = border;
                x < width - border;
                x += step
            ) {

                const i =
                    y * width + x;

                const gx =
                    gray[i + 1] -
                    gray[i - 1];

                const gy =
                    gray[i + width] -
                    gray[i - width];

                const g =
                    Math.sqrt(
                        gx * gx +
                        gy * gy
                    );

                if (g < 12) continue;

                const gxx =
                    gray[i + 1] +
                    gray[i - 1] -
                    2 * gray[i];

                const gyy =
                    gray[i + width] +
                    gray[i - width] -
                    2 * gray[i];

                const corner =
                    Math.abs(gxx * gyy);

                const score =
                    g * 0.7 +
                    corner * 0.3;

                candidates.push({
                    x,
                    y,
                    score
                });
            }
        }

        candidates.sort(
            (a, b) =>
                b.score - a.score
        );

        const selected = [];

        const minDistance =
            Math.max(
                8,
                Math.min(width, height) / 35
            );

        const minDistSq =
            minDistance *
            minDistance;

        for (const point of candidates) {

            let valid = true;

            for (const chosen of selected) {

                const dx =
                    point.x - chosen.x;

                const dy =
                    point.y - chosen.y;

                if (
                    dx * dx +
                    dy * dy <
                    minDistSq
                ) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                selected.push(point);
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

    function describeFeature(
        grayObj,
        point
    ) {

        const {
            gray,
            width,
            height
        } = grayObj;

        const r =
            CONFIG.patchRadius;

        const descriptor = [];

        for (
            let dy = -r;
            dy <= r;
            dy++
        ) {

            for (
                let dx = -r;
                dx <= r;
                dx++
            ) {

                const x =
                    clamp(
                        Math.round(point.x + dx),
                        0,
                        width - 1
                    );

                const y =
                    clamp(
                        Math.round(point.y + dy),
                        0,
                        height - 1
                    );

                descriptor.push(
                    gray[
                        y * width + x
                    ]
                );
            }
        }

        let mean = 0;

        for (const value of descriptor) {
            mean += value;
        }

        mean /=
            descriptor.length;

        let variance = 0;

        for (let i = 0; i < descriptor.length; i++) {

            descriptor[i] -= mean;

            variance +=
                descriptor[i] *
                descriptor[i];
        }

        const std =
            Math.sqrt(
                variance /
                descriptor.length
            ) || 1;

        for (let i = 0; i < descriptor.length; i++) {
            descriptor[i] /= std;
        }

        return descriptor;
    }


    function buildDescriptors(
        grayObj,
        features
    ) {

        return features.map(
            point => ({
                ...point,
                descriptor:
                    describeFeature(
                        grayObj,
                        point
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

        for (let i = 0; i < length; i++) {

            const d =
                a[i] - b[i];

            sum += d * d;
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

            let best = null;
            let second = null;

            for (
                let j = 0;
                j < featuresB.length;
                j++
            ) {

                const distance =
                    descriptorDistance(
                        featuresA[i].descriptor,
                        featuresB[j].descriptor
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
                    distance < second.distance
                ) {

                    second = {
                        a: i,
                        b: j,
                        distance
                    };
                }
            }

            if (
                best &&
                second &&
                best.distance <
                second.distance *
                CONFIG.matchRatio
            ) {

                forward.push(best);
            }
        }

        return forward;
    }


    /* =========================================================
       MUTUAL MATCH FILTER
    ========================================================= */

    function reciprocalMatches(
        featuresA,
        featuresB,
        matches
    ) {

        const reverseBest =
            new Map();

        for (
            let j = 0;
            j < featuresB.length;
            j++
        ) {

            let bestIndex = -1;
            let bestDistance = Infinity;

            for (
                let i = 0;
                i < featuresA.length;
                i++
            ) {

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

            reverseBest.set(
                j,
                bestIndex
            );
        }

        return matches.filter(
            match =>
                reverseBest.get(match.b) ===
                match.a
        );
    }


    /* =========================================================
       AFFINE MODEL
    ========================================================= */

    function solveAffine(
        p1,
        p2,
        p3,
        q1,
        q2,
        q3
    ) {

        const A = [
            [
                p1.x,
                p1.y,
                1
            ],
            [
                p2.x,
                p2.y,
                1
            ],
            [
                p3.x,
                p3.y,
                1
            ]
        ];

        const bx = [
            q1.x,
            q2.x,
            q3.x
        ];

        const by = [
            q1.y,
            q2.y,
            q3.y
        ];

        function solve(M, b) {

            const m =
                M.map(
                    row => row.slice()
                );

            const v =
                b.slice();

            for (let i = 0; i < 3; i++) {

                let pivot = i;

                for (
                    let r = i + 1;
                    r < 3;
                    r++
                ) {

                    if (
                        Math.abs(m[r][i]) >
                        Math.abs(m[pivot][i])
                    ) {
                        pivot = r;
                    }
                }

                if (
                    Math.abs(m[pivot][i]) <
                    1e-8
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
                    v[i],
                    v[pivot]
                ] = [
                    v[pivot],
                    v[i]
                ];

                const divisor =
                    m[i][i];

                for (
                    let c = i;
                    c < 3;
                    c++
                ) {
                    m[i][c] /=
                        divisor;
                }

                v[i] /=
                    divisor;

                for (
                    let r = 0;
                    r < 3;
                    r++
                ) {

                    if (r === i) continue;

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

                    v[r] -=
                        factor *
                        v[i];
                }
            }

            return v;
        }

        const sx =
            solve(A, bx);

        const sy =
            solve(A, by);

        if (!sx || !sy) {
            return null;
        }

        return {
            a: sx[0],
            b: sx[1],
            tx: sx[2],
            c: sy[0],
            d: sy[1],
            ty: sy[2]
        };
    }


    function transformPoint(
        model,
        p
    ) {

        return {
            x:
                model.a * p.x +
                model.b * p.y +
                model.tx,

            y:
                model.c * p.x +
                model.d * p.y +
                model.ty
        };
    }


    /* =========================================================
       RANSAC GEOMETRIC VERIFICATION
    ========================================================= */

    function verifyGeometry(
        featuresA,
        featuresB,
        matches
    ) {

        if (matches.length < 3) {

            return {
                model: null,
                inliers: [],
                ratio: 0,
                consistency: 0
            };
        }

        let bestModel = null;
        let bestInliers = [];

        for (
            let iteration = 0;
            iteration <
            CONFIG.ransacIterations;
            iteration++
        ) {

            const indices = [];

            while (
                indices.length < 3
            ) {

                const index =
                    Math.floor(
                        Math.random() *
                        matches.length
                    );

                if (
                    !indices.includes(index)
                ) {
                    indices.push(index);
                }
            }

            const m1 =
                matches[indices[0]];

            const m2 =
                matches[indices[1]];

            const m3 =
                matches[indices[2]];

            const model =
                solveAffine(
                    featuresA[m1.a],
                    featuresA[m2.a],
                    featuresA[m3.a],
                    featuresB[m1.b],
                    featuresB[m2.b],
                    featuresB[m3.b]
                );

            if (!model) continue;

            const inliers = [];

            for (
                let i = 0;
                i < matches.length;
                i++
            ) {

                const match =
                    matches[i];

                const p =
                    featuresA[match.a];

                const q =
                    featuresB[match.b];

                const projected =
                    transformPoint(
                        model,
                        p
                    );

                const dx =
                    projected.x - q.x;

                const dy =
                    projected.y - q.y;

                const error =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );

                if (
                    error <=
                    CONFIG.ransacThreshold
                ) {
                    inliers.push(i);
                }
            }

            if (
                inliers.length >
                bestInliers.length
            ) {

                bestModel = model;
                bestInliers = inliers;
            }
        }

        const ratio =
            bestInliers.length /
            Math.max(1, matches.length);

        const consistency =
            clamp(
                ratio * 100,
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
       COVERAGE
    ========================================================= */

    function calculateCoverage(
        featuresA,
        inliers
    ) {

        if (!featuresA.length) {
            return 0;
        }

        const cells = new Set();

        const minX =
            Math.min(
                ...featuresA.map(p => p.x)
            );

        const maxX =
            Math.max(
                ...featuresA.map(p => p.x)
            );

        const minY =
            Math.min(
                ...featuresA.map(p => p.y)
            );

        const maxY =
            Math.max(
                ...featuresA.map(p => p.y)
            );

        const rangeX =
            Math.max(1, maxX - minX);

        const rangeY =
            Math.max(1, maxY - minY);

        for (const index of inliers) {

            const p =
                featuresA[index.a];

            const gx =
                Math.floor(
                    ((p.x - minX) /
                    rangeX) * 5
                );

            const gy =
                Math.floor(
                    ((p.y - minY) /
                    rangeY) * 5
                );

            cells.add(
                `${clamp(gx, 0, 4)}:${clamp(gy, 0, 4)}`
            );
        }

        return (
            cells.size /
            25
        ) * 100;
    }


    /* =========================================================
       MATCH SCORE
    ========================================================= */

    function calculateScore(
        verified,
        candidate,
        coverage,
        geometry,
        quality
    ) {

        const verificationScore =
            clamp(
                verified /
                Math.max(1, candidate) *
                100,
                0,
                100
            );

        const featureScore =
            clamp(
                verified / 30 * 100,
                0,
                100
            );

        return clamp(
            verificationScore * 0.30 +
            featureScore * 0.20 +
            coverage * 0.15 +
            geometry * 0.25 +
            quality * 0.10,
            0,
            100
        );
    }


    function confidenceLabel(score) {

        if (score >= 80) {
            return "HIGH";
        }

        if (score >= 60) {
            return "MODERATE";
        }

        if (score >= 40) {
            return "LOW";
        }

        return "VERY LOW";
    }


    /* =========================================================
       CORRESPONDENCE MAP
    ========================================================= */

    function createCorrespondenceMap(
        imageA,
        imageB,
        featuresA,
        featuresB,
        matches,
        inlierIndexes
    ) {

        const map =
            $("correspondenceMap");

        const placeholder =
            $("visualPlaceholder");

        if (!map) return;

        const canvas =
            document.createElement("canvas");

        const maxWidth = 1400;
        const gap = 20;

        const scale =
            Math.min(
                1,
                maxWidth /
                Math.max(
                    imageA.naturalWidth,
                    imageB.naturalWidth
                )
            );

        const widthA =
            Math.max(
                1,
                Math.round(
                    imageA.naturalWidth *
                    scale
                )
            );

        const widthB =
            Math.max(
                1,
                Math.round(
                    imageB.naturalWidth *
                    scale
                )
            );

        const heightA =
            Math.max(
                1,
                Math.round(
                    imageA.naturalHeight *
                    scale
                )
            );

        const heightB =
            Math.max(
                1,
                Math.round(
                    imageB.naturalHeight *
                    scale
                )
            );

        const height =
            Math.max(
                heightA,
                heightB
            );

        canvas.width =
            widthA +
            widthB +
            gap;

        canvas.height =
            height;

        const ctx =
            canvas.getContext("2d");

        ctx.fillStyle =
            "#05070a";

        ctx.fillRect(
            0,
            0,
            canvas.width,
            canvas.height
        );

        ctx.drawImage(
            imageA,
            0,
            0,
            widthA,
            heightA
        );

        ctx.drawImage(
            imageB,
            widthA + gap,
            0,
            widthB,
            heightB
        );

        const inlierSet =
            new Set(
                inlierIndexes
            );

        for (
            let i = 0;
            i < matches.length;
            i++
        ) {

            const match =
                matches[i];

            const p =
                featuresA[match.a];

            const q =
                featuresB[match.b];

            const x1 =
                p.x *
                scale;

            const y1 =
                p.y *
                scale;

            const x2 =
                q.x *
                scale +
                widthA +
                gap;

            const y2 =
                q.y *
                scale;

            const verified =
                inlierSet.has(i);

            ctx.beginPath();

            ctx.moveTo(
                x1,
                y1
            );

            ctx.lineTo(
                x2,
                y2
            );

            ctx.globalAlpha =
                verified
                    ? 0.75
                    : 0.12;

            ctx.lineWidth =
                verified
                    ? 1.5
                    : 0.6;

            ctx.strokeStyle =
                verified
                    ? "#d8f6ff"
                    : "#78828c";

            ctx.stroke();

            ctx.globalAlpha = 1;

            if (verified) {

                ctx.beginPath();

                ctx.arc(
                    x1,
                    y1,
                    3,
                    0,
                    Math.PI * 2
                );

                ctx.fillStyle =
                    "#ffffff";

                ctx.fill();

                ctx.beginPath();

                ctx.arc(
                    x2,
                    y2,
                    3,
                    0,
                    Math.PI * 2
                );

                ctx.fill();
            }
        }

        map.src =
            canvas.toDataURL(
                "image/png"
            );

        map.style.display =
            "block";

        if (placeholder) {
            placeholder.style.display =
                "none";
        }
    }


    /* =========================================================
       DISPLAY RESULTS
    ========================================================= */

    function displayResults(result) {

        setText(
            "score",
            `${round(result.score)}%`
        );

        setText(
            "features",
            result.verifiedMatches
        );

        setText(
            "confidence",
            `${result.confidence.label} (${round(result.confidence.value)}%)`
        );

        setText(
            "quality",
            percent(result.averageQuality)
        );

        setText(
            "time",
            `${round(result.processTime)} ms`
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
            percent(result.imageA.contrast)
        );

        setText(
            "sharpnessA",
            percent(result.imageA.sharpness)
        );

        setText(
            "qualityScoreA",
            percent(result.imageA.quality)
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
            percent(result.imageB.contrast)
        );

        setText(
            "sharpnessB",
            percent(result.imageB.sharpness)
        );

        setText(
            "qualityScoreB",
            percent(result.imageB.quality)
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
            percent(result.featureCoverage)
        );

        setText(
            "correspondenceStrength",
            percent(result.correspondenceStrength)
        );


        /* GEOMETRY */

        setText(
            "inlierRatio",
            percent(result.inlierRatio)
        );

        setText(
            "geometricConsistency",
            percent(result.geometricConsistency)
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

        setText(
            "visualNote",
            `● ${result.verifiedMatches} verified feature correspondences visualized by the correspondence engine.`
        );

        setDisabled(
            "downloadReportBtn",
            false
        );
    }


    /* =========================================================
       INTERPRETATION
    ========================================================= */

    function generateInterpretation(result) {

        const score =
            result.score;

        if (score >= 80) {

            return (
                "The analysis indicates a strong visual correspondence between the supplied lunar observations. " +
                "A substantial number of candidate features were geometrically consistent after robust verification. " +
                "The result should be treated as a computational correspondence assessment rather than a scientific identification."
            );
        }

        if (score >= 60) {

            return (
                "The analysis indicates a moderate correspondence between the supplied lunar observations. " +
                "Several visual features demonstrate consistent spatial relationships, although the evidence is not strong enough to establish a definitive correspondence."
            );
        }

        if (score >= 40) {

            return (
                "The analysis identified limited correspondence between the supplied lunar observations. " +
                "Some candidate features were matched, but geometric consistency remains relatively weak."
            );
        }

        return (
            "The analysis found insufficient verified correspondence between the supplied lunar observations. " +
            "Differences in illumination, scale, viewing geometry, image quality or surface appearance may contribute to the low correspondence score."
        );
    }


    /* =========================================================
       MAIN ANALYSIS
    ========================================================= */

    async function analyzeImages() {

        const analyzeButton =
            $("compareBtn");

        if (
            !state.imageAFile ||
            !state.imageBFile
        ) {

            updateStatus(
                "SYSTEM WAITING — SELECT BOTH IMAGES"
            );

            return;
        }

        if (analyzeButton) {
            analyzeButton.disabled = true;
        }

        const startTime =
            performance.now();

        try {

            resetPipeline();

            /* ---------------------------------------------
               01 ACQUIRE
            --------------------------------------------- */

            stageActive("stageAcquire");

            updateStatus(
                "ACQUIRING LUNAR IMAGERY..."
            );

            state.imageA =
                await loadImage(
                    state.imageAFile
                );

            state.imageB =
                await loadImage(
                    state.imageBFile
                );

            stageDone("stageAcquire");

            await sleep(100);


            /* ---------------------------------------------
               02 PREPROCESS
            --------------------------------------------- */

            stageActive(
                "stagePreprocess"
            );

            updateStatus(
                "PREPROCESSING IMAGE DATA..."
            );

            const dataA =
                canvasFromImage(
                    state.imageA
                );

            const dataB =
                canvasFromImage(
                    state.imageB
                );

            const grayA =
                toGray(dataA);

            const grayB =
                toGray(dataB);

            const qualityA =
                calculateQuality(
                    grayA
                );

            const qualityB =
                calculateQuality(
                    grayB
                );

            stageDone(
                "stagePreprocess"
            );

            await sleep(100);


            /* ---------------------------------------------
               03 EXTRACT
            --------------------------------------------- */

            stageActive(
                "stageExtract"
            );

            updateStatus(
                "EXTRACTING VISUAL FEATURES..."
            );

            const pointsA =
                detectFeatures(
                    grayA
                );

            const pointsB =
                detectFeatures(
                    grayB
                );

            const featuresA =
                buildDescriptors(
                    grayA,
                    pointsA
                );

            const featuresB =
                buildDescriptors(
                    grayB,
                    pointsB
                );

            stageDone(
                "stageExtract"
            );

            await sleep(100);


            /* ---------------------------------------------
               04 MATCH
            --------------------------------------------- */

            stageActive(
                "stageMatch"
            );

            updateStatus(
                "MATCHING FEATURE DESCRIPTORS..."
            );

            const raw =
                matchFeatures(
                    featuresA,
                    featuresB
                );

            const reciprocal =
                reciprocalMatches(
                    featuresA,
                    featuresB,
                    raw
                );

            stageDone(
                "stageMatch"
            );

            await sleep(100);


            /* ---------------------------------------------
               05 VERIFY
            --------------------------------------------- */

            stageActive(
                "stageVerify"
            );

            updateStatus(
                "RUNNING GEOMETRIC VERIFICATION..."
            );

            const geometry =
                verifyGeometry(
                    featuresA,
                    featuresB,
                    reciprocal
                );

            const inlierMatches =
                geometry.inliers.map(
                    index =>
                        reciprocal[index]
                );

            const coverage =
                calculateCoverage(
                    featuresA,
                    inlierMatches
                );

            stageDone(
                "stageVerify"
            );

            await sleep(100);


            /* ---------------------------------------------
               06 SCORE
            --------------------------------------------- */

            stageActive(
                "stageScore"
            );

            updateStatus(
                "CALCULATING CORRESPONDENCE SCORE..."
            );

            const averageQuality =
                (
                    qualityA.quality +
                    qualityB.quality
                ) / 2;

            const score =
                calculateScore(
                    inlierMatches.length,
                    reciprocal.length,
                    coverage,
                    geometry.consistency,
                    averageQuality
                );

            const confidenceValue =
                clamp(
                    score * 0.92 +
                    geometry.consistency * 0.08,
                    0,
                    100
                );

            const result = {

                score,

                confidence: {
                    value: confidenceValue,
                    label:
                        confidenceLabel(
                            confidenceValue
                        )
                },

                averageQuality,

                processTime: 0,

                imageA: {
                    width:
                        state.imageA.naturalWidth,

                    height:
                        state.imageA.naturalHeight,

                    keypoints:
                        featuresA.length,

                    contrast:
                        qualityA.contrast,

                    sharpness:
                        qualityA.sharpness,

                    quality:
                        qualityA.quality
                },

                imageB: {
                    width:
                        state.imageB.naturalWidth,

                    height:
                        state.imageB.naturalHeight,

                    keypoints:
                        featuresB.length,

                    contrast:
                        qualityB.contrast,

                    sharpness:
                        qualityB.sharpness,

                    quality:
                        qualityB.quality
                },

                rawMatches:
                    raw.length,

                candidateMatches:
                    reciprocal.length,

                verifiedMatches:
                    inlierMatches.length,

                featureCoverage:
                    coverage,

                correspondenceStrength:
                    geometry.consistency,

                inlierRatio:
                    geometry.ratio * 100,

                geometricConsistency:
                    geometry.consistency,

                homographyStatus:
                    geometry.model
                        ? "ESTABLISHED"
                        : "NOT ESTABLISHED",

                verificationStatus:
                    geometry.model &&
                    inlierMatches.length >= 5
                        ? "VERIFIED"
                        : "LIMITED",

                interpretation: ""
            };

            result.processTime =
                performance.now() -
                startTime;

            result.interpretation =
                generateInterpretation(
                    result
                );

            stageDone(
                "stageScore"
            );

            await sleep(100);


            /* ---------------------------------------------
               MAP
            --------------------------------------------- */

            createCorrespondenceMap(
                state.imageA,
                state.imageB,
                featuresA,
                featuresB,
                reciprocal,
                geometry.inliers
            );


            /* ---------------------------------------------
               07 REPORT
            --------------------------------------------- */

            stageActive(
                "stageReport"
            );

            updateStatus(
                "GENERATING ANALYSIS OUTPUT..."
            );

            displayResults(
                result
            );

            state.lastAnalysis =
                result;

            stageDone(
                "stageReport"
            );

            updateStatus(
                "ANALYSIS COMPLETE — CORRESPONDENCE RESULT READY"
            );

        } catch (error) {

            console.error(
                "LUNARMATCH analysis error:",
                error
            );

            updateStatus(
                `ANALYSIS ERROR — ${error.message || "Unable to complete analysis."}`
            );

            resetPipeline();

        } finally {

            if (analyzeButton) {
                analyzeButton.disabled = false;
            }
        }
    }


    /* =========================================================
       PDF / REPORT
    ========================================================= */

    function downloadReport() {

       async function downloadReport() {
    if (!state.lastAnalysis) {
        alert("Please run the correspondence analysis first.");
        return;
    }

    const result = state.lastAnalysis;

    try {
        // Load jsPDF dynamically without changing the rest of the website
        if (!window.jspdf) {
            await new Promise((resolve, reject) => {
                const script = document.createElement("script");
                script.src =
                    "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";

                script.onload = resolve;
                script.onerror = () =>
                    reject(new Error("Unable to load PDF generator."));

                document.head.appendChild(script);
            });
        }

        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF();

        const margin = 18;
        let y = 20;

        function line(text, size = 10, bold = false) {
            pdf.setFontSize(size);
            pdf.setFont("helvetica", bold ? "bold" : "normal");

            const lines = pdf.splitTextToSize(
                String(text ?? "—"),
                174
            );

            pdf.text(lines, margin, y);
            y += lines.length * 6 + 2;

            if (y > 275) {
                pdf.addPage();
                y = 20;
            }
        }

        // Header
        line("LUNARMATCH", 22, true);
        line("Lunar Intelligence Platform", 11, false);
        line("Lunar Image Correspondence Analysis Report", 12, true);

        y += 4;

        // Summary
        line("ANALYSIS SUMMARY", 14, true);

        line(`Correspondence Score: ${result.score ?? "—"}%`);
        line(
            `Confidence: ${
                result.confidence?.label ?? "—"
            } (${result.confidence?.value ?? "—"}%)`
        );
        line(
            `Verified Matches: ${
                result.verifiedMatches ?? "—"
            }`
        );
        line(
            `Feature Coverage: ${
                result.featureCoverage ?? "—"
            }%`
        );
        line(
            `Correspondence Strength: ${
                result.correspondenceStrength ?? "—"
            }%`
        );
        line(
            `Inlier Ratio: ${
                result.inlierRatio ?? "—"
            }%`
        );
        line(
            `Geometric Consistency: ${
                result.geometricConsistency ?? "—"
            }%`
        );
        line(
            `Homography Status: ${
                result.homographyStatus ?? "—"
            }`
        );
        line(
            `Verification Status: ${
                result.verificationStatus ?? "—"
            }`
        );

        y += 4;

        // Image A
        line("SOURCE IMAGE A", 14, true);

        line(
            `Resolution: ${
                result.resolutionA ?? "—"
            }`
        );
        line(
            `Keypoints: ${
                result.keypointsA ?? "—"
            }`
        );
        line(
            `Contrast: ${
                result.contrastA ?? "—"
            }`
        );
        line(
            `Sharpness: ${
                result.sharpnessA ?? "—"
            }`
        );
        line(
            `Quality Score: ${
                result.qualityScoreA ?? "—"
            }`
        );

        y += 4;

        // Image B
        line("REFERENCE IMAGE B", 14, true);

        line(
            `Resolution: ${
                result.resolutionB ?? "—"
            }`
        );
        line(
            `Keypoints: ${
                result.keypointsB ?? "—"
            }`
        );
        line(
            `Contrast: ${
                result.contrastB ?? "—"
            }`
        );
        line(
            `Sharpness: ${
                result.sharpnessB ?? "—"
            }`
        );
        line(
            `Quality Score: ${
                result.qualityScoreB ?? "—"
            }`
        );

        y += 4;

        // Matching statistics
        line("MATCHING STATISTICS", 14, true);

        line(
            `Raw Matches: ${
                result.rawMatches ?? "—"
            }`
        );
        line(
            `Candidate Matches: ${
                result.candidateMatches ?? "—"
            }`
        );
        line(
            `Verified Matches: ${
                result.verifiedMatches ?? "—"
            }`
        );

        y += 4;

        // Interpretation
        line("SCIENTIFIC INTERPRETATION", 14, true);
        line(result.interpretation ?? "No interpretation available.");

        y += 4;

        line("LUNARMATCH — Lunar Image Correspondence System", 9, false);
        line(
            `Generated: ${new Date().toLocaleString()}`,
            9,
            false
        );

        // Download directly as a PDF
        pdf.save("LUNARMATCH_Correspondence_Report.pdf");

        setText(
            "status",
            "REPORT DOWNLOADED — PDF READY"
        );

    } catch (error) {
        console.error("PDF generation error:", error);

        setText(
            "status",
            "REPORT ERROR — PDF COULD NOT BE GENERATED"
        );

        alert(
            "The PDF could not be generated. Please check your internet connection and try again."
        );
    }
}
    }


    /* =========================================================
       NAVIGATION
    ========================================================= */

    function setupNavigation() {

        const links =
            document.querySelectorAll(
                'nav a[href^="#"]'
            );

        links.forEach(link => {

            link.addEventListener(
                "click",
                () => {

                    links.forEach(
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
       EFFECTS
    ========================================================= */

    function setupEffects() {

        const zones =
            document.querySelectorAll(
                ".drop-new"
            );

        zones.forEach(zone => {

            zone.addEventListener(
                "dragenter",
                () => {
                    zone.classList.add(
                        "drag-active"
                    );
                }
            );

            zone.addEventListener(
                "dragleave",
                () => {
                    zone.classList.remove(
                        "drag-active"
                    );
                }
            );
        });
    }


    /* =========================================================
       BUTTONS
    ========================================================= */

    function setupButtons() {

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

                    updateStatus(
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

                    const section =
                        $("contact");

                    if (section) {
                        section.scrollIntoView({
                            behavior: "smooth"
                        });
                    }
                }
            );
        }
    }


    /* =========================================================
       GLOBAL ERROR HANDLING
    ========================================================= */

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

        state.initialized = true;


        /* Image inputs */

        setupImageInput(
            "fileA",
            "A"
        );

        setupImageInput(
            "fileB",
            "B"
        );


        /* Drop zones */

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


        /* Buttons */

        setupButtons();


        /* Navigation */

        setupNavigation();


        /* Visual effects */

        setupEffects();


        /* Initial UI */

        resetPipeline();

        resetResults();

        updateStatus(
            "SYSTEM READY — AWAITING LUNAR IMAGERY"
        );


        console.log(
            "LUNARMATCH — Lunar Correspondence Engine initialized successfully."
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
