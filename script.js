// =========================================================
// LUNARMATCH — FREE BROWSER ANALYSIS ENGINE
// Browser-side ORB + Robust Geometric Verification
//
// Captain's V5.3 server.py remains preserved separately.
// This frontend engine avoids Render/PyTorch RAM limits.
// =========================================================

const fileA = document.getElementById("fileA");
const fileB = document.getElementById("fileB");

const previewA = document.getElementById("previewA");
const previewB = document.getElementById("previewB");

const dropA = document.querySelectorAll(".drop-new")[0];
const dropB = document.querySelectorAll(".drop-new")[1];

const compareBtn = document.getElementById("compareBtn");
const downloadReportBtn =
    document.getElementById("downloadReportBtn");

const correspondenceMap =
    document.getElementById("correspondenceMap");

const visualPlaceholder =
    document.getElementById("visualPlaceholder");

const visualNote =
    document.getElementById("visualNote");

let previewURL_A = null;
let previewURL_B = null;
let lastAnalysis = null;

let cvReady = false;


// =========================================================
// HELPERS
// =========================================================

function setText(id, value) {

    const el = document.getElementById(id);

    if (!el) return;

    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        el.textContent = "—";
        return;
    }

    el.textContent = value;
}


function numberValue(...values) {

    for (const value of values) {

        if (
            value === undefined ||
            value === null ||
            value === ""
        ) {
            continue;
        }

        const n = Number(value);

        if (Number.isFinite(n)) {
            return n;
        }
    }

    return NaN;
}


function numberFormat(value) {

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return "—";
    }

    return n.toLocaleString();
}


function percent(value, decimals = 1) {

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return "—";
    }

    return `${n.toFixed(decimals)}%`;
}


function seconds(value) {

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return "—";
    }

    return `${n.toFixed(2)} sec`;
}


function clamp(value, min, max) {

    return Math.max(
        min,
        Math.min(max, value)
    );
}


// =========================================================
// LOAD OPENCV.JS
// =========================================================

function loadOpenCV() {

    return new Promise((resolve, reject) => {

        if (
            window.cv &&
            typeof window.cv.Mat === "function"
        ) {

            cvReady = true;

            resolve();

            return;
        }


        const existing =
            document.querySelector(
                'script[data-lunarmatch-opencv="true"]'
            );


        if (existing) {

            existing.addEventListener(
                "load",
                () => {

                    waitForOpenCV(
                        resolve,
                        reject
                    );
                }
            );

            existing.addEventListener(
                "error",
                reject
            );

            return;
        }


        const script =
            document.createElement("script");


        script.src =
            "https://docs.opencv.org/4.x/opencv.js";


        script.async = true;

        script.dataset.lunarmatchOpencv =
            "true";


        script.onload = () => {

            waitForOpenCV(
                resolve,
                reject
            );
        };


        script.onerror = () => {

            reject(
                new Error(
                    "OpenCV.js could not be loaded. Please check your internet connection and try again."
                )
            );
        };


        document.head.appendChild(
            script
        );
    });
}


function waitForOpenCV(
    resolve,
    reject
) {

    const start =
        Date.now();


    const timer =
        setInterval(() => {

            if (
                window.cv &&
                typeof window.cv.Mat === "function"
            ) {

                clearInterval(timer);

                cvReady = true;

                resolve();

                return;
            }


            if (
                Date.now() - start >
                15000
            ) {

                clearInterval(timer);

                reject(
                    new Error(
                        "OpenCV.js initialization timed out."
                    )
                );
            }

        }, 100);
}


// =========================================================
// IMAGE INPUT
// =========================================================

function setupImageInput(
    input,
    preview,
    drop,
    name
) {

    if (!input || !preview || !drop) {
        return;
    }


    input.addEventListener(
        "change",
        () => {

            const file =
                input.files[0];

            if (!file) return;


            if (
                !file.type.startsWith("image/")
            ) {

                alert(
                    `${name}: Please select a valid image file.`
                );

                input.value = "";

                return;
            }


            showPreview(
                file,
                preview,
                drop,
                name
            );
        }
    );


    drop.addEventListener(
        "dragover",
        event => {

            event.preventDefault();

            drop.classList.add(
                "dragging"
            );
        }
    );


    drop.addEventListener(
        "dragleave",
        () => {

            drop.classList.remove(
                "dragging"
            );
        }
    );


    drop.addEventListener(
        "drop",
        event => {

            event.preventDefault();

            drop.classList.remove(
                "dragging"
            );


            const file =
                event.dataTransfer.files[0];


            if (
                !file ||
                !file.type.startsWith("image/")
            ) {

                alert(
                    `${name}: Please drop a valid image file.`
                );

                return;
            }


            try {

                const transfer =
                    new DataTransfer();

                transfer.items.add(file);

                input.files =
                    transfer.files;

            } catch (error) {

                console.warn(
                    "File assignment warning:",
                    error
                );
            }


            showPreview(
                file,
                preview,
                drop,
                name
            );
        }
    );
}


function showPreview(
    file,
    preview,
    drop,
    name
) {

    if (
        name === "IMAGE A" &&
        previewURL_A
    ) {

        URL.revokeObjectURL(
            previewURL_A
        );
    }


    if (
        name === "IMAGE B" &&
        previewURL_B
    ) {

        URL.revokeObjectURL(
            previewURL_B
        );
    }


    const url =
        URL.createObjectURL(file);


    if (name === "IMAGE A") {

        previewURL_A = url;

    } else {

        previewURL_B = url;
    }


    preview.src = url;

    preview.style.display =
        "block";

    preview.style.visibility =
        "visible";

    preview.style.opacity =
        "1";


    const content =
        drop.querySelector(
            ".drop-content"
        );


    if (content) {

        content.style.opacity =
            "0";
    }


    drop.classList.add(
        "has-image"
    );


    drop.dataset.filename =
        file.name;


    const strong =
        drop.querySelector(
            ".drop-content strong"
        );


    if (strong) {

        strong.textContent =
            file.name;
    }


    lastAnalysis = null;


    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            true;
    }
}


if (
    fileA &&
    previewA &&
    dropA
) {

    setupImageInput(
        fileA,
        previewA,
        dropA,
        "IMAGE A"
    );
}


if (
    fileB &&
    previewB &&
    dropB
) {

    setupImageInput(
        fileB,
        previewB,
        dropB,
        "IMAGE B"
    );
}


// =========================================================
// RESET
// =========================================================

function resetResults() {

    setText(
        "status",
        "READY FOR ANALYSIS"
    );

    setText(
        "score",
        "—"
    );

    setText(
        "features",
        "—"
    );

    setText(
        "confidence",
        "—"
    );

    setText(
        "quality",
        "—"
    );

    setText(
        "time",
        "—"
    );


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


    ids.forEach(
        id => setText(id, "—")
    );


    [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"

    ].forEach(id => {

        const el =
            document.getElementById(id);

        if (!el) return;

        el.classList.remove(
            "complete",
            "active",
            "failed"
        );
    });


    if (correspondenceMap) {

        correspondenceMap.src = "";

        correspondenceMap.style.display =
            "none";
    }


    if (visualPlaceholder) {

        visualPlaceholder.style.display =
            "block";
    }


    if (visualNote) {

        visualNote.textContent =
            "● Verified feature correspondences will appear here after analysis.";
    }


    setText(
        "interpretation",
        "Upload two lunar images and run the correspondence engine to generate a detailed assessment."
    );


    lastAnalysis = null;
}


resetResults();


// =========================================================
// PIPELINE
// =========================================================

function pipeline(id, state) {

    const el =
        document.getElementById(id);

    if (!el) return;


    el.classList.remove(
        "complete",
        "active",
        "failed"
    );


    if (state) {

        el.classList.add(
            state
        );
    }
}


function pipelineRunning() {

    pipeline(
        "stageAcquire",
        "complete"
    );

    pipeline(
        "stagePreprocess",
        "active"
    );

    pipeline(
        "stageExtract",
        ""
    );

    pipeline(
        "stageMatch",
        ""
    );

    pipeline(
        "stageVerify",
        ""
    );

    pipeline(
        "stageScore",
        ""
    );

    pipeline(
        "stageReport",
        ""
    );
}


function pipelineComplete() {

    [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"

    ].forEach(
        id =>
            pipeline(
                id,
                "complete"
            )
    );
}


function pipelineFailed() {

    [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore"

    ].forEach(id => {

        const el =
            document.getElementById(id);

        if (!el) return;


        el.classList.remove(
            "active"
        );


        el.classList.add(
            "failed"
        );
    });
}


// =========================================================
// IMAGE QUALITY
// =========================================================

function calculateImageQuality(
    image
) {

    const width =
        image.cols;

    const height =
        image.rows;


    const gray =
        new cv.Mat();


    cv.cvtColor(
        image,
        gray,
        cv.COLOR_RGBA2GRAY
    );


    const mean =
        new cv.Mat();


    const stddev =
        new cv.Mat();


    cv.meanStdDev(
        gray,
        mean,
        stddev
    );


    const contrast =
        stddev.doubleAt(0, 0);


    const laplacian =
        new cv.Mat();


    cv.Laplacian(
        gray,
        laplacian,
        cv.CV_64F
    );


    const lapMean =
        new cv.Mat();


    const lapStd =
        new cv.Mat();


    cv.meanStdDev(
        laplacian,
        lapMean,
        lapStd
    );


    const sharpness =
        Math.pow(
            lapStd.doubleAt(0, 0),
            2
        );


    const resolutionPixels =
        width * height;


    const resolutionScore =
        clamp(
            (
                Math.sqrt(
                    resolutionPixels
                ) / 1200
            ) * 100,
            0,
            100
        );


    const contrastScore =
        clamp(
            contrast * 2.2,
            0,
            100
        );


    const sharpnessScore =
        clamp(
            Math.log10(
                Math.max(
                    sharpness,
                    1
                )
            ) * 18,
            0,
            100
        );


    const score =
        (
            resolutionScore * 0.25 +
            contrastScore * 0.30 +
            sharpnessScore * 0.45
        );


    let quality =
        "LIMITED";


    if (score >= 75) {

        quality =
            "EXCELLENT";

    } else if (score >= 55) {

        quality =
            "GOOD";

    } else if (score >= 35) {

        quality =
            "FAIR";
    }


    const result = {

        resolution:
            `${width} × ${height}`,

        contrast,

        sharpness,

        score,

        quality
    };


    gray.delete();
    mean.delete();
    stddev.delete();
    laplacian.delete();
    lapMean.delete();
    lapStd.delete();


    return result;
}


// =========================================================
// IMAGE LOADING
// =========================================================

function fileToMat(file) {

    return new Promise(
        (resolve, reject) => {

            const url =
                URL.createObjectURL(
                    file
                );


            const image =
                new Image();


            image.onload = () => {

                try {

                    const canvas =
                        document.createElement(
                            "canvas"
                        );


                    const MAX =
                        1000;


                    let width =
                        image.naturalWidth;


                    let height =
                        image.naturalHeight;


                    const scale =
                        Math.min(
                            1,
                            MAX /
                            Math.max(
                                width,
                                height
                            )
                        );


                    width =
                        Math.max(
                            1,
                            Math.round(
                                width * scale
                            )
                        );


                    height =
                        Math.max(
                            1,
                            Math.round(
                                height * scale
                            )
                        );


                    canvas.width =
                        width;

                    canvas.height =
                        height;


                    const ctx =
                        canvas.getContext(
                            "2d",
                            {
                                willReadFrequently:
                                    true
                            }
                        );


                    ctx.drawImage(
                        image,
                        0,
                        0,
                        width,
                        height
                    );


                    const mat =
                        cv.imread(
                            canvas
                        );


                    URL.revokeObjectURL(
                        url
                    );


                    resolve(mat);

                } catch (error) {

                    URL.revokeObjectURL(
                        url
                    );

                    reject(error);
                }
            };


            image.onerror = () => {

                URL.revokeObjectURL(
                    url
                );

                reject(
                    new Error(
                        "Unable to read one of the uploaded images."
                    )
                );
            };


            image.src =
                url;
        }
    );
}


// =========================================================
// ORB FEATURE EXTRACTION
// =========================================================

function extractFeatures(
    image
) {

    const gray =
        new cv.Mat();


    cv.cvtColor(
        image,
        gray,
        cv.COLOR_RGBA2GRAY
    );


    const clahe =
        new cv.Mat();


    const claheObject =
        new cv.CLAHE(
            2.0,
            new cv.Size(
                8,
                8
            )
        );


    claheObject.apply(
        gray,
        clahe
    );


    const keypoints =
        new cv.KeyPointVector();


    const descriptors =
        new cv.Mat();


    const orb =
        new cv.ORB(
            4000,
            1.2,
            8,
            31,
            0,
            2,
            cv.ORB_HARRIS_SCORE,
            31,
            20
        );


    orb.detectAndCompute(
        clahe,
        new cv.Mat(),
        keypoints,
        descriptors
    );


    const result = {

        gray,
        clahe,
        keypoints,
        descriptors,

        count:
            keypoints.size()
    };


    claheObject.delete();
    orb.delete();


    return result;
}


// =========================================================
// MATCH FEATURES
// =========================================================

function matchFeatures(
    descriptorsA,
    descriptorsB
) {

    if (
        descriptorsA.empty() ||
        descriptorsB.empty()
    ) {

        return [];
    }


    const matcher =
        new cv.BFMatcher(
            cv.NORM_HAMMING,
            false
        );


    const knn =
        new cv.DMatchVectorVector();


    matcher.knnMatch(
        descriptorsA,
        descriptorsB,
        knn,
        2
    );


    const goodMatches = [];


    for (
        let i = 0;
        i < knn.size();
        i++
    ) {

        const pair =
            knn.get(i);


        if (
            pair.size() < 2
        ) {

            continue;
        }


        const first =
            pair.get(0);


        const second =
            pair.get(1);


        // Lowe ratio test
        if (
            first.distance <
            0.78 *
            second.distance
        ) {

            goodMatches.push({
                queryIdx:
                    first.queryIdx,

                trainIdx:
                    first.trainIdx,

                distance:
                    first.distance
            });
        }


        pair.delete();
    }


    knn.delete();
    matcher.delete();


    return goodMatches;
}


// =========================================================
// GEOMETRIC VERIFICATION
// =========================================================

function verifyGeometry(
    featuresA,
    featuresB,
    matches
) {

    if (
        matches.length < 4
    ) {

        return {

            verifiedMatches: 0,

            inlierRatio: 0,

            geometricConsistency: 0,

            coverage: 0,

            model:
                "NOT ESTABLISHED",

            homography:
                null,

            mask:
                null
        };
    }


    const pointsA =
        new cv.Mat();


    const pointsB =
        new cv.Mat();


    const pointArrayA = [];
    const pointArrayB = [];


    for (
        const match of matches
    ) {

        const kpA =
            featuresA.keypoints.get(
                match.queryIdx
            );


        const kpB =
            featuresB.keypoints.get(
                match.trainIdx
            );


        pointArrayA.push(
            kpA.pt.x,
            kpA.pt.y
        );


        pointArrayB.push(
            kpB.pt.x,
            kpB.pt.y
        );
    }


    pointsA.create(
        matches.length,
        1,
        cv.CV_32FC2
    );


    pointsB.create(
        matches.length,
        1,
        cv.CV_32FC2
    );


    for (
        let i = 0;
        i < matches.length;
        i++
    ) {

        pointsA.data32F[
            i * 2
        ] =
            pointArrayA[
                i * 2
            ];

        pointsA.data32F[
            i * 2 + 1
        ] =
            pointArrayA[
                i * 2 + 1
            ];


        pointsB.data32F[
            i * 2
        ] =
            pointArrayB[
                i * 2
            ];

        pointsB.data32F[
            i * 2 + 1
        ] =
            pointArrayB[
                i * 2 + 1
            ];
    }


    const mask =
        new cv.Mat();


    let homography =
        null;


    try {

        homography =
            cv.findHomography(
                pointsA,
                pointsB,
                cv.RANSAC,
                5,
                mask
            );

    } catch (error) {

        console.warn(
            "Homography verification failed:",
            error
        );
    }


    let verified =
        0;


    if (
        !mask.empty()
    ) {

        for (
            let i = 0;
            i < mask.rows;
            i++
        ) {

            if (
                mask.ucharAt(
                    i,
                    0
                ) > 0
            ) {

                verified++;
            }
        }
    }


    const inlierRatio =
        matches.length > 0
            ? (
                verified /
                matches.length
            ) * 100
            : 0;


    // 4 × 4 spatial coverage
    const occupied =
        new Set();


    for (
        let i = 0;
        i < matches.length;
        i++
    ) {

        if (
            mask.empty() ||
            mask.ucharAt(
                i,
                0
            ) === 0
        ) {

            continue;
        }


        const x =
            pointArrayA[
                i * 2
            ];


        const y =
            pointArrayA[
                i * 2 + 1
            ];


        const gridX =
            Math.min(
                3,
                Math.floor(
                    x /
                    featuresA.gray.cols *
                    4
                )
            );


        const gridY =
            Math.min(
                3,
                Math.floor(
                    y /
                    featuresA.gray.rows *
                    4
                )
            );


        occupied.add(
            `${gridX}-${gridY}`
        );
    }


    const coverage =
        occupied.size /
        16 *
        100;


    const geometricConsistency =
        clamp(
            inlierRatio * 0.65 +
            coverage * 0.35,
            0,
            100
        );


    pointsA.delete();
    pointsB.delete();


    return {

        verifiedMatches:
            verified,

        inlierRatio,

        geometricConsistency,

        coverage,

        model:
            verified >= 4
                ? "HOMOGRAPHY / RANSAC"
                : "NOT ESTABLISHED",

        homography,

        mask
    };
}


// =========================================================
// SCORE
// =========================================================

function calculateScore(
    candidateMatches,
    verifiedMatches,
    inlierRatio,
    coverage,
    geometry,
    qualityA,
    qualityB
) {

    const matchStrength =
        clamp(
            (
                verifiedMatches /
                Math.max(
                    candidateMatches,
                    1
                )
            ) * 100,
            0,
            100
        );


    const quantityScore =
        clamp(
            (
                verifiedMatches /
                80
            ) * 100,
            0,
            100
        );


    const qualityScore =
        (
            qualityA.score +
            qualityB.score
        ) / 2;


    const score =
        clamp(

            matchStrength * 0.30 +

            quantityScore * 0.20 +

            inlierRatio * 0.25 +

            coverage * 0.10 +

            geometry * 0.10 +

            qualityScore * 0.05,

            0,
            100
        );


    return score;
}


// =========================================================
// CONFIDENCE
// =========================================================

function classify(
    score,
    verified,
    geometry,
    coverage
) {

    if (
        score >= 65 &&
        verified >= 25 &&
        geometry >= 55 &&
        coverage >= 25
    ) {

        return {
            confidence:
                "HIGH",

            decision:
                "STRONG CORRESPONDENCE",

            matchFound:
                true
        };
    }


    if (
        score >= 45 &&
        verified >= 12 &&
        geometry >= 35 &&
        coverage >= 15
    ) {

        return {
            confidence:
                "MEDIUM",

            decision:
                "MODERATE CORRESPONDENCE",

            matchFound:
                true
        };
    }


    return {

        confidence:
            "LOW",

        decision:
            "LIMITED CORRESPONDENCE",

        matchFound:
            false
    };
}


// =========================================================
// DRAW CORRESPONDENCE MAP
// =========================================================

function createCorrespondenceMap(
    imageA,
    imageB,
    featuresA,
    featuresB,
    matches,
    geometry
) {

    const widthA =
        imageA.cols;


    const heightA =
        imageA.rows;


    const widthB =
        imageB.cols;


    const heightB =
        imageB.rows;


    const gap =
        40;


    const outputWidth =
        widthA +
        widthB +
        gap;


    const outputHeight =
        Math.max(
            heightA,
            heightB
        );


    const output =
        new cv.Mat(
            outputHeight,
            outputWidth,
            cv.CV_8UC4,
            new cv.Scalar(
                12,
                12,
                12,
                255
            )
        );


    const roiA =
        output.roi(
            new cv.Rect(
                0,
                0,
                widthA,
                heightA
            )
        );


    const roiB =
        output.roi(
            new cv.Rect(
                widthA + gap,
                0,
                widthB,
                heightB
            )
        );


    imageA.copyTo(
        roiA
    );


    imageB.copyTo(
        roiB
    );


    roiA.delete();
    roiB.delete();


    const verifiedIndices =
        [];


    if (
        geometry.mask &&
        !geometry.mask.empty()
    ) {

        for (
            let i = 0;
            i < matches.length;
            i++
        ) {

            if (
                geometry.mask.ucharAt(
                    i,
                    0
                ) > 0
            ) {

                verifiedIndices.push(
                    i
                );
            }
        }
    }


    // Limit drawn lines for performance
    const drawLimit =
        Math.min(
            verifiedIndices.length,
            80
        );


    for (
        let j = 0;
        j < drawLimit;
        j++
    ) {

        const index =
            verifiedIndices[j];


        const match =
            matches[index];


        const kpA =
            featuresA.keypoints.get(
                match.queryIdx
            );


        const kpB =
            featuresB.keypoints.get(
                match.trainIdx
            );


        const ptA =
            new cv.Point(
                Math.round(
                    kpA.pt.x
                ),
                Math.round(
                    kpA.pt.y
                )
            );


        const ptB =
            new cv.Point(
                Math.round(
                    kpB.pt.x +
                    widthA +
                    gap
                ),
                Math.round(
                    kpB.pt.y
                )
            );


        cv.line(
            output,
            ptA,
            ptB,
            new cv.Scalar(
                0,
                255,
                120,
                255
            ),
            1
        );


        cv.circle(
            output,
            ptA,
            4,
            new cv.Scalar(
                0,
                255,
                120,
                255
            ),
            1
        );


        cv.circle(
            output,
            ptB,
            4,
            new cv.Scalar(
                0,
                255,
                120,
                255
            ),
            1
        );
    }


    const canvas =
        document.createElement(
            "canvas"
        );


    canvas.width =
        output.cols;

    canvas.height =
        output.rows;


    cv.imshow(
        canvas,
        output
    );


    const dataURL =
        canvas.toDataURL(
            "image/jpeg",
            0.88
        );


    output.delete();


    return dataURL;
}


// =========================================================
// DISPLAY RESULTS
// =========================================================

function displayResults(
    data
) {

    lastAnalysis =
        data;


    setText(
        "status",
        data.decision
    );


    setText(
        "score",
        percent(
            data.match_percentage
        )
    );


    setText(
        "features",
        `${numberFormat(data.verified_matches)} / ${numberFormat(data.candidate_matches)}`
    );


    setText(
        "confidence",
        data.confidence
    );


    setText(
        "quality",
        data.overall_quality
    );


    setText(
        "time",
        seconds(
            data.processing_time
        )
    );


    // IMAGE A
    setText(
        "resolutionA",
        data.image_quality_a.resolution
    );


    setText(
        "keypointsA",
        numberFormat(
            data.feature_count_a
        )
    );


    setText(
        "contrastA",
        data.image_quality_a.contrast.toFixed(1)
    );


    setText(
        "sharpnessA",
        data.image_quality_a.sharpness.toFixed(1)
    );


    setText(
        "qualityScoreA",
        data.image_quality_a.score.toFixed(1)
    );


    // IMAGE B
    setText(
        "resolutionB",
        data.image_quality_b.resolution
    );


    setText(
        "keypointsB",
        numberFormat(
            data.feature_count_b
        )
    );


    setText(
        "contrastB",
        data.image_quality_b.contrast.toFixed(1)
    );


    setText(
        "sharpnessB",
        data.image_quality_b.sharpness.toFixed(1)
    );


    setText(
        "qualityScoreB",
        data.image_quality_b.score.toFixed(1)
    );


    // MATCHES
    setText(
        "rawMatches",
        numberFormat(
            data.raw_matches
        )
    );


    setText(
        "candidateMatches",
        numberFormat(
            data.candidate_matches
        )
    );


    setText(
        "verifiedMatches",
        numberFormat(
            data.verified_matches
        )
    );


    // COVERAGE
    setText(
        "featureCoverage",
        percent(
            data.feature_coverage
        )
    );


    setText(
        "correspondenceStrength",
        percent(
            data.correspondence_strength
        )
    );


    // GEOMETRY
    setText(
        "inlierRatio",
        percent(
            data.inlier_ratio
        )
    );


    setText(
        "geometricConsistency",
        percent(
            data.geometric_consistency
        )
    );


    setText(
        "homographyStatus",
        data.geometry_model
    );


    setText(
        "verificationStatus",
        data.verification_status
    );


    // VISUALIZATION
    if (
        data.visualization &&
        correspondenceMap
    ) {

        correspondenceMap.src =
            data.visualization;


        correspondenceMap.style.display =
            "block";


        if (visualPlaceholder) {

            visualPlaceholder.style.display =
                "none";
        }
    }


    if (visualNote) {

        visualNote.textContent =
            `● ${numberFormat(data.verified_matches)} verified ORB correspondences · ${data.geometry_model}`;
    }


    const visualEngine =
        document.getElementById(
            "visualEngine"
        );


    if (visualEngine) {

        visualEngine.textContent =
            "BROWSER ORB + RANSAC";
    }


    generateInterpretation(
        data
    );


    pipelineComplete();


    // Report button remains available as a data export.
    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            false;
    }


    console.log(
        "LUNARMATCH BROWSER RESULT:",
        data
    );
}


// =========================================================
// INTERPRETATION
// =========================================================

function generateInterpretation(
    data
) {

    let assessment;


    if (
        data.confidence ===
        "HIGH"
    ) {

        assessment =
            "The browser correspondence engine detected strong supporting visual evidence with multiple geometrically consistent relationships.";

    } else if (
        data.confidence ===
        "MEDIUM"
    ) {

        assessment =
            "The browser correspondence engine detected meaningful visual correspondence, but the available geometric evidence does not support a high-confidence classification.";

    } else {

        assessment =
            "The engine detected limited correspondence evidence. The result should be treated cautiously when illumination, scale, viewpoint, terrain appearance, or image quality differ significantly.";
    }


    let scoreText;


    if (
        data.match_percentage >=
        70
    ) {

        scoreText =
            "The overall correspondence evidence is strong.";

    } else if (
        data.match_percentage >=
        40
    ) {

        scoreText =
            "The overall correspondence evidence is moderate.";

    } else if (
        data.match_percentage >=
        20
    ) {

        scoreText =
            "The overall correspondence evidence is limited.";

    } else {

        scoreText =
            "The overall correspondence evidence is low.";
    }


    const text =

        `${assessment} ` +

        `${scoreText} ` +

        `The analysis produced ${numberFormat(data.candidate_matches)} candidate correspondences, of which ${numberFormat(data.verified_matches)} survived robust geometric verification using ${data.geometry_model}. ` +

        `The final correspondence score is ${percent(data.match_percentage)} with ${data.confidence.toLowerCase()} confidence. ` +

        `This score represents measured image-correspondence evidence, not a probability that the two images depict the same geographic location.`;


    setText(
        "interpretation",
        text
    );
}


// =========================================================
// RUN BROWSER ANALYSIS
// =========================================================

async function compareImages() {

    if (
        !fileA ||
        !fileB
    ) {

        console.error(
            "LUNARMATCH: input elements not found."
        );

        return;
    }


    const imageFileA =
        fileA.files[0];


    const imageFileB =
        fileB.files[0];


    if (
        !imageFileA ||
        !imageFileB
    ) {

        alert(
            "Please upload both Image A and Image B before starting the analysis."
        );

        return;
    }


    if (
        !imageFileA.type.startsWith("image/") ||
        !imageFileB.type.startsWith("image/")
    ) {

        alert(
            "Both files must be valid image files."
        );

        return;
    }


    if (compareBtn) {

        compareBtn.disabled =
            true;

        compareBtn.innerHTML =
            `<span>ANALYZING LUNAR DATA...</span><b>◌</b>`;
    }


    setText(
        "status",
        "ANALYSIS IN PROGRESS"
    );


    setText(
        "score",
        "···"
    );


    setText(
        "features",
        "Processing"
    );


    setText(
        "confidence",
        "Processing"
    );


    setText(
        "quality",
        "Processing"
    );


    setText(
        "time",
        "Processing"
    );


    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            true;
    }


    if (correspondenceMap) {

        correspondenceMap.src = "";

        correspondenceMap.style.display =
            "none";
    }


    if (visualPlaceholder) {

        visualPlaceholder.style.display =
            "block";
    }


    if (visualNote) {

        visualNote.textContent =
            "● Running browser-side feature extraction and geometric verification...";
    }


    pipelineRunning();


    const startTime =
        performance.now();


    let matA =
        null;

    let matB =
        null;

    let featuresA =
        null;

    let featuresB =
        null;

    let geometry =
        null;


    try {

        // -------------------------------------------------
        // LOAD OPENCV
        // -------------------------------------------------

        setText(
            "status",
            "LOADING ANALYSIS ENGINE"
        );


        await loadOpenCV();


        // -------------------------------------------------
        // READ IMAGES
        // -------------------------------------------------

        setText(
            "status",
            "READING LUNAR IMAGES"
        );


        matA =
            await fileToMat(
                imageFileA
            );


        matB =
            await fileToMat(
                imageFileB
            );


        pipeline(
            "stageAcquire",
            "complete"
        );


        // -------------------------------------------------
        // QUALITY
        // -------------------------------------------------

        setText(
            "status",
            "PREPROCESSING IMAGES"
        );


        const qualityA =
            calculateImageQuality(
                matA
            );


        const qualityB =
            calculateImageQuality(
                matB
            );


        pipeline(
            "stagePreprocess",
            "complete"
        );


        // -------------------------------------------------
        // FEATURES
        // -------------------------------------------------

        setText(
            "status",
            "EXTRACTING VISUAL FEATURES"
        );


        pipeline(
            "stageExtract",
            "active"
        );


        featuresA =
            extractFeatures(
                matA
            );


        featuresB =
            extractFeatures(
                matB
            );


        pipeline(
            "stageExtract",
            "complete"
        );


        // -------------------------------------------------
        // MATCH
        // -------------------------------------------------

        setText(
            "status",
            "MATCHING CORRESPONDENCES"
        );


        pipeline(
            "stageMatch",
            "active"
        );


        const matches =
            matchFeatures(
                featuresA.descriptors,
                featuresB.descriptors
            );


        pipeline(
            "stageMatch",
            "complete"
        );


        // -------------------------------------------------
        // GEOMETRY
        // -------------------------------------------------

        setText(
            "status",
            "VERIFYING GEOMETRY"
        );


        pipeline(
            "stageVerify",
            "active"
        );


        geometry =
            verifyGeometry(
                featuresA,
                featuresB,
                matches
            );


        pipeline(
            "stageVerify",
            "complete"
        );


        // -------------------------------------------------
        // SCORE
        // -------------------------------------------------

        setText(
            "status",
            "CALCULATING CORRESPONDENCE SCORE"
        );


        pipeline(
            "stageScore",
            "active"
        );


        const score =
            calculateScore(

                matches.length,

                geometry.verifiedMatches,

                geometry.inlierRatio,

                geometry.coverage,

                geometry.geometricConsistency,

                qualityA,

                qualityB
            );


        const classification =
            classify(

                score,

                geometry.verifiedMatches,

                geometry.geometricConsistency,

                geometry.coverage
            );


        // -------------------------------------------------
        // VISUALIZATION
        // -------------------------------------------------

        let visualization =
            null;


        try {

            visualization =
                createCorrespondenceMap(

                    matA,

                    matB,

                    featuresA,

                    featuresB,

                    matches,

                    geometry
                );

        } catch (visualError) {

            console.warn(
                "Visualization warning:",
                visualError
            );
        }


        pipeline(
            "stageScore",
            "complete"
        );


        // -------------------------------------------------
        // FINAL RESULT
        // -------------------------------------------------

        const processingTime =
            (
                performance.now() -
                startTime
            ) / 1000;


        const overallQuality =
            (
                qualityA.score +
                qualityB.score
            ) / 2;


        const result = {

            success:
                true,

            version:
                "LUNARMATCH BROWSER 1.0",

            engine:
                "BROWSER ORB + RANSAC",

            method:
                "ORB + BFMatcher + Homography RANSAC",

            decision:
                classification.decision,

            confidence:
                classification.confidence,

            match_found:
                classification.matchFound,

            match_percentage:
                score,

            correspondence_score:
                score,

            score,

            raw_matches:
                matches.length,

            candidate_matches:
                matches.length,

            verified_matches:
                geometry.verifiedMatches,

            feature_count_a:
                featuresA.count,

            feature_count_b:
                featuresB.count,

            feature_coverage:
                geometry.coverage,

            spatial_coverage:
                geometry.coverage,

            correspondence_strength:
                geometry.inlierRatio,

            inlier_ratio:
                geometry.inlierRatio,

            geometric_consistency:
                geometry.geometricConsistency,

            geometry_score:
                geometry.geometricConsistency,

            geometry_model:
                geometry.model,

            verification_status:
                geometry.verifiedMatches >= 4
                    ? "CORRESPONDENCES VERIFIED"
                    : "INSUFFICIENT VERIFIED CORRESPONDENCES",

            image_quality_a:
                qualityA,

            image_quality_b:
                qualityB,

            overall_quality:
                overallQuality >= 75
                    ? "EXCELLENT"
                    : overallQuality >= 55
                        ? "GOOD"
                        : overallQuality >= 35
                            ? "FAIR"
                            : "LIMITED",

            visualization,

            processing_time:
                processingTime
        };


        pipeline(
            "stageReport",
            "complete"
        );


        displayResults(
            result
        );


    } catch (error) {

        console.error(
            "LUNARMATCH BROWSER ERROR:",
            error
        );


        setText(
            "status",
            "ANALYSIS FAILED"
        );


        setText(
            "score",
            "—"
        );


        setText(
            "features",
            "—"
        );


        setText(
            "confidence",
            "—"
        );


        setText(
            "quality",
            "—"
        );


        setText(
            "time",
            "—"
        );


        pipelineFailed();


        if (correspondenceMap) {

            correspondenceMap.src =
                "";

            correspondenceMap.style.display =
                "none";
        }


        if (visualPlaceholder) {

            visualPlaceholder.style.display =
                "block";
        }


        if (visualNote) {

            visualNote.textContent =
                "● Browser-side analysis could not be completed.";
        }


        if (downloadReportBtn) {

            downloadReportBtn.disabled =
                true;
        }


        alert(
            error.message ||
            "Unable to complete image analysis."
        );

    } finally {

        // -------------------------------------------------
        // CLEAN OPENCV MEMORY
        // -------------------------------------------------

        try {

            if (geometry) {

                if (
                    geometry.homography &&
                    typeof geometry.homography.delete ===
                    "function"
                ) {

                    geometry.homography.delete();
                }


                if (
                    geometry.mask &&
                    typeof geometry.mask.delete ===
                    "function"
                ) {

                    geometry.mask.delete();
                }
            }


            if (featuresA) {

                if (
                    featuresA.gray &&
                    typeof featuresA.gray.delete ===
                    "function"
                ) {

                    featuresA.gray.delete();
                }


                if (
                    featuresA.clahe &&
                    typeof featuresA.clahe.delete ===
                    "function"
                ) {

                    featuresA.clahe.delete();
                }


                if (
                    featuresA.keypoints &&
                    typeof featuresA.keypoints.delete ===
                    "function"
                ) {

                    featuresA.keypoints.delete();
                }


                if (
                    featuresA.descriptors &&
                    typeof featuresA.descriptors.delete ===
                    "function"
                ) {

                    featuresA.descriptors.delete();
                }
            }


            if (featuresB) {

                if (
                    featuresB.gray &&
                    typeof featuresB.gray.delete ===
                    "function"
                ) {

                    featuresB.gray.delete();
                }


                if (
                    featuresB.clahe &&
                    typeof featuresB.clahe.delete ===
                    "function"
                ) {

                    featuresB.clahe.delete();
                }


                if (
                    featuresB.keypoints &&
                    typeof featuresB.keypoints.delete ===
                    "function"
                ) {

                    featuresB.keypoints.delete();
                }


                if (
                    featuresB.descriptors &&
                    typeof featuresB.descriptors.delete ===
                    "function"
                ) {

                    featuresB.descriptors.delete();
                }
            }


            if (matA) {

                matA.delete();
            }


            if (matB) {

                matB.delete();
            }

        } catch (cleanupError) {

            console.warn(
                "OpenCV cleanup warning:",
                cleanupError
            );
        }


        if (compareBtn) {

            compareBtn.disabled =
                false;

            compareBtn.innerHTML =
                `<span>RUN CORRESPONDENCE ENGINE</span><b>→</b>`;
        }
    }
}


if (compareBtn) {

    compareBtn.addEventListener(
        "click",
        compareImages
    );
}


// =========================================================
// REPORT / EXPORT
// =========================================================

async function downloadReport() {

    if (!lastAnalysis) {

        alert(
            "Please run the correspondence analysis first."
        );

        return;
    }


    /*
       The old /api/report endpoint depends on the
       heavy Python backend.

       For the free browser deployment we instead
       export the completed analysis as a local JSON
       report. No server required.
    */

    try {

        const report = {

            project:
                "LUNARMATCH",

            title:
                "LUNARMATCH Lunar Image Correspondence Report",

            generated:
                new Date().toISOString(),

            engine:
                lastAnalysis.engine,

            analysis:
                lastAnalysis
        };


        const blob =
            new Blob(
                [
                    JSON.stringify(
                        report,
                        null,
                        2
                    )
                ],
                {
                    type:
                        "application/json"
                }
            );


        const url =
            URL.createObjectURL(
                blob
            );


        const link =
            document.createElement(
                "a"
            );


        link.href =
            url;


        link.download =
            "LUNARMATCH_Analysis_Report.json";


        document.body.appendChild(
            link
        );


        link.click();


        link.remove();


        URL.revokeObjectURL(
            url
        );

    } catch (error) {

        console.error(
            "REPORT ERROR:",
            error
        );


        alert(
            "Unable to export the analysis report."
        );
    }
}


if (downloadReportBtn) {

    downloadReportBtn.addEventListener(
        "click",
        downloadReport
    );
}


// =========================================================
// NAVIGATION
// =========================================================

document
    .querySelectorAll("nav a")
    .forEach(link => {

        link.addEventListener(
            "click",
            () => {

                document
                    .querySelectorAll(
                        "nav a"
                    )
                    .forEach(item => {

                        item.classList.remove(
                            "active"
                        );
                    });


                link.classList.add(
                    "active"
                );
            }
        );
    });


// =========================================================
// CLEANUP
// =========================================================

window.addEventListener(
    "beforeunload",
    () => {

        if (previewURL_A) {

            URL.revokeObjectURL(
                previewURL_A
            );
        }


        if (previewURL_B) {

            URL.revokeObjectURL(
                previewURL_B
            );
        }
    }
);


// =========================================================
// STARTUP
// =========================================================

console.log(
    "LUNARMATCH browser analysis engine initialized."
);

console.log(
    "Engine: OpenCV.js ORB + BFMatcher + Homography RANSAC"
);
