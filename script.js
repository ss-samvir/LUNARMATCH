// =========================================================
// LUNARMATCH V5 — HYBRID AI FRONTEND
// LoFTR + Multi-view SIFT + Robust Geometric Verification
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


function stringValue(...values) {

    for (const value of values) {

        if (
            value !== undefined &&
            value !== null &&
            String(value).trim() !== ""
        ) {
            return String(value);
        }
    }

    return "";
}


function percent(value, decimals = 1) {

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return "—";
    }

    return `${n.toFixed(decimals)}%`;
}


function numberFormat(value) {

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return "—";
    }

    return n.toLocaleString();
}


function seconds(value) {

    const n = Number(value);

    if (!Number.isFinite(n)) {
        return "—";
    }

    return `${n.toFixed(2)} sec`;
}


function normalizeConfidence(value) {

    return String(value || "")
        .trim()
        .toLowerCase();
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

            const file = input.files[0];

            if (!file) {
                return;
            }

            if (!file.type.startsWith("image/")) {

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

            drop.classList.add("dragging");
        }
    );


    drop.addEventListener(
        "dragleave",
        () => {

            drop.classList.remove("dragging");
        }
    );


    drop.addEventListener(
        "drop",
        event => {

            event.preventDefault();

            drop.classList.remove("dragging");

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
        el.classList.add(state);
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
        id => pipeline(
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

function displayQuality(
    data,
    prefix
) {

    if (!data) return;


    setText(
        `resolution${prefix}`,
        data.resolution
    );


    const contrast =
        numberValue(
            data.contrast
        );


    setText(
        `contrast${prefix}`,
        Number.isFinite(contrast)
            ? contrast.toFixed(1)
            : "—"
    );


    const sharpness =
        numberValue(
            data.sharpness
        );


    setText(
        `sharpness${prefix}`,
        Number.isFinite(sharpness)
            ? sharpness.toFixed(1)
            : "—"
    );


    const score =
        numberValue(
            data.score,
            data.quality_score
        );


    setText(
        `qualityScore${prefix}`,
        Number.isFinite(score)
            ? score.toFixed(1)
            : "—"
    );
}


function overallQuality(data) {

    const a =
        numberValue(
            data.image_quality_a?.score,
            data.image_quality_a?.quality_score
        );


    const b =
        numberValue(
            data.image_quality_b?.score,
            data.image_quality_b?.quality_score
        );


    const values = [
        a,
        b
    ].filter(
        Number.isFinite
    );


    if (!values.length) {
        return "—";
    }


    const average =
        values.reduce(
            (x, y) => x + y,
            0
        ) / values.length;


    if (average >= 75) {
        return "EXCELLENT";
    }

    if (average >= 55) {
        return "GOOD";
    }

    if (average >= 35) {
        return "FAIR";
    }

    return "LIMITED";
}


// =========================================================
// RESULTS
// =========================================================

function displayResults(data) {

    lastAnalysis =
        data;


    // =====================================================
    // MAIN SCORE
    // =====================================================

    const score =
        numberValue(
            data.match_percentage,
            data.correspondence_score,
            data.score
        );


    const confidence =
        stringValue(
            data.confidence
        );


    const decision =
        stringValue(
            data.decision,
            data.status
        );


    // =====================================================
    // LEARNED MATCHER COUNTS
    // =====================================================

    const loftrCandidates =
        numberValue(
            data.loftr_candidate_matches,
            data.learned_candidate_matches,
            data.candidate_matches
        );


    const loftrVerified =
        numberValue(
            data.loftr_verified_matches,
            data.learned_verified_matches,
            data.verified_matches
        );


    // =====================================================
    // SIFT COUNTS
    // =====================================================

    const siftCandidates =
        numberValue(
            data.sift_candidate_matches
        );


    const siftVerified =
        numberValue(
            data.sift_verified_matches
        );


    // =====================================================
    // TOP RESULT
    // =====================================================

    let statusText =
        decision;


    if (!statusText) {

        statusText =
            data.match_found
                ? "CORRESPONDENCE DETECTED"
                : "NO RELIABLE CORRESPONDENCE";
    }


    setText(
        "status",
        statusText
    );


    setText(
        "score",
        percent(score)
    );


    if (
        Number.isFinite(loftrVerified) &&
        Number.isFinite(loftrCandidates)
    ) {

        setText(
            "features",
            `${numberFormat(loftrVerified)} / ${numberFormat(loftrCandidates)}`
        );

    } else {

        setText(
            "features",
            "—"
        );
    }


    setText(
        "confidence",
        confidence || "—"
    );


    setText(
        "quality",
        overallQuality(data)
    );


    setText(
        "time",
        seconds(
            data.processing_time
        )
    );


    // =====================================================
    // IMAGE A
    // =====================================================

    displayQuality(
        data.image_quality_a,
        "A"
    );


    setText(
        "keypointsA",
        numberFormat(
            data.feature_count_a
        )
    );


    // =====================================================
    // IMAGE B
    // =====================================================

    displayQuality(
        data.image_quality_b,
        "B"
    );


    setText(
        "keypointsB",
        numberFormat(
            data.feature_count_b
        )
    );


    // =====================================================
    // MATCH COUNTS
    // =====================================================

    /*
       RAW MATCHES

       Prefer an explicitly returned raw-match count.
       Otherwise use learned candidates.

       This prevents us from inventing a value.
    */

    const rawMatches =
        numberValue(
            data.raw_matches,
            data.raw_match_count
        );


    setText(
        "rawMatches",
        Number.isFinite(rawMatches)
            ? numberFormat(rawMatches)
            : Number.isFinite(loftrCandidates)
                ? numberFormat(loftrCandidates)
                : "—"
    );


    setText(
        "candidateMatches",
        Number.isFinite(loftrCandidates)
            ? numberFormat(loftrCandidates)
            : "—"
    );


    setText(
        "verifiedMatches",
        Number.isFinite(loftrVerified)
            ? numberFormat(loftrVerified)
            : "—"
    );


    // =====================================================
    // FEATURE / SPATIAL COVERAGE
    // =====================================================

    const coverage =
        numberValue(
            data.feature_coverage,
            data.spatial_coverage,
            data.coverage
        );


    setText(
        "featureCoverage",
        Number.isFinite(coverage)
            ? percent(coverage)
            : "—"
    );


    // =====================================================
    // CORRESPONDENCE STRENGTH
    // =====================================================

    const strength =
        numberValue(
            data.correspondence_strength,
            data.match_percentage,
            data.correspondence_score
        );


    setText(
        "correspondenceStrength",
        Number.isFinite(strength)
            ? percent(strength)
            : "—"
    );


    // =====================================================
    // GEOMETRIC VERIFICATION
    // =====================================================

    /*
       IMPORTANT:

       Do NOT calculate geometric consistency from
       spatial coverage.

       If the backend provides a real geometry score,
       use that.

       Otherwise show the actual inlier ratio.
    */

    const backendInlierRatio =
        numberValue(
            data.inlier_ratio,
            data.geometry_inlier_ratio
        );


    let inlierRatio =
        backendInlierRatio;


    if (
        !Number.isFinite(inlierRatio) &&
        Number.isFinite(loftrCandidates) &&
        loftrCandidates > 0 &&
        Number.isFinite(loftrVerified)
    ) {

        inlierRatio =
            (
                loftrVerified /
                loftrCandidates
            ) * 100;
    }


    setText(
        "inlierRatio",
        Number.isFinite(inlierRatio)
            ? percent(inlierRatio)
            : "—"
    );


    const geometryConsistency =
        numberValue(
            data.geometric_consistency,
            data.geometry_consistency,
            data.geometry_score
        );


    setText(
        "geometricConsistency",
        Number.isFinite(geometryConsistency)
            ? percent(geometryConsistency)
            : "—"
    );


    // =====================================================
    // GEOMETRIC MODEL
    // =====================================================

    const model =
        stringValue(
            data.geometry_model,
            data.verification_model
        )
        .toLowerCase();


    let modelDisplay =
        "NOT ESTABLISHED";


    if (
        model.includes("homography")
    ) {

        modelDisplay =
            "HOMOGRAPHY / RANSAC";

    } else if (
        model.includes("affine")
    ) {

        modelDisplay =
            "AFFINE / RANSAC";

    } else if (
        model.includes("fundamental")
    ) {

        modelDisplay =
            "FUNDAMENTAL / RANSAC";

    } else if (
        model.includes("ransac")
    ) {

        modelDisplay =
            "RANSAC";
    }


    setText(
        "homographyStatus",
        modelDisplay
    );


    // =====================================================
    // VERIFICATION STATUS
    // =====================================================

    const verification =
        stringValue(
            data.verification_status
        );


    if (verification) {

        setText(
            "verificationStatus",
            verification
        );

    } else if (
        Number.isFinite(loftrVerified) &&
        loftrVerified >= 1
    ) {

        setText(
            "verificationStatus",
            "CORRESPONDENCES VERIFIED"
        );

    } else {

        setText(
            "verificationStatus",
            "NO VERIFIED CORRESPONDENCES"
        );
    }


    // =====================================================
    // VISUALIZATION
    // =====================================================

    if (
        data.visualization &&
        correspondenceMap
    ) {

        let visualizationURL =
            String(
                data.visualization
            );


        /*
           Cache-busting is only needed for the same
           output filename being overwritten repeatedly.
        */

        const separator =
            visualizationURL.includes("?")
                ? "&"
                : "?";


        visualizationURL +=
            `${separator}t=${Date.now()}`;


        correspondenceMap.src =
            visualizationURL;


        correspondenceMap.style.display =
            "block";


        if (visualPlaceholder) {

            visualPlaceholder.style.display =
                "none";
        }
    }


    // =====================================================
    // VISUALIZATION NOTE
    // =====================================================

    if (visualNote) {

        const geometry =
            stringValue(
                data.geometry_model,
                data.verification_model
            );


        if (
            Number.isFinite(loftrVerified)
        ) {

            visualNote.textContent =
                `● ${numberFormat(loftrVerified)} verified learned correspondences · ${geometry || "robust geometric verification"}`;

        } else if (
            Number.isFinite(siftVerified)
        ) {

            visualNote.textContent =
                `● ${numberFormat(siftVerified)} verified classical correspondences · ${geometry || "robust geometric verification"}`;

        } else {

            visualNote.textContent =
                "● Verified feature correspondences generated by the hybrid analysis engine.";
        }
    }


    // =====================================================
    // ENGINE
    // =====================================================

    const visualEngine =
        document.getElementById(
            "visualEngine"
        );


    if (visualEngine) {

        visualEngine.textContent =
            stringValue(
                data.engine,
                data.method
            ) ||
            "HYBRID LOFTR + MULTI-VIEW SIFT";
    }


    // =====================================================
    // INTERPRETATION
    // =====================================================

    generateInterpretation(
        data
    );


    // =====================================================
    // PIPELINE
    // =====================================================

    pipelineComplete();


    // =====================================================
    // REPORT
    // =====================================================

    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            false;
    }


    console.log(
        "LUNARMATCH V5 RESULT:",
        data
    );
}


// =========================================================
// INTERPRETATION
// =========================================================

function generateInterpretation(data) {

    const score =
        numberValue(
            data.match_percentage,
            data.correspondence_score,
            data.score
        );


    const verified =
        numberValue(
            data.loftr_verified_matches,
            data.learned_verified_matches,
            data.verified_matches
        );


    const candidates =
        numberValue(
            data.loftr_candidate_matches,
            data.learned_candidate_matches,
            data.candidate_matches
        );


    const confidence =
        stringValue(
            data.confidence
        );


    const geometry =
        stringValue(
            data.geometry_model,
            data.verification_model
        );


    const confidenceLevel =
        normalizeConfidence(
            confidence
        );


    // =====================================================
    // ASSESSMENT
    // =====================================================

    let assessment;


    if (
        confidenceLevel === "high"
    ) {

        assessment =
            "The hybrid correspondence engine detected strong supporting visual evidence with multiple geometrically consistent relationships.";

    } else if (
        confidenceLevel === "medium"
    ) {

        assessment =
            "The hybrid correspondence engine detected meaningful visual correspondence, but the available geometric evidence does not support a high-confidence classification.";

    } else {

        assessment =
            "The engine detected limited correspondence evidence. The result should therefore be treated cautiously, particularly when illumination, scale, viewpoint, terrain appearance, or image quality differ significantly.";
    }


    // =====================================================
    // SCORE INTERPRETATION
    // =====================================================

    let scoreText;


    if (
        Number.isFinite(score) &&
        score >= 70
    ) {

        scoreText =
            "The overall correspondence evidence is strong.";

    } else if (
        Number.isFinite(score) &&
        score >= 40
    ) {

        scoreText =
            "The overall correspondence evidence is moderate.";

    } else if (
        Number.isFinite(score) &&
        score >= 20
    ) {

        scoreText =
            "The overall correspondence evidence is limited.";

    } else {

        scoreText =
            "The overall correspondence evidence is low.";
    }


    // =====================================================
    // MATCH COUNTS
    // =====================================================

    const verifiedText =
        Number.isFinite(verified)
            ? numberFormat(verified)
            : "an unknown number of";


    const candidateText =
        Number.isFinite(candidates)
            ? numberFormat(candidates)
            : "an unknown number of";


    const geometryText =
        geometry ||
        "a stable geometric model";


    // =====================================================
    // FINAL INTERPRETATION
    // =====================================================

    const text =

        `${assessment} ` +

        `${scoreText} ` +

        `The analysis produced ${candidateText} learned candidate correspondences, ` +

        `of which ${verifiedText} survived robust geometric verification using ${geometryText}. ` +

        `The final correspondence score is ${percent(score)} with ${confidence || "unreported"} confidence. ` +

        `This score represents measured image-correspondence evidence, not a probability that the two images depict the same geographic location.`;



    setText(
        "interpretation",
        text
    );
}


// =========================================================
// RUN ANALYSIS
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


    const imageA =
        fileA.files[0];


    const imageB =
        fileB.files[0];


    // =====================================================
    // VALIDATE INPUTS
    // =====================================================

    if (
        !imageA ||
        !imageB
    ) {

        alert(
            "Please upload both Image A and Image B before starting the analysis."
        );

        return;
    }


    if (
        !imageA.type.startsWith("image/") ||
        !imageB.type.startsWith("image/")
    ) {

        alert(
            "Both files must be valid image files."
        );

        return;
    }


    // =====================================================
    // BUTTON STATE
    // =====================================================

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
            "● Running learned correspondence and geometric verification...";
    }


    pipelineRunning();


    // =====================================================
    // REQUEST
    // =====================================================

    const formData =
        new FormData();


    /*
       IMPORTANT

       The current frontend/backend contract uses:

       imageA
       imageB

       Do NOT change these to image1/image2 unless
       the Flask backend is changed too.
    */

    formData.append(
        "imageA",
        imageA,
        imageA.name
    );


    formData.append(
        "imageB",
        imageB,
        imageB.name
    );


    const startTime =
        performance.now();


    try {

        const response =
            await fetch(
                "/api/match",
                {
                    method: "POST",
                    body: formData
                }
            );


        let data;


        try {

            data =
                await response.json();

        } catch (error) {

            throw new Error(
                "The analysis server returned an invalid response."
            );
        }


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Backend analysis failed."
            );
        }


        if (
            data.processing_time ===
            undefined
        ) {

            data.processing_time =
                (
                    performance.now() -
                    startTime
                ) / 1000;
        }


        displayResults(
            data
        );

    } catch (error) {

        console.error(
            "LUNARMATCH ERROR:",
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
                "● Correspondence visualization unavailable because the analysis failed.";
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
// PDF REPORT
// =========================================================

async function downloadReport() {

    if (
        !fileA ||
        !fileB
    ) {
        return;
    }


    if (!lastAnalysis) {

        alert(
            "Please run the correspondence analysis first."
        );

        return;
    }


    try {

        /*
           First attempt:

           Send the original images + analysis data.
           This is more useful for a Flask report endpoint
           that regenerates the PDF from the source images.
        */

        const formData =
            new FormData();


        const imageA =
            fileA.files[0];


        const imageB =
            fileB.files[0];


        if (imageA) {

            formData.append(
                "imageA",
                imageA,
                imageA.name
            );
        }


        if (imageB) {

            formData.append(
                "imageB",
                imageB,
                imageB.name
            );
        }


        formData.append(
            "analysis",
            JSON.stringify(
                lastAnalysis
            )
        );


        const response =
            await fetch(
                "/api/report",
                {
                    method: "POST",
                    body: formData
                }
            );


        const contentType =
            response.headers.get(
                "content-type"
            ) || "";


        if (!response.ok) {

            let errorText =
                "Could not generate report.";


            try {

                if (
                    contentType.includes(
                        "application/json"
                    )
                ) {

                    const data =
                        await response.json();


                    if (data.error) {

                        errorText =
                            data.error;
                    }
                }

            } catch (_) {}


            throw new Error(
                errorText
            );
        }


        // =================================================
        // JSON REPORT URL
        // =================================================

        if (
            contentType.includes(
                "application/json"
            )
        ) {

            const data =
                await response.json();


            const reportURL =
                data.report ||
                data.url ||
                data.pdf;


            if (reportURL) {

                const link =
                    document.createElement(
                        "a"
                    );


                link.href =
                    reportURL;


                link.target =
                    "_blank";


                link.rel =
                    "noopener";


                document.body.appendChild(
                    link
                );


                link.click();


                link.remove();


                return;
            }
        }


        // =================================================
        // DIRECT PDF RESPONSE
        // =================================================

        const blob =
            await response.blob();


        const url =
            window.URL.createObjectURL(
                blob
            );


        const link =
            document.createElement(
                "a"
            );


        link.href =
            url;


        link.download =
            "LUNARMATCH_Analysis_Report.pdf";


        document.body.appendChild(
            link
        );


        link.click();


        link.remove();


        window.URL.revokeObjectURL(
            url
        );

    } catch (error) {

        console.error(
            "REPORT ERROR:",
            error
        );


        /*
           Fallback:

           If the current backend expects JSON rather
           than multipart form data, try that too.
        */

        try {

            const fallbackResponse =
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
                                lastAnalysis
                            )
                    }
                );


            if (
                fallbackResponse.ok
            ) {

                const contentType =
                    fallbackResponse.headers.get(
                        "content-type"
                    ) || "";


                if (
                    contentType.includes(
                        "application/json"
                    )
                ) {

                    const data =
                        await fallbackResponse.json();


                    const reportURL =
                        data.report ||
                        data.url ||
                        data.pdf;


                    if (reportURL) {

                        window.open(
                            reportURL,
                            "_blank"
                        );

                        return;
                    }
                }


                const blob =
                    await fallbackResponse.blob();


                const url =
                    window.URL.createObjectURL(
                        blob
                    );


                const link =
                    document.createElement(
                        "a"
                    );


                link.href =
                    url;


                link.download =
                    "LUNARMATCH_Analysis_Report.pdf";


                document.body.appendChild(
                    link
                );


                link.click();


                link.remove();


                window.URL.revokeObjectURL(
                    url
                );


                return;
            }

        } catch (fallbackError) {

            console.error(
                "REPORT FALLBACK ERROR:",
                fallbackError
            );
        }


        alert(
            error.message ||
            "Unable to generate report."
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
    "LUNARMATCH V5 frontend initialized."
);

console.log(
    "Engine: LoFTR + Multi-view SIFT + Robust Geometric Verification"
);