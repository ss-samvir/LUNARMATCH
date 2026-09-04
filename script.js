// =========================================================
// LUNARMATCH — PREMIUM FRONTEND ANALYSIS SYSTEM
// =========================================================

const fileA = document.getElementById("fileA");
const fileB = document.getElementById("fileB");

const previewA = document.getElementById("previewA");
const previewB = document.getElementById("previewB");

const dropA = document.querySelectorAll(".drop-new")[0];
const dropB = document.querySelectorAll(".drop-new")[1];

const compareBtn = document.getElementById("compareBtn");
const downloadReportBtn = document.getElementById("downloadReportBtn");

const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const featuresEl = document.getElementById("features");
const confidenceEl = document.getElementById("confidence");
const qualityEl = document.getElementById("quality");
const timeEl = document.getElementById("time");

const correspondenceMap = document.getElementById("correspondenceMap");
const visualPlaceholder = document.getElementById("visualPlaceholder");
const visualNote = document.getElementById("visualNote");

let previewURL_A = null;
let previewURL_B = null;
let lastAnalysis = null;


// =========================================================
// SAFE DOM HELPER
// =========================================================

function setText(id, value) {

    const element = document.getElementById(id);

    if (element) {

        element.textContent =
            value === undefined ||
            value === null ||
            value === ""
                ? "—"
                : value;
    }
}


// =========================================================
// IMAGE INPUT
// =========================================================

function setupImageInput(input, preview, drop, name) {

    input.addEventListener("change", () => {

        const file = input.files[0];

        if (!file) return;

        if (!file.type.startsWith("image/")) {

            alert(`${name}: Please select a valid image file.`);

            input.value = "";

            return;
        }

        showPreview(
            file,
            preview,
            drop,
            name
        );
    });


    drop.addEventListener("dragover", (event) => {

        event.preventDefault();

        drop.classList.add("dragging");
    });


    drop.addEventListener("dragleave", () => {

        drop.classList.remove("dragging");
    });


    drop.addEventListener("drop", (event) => {

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

            const dataTransfer =
                new DataTransfer();

            dataTransfer.items.add(file);

            input.files =
                dataTransfer.files;

        } catch (error) {

            console.error(
                "File assignment error:",
                error
            );
        }


        showPreview(
            file,
            preview,
            drop,
            name
        );
    });
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


    const objectURL =
        URL.createObjectURL(file);


    if (name === "IMAGE A") {

        previewURL_A =
            objectURL;

    } else {

        previewURL_B =
            objectURL;
    }


    preview.src =
        objectURL;

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
}


setupImageInput(
    fileA,
    previewA,
    dropA,
    "IMAGE A"
);


setupImageInput(
    fileB,
    previewB,
    dropB,
    "IMAGE B"
);


// =========================================================
// RESET RESULTS
// =========================================================

function resetResults() {

    if (statusEl)
        statusEl.textContent =
            "READY FOR ANALYSIS";

    if (scoreEl)
        scoreEl.textContent =
            "—";

    if (featuresEl)
        featuresEl.textContent =
            "—";

    if (confidenceEl)
        confidenceEl.textContent =
            "—";

    if (qualityEl)
        qualityEl.textContent =
            "—";

    if (timeEl)
        timeEl.textContent =
            "—";


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


    const pipelineIds = [

        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"
    ];


    pipelineIds.forEach(id => {

        const element =
            document.getElementById(id);

        if (element) {

            element.classList.remove(
                "complete",
                "active",
                "failed"
            );
        }
    });


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
            "● Verified feature correspondences will appear here after analysis.";
    }


    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            true;
    }


    const interpretation =
        document.getElementById(
            "interpretation"
        );

    if (interpretation) {

        interpretation.textContent =
            "Upload two lunar images and run the correspondence engine to generate a detailed assessment.";
    }


    lastAnalysis = null;
}


resetResults();


// =========================================================
// PIPELINE STATUS
// =========================================================

function setPipelineStage(
    id,
    state
) {

    const element =
        document.getElementById(id);

    if (!element) return;

    element.classList.remove(
        "complete",
        "active",
        "failed"
    );

    if (state) {

        element.classList.add(
            state
        );
    }
}


function setPipelineRunning() {

    setPipelineStage(
        "stageAcquire",
        "complete"
    );

    setPipelineStage(
        "stagePreprocess",
        "active"
    );

    setPipelineStage(
        "stageExtract",
        ""
    );

    setPipelineStage(
        "stageMatch",
        ""
    );

    setPipelineStage(
        "stageVerify",
        ""
    );

    setPipelineStage(
        "stageScore",
        ""
    );

    setPipelineStage(
        "stageReport",
        ""
    );
}


function setPipelineComplete() {

    [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore",
        "stageReport"
    ].forEach(id => {

        setPipelineStage(
            id,
            "complete"
        );
    });
}


function setPipelineFailed() {

    [
        "stageAcquire",
        "stagePreprocess",
        "stageExtract",
        "stageMatch",
        "stageVerify",
        "stageScore"
    ].forEach(id => {

        const element =
            document.getElementById(id);

        if (element) {

            element.classList.remove(
                "active"
            );

            element.classList.add(
                "failed"
            );
        }
    });
}


// =========================================================
// FORMATTING
// =========================================================

function formatPercent(value) {

    const number =
        Number(value);

    if (!Number.isFinite(number)) {

        return "—";
    }

    return number.toFixed(1) + "%";
}


function formatNumber(value) {

    const number =
        Number(value);

    if (!Number.isFinite(number)) {

        return "—";
    }

    return number.toLocaleString();
}


function formatScore(value) {

    const number =
        Number(value);

    if (!Number.isFinite(number)) {

        return "—";
    }

    return number.toFixed(1) + "%";
}


// =========================================================
// IMAGE QUALITY DISPLAY
// =========================================================

function displayImageQuality(
    data,
    prefix
) {

    if (!data) return;


    setText(
        `resolution${prefix}`,
        data.resolution
    );


    setText(
        `contrast${prefix}`,
        data.contrast_score !== undefined
            ? Number(data.contrast_score).toFixed(1)
            : "—"
    );


    setText(
        `sharpness${prefix}`,
        data.sharpness_score !== undefined
            ? Number(data.sharpness_score).toFixed(1)
            : "—"
    );


    setText(
        `qualityScore${prefix}`,
        data.quality_score !== undefined
            ? Number(data.quality_score).toFixed(1)
            : "—"
    );
}


// =========================================================
// AUTOMATED INTERPRETATION
// =========================================================

function generateInterpretation(
    data
) {

    const score =
        Number(data.match_percentage) || 0;

    const verified =
        Number(
            data.verified_matches ??
            data.corresponding_features ??
            0
        );

    const candidate =
        Number(
            data.candidate_matches ??
            0
        );

    const geometric =
        Number(
            data.geometric_consistency ??
            data.inlier_ratio ??
            0
        );

    const confidence =
        typeof data.confidence === "string"
            ? data.confidence
            : "Unclassified";

    const quality =
        typeof data.quality === "string"
            ? data.quality
            : "Unknown";


    let assessment;


    if (
        confidence === "High"
    ) {

        assessment =
            "The correspondence evidence is strong, with a substantial number of verified feature relationships and good geometric consistency.";

    } else if (
        confidence === "Medium"
    ) {

        assessment =
            "The system detected meaningful correspondence evidence, although the geometric or feature evidence is not strong enough for the highest confidence classification.";

    } else {

        assessment =
            "The system detected limited correspondence evidence. The result should be treated cautiously and may benefit from higher-quality imagery or images with greater shared surface detail.";
    }


    let matchDescription;


    if (score >= 70) {

        matchDescription =
            "The overall correspondence score is strong.";

    } else if (score >= 40) {

        matchDescription =
            "The overall correspondence score indicates a moderate level of correspondence.";

    } else if (score >= 15) {

        matchDescription =
            "The overall correspondence score indicates limited-to-moderate correspondence.";

    } else {

        matchDescription =
            "The overall correspondence score is low.";
    }


    const interpretationText =

        `${assessment} ${matchDescription} ` +

        `The analysis identified ${formatNumber(verified)} verified correspondences ` +

        `from ${formatNumber(candidate)} candidate matches, ` +

        `with approximately ${formatPercent(geometric)} geometric consistency. ` +

        `The assessed image quality was ${quality.toLowerCase()}. ` +

        `This computer-vision result represents measurable image correspondence evidence rather than a probability of geographic identity.`;


    setText(
        "interpretation",
        interpretationText
    );
}


// =========================================================
// DISPLAY ANALYSIS RESULTS
// =========================================================

function displayResults(
    data
) {

    lastAnalysis =
        data;


    const matchFound =
        Boolean(data.match_found);


    statusEl.textContent =
        matchFound
            ? "MATCH FOUND"
            : "NO STRONG MATCH";


    scoreEl.textContent =
        formatScore(
            data.match_percentage
        );


    const verified =
        Number(
            data.verified_matches ??
            data.corresponding_features ??
            0
        );


    const candidate =
        Number(
            data.candidate_matches ??
            0
        );


    featuresEl.textContent =
        candidate > 0
            ? `${formatNumber(verified)} / ${formatNumber(candidate)}`
            : formatNumber(verified);


    confidenceEl.textContent =
        data.confidence ||
        "—";


    qualityEl.textContent =
        data.quality ||
        "—";


    const processingTime =
        Number(
            data.processing_time ??
            data.engine_time
        );


    timeEl.textContent =
        Number.isFinite(
            processingTime
        )
            ? processingTime.toFixed(2) + " sec"
            : "—";


    displayImageQuality(
        data.image_quality_a,
        "A"
    );


    setText(
        "keypointsA",
        formatNumber(
            data.feature_count_a
        )
    );


    displayImageQuality(
        data.image_quality_b,
        "B"
    );


    setText(
        "keypointsB",
        formatNumber(
            data.feature_count_b
        )
    );


    setText(
        "rawMatches",
        formatNumber(
            data.raw_matches
        )
    );


    setText(
        "candidateMatches",
        formatNumber(
            data.candidate_matches
        )
    );


    setText(
        "verifiedMatches",
        formatNumber(
            verified
        )
    );


    let coverage =
        data.feature_coverage;


    if (
        coverage === undefined ||
        coverage === null
    ) {

        const minimumFeatures =
            Math.max(
                1,
                Math.min(
                    Number(
                        data.feature_count_a
                    ) || 1,
                    Number(
                        data.feature_count_b
                    ) || 1
                )
            );


        coverage =
            (
                verified /
                minimumFeatures
            ) * 100;
    }


    setText(
        "featureCoverage",
        formatPercent(
            coverage
        )
    );


    const correspondenceStrength =
        data.correspondence_strength ??
        data.match_percentage;


    setText(
        "correspondenceStrength",
        formatPercent(
            correspondenceStrength
        )
    );


    const inlierRatio =
        data.inlier_ratio ??
        data.geometric_consistency;


    setText(
        "inlierRatio",
        formatPercent(
            inlierRatio
        )
    );


    setText(
        "geometricConsistency",
        formatPercent(
            data.geometric_consistency
        )
    );


    setText(
        "homographyStatus",
        data.homography_verified
            ? "VERIFIED"
            : "NOT VERIFIED"
    );


    setText(
        "verificationStatus",
        verified >= 1
            ? "CORRESPONDENCES VERIFIED"
            : "NO VERIFIED CORRESPONDENCES"
    );


    if (
        data.visualization &&
        correspondenceMap
    ) {

        correspondenceMap.src =
            "data:image/jpeg;base64," +
            data.visualization;


        correspondenceMap.style.display =
            "block";


        if (visualPlaceholder) {

            visualPlaceholder.style.display =
                "none";
        }
    }


    if (visualNote) {

        const geometric =
            Number(
                data.geometric_consistency
            );


        if (
            Number.isFinite(
                geometric
            )
        ) {

            visualNote.textContent =
                `● ${formatNumber(verified)} verified correspondences · ${formatPercent(geometric)} geometric consistency`;

        } else {

            visualNote.textContent =
                `● ${formatNumber(verified)} verified feature correspondences returned by the analysis engine.`;
        }
    }


    const visualEngine =
        document.getElementById(
            "visualEngine"
        );


    if (visualEngine) {

        visualEngine.textContent =
            "SIFT + FLANN + RANSAC";
    }


    generateInterpretation(
        data
    );


    setPipelineComplete();


    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            false;
    }
}


// =========================================================
// RUN ANALYSIS
// =========================================================

async function compareImages() {

    const imageA =
        fileA.files[0];

    const imageB =
        fileB.files[0];


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


    compareBtn.disabled =
        true;


    compareBtn.innerHTML =
        `<span>ANALYZING LUNAR DATA...</span><b>◌</b>`;


    statusEl.textContent =
        "ANALYSIS IN PROGRESS";


    scoreEl.textContent =
        "···";


    featuresEl.textContent =
        "Processing";


    confidenceEl.textContent =
        "Processing";


    qualityEl.textContent =
        "Processing";


    timeEl.textContent =
        "Processing";


    if (downloadReportBtn) {

        downloadReportBtn.disabled =
            true;
    }


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


    setPipelineRunning();


    // =====================================================
    // IMPORTANT:
    // Backend expects imageA and imageB
    // =====================================================

    const formData =
        new FormData();


    formData.append(
        "imageA",
        imageA
    );


    formData.append(
        "imageB",
        imageB
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

        } catch (jsonError) {

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
            data.processing_time === undefined
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


        console.log(
            "LUNARMATCH PREMIUM ANALYSIS:",
            data
        );


    } catch (error) {

        console.error(
            "LUNARMATCH ERROR:",
            error
        );


        statusEl.textContent =
            "ANALYSIS FAILED";


        scoreEl.textContent =
            "—";


        featuresEl.textContent =
            "—";


        confidenceEl.textContent =
            "—";


        qualityEl.textContent =
            "—";


        timeEl.textContent =
            "—";


        setPipelineFailed();


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


        if (downloadReportBtn) {

            downloadReportBtn.disabled =
                true;
        }


        alert(
            error.message ||
            "Unable to complete image analysis."
        );


    } finally {

        compareBtn.disabled =
            false;


        compareBtn.innerHTML =
            `<span>RUN CORRESPONDENCE ENGINE</span><b>→</b>`;
    }
}


compareBtn.addEventListener(
    "click",
    compareImages
);


// =========================================================
// PDF REPORT
// =========================================================

async function downloadReport() {

    const imageA =
        fileA.files[0];

    const imageB =
        fileB.files[0];


    if (
        !imageA ||
        !imageB
    ) {

        alert(
            "Please upload both images before generating the report."
        );

        return;
    }


    if (!lastAnalysis) {

        alert(
            "Please run the correspondence analysis first."
        );

        return;
    }


    downloadReportBtn.disabled =
        true;


    downloadReportBtn.innerHTML =
        "GENERATING REPORT...";


    const formData =
        new FormData();


    // =====================================================
    // IMPORTANT:
    // Backend expects imageA and imageB
    // =====================================================

    formData.append(
        "imageA",
        imageA
    );


    formData.append(
        "imageB",
        imageB
    );


    try {

        const response =
            await fetch(
                "/api/report",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            let message =
                "Could not generate PDF report.";


            try {

                const data =
                    await response.json();

                if (data.error) {

                    message =
                        data.error;
                }

            } catch (_) {}


            throw new Error(
                message
            );
        }


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
            "PDF REPORT ERROR:",
            error
        );


        alert(
            error.message ||
            "Unable to generate the PDF report."
        );


    } finally {

        downloadReportBtn.disabled =
            false;


        downloadReportBtn.innerHTML =
            `DOWNLOAD PDF REPORT <span>↓</span>`;
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
// INITIAL SYSTEM MESSAGE
// =========================================================

console.log(
    "LUNARMATCH Premium Frontend initialized."
);
