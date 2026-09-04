// =========================================================
// LUNARMATCH — FRONTEND ANALYSIS ENGINE
// =========================================================

// -----------------------------
// ELEMENT REFERENCES
// -----------------------------

const fileA = document.getElementById("fileA");
const fileB = document.getElementById("fileB");

const previewA = document.getElementById("previewA");
const previewB = document.getElementById("previewB");

const dropA = document.querySelectorAll(".drop-new")[0];
const dropB = document.querySelectorAll(".drop-new")[1];

const compareBtn = document.getElementById("compareBtn");

const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const featuresEl = document.getElementById("features");
const confidenceEl = document.getElementById("confidence");
const qualityEl = document.getElementById("quality");
const timeEl = document.getElementById("time");


// =========================================================
// STATE
// =========================================================

let previewURL_A = null;
let previewURL_B = null;


// =========================================================
// IMAGE INPUT
// =========================================================

function setupImageInput(input, preview, drop, name) {

  // -----------------------------
  // NORMAL FILE SELECTION
  // -----------------------------

  input.addEventListener("change", () => {

    const file = input.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {

      alert(`${name}: Please select a valid image file.`);

      input.value = "";

      return;
    }

    showPreview(file, preview, drop, name);

  });


  // -----------------------------
  // DRAG OVER
  // -----------------------------

  drop.addEventListener("dragover", (event) => {

    event.preventDefault();

    drop.classList.add("dragging");

  });


  // -----------------------------
  // DRAG LEAVE
  // -----------------------------

  drop.addEventListener("dragleave", () => {

    drop.classList.remove("dragging");

  });


  // -----------------------------
  // DROP
  // -----------------------------

  drop.addEventListener("drop", (event) => {

    event.preventDefault();

    drop.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (!file || !file.type.startsWith("image/")) {

      alert(`${name}: Please drop a valid image file.`);

      return;
    }

    try {

      const dataTransfer = new DataTransfer();

      dataTransfer.items.add(file);

      input.files = dataTransfer.files;

    } catch (error) {

      console.error("File assignment error:", error);

    }

    showPreview(file, preview, drop, name);

  });

}


// =========================================================
// SHOW SINGLE IMAGE PREVIEW
// =========================================================

function showPreview(file, preview, drop, name) {

  // Revoke old preview URL
  if (name === "IMAGE A" && previewURL_A) {

    URL.revokeObjectURL(previewURL_A);

  }

  if (name === "IMAGE B" && previewURL_B) {

    URL.revokeObjectURL(previewURL_B);

  }


  const objectURL = URL.createObjectURL(file);


  if (name === "IMAGE A") {

    previewURL_A = objectURL;

  } else {

    previewURL_B = objectURL;

  }


  // IMPORTANT:
  // One input = one preview image.

  preview.src = objectURL;

  preview.style.display = "block";

  preview.style.visibility = "visible";

  preview.style.opacity = "1";


  const content = drop.querySelector(".drop-content");

  if (content) {

    content.style.opacity = "0";

  }


  drop.classList.add("has-image");

  drop.dataset.filename = file.name;


  // Update visual label

  const strong = drop.querySelector(".drop-content strong");

  if (strong) {

    strong.textContent = file.name;

  }

}


// =========================================================
// INITIALIZE IMAGE INPUTS
// =========================================================

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
// RESULT VISUALIZATION
// =========================================================
//
// The backend returns ONE correspondence map containing
// Image A + Image B + verified feature connections.
//
// Therefore we must NOT place the same image into resultA
// and resultB.
//
// Instead, create ONE dedicated correspondence image.
// =========================================================

const visualImages = document.querySelector(".visual-images");

let correspondenceMap = null;


function createCorrespondenceViewer() {

  if (!visualImages) return;

  // Remove previous generated viewer

  if (correspondenceMap) {

    correspondenceMap.remove();

    correspondenceMap = null;

  }


  // Hide the two old result images.
  // They were causing the combined map to appear twice.

  if (document.getElementById("resultA")) {

    document.getElementById("resultA").style.display = "none";

  }

  if (document.getElementById("resultB")) {

    document.getElementById("resultB").style.display = "none";

  }


  // Clear existing generated viewer

  const oldViewer =
    visualImages.querySelector(".correspondence-viewer");

  if (oldViewer) {

    oldViewer.remove();

  }


  // Create one clean correspondence viewer

  const wrapper = document.createElement("div");

  wrapper.className = "correspondence-viewer";

  wrapper.style.width = "100%";
  wrapper.style.display = "block";


  const label = document.createElement("label");

  label.textContent = "IMAGE A  ↔  IMAGE B";

  label.style.display = "block";
  label.style.marginBottom = "8px";


  const image = document.createElement("img");

  image.id = "correspondenceMap";

  image.alt =
    "LUNARMATCH verified feature correspondence map";

  image.style.width = "100%";
  image.style.height = "auto";
  image.style.display = "none";
  image.style.borderRadius = "8px";


  wrapper.appendChild(label);

  wrapper.appendChild(image);

  visualImages.appendChild(wrapper);

  correspondenceMap = image;

}


// Create viewer immediately

createCorrespondenceViewer();


// =========================================================
// RESET RESULT
// =========================================================

function resetResults() {

  statusEl.textContent = "READY FOR ANALYSIS";

  scoreEl.textContent = "—";

  featuresEl.textContent = "—";

  confidenceEl.textContent = "—";

  qualityEl.textContent = "—";

  timeEl.textContent = "—";


  if (correspondenceMap) {

    correspondenceMap.src = "";

    correspondenceMap.style.display = "none";

  }

}


// =========================================================
// DISPLAY DETAILED RESULT
// =========================================================

function displayResults(data) {

  // -----------------------------
  // STATUS
  // -----------------------------

  if (data.match_found) {

    statusEl.textContent = "MATCH FOUND";

  } else {

    statusEl.textContent = "NO STRONG MATCH";

  }


  // -----------------------------
  // MATCH SCORE
  // -----------------------------

  const matchScore =
    Number(data.match_percentage);

  if (Number.isFinite(matchScore)) {

    scoreEl.textContent =
      matchScore.toFixed(1) + "%";

  } else {

    scoreEl.textContent = "—";

  }


  // -----------------------------
  // VERIFIED FEATURES
  // -----------------------------

  const verified =
    data.verified_matches ??
    data.corresponding_features ??
    0;

  const candidate =
    data.candidate_matches ??
    0;


  if (candidate > 0) {

    featuresEl.textContent =
      `${verified} / ${candidate}`;

  } else {

    featuresEl.textContent =
      String(verified);

  }


  // -----------------------------
  // CONFIDENCE
  // -----------------------------

  confidenceEl.textContent =
    data.confidence ?? "—";


  // -----------------------------
  // QUALITY
  // -----------------------------

  qualityEl.textContent =
    data.quality ?? "—";


  // -----------------------------
  // PROCESSING TIME
  // -----------------------------

  const processingTime =
    Number(data.processing_time);

  if (Number.isFinite(processingTime)) {

    timeEl.textContent =
      processingTime.toFixed(2) + " sec";

  } else {

    timeEl.textContent = "—";

  }


  // -----------------------------
  // CORRESPONDENCE MAP
  // -----------------------------

  if (
    data.visualization &&
    correspondenceMap
  ) {

    correspondenceMap.src =
      "data:image/jpeg;base64," +
      data.visualization;

    correspondenceMap.style.display =
      "block";

  }


  // -----------------------------
  // UPDATE VISUALIZATION LABEL
  // -----------------------------

  const visualHeader =
    document.querySelector(".visual-header small");

  if (visualHeader) {

    visualHeader.textContent =
      "SIFT + RANSAC";

  }


  // -----------------------------
  // UPDATE VISUAL NOTE
  // -----------------------------

  const visualNote =
    document.querySelector(".visual-note");

  if (visualNote) {

    const geometric =
      data.geometric_consistency;

    const verifiedCount =
      data.verified_matches ??
      data.corresponding_features ??
      0;

    if (geometric !== undefined) {

      visualNote.textContent =
        `● ${verifiedCount} verified correspondences · ` +
        `${Number(geometric).toFixed(1)}% geometric consistency`;

    } else {

      visualNote.textContent =
        "● Verified feature correspondences returned by the analysis engine.";

    }

  }

}


// =========================================================
// ANALYSIS
// =========================================================

async function compareImages() {

  // -----------------------------
  // VALIDATE INPUTS
  // -----------------------------

  const imageA =
    fileA.files[0];

  const imageB =
    fileB.files[0];


  if (!imageA || !imageB) {

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


  // -----------------------------
  // BUTTON
  // -----------------------------

  compareBtn.disabled = true;

  compareBtn.innerHTML = `
    <span>ANALYZING LUNAR DATA...</span>
    <b>◌</b>
  `;


  // -----------------------------
  // STATUS
  // -----------------------------

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


  if (correspondenceMap) {

    correspondenceMap.src = "";

    correspondenceMap.style.display =
      "none";

  }


  // -----------------------------
  // FORM DATA
  // -----------------------------

  const formData =
    new FormData();

  formData.append(
    "image1",
    imageA
  );

  formData.append(
    "image2",
    imageB
  );


  // -----------------------------
  // SEND TO BACKEND
  // -----------------------------

  try {

    const response =
      await fetch(
        "/api/match",
        {
          method: "POST",
          body: formData
        }
      );


    // -----------------------------
    // RESPONSE
    // -----------------------------

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


    // -----------------------------
    // DISPLAY RESULT
    // -----------------------------

    displayResults(data);


    console.log(
      "LUNARMATCH ANALYSIS:",
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


    if (correspondenceMap) {

      correspondenceMap.src = "";

      correspondenceMap.style.display =
        "none";

    }


    alert(
      error.message ||
      "Unable to complete image analysis."
    );


  } finally {

    // -----------------------------
    // RESTORE BUTTON
    // -----------------------------

    compareBtn.disabled = false;

    compareBtn.innerHTML = `
      <span>RUN CORRESPONDENCE ENGINE</span>
      <b>→</b>
    `;

  }

}


// =========================================================
// COMPARE BUTTON
// =========================================================

compareBtn.addEventListener(
  "click",
  compareImages
);


// =========================================================
// NAVIGATION
// =========================================================

document
  .querySelectorAll("nav a")
  .forEach((link) => {

    link.addEventListener(
      "click",
      () => {

        document
          .querySelectorAll("nav a")
          .forEach((item) => {

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
