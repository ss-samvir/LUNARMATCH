const fileA = document.getElementById("fileA");
const fileB = document.getElementById("fileB");

const previewA = document.getElementById("previewA");
const previewB = document.getElementById("previewB");

const dropA = document.querySelectorAll(".drop-new")[0];
const dropB = document.querySelectorAll(".drop-new")[1];


// =========================================================
// IMAGE INPUT
// =========================================================

function setupImageInput(input, preview, drop, name) {

  input.addEventListener("change", () => {

    const file = input.files[0];

    if (!file) return;

    showPreview(file, preview, drop, name);

  });


  // Drag & drop

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

    const file = event.dataTransfer.files[0];

    if (!file || !file.type.startsWith("image/")) {

      alert("Please select a valid image file.");

      return;
    }

    const dataTransfer = new DataTransfer();

    dataTransfer.items.add(file);

    input.files = dataTransfer.files;

    showPreview(file, preview, drop, name);

  });

}


function showPreview(file, preview, drop, name) {

  preview.src = URL.createObjectURL(file);

  preview.style.display = "block";

  const content = drop.querySelector(".drop-content");

  if (content) {

    content.style.opacity = "0";

  }

  drop.classList.add("has-image");

  drop.dataset.filename = file.name;

}


// Initialize

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
// SMOOTH NAVIGATION
// =========================================================

document.querySelectorAll("nav a").forEach((link) => {

  link.addEventListener("click", () => {

    document.querySelectorAll("nav a")
      .forEach((item) => item.classList.remove("active"));

    link.classList.add("active");

  });

});


// =========================================================
// ANALYSIS
// =========================================================

async function compareImages() {

  if (!fileA.files[0] || !fileB.files[0]) {

    alert("Please upload both Image A and Image B first.");

    return;
  }


  const btn = document.getElementById("compareBtn");

  const status = document.getElementById("status");

  const score = document.getElementById("score");


  // Loading state

  btn.disabled = true;

  btn.innerHTML = `
    <span>ANALYZING LUNAR DATA...</span>
    <b>◌</b>
  `;

  status.textContent = "ANALYSIS IN PROGRESS";

  score.textContent = "···";


  // Reset result values

  document.getElementById("features").textContent = "Processing";

  document.getElementById("confidence").textContent = "Processing";

  document.getElementById("quality").textContent = "Processing";

  document.getElementById("time").textContent = "Processing";


  const formData = new FormData();

  formData.append("image1", fileA.files[0]);

  formData.append("image2", fileB.files[0]);


  try {

    const response = await fetch("/api/match", {

      method: "POST",

      body: formData

    });


    let data;

    try {

      data = await response.json();

    } catch {

      throw new Error(
        "The analysis server returned an invalid response."
      );

    }


    if (!response.ok) {

      throw new Error(
        data.error || "Backend analysis failed."
      );

    }


    // =====================================================
    // RESULTS
    // =====================================================

    status.textContent =
      data.match_found
        ? "MATCH FOUND"
        : "NO STRONG MATCH";


    score.textContent =
      Number(data.match_percentage).toFixed(1) + "%";


    document.getElementById("features").textContent =
      data.corresponding_features ?? "—";


    document.getElementById("confidence").textContent =
      data.confidence ?? "—";


    document.getElementById("quality").textContent =
      data.quality ?? "—";


    document.getElementById("time").textContent =
      Number(data.processing_time).toFixed(2) + " sec";


    // Visualization

    if (data.visualization) {

      const image =
        "data:image/jpeg;base64," +
        data.visualization;

      document.getElementById("resultA").src = image;

      document.getElementById("resultB").src = image;

    }


  } catch (error) {

    console.error("LUNARMATCH:", error);


    status.textContent = "ANALYSIS FAILED";

    score.textContent = "—";

    document.getElementById("features").textContent = "—";

    document.getElementById("confidence").textContent = "—";

    document.getElementById("quality").textContent = "—";

    document.getElementById("time").textContent = "—";


    alert(error.message);

  }


  finally {

    btn.disabled = false;

    btn.innerHTML = `
      <span>RUN CORRESPONDENCE ENGINE</span>
      <b>→</b>
    `;

  }

}


// =========================================================
// BUTTON
// =========================================================

document
  .getElementById("compareBtn")
  .addEventListener("click", compareImages);
