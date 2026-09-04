const fileA = document.getElementById("fileA");
const fileB = document.getElementById("fileB");

function setup(input, img, drop) {
    input.addEventListener("change", () => {
        const f = input.files[0];

        if (!f) return;

        img.src = URL.createObjectURL(f);
        img.style.display = "block";

        const icon = drop.querySelector(".upload-icon");
        const strong = drop.querySelector("strong");
        const small = drop.querySelector("small");

        if (icon) icon.style.display = "none";
        if (strong) strong.style.display = "none";
        if (small) small.style.display = "none";
    });
}

setup(
    fileA,
    document.getElementById("previewA"),
    document.querySelectorAll(".drop")[0]
);

setup(
    fileB,
    document.getElementById("previewB"),
    document.querySelectorAll(".drop")[1]
);

async function compareImages() {

    if (!fileA.files[0] || !fileB.files[0]) {
        alert("Please upload both Image A and Image B first.");
        return;
    }

    const btn = document.getElementById("compareBtn");
    const status = document.getElementById("status");
    const score = document.getElementById("score");

    btn.disabled = true;
    btn.textContent = "Analyzing lunar images...";
    status.textContent = "ANALYZING...";
    score.textContent = "...";

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
                "The server returned an invalid response. Please check the backend."
            );
        }

        if (!response.ok) {
            throw new Error(
                data.error || `Server error (${response.status})`
            );
        }

        status.textContent = data.match_found
            ? "MATCH FOUND"
            : "NO STRONG MATCH";

        score.textContent =
            Number(data.match_percentage || 0).toFixed(1) + "%";

        document.getElementById("features").textContent =
            data.corresponding_features ?? "—";

        document.getElementById("confidence").textContent =
            data.confidence ?? "—";

        document.getElementById("quality").textContent =
            data.quality ?? "—";

        document.getElementById("time").textContent =
            Number(data.processing_time || 0).toFixed(2) + " sec";

        if (data.visualization) {
            const image =
                "data:image/jpeg;base64," + data.visualization;

            document.getElementById("resultA").src = image;
            document.getElementById("resultB").src = image;
        }

    } catch (error) {

        console.error("Analysis error:", error);

        alert(error.message);

        status.textContent = "ANALYSIS FAILED";
        score.textContent = "—";

    } finally {

        btn.disabled = false;
        btn.textContent = "⌕  COMPARE IMAGES";

    }
}

document
    .getElementById("compareBtn")
    .addEventListener("click", compareImages);
