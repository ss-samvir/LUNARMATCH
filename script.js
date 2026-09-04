(() => {
  "use strict";

  /*
   * ============================================================
   * LUNARMATCH V7 — ENHANCED LUNAR CORRESPONDENCE ENGINE
   * ============================================================
   *
   * Browser-side computer vision.
   *
   * Pipeline:
   *
   * ACQUIRE
   *     ↓
   * PREPROCESS
   *     ↓
   * MULTI-SCALE FEATURE EXTRACTION
   *     ↓
   * GRADIENT + PATCH DESCRIPTORS
   *     ↓
   * ADAPTIVE MUTUAL MATCHING
   *     ↓
   * RANSAC GEOMETRIC VERIFICATION
   *     ↓
   * SPATIAL CONSISTENCY
   *     ↓
   * EVIDENCE-BASED CONFIDENCE
   *     ↓
   * CORRESPONDENCE MAP
   *     ↓
   * PDF REPORT
   *
   * No OpenCV.js / WASM dependency.
   * Designed for desktop + mobile stability.
   */

  /* ============================================================
     CONFIGURATION
     ============================================================ */

  const MAX_IMAGE_DIMENSION = 1200;
  const WORK_MAX_DIMENSION = 720;

  const MAX_KEYPOINTS = 900;
  const MAX_MATCH_FEATURES = 260;

  const PATCH_RADIUS = 12;
  const DESCRIPTOR_STEP = 2;

  const BASE_RATIO = 0.90;
  const MIN_RATIO = 0.78;
  const MAX_RATIO = 0.94;

  const RANSAC_ITERATIONS = 420;
  const RANSAC_ERROR_PIXELS = 7.5;

  const MAX_VISUAL_MATCHES = 90;

  const GRID_SIZE = 5;

  /* ============================================================
     GLOBAL STATE
     ============================================================ */

  let imageAFile = null;
  let imageBFile = null;

  let imageAData = null;
  let imageBData = null;

  let lastAnalysis = null;

  /* ============================================================
     DOM HELPERS
     ============================================================ */

  const $ = id => document.getElementById(id);

  function setText(id, value) {
    const element = $(id);

    if (element) {
      element.textContent = value;
    }
  }

  function setDisabled(id, state) {
    const element = $(id);

    if (element) {
      element.disabled = state;
    }
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function round(value, digits = 2) {
    const factor = Math.pow(10, digits);
    return Math.round(value * factor) / factor;
  }

  function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function formatTime(ms) {
    return `${round(ms / 1000, 2)} sec`;
  }

  /* ============================================================
     PIPELINE UI
     ============================================================ */

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
      const element = $(id);

      if (!element) return;

      element.classList.remove(
        "active",
        "complete",
        "error"
      );
    });
  }

  function pipelineActive(id) {
    const element = $(id);

    if (!element) return;

    element.classList.remove(
      "complete",
      "error"
    );

    element.classList.add("active");
  }

  function pipelineComplete(id) {
    const element = $(id);

    if (!element) return;

    element.classList.remove(
      "active",
      "error"
    );

    element.classList.add("complete");
  }

  function pipelineError(id) {
    const element = $(id);

    if (!element) return;

    element.classList.remove(
      "active",
      "complete"
    );

    element.classList.add("error");
  }

  /* ============================================================
     RESULT RESET
     ============================================================ */

  function resetResults() {
    setText("status", "SYSTEM READY");

    setText("score", "--");
    setText("features", "--");
    setText("confidence", "--");
    setText("quality", "--");
    setText("time", "--");

    const fields = [
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

    fields.forEach(id => setText(id, "--"));

    const map = $("correspondenceMap");

    if (map) {
      map.removeAttribute("src");
      map.style.display = "none";
    }

    const placeholder = $("visualPlaceholder");

    if (placeholder) {
      placeholder.style.display = "";
    }

    setText(
      "interpretation",
      "Upload two lunar images and run the analysis to generate a correspondence assessment."
    );

    resetPipeline();

    lastAnalysis = null;

    setDisabled(
      "downloadReportBtn",
      true
    );
  }

  /* ============================================================
     FILE VALIDATION
     ============================================================ */

  function validateFile(file) {
    if (!file) {
      throw new Error("No image selected.");
    }

    const validMime =
      /^image\/(jpeg|png|webp|bmp|tiff?)$/i.test(
        file.type
      );

    const extension =
      /\.([a-z0-9]+)$/i.exec(file.name)?.[1]?.toLowerCase();

    const validExtension = [
      "jpg",
      "jpeg",
      "png",
      "webp",
      "tif",
      "tiff",
      "bmp"
    ].includes(extension);

    if (!validMime && !validExtension) {
      throw new Error(
        "Unsupported image format. Use JPG, PNG, WEBP, TIF or BMP."
      );
    }

    if (file.size > 25 * 1024 * 1024) {
      throw new Error(
        "Image is too large. Maximum size is 25 MB."
      );
    }
  }

  /* ============================================================
     FILE → IMAGE
     ============================================================ */

  function readDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        resolve(reader.result);
      };

      reader.onerror = () => {
        reject(
          new Error("Could not read the image.")
        );
      };

      reader.readAsDataURL(file);
    });
  }

  function decodeImage(dataURL) {
    return new Promise((resolve, reject) => {
      const image = new Image();

      image.onload = () => resolve(image);

      image.onerror = () => {
        reject(
          new Error("Could not decode the image.")
        );
      };

      image.src = dataURL;
    });
  }

  async function imageToGray(file) {
    validateFile(file);

    const dataURL =
      await readDataURL(file);

    const image =
      await decodeImage(dataURL);

    let width =
      image.naturalWidth ||
      image.width;

    let height =
      image.naturalHeight ||
      image.height;

    const scale =
      Math.min(
        1,
        MAX_IMAGE_DIMENSION /
        Math.max(width, height)
      );

    width =
      Math.max(
        96,
        Math.round(width * scale)
      );

    height =
      Math.max(
        96,
        Math.round(height * scale)
      );

    const canvas =
      document.createElement("canvas");

    canvas.width = width;
    canvas.height = height;

    const context =
      canvas.getContext(
        "2d",
        {
          willReadFrequently: true
        }
      );

    context.drawImage(
      image,
      0,
      0,
      width,
      height
    );

    const rgba =
      context.getImageData(
        0,
        0,
        width,
        height
      ).data;

    const gray =
      new Float32Array(
        width * height
      );

    for (
      let i = 0, p = 0;
      i < gray.length;
      i++, p += 4
    ) {
      gray[i] =
        0.299 * rgba[p] +
        0.587 * rgba[p + 1] +
        0.114 * rgba[p + 2];
    }

    return {
      width,
      height,
      gray,
      canvas,
      dataURL
    };
  }

  /* ============================================================
     RESIZE
     ============================================================ */

  function resizeGray(
    source,
    sourceWidth,
    sourceHeight,
    targetWidth,
    targetHeight
  ) {
    const result =
      new Float32Array(
        targetWidth *
        targetHeight
      );

    const scaleX =
      sourceWidth /
      targetWidth;

    const scaleY =
      sourceHeight /
      targetHeight;

    for (
      let y = 0;
      y < targetHeight;
      y++
    ) {
      const sourceY =
        Math.min(
          sourceHeight - 1,
          Math.floor(
            (y + 0.5) *
            scaleY
          )
        );

      for (
        let x = 0;
        x < targetWidth;
        x++
      ) {
        const sourceX =
          Math.min(
            sourceWidth - 1,
            Math.floor(
              (x + 0.5) *
              scaleX
            )
          );

        result[
          y * targetWidth + x
        ] =
          source[
            sourceY *
            sourceWidth +
            sourceX
          ];
      }
    }

    return result;
  }

  /* ============================================================
     PREPROCESSING
     ============================================================ */

  function localNormalize(image) {
    const scale =
      Math.min(
        1,
        WORK_MAX_DIMENSION /
        Math.max(
          image.width,
          image.height
        )
      );

    const width =
      Math.max(
        128,
        Math.round(
          image.width *
          scale
        )
      );

    const height =
      Math.max(
        128,
        Math.round(
          image.height *
          scale
        )
      );

    const source =
      resizeGray(
        image.gray,
        image.width,
        image.height,
        width,
        height
      );

    const normalized =
      new Float32Array(
        width * height
      );

    /*
     * Global mean/std first.
     * This prevents very dark or bright lunar images
     * from producing incompatible descriptors.
     */

    let sum = 0;

    for (
      let i = 0;
      i < source.length;
      i++
    ) {
      sum += source[i];
    }

    const mean =
      sum /
      source.length;

    let variance = 0;

    for (
      let i = 0;
      i < source.length;
      i++
    ) {
      const d =
        source[i] -
        mean;

      variance +=
        d * d;
    }

    variance /=
      source.length;

    const std =
      Math.sqrt(
        variance
      ) || 1;

    /*
     * Mild contrast normalization.
     */

    for (
      let i = 0;
      i < source.length;
      i++
    ) {
      normalized[i] =
        clamp(
          128 +
          (
            source[i] -
            mean
          ) /
          std *
          48,
          0,
          255
        );
    }

    /*
     * Small smoothing pass.
     */

    const smooth =
      new Float32Array(
        normalized.length
      );

    for (
      let y = 0;
      y < height;
      y++
    ) {
      for (
        let x = 0;
        x < width;
        x++
      ) {
        const i =
          y * width +
          x;

        if (
          x === 0 ||
          y === 0 ||
          x === width - 1 ||
          y === height - 1
        ) {
          smooth[i] =
            normalized[i];

          continue;
        }

        smooth[i] =
          (
            normalized[i] * 4 +
            normalized[i - 1] +
            normalized[i + 1] +
            normalized[i - width] +
            normalized[i + width]
          ) / 8;
      }
    }

    return {
      gray: smooth,
      width,
      height
    };
  }

  /* ============================================================
     IMAGE QUALITY
     ============================================================ */

  function percentileSample(array, q) {
    if (!array.length) {
      return 0;
    }

    /*
     * Sampling keeps mobile analysis responsive.
     */

    const maxSamples = 60000;
    const step =
      Math.max(
        1,
        Math.floor(
          array.length /
          maxSamples
        )
      );

    const values = [];

    for (
      let i = 0;
      i < array.length;
      i += step
    ) {
      values.push(
        array[i]
      );
    }

    values.sort(
      (a, b) => a - b
    );

    const position =
      (values.length - 1) *
      q;

    const lower =
      Math.floor(position);

    const upper =
      Math.ceil(position);

    if (
      lower === upper
    ) {
      return values[lower];
    }

    return (
      values[lower] +
      (
        values[upper] -
        values[lower]
      ) *
      (
        position -
        lower
      )
    );
  }

  function imageQuality(image) {
    const {
      gray,
      width,
      height
    } = image;

    let sum = 0;

    for (
      let i = 0;
      i < gray.length;
      i++
    ) {
      sum += gray[i];
    }

    const mean =
      sum /
      gray.length;

    let variance = 0;

    for (
      let i = 0;
      i < gray.length;
      i++
    ) {
      const d =
        gray[i] -
        mean;

      variance +=
        d * d;
    }

    variance /=
      gray.length;

    const contrast =
      Math.sqrt(
        variance
      );

    let lapSquared = 0;
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
          y * width +
          x;

        const lap =
          gray[i - width] +
          gray[i + width] +
          gray[i - 1] +
          gray[i + 1] -
          4 * gray[i];

        lapSquared +=
          lap * lap;

        count++;
      }
    }

    const sharpness =
      count
        ? lapSquared /
          count
        : 0;

    const p95 =
      percentileSample(
        gray,
        0.95
      );

    const p05 =
      percentileSample(
        gray,
        0.05
      );

    const dynamicRange =
      p95 - p05;

    const contrastScore =
      clamp(
        contrast /
        52 *
        100,
        0,
        100
      );

    const sharpnessScore =
      clamp(
        Math.log1p(
          sharpness
        ) /
        Math.log1p(
          900
        ) *
        100,
        0,
        100
      );

    const rangeScore =
      clamp(
        dynamicRange /
        190 *
        100,
        0,
        100
      );

    const exposurePenalty =
      Math.abs(
        mean - 128
      ) /
      128;

    const qualityScore =
      clamp(
        0.40 *
          contrastScore +

        0.40 *
          sharpnessScore +

        0.20 *
          rangeScore -

        8 *
          exposurePenalty,

        0,
        100
      );

    let label =
      "FAIR";

    if (
      qualityScore >= 82
    ) {
      label =
        "EXCELLENT";
    } else if (
      qualityScore >= 68
    ) {
      label =
        "GOOD";
    } else if (
      qualityScore < 45
    ) {
      label =
        "LIMITED";
    }

    return {
      resolution:
        `${width} × ${height}`,

      contrast:
        round(
          contrast,
          2
        ),

      sharpness:
        round(
          sharpness,
          2
        ),

      qualityScore:
        round(
          qualityScore,
          1
        ),

      qualityLabel:
        label
    };
  }

  /* ============================================================
     GRADIENT
     ============================================================ */

  function gradientAt(
    gray,
    width,
    height,
    x,
    y
  ) {
    const px =
      Math.round(x);

    const py =
      Math.round(y);

    if (
      px < 1 ||
      py < 1 ||
      px >= width - 1 ||
      py >= height - 1
    ) {
      return [0, 0];
    }

    const index =
      py * width +
      px;

    return [
      (
        gray[index + 1] -
        gray[index - 1]
      ) / 2,

      (
        gray[index + width] -
        gray[index - width]
      ) / 2
    ];
  }

  /* ============================================================
     FEATURE DETECTOR
     ============================================================ */

  function detectFeatures(image) {
    const {
      gray,
      width,
      height
    } = image;

    const candidates = [];

    /*
     * FAST-style circular neighbourhood.
     */

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

    const step =
      Math.max(
        2,
        Math.floor(
          Math.min(
            width,
            height
          ) / 320
        )
      );

    /*
     * Adaptive threshold.
     */

    let globalSum = 0;

    for (
      let i = 0;
      i < gray.length;
      i += 4
    ) {
      globalSum +=
        gray[i];
    }

    const globalMean =
      globalSum /
      Math.ceil(
        gray.length / 4
      );

    const threshold =
      globalMean < 90
        ? 13
        : globalMean > 175
          ? 22
          : 17;

    for (
      let y = 8;
      y < height - 8;
      y += step
    ) {
      for (
        let x = 8;
        x < width - 8;
        x += step
      ) {
        const center =
          gray[
            y * width +
            x
          ];

        let brighter = 0;
        let darker = 0;

        for (
          let k = 0;
          k < circle.length;
          k++
        ) {
          const value =
            gray[
              (
                y +
                circle[k][1]
              ) *
              width +
              (
                x +
                circle[k][0]
              )
            ];

          if (
            value >
            center +
            threshold
          ) {
            brighter++;
          }

          if (
            value <
            center -
            threshold
          ) {
            darker++;
          }
        }

        if (
          brighter < 6 &&
          darker < 6
        ) {
          continue;
        }

        /*
         * Harris-like local structure score.
         */

        let sxx = 0;
        let syy = 0;
        let sxy = 0;
        let gradientEnergy = 0;

        for (
          let yy = -2;
          yy <= 2;
          yy++
        ) {
          for (
            let xx = -2;
            xx <= 2;
            xx++
          ) {
            const [
              gx,
              gy
            ] =
              gradientAt(
                gray,
                width,
                height,
                x + xx,
                y + yy
              );

            sxx +=
              gx * gx;

            syy +=
              gy * gy;

            sxy +=
              gx * gy;

            gradientEnergy +=
              Math.abs(gx) +
              Math.abs(gy);
          }
        }

        const determinant =
          sxx * syy -
          sxy * sxy;

        const trace =
          sxx +
          syy +
          1e-6;

        const harris =
          determinant /
          trace;

        const circleContrast =
          Math.max(
            brighter,
            darker
          );

        const score =
          harris *
          (
            1 +
            circleContrast /
            16
          ) +
          gradientEnergy *
          0.25;

        if (
          score >
          28
        ) {
          candidates.push({
            x,
            y,
            score
          });
        }
      }
    }

    candidates.sort(
      (a, b) =>
        b.score -
        a.score
    );

    /*
     * Spatially distributed non-maximum suppression.
     */

    const selected = [];

    const minimumDistance =
      Math.max(
        10,
        Math.round(
          Math.min(
            width,
            height
          ) / 48
        )
      );

    for (
      const candidate of
      candidates
    ) {
      let accepted =
        true;

      for (
        const existing of
        selected
      ) {
        const distance =
          Math.hypot(
            candidate.x -
            existing.x,

            candidate.y -
            existing.y
          );

        if (
          distance <
          minimumDistance
        ) {
          accepted =
            false;
          break;
        }
      }

      if (!accepted) {
        continue;
      }

      selected.push(
        candidate
      );

      if (
        selected.length >=
        MAX_KEYPOINTS
      ) {
        break;
      }
    }

    return selected;
  }

  /* ============================================================
     ENHANCED DESCRIPTOR
     ============================================================ */

  function describeFeature(
    image,
    point
  ) {
    const {
      gray,
      width,
      height
    } = image;

    const radius =
      PATCH_RADIUS + 3;

    if (
      point.x < radius ||
      point.y < radius ||
      point.x >=
        width - radius ||
      point.y >=
        height - radius
    ) {
      return null;
    }

    /*
     * Dominant local orientation.
     */

    let directionX = 0;
    let directionY = 0;

    for (
      let y = -8;
      y <= 8;
      y += 2
    ) {
      for (
        let x = -8;
        x <= 8;
        x += 2
      ) {
        const [
          gx,
          gy
        ] =
          gradientAt(
            gray,
            width,
            height,
            point.x + x,
            point.y + y
          );

        const magnitude =
          Math.hypot(
            gx,
            gy
          );

        directionX +=
          gx * magnitude;

        directionY +=
          gy * magnitude;
      }
    }

    const angle =
      Math.atan2(
        directionY,
        directionX
      );

    const cos =
      Math.cos(angle);

    const sin =
      Math.sin(angle);

    const descriptor = [];

    /*
     * Rotated normalized intensity patch.
     */

    for (
      let y = -10;
      y <= 10;
      y += DESCRIPTOR_STEP
    ) {
      for (
        let x = -10;
        x <= 10;
        x += DESCRIPTOR_STEP
      ) {
        const rx =
          Math.round(
            point.x +
            x * cos -
            y * sin
          );

        const ry =
          Math.round(
            point.y +
            x * sin +
            y * cos
          );

        if (
          rx < 0 ||
          ry < 0 ||
          rx >= width ||
          ry >= height
        ) {
          return null;
        }

        descriptor.push(
          gray[
            ry * width +
            rx
          ] / 255
        );
      }
    }

    /*
     * Gradient orientation samples.
     */

    for (
      let y = -8;
      y <= 8;
      y += 4
    ) {
      for (
        let x = -8;
        x <= 8;
        x += 4
      ) {
        const rx =
          Math.round(
            point.x +
            x * cos -
            y * sin
          );

        const ry =
          Math.round(
            point.y +
            x * sin +
            y * cos
          );

        const [
          gx,
          gy
        ] =
          gradientAt(
            gray,
            width,
            height,
            rx,
            ry
          );

        const magnitude =
          Math.hypot(
            gx,
            gy
          );

        const orientation =
          Math.atan2(
            gy,
            gx
          ) -
          angle;

        descriptor.push(
          Math.cos(
            orientation
          ) *
          magnitude /
          255
        );

        descriptor.push(
          Math.sin(
            orientation
          ) *
          magnitude /
          255
        );
      }
    }

    /*
     * Zero mean + L2 normalization.
     */

    let mean = 0;

    for (
      const value of
      descriptor
    ) {
      mean += value;
    }

    mean /=
      descriptor.length;

    for (
      let i = 0;
      i < descriptor.length;
      i++
    ) {
      descriptor[i] -=
        mean;
    }

    let norm = 0;

    for (
      const value of
      descriptor
    ) {
      norm +=
        value * value;
    }

    norm =
      Math.sqrt(norm) ||
      1;

    for (
      let i = 0;
      i < descriptor.length;
      i++
    ) {
      descriptor[i] /=
        norm;
    }

    return {
      descriptor,
      angle
    };
  }

  function extractFeatures(image) {
    const points =
      detectFeatures(
        image
      );

    const features = [];

    for (
      const point of
      points
    ) {
      const descriptor =
        describeFeature(
          image,
          point
        );

      if (!descriptor) {
        continue;
      }

      features.push({
        x:
          point.x,

        y:
          point.y,

        score:
          point.score,

        descriptor:
          descriptor.descriptor,

        angle:
          descriptor.angle
      });
    }

    return features;
  }

  /* ============================================================
     DESCRIPTOR DISTANCE
     ============================================================ */

  function descriptorDistance(
    a,
    b
  ) {
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
      const d =
        a[i] -
        b[i];

      sum +=
        d * d;
    }

    return Math.sqrt(
      sum /
      length
    );
  }

  /* ============================================================
     ADAPTIVE FEATURE MATCHING
     ============================================================ */

  function matchFeatures(
    featuresA,
    featuresB
  ) {
    const A =
      featuresA
        .slice()
        .sort(
          (a, b) =>
            b.score -
            a.score
        )
        .slice(
          0,
          MAX_MATCH_FEATURES
        );

    const B =
      featuresB
        .slice()
        .sort(
          (a, b) =>
            b.score -
            a.score
        )
        .slice(
          0,
          MAX_MATCH_FEATURES
        );

    if (
      A.length < 4 ||
      B.length < 4
    ) {
      return [];
    }

    /*
     * Estimate descriptor noise.
     */

    const roughDistances = [];

    for (
      let i = 0;
      i < Math.min(
        A.length,
        80
      );
      i++
    ) {
      let best =
        Infinity;

      for (
        let j = 0;
        j < Math.min(
          B.length,
          80
        );
        j++
      ) {
        const d =
          descriptorDistance(
            A[i].descriptor,
            B[j].descriptor
          );

        if (
          d < best
        ) {
          best = d;
        }
      }

      if (
        Number.isFinite(best)
      ) {
        roughDistances.push(
          best
        );
      }
    }

    roughDistances.sort(
      (a, b) =>
        a - b
    );

    const medianNoise =
      roughDistances.length
        ? roughDistances[
            Math.floor(
              roughDistances.length /
              2
            )
          ]
        : 0.25;

    const adaptiveRatio =
      clamp(
        BASE_RATIO +
        (
          0.28 -
          medianNoise
        ) *
        0.18,
        MIN_RATIO,
        MAX_RATIO
      );

    const forward = [];

    for (
      let i = 0;
      i < A.length;
      i++
    ) {
      let best =
        Infinity;

      let second =
        Infinity;

      let bestIndex =
        -1;

      for (
        let j = 0;
        j < B.length;
        j++
      ) {
        const distance =
          descriptorDistance(
            A[i].descriptor,
            B[j].descriptor
          );

        if (
          distance <
          best
        ) {
          second =
            best;

          best =
            distance;

          bestIndex =
            j;

        } else if (
          distance <
          second
        ) {
          second =
            distance;
        }
      }

      if (
        bestIndex < 0 ||
        !Number.isFinite(
          second
        )
      ) {
        continue;
      }

      const ratio =
        second > 1e-8
          ? best /
            second
          : 1;

      /*
       * Both ratio and absolute descriptor distance
       * must be reasonable.
       */

      const absoluteLimit =
        Math.max(
          0.24,
          medianNoise *
          1.85
        );

      if (
        ratio <
          adaptiveRatio &&
        best <
          absoluteLimit
      ) {
        forward.push({
          ai: i,
          bi: bestIndex,
          ratio,
          distance: best
        });
      }
    }

    /*
     * Reverse nearest-neighbour map.
     */

    const reverse =
      new Map();

    for (
      let j = 0;
      j < B.length;
      j++
    ) {
      let best =
        Infinity;

      let bestIndex =
        -1;

      for (
        let i = 0;
        i < A.length;
        i++
      ) {
        const distance =
          descriptorDistance(
            B[j].descriptor,
            A[i].descriptor
          );

        if (
          distance <
          best
        ) {
          best =
            distance;

          bestIndex =
            i;
        }
      }

      if (
        bestIndex >= 0
      ) {
        reverse.set(
          j,
          bestIndex
        );
      }
    }

    /*
     * Mutual matching.
     */

    const mutual =
      forward.filter(
        match =>
          reverse.get(
            match.bi
          ) ===
          match.ai
      );

    /*
     * Sort strongest matches first.
     */

    mutual.sort(
      (a, b) =>
        a.ratio -
        b.ratio
    );

    return mutual.map(
      match => ({
        ai:
          featuresA.indexOf(
            A[match.ai]
          ),

        bi:
          featuresB.indexOf(
            B[match.bi]
          ),

        distance:
          match.distance,

        ratio:
          match.ratio
      })
    );
  }

  /* ============================================================
     AFFINE MODEL SOLVER
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
      [p1.x, p1.y, 1],
      [p2.x, p2.y, 1],
      [p3.x, p3.y, 1]
    ];

    const targetX = [
      q1.x,
      q2.x,
      q3.x
    ];

    const targetY = [
      q1.y,
      q2.y,
      q3.y
    ];

    function gaussian(
      source,
      target
    ) {
      const M =
        source.map(
          (row, index) =>
            [
              ...row,
              target[index]
            ]
        );

      for (
        let column = 0;
        column < 3;
        column++
      ) {
        let pivot =
          column;

        for (
          let row =
            column + 1;
          row < 3;
          row++
        ) {
          if (
            Math.abs(
              M[row][column]
            ) >
            Math.abs(
              M[pivot][column]
            )
          ) {
            pivot =
              row;
          }
        }

        if (
          Math.abs(
            M[pivot][column]
          ) <
          1e-8
        ) {
          return null;
        }

        [
          M[column],
          M[pivot]
        ] =
        [
          M[pivot],
          M[column]
        ];

        const divisor =
          M[column][column];

        for (
          let c =
            column;
          c < 4;
          c++
        ) {
          M[column][c] /=
            divisor;
        }

        for (
          let row = 0;
          row < 3;
          row++
        ) {
          if (
            row ===
            column
          ) {
            continue;
          }

          const factor =
            M[row][column];

          for (
            let c =
              column;
            c < 4;
            c++
          ) {
            M[row][c] -=
              factor *
              M[column][c];
          }
        }
      }

      return [
        M[0][3],
        M[1][3],
        M[2][3]
      ];
    }

    const X =
      gaussian(
        matrix,
        targetX
      );

    const Y =
      gaussian(
        matrix,
        targetY
      );

    if (
      !X ||
      !Y
    ) {
      return null;
    }

    return {
      a: X[0],
      b: X[1],
      c: X[2],

      d: Y[0],
      e: Y[1],
      f: Y[2]
    };
  }

  function transform(
    model,
    point
  ) {
    return {
      x:
        model.a *
          point.x +
        model.b *
          point.y +
        model.c,

      y:
        model.d *
          point.x +
        model.e *
          point.y +
        model.f
    };
  }

  function triangleArea(
    a,
    b,
    c
  ) {
    return Math.abs(
      (
        b.x -
        a.x
      ) *
      (
        c.y -
        a.y
      ) -

      (
        b.y -
        a.y
      ) *
      (
        c.x -
        a.x
      )
    );
  }

  /* ============================================================
     RANSAC GEOMETRIC VERIFICATION
     ============================================================ */

  function verifyGeometry(
    matches,
    featuresA,
    featuresB
  ) {
    if (
      matches.length <
      4
    ) {
      return {
        model: null,
        inliers: [],
        ratio: 0,
        consistency: 0,
        medianError: Infinity
      };
    }

    let bestModel =
      null;

    let bestInliers =
      [];

    let bestError =
      Infinity;

    for (
      let iteration = 0;
      iteration <
      RANSAC_ITERATIONS;
      iteration++
    ) {
      const i1 =
        Math.floor(
          Math.random() *
          matches.length
        );

      let i2 =
        Math.floor(
          Math.random() *
          matches.length
        );

      let i3 =
        Math.floor(
          Math.random() *
          matches.length
        );

      if (
        i2 === i1
      ) {
        i2 =
          (
            i2 + 1
          ) %
          matches.length;
      }

      if (
        i3 === i1 ||
        i3 === i2
      ) {
        i3 =
          (
            i3 + 2
          ) %
          matches.length;
      }

      const m1 =
        matches[i1];

      const m2 =
        matches[i2];

      const m3 =
        matches[i3];

      const p1 =
        featuresA[m1.ai];

      const p2 =
        featuresA[m2.ai];

      const p3 =
        featuresA[m3.ai];

      const q1 =
        featuresB[m1.bi];

      const q2 =
        featuresB[m2.bi];

      const q3 =
        featuresB[m3.bi];

      /*
       * Reject degenerate triangles.
       */

      if (
        triangleArea(
          p1,
          p2,
          p3
        ) <
        80
      ) {
        continue;
      }

      if (
        triangleArea(
          q1,
          q2,
          q3
        ) <
        80
      ) {
        continue;
      }

      const model =
        solveAffine(
          p1,
          p2,
          p3,
          q1,
          q2,
          q3
        );

      if (!model) {
        continue;
      }

      /*
       * Reject absurd transformations.
       */

      const determinant =
        model.a *
          model.e -
        model.b *
          model.d;

      if (
        Math.abs(
          determinant
        ) <
        0.15 ||
        Math.abs(
          determinant
        ) >
        6
      ) {
        continue;
      }

      const inliers =
        [];

      let totalError =
        0;

      for (
        const match of
        matches
      ) {
        const source =
          featuresA[
            match.ai
          ];

        const target =
          featuresB[
            match.bi
          ];

        const predicted =
          transform(
            model,
            source
          );

        const error =
          Math.hypot(
            predicted.x -
              target.x,

            predicted.y -
              target.y
          );

        /*
         * Slightly weighted threshold:
         * very strong descriptor matches get a little
         * more tolerance.
         */

        const threshold =
          match.ratio <
          0.78
            ? RANSAC_ERROR_PIXELS +
              1.5
            : RANSAC_ERROR_PIXELS;

        if (
          error <=
          threshold
        ) {
          inliers.push(
            match
          );

          totalError +=
            error;
        }
      }

      const meanError =
        inliers.length
          ? totalError /
            inliers.length
          : Infinity;

      /*
       * Primary objective = inlier count.
       * Secondary objective = geometric error.
       */

      if (
        inliers.length >
          bestInliers.length ||
        (
          inliers.length ===
            bestInliers.length &&
          meanError <
            bestError
        )
      ) {
        bestInliers =
          inliers;

        bestModel =
          model;

        bestError =
          meanError;
      }
    }

    if (
      !bestModel ||
      !bestInliers.length
    ) {
      return {
        model: null,
        inliers: [],
        ratio: 0,
        consistency: 0,
        medianError: Infinity
      };
    }

    const errors =
      [];

    for (
      const match of
      bestInliers
    ) {
      const source =
        featuresA[
          match.ai
        ];

      const target =
        featuresB[
          match.bi
        ];

      const predicted =
        transform(
          bestModel,
          source
        );

      errors.push(
        Math.hypot(
          predicted.x -
            target.x,

          predicted.y -
            target.y
        )
      );
    }

    errors.sort(
      (a, b) =>
        a - b
    );

    const medianError =
      errors[
        Math.floor(
          errors.length /
          2
        )
      ];

    const ratio =
      bestInliers.length /
      matches.length;

    const consistency =
      clamp(
        100 -
        medianError *
        10,
        0,
        100
      );

    return {
      model:
        bestModel,

      inliers:
        bestInliers,

      ratio,

      consistency,

      medianError
    };
  }

  /* ============================================================
     SPATIAL COVERAGE
     ============================================================ */

  function calculateCoverage(
    inliers,
    features,
    width,
    height
  ) {
    if (
      !inliers.length
    ) {
      return 0;
    }

    const occupied =
      new Set();

    for (
      const match of
      inliers
    ) {
      const point =
        features[
          match.ai
        ];

      const gx =
        clamp(
          Math.floor(
            point.x /
            width *
            GRID_SIZE
          ),
          0,
          GRID_SIZE - 1
        );

      const gy =
        clamp(
          Math.floor(
            point.y /
            height *
            GRID_SIZE
          ),
          0,
          GRID_SIZE - 1
        );

      occupied.add(
        `${gx}:${gy}`
      );
    }

    return (
      occupied.size /
      (
        GRID_SIZE *
        GRID_SIZE
      ) *
      100
    );
  }

  /* ============================================================
     MATCH QUALITY
     ============================================================ */

  function calculateMatchQuality(
    matches
  ) {
    if (
      !matches.length
    ) {
      return 0;
    }

    let sum = 0;

    for (
      const match of
      matches
    ) {
      /*
       * Lower ratio = stronger match.
       */

      sum +=
        clamp(
          (
            1 -
            match.ratio
          ) *
          100,
          0,
          100
        );
    }

    return (
      sum /
      matches.length
    );
  }

  /* ============================================================
     SCORE
     ============================================================ */

  function calculateScore(
    metricsA,
    metricsB,
    featuresA,
    featuresB,
    matches,
    verification,
    width,
    height
  ) {
    const verified =
      verification.inliers.length;

    const candidates =
      matches.length;

    const featureCount =
      Math.max(
        1,
        Math.min(
          featuresA.length,
          featuresB.length
        )
      );

    /*
     * Correspondence density.
     */

    const correspondenceDensity =
      clamp(
        verified /
        featureCount *
        260,
        0,
        100
      );

    /*
     * Geometric agreement.
     */

    const geometricScore =
      clamp(
        verification.ratio *
        180,
        0,
        100
      );

    /*
     * Spatial coverage.
     */

    const coverage =
      calculateCoverage(
        verification.inliers,
        featuresA,
        width,
        height
      );

    /*
     * Descriptor quality.
     */

    const matchQuality =
      calculateMatchQuality(
        verification.inliers
      );

    /*
     * Image quality.
     */

    const quality =
      (
        metricsA.qualityScore +
        metricsB.qualityScore
      ) /
      2;

    /*
     * Scale confidence:
     * good results should have at least several
     * distributed verified relationships.
     */

    const evidenceDepth =
      clamp(
        verified /
        30 *
        100,
        0,
        100
      );

    /*
     * Final evidence score.
     */

    let score =
      0.28 *
        correspondenceDensity +

      0.25 *
        geometricScore +

      0.17 *
        verification.consistency +

      0.12 *
        coverage +

      0.10 *
        matchQuality +

      0.05 *
        quality +

      0.03 *
        evidenceDepth;

    /*
     * Hard evidence safeguards.
     *
     * These prevent a handful of accidental matches
     * from generating an impressive confidence number.
     */

    if (
      verified < 4
    ) {
      score =
        Math.min(
          score,
          24
        );
    }

    if (
      verified < 6
    ) {
      score =
        Math.min(
          score,
          39
        );
    }

    if (
      verified < 10
    ) {
      score =
        Math.min(
          score,
          57
        );
    }

    if (
      verification.ratio <
      0.12
    ) {
      score =
        Math.min(
          score,
          48
        );
    }

    if (
      coverage < 12 &&
      verified < 15
    ) {
      score =
        Math.min(
          score,
          55
        );
    }

    score =
      clamp(
        score,
        0,
        100
      );

    let classification =
      "NO RELIABLE CORRESPONDENCE";

    if (
      verified >= 22 &&
      score >= 80 &&
      verification.ratio >= 0.36 &&
      coverage >= 28 &&
      verification.consistency >= 70
    ) {
      classification =
        "STRONG CORRESPONDENCE";

    } else if (
      verified >= 14 &&
      score >= 65 &&
      verification.ratio >= 0.28 &&
      coverage >= 20
    ) {
      classification =
        "CORRESPONDENCE FOUND";

    } else if (
      verified >= 7 &&
      score >= 48 &&
      verification.ratio >= 0.20 &&
      coverage >= 16
    ) {
      classification =
        "POSSIBLE CORRESPONDENCE";
    }

    return {
      score:
        round(
          score,
          1
        ),

      verifiedMatches:
        verified,

      candidateMatches:
        candidates,

      featureCoverage:
        round(
          coverage,
          1
        ),

      correspondenceStrength:
        round(
          correspondenceDensity,
          1
        ),

      inlierRatio:
        round(
          verification.ratio *
          100,
          1
        ),

      geometricConsistency:
        round(
          verification.consistency,
          1
        ),

      matchQuality:
        round(
          matchQuality,
          1
        ),

      classification,

      qualityScore:
        round(
          quality,
          1
        )
    };
  }

  /* ============================================================
     INTERPRETATION
     ============================================================ */

  function buildInterpretation(
    result
  ) {
    if (
      result.score >= 80
    ) {
      return (
        `Strong visual correspondence detected. ` +
        `${result.verifiedMatches} feature relationships ` +
        `survived geometric verification with ` +
        `${round(result.featureCoverage, 0)}% spatial coverage. ` +
        `The verified relationships show ` +
        `${round(result.geometricConsistency, 0)}% geometric ` +
        `consistency, providing strong evidence of correspondence.`
      );
    }

    if (
      result.score >= 65
    ) {
      return (
        `Meaningful lunar correspondence was detected. ` +
        `${result.verifiedMatches} feature relationships ` +
        `survived geometric filtering, with ` +
        `${round(result.featureCoverage, 0)}% spatial coverage ` +
        `and ${round(result.geometricConsistency, 0)}% geometric ` +
        `consistency. The result provides supportive evidence ` +
        `but should be reviewed using the correspondence map.`
      );
    }

    if (
      result.score >= 48
    ) {
      return (
        `Potential local correspondence was detected, but the ` +
        `evidence remains limited. ${result.verifiedMatches} ` +
        `relationships survived geometric verification. ` +
        `Differences in illumination, scale, viewpoint, focus, ` +
        `or lunar surface appearance may be affecting the result.`
      );
    }

    return (
      `No reliable global correspondence was established. ` +
      `The detected local similarities did not provide enough ` +
      `consistent geometric evidence. Try higher-resolution, ` +
      `well-focused images showing overlapping lunar terrain.`
    );
  }

  /* ============================================================
     VISUALIZATION
     ============================================================ */

  function createCorrespondenceMap(
    featuresA,
    featuresB,
    inliers,
    sourceA,
    sourceB,
    workA,
    workB
  ) {
    const canvas =
      document.createElement(
        "canvas"
      );

    const gap = 12;

    const maximumWidth =
      1200;

    const scale =
      Math.min(
        1,
        maximumWidth /
        (
          sourceA.width +
          sourceB.width +
          gap
        )
      );

    const widthA =
      Math.round(
        sourceA.width *
        scale
      );

    const heightA =
      Math.round(
        sourceA.height *
        scale
      );

    const widthB =
      Math.round(
        sourceB.width *
        scale
      );

    const heightB =
      Math.round(
        sourceB.height *
        scale
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

    const context =
      canvas.getContext(
        "2d"
      );

    context.fillStyle =
      "#080b12";

    context.fillRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    context.drawImage(
      sourceA.canvas,
      0,
      0,
      widthA,
      heightA
    );

    context.drawImage(
      sourceB.canvas,
      widthA +
        gap,
      0,
      widthB,
      heightB
    );

    const visualMatches =
      inliers
        .slice()
        .sort(
          (a, b) =>
            a.distance -
            b.distance
        )
        .slice(
          0,
          MAX_VISUAL_MATCHES
        );

    /*
     * Feature coordinates belong to work images.
     * Scale them properly to source images.
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

    context.lineWidth =
      1.2;

    visualMatches.forEach(
      (match, index) => {
        const pointA =
          featuresA[
            match.ai
          ];

        const pointB =
          featuresB[
            match.bi
          ];

        const x1 =
          pointA.x *
          scaleAX *
          scale;

        const y1 =
          pointA.y *
          scaleAY *
          scale;

        const x2 =
          widthA +
          gap +
          pointB.x *
          scaleBX *
          scale;

        const y2 =
          pointB.y *
          scaleBY *
          scale;

        context.strokeStyle =
          `hsl(${150 + index % 30}, 85%, 64%)`;

        context.beginPath();

        context.moveTo(
          x1,
          y1
        );

        context.lineTo(
          x2,
          y2
        );

        context.stroke();

        context.fillStyle =
          "#ffffff";

        context.beginPath();

        context.arc(
          x1,
          y1,
          2.5,
          0,
          Math.PI * 2
        );

        context.fill();

        context.beginPath();

        context.arc(
          x2,
          y2,
          2.5,
          0,
          Math.PI * 2
        );

        context.fill();
      }
    );

    context.fillStyle =
      "rgba(0,0,0,.65)";

    context.fillRect(
      0,
      0,
      140,
      28
    );

    context.fillRect(
      widthA +
        gap,
      0,
      140,
      28
    );

    context.fillStyle =
      "#ffffff";

    context.font =
      "600 12px Inter, Arial, sans-serif";

    context.fillText(
      "IMAGE A",
      12,
      19
    );

    context.fillText(
      "IMAGE B",
      widthA +
        gap +
        12,
      19
    );

    return canvas.toDataURL(
      "image/jpeg",
      0.88
    );
  }

  /* ============================================================
     DISPLAY RESULTS
     ============================================================ */

  function displayResults(
    result
  ) {
    setText(
      "status",
      result.classification
    );

    setText(
      "score",
      `${result.score}%`
    );

    setText(
      "features",
      result.verifiedMatches
    );

    setText(
      "confidence",
      result.score >= 80
        ? "HIGH"
        : result.score >= 65
          ? "MODERATE"
          : result.score >= 48
            ? "LIMITED"
            : "LOW"
    );

    setText(
      "quality",
      result.qualityScore >= 80
        ? "EXCELLENT"
        : result.qualityScore >= 65
          ? "GOOD"
          : "FAIR"
    );

    setText(
      "time",
      formatTime(
        result.processingMs
      )
    );

    setText(
      "resolutionA",
      result.metricsA.resolution
    );

    setText(
      "keypointsA",
      result.featuresACount
    );

    setText(
      "contrastA",
      result.metricsA.contrast
    );

    setText(
      "sharpnessA",
      result.metricsA.sharpness
    );

    setText(
      "qualityScoreA",
      `${result.metricsA.qualityScore}/100`
    );

    setText(
      "resolutionB",
      result.metricsB.resolution
    );

    setText(
      "keypointsB",
      result.featuresBCount
    );

    setText(
      "contrastB",
      result.metricsB.contrast
    );

    setText(
      "sharpnessB",
      result.metricsB.sharpness
    );

    setText(
      "qualityScoreB",
      `${result.metricsB.qualityScore}/100`
    );

    setText(
      "rawMatches",
      result.rawFeatureComparisons
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
      `${result.featureCoverage}%`
    );

    setText(
      "correspondenceStrength",
      `${result.correspondenceStrength}%`
    );

    setText(
      "inlierRatio",
      `${result.inlierRatio}%`
    );

    setText(
      "geometricConsistency",
      `${result.geometricConsistency}%`
    );

    setText(
      "homographyStatus",
      result.homographyStatus
    );

    setText(
      "verificationStatus",
      result.verificationStatus
    );

    const map =
      $("correspondenceMap");

    if (
      map &&
      result.visualization
    ) {
      map.src =
        result.visualization;

      map.style.display =
        "block";
    }

    const placeholder =
      $("visualPlaceholder");

    if (placeholder) {
      placeholder.style.display =
        "none";
    }

    setText(
      "interpretation",
      buildInterpretation(
        result
      )
    );

    setDisabled(
      "downloadReportBtn",
      false
    );
  }

  /* ============================================================
     MAIN ANALYSIS ENGINE
     ============================================================ */

  async function analyzeImages() {
    if (
      !imageAFile ||
      !imageBFile
    ) {
      setText(
        "status",
        "SELECT BOTH IMAGES"
      );

      return;
    }

    setDisabled(
      "compareBtn",
      true
    );

    const start =
      performance.now();

    try {
      resetPipeline();

      /* --------------------------------------------------------
         ACQUIRE
         -------------------------------------------------------- */

      pipelineActive(
        "stageAcquire"
      );

      setText(
        "status",
        "ACQUIRING"
      );

      await delay(30);

      const [
        sourceA,
        sourceB
      ] =
        await Promise.all([
          imageToGray(
            imageAFile
          ),

          imageToGray(
            imageBFile
          )
        ]);

      pipelineComplete(
        "stageAcquire"
      );

      /* --------------------------------------------------------
         PREPROCESS
         -------------------------------------------------------- */

      pipelineActive(
        "stagePreprocess"
      );

      setText(
        "status",
        "PREPROCESSING"
      );

      await delay(30);

      const workA =
        localNormalize(
          sourceA
        );

      const workB =
        localNormalize(
          sourceB
        );

      pipelineComplete(
        "stagePreprocess"
      );

      /* --------------------------------------------------------
         FEATURE EXTRACTION
         -------------------------------------------------------- */

      pipelineActive(
        "stageExtract"
      );

      setText(
        "status",
        "EXTRACTING LUNAR FEATURES"
      );

      await delay(40);

      const featuresA =
        extractFeatures(
          workA
        );

      await delay(20);

      const featuresB =
        extractFeatures(
          workB
        );

      pipelineComplete(
        "stageExtract"
      );

      /* --------------------------------------------------------
         MATCHING
         -------------------------------------------------------- */

      pipelineActive(
        "stageMatch"
      );

      setText(
        "status",
        "MATCHING CORRESPONDENCES"
      );

      await delay(40);

      const matches =
        matchFeatures(
          featuresA,
          featuresB
        );

      pipelineComplete(
        "stageMatch"
      );

      /* --------------------------------------------------------
         GEOMETRIC VERIFICATION
         -------------------------------------------------------- */

      pipelineActive(
        "stageVerify"
      );

      setText(
        "status",
        "VERIFYING GEOMETRY"
      );

      await delay(40);

      const verification =
        verifyGeometry(
          matches,
          featuresA,
          featuresB
        );

      pipelineComplete(
        "stageVerify"
      );

      /* --------------------------------------------------------
         SCORE
         -------------------------------------------------------- */

      pipelineActive(
        "stageScore"
      );

      setText(
        "status",
        "CALCULATING EVIDENCE"
      );

      await delay(40);

      const metricsA =
        imageQuality(
          sourceA
        );

      const metricsB =
        imageQuality(
          sourceB
        );

      const scored =
        calculateScore(
          metricsA,
          metricsB,
          featuresA,
          featuresB,
          matches,
          verification,
          workA.width,
          workA.height
        );

      pipelineComplete(
        "stageScore"
      );

      /* --------------------------------------------------------
         REPORT
         -------------------------------------------------------- */

      pipelineActive(
        "stageReport"
      );

      setText(
        "status",
        "BUILDING CORRESPONDENCE MAP"
      );

      await delay(30);

      const visualization =
        createCorrespondenceMap(
          featuresA,
          featuresB,
          verification.inliers,
          sourceA,
          sourceB,
          workA,
          workB
        );

      const processingMs =
        performance.now() -
        start;

      const result = {
        ...scored,

        processingMs,

        metricsA,
        metricsB,

        featuresACount:
          featuresA.length,

        featuresBCount:
          featuresB.length,

        rawFeatureComparisons:
          Math.min(
            featuresA.length,
            MAX_MATCH_FEATURES
          ) *
          Math.min(
            featuresB.length,
            MAX_MATCH_FEATURES
          ),

        homographyStatus:
          verification.model
            ? "AFFINE MODEL VERIFIED"
            : "NOT ESTABLISHED",

        verificationStatus:
          verification.inliers.length >= 7
            ? "RANSAC CONSISTENT"
            : "INSUFFICIENT INLIERS",

        visualization,

        generatedAt:
          new Date().toLocaleString()
      };

      lastAnalysis =
        result;

      displayResults(
        result
      );

      pipelineComplete(
        "stageReport"
      );

      setText(
        "status",
        result.classification
      );

      console.log(
        "LUNARMATCH V7 ANALYSIS:",
        {
          featuresA:
            featuresA.length,

          featuresB:
            featuresB.length,

          candidates:
            matches.length,

          verified:
            verification.inliers.length,

          coverage:
            result.featureCoverage,

          inlierRatio:
            result.inlierRatio,

          consistency:
            result.geometricConsistency,

          score:
            result.score
        }
      );

    } catch (error) {
      console.error(
        "LUNARMATCH ENGINE ERROR:",
        error
      );

      setText(
        "status",
        "ANALYSIS ERROR"
      );

      setText(
        "interpretation",
        error.message ||
        "The analysis could not be completed."
      );

      PIPELINE.forEach(
        id => {
          const element =
            $(id);

          if (
            element &&
            element.classList.contains(
              "active"
            )
          ) {
            pipelineError(id);
          }
        }
      );

    } finally {
      setDisabled(
        "compareBtn",
        false
      );
    }
  }

  /* ============================================================
     PDF ENGINE
     ============================================================ */

  function loadPDFEngine() {
    if (
      window.jspdf &&
      window.jspdf.jsPDF
    ) {
      return Promise.resolve(
        window.jspdf.jsPDF
      );
    }

    return new Promise(
      (resolve, reject) => {
        const existing =
          document.querySelector(
            'script[data-lunarmatch-jspdf="1"]'
          );

        if (existing) {
          if (
            window.jspdf &&
            window.jspdf.jsPDF
          ) {
            resolve(
              window.jspdf.jsPDF
            );

            return;
          }

          existing.addEventListener(
            "load",
            () => {
              if (
                window.jspdf &&
                window.jspdf.jsPDF
              ) {
                resolve(
                  window.jspdf.jsPDF
                );
              } else {
                reject(
                  new Error(
                    "PDF engine loaded but jsPDF is unavailable."
                  )
                );
              }
            },
            {
              once: true
            }
          );

          existing.addEventListener(
            "error",
            () =>
              reject(
                new Error(
                  "PDF engine could not be loaded."
                )
              ),
            {
              once: true
            }
          );

          return;
        }

        const script =
          document.createElement(
            "script"
          );

        script.src =
          "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";

        script.async = true;

        script.dataset.lunarmatchJspdf =
          "1";

        script.onload =
          () => {
            if (
              window.jspdf &&
              window.jspdf.jsPDF
            ) {
              resolve(
                window.jspdf.jsPDF
              );
            } else {
              reject(
                new Error(
                  "PDF engine loaded but jsPDF is unavailable."
                )
              );
            }
          };

        script.onerror =
          () => {
            reject(
              new Error(
                "PDF engine could not be loaded."
              )
            );
          };

        document.head.appendChild(
          script
        );
      }
    );
  }

  /* ============================================================
     PDF REPORT
     ============================================================ */

  async function downloadReport() {
    if (!lastAnalysis) {
      setText(
        "status",
        "RUN ANALYSIS FIRST"
      );

      return;
    }

    setDisabled(
      "downloadReportBtn",
      true
    );

    try {
      const jsPDF =
        await loadPDFEngine();

      const documentPDF =
        new jsPDF({
          unit: "mm",
          format: "a4"
        });

      const margin =
        16;

      let y =
        18;

      documentPDF.setFont(
        "helvetica",
        "bold"
      );

      documentPDF.setFontSize(
        20
      );

      documentPDF.text(
        "LUNARMATCH",
        margin,
        y
      );

      y += 7;

      documentPDF.setFont(
        "helvetica",
        "normal"
      );

      documentPDF.setFontSize(
        9
      );

      documentPDF.text(
        "Lunar Image Correspondence Analysis Report",
        margin,
        y
      );

      y += 10;

      documentPDF.setFontSize(
        10
      );

      const statistics = [
        `Generated: ${lastAnalysis.generatedAt}`,

        `Result: ${lastAnalysis.classification}`,

        `Correspondence Score: ${lastAnalysis.score}%`,

        `Confidence: ${
          lastAnalysis.score >= 80
            ? "HIGH"
            : lastAnalysis.score >= 65
              ? "MODERATE"
              : "LIMITED"
        }`,

        `Verified Features: ${lastAnalysis.verifiedMatches}`,

        `Candidate Matches: ${lastAnalysis.candidateMatches}`,

        `Inlier Ratio: ${lastAnalysis.inlierRatio}%`,

        `Spatial Coverage: ${lastAnalysis.featureCoverage}%`,

        `Geometric Consistency: ${lastAnalysis.geometricConsistency}%`,

        `Image A Quality: ${lastAnalysis.metricsA.qualityScore}/100`,

        `Image B Quality: ${lastAnalysis.metricsB.qualityScore}/100`,

        `Processing Time: ${formatTime(lastAnalysis.processingMs)}`
      ];

      for (
        const line of
        statistics
      ) {
        documentPDF.text(
          line,
          margin,
          y
        );

        y += 6;
      }

      y += 4;

      documentPDF.setFont(
        "helvetica",
        "bold"
      );

      documentPDF.text(
        "Interpretation",
        margin,
        y
      );

      y += 6;

      documentPDF.setFont(
        "helvetica",
        "normal"
      );

      const interpretation =
        buildInterpretation(
          lastAnalysis
        );

      const wrapped =
        documentPDF.splitTextToSize(
          interpretation,
          178
        );

      documentPDF.text(
        wrapped,
        margin,
        y
      );

      y +=
        wrapped.length *
        5 +
        7;

      if (
        lastAnalysis.visualization
      ) {
        documentPDF.setFont(
          "helvetica",
          "bold"
        );

        documentPDF.text(
          "Correspondence Visualization",
          margin,
          y
        );

        y += 5;

        const imageWidth =
          178;

        const imageHeight =
          imageWidth *
          0.55;

        if (
          y +
            imageHeight >
          282
        ) {
          documentPDF.addPage();

          y = 18;
        }

        documentPDF.addImage(
          lastAnalysis.visualization,
          "JPEG",
          margin,
          y,
          imageWidth,
          imageHeight
        );
      }

      documentPDF.setFontSize(
        7
      );

      documentPDF.setTextColor(
        100
      );

      documentPDF.text(
        "LUNARMATCH browser correspondence engine — computational estimate for research/prototype use.",
        margin,
        288
      );

      documentPDF.save(
        "LUNARMATCH_Analysis_Report.pdf"
      );

      setText(
        "status",
        "REPORT READY"
      );

    } catch (error) {
      console.error(
        "PDF ERROR:",
        error
      );

      setText(
        "status",
        "PDF ERROR"
      );

      setText(
        "interpretation",
        error.message ||
        "The PDF report could not be generated."
      );

    } finally {
      setDisabled(
        "downloadReportBtn",
        false
      );
    }
  }

  /* ============================================================
     PREVIEW
     ============================================================ */

  function updatePreview(
    previewId,
    dataURL
  ) {
    const image =
      $(previewId);

    if (!image) {
      console.warn(
        `LUNARMATCH: Preview element ${previewId} not found.`
      );

      return;
    }

    image.src =
      dataURL;

    image.alt =
      "Selected lunar image preview";

    image.style.display =
      "block";

    image.removeAttribute(
      "hidden"
    );

    console.log(
      `LUNARMATCH: Preview updated for ${previewId}`
    );
  }

  /* ============================================================
     IMAGE HANDLING
     ============================================================ */

  async function handleImage(
    file,
    slot,
    previewId
  ) {
    try {
      validateFile(
        file
      );

      setText(
        "status",
        `LOADING IMAGE ${slot}`
      );

      const dataURL =
        await readDataURL(
          file
        );

      if (
        slot === "A"
      ) {
        imageAFile =
          file;

        imageAData =
          dataURL;
      } else {
        imageBFile =
          file;

        imageBData =
          dataURL;
      }

      updatePreview(
        previewId,
        dataURL
      );

      /*
       * New image invalidates previous analysis.
       */

      lastAnalysis =
        null;

      setDisabled(
        "downloadReportBtn",
        true
      );

      if (
        imageAFile &&
        imageBFile
      ) {
        setText(
          "status",
          "READY TO ANALYZE"
        );
      } else {
        setText(
          "status",
          `IMAGE ${slot} LOADED`
        );
      }

      console.log(
        `LUNARMATCH: Image ${slot} loaded successfully:`,
        file.name
      );

    } catch (error) {
      console.error(
        `LUNARMATCH: Image ${slot} error:`,
        error
      );

      setText(
        "status",
        "INVALID IMAGE"
      );

      setText(
        "interpretation",
        error.message
      );
    }
  }

  /* ============================================================
     ROBUST FILE INPUT SETUP
     ============================================================ */

  function setupImageInput(
    inputId,
    previewId,
    slot
  ) {
    const input =
      $(inputId);

    if (!input) {
      console.error(
        `LUNARMATCH: ${inputId} not found.`
      );

      return;
    }

    input.type =
      "file";

    input.accept =
      "image/jpeg,image/png,image/webp,image/bmp,image/tiff,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff";

    input.addEventListener(
      "change",
      async event => {
        const files =
          event.target.files;

        if (
          !files ||
          !files.length
        ) {
          console.log(
            `LUNARMATCH: No Image ${slot} selected.`
          );

          return;
        }

        const file =
          files[0];

        console.log(
          `LUNARMATCH: Image ${slot} selected:`,
          file.name,
          file.type,
          file.size
        );

        await handleImage(
          file,
          slot,
          previewId
        );
      }
    );

    console.log(
      `LUNARMATCH: File input ${inputId} initialized.`
    );
  }

  /* ============================================================
     DRAG & DROP + CLICKABLE UPLOAD ZONE
     ============================================================ */

  function setupDropZone(
    zone,
    input,
    previewId,
    slot
  ) {
    if (
      !zone ||
      !input
    ) {
      console.warn(
        `LUNARMATCH: Drop zone or input missing for Image ${slot}.`
      );

      return;
    }

    zone.addEventListener(
      "click",
      event => {
        if (
          event.target ===
          input
        ) {
          return;
        }

        console.log(
          `LUNARMATCH: Opening Image ${slot} file picker.`
        );

        input.click();
      }
    );

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
              "dragging"
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
              "dragging"
            );
          }
        );
      }
    );

    zone.addEventListener(
      "drop",
      async event => {
        const files =
          event.dataTransfer?.files;

        if (
          !files ||
          !files.length
        ) {
          return;
        }

        const file =
          files[0];

        console.log(
          `LUNARMATCH: Image ${slot} dropped:`,
          file.name,
          file.type,
          file.size
        );

        await handleImage(
          file,
          slot,
          previewId
        );
      }
    );

    console.log(
      `LUNARMATCH: Upload zone for Image ${slot} initialized.`
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
      .forEach(
        link => {
          link.addEventListener(
            "click",
            event => {
              const targetID =
                link.getAttribute(
                  "href"
                );

              const target =
                targetID
                  ? $(
                      targetID.slice(1)
                    )
                  : null;

              if (!target) {
                return;
              }

              event.preventDefault();

              target.scrollIntoView({
                behavior:
                  "smooth",

                block:
                  "start"
              });
            }
          );
        }
      );
  }

  /* ============================================================
     BUTTON EFFECTS
     ============================================================ */

  function setupButtonEffects() {
    document
      .querySelectorAll(
        "button"
      )
      .forEach(
        button => {
          button.addEventListener(
            "click",
            () => {
              button.classList.remove(
                "button-pulse"
              );

              void button.offsetWidth;

              button.classList.add(
                "button-pulse"
              );
            }
          );
        }
      );
  }

  /* ============================================================
     CONTACT / REGISTRATION
     ============================================================ */

  function setupPlaceholderActions() {
    const registration =
      $("registrationBtn");

    if (registration) {
      registration.addEventListener(
        "click",
        () => {
          setText(
            "status",
            "REGISTRATION MODULE"
          );

          setText(
            "interpretation",
            "Account registration is reserved for the next LUNARMATCH platform module."
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
          const target =
            $("contact");

          if (target) {
            target.scrollIntoView({
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

  document.addEventListener(
    "DOMContentLoaded",
    () => {
      console.log(
        "LUNARMATCH: DOM loaded."
      );

      setupImageInput(
        "fileA",
        "previewA",
        "A"
      );

      setupImageInput(
        "fileB",
        "previewB",
        "B"
      );

      const inputA =
        $("fileA");

      const inputB =
        $("fileB");

      const dropZones =
        document.querySelectorAll(
          ".drop-new"
        );

      console.log(
        "LUNARMATCH upload zones:",
        dropZones.length
      );

      setupDropZone(
        dropZones[0],
        inputA,
        "previewA",
        "A"
      );

      setupDropZone(
        dropZones[1],
        inputB,
        "previewB",
        "B"
      );

      const compare =
        $("compareBtn");

      if (compare) {
        compare.addEventListener(
          "click",
          analyzeImages
        );
      } else {
        console.warn(
          "LUNARMATCH: compareBtn not found."
        );
      }

      const report =
        $("downloadReportBtn");

      if (report) {
        report.addEventListener(
          "click",
          downloadReport
        );
      } else {
        console.warn(
          "LUNARMATCH: downloadReportBtn not found."
        );
      }

      setupNavigation();

      setupButtonEffects();

      setupPlaceholderActions();

      resetResults();

      console.log(
        "LUNARMATCH V7 — Enhanced Lunar Correspondence Engine ONLINE"
      );

      console.log(
        "Engine: Adaptive local normalization + enhanced features + gradient descriptors + mutual matching + RANSAC"
      );

      console.log(
        "Upload system: ACTIVE"
      );

      console.log(
        "Image A input:",
        !!inputA
      );

      console.log(
        "Image B input:",
        !!inputB
      );
    }
  );

})();
