(() => {
  "use strict";

  /*
   * ============================================================
   * LUNARMATCH V6 — LUNAR CORRESPONDENCE ENGINE
   * ============================================================
   *
   * Browser-side computer vision engine.
   *
   * Pipeline:
   *
   * ACQUIRE
   *    ↓
   * PREPROCESS
   *    ↓
   * FEATURE EXTRACTION
   *    ↓
   * DESCRIPTOR MATCHING
   *    ↓
   * RANSAC GEOMETRIC VERIFICATION
   *    ↓
   * EVIDENCE-BASED SCORE
   *    ↓
   * CORRESPONDENCE MAP
   *    ↓
   * PDF REPORT
   *
   * No OpenCV.js / WASM dependency.
   * Designed for desktop + mobile stability.
   */

  /* ============================================================
     CONFIGURATION
     ============================================================ */

  const MAX_IMAGE_DIMENSION = 1000;
  const WORK_MAX_DIMENSION = 640;

  const MAX_KEYPOINTS = 700;
  const MAX_MATCH_FEATURES = 180;

  const PATCH_RADIUS = 10;
  const DESCRIPTOR_STEP = 2;

  const LOWE_RATIO = 0.86;

  const RANSAC_ITERATIONS = 260;
  const RANSAC_ERROR_PIXELS = 6;

  const MAX_VISUAL_MATCHES = 75;


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
        64,
        Math.round(width * scale)
      );

    height =
      Math.max(
        64,
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
        targetWidth * targetHeight
      );

    const scaleX =
      sourceWidth / targetWidth;

    const scaleY =
      sourceHeight / targetHeight;

    for (
      let y = 0;
      y < targetHeight;
      y++
    ) {

      const sourceY =
        Math.min(
          sourceHeight - 1,
          Math.floor(
            (y + 0.5) * scaleY
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
              (x + 0.5) * scaleX
            )
          );

        result[
          y * targetWidth + x
        ] =
          source[
            sourceY * sourceWidth +
            sourceX
          ];
      }
    }

    return result;
  }


  /* ============================================================
     FAST LOCAL NORMALIZATION
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
        96,
        Math.round(
          image.width * scale
        )
      );

    const height =
      Math.max(
        96,
        Math.round(
          image.height * scale
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

    const integral =
      new Float64Array(
        (width + 1) *
        (height + 1)
      );

    const integralSq =
      new Float64Array(
        (width + 1) *
        (height + 1)
      );

    for (
      let y = 1;
      y <= height;
      y++
    ) {

      let rowSum = 0;
      let rowSq = 0;

      for (
        let x = 1;
        x <= width;
        x++
      ) {

        const value =
          source[
            (y - 1) * width +
            (x - 1)
          ];

        rowSum += value;
        rowSq += value * value;

        const index =
          y * (width + 1) + x;

        integral[index] =
          integral[
            (y - 1) * (width + 1) + x
          ] + rowSum;

        integralSq[index] =
          integralSq[
            (y - 1) * (width + 1) + x
          ] + rowSq;
      }
    }

    const normalized =
      new Float32Array(
        width * height
      );

    const radius = 4;

    function areaSum(
      table,
      x0,
      y0,
      x1,
      y1
    ) {

      const stride =
        width + 1;

      return (
        table[y1 * stride + x1] -
        table[y0 * stride + x1] -
        table[y1 * stride + x0] +
        table[y0 * stride + x0]
      );
    }

    for (
      let y = 0;
      y < height;
      y++
    ) {

      const y0 =
        Math.max(0, y - radius);

      const y1 =
        Math.min(
          height - 1,
          y + radius
        );

      for (
        let x = 0;
        x < width;
        x++
      ) {

        const x0 =
          Math.max(0, x - radius);

        const x1 =
          Math.min(
            width - 1,
            x + radius
          );

        const ax = x0;
        const ay = y0;
        const bx = x1 + 1;
        const by = y1 + 1;

        const count =
          (bx - ax) *
          (by - ay);

        const sum =
          areaSum(
            integral,
            ax,
            ay,
            bx,
            by
          );

        const sq =
          areaSum(
            integralSq,
            ax,
            ay,
            bx,
            by
          );

        const mean =
          sum / count;

        const variance =
          Math.max(
            25,
            sq / count -
            mean * mean
          );

        const value =
          source[
            y * width + x
          ];

        const z =
          (value - mean) /
          Math.sqrt(variance);

        normalized[
          y * width + x
        ] =
          clamp(
            128 + z * 40,
            0,
            255
          );
      }
    }

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
          y * width + x;

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

  function percentile(array, q) {

    const values =
      Array.from(array)
        .sort(
          (a, b) => a - b
        );

    if (!values.length) {
      return 0;
    }

    const position =
      (values.length - 1) * q;

    const lower =
      Math.floor(position);

    const upper =
      Math.ceil(position);

    if (lower === upper) {
      return values[lower];
    }

    return (
      values[lower] +
      (
        values[upper] -
        values[lower]
      ) *
      (
        position - lower
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
      sum / gray.length;

    let variance = 0;

    for (
      let i = 0;
      i < gray.length;
      i++
    ) {

      const difference =
        gray[i] - mean;

      variance +=
        difference *
        difference;
    }

    variance /=
      gray.length;

    const contrast =
      Math.sqrt(variance);

    let lapSum = 0;
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
          y * width + x;

        const lap =
          gray[i - width] +
          gray[i + width] +
          gray[i - 1] +
          gray[i + 1] -
          4 * gray[i];

        lapSum += lap;
        lapSquared +=
          lap * lap;

        count++;
      }
    }

    const lapMean =
      count
        ? lapSum / count
        : 0;

    const sharpness =
      count
        ? Math.max(
            0,
            lapSquared / count -
            lapMean * lapMean
          )
        : 0;

    const p95 =
      percentile(gray, 0.95);

    const p05 =
      percentile(gray, 0.05);

    const dynamicRange =
      p95 - p05;

    const contrastScore =
      clamp(
        contrast / 52 * 100,
        0,
        100
      );

    const sharpnessScore =
      clamp(
        Math.log1p(sharpness) /
        Math.log1p(900) *
        100,
        0,
        100
      );

    const rangeScore =
      clamp(
        dynamicRange / 190 * 100,
        0,
        100
      );

    const exposurePenalty =
      Math.abs(
        mean - 128
      ) / 128;

    const qualityScore =
      clamp(
        0.42 * contrastScore +
        0.38 * sharpnessScore +
        0.20 * rangeScore -
        10 * exposurePenalty,
        0,
        100
      );

    let label = "FAIR";

    if (qualityScore >= 82) {
      label = "EXCELLENT";
    } else if (qualityScore >= 68) {
      label = "GOOD";
    } else if (qualityScore < 45) {
      label = "LIMITED";
    }

    return {
      resolution:
        `${width} × ${height}`,

      contrast:
        round(contrast, 2),

      sharpness:
        round(sharpness, 2),

      qualityScore:
        round(qualityScore, 1),

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
      py * width + px;

    return [
      gray[index + 1] -
      gray[index - 1],

      gray[index + width] -
      gray[index - width]
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

    const samplingStep =
      Math.max(
        2,
        Math.floor(
          Math.min(
            width,
            height
          ) / 250
        )
      );

    const threshold = 18;

    for (
      let y = 6;
      y < height - 6;
      y += samplingStep
    ) {

      for (
        let x = 6;
        x < width - 6;
        x += samplingStep
      ) {

        const center =
          gray[
            y * width + x
          ];

        let brighter = 0;
        let darker = 0;

        for (
          let k = 0;
          k < circle.length;
          k++
        ) {

          const dx =
            circle[k][0];

          const dy =
            circle[k][1];

          const value =
            gray[
              (y + dy) * width +
              (x + dx)
            ];

          if (
            value >
            center + threshold
          ) {
            brighter++;
          }

          if (
            value <
            center - threshold
          ) {
            darker++;
          }
        }

        if (
          brighter < 7 &&
          darker < 7
        ) {
          continue;
        }

        let sxx = 0;
        let syy = 0;
        let sxy = 0;

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

            sxx += gx * gx;
            syy += gy * gy;
            sxy += gx * gy;
          }
        }

        const determinant =
          sxx * syy -
          sxy * sxy;

        const trace =
          sxx +
          syy +
          1e-6;

        const response =
          determinant / trace;

        if (response > 18) {

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

    const minimumDistance = 14;

    for (
      const candidate of candidates
    ) {

      let accepted = true;

      for (
        const existing of selected
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

          accepted = false;
          break;
        }
      }

      if (!accepted) {
        continue;
      }

      selected.push(candidate);

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
     DESCRIPTOR
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
      PATCH_RADIUS + 2;

    if (
      point.x < radius ||
      point.y < radius ||
      point.x >= width - radius ||
      point.y >= height - radius
    ) {
      return null;
    }

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

        directionX += gx;
        directionY += gy;
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

    for (
      let y = -8;
      y <= 8;
      y += DESCRIPTOR_STEP
    ) {

      for (
        let x = -8;
        x <= 8;
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

        const value =
          gray[
            ry * width + rx
          ] / 255;

        descriptor.push(value);
      }
    }

    let mean = 0;

    for (
      const value of descriptor
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

      descriptor[i] -= mean;
    }

    let norm = 0;

    for (
      const value of descriptor
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
      detectFeatures(image);

    const features = [];

    for (
      const point of points
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
        x: point.x,
        y: point.y,
        score: point.score,
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

    for (
      let i = 0;
      i < a.length;
      i++
    ) {

      const difference =
        a[i] - b[i];

      sum +=
        difference *
        difference;
    }

    return Math.sqrt(sum);
  }


  /* ============================================================
     FEATURE MATCHING
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
            b.score - a.score
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
            b.score - a.score
        )
        .slice(
          0,
          MAX_MATCH_FEATURES
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
          distance < best
        ) {

          second = best;
          best = distance;
          bestIndex = j;

        } else if (
          distance < second
        ) {

          second = distance;
        }
      }

      if (
        bestIndex >= 0 &&
        second > 0 &&
        best / second <
        LOWE_RATIO
      ) {

        forward.push({
          ai: i,
          bi: bestIndex,
          distance:
            best / second
        });
      }
    }

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
          distance < best
        ) {

          best = distance;
          bestIndex = i;
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

    const mutual =
      forward.filter(
        match =>
          reverse.get(
            match.bi
          ) === match.ai
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
          match.distance
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

            pivot = row;
          }
        }

        if (
          Math.abs(
            M[pivot][column]
          ) < 1e-8
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
          let c = column;
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
            row === column
          ) {
            continue;
          }

          const factor =
            M[row][column];

          for (
            let c = column;
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

    if (!X || !Y) {
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
        model.a * point.x +
        model.b * point.y +
        model.c,

      y:
        model.d * point.x +
        model.e * point.y +
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
        b.x - a.x
      ) *
      (
        c.y - a.y
      ) -

      (
        b.y - a.y
      ) *
      (
        c.x - a.x
      )
    );
  }


  /* ============================================================
     RANSAC
     ============================================================ */

  function verifyGeometry(
    matches,
    featuresA,
    featuresB
  ) {

    if (
      matches.length < 4
    ) {

      return {
        model: null,
        inliers: [],
        ratio: 0,
        consistency: 0
      };
    }

    let bestModel =
      null;

    let bestInliers = [];

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
          (i2 + 1) %
          matches.length;
      }

      if (
        i3 === i1 ||
        i3 === i2
      ) {
        i3 =
          (i3 + 2) %
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

      if (
        triangleArea(
          p1,
          p2,
          p3
        ) < 25
      ) {
        continue;
      }

      if (
        triangleArea(
          q1,
          q2,
          q3
        ) < 25
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

      const inliers = [];

      for (
        const match of matches
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

        if (
          error <=
          RANSAC_ERROR_PIXELS
        ) {

          inliers.push(
            match
          );
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

    const ratio =
      matches.length
        ? bestInliers.length /
          matches.length
        : 0;

    let consistency = 0;

    if (
      bestModel &&
      bestInliers.length >= 4
    ) {

      let totalError = 0;

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

        totalError +=
          Math.hypot(
            predicted.x -
              target.x,

            predicted.y -
              target.y
          );
      }

      const meanError =
        totalError /
        bestInliers.length;

      consistency =
        clamp(
          100 -
          meanError * 12,
          0,
          100
        );
    }

    return {
      model:
        bestModel,

      inliers:
        bestInliers,

      ratio,

      consistency
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

    if (!inliers.length) {
      return 0;
    }

    const occupied =
      new Set();

    for (
      const match of inliers
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
            4
          ),
          0,
          3
        );

      const gy =
        clamp(
          Math.floor(
            point.y /
            height *
            4
          ),
          0,
          3
        );

      occupied.add(
        `${gx}:${gy}`
      );
    }

    return (
      occupied.size /
      16 *
      100
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

    const correspondenceDensity =
      clamp(
        verified /
        featureCount *
        180,
        0,
        100
      );

    const geometricScore =
      clamp(
        verification.ratio *
        150,
        0,
        100
      );

    const coverage =
      calculateCoverage(
        verification.inliers,
        featuresA,
        width,
        height
      );

    const quality =
      (
        metricsA.qualityScore +
        metricsB.qualityScore
      ) / 2;

    const score =
      clamp(
        0.34 *
          correspondenceDensity +

        0.30 *
          geometricScore +

        0.18 *
          verification.consistency +

        0.10 *
          coverage +

        0.08 *
          quality,

        0,
        100
      );

    let classification =
      "NO RELIABLE CORRESPONDENCE";

    if (
      verified >= 18 &&
      score >= 78 &&
      verification.ratio >= 0.35
    ) {

      classification =
        "STRONG CORRESPONDENCE";

    } else if (
      verified >= 10 &&
      score >= 62 &&
      verification.ratio >= 0.25
    ) {

      classification =
        "CORRESPONDENCE FOUND";

    } else if (
      verified >= 6 &&
      score >= 48 &&
      coverage >= 20
    ) {

      classification =
        "POSSIBLE CORRESPONDENCE";
    }

    return {

      score:
        round(score, 1),

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
      result.score >= 78
    ) {

      return (
        `Strong visual correspondence detected. ` +
        `${result.verifiedMatches} geometrically verified ` +
        `feature relationships were retained, with ` +
        `${round(result.featureCoverage, 0)}% spatial coverage ` +
        `and ${round(result.geometricConsistency, 0)}% ` +
        `geometric consistency. The evidence supports a ` +
        `high-confidence correspondence assessment.`
      );
    }

    if (
      result.score >= 62
    ) {

      return (
        `Meaningful correspondence was detected. ` +
        `${result.verifiedMatches} verified relationships ` +
        `survived geometric filtering. Spatial coverage ` +
        `reached ${round(result.featureCoverage, 0)}%, while ` +
        `geometric consistency was ` +
        `${round(result.geometricConsistency, 0)}%. ` +
        `The result is supportive but should be reviewed ` +
        `alongside the correspondence visualization.`
      );
    }

    if (
      result.score >= 48
    ) {

      return (
        `Some local similarities were detected, but the ` +
        `evidence is limited. Only ` +
        `${result.verifiedMatches} relationships survived ` +
        `geometric verification. Differences in illumination, ` +
        `viewpoint, scale, image quality, or surface appearance ` +
        `may be affecting the result.`
      );
    }

    return (
      `No reliable global correspondence was established. ` +
      `The detected local similarities were not sufficiently ` +
      `consistent under geometric verification. Try higher-` +
      `resolution, better-focused images with overlapping ` +
      `lunar terrain.`
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
    sourceB
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
      widthA + gap,
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

    context.lineWidth = 1.2;

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
          scale;

        const y1 =
          pointA.y *
          scale;

        const x2 =
          widthA +
          gap +
          pointB.x *
          scale;

        const y2 =
          pointB.y *
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
          2.4,
          0,
          Math.PI * 2
        );

        context.fill();

        context.beginPath();

        context.arc(
          x2,
          y2,
          2.4,
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
      widthA + gap,
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
      widthA + gap + 12,
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
      result.score >= 78
        ? "HIGH"
        : result.score >= 62
          ? "MODERATE"
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

      pipelineActive(
        "stageExtract"
      );

      setText(
        "status",
        "EXTRACTING FEATURES"
      );

      await delay(30);

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

      pipelineActive(
        "stageMatch"
      );

      setText(
        "status",
        "MATCHING CORRESPONDENCES"
      );

      await delay(30);

      const matches =
        matchFeatures(
          featuresA,
          featuresB
        );

      pipelineComplete(
        "stageMatch"
      );

      pipelineActive(
        "stageVerify"
      );

      setText(
        "status",
        "VERIFYING GEOMETRY"
      );

      await delay(30);

      const verification =
        verifyGeometry(
          matches,
          featuresA,
          featuresB
        );

      pipelineComplete(
        "stageVerify"
      );

      pipelineActive(
        "stageScore"
      );

      setText(
        "status",
        "CALCULATING CONFIDENCE"
      );

      await delay(30);

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

      pipelineActive(
        "stageReport"
      );

      setText(
        "status",
        "BUILDING RESULT"
      );

      await delay(30);

      const visualization =
        createCorrespondenceMap(
          featuresA,
          featuresB,
          verification.inliers,
          sourceA,
          sourceB
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
          verification.inliers.length >= 6
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

          existing.addEventListener(
            "load",
            () =>
              resolve(
                window.jspdf?.jsPDF
              )
          );

          existing.addEventListener(
            "error",
            () =>
              reject(
                new Error(
                  "PDF engine could not be loaded."
                )
              )
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

      const margin = 16;

      let y = 18;

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
        const line of statistics
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
          imageWidth * 0.55;

        if (
          y + imageHeight >
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
     IMAGE INPUT
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

    /*
     * Force the correct input type.
     */

    input.type =
      "file";

    /*
     * Explicitly allow common lunar image formats.
     */

    input.accept =
      "image/jpeg,image/png,image/webp,image/bmp,image/tiff,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff";

    /*
     * Handle normal file selection.
     */

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

    /*
     * Make the entire upload card clickable.
     */

    zone.addEventListener(
      "click",
      event => {

        /*
         * If the actual file input itself was clicked,
         * don't trigger another click.
         */

        if (
          event.target === input
        ) {
          return;
        }

        console.log(
          `LUNARMATCH: Opening Image ${slot} file picker.`
        );

        input.click();
      }
    );


    /*
     * Drag enter / drag over.
     */

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


    /*
     * Drag leave / drop.
     */

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


    /*
     * Handle dropped image.
     */

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


      /*
       * Initialize Image A input.
       */

      setupImageInput(
        "fileA",
        "previewA",
        "A"
      );


      /*
       * Initialize Image B input.
       */

      setupImageInput(
        "fileB",
        "previewB",
        "B"
      );


      /*
       * Get actual input elements.
       */

      const inputA =
        $("fileA");

      const inputB =
        $("fileB");


      /*
       * Find upload cards.
       */

      const dropZones =
        document.querySelectorAll(
          ".drop-new"
        );


      console.log(
        "LUNARMATCH upload zones:",
        dropZones.length
      );


      /*
       * Initialize Image A upload card.
       */

      setupDropZone(
        dropZones[0],
        inputA,
        "previewA",
        "A"
      );


      /*
       * Initialize Image B upload card.
       */

      setupDropZone(
        dropZones[1],
        inputB,
        "previewB",
        "B"
      );


      /*
       * Compare button.
       */

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


      /*
       * PDF report button.
       */

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


      /*
       * Other UI systems.
       */

      setupNavigation();

      setupButtonEffects();

      setupPlaceholderActions();

      resetResults();


      /*
       * Startup diagnostics.
       */

      console.log(
        "LUNARMATCH V6 — Lunar Correspondence Engine ONLINE"
      );

      console.log(
        "Engine: Local normalization + feature detection + descriptor matching + RANSAC"
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
