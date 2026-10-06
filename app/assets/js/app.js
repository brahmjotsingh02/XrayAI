(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzoneContent = document.getElementById("dropzone-content");
  const dropzonePreview = document.getElementById("dropzone-preview");
  const previewImg = document.getElementById("preview-img");
  const btnClear = document.getElementById("btn-clear");
  const btnAnalyze = document.getElementById("btn-analyze");
  const btnSpinner = btnAnalyze.querySelector(".btn-spinner");
  const btnText = btnAnalyze.querySelector(".btn-text");

  const resultsEmpty = document.getElementById("results-empty");
  const resultsContent = document.getElementById("results-content");
  const predictionValue = document.getElementById("prediction-value");
  const predictionConfidence = document.getElementById("prediction-confidence");
  const confidenceList = document.getElementById("confidence-list");

  const heatmapOriginal = document.getElementById("heatmap-original");
  const heatmapOverlay = document.getElementById("heatmap-overlay");

  let currentFile = null;
  let currentImageUrl = null;


  // ============================================================
  // Helpers
  // ============================================================

  function setAnalyzeEnabled(enabled) {
    btnAnalyze.disabled = !enabled;
  }


  // ============================================================
  // Clear uploaded image
  // ============================================================

  function clearImage() {
    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }

    currentFile = null;
    fileInput.value = "";

    dropzone.classList.remove("has-file");
    dropzoneContent.classList.remove("hidden");
    dropzonePreview.classList.add("hidden");

    previewImg.removeAttribute("src");

    setAnalyzeEnabled(false);

    resetResults();
    resetHeatmap();
  }


  // ============================================================
  // Reset prediction results
  // ============================================================

  function resetResults() {
    resultsEmpty.classList.remove("hidden");
    resultsContent.classList.add("hidden");
    confidenceList.innerHTML = "";
  }


  // ============================================================
  // Reset heatmap
  // ============================================================

  function resetHeatmap() {
    heatmapOriginal.innerHTML =
      '<span class="placeholder-text">Original X-ray</span>';

    heatmapOriginal.classList.remove("has-image");

    heatmapOverlay.innerHTML =
      '<span class="placeholder-text">Heatmap overlay</span>';

    heatmapOverlay.classList.remove("has-image");
  }


  // ============================================================
  // Load selected image
  // ============================================================

  function loadImage(file) {
    if (!file || !file.type.startsWith("image/")) {
      return;
    }

    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
    }

    currentFile = file;
    currentImageUrl = URL.createObjectURL(file);

    previewImg.src = currentImageUrl;

    dropzone.classList.add("has-file");
    dropzoneContent.classList.add("hidden");
    dropzonePreview.classList.remove("hidden");

    setAnalyzeEnabled(true);

    resetResults();
    resetHeatmap();
  }


  // ============================================================
  // File selection
  // ============================================================

  dropzone.addEventListener("click", function (e) {
    if (e.target.closest(".btn-clear")) {
      return;
    }

    if (!currentFile) {
      fileInput.click();
    }
  });


  dropzone.addEventListener("keydown", function (e) {
    if (
      (e.key === "Enter" || e.key === " ") &&
      !currentFile
    ) {
      e.preventDefault();
      fileInput.click();
    }
  });


  fileInput.addEventListener("change", function () {
    if (fileInput.files[0]) {
      loadImage(fileInput.files[0]);
    }
  });


  // ============================================================
  // Clear button
  // ============================================================

  btnClear.addEventListener("click", function (e) {
    e.stopPropagation();
    clearImage();
  });


  // ============================================================
  // Drag and drop
  // ============================================================

  ["dragenter", "dragover"].forEach(function (eventName) {
    dropzone.addEventListener(eventName, function (e) {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });


  ["dragleave", "drop"].forEach(function (eventName) {
    dropzone.addEventListener(eventName, function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });


  dropzone.addEventListener("drop", function (e) {
    const file = e.dataTransfer.files[0];

    if (file) {
      loadImage(file);
    }
  });


  // ============================================================
  // Render AI prediction results
  // ============================================================

  function renderResults(probabilities, prediction, confidence) {

    /*
     * IMPORTANT:
     *
     * The backend uses these exact class names:
     *
     * COVID
     * Lung_Opacity
     * Normal
     * Viral Pneumonia
     *
     * We keep the backend names internally and only use
     * friendly names for displaying them to the user.
     */

    const displayNames = {
      "COVID": "COVID-19",
      "Lung_Opacity": "Lung Opacity",
      "Normal": "Normal",
      "Viral Pneumonia": "Viral Pneumonia"
    };


    const indexed = Object.keys(probabilities).map(function (name) {
      return {
        name: name,
        displayName: displayNames[name] || name,
        score: Number(probabilities[name]) || 0
      };
    });


    indexed.sort(function (a, b) {
      return b.score - a.score;
    });


    // Use the prediction returned by the model,
    // rather than calculating a separate prediction
    // in the browser.
    const predictedDisplayName =
      displayNames[prediction] || prediction;


    predictionValue.textContent =
      predictedDisplayName;


    predictionConfidence.textContent =
      Number(confidence).toFixed(1) +
      "% confidence";


    confidenceList.innerHTML = indexed
      .map(function (item, rank) {

        return (
          '<li class="confidence-item' +
          (item.name === prediction ? " is-top" : "") +
          '">' +

          '<span class="confidence-name">' +
          item.displayName +
          "</span>" +

          '<span class="confidence-pct">' +
          item.score.toFixed(1) +
          "%" +
          "</span>" +

          '<div class="confidence-bar-track">' +

          '<div class="confidence-bar-fill" ' +
          'style="transform:scaleX(' +
          (item.score / 100) +
          ')">' +

          "</div>" +

          "</div>" +

          "</li>"
        );
      })
      .join("");


    resultsEmpty.classList.add("hidden");
    resultsContent.classList.remove("hidden");


    requestAnimationFrame(function () {

      confidenceList
        .querySelectorAll(".confidence-bar-fill")
        .forEach(function (bar) {

          const transform =
            bar.style.transform;

          bar.style.transform = "scaleX(0)";

          requestAnimationFrame(function () {
            bar.style.transform = transform;
          });

        });

    });
  }


  // ============================================================
  // Render REAL Grad-CAM
  // ============================================================

  function renderHeatmap(imageUrl, heatmapBase64) {

    // Original X-ray
    heatmapOriginal.innerHTML =
      '<img src="' +
      imageUrl +
      '" alt="Original chest X-ray" />';

    heatmapOriginal.classList.add("has-image");


    // Check that backend returned Grad-CAM
    if (!heatmapBase64) {

      heatmapOverlay.innerHTML =
        '<span class="placeholder-text">' +
        "Grad-CAM unavailable" +
        "</span>";

      heatmapOverlay.classList.remove(
        "has-image"
      );

      return;
    }


    /*
     * The backend returns the Grad-CAM as a
     * Base64-encoded PNG.
     *
     * We convert it into a browser image using
     * a data URL.
     */

    const overlayImg =
      document.createElement("img");

    overlayImg.src =
      "data:image/png;base64," +
      heatmapBase64;

    overlayImg.alt =
      "Grad-CAM heatmap showing regions that influenced the AI prediction";

    overlayImg.className =
      "heatmap-canvas";


    heatmapOverlay.innerHTML = "";

    heatmapOverlay.appendChild(
      overlayImg
    );

    heatmapOverlay.classList.add(
      "has-image"
    );
  }


  // ============================================================
  // Loading state
  // ============================================================

  function setLoading(loading) {

    btnAnalyze.classList.toggle(
      "loading",
      loading
    );

    btnSpinner.classList.toggle(
      "hidden",
      !loading
    );

    btnText.textContent =
      loading
        ? "AI Analyzing…"
        : "Run AI Analysis";


    btnAnalyze.disabled =
      loading || !currentFile;


    document.body.classList.toggle(
      "is-analyzing",
      loading
    );
  }


  // ============================================================
  // Run AI analysis
  // ============================================================

  btnAnalyze.addEventListener(
    "click",
    function () {

      if (
        !currentFile ||
        btnAnalyze.classList.contains("loading")
      ) {
        return;
      }


      setLoading(true);


      const formData =
        new FormData();

      formData.append(
        "file",
        currentFile
      );


      fetch("/predict", {
          method: "POST",
          body: formData
        }
      )

        .then(function (response) {

          if (!response.ok) {

            return response
              .json()
              .then(function (error) {

                throw new Error(
                  error.detail ||
                  "The AI server returned an error."
                );

              });

          }

          return response.json();
        })


        .then(function (data) {

          /*
           * Use the actual values returned
           * by the AI backend.
           */

          renderResults(
            data.probabilities,
            data.prediction,
            data.confidence
          );


          /*
           * Display the REAL Grad-CAM
           * returned by the backend.
           */

          renderHeatmap(
            currentImageUrl,
            data.heatmap
          );


          setLoading(false);
        })


        .catch(function (error) {

          console.error(
            "Prediction error:",
            error
          );


          setLoading(false);


          predictionValue.textContent =
            "Analysis failed";

          predictionConfidence.textContent =
            error.message ||
            "Unable to analyze the image.";


          resultsEmpty.classList.add(
            "hidden"
          );

          resultsContent.classList.remove(
            "hidden"
          );

        });

    }
  );

})();