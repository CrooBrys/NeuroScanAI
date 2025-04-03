document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const imageInput = document.getElementById("image");
  const modelSelect = document.getElementById("model-select");
  const preview = document.getElementById("preview");
  const loader = document.getElementById("loader");
  const result = document.getElementById("result");
  const history = document.getElementById("history");
  const historyContainer = document.getElementById("history-entries");
  const submitBtn = form.querySelector('button[type="submit"]');
  const toggle = document.getElementById("dark-toggle");


  // Dark mode toggle
  toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark", toggle.checked);
  });

  // Clear inputs and history on page load
  window.addEventListener('load', () => {
    imageInput.value = '';
    preview.style.display = 'none';
    historyContainer.innerHTML = "";
    const scrollHint = document.getElementById("scroll-hint");
    if (scrollHint) scrollHint.classList.remove("visible");
  });

  // Preview image when selected
  imageInput.addEventListener("change", () => {
    result.innerHTML = "";
    const file = imageInput.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = "block";
      };
      reader.readAsDataURL(file);
    } else {
      preview.style.display = "none";
    }
  });

  // Clear result on model change
  modelSelect.addEventListener("change", () => {
    result.innerHTML = "";
  });

  // Form submit handler
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = imageInput.files[0];
    const selectedModel = modelSelect.value;

    if (!file || !selectedModel) {
      result.innerHTML = `<p style="color:red;">Please select a model and image.</p>`;
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    const url = `/upload?model=${encodeURIComponent(selectedModel)}`;

    imageInput.disabled = true;
    modelSelect.disabled = true;
    submitBtn.disabled = true;
    loader.style.display = "block";
    result.innerHTML = "";

    try {
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      loader.style.display = "none";

      if (data.error) {
        result.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
      } else {
        const fill = document.createElement("div");
        fill.className = "confidence-fill";
        fill.style.width = `${data.confidence}%`;
        fill.textContent = `${data.confidence}%`;
        fill.style.background = getConfidenceColor(data.confidence);

        result.innerHTML = `
          <p>Prediction: <strong>${data.class}</strong></p>
          <p>Confidence: ${data.confidence}%</p>
          <div class="confidence-bar"></div>
        `;
        result.querySelector(".confidence-bar").appendChild(fill);

        const fileName = file.name;
        const timestamp = new Date().toLocaleString();

        const historyEntry = document.createElement("div");
        historyEntry.className = "history-entry";

        const historyFill = document.createElement("div");
        historyFill.className = "confidence-fill";
        historyFill.style.width = `${data.confidence}%`;
        historyFill.textContent = `${data.confidence}%`;
        historyFill.style.background = getHistoryGradient(data.confidence);

        historyEntry.innerHTML = `
          <div class="history-meta">
            <strong>${fileName}</strong> &mdash; ${timestamp}
          </div>
          <div>Prediction: <strong>${data.class}</strong></div>
          <div class="confidence-bar history-bar"></div>
          <hr>
        `;
        historyEntry.querySelector(".history-bar").appendChild(historyFill);
        historyContainer.prepend(historyEntry);

        checkHistoryScrollState(); // Update scrollbar logic
      }
    } catch (err) {
      loader.style.display = "none";
      result.innerHTML = `<p style="color:red;">Request failed. Please try again later.</p>`;
    } finally {
      imageInput.disabled = false;
      modelSelect.disabled = false;
      submitBtn.disabled = false;
    }
  });

  // Color logic for live prediction bar
  function getConfidenceColor(confidence) {
    const baseHue = 180;
    const lightness = 70 - (confidence / 100) * 30;
    return `hsl(${baseHue}, 70%, ${lightness}%)`;
  }

  // Color logic for history bar (green, yellow, red)
  function getHistoryGradient(confidence) {
    const clamp = (val, min, max) => Math.max(min, Math.min(max, val));
    const r = confidence < 60
      ? 239 + (confidence / 60) * (255 - 239)
      : confidence < 85
      ? 255 - ((confidence - 60) / 25) * (255 - 76)
      : 76 - ((confidence - 85) / 15) * 30;

    const g = confidence < 60
      ? 83 + (confidence / 60) * (167 - 83)
      : confidence < 85
      ? 167 + ((confidence - 60) / 25) * (175 - 167)
      : 175 - ((confidence - 85) / 15) * 30;

    const b = confidence < 60
      ? 80 + (confidence / 60) * (38 - 80)
      : confidence < 85
      ? 38 + ((confidence - 60) / 25) * (80 - 38)
      : 80 - ((confidence - 85) / 15) * 10;

    return `rgb(${clamp(r, 0, 255)}, ${clamp(g, 0, 255)}, ${clamp(b, 0, 255)})`;
  }

  // Handle scroll behavior for upload history
  let historyScrollRevealed = false;

  function checkHistoryScrollState() {
    const entries = historyContainer.querySelectorAll(".history-entry");
  
    if (entries.length > 2 && !historyScrollRevealed) {
      history.classList.add("scroll-reveal");
  
      const scrollHint = document.getElementById("scroll-hint");
      console.log("Applying visible class to scroll hint");
      if (scrollHint) scrollHint.classList.add("visible");
  
      // Once the user hovers over the box, hide the hint and lock the state
      const onFirstHover = () => {
        historyScrollRevealed = true;
        history.classList.remove("scroll-reveal");
        history.classList.add("scrolled");
  
        if (scrollHint) {
          scrollHint.classList.remove("visible");
        }
  
        history.removeEventListener("mouseenter", onFirstHover);
      };
  
      history.addEventListener("mouseenter", onFirstHover);
    }
    console.log("Checking history scroll state", entries.length);
  }  
});
