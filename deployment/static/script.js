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

  // 🌙 Dark mode toggle
  toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark", toggle.checked);
  });

  // 🔄 Reset state on page load
  window.addEventListener('load', () => {
    imageInput.value = '';
    preview.style.display = 'none';
    historyContainer.innerHTML = ""; // Clear previous entries
  });

  // 📷 Image preview handler
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

  // 🧪 Clear result when switching model
  modelSelect.addEventListener("change", () => {
    result.innerHTML = "";
  });

  // 🧠 Form submission and prediction
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
        // 🎯 Show result
        result.innerHTML = `
          <p>Prediction: <strong>${data.class}</strong></p>
          <p>Confidence: ${data.confidence}%</p>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width:${data.confidence}%;">
              ${data.confidence}%
            </div>
          </div>
        `;

        // 📝 Add to history
        const fileName = file.name;
        const timestamp = new Date().toLocaleString();
        const confidenceClass =
          data.confidence >= 85
            ? "high"
            : data.confidence >= 60
            ? "medium"
            : "low";

        const historyEntry = document.createElement("div");
        historyEntry.className = "history-entry";
        historyEntry.innerHTML = `
          <div class="history-meta">
            <strong>${fileName}</strong> &mdash; ${timestamp}
          </div>
          <div>Prediction: <strong>${data.class}</strong></div>
          <div class="confidence-bar history-bar ${confidenceClass}">
            <div class="confidence-fill" style="width:${data.confidence}%;">
              ${data.confidence}%
            </div>
          </div>
          <hr>
        `;

        if (historyContainer) {
          historyContainer.prepend(historyEntry);
        }
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
});
