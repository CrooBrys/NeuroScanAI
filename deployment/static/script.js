document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const imageInput = document.getElementById("image");
  const modelSelect = document.getElementById("model-select");
  const preview = document.getElementById("preview");
  const loader = document.getElementById("loader");
  const result = document.getElementById("result");
  const submitBtn = form.querySelector('button[type="submit"]');

  window.addEventListener('load', () => {
    imageInput.value = '';
    preview.style.display = 'none';
  });

  // Clear result on new image selection
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

  // Handle form submission
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

    // Disable all inputs and show loader
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
        result.innerHTML = `
          <p>Prediction: <strong>${data.class}</strong></p>
          <p>Confidence: ${data.confidence}%</p>
        `;
      }
    } catch (err) {
      loader.style.display = "none";
      result.innerHTML = `<p style="color:red;">Request failed. Please try again later.</p>`;
    } finally {
      // Re-enable all inputs
      imageInput.disabled = false;
      modelSelect.disabled = false;
      submitBtn.disabled = false;
    }
  });
});