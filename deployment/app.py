import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF warnings/info

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Class names must match your model output order
CLASS_NAMES = ['Glioma', 'Meningioma', 'None', 'Pituitary']

def preprocess_image(image_bytes):
    """Preprocess the uploaded image to fit model input."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # Resize to model input size
    image = np.asarray(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    model_name = request.args.get('model')
    if not model_name:
        return jsonify({'error': 'Model not specified'}), 400

    model_path = os.path.join("../training", "models", model_name)

    if not os.path.isfile(model_path):
        return jsonify({'error': f'Model \"{model_name}\" not found.'}), 404

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"[DEBUG] Model loading error: {e}")
        return jsonify({'error': 'Failed to load model.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            image_bytes = file.read()
            image = preprocess_image(image_bytes)
            prediction = model.predict(image)[0]
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            return jsonify({
                'class': predicted_class,
                'confidence': round(confidence * 100, 2)
            })
        except Exception as e:
            print(f"[DEBUG] Prediction error: {e}")
            return jsonify({'error': 'Prediction failed.'}), 500
    else:
        return jsonify({'error': 'Unsupported file type. Only .jpg, .jpeg, .png allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)