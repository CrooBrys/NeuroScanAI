import os
import cv2
import traceback
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet50, vgg16, efficientnet, inception_v3
from PIL import Image
from google.cloud import storage

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
bucket_name = os.environ.get("GCS_BUCKET")
IS_LOCAL = True  # Set to False if deploying with GCS

CLASS_NAMES = ['Glioma', 'Meningioma', 'None', 'Pituitary']
model_cache = {}  # Cache for loaded models


def preprocess_image(image_bytes, model_name):
    try:
        print("[DEBUG] Starting image preprocessing...")

        # Decode image
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype("float32")

        # Model-specific preprocessing
        if model_name == "ResNet50.keras":
            img = resnet50.preprocess_input(img)
        elif model_name == "VGG16.keras":
            img = vgg16.preprocess_input(img)
        elif model_name == "EfficientNetB0.keras":
            img = efficientnet.preprocess_input(img)
        elif model_name == "InceptionV3.keras":
            img = inception_v3.preprocess_input(img)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        print("[DEBUG] Preprocessing complete.")
        return np.expand_dims(img, axis=0)

    except Exception as e:
        print(f"[DEBUG] Error during preprocessing: {e}")
        raise


@app.route('/')
def index():
    try:
        print("[DEBUG] Serving index.html")
        return render_template('index.html')
    except Exception as e:
        print(f"[DEBUG] Error rendering index: {e}")
        traceback.print_exc()
        return "Internal Server Error", 500


@app.route('/upload', methods=['POST'])
def upload():
    print("[DEBUG] Upload route called")

    model_name = request.args.get('model')
    if not model_name:
        return jsonify({'error': 'Model not specified'}), 400

    try:
        # Load and cache model
        if model_name not in model_cache:
            if IS_LOCAL:
                model_path = os.path.join("..", "training", "models", model_name)
                print(f"[DEBUG] Loading and caching model from: {model_path}")
                model_cache[model_name] = load_model(model_path)
            else:
                if not bucket_name:
                    return jsonify({'error': 'GCS_BUCKET not set'}), 500

                print(f"[DEBUG] Loading model from GCS bucket: {bucket_name}")
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(f"{model_name}")
                model_temp_path = f"/tmp/{model_name}"
                blob.download_to_filename(model_temp_path)
                model_cache[model_name] = load_model(model_temp_path)

        model = model_cache[model_name]

    except Exception as e:
        print(f"[DEBUG] Model loading error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to load model.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            print("[DEBUG] Processing uploaded image...")
            image_bytes = file.read()
            image = preprocess_image(image_bytes, model_name)

            # Prediction
            prediction = model.predict(image)[0]
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            print("[DEBUG] Prediction:", prediction)

            return jsonify({
                'class': predicted_class,
                'confidence': round(confidence * 100, 2)
            })

        except Exception as e:
            print(f"[DEBUG] Prediction error: {e}")
            traceback.print_exc()
            return jsonify({'error': 'Prediction failed.'}), 500
    else:
        return jsonify({'error': 'Unsupported file type. Only .jpg, .jpeg, .png allowed.'}), 400

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 8080))
        host = 'localhost' if IS_LOCAL else '0.0.0.0'
        print(f"[DEBUG] Starting Flask app on {host}:{port}")
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        print(f"[DEBUG] Error starting Flask app: {e}")
        traceback.print_exc()