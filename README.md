# 🧠 NeuroScanAI

NeuroScanAI is a deep learning-based tool for classifying brain MRI scans into one of four categories: **Glioma**, **Meningioma**, **Pituitary**, or **None** (no tumor). It offers both a model training pipeline and an interactive web interface for live predictions.

We evaluate and compare the performance of four popular convolutional neural networks:
**ResNet50**, **VGG16**, **EfficientNetB0**, and **InceptionV3** — each trained and validated using Stratified K-Fold cross-validation.

---

## Live Demo (Google Cloud Run)
> Try the deployed version here: [**NeuroScanAI Demo**](https://brain-app-427956346530.us-central1.run.app)

This deployment is hosted using **Google Cloud Run**, a fully managed, serverless platform that runs containerized applications with automatic scaling. We containerized our Flask app using the `Dockerfile` in the `deployment/` directory, then uploaded it to Cloud Run. Our four trained `.keras` models are stored in **Google Cloud Storage** and loaded dynamically at runtime.

---

## Dataset Info

- **Source**: [Brain Tumor Classification (MRI) Dataset - Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **License**: MIT License
- **Provided Format**:
  - Grayscale `.jpg` MRI scans with black backgrounds
  - Pre-split into `Training/` and `Testing/` directories, each containing four subfolders (one per class)
- **Preprocessing**:
  - Images were converted to **RGB** and resized to **224x224** for model input compatibility
  - We **merged all images into a single `data/` folder**, organized by four class folders:
    - `Glioma/`
    - `Meningioma/`
    - `Pituitary/`
    - `None/`
- **Classification Task**: Supervised multi-class classification (4 categories)

---

## Project Structure

The repository is organized into two main components: `deployment/` for the web app, and `training/` for the model development pipeline.

```bash
cs4442_final_project/
│
├── deployment/               # Flask-based frontend + backend for prediction
│   ├── static/               # Static assets (JS, CSS, and tooltip examples)
│   │   ├── images/           # Sample MRI images for each tumor class
│   │   ├── script.js         # Handles image upload, preview, results, tooltips
│   │   └── style.css         # Light/dark theme and component styling
│   │
│   ├── templates/            # HTML templates
│   │   └── index.html        # Main web interface layout
│   │
│   ├── app.py                # Main Flask server application
│   ├── Dockerfile            # Docker config for Cloud Run deployment
│   └── requirements.txt      # Dependencies for deployment environment
│
├── training/                 # All training scripts, data, and notebooks
│   ├── data/                 # Merged dataset with subfolders per class
│   │   ├── Glioma/
│   │   ├── Meningioma/
│   │   ├── None/
│   │   └── Pituitary/
│   │
│   ├── models/               # Saved Keras model files from training
│   │
│   ├── brain_tumor_classification.ipynb   # End-to-end training & analysis notebook
│   ├── config.py                         # Hyperparameters and constants
│   ├── data_augmentation.py              # Functions for augmenting MRI data
│   ├── data_preprocessing.py             # Image preprocessing routines
│   ├── feature_extraction.py             # Model feature extraction functions
│   ├── model_evaluation.py               # Evaluation, visualization, and metrics
│   ├── model_training.py                 # K-fold training logic with saving/loading
│   ├── requirements.txt                  # Dependencies for training pipeline
│   └── utils.py                          # Reusable helper functions for training
│
├── .gitignore                # Standard Git ignore rules
└── README.md                 # Project overview and usage guide

```

---

## Getting Started

Ensure you have **Python 3.11** installed before proceeding.

Clone the repository:

```bash
git clone https://github.com/chiggi24/cs4442_final_project.git
cd cs4442_final_project
```

---

## Running the Training Notebook

> Located in `training/brain_tumor_classification.ipynb`

- Make sure you're inside the cloned `cs4442_final_project` directory.
- Navigate to the `training/` folder:

```bash
cd training
```
- Create and activate a virtual environment using Python 3.11:

```bash
python3.11 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
- Install the required dependencies:
```bash
pip install -r requirements.txt
```
- Register the environment as a new Jupyter kernel:
```bash
python -m ipykernel install --user --name=neuroscanai --display-name "NeuroScanAI"
```

- Launch Jupyter Notebook and in the top-right dropdown, select the `NeuroScanAI` kernel
- All training code, plots, and evaluation logic are included
- Trained `.keras` models will be saved into the `training/models/` directory

---

## Local Deployment (Flask Web App)

> Ensure you're using **Python 3.11** for TensorFlow compatibility.

- From the project root (`cs4442_final_project/`), navigate to the `deployment/` folder:

```bash
cd deployment
```
- Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
- Install dependencies:
```bash
pip install -r requirements.txt
```
Run the Flask app:
```bash
python app.py 
```
- Open your browser and go to: [http://localhost:8080](http://localhost:8080)

> **Note**: To speed up training during testing, you can reduce the number of epochs and K-folds in `training/config.py`:

```python
# Number of training epochs
EPOCHS = 2  

# Number of folds for K-Fold Cross-Validation 
K_FOLDS = 2
```

---

## Dependencies

The project maintains **separate environments** for training and deployment, each with its own set of dependencies defined in `requirements.txt`.

### Key libraries:
- **TensorFlow** – Core deep learning framework for model development and inference
- **OpenCV** – Image preprocessing and format handling
- **Matplotlib** & **Seaborn** – Visualization of metrics and results
- **Scikit-learn** – Metrics, preprocessing, and validation utilities
- **Flask** – Lightweight web framework used for deployment
- **Google Cloud Storage** – Access to hosted `.keras` models in production
- **Pillow** – Support for image file operations

---

## Contributors  
- **Christopher Higgins**  
- **Bryson Crook**  
- **Mohamed El Dogdog**