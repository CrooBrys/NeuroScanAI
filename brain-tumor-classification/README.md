# Brain Tumor Classification (MRI)

---

## Overview  
This project aims to classify brain tumors using MRI scans. The dataset consists of four categories: **Glioma, Meningioma, Pituitary, and None (No tumor)**. The goal is to train and evaluate multiple models to determine the best-performing classifier and deploy a simple application for real-time MRI classification.

---

## Dataset  
**Source:** [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  

- **Image Format:** JPG  
- **Classes:** Glioma, Meningioma, Pituitary, None  
- **Pre-split Dataset:** Training & Testing  

---

## Project Structure  
```
/brain-tumor-classification
│── /deployment (Flask Web App)
│   │── /static (CSS, JS, images for the web interface)
│   │── /templates (HTML templates for the web app)
│   │── app.py (Flask server script)
│   │── model.pth or model.h5 (Trained model for inference)
│   │── requirements.txt (Dependencies for running Flask app)
│
│── /training (Machine Learning Code)
│   │── /data (Dataset Folder)
│   │   │── /testing
│   │   │   │── Glioma (Images of Glioma tumors)
│   │   │   │── Meningioma (Images of Meningioma tumors)
│   │   │   │── None (Images without tumors)
│   │   │   │── Pituitary (Images of Pituitary tumors)
│   │   │── /training
│   │   │   │── Glioma (Images of Glioma tumors)
│   │   │   │── Meningioma (Images of Meningioma tumors)
│   │   │   │── None (Images without tumors)
│   │   │   │── Pituitary (Images of Pituitary tumors)
│   │── brain_tumor_classification.ipynb (Jupyter Notebook for experiments, visualizations, and analysis)
│   │── evaluate.py (Model evaluation and metrics calculation)
│   │── model_utils.py (Helper functions for model building and training)
│   │── preprocess.py (Handles data preprocessing and augmentation)
│   │── train.py (Model training script)
│
│── .gitignore (Files to ignore in version control)
│── README.md (Project documentation)
│── requirements.txt (General dependencies)
```

---

## Project Workflow  
### **1. Data Preprocessing**  
- Merge existing training & test datasets.  
- Create a custom **70/30 train-test split** with **k-fold cross-validation**.  
- Check image integrity and standardize dimensions.  

### **2. Data Augmentation**  
- Apply geometric transformations (rotation, flipping, zooming, etc.).  
- Adjust brightness, contrast, and add noise.  

### **3. Feature Engineering & Normalization**  
- Scale pixel values to **0-1**.  
- Extract features using **pre-trained CNNs (ResNet, VGG, EfficientNet)**.  

### **4. Model Selection & Training**  
- Train **CNN-based models**.  
- Experiment with the most popular models for this task:  
  - **ResNet50**  
  - **VGG16**  
  - **EfficientNetB0**  
  - **InceptionV3**  
- Apply **hyperparameter tuning** and **k-fold cross-validation**.  

### **5. Model Evaluation**  
- Use metrics like **F1-score, Recall, Precision, AUC-ROC**.  
- Analyze results with a confusion matrix.  
- The **Jupyter Notebook (`brain_tumor_classification.ipynb`) generates plots** of:
  - Loss and accuracy curves
  - Confusion matrices
  - ROC curves
  - Sample image predictions with model confidence scores

### **6. Deployment**  
- Develop a simple **Flask-based local application**.  
- Allow users to upload MRI images for classification.  

---

## Dependencies  
- Python 3.x  
- TensorFlow / PyTorch  
- OpenCV  
- Scikit-learn  
- Matplotlib / Seaborn  
- Flask (for deployment)  

---

## Usage  
### **1. Clone the Repository**  
```bash
git clone https://github.com/chiggi24/cs4442_final_project.git
cd brain-tumor-classification
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Running the Code**  
Use an editor like **VS Code, Jupyter Notebook, or any Python IDE** to open and execute the `.ipynb` or `.py` scripts.  

### **4. Training the Model**  
Run the script that contains the model training steps:  
```bash
python training/train.py
```
To **visualize training results**, open the `brain_tumor_classification.ipynb` notebook and run the cells to generate:
- Model accuracy & loss curves
- Confusion matrix
- ROC curve
- Sample MRI classification predictions with confidence scores

### **5. Deployment (Local Flask App)**  
Run the Flask application locally:  
```bash
cd deployment
python app.py
```
Then open a browser and go to:  
```bash
http://127.0.0.1:5000
```
Here, you can upload MRI images for classification.  

---

## Contributors  
- **Christopher Higgins**
- **Bryson Crook**
- **Mohamed El Dogdog**

---

## License  
MIT License  

---
