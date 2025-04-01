import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR, IMAGE_SIZE

def load_images():
    images, labels = [], []
    dataset_dir = os.path.join(DATA_DIR)
    
    if not os.path.exists(dataset_dir) or not any(os.listdir(dataset_dir)):
        print(f"Error: Dataset {dataset_dir} is empty or missing.")
        return np.array(images), np.array(labels)
    
    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path) or not any(os.listdir(category_path)):
            print(f"Warning: {category_path} is empty, skipping.")
            continue
        
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image {img_path}, skipping.")
                continue
            
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            labels.append(category)
    
    return np.array(images), np.array(labels)

def preprocess_data():
    images, labels = load_images()
    if len(images) == 0:
        print("Error: No images loaded.")
        return None, None, None, None
    
    images = images / 255.0  # Normalize pixel values
    
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels)
    
    return X_train, X_test, y_train, y_test
