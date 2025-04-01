import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from config import DATA_DIR, IMAGE_SIZE

def merge_datasets():
    # Merge training and testing datasets into one "merged" folder
    merged_dir = os.path.join(DATA_DIR, "merged")
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)  # Remove the old merged directory if it exists
    os.makedirs(merged_dir)  # Create a new merged directory
    
    for split in ["training", "testing"]:  # Iterate over 'training' and 'testing' directories
        split_path = os.path.join(DATA_DIR, split)
        for category in os.listdir(split_path):  # Iterate over categories (e.g., 'glioma', 'meningioma', etc.)
            category_path = os.path.join(split_path, category)
            merged_category_path = os.path.join(merged_dir, category)
            os.makedirs(merged_category_path, exist_ok=True)  # Create category directory in the merged folder
            
            for img_name in os.listdir(category_path):  # Iterate over images in each category
                shutil.copy(os.path.join(category_path, img_name), merged_category_path)  # Copy image to merged category folder

    return merged_dir  # Return path to the merged directory

def load_images():
    images, labels = [], []
    
    # Get the path to the merged dataset
    dataset_dir = merge_datasets()
    for category in os.listdir(dataset_dir):  # Iterate over categories in the merged folder
        category_path = os.path.join(dataset_dir, category)
        
        for img_name in os.listdir(category_path):  # Iterate over images in each category
            img_path = os.path.join(category_path, img_name)
            
            # Read image using OpenCV
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMAGE_SIZE)
            
            # Append the image and its label
            images.append(img)
            labels.append(category)
    
    # Convert the lists to numpy arrays
    return np.array(images), np.array(labels)

def preprocess_data():
    # Load images and their labels
    images, labels = load_images()
    
    # Normalize pixel values to range [0, 1]
    images = images / 255.0
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels)
    return X_train, X_test, y_train, y_test
