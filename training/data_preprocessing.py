import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR, IMAGE_SIZE

def merge_datasets():
    merged_dir = os.path.join(DATA_DIR, "merged")
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    os.makedirs(merged_dir)

    for split in ["training", "testing"]:
        split_path = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping.")
            continue

        for category in os.listdir(split_path):
            category_path = os.path.join(split_path, category)
            if not os.path.isdir(category_path):
                continue

            merged_category_path = os.path.join(merged_dir, category)
            os.makedirs(merged_category_path, exist_ok=True)

            images_copied = 0
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if os.path.isfile(img_path):
                    shutil.copy(img_path, merged_category_path)
                    images_copied += 1

            if images_copied == 0:
                print(f"Warning: No images found in {category_path}, directory will be empty.")

    return merged_dir

def load_images():
    images, labels = [], []
    dataset_dir = merge_datasets()
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"Error: Merged dataset {dataset_dir} is empty.")
        return np.array(images), np.array(labels)

    for category in os.listdir(dataset_dir):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path) or len(os.listdir(category_path)) == 0:
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

    # Convert labels to numerical values
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels)

    return X_train, X_test, y_train, y_test
