import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR, IMAGE_SIZE
from data_augmentation import balance_with_augmentation
from tensorflow.keras.applications import resnet50, vgg16, efficientnet, inception_v3

def apply_model_preprocessing(X, model_name):
    if model_name == "ResNet50":
        return resnet50.preprocess_input(X)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(X)
    elif model_name == "EfficientNetB0":
        return efficientnet.preprocess_input(X)
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(X)
    else:
        return X

def load_all_images():
    images, labels = [], []

    if not os.path.exists(DATA_DIR):
        print(f"Error: DATA_DIR '{DATA_DIR}' does not exist.")
        return np.array(images), np.array(labels)

    for category in sorted(os.listdir(DATA_DIR)):
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.isdir(category_path):
            continue

        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"Warning: No images found in '{category_path}', skipping.")
            continue

        for img_name in image_files:
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read '{img_path}', skipping.")
                continue

            try:
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error resizing '{img_path}': {e}")

    return np.array(images), np.array(labels)

def print_class_distributions(y, class_names):
    unique, counts = np.unique(y, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  {class_names[i]}: {count} images")

def preprocess_data(model_name, verbose=True):
    # Step 1: Load all images and labels
    X, y_raw = load_all_images()

    if len(X) == 0 or len(y_raw) == 0:
        print("Error: No data loaded.")
        return None, None, None, None, None

    # Step 2: Apply model-specific preprocessing
    X = apply_model_preprocessing(X.astype('float32'), model_name)

    # Step 3: Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = encoder.classes_

    if verbose:
        print("Class mapping:")
        for i, label in enumerate(class_names):
            print(f"  {i} = {label}")

    # Step 4: Balance dataset via augmentation
    X_balanced, y_balanced = balance_with_augmentation(X, y, verbose=verbose)

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )

    if verbose:
        print("Train dataset distribution:")
        print_class_distributions(y_train, class_names)

        print("Test dataset distribution:")
        print_class_distributions(y_test, class_names)

    return X_train, X_test, y_train, y_test, class_names
