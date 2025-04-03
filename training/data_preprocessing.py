# Import necessary libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from config import DATA_DIR, IMAGE_SIZE
from data_augmentation import balance_with_augmentation
from tensorflow.keras.applications import resnet50, vgg16, efficientnet, inception_v3

# Function to apply model-specific preprocessing
def apply_model_preprocessing(X, model_name):
    X = X.astype("float32")  # Ensure float type for efficiency
    if model_name == "ResNet50":
        return resnet50.preprocess_input(X)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(X)
    elif model_name == "EfficientNetB0":
        return efficientnet.preprocess_input(X)
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(X)
    else:
        return X / 255.0  # Normalize to [0,1] for unknown models

# Function to load images and labels
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error resizing '{img_path}': {e}")

    return np.array(images), np.array(labels)

# Function to plot class distribution
def plot_class_distribution(y, class_names, title="Class Distribution"):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette="viridis", hue=y)  # Explicitly use hue=y
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.xlabel("Tumor Type")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.legend(title="Tumor Types", loc='upper right')  # Add legend to match the color scheme
    plt.show()

# Function to visualize PCA of image dataset
def visualize_pca(X, y, title="PCA of Image Dataset"):
    X_flat = X.reshape(X.shape[0], -1)  # Flatten images
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Class Labels")
    plt.title(title)
    plt.show()

# Function to preprocess and split data
def preprocess_data(model_name, verbose=True):
    X, y_raw = load_all_images()

    if len(X) == 0 or len(y_raw) == 0:
        print("Error: No data loaded.")
        return None, None, None, None, None

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = encoder.classes_

    if verbose:
        print("\nDataset Summary:")
        print(f"  Total Images: {len(X)}")
        print(f"  Image Shape: {X.shape[1:]}")

    # Dynamically determine max class size for augmentation
    max_class_size = max(np.bincount(y))
    X_balanced, y_balanced = balance_with_augmentation(
        X, y, class_names, target_class_size=max_class_size, verbose=verbose
    )

    X_balanced = apply_model_preprocessing(X_balanced.astype("float32"), model_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )

    if verbose:
        print("Train dataset distribution:", np.bincount(y_train).tolist())
        print("Test dataset distribution:", np.bincount(y_test).tolist())
        plot_class_distribution(y_train, class_names, "Training Class Distribution")
        plot_class_distribution(y_test, class_names, "Testing Class Distribution")
        visualize_pca(X_train, y_train, "PCA of Training Data")

    return X_train, X_test, y_train, y_test, class_names
