import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def show_model_preprocessing_comparison(image, title="image(2).jpg"):
    img_array = image.astype('float32')

    models = {
        "Original": img_array,
        "ResNet50": resnet50.preprocess_input(img_array.copy()),
        "VGG16": vgg16.preprocess_input(img_array.copy()),
        "InceptionV3": inception_v3.preprocess_input(img_array.copy()),
        "EfficientNetB0": efficientnet.preprocess_input(img_array.copy()),
    }

    fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
    for ax, (name, processed) in zip(axes, models.items()):
        # Normalize to [0, 255] for display
        disp_img = processed.copy()
        if name != "Original":
            disp_img = ((disp_img - disp_img.min()) / (disp_img.max() - disp_img.min()) * 255).astype("uint8")
        ax.imshow(disp_img.astype("uint8"))
        ax.set_title(name)
        ax.axis("off")

    plt.suptitle(f"Model Preprocessing Comparison - {title}")
    plt.tight_layout()
    plt.show()

    print("Mean pixel values after preprocessing:")
    for name, processed in models.items():
        mean = processed.mean()
        std = processed.std()
        print(f"  {name}: mean = {mean:.2f}, std = {std:.2f}")

def preprocess_data(model_name, verbose=True):
    # Step 1: Load raw images and labels
    X, y_raw = load_all_images()
    if len(X) == 0 or len(y_raw) == 0:
        print("Error: No data loaded.")
        return None, None, None, None, None

    # Step 2: Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = encoder.classes_

    if verbose:
        print("Class mapping:")
        for i, label in enumerate(class_names):
            print(f"  {i} = {label}")
        print()

    # Optional visualization
    if verbose and len(X) > 2:
        print("Visualizing model-specific preprocessing on sample image (Glioma image[2]):")
        show_model_preprocessing_comparison(X[2], title="image(2).jpg")

    # Step 3: Apply augmentation BEFORE model preprocessing
    X_balanced, y_balanced = balance_with_augmentation(
        X, y, target_class_size=None, verbose=verbose, samples_per_class=2
    )

    # Step 4: Model-specific preprocessing AFTER augmentation
    X_balanced = apply_model_preprocessing(X_balanced.astype('float32'), model_name)

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )

    if verbose:
        print("Train dataset distribution:", np.bincount(y_train).tolist())
        print("Test dataset distribution:", np.bincount(y_test).tolist())

    return X_train, X_test, y_train, y_test, class_names