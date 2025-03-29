import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "data"
OUTPUT_DIR = "processed_data"
IMG_SIZE = (224, 224)
CLASSES = ["Glioma", "Meningioma", "Pituitary", "None"]

def load_images():
    images, labels = [], []
    for label, category in enumerate(CLASSES):
        for phase in ['training', 'testing']:
            path = os.path.join(DATA_DIR, phase, category)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_data():
    images, labels = load_images()
    images = images / 255.0  # Normalize pixel values
    
    # Custom train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)

    # Save processed data
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    print("Data preprocessing and augmentation completed.")
    return X_train, X_test, y_train, y_test
