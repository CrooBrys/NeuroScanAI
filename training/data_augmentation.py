import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def balance_with_augmentation(X, y, target_class_size=None):
    """
    Balances the dataset using data augmentation to match the highest class count.

    Parameters:
    - X: numpy array of images (shape: [num_samples, height, width, channels])
    - y: numpy array of labels (integers)
    - target_class_size: if specified, all classes will be augmented to this size

    Returns:
    - X_balanced: numpy array of balanced images
    - y_balanced: numpy array of corresponding labels
    """

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_samples = target_class_size if target_class_size else np.max(class_counts)

    print("Original class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} images")

    X_balanced = []
    y_balanced = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        y_cls = y[cls_indices]

        num_to_add = max_samples - len(X_cls)

        # Keep original samples
        X_balanced.extend(X_cls)
        y_balanced.extend(y_cls)

        if num_to_add > 0:
            print(f"Augmenting class {cls} with {num_to_add} synthetic samples...")
            temp_gen = datagen.flow(X_cls, y_cls, batch_size=1)
            for _ in range(num_to_add):
                x_aug, y_aug = next(temp_gen)
                X_balanced.append(x_aug[0])
                y_balanced.append(y_aug[0])

    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)

    print("Final balanced dataset shape:")
    print("  Images:", X_balanced.shape)
    print("  Labels:", y_balanced.shape)

    return X_balanced, y_balanced