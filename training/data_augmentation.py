import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def balance_with_augmentation(X, y, target_class_size=None, verbose=True):
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

    if verbose:
        print("Original class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} images")
        print()

    X_balanced, y_balanced = [], []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        y_cls = y[cls_indices]
        num_to_add = max_samples - len(X_cls)
        X_balanced.extend(X_cls)
        y_balanced.extend(y_cls)
        if num_to_add > 0 and verbose:
            print(f"Augmenting class {cls} with {num_to_add} synthetic samples...")
        temp_gen = datagen.flow(X_cls, y_cls, batch_size=1)
        for _ in range(num_to_add):
            x_aug, y_aug = next(temp_gen)
            X_balanced.append(x_aug[0])
            y_balanced.append(y_aug[0])

    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    if verbose:
        print("\nAugmented balanced class distribution:")
        final_counts = np.bincount(y_balanced)
        for cls, count in enumerate(final_counts):
            print(f"  Class {cls}: {count} images")
        print()
        print("Final balanced dataset shape:")
        print(f"  Images: {X_balanced.shape}")
        print(f"  Labels: {y_balanced.shape}\n")
    return X_balanced, y_balanced
