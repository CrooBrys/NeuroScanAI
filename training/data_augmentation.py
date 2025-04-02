import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def balance_with_augmentation(X, y, target_class_size=None, verbose=True, samples_per_class=3, random_state=42):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        brightness_range=[0.95, 1.05],
        fill_mode='nearest',
        horizontal_flip=True  # now acceptable, but optional
    )
    
    # Set numpy random seed
    np.random.seed(random_state)

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

        # Set a seed for each class for reproducibility
        temp_gen = datagen.flow(
            X_cls, y_cls, batch_size=1, shuffle=False, seed=random_state + int(cls)
        )

        if verbose and num_to_add > 0:
            print(f"Augmenting class {cls} with {num_to_add} synthetic samples...")

        augment_idx = 0
        for i in range(num_to_add):
            x_aug, y_aug = next(temp_gen)
            original_idx = augment_idx % len(X_cls)
            original_img = X_cls[original_idx]

            if verbose and i < samples_per_class:
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(original_img.astype('uint8'))
                axes[0].set_title("Original")
                axes[0].axis("off")
                axes[1].imshow(x_aug[0].astype('uint8'))
                axes[1].set_title("Augmented")
                axes[1].axis("off")
                plt.suptitle(f"Class {cls} - Sample {i+1}")
                plt.tight_layout()
                plt.show()

            X_balanced.append(x_aug[0])
            y_balanced.append(y_aug[0])
            augment_idx += 1

    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)

    if verbose:
        print("Augmented balanced class distribution:")
        final_counts = np.bincount(y_balanced)
        for cls, count in enumerate(final_counts):
            print(f"  Class {cls}: {count} images")
        print()
        print("Final balanced dataset shape:")
        print(f"  Images: {X_balanced.shape}")
        print(f"  Labels: {y_balanced.shape}\n")

    return X_balanced, y_balanced