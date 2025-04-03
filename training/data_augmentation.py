# Import necessary libraries for image processing and augmentation
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to balance the dataset by augmenting images in under-represented classes
# This function will create synthetic images using data augmentation techniques
def balance_with_augmentation(X, y, target_class_size=None, verbose=True, samples_per_class=3, random_state=42):
    # Define the augmentation techniques to apply to images
    datagen = ImageDataGenerator(
        rotation_range=15,  # Random rotations between -15 and 15 degrees
        width_shift_range=0.05,  # Horizontal shift
        height_shift_range=0.05,  # Vertical shift
        zoom_range=0.1,  # Random zoom
        brightness_range=[0.95, 1.05],  # Random brightness adjustment
        fill_mode='nearest',  # Fill missing pixels after transformations
        horizontal_flip=True  # Random horizontal flipping
    )
    
    # Set the random seed for reproducibility
    np.random.seed(random_state)

    # Get the unique classes and their counts in the dataset
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    # Determine the target number of samples per class
    max_samples = target_class_size if target_class_size else np.max(class_counts)

    # Optionally print the original class distribution
    if verbose:
        print("\nOriginal class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} images")
        print()

    X_balanced, y_balanced = [], []  # Lists to store augmented data

    # Loop through each unique class
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]  # Get indices of images belonging to the current class
        X_cls = X[cls_indices]  # Select images of the current class
        y_cls = y[cls_indices]  # Select labels for the current class
        num_to_add = max_samples - len(X_cls)  # Calculate how many samples need to be added to balance

        # Add the original images of the current class to the balanced lists
        X_balanced.extend(X_cls)
        y_balanced.extend(y_cls)

        # Initialize the ImageDataGenerator for the current class with a unique seed for reproducibility
        temp_gen = datagen.flow(
            X_cls, y_cls, batch_size=1, shuffle=False, seed=random_state + int(cls)
        )

        # Optionally print information about the augmentation process
        if verbose and num_to_add > 0:
            print(f"Augmenting class {cls} with {num_to_add} synthetic samples...")

        augment_idx = 0  # Counter to keep track of augmented samples

        # Generate augmented images until the class reaches the target size
        for i in range(num_to_add):
            x_aug, y_aug = next(temp_gen)  # Get the next augmented image and its label
            original_idx = augment_idx % len(X_cls)  # Ensure we cycle through the original images
            original_img = X_cls[original_idx]

            # Optionally visualize the original and augmented images
            if verbose and i < samples_per_class:
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))
                axes[0].imshow(original_img.astype('uint8'))  # Display original image
                axes[0].set_title("Original")
                axes[0].axis("off")
                axes[1].imshow(x_aug[0].astype('uint8'))  # Display augmented image
                axes[1].set_title("Augmented")
                axes[1].axis("off")
                plt.suptitle(f"Class {cls} - Sample {i+1}")
                plt.tight_layout()
                plt.show()

            # Add the augmented image and its label to the balanced dataset
            X_balanced.append(x_aug[0])
            y_balanced.append(y_aug[0])
            augment_idx += 1

    # Convert the balanced lists into NumPy arrays
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)

    # Optionally print the final balanced class distribution
    if verbose:
        print("Augmented balanced class distribution:")
        final_counts = np.bincount(y_balanced)
        for cls, count in enumerate(final_counts):
            print(f"  Class {cls}: {count} images")
        print()
        print("Final balanced dataset shape:")
        print(f"  Images: {X_balanced.shape}")
        print(f"  Labels: {y_balanced.shape}\n")

    # Return the balanced dataset (augmented and original images)
    return X_balanced, y_balanced
