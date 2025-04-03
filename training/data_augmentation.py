# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_preprocessing import plot_class_distribution

# Function to balance the dataset by augmenting under-represented classes
def balance_with_augmentation(X, y, class_names, target_class_size=None, verbose=True, samples_per_class=3, random_state=42):
    """
    Balances the dataset by augmenting the under-represented classes with synthetic images.

    Args:
        X (numpy array): Image data.
        y (numpy array): Labels corresponding to the image data.
        class_names (list): List of class names.
        target_class_size (int, optional): The target size for each class after augmentation. If None, uses the max class size.
        verbose (bool, optional): Whether to print detailed output.
        samples_per_class (int, optional): Number of augmented samples to display per class.
        random_state (int, optional): Seed for random number generator to ensure reproducibility.

    Returns:
        numpy array: Augmented image data.
        numpy array: Augmented labels.
    """
    # Define the augmentation techniques with a variety of transformations
    datagen = ImageDataGenerator(
        rotation_range=15,  # Random rotation between -15 and 15 degrees
        width_shift_range=0.05,  # Horizontal shift of up to 5% of the image width
        height_shift_range=0.05,  # Vertical shift of up to 5% of the image height
        zoom_range=0.1,  # Random zoom between 90% and 110% of the original image size
        brightness_range=[0.95, 1.05],  # Random brightness adjustments
        fill_mode='nearest',  # Strategy for filling missing pixels after transformations
        horizontal_flip=True  # Random horizontal flipping
    )

    np.random.seed(random_state)  # Set the seed for reproducibility of random transformations

    unique_classes, class_counts = np.unique(y, return_counts=True)  # Get unique classes and their counts
    max_samples = target_class_size if target_class_size else np.max(class_counts)  # Determine the target size for classes

    if verbose:
        print("\nOriginal class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} images")  # Print the original class distribution
        plot_class_distribution(y, class_names, "Original Class Distribution")  # Plot the original class distribution

    # Initialize lists to hold the balanced data
    X_balanced, y_balanced = [], []

    # Loop through each class to balance the dataset
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]  # Get indices of images belonging to the current class
        X_cls = X[cls_indices]  # Extract images of the current class
        y_cls = y[cls_indices]  # Extract labels of the current class
        num_to_add = max_samples - len(X_cls)  # Calculate how many new samples are needed

        # Append the original images and labels to the balanced dataset
        X_balanced.extend(X_cls)
        y_balanced.extend(y_cls)

        # Create an image generator for augmenting the current class
        temp_gen = datagen.flow(X_cls, y_cls, batch_size=1, shuffle=False, seed=random_state + int(cls))

        if verbose and num_to_add > 0:
            print(f"Augmenting class {cls} with {num_to_add} synthetic samples...")  # Print progress message

        # Generate synthetic samples for the current class
        for i in range(num_to_add):
            x_aug, y_aug = next(temp_gen)  # Generate one augmented image and its label
            X_balanced.append(x_aug[0])  # Add augmented image to the balanced dataset
            y_balanced.append(y_aug[0])  # Add corresponding label to the balanced dataset

            # Optionally display a comparison of original and augmented samples
            if verbose and i < samples_per_class:
                fig, axes = plt.subplots(1, 2, figsize=(6, 3))  # Create subplots to compare original and augmented images
                axes[0].imshow(X_cls[i % len(X_cls)].astype('uint8'))  # Display original image
                axes[0].set_title("Original")
                axes[0].axis("off")  # Hide axis for better visualization
                axes[1].imshow(x_aug[0].astype('uint8'))  # Display augmented image
                axes[1].set_title("Augmented")
                axes[1].axis("off")  # Hide axis for better visualization
                plt.suptitle(f"Class {cls} - Sample {i+1}")  # Add a title to the subplot
                plt.tight_layout()  # Adjust layout to prevent overlap
                plt.show()  # Show the subplot with the original and augmented images

    X_balanced, y_balanced = np.array(X_balanced), np.array(y_balanced)  # Convert lists to numpy arrays

    if verbose:
        # Print the distribution of classes in the augmented dataset
        print("\nAugmented class distribution:")
        for cls, count in enumerate(np.bincount(y_balanced)):
            print(f"  Class {cls}: {count} images")  # Print class distribution after augmentation
        print(f"\nFinal dataset shape: {X_balanced.shape}, Labels: {y_balanced.shape}\n")  # Print final dataset shape

    return X_balanced, y_balanced  # Return the balanced (augmented) image data and labels
