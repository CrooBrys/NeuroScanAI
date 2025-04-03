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
    """
    Applies model-specific preprocessing to the input image data.

    Args:
        X (numpy array): Input image data.
        model_name (str): The model for which preprocessing is required.

    Returns:
        numpy array: Preprocessed image data.
    """
    X = X.astype("float32")  # Convert to float32 for consistency and efficient processing
    # Check for the model type and apply corresponding preprocessing
    if model_name == "ResNet50":
        return resnet50.preprocess_input(X)  # Apply ResNet50 specific preprocessing
    elif model_name == "VGG16":
        return vgg16.preprocess_input(X)  # Apply VGG16 specific preprocessing
    elif model_name == "EfficientNetB0":
        return efficientnet.preprocess_input(X)  # Apply EfficientNetB0 specific preprocessing
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(X)  # Apply InceptionV3 specific preprocessing
    else:
        return X / 255.0  # Normalize pixel values to [0,1] for unknown models

# Function to load images and labels
def load_all_images():
    """
    Loads all images and their corresponding labels from the specified dataset directory.

    Returns:
        numpy array: Array of image data.
        numpy array: Array of labels corresponding to the images.
    """
    images, labels = [], []  # Initialize lists for images and labels

    # Check if the dataset directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: DATA_DIR '{DATA_DIR}' does not exist.")  # Print error if directory doesn't exist
        return np.array(images), np.array(labels)  # Return empty arrays if directory is invalid

    # Iterate through each category (subdirectory) in the dataset directory
    for category in sorted(os.listdir(DATA_DIR)):
        category_path = os.path.join(DATA_DIR, category)  # Get the path of the current category

        if not os.path.isdir(category_path):  # Skip if it's not a directory
            continue

        # Get list of image files (PNG, JPG, JPEG) within the category
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  
        if not image_files:  # If no images are found, print a warning and skip
            print(f"Warning: No images found in '{category_path}', skipping.")
            continue

        # Process each image file in the category
        for img_name in image_files:
            img_path = os.path.join(category_path, img_name)  # Get the full path of the image file
            img = cv2.imread(img_path)  # Read the image using OpenCV
            if img is None:  # If the image couldn't be read, print a warning and skip
                print(f"Warning: Unable to read '{img_path}', skipping.")
                continue
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
                img = cv2.resize(img, IMAGE_SIZE)  # Resize the image to the predefined size
                images.append(img)  # Append the image to the images list
                labels.append(category)  # Append the category (label) to the labels list
            except Exception as e:
                print(f"Error resizing '{img_path}': {e}")  # Print an error if resizing fails

    return np.array(images), np.array(labels)  # Return the loaded images and labels as numpy arrays

# Function to plot class distribution
def plot_class_distribution(y, class_names, title="Class Distribution"):
    """
    Plots the distribution of classes in the dataset.

    Args:
        y (numpy array): Array of labels.
        class_names (list): List of class names.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 4))  # Set the figure size
    sns.countplot(x=y, palette="viridis", hue=y)  # Create a count plot of the class distribution
    plt.xticks(range(len(class_names)), class_names, rotation=45)  # Set x-axis labels to class names
    plt.xlabel("Tumor Type")  # Label for the x-axis
    plt.ylabel("Number of Samples")  # Label for the y-axis
    plt.title(title)  # Set the plot title
    plt.legend(title="Tumor Types", loc='upper right')  # Add a legend to indicate class labels
    plt.show()  # Display the plot

# Function to visualize PCA of image dataset
def visualize_pca(X, y, title="PCA of Image Dataset"):
    """
    Visualizes the PCA of the dataset in 2D space.

    Args:
        X (numpy array): Image data.
        y (numpy array): Labels.
        title (str): Title for the plot.
    """
    X_flat = X.reshape(X.shape[0], -1)  # Flatten the images into 1D arrays
    pca = PCA(n_components=2)  # Initialize PCA to reduce dimensions to 2
    X_pca = pca.fit_transform(X_flat)  # Apply PCA transformation to the flattened data

    plt.figure(figsize=(8, 6))  # Set the figure size for the plot
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)  # Plot the PCA scatter plot
    plt.colorbar(scatter, label="Class Labels")  # Add color bar to indicate class labels
    plt.title(title)  # Set the plot title
    plt.show()  # Display the plot

# Function to preprocess and split data
def preprocess_data(model_name, verbose=True):
    """
    Prepares the dataset for model training by loading, preprocessing, augmenting, 
    and splitting the data into training and testing sets.

    Args:
        model_name (str): Name of the model to preprocess for (e.g., ResNet50).
        verbose (bool): Whether to display verbose output.

    Returns:
        tuple: Contains training data, test data, training labels, test labels, and class names.
    """
    X, y_raw = load_all_images()  # Load the images and labels

    if len(X) == 0 or len(y_raw) == 0:  # Check if data was loaded successfully
        print("Error: No data loaded.")  # Print error if data is empty
        return None, None, None, None, None

    encoder = LabelEncoder()  # Initialize the label encoder
    y = encoder.fit_transform(y_raw)  # Encode the labels into numerical format
    class_names = encoder.classes_  # Get the class names from the encoder

    if verbose:  # If verbose output is enabled
        print("\nDataset Summary:")
        print(f"  Total Images: {len(X)}")  # Print total number of images
        print(f"  Image Shape: {X.shape[1:]}")  # Print the shape of the images (height, width, channels)

    # Dynamically determine max class size for augmentation
    max_class_size = max(np.bincount(y))  # Find the maximum number of samples in any class
    # Balance the dataset by augmenting the classes to match the largest class size
    X_balanced, y_balanced = balance_with_augmentation(
        X, y, class_names, target_class_size=max_class_size, verbose=verbose
    )

    X_balanced = apply_model_preprocessing(X_balanced.astype("float32"), model_name)  # Apply preprocessing to the balanced dataset

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )

    if verbose:  # If verbose output is enabled
        print("Train dataset distribution:", np.bincount(y_train).tolist())  # Print the distribution of classes in the training set
        print("Test dataset distribution:", np.bincount(y_test).tolist())  # Print the distribution of classes in the testing set
        plot_class_distribution(y_train, class_names, "Training Class Distribution")  # Plot the training class distribution
        plot_class_distribution(y_test, class_names, "Testing Class Distribution")  # Plot the testing class distribution
        visualize_pca(X_train, y_train, "PCA of Training Data")  # Visualize PCA of the training data

    return X_train, X_test, y_train, y_test, class_names  # Return the processed data
