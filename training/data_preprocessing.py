# Import necessary libraries for image processing, model training, and evaluation
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import DATA_DIR, IMAGE_SIZE
from data_augmentation import balance_with_augmentation
from tensorflow.keras.applications import resnet50, vgg16, efficientnet, inception_v3

# Function to apply model-specific preprocessing to the input images
# Preprocessing steps are specific to the selected model architecture
def apply_model_preprocessing(X, model_name):
    # Check which model is selected and apply the respective preprocessing
    if model_name == "ResNet50":
        return resnet50.preprocess_input(X)  # Preprocess for ResNet50
    elif model_name == "VGG16":
        return vgg16.preprocess_input(X)  # Preprocess for VGG16
    elif model_name == "EfficientNetB0":
        return efficientnet.preprocess_input(X)  # Preprocess for EfficientNetB0
    elif model_name == "InceptionV3":
        return inception_v3.preprocess_input(X)  # Preprocess for InceptionV3
    else:
        return X  # If no known model is selected, return the input as-is

# Function to load images and their associated labels from the dataset directory
# Resizes images to the required input size for the model
def load_all_images():
    images, labels = [], []
    
    # Check if the specified data directory exists
    if not os.path.exists(DATA_DIR):  
        print(f"Error: DATA_DIR '{DATA_DIR}' does not exist.")
        return np.array(images), np.array(labels)

    # Iterate over category folders in the dataset
    for category in sorted(os.listdir(DATA_DIR)):
        category_path = os.path.join(DATA_DIR, category)
        
        # Skip non-directories
        if not os.path.isdir(category_path):
            continue

        # Find image files in the category folder
        image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  
        if not image_files:  # If no images are found, skip the folder
            print(f"Warning: No images found in '{category_path}', skipping.")
            continue

        # Loop through each image file
        for img_name in image_files:  
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)  # Read the image file
            if img is None:  # If image cannot be read, skip
                print(f"Warning: Unable to read '{img_path}', skipping.")
                continue
            try:
                # Resize the image to the required input size
                img = cv2.resize(img, IMAGE_SIZE)  
                images.append(img)
                labels.append(category)  # Store the corresponding label
            except Exception as e:
                # Handle potential resizing errors
                print(f"Error resizing '{img_path}': {e}")
    
    # Return the loaded and resized images along with their labels as NumPy arrays
    return np.array(images), np.array(labels)  

# Function to visualize and compare the effects of different model-specific preprocessing
# This helps to evaluate how different models alter the image
def show_model_preprocessing_comparison(image, title="image(2).jpg"):
    img_array = image.astype('float32')  # Convert image to float32 for processing
    
    # Create a dictionary with each model's preprocessing applied to the image
    models = {
        "Original": img_array,  
        "ResNet50": resnet50.preprocess_input(img_array.copy()),
        "VGG16": vgg16.preprocess_input(img_array.copy()),
        "InceptionV3": inception_v3.preprocess_input(img_array.copy()),
        "EfficientNetB0": efficientnet.preprocess_input(img_array.copy()),
    }

    # Create subplots for visualizing the effect of preprocessing for each model
    fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
    for ax, (name, processed) in zip(axes, models.items()):
        # Normalize the processed image to a viewable range if not the original
        disp_img = processed.copy()
        if name != "Original":
            disp_img = ((disp_img - disp_img.min()) / (disp_img.max() - disp_img.min()) * 255).astype("uint8")
        ax.imshow(disp_img.astype("uint8"))  # Display the processed image
        ax.set_title(name)  # Title of the subplot
        ax.axis("off")  # Hide axes

    plt.suptitle(f"Model Preprocessing Comparison - {title}")  # Add title to the figure
    plt.tight_layout()
    plt.show()

    # Print the mean and standard deviation of pixel values after preprocessing
    print("Mean pixel values after preprocessing:")
    for name, processed in models.items():
        mean = processed.mean()  # Calculate mean pixel value
        std = processed.std()  # Calculate standard deviation of pixel values
        print(f"  {name}: mean = {mean:.2f}, std = {std:.2f}")

# Function to prepare data by loading, augmenting, preprocessing, and splitting into training and testing sets
def preprocess_data(model_name, verbose=True):
    # Load the images and labels from the dataset
    X, y_raw = load_all_images()
    
    # If no data is loaded, return None
    if len(X) == 0 or len(y_raw) == 0:
        print("Error: No data loaded.")
        return None, None, None, None, None

    # Encode string labels into numerical values for model compatibility
    encoder = LabelEncoder()  
    y = encoder.fit_transform(y_raw)  
    class_names = encoder.classes_  # Get the class names for reference

    if verbose:  # Optionally print the class mapping for better understanding
        print("Class mapping:")
        for i, label in enumerate(class_names):
            print(f"  {i} = {label}")
        print()

    if verbose and len(X) > 2:  # Optionally visualize preprocessing on a sample image
        print("Visualizing model-specific preprocessing on sample image (Glioma image[2]):")
        show_model_preprocessing_comparison(X[2], title="image(2).jpg")

    # Apply class balancing and augmentation to handle imbalanced datasets
    X_balanced, y_balanced = balance_with_augmentation(
        X, y, target_class_size=None, verbose=verbose, samples_per_class=2
    )

    # Apply model-specific preprocessing to the augmented and balanced data
    X_balanced = apply_model_preprocessing(X_balanced.astype('float32'), model_name)

    # Split the data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, stratify=y_balanced, random_state=42
    )

    if verbose:  # Optionally print class distribution in training and testing sets
        print("Train dataset distribution:", np.bincount(y_train).tolist())
        print("Test dataset distribution:", np.bincount(y_test).tolist())

    # Return the training and testing data, along with class names for reference
    return X_train, X_test, y_train, y_test, class_names
