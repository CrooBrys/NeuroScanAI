# Import necessary libraries
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout

# Function to get a feature extractor model with optional fine-tuning
def get_feature_extractor(model_name, fine_tune=False, unfreeze_layers=10, input_shape=(224, 224, 3)):
    """
    Retrieves a feature extraction model based on the selected pre-trained model.
    Optionally enables fine-tuning by unfreezing the last 'unfreeze_layers' layers.

    Args:
        model_name (str): The name of the pre-trained model to use ("ResNet50", "VGG16", "EfficientNetB0", "InceptionV3").
        fine_tune (bool): Whether to fine-tune the model by unfreezing the last layers. Default is False.
        unfreeze_layers (int): The number of layers to unfreeze from the end for fine-tuning. Default is 10.
        input_shape (tuple): The shape of input images, default is (224, 224, 3).

    Returns:
        tensorflow.keras.Model: A model object with the feature extractor and a custom classification head.
    """
    # Select the pre-trained model based on the given model name
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model: {model_name}")  # Error handling for unsupported models

    # Freeze all layers initially to prevent updates during training
    base_model.trainable = False

    # If fine-tuning is enabled, unfreeze the last 'unfreeze_layers' layers for training
    if fine_tune:
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True  # Unfreeze the selected layers for training

        # Print the number of trainable layers for debugging and tracking progress
        trainable_count = sum([layer.trainable for layer in base_model.layers])
        print(f"Fine-tuning enabled: {unfreeze_layers} layers unfrozen. Total trainable layers: {trainable_count}")

    # Add a custom classification head on top of the pre-trained base model
    x = Flatten()(base_model.output)  # Flatten the output of the base model for input into the dense layers
    x = Dense(128, activation='relu')(x)  # Fully connected layer with 128 units and ReLU activation
    x = BatchNormalization()(x)  # Normalize activations to improve training stability
    x = Dropout(0.5)(x)  # Dropout to reduce overfitting by randomly setting 50% of units to zero
    x = Dense(4, activation='softmax')(x)  # Output layer with 4 units for 4-class classification (softmax for multi-class)

    # Return the complete model (base model + custom classification head)
    return Model(inputs=base_model.input, outputs=x)  # Construct the model and return it
