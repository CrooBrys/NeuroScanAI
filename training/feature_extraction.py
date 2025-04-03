# Import pre-trained models from TensorFlow Keras applications for feature extraction
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout

# Function to get a feature extractor model with optional fine-tuning
def get_feature_extractor(model_name, fine_tune=False, unfreeze_layers=10, input_shape=(224, 224, 3)):
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
            layer.trainable = True

        # Print the number of trainable layers for debugging
        trainable_count = sum([layer.trainable for layer in base_model.layers])
        print(f"Fine-tuning enabled: {unfreeze_layers} layers unfrozen. Total trainable layers: {trainable_count}")

    # Add a custom classification head on top of the pre-trained base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)  # Normalize activations for stability
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    x = Dense(4, activation='softmax')(x)  # Output layer for 4-class classification

    # Return the complete model
    return Model(inputs=base_model.input, outputs=x)
