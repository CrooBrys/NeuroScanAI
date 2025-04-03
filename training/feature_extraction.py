# Import pre-trained models from TensorFlow Keras applications for feature extraction
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

# Function to get a feature extractor model with an optional fine-tuning mechanism
# Returns a Keras model with a custom classification head added on top of the selected pre-trained model
def get_feature_extractor(model_name, fine_tune=False, unfreeze_layers=10):
    # Select the pre-trained model based on the given model name
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unsupported model: {model_name}")  # Error handling for unsupported models

    # Freeze all layers initially to prevent them from being updated during training
    base_model.trainable = False

    # If fine-tuning is enabled, unfreeze the top 'unfreeze_layers' layers for training
    if fine_tune:
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True

    # Add a custom classification head on top of the pre-trained base model
    x = Flatten()(base_model.output)  # Flatten the output of the base model to feed into dense layers
    x = Dense(128, activation='relu')(x)  # Add a fully connected layer with ReLU activation
    x = Dense(4, activation='softmax')(x)  # Final softmax layer for multi-class classification (4 classes)

    # Return the complete model, consisting of the base model + custom classification head
    return Model(inputs=base_model.input, outputs=x)
