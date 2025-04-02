from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense

def get_feature_extractor(model_name, fine_tune=False, unfreeze_layers=10):
    if model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Freeze all layers initially
    base_model.trainable = False

    # Unfreeze the top `unfreeze_layers` if fine-tuning
    if fine_tune:
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True

    # Add custom classification head
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=x)