import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

CLASSES = ["Glioma", "Meningioma", "Pituitary", "None"]

# Build Model
def build_model(base_model_name="ResNet50"):
    if base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif base_model_name == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    base_model.trainable = False  # Freeze base model

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# K-fold Cross-Validation and Hyperparameter Tuning
def train_k_fold(X_train, y_train, model_name):
    model = build_model(model_name)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Training fold {fold + 1}/5 for {model_name}")
        
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        checkpoint = ModelCheckpoint(f"trained_{model_name}_fold_{fold + 1}.h5", save_best_only=True)
        
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[checkpoint])
        
        model.load_weights(f"trained_{model_name}_fold_{fold + 1}.h5")
        print(f"Model saved for fold {fold + 1}")

    model.save(f"final_trained_{model_name}.h5")
    return model
