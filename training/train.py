import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

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

def train_k_fold(X_train, y_train, model_name, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(model_name)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    n_splits = 2
    kf = KFold(n_splits, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"Training fold {fold + 1}/{n_splits} for {model_name}")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_fold, y_train_fold, epochs=1, batch_size=32,
                  validation_data=(X_val_fold, y_val_fold))

        y_pred_val = model.predict(X_val_fold)
        y_val_bin = LabelBinarizer().fit_transform(y_val_fold)
        fold_auc = roc_auc_score(y_val_bin, y_pred_val, multi_class='ovr')
        auc_scores.append(fold_auc)

    avg_roc_auc = np.mean(auc_scores)
    print(f"Average ROC AUC for {model_name}: {avg_roc_auc:.4f}")

    final_model_path = os.path.join(save_dir, f"final_trained_{model_name}.keras")
    model.save(final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    return model, avg_roc_auc