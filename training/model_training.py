# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import EPOCHS, BATCH_SIZE, K_FOLDS
from tensorflow.keras.callbacks import TensorBoard

# Enable mixed precision if GPU supports it (optional, speeds up training)
if tf.config.experimental.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Function to compute class weights for handling class imbalance
def compute_class_weights(y_train):
    """
    Computes class weights to handle class imbalance by giving more weight to less frequent classes.

    Args:
        y_train (numpy array): Array of class labels for training data.

    Returns:
        dict: A dictionary where keys are class labels and values are corresponding class weights.
    """
    class_counts = np.bincount(y_train)  # Count occurrences of each class in the training set
    total_samples = len(y_train)  # Total number of samples in the training data
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}  # Compute weights inversely proportional to class frequency
    return class_weights

# Function to train the model using K-Fold cross-validation
def train_model(model, model_name, X_train, y_train):
    """
    Trains the given model using K-Fold cross-validation, saves the best-performing model,
    generates average training/validation plots, and returns fold-wise performance.

    Args:
        model (tensorflow.keras.Model): The model to train.
        model_name (str): Filename identifier for saving the best model.
        X_train (numpy array): Input image data.
        y_train (numpy array): Corresponding class labels.

    Returns:
        tensorflow.keras.Model: The trained model with the best weights.
        list: Validation accuracy from each fold.
    """
    os.makedirs("models", exist_ok=True)

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    best_val_accuracy = 0.0
    best_model_weights = None

    all_train_losses, all_val_losses = [], []
    all_train_accuracies, all_val_accuracies = [], []
    fold_val_accuracies = []

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint_path = f"models/{model_name}.keras"
    model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max")

    # Compute class weights for imbalance handling
    class_weights = compute_class_weights(y_train)

    # K-Fold training loop
    for train_index, val_index in kf.split(X_train, y_train):
        print(f"\nTraining Fold {fold_no}...")

        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model.compile(
            optimizer=Adam(learning_rate=1e-3, weight_decay=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=[early_stopping, lr_scheduler, model_checkpoint],
            verbose=1
        )

        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])
        all_train_accuracies.append(history.history['accuracy'])
        all_val_accuracies.append(history.history['val_accuracy'])

        val_acc = max(history.history['val_accuracy'])
        fold_val_accuracies.append(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_weights = model.get_weights()

        fold_no += 1

    # Load the best weights into the model
    model.set_weights(best_model_weights)

    # Reload from saved file to ensure the returned model is consistent with what was saved
    model = tf.keras.models.load_model(checkpoint_path)
    print(f"\nBest model saved to: {checkpoint_path}")

    def pad_with_nan(sequences, maxlen):
        return np.array([
            np.pad(seq, (0, maxlen - len(seq)), constant_values=np.nan)
            for seq in sequences
        ])

    max_len = max(len(x) for x in all_train_losses)

    all_train_losses_padded = pad_with_nan(all_train_losses, max_len)
    all_val_losses_padded = pad_with_nan(all_val_losses, max_len)
    all_train_accuracies_padded = pad_with_nan(all_train_accuracies, max_len)
    all_val_accuracies_padded = pad_with_nan(all_val_accuracies, max_len)

    avg_train_loss = np.nanmean(all_train_losses_padded, axis=0)
    avg_val_loss = np.nanmean(all_val_losses_padded, axis=0)
    avg_train_accuracy = np.nanmean(all_train_accuracies_padded, axis=0)
    avg_val_accuracy = np.nanmean(all_val_accuracies_padded, axis=0)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss', marker='o')
    plt.plot(avg_val_loss, label='Validation Loss', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f'{model_name} - Average Loss Across Folds')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracy, label='Train Accuracy', marker='o')
    plt.plot(avg_val_accuracy, label='Validation Accuracy', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f'{model_name} - Average Accuracy Across Folds')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"{model_name} - K-Fold Training Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return model, fold_val_accuracies