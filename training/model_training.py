# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from config import EPOCHS, BATCH_SIZE, K_FOLDS

# Enable mixed precision if GPU supports it (optional, speeds up training)
if tf.config.experimental.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Function to compute class weights for handling class imbalance
def compute_class_weights(y_train):
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    return class_weights

# Function to train the model using K-Fold cross-validation
def train_model(model, model_name, X_train, y_train):
    os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_no = 1
    best_val_accuracy = 0.0
    best_model_weights = None

    all_train_losses, all_val_losses = [], []
    all_train_accuracies, all_val_accuracies = [], []
    fold_val_accuracies = []

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(f"models/{model_name}_best.keras", save_best_only=True, monitor="val_accuracy", mode="max")

    # Compute class weights for imbalance handling
    class_weights = compute_class_weights(y_train)

    for train_index, val_index in kf.split(X_train, y_train):
        print(f"\nTraining Fold {fold_no}...")

        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Set up TensorBoard logging
        log_dir = f"logs/{model_name}_fold_{fold_no}"
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

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
            callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard_callback],
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

    model.set_weights(best_model_weights)
    final_model_path = f"models/{model_name}.keras"
    model.save(final_model_path)
    print(f"\nBest model saved to: {final_model_path}")

    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

    # Plot training results
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
