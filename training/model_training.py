import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import numpy as np
from config import EPOCHS, BATCH_SIZE, K_FOLDS
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

def train_model(model, model_name, X_train, y_train):
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Stratified K-fold cross-validation
    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_no = 1

    # For tracking the results across folds
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    # Callbacks (reused across folds)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Compile model
    optimizer = Adam(learning_rate=1e-3, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    best_val_accuracy = 0.0
    best_model_weights = None

    # Perform K-Fold training
    for train_index, val_index in kf.split(X_train, y_train):
        print(f"\nTraining Fold {fold_no}...")

        # Get current fold's train/val splits
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train the model
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        # Store metrics for plotting later
        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])
        all_train_accuracies.append(history.history['accuracy'])
        all_val_accuracies.append(history.history['val_accuracy'])

        # Check if this fold has the best validation accuracy
        val_acc = history.history['val_accuracy'][-1]
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_weights = model.get_weights()

        fold_no += 1

    # Load best model weights and save
    model.set_weights(best_model_weights)
    model_path = os.path.join("models", f"{model_name}.keras")
    model.save(model_path)
    print(f"\nBest model across folds saved to: {model_path}")

    # Calculate mean metrics across folds
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

    # Plot training and validation curves
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Average Loss Curve Across Folds')

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracy, label='Train Accuracy')
    plt.plot(avg_val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Average Accuracy Curve Across Folds')

    plt.tight_layout()
    plt.show()

    return model