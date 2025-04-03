# Import necessary libraries
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import numpy as np
from config import EPOCHS, BATCH_SIZE, K_FOLDS
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Function to train the model with K-Fold cross-validation
def train_model(model, model_name, X_train, y_train):
    # Ensure the models directory exists where we will save the best model
    os.makedirs("models", exist_ok=True)

    # Initialize Stratified K-Fold cross-validation
    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)  # Split dataset into K_FOLDS
    fold_no = 1  # Start fold numbering

    # Lists to store metrics for tracking and plotting across folds
    all_train_losses = []  # Training losses
    all_val_losses = []  # Validation losses
    all_train_accuracies = []  # Training accuracies
    all_val_accuracies = []  # Validation accuracies
    fold_val_accuracies = []  # Best validation accuracies for each fold

    # Define training callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Compile the model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric
    optimizer = Adam(learning_rate=1e-3, weight_decay=1e-4)  # Optimizer setup
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    best_val_accuracy = 0.0  # Initialize the best validation accuracy
    best_model_weights = None  # To store the best weights during training

    # Perform K-Fold training
    for train_index, val_index in kf.split(X_train, y_train):
        print(f"\nTraining Fold {fold_no}...")

        # Get the current fold's training and validation data
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Train the model on the current fold's data
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, lr_scheduler],  # Apply early stopping and learning rate reduction
            verbose=1  # Print training progress
        )

        # Store training and validation losses/accuracies for plotting later
        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])
        all_train_accuracies.append(history.history['accuracy'])
        all_val_accuracies.append(history.history['val_accuracy'])

        # Track the best validation accuracy across epochs in this fold
        val_acc = max(history.history['val_accuracy'])
        fold_val_accuracies.append(val_acc)

        # Update the best model weights if current fold has better validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_weights = model.get_weights()

        fold_no += 1  # Increment fold number

    # After all folds, restore the best model weights and save the best model
    model.set_weights(best_model_weights)
    model_path = os.path.join("models", f"{model_name}.keras")
    model.save(model_path)
    print(f"\nBest model across folds saved to: {model_path}")

    # Calculate mean training/validation losses and accuracies across all folds
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

    # Plot average loss and accuracy curves for training and validation across folds
    plt.figure(figsize=(14, 6))

    # Loss curve plot
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss', marker='o')
    plt.plot(avg_val_loss, label='Validation Loss', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f'{model_name} - Average Loss Across Folds')
    plt.legend()
    plt.grid(True)

    # Accuracy curve plot
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracy, label='Train Accuracy', marker='o')
    plt.plot(avg_val_accuracy, label='Validation Accuracy', marker='x')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f'{model_name} - Average Accuracy Across Folds')
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"{model_name} - K-Fold Training Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    plt.show()

    return model, fold_val_accuracies  # Return the model with best weights and per-fold validation accuracies
