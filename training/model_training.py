# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from config import EPOCHS, BATCH_SIZE, K_FOLDS

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
    Trains the given model using K-Fold cross-validation, stores the best model, and generates training/validation plots.

    Args:
        model (tensorflow.keras.Model): The model to train.
        model_name (str): Name of the model, used for saving the best model.
        X_train (numpy array): Training data (images).
        y_train (numpy array): Labels for the training data.

    Returns:
        tensorflow.keras.Model: The trained model with the best weights.
        list: A list containing the validation accuracies from each fold.
    """
    os.makedirs("models", exist_ok=True)  # Create the "models" directory if it doesn't exist

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)  # Setup for K-Fold cross-validation
    fold_no = 1  # Start with the first fold
    best_val_accuracy = 0.0  # Track the best validation accuracy achieved
    best_model_weights = None  # Store the best model weights

    # Initialize lists to store losses and accuracies across folds
    all_train_losses, all_val_losses = [], []
    all_train_accuracies, all_val_accuracies = [], []
    fold_val_accuracies = []  # To store validation accuracies for each fold

    # Set up callbacks to improve training performance
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Stop training if validation loss doesn't improve
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)  # Reduce learning rate if validation loss plateaus
    model_checkpoint = ModelCheckpoint(f"models/{model_name}_best.keras", save_best_only=True, monitor="val_accuracy", mode="max")  # Save the best model based on validation accuracy

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weights(y_train)

    # Loop through the K-Folds for training
    for train_index, val_index in kf.split(X_train, y_train):
        print(f"\nTraining Fold {fold_no}...")

        # Split data into training and validation sets for the current fold
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Compile the model with the Adam optimizer and loss function
        model.compile(
            optimizer=Adam(learning_rate=1e-3, weight_decay=1e-4),  # Optimizer with learning rate and weight decay
            loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
            metrics=['accuracy']  # Evaluation metric for accuracy
        )

        # Fit the model on the current fold's data
        history = model.fit(
            X_train_fold, y_train_fold,  # Training data and labels
            validation_data=(X_val_fold, y_val_fold),  # Validation data and labels
            epochs=EPOCHS,  # Number of epochs for training
            batch_size=BATCH_SIZE,  # Batch size for training
            class_weight=class_weights,  # Class weights to handle class imbalance
            callbacks=[early_stopping, lr_scheduler, model_checkpoint, tensorboard_callback],  # Callbacks for training
            verbose=1  # Display training progress
        )

        # Store training and validation losses and accuracies
        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])
        all_train_accuracies.append(history.history['accuracy'])
        all_val_accuracies.append(history.history['val_accuracy'])

        # Track the highest validation accuracy achieved
        val_acc = max(history.history['val_accuracy'])
        fold_val_accuracies.append(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc  # Update the best validation accuracy
            best_model_weights = model.get_weights()  # Save the best model weights

        fold_no += 1  # Move to the next fold

    # Set the model weights to the best weights found during training
    model.set_weights(best_model_weights)
    final_model_path = f"models/{model_name}.keras"  # Path to save the final model
    model.save(final_model_path)  # Save the final model
    print(f"\nBest model saved to: {final_model_path}")

    # Calculate and plot average training and validation losses/accuracies
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

    # Plot training and validation results
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

    return model, fold_val_accuracies  # Return the trained model and validation accuracies for each fold
