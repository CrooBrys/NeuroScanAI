import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
import numpy as np
from config import EPOCHS, BATCH_SIZE
from data_augmentation import get_data_generator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def train_model(model, X_train, y_train, X_test, y_test):
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    # Set up K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust the number of splits as needed
    fold_no = 1

    # For tracking the results across folds
    all_train_losses = []
    all_val_losses = []
    all_train_accuracies = []
    all_val_accuracies = []

    # Initialize callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    for train_index, val_index in kf.split(X_train):
        print(f"\nTraining Fold {fold_no}...")

        # Split data into train and validation for this fold
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Data Augmentation using the imported function
        datagen = get_data_generator()
        datagen.fit(X_train_fold)

        # Compile model with Adam optimizer and weight decay
        optimizer = Adam(learning_rate=1e-3, weight_decay=1e-4)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model with class weights, data augmentation, and callbacks
        history = model.fit(datagen.flow(X_train_fold, y_train_fold, batch_size=BATCH_SIZE),
                            validation_data=(X_val_fold, y_val_fold),
                            epochs=EPOCHS,
                            class_weight=class_weights,
                            callbacks=[early_stopping, lr_scheduler, model_checkpoint])

        # Save history of the current fold
        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])
        all_train_accuracies.append(history.history['accuracy'])
        all_val_accuracies.append(history.history['val_accuracy'])

        fold_no += 1

    # Calculate the mean loss and accuracy across all folds
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracies, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracies, axis=0)

    # Plot loss and accuracy curves for K-fold results
    plt.figure(figsize=(12, 5))

    # Plot Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Average Loss Curve Across Folds')

    # Plot Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(avg_train_accuracy, label='Train Accuracy')
    plt.plot(avg_val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Average Accuracy Curve Across Folds')

    plt.show()

    return model
