from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_balanced_data_generator(X_train, y_train, batch_size):
    """Creates a data generator with equalized class distribution."""
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    # Find the max class count to oversample the smaller classes
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_samples = np.max(class_counts)

    # Create balanced dataset by oversampling
    balanced_X, balanced_y = [], []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        oversampled_indices = np.random.choice(cls_indices, max_samples, replace=True)
        balanced_X.extend(X_train[oversampled_indices])
        balanced_y.extend(y_train[oversampled_indices])

    balanced_X = np.array(balanced_X)
    balanced_y = np.array(balanced_y)

    print("Balanced Data Shape:", balanced_X.shape, balanced_y.shape)

    return datagen.flow(balanced_X, balanced_y, batch_size=batch_size, shuffle=True)
