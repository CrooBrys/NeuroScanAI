# Path to the dataset directory  
DATA_DIR = "data"  

# Standardized image size for model input (width, height)  
IMAGE_SIZE = (224, 224)  

# Number of images processed per batch during training  
BATCH_SIZE = 32  

# Number of training epochs
EPOCHS = 10

# Number of folds for K-Fold Cross-Validation 
K_FOLDS = 5

# List of CNN models used for feature extraction and classification  
MODELS = ["ResNet50", "VGG16", "EfficientNetB0", "InceptionV3"]  

# Number of threads used within individual TensorFlow operations  
INTRA_THREADS = 12  # Controls the parallelism of matrix computations, convolutions, etc. inside operations

# Number of threads used between independent TensorFlow operations  
INTER_THREADS = 4  # Controls how many operations can run concurrently across CPU cores
