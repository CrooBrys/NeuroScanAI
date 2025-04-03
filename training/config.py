# Path to the dataset directory  
DATA_DIR = "data"  

# Standardized image size for model input (width, height)  
IMAGE_SIZE = (224, 224)  

# Number of images processed per batch during training  
BATCH_SIZE = 32  

# Number of training epochs
EPOCHS = 2  

# Number of folds for K-Fold Cross-Validation 
K_FOLDS = 2

# List of CNN models used for feature extraction and classification  
MODELS = ["ResNet50", "VGG16", "EfficientNetB0", "InceptionV3"]  
