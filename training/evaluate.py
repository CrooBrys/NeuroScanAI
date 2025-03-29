import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

CLASSES = ["Glioma", "Meningioma", "Pituitary", "None"]

# Function to evaluate the model on test data
def evaluate_model(model_name, X_test, y_test, model_dir="saved_models"):
    model_path = os.path.join(model_dir, f"final_trained_{model_name}.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = load_model(model_path)
    
    y_pred_test = model.predict(X_test)
    y_pred_class_test = np.argmax(y_pred_test, axis=1)

    print(f"Classification Report for {model_name} on Test Set:")
    print(classification_report(y_test, y_pred_class_test, target_names=CLASSES))

    cm = confusion_matrix(y_test, y_pred_class_test)
    print(f"Confusion Matrix for {model_name}:")
    print(cm)

    # ROC Curve and AUC
    y_test_bin = LabelBinarizer().fit_transform(y_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_test.ravel())
    roc_auc = auc(fpr, tpr)
    print(f"AUC for {model_name}: {roc_auc}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

    # Plot ROC Curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc='lower right')
    plt.show()