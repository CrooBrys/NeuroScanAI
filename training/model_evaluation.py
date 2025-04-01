import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Get class labels
    class_labels = np.unique(y_test)  # Automatically extracts class names
    
    # Predict probabilities and labels
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels.astype(str)))

    # Calculate accuracy (as a single metric)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # Convert y_test to One-Hot Encoding for ROC calculation
    y_test_bin = label_binarize(y_test, classes=class_labels)

    # Compute ROC AUC Score for Multiclass (One-vs-Rest)
    roc_auc = roc_auc_score(y_test_bin, y_pred_probs, average="macro", multi_class="ovr")
    print(f"Multiclass ROC AUC Score: {roc_auc:.2f}")

    # Plot ROC Curves for Each Class
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass OvR)")
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_labels):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f"Class {class_label} (AUC = {auc(recall, precision):.2f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    # Calibration Curve
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_labels):
        prob_true, prob_pred = calibration_curve(y_test_bin[:, i], y_pred_probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, label=f"Class {class_label}")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.show()

    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(model.history.history['accuracy'], label='Training Accuracy')
    plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # Class Distribution: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.countplot(x=y_test)
    plt.title("True Class Distribution")

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_pred)
    plt.title("Predicted Class Distribution")

    plt.show()