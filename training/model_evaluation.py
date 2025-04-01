import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Get class labels
    class_labels = np.unique(y_test)  # Automatically extracts class names
    
    # Predict probabilities and labels
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=class_labels.astype(str)))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
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
