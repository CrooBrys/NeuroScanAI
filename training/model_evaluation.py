import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report

def evaluate_single_model(model, X_test, y_test, class_names):
    """Evaluate a single model with detailed metrics and visualizations."""
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    class_indices = np.arange(len(class_names))
    y_test_bin = label_binarize(y_test, classes=class_indices)

    roc_auc = roc_auc_score(y_test_bin, y_pred_probs, average="macro", multi_class="ovr")
    print(f"Multiclass ROC AUC Score: {roc_auc:.2f}")

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass OvR)")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f"{class_label} (AUC = {auc(recall, precision):.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

def compare_models(trained_models):
    """Compare models and format results similar to the provided table."""
    
    all_results = {}
    print("Evaluating Models...\n")

    for model_name, data in trained_models.items():
        y_pred_probs = data["model"].predict(data["X_test"])
        y_pred = y_pred_probs.argmax(axis=1)
        y_test_bin = label_binarize(data["y_test"], classes=np.arange(len(data["class_names"])))

        per_class_metrics = []

        for class_idx in range(len(data["class_names"])):
            class_precision = precision_score(data["y_test"], y_pred, labels=[class_idx], average="macro", zero_division=0)
            class_recall = recall_score(data["y_test"], y_pred, labels=[class_idx], average="macro", zero_division=0)
            class_f1 = f1_score(data["y_test"], y_pred, labels=[class_idx], average="macro", zero_division=0)
            per_class_metrics.append([class_precision, class_recall, class_f1])

        per_class_df = pd.DataFrame(per_class_metrics, columns=["Precision", "Recall", "F1-score"])

        accuracy = np.mean(y_pred == data["y_test"])
        macro_precision = precision_score(data["y_test"], y_pred, average="macro")
        macro_recall = recall_score(data["y_test"], y_pred, average="macro")
        macro_f1 = f1_score(data["y_test"], y_pred, average="macro")

        per_class_df.loc["Average"] = [macro_precision, macro_recall, macro_f1]
        per_class_df.loc["Accuracy"] = [accuracy, np.nan, np.nan]

        all_results[model_name] = per_class_df

    # Combine into one DataFrame
    combined_df = pd.concat(all_results, axis=1)
    
    # Formatting
    combined_df = combined_df.round(2)  # Round to match visual style
    combined_df.fillna("", inplace=True)  # Remove NaNs for a clean table
    combined_df.index.name = "CLASS"  # Label index

    return combined_df