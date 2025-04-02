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
    """Compare multiple models using accuracy, F1-score, precision, recall, ROC AUC, and Precision-Recall AUC, and rank them based on validation accuracy in a table."""
    results = []
    print(f"Evaluating Models...")
    for model_name, data in trained_models.items():
        
        # Predict probabilities and labels
        y_pred_probs = data["model"].predict(data["X_test"])
        y_pred = y_pred_probs.argmax(axis=1)
        
        # Binarize the labels for ROC AUC computation
        y_test_bin = label_binarize(data["y_test"], classes=np.arange(len(data["class_names"])))
        
        # Calculate metrics
        accuracy = np.mean(y_pred == data["y_test"])
        f1 = f1_score(data["y_test"], y_pred, average="macro")
        roc_auc = roc_auc_score(y_test_bin, y_pred_probs, average="macro", multi_class="ovr")
        precision = precision_score(data["y_test"], y_pred, average="macro")
        recall = recall_score(data["y_test"], y_pred, average="macro")
        
        # Precision-Recall AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_test_bin.ravel(), y_pred_probs.ravel())
        precision_recall_auc = auc(recall_vals, precision_vals)
        
        # Append results to the list
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "Precision-Recall AUC": precision_recall_auc
        })
    
    # Convert results to a pandas DataFrame
    results_df = pd.DataFrame(results)

    # Sort the models by accuracy in descending order for ranking
    results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    # Display the table
    print("\nModel Comparison Table:")
    
    return results_df