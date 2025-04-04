# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report, matthews_corrcoef, cohen_kappa_score

# Function to evaluate a single model on a test set with detailed metrics and visualizations
def evaluate_single_model(model, X_test, y_test, class_names):
    """
    Evaluates a model on a test set, providing classification report, accuracy, confusion matrix, and ROC/PR curves.

    Args:
        model: The trained model to evaluate.
        X_test (numpy array): Test data for evaluation.
        y_test (numpy array): True labels corresponding to the test data.
        class_names (list): List of class names corresponding to the model output.

    Prints evaluation results including:
        - Classification report (precision, recall, F1-score, and support)
        - Accuracy score
        - Confusion matrix (regular and normalized)
        - ROC and Precision-Recall curves for each class
        - Matthews Correlation Coefficient (MCC) and Cohen's Kappa score
    """
    # Get the predicted probabilities and convert them to class predictions
    y_pred_probs = model.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=1)

    # Print classification report including precision, recall, f1-score, and support for each class
    print(f"\n---\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"---\n")
    
    # Calculate and print accuracy of the model
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"\n---\n")
    
    # Compute confusion matrix and plot it as a heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Normalize confusion matrix and plot it
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # Binarize the true labels for multiclass ROC AUC calculation
    class_indices = np.arange(len(class_names))
    y_test_bin = label_binarize(y_test, classes=class_indices)

    # Calculate and print ROC AUC score for multiclass classification
    roc_auc = roc_auc_score(y_test_bin, y_pred_probs, average="macro", multi_class="ovr")
    print(f"---\n")
    print(f"Multiclass ROC AUC Score: {roc_auc:.2f}")

    # Plot ROC curve for each class and display AUC values
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_label} (AUC = {auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classifier
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass OvR)")
    plt.legend()
    plt.show()
    print(f"---\n")
    
    # Plot precision-recall curve for each class and display AUC values
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, label=f"{class_label} (AUC = {auc(recall, precision):.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    print(f"---\n")

    # Calculate additional metrics
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"MCC: {mcc:.2f}")
    print(f"\n---\n")

    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen's Kappa: {kappa:.2f}")

# Function to compare multiple models and present their performance in a formatted table
def compare_models(trained_models):
    """
    Compares the performance of multiple trained models and presents the results in a grouped DataFrame.

    Args:
        trained_models (dict): Dictionary where keys are model names and values are the corresponding model data.

    Returns:
        pd.DataFrame: Multi-index column DataFrame with performance metrics grouped by model.
    """
    all_results = {}  # Dictionary to hold results
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

        # Add summary rows
        accuracy = np.mean(y_pred == data["y_test"])
        macro_precision = precision_score(data["y_test"], y_pred, average="macro")
        macro_recall = recall_score(data["y_test"], y_pred, average="macro")
        macro_f1 = f1_score(data["y_test"], y_pred, average="macro")

        per_class_df.loc["Average"] = [macro_precision, macro_recall, macro_f1]
        per_class_df.loc["Accuracy"] = [accuracy, np.nan, np.nan]

        # Create MultiIndex columns: (model_name, metric)
        per_class_df.columns = pd.MultiIndex.from_product([[model_name], per_class_df.columns])
        all_results[model_name] = per_class_df

    # Concatenate all models horizontally
    combined_df = pd.concat(all_results.values(), axis=1)

    # Formatting
    combined_df = combined_df.round(2)
    combined_df.fillna("", inplace=True)
    combined_df.index.name = "CLASS"

    return combined_df