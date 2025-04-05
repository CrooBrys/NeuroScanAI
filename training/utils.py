import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot class distribution
def plot_class_distribution(y, class_names, title="Class Distribution"):
    """
    Plots the distribution of classes in the dataset.

    Args:
        y (numpy array): Array of labels.
        class_names (list): List of class names.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 4))  # Set the figure size
    sns.countplot(x=y, palette="viridis", hue=y)  # Create a count plot of the class distribution
    plt.xticks(range(len(class_names)), class_names, rotation=45)  # Set x-axis labels to class names
    plt.xlabel("Tumor Type")  # Label for the x-axis
    plt.ylabel("Number of Samples")  # Label for the y-axis
    plt.title(title)  # Set the plot title
    plt.legend(title="Tumor Types", loc='upper right')  # Add a legend to indicate class labels
    plt.show()  # Display the plot
    
def plot_accuracy_bar_with_std(data):
    """
    Plots mean validation accuracy per model with standard deviation error bars.
    Bars are color-coded, values are labeled above the bars, and layout is clean for reports.
    """
    model_names = list(data.keys())
    means = [np.mean(v) for v in data.values()]
    stds = [np.std(v) for v in data.values()]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']  # Distinct colors

    plt.figure(figsize=(8, 4))
    bars = plt.bar(model_names, means, yerr=stds, capsize=6,
                   color=colors, edgecolor='black')

    # Compute max y to adjust plot range
    max_y = max(mean + std for mean, std in zip(means, stds))
    plt.ylim(0.90, max_y + 0.01)

    # Add labels slightly above the error bar (clear of the line)
    for i, (mean, std) in enumerate(zip(means, stds)):
        label_y = mean + std + 0.004  # Give some buffer above the error bar
        plt.text(i, label_y, f"{mean:.4f}", ha='center', fontsize=10, fontweight='bold', color='black')

    plt.ylabel("Mean Validation Accuracy")
    plt.title("Model Accuracy with Standard Deviation (5-Fold CV)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()