import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

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
    
def plot_validation_accuracy_boxplot(validation_accuracies_per_model):
    """
    Plots a boxplot comparing validation accuracy distributions for multiple models.

    Args:
        validation_accuracies_per_model (dict): Dictionary with model names as keys and
                                                lists of validation accuracies as values.
    """
    plt.figure(figsize=(10, 6))
    model_names = list(validation_accuracies_per_model.keys())
    data = [validation_accuracies_per_model[name] for name in model_names]

    box = plt.boxplot(data, patch_artist=True, tick_labels=model_names)
    cmap = cm.get_cmap('tab10')

    for i, patch in enumerate(box['boxes']):
        color = cmap(i % cmap.N)
        patch.set_facecolor(color)

    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    plt.title("Model Comparison: Validation Accuracy Distribution")
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Model")
    plt.grid(True)
    plt.tight_layout()
    plt.show()