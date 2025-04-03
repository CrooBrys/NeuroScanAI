import matplotlib.pyplot as plt
import seaborn as sns

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