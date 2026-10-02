# Module responsible for calculating metrics and visualizing errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluates model performance on unseen test data.
    Prints Precision, Recall, and F1-score, and exports a Confusion Matrix plot.
    """
    # Generate predictions on the 20% validation split
    predictions = model.predict(X_test)

    # Print detailed text report (Precision, Recall, F1-Score per class)
    print("=== Multi-Class Classification Report ===")
    print(classification_report(y_test, predictions, target_names=target_names))

    # Generate Confusion Matrix array to analyze prediction cross-class errors
    cm = confusion_matrix(y_test, predictions)
    
    # Plot heatmap visualization using Seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,          # Show exact numeric counts in matrix cells
        fmt="d",             # Format counts as integers
        cmap="Blues",        # Color theme
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Iris Multi-Class Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    # Save output graphic file directly to disk
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix plot to 'confusion_matrix.png'")