from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('confusion.csv')

    # Calculate the confusion matrix
    confusion = confusion_matrix(df['True'], df['Predicted'])

    # Plot the confusion matrix
    plt.figure(figsize=(20, 14))
    sns.heatmap(confusion, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
