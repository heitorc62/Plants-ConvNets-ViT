import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
import joblib
import argparse

def make_path(path):
    dir = os.path.dirname(path)
    if dir: 
        if not os.path.exists(dir):
            os.makedirs(dir)

def save_model(output_path, model):
    make_path(output_path)
    joblib.dump(model, output_path + "/random_forest_model.joblib")
    
def save_report(output_path, report):
    make_path(output_path)
    with open(output_path + "/random_forest_report.txt", 'w') as f:
        f.write(report)
    

def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    label_names = os.listdir(dataset_dir)

    for label in label_names:
        class_dir = os.path.join(dataset_dir, label)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = Image.open(image_path)
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((64, 64))  # Resize to a standard size
            images.append(np.array(image).flatten())  # Flatten the image
            labels.append(label)

    return np.array(images), np.array(labels)

def main(dataset_path, output_path):
    # Load the dataset
    X, y = load_images_and_labels(dataset_path)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # Evaluate the classifier
    accuracy = clf.score(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    save_model(output_path, clf)
    
    report = classification_report(y_test, y_pred, target_names=np.unique(y_test))
    save_report(output_path, report)
    
    
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--output_path', type=str, help='')
    # Parse the arguments
    args = parser.parse_args()
    main(args.dataset_path, args.output_path)

    
