from tsnecuda import TSNE
import argparse, cv2, os
import numpy as np


def normalize_image(image):
    # Convert image to float32 format
    image = image.astype(np.float32) / 255.0
    # Subtract mean
    mean = [0.485, 0.456, 0.406]
    image -= mean
    # Divide by standard deviation
    std = [0.229, 0.224, 0.225]
    image /= std
    
    return image

def load_PlantVillage(dataset_path):
    # Load images and labels
    images = []
    labels = []

    # Iterate over each class folder
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                # Load and preprocess the image
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))
                image = normalize_image(image)
                images.append(image)
                labels.append(class_folder)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    target_names = np.unique(labels)
    return {'X': images, 'y': labels, 'target_names': target_names}


def plot_embedding(X, target_names, title="t-SNE embedding of the PlantVillage dataset"):
    pass



def main(data_path, perplexity_value):
    data = load_PlantVillage(data_path)
    X = data['X']
    X_embedded = TSNE(n_components=2, perplexity=perplexity_value, learning_rate=10).fit_transform(X)
    plot_embedding(X_embedded, data['target_names'])
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot t-SNE of PlantVillage dataset')

    parser.add_argument('--perplexity', type=int, default=15, help='Perplexity value.')
    parser.add_argument('--path', type=str, default="/home/heitorc62/PlantsConv/dataset/Plant_leave_diseases_dataset_without_augmentation", help='Dataset path.')

    args = parser.parse_args()
    main(args.path, args.perplexity)