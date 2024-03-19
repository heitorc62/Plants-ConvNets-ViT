import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from PIL import ImageFile
import openTSNE
import argparse

def opentsne_fit(data, n=2):
    tsne = openTSNE.TSNE(
        n_components=n,
        perplexity=30,
        initialization="pca",
        metric="cosine",
        random_state=0,
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tsne_results = tsne.fit(data)
    return tsne_results

def save_plot(tsne_results, labels, label_mappings, output_path, fig_size=40):
    plt.figure(figsize=(fig_size, fig_size))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(label_mappings)))
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)
        subset = tsne_results[indices]
        plt.scatter(subset[:, 0], subset[:, 1], color=colors[i], label=label_mappings[label])
    plt.legend()
    
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot of 8px PlantVillage images in 2D using t-SNE')
    plt.savefig(output_path)
    plt.close()
    

def main(dataset_path, output_path):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transforms.ToTensor())
    
    label_mappings = {i: label for i, label in enumerate(dataset.classes)}
    
    # Prepare data for t-SNE
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1).numpy()  # Flatten images
    
    # Fit t-SNE
    tsne_results = opentsne_fit(images)
    
    save_plot(tsne_results, labels, label_mappings, output_path)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--output_path', type=str, help='')
    # Parse the arguments
    args = parser.parse_args()
    main(args.dataset_path, args.output_path)
