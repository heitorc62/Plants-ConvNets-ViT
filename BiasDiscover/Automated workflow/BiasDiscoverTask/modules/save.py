import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def make_path(path):
    dir = os.path.dirname(path)
    if dir: 
        if not os.path.exists(dir):
            os.makedirs(dir)

def save_statistics(losses, current_dir):
    losses_np = np.array([loss for loss in losses])
    losses_path = os.path.join(current_dir, "statistics/losses.csv")
    make_path(losses_path)
    np.savetxt(losses_path, losses_np, delimiter=",")

def save_graphics(losses, current_dir):
    plt.figure(figsize=(10,5))
    plt.title("Total Variation Loss During Training")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plot_path = os.path.join(current_dir, "statistics/training_loss_plot.png")
    make_path(plot_path)
    plt.savefig(plot_path)
    plt.close()  

def save_discoverer(biased_discoverer, current_dir):
    hyperplane_path = os.path.join(current_dir, "hyperplane/hyperplane.pth")
    make_path(hyperplane_path)
    torch.save(biased_discoverer.state_dict(), hyperplane_path)

def save_images_with_scores(images, scores, current_dir):
    fig, ax = plt.subplots(1, len(images), figsize=(15, 5))
    
    for i, (img, score) in enumerate(zip(images, scores)):
        # Display image
        ax[i].imshow(img)
        ax[i].axis('off')  # hide axes
        
        # Annotate the score. Adjust (x,y) values as per your needs.
        ax[i].annotate(f"{score:.2f}", (0, img.shape[0] + 15), color="black", weight="bold", fontsize=12, ha='left')
    
    plt.tight_layout()
    image_path = os.path.join(current_dir, "traversal_images/traversal_images.png")
    make_path(image_path)
    plt.savefig(image_path)
    plt.close()    

