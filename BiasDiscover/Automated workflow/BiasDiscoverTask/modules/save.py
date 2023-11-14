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

def adjust_lists(images, probs):
    print("len(images) = ", len(images))
    print("len(probs) = ", len(probs))
    print("images[0].shape", images[0].shape)
    print("probs[0].shape", probs[0].shape)
    new_images = []
    new_probs = []

    # Iterate through each position of the batches (0 to 31)
    for i in range(len(images[0])):
        # Collect the i-th image from each batch and form a new batch
        new_batch = torch.stack([batch[i] for batch in images])
        new_images.append(new_batch)

        # Collect the i-th probability from each batch and form a new batch
        new_batch_probs = torch.stack([batch[i] for batch in probs])
        new_probs.append(new_batch_probs)
    
    print(f"len(new_images) = {len(new_images)}")
    print(f"len(new_probs) = {len(new_probs)}")
    print(f"new_images[0].shape = {new_images[0].shape}")
    print(f"new_probs[0].shape = {new_probs[0].shape}")
    return new_images, new_probs


def save_images_with_scores(list_of_images, inverted_scores, current_dir):
    traversal_images_batch, scores_batch = adjust_lists(list_of_images, inverted_scores)
    for batch_index, (traversal_images, traversal_scores) in enumerate(zip(traversal_images_batch, scores_batch)):

        fig, ax = plt.subplots(1, len(traversal_images), figsize=(15, 5))
        
        for i, (img, score) in enumerate(zip(traversal_images, traversal_scores)):
            if isinstance(img, torch.Tensor):
                if img.requires_grad:
                    img = img.detach()
                # Permute the tensor from (C, H, W) to (H, W, C) for imshow
                img = img.permute(1, 2, 0)
                img = img.cpu().numpy()  # Also make sure it's a numpy array

            # Display image
            ax[i].imshow(img)
            ax[i].axis('off')  # hide axes
            
            if isinstance(score, torch.Tensor) and score.numel() == 1:
                # Convert tensor with a single value to a Python float
                score_value = score.item()
            else:
            # Handle cases where 'score' is not a single-element tensor
                score_value = score


            # Annotate the score. Adjust (x,y) values as per your needs.
            ax[i].annotate(f"{score_value:.2f}", (0, img.shape[0] + 15), color="black", weight="bold", fontsize=12, ha='left')
        
        plt.tight_layout()
        image_path = os.path.join(current_dir, f"traversal_images/traversal_images_{batch_index}.png")
        make_path(image_path)
        plt.savefig(image_path)
        plt.close()    

