import os
import numpy as np
import matplotlib.pyplot as plt

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