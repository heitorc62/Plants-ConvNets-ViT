import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def make_path(path):
    dir = os.path.dirname(path)
    if dir: 
        if not os.path.exists(dir):
            os.makedirs(dir)


def save_models(netG, netD, current_dir):
    # Save the Generator
    netG_path = os.path.join(current_dir, "models/netG.pth")
    make_path(netG_path)
    torch.save(netG.state_dict(), netG_path)

    # Save the Discriminator
    netD_path = os.path.join(current_dir, "models/netD.pth")
    make_path(netD_path)
    torch.save(netD.state_dict(), netD_path)

def save_stats(G_losses, D_losses, current_dir):
    # Save the Generator loss history
    G_losses_np = np.array([loss for loss in G_losses])
    G_losses_path = os.path.join(current_dir, "statistics/G_losses.csv")
    make_path(G_losses_path)
    np.savetxt(G_losses_path, G_losses_np, delimiter=",")

    #Save the Discriminator loss history
    D_losses_np = np.array([loss for loss in D_losses])
    D_losses_path = os.path.join(current_dir, "statistics/D_losses.csv")
    make_path(D_losses_path)
    np.savetxt(D_losses_path, D_losses_np, delimiter=",")

def save_graphics(G_losses, D_losses, img_list, current_dir):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plot_path = os.path.join(current_dir, "statistics/training_loss_plot.png")
    make_path(plot_path)
    plt.savefig(plot_path)
    plt.close()    

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # Save the animation
    ani.save("StyleGAN2_results.gif", writer=PillowWriter(fps=1))