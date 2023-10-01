from modules.dcgan import Generator, Discriminator, weights_init
from modules.train import train_model
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def create_dataloader(image_size, data_dir, batch_size, workers):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return dataloader


def create_networks(ngpu, nz, ngf, nc, ndf, device):
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netD = Discriminator(ngpu, nc, ndf).to(device)

    # Handle multi-GPU if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #    netG = nn.DataParallel(netG, list(range(ngpu)))
    #    netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    _ = netG.apply(weights_init)
    _ = netD.apply(weights_init)

    return netG, netD

def save_models(netG, netD, current_dir):
    # Save the Generator
    netG_path = os.path.join(current_dir, "models/netG.pth")
    torch.save(netG.state_dict(), netG_path)

    # Save the Discriminator
    netD_path = os.path.join(current_dir, "models/netD.pth")
    torch.save(netD.state_dict(), netD_path)

def save_stats(G_losses, D_losses, current_dir):
    # Save the Generator loss history
    G_losses_np = np.array([loss.item() for loss in G_losses])
    G_losses_path = os.path.join(current_dir, "statistics/G_losses.csv")
    np.savetxt(G_losses_path, G_losses_np, delimiter=",")

    #Save the Discriminator loss history
    D_losses_np = np.array([loss.item() for loss in D_losses])
    D_losses_path = os.path.join(current_dir, "statistics/D_losses.csv")
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
    plt.savefig(plot_path)
    plt.close()    

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # Save the animation
    ani.save("GAN_results.gif", writer=PillowWriter(fps=1))


if __name__ == "__main__":
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    current_dir = os.path.dirname(os.path.realpath(__file__))                         # Get the directory of the current file (python scripts)                          
    data_dir = "../../../dataset/Plant_leave_diseases_dataset_without_augmentation"   # Root directory for dataset
    workers = 4            # Number of workers for dataloader
    batch_size = 128       # Batch size during training
    image_size = 256       # Spatial size of training images
    nc = 3                 # Number of channels in the training images
    nz = 100               # Size of z latent vector (i.e. size of generator input)
    ngf = 64               # Size of feature maps in generator
    ndf = 64               # Size of feature maps in discriminator
    num_epochs = 5         # Number of training epochs
    lr = 0.0002            # Learning rate for optimizers
    beta1 = 0.5            # Beta1 hyperparameter for Adam optimizers
    ngpu = 2               # Number of GPUs available. Use 0 for CPU mode.
    real_label = 1.
    fake_label = 0.
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    dataloader = create_dataloader(image_size, data_dir, batch_size, workers)
    netG, netD = create_networks(ngpu, nz, ngf, nc, ndf, device)
    # Initialize the BCELoss function
    criterion = nn.BCELoss()
    # Create batch of latent vectors to visualize the progression of the generator
    fixed_noise = torch.randn(batch_size//2, nz, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    netG, netD, G_losses, D_losses, img_list = train_model(netG, netD, criterion, real_label, fake_label, optimizerD, optimizerG, dataloader, fixed_noise, device, nz, num_epochs)

    save_models(netG, netD, current_dir)
    save_stats(G_losses, D_losses, current_dir)
    save_graphics(G_losses, D_losses, img_list, current_dir)