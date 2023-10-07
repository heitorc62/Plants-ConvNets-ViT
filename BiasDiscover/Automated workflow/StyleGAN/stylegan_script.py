from modules.train import train_model
from modules.stylegan import MappingNetwork, Generator, Discriminator
from modules.stylegan_modules import PathLengthPenalty
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import random



def get_loader(LOG_RESOLUTION, DATASET, BATCH_SIZE):
    transform = transforms.Compose(
        [
            transforms.Resize((2 ** LOG_RESOLUTION, 2 ** LOG_RESOLUTION)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    return loader


def save_models(netG, netD, current_dir):
    # Save the Generator
    netG_path = os.path.join(current_dir, "models/netG.pth")
    torch.save(netG.state_dict(), netG_path)

    # Save the Discriminator
    netD_path = os.path.join(current_dir, "models/netD.pth")
    torch.save(netD.state_dict(), netD_path)

def save_stats(G_losses, D_losses, current_dir):
    # Save the Generator loss history
    G_losses_np = np.array([loss for loss in G_losses])
    G_losses_path = os.path.join(current_dir, "statistics/G_losses.csv")
    np.savetxt(G_losses_path, G_losses_np, delimiter=",")

    #Save the Discriminator loss history
    D_losses_np = np.array([loss for loss in D_losses])
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






def main():
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    current_dir = os.path.dirname(os.path.realpath(__file__))

    DATASET                 = "../../../../dataset/Plant_leave_diseases_dataset_without_augmentation/"
    DEVICE                  = "cuda0" if torch.cuda.is_available() else "cpu"
    EPOCHS                  = 1
    LEARNING_RATE           = 1e-3
    BATCH_SIZE              = 32
    LOG_RESOLUTION          = 7 #for 128*128
    Z_DIM                   = 256
    W_DIM                   = 256
    LAMBDA_GP               = 10

    loader              = get_loader(LOG_RESOLUTION, DATASET, BATCH_SIZE)
    gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
    critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)
    mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)
    path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

    opt_gen             = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic          = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    gen.train()
    critic.train()
    mapping_network.train()

    # Create batch of latent vectors to visualize the progression of the generator
    fixed_noise = torch.randn(BATCH_SIZE//2, Z_DIM, 1, 1, device=DEVICE)

    netG, netD, G_losses, D_losses, img_list = train_model(
        critic, gen, path_length_penalty, loader, fixed_noise,
        opt_critic, opt_gen, opt_mapping_network, mapping_network,
        DEVICE, LAMBDA_GP, W_DIM, LOG_RESOLUTION, EPOCHS
    )

    save_models(netG, netD, current_dir)
    save_stats(G_losses, D_losses, current_dir)
    save_graphics(G_losses, D_losses, img_list, current_dir)




if __name__ == '__main__':
    main()
