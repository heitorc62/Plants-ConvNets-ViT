from modules.train import train_model
from modules.stylegan import MappingNetwork, Generator, Discriminator
from modules.stylegan_modules import PathLengthPenalty
import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
from modules.save import save_graphics, save_models, save_stats



def get_loader(LOG_RESOLUTION, DATASET, BATCH_SIZE, WORKERS):
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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=WORKERS)
    return loader



def main(args):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # Assigning values from args to variables
    DATASET = args.DATASET
    DEVICE = args.DEVICE
    EPOCHS = args.EPOCHS
    LEARNING_RATE = args.LEARNING_RATE
    BATCH_SIZE = args.BATCH_SIZE
    LOG_RESOLUTION = args.LOG_RESOLUTION
    Z_DIM = args.Z_DIM
    W_DIM = args.W_DIM
    LAMBDA_GP = args.LAMBDA_GP
    WORKERS = args.WORKERS

    loader              = get_loader(LOG_RESOLUTION, DATASET, BATCH_SIZE, WORKERS)
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
    fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)

    netG, netD, G_losses, D_losses, img_list = train_model(
        critic, gen, path_length_penalty, loader, fixed_noise,
        opt_critic, opt_gen, opt_mapping_network, mapping_network,
        DEVICE, LAMBDA_GP, W_DIM, LOG_RESOLUTION, EPOCHS
    )

    save_models(netG, netD, current_dir)
    save_stats(G_losses, D_losses, current_dir)
    save_graphics(G_losses, D_losses, img_list, current_dir)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on some data.")
    
    # Add arguments
 # Add arguments
    parser.add_argument("--DATASET", type=str, default="../../../../dataset/Plant_leave_diseases_dataset_without_augmentation/", help="Path to the dataset")

    parser.add_argument("--DEVICE", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="Device to run")
    
    parser.add_argument("--EPOCHS", type=int, default=300, help="Total number of training epochs")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Batch size for training")
    parser.add_argument("--LOG_RESOLUTION", type=int, default=8, help="Log resolution for the model")
    parser.add_argument("--Z_DIM", type=int, default=256, help="Dimension of Z")
    parser.add_argument("--W_DIM", type=int, default=256, help="Dimension of W")
    parser.add_argument("--LAMBDA_GP", type=float, default=10.0, help="Gradient penalty coefficient")
    parser.add_argument("--WORKERS", type=int, default=4, help="Number of workers to dataloader")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args)

