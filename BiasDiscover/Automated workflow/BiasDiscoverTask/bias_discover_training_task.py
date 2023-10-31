from modules.discoverer import GenerativeModel, BiasDiscoverer
from modules.model_loader import load_gen_model, load_classifier
from modules.train import optimize_hyperplane
from modules.save import save_statistics, save_graphics, save_discoverer
import torch
from torch import optim
import os
import argparse


def main(args):
    # Z_DIM = args.Z_DIM
    # LEARNING_RATE = args.LEARNING_RATE
    # EPOCHS = args.EPOCHS
    # BATCH_SIZE = args.BATCH_SIZE
    # DEVICE = args.DEVICE
    # GENERATOR_PATH = args.GENERATOR_PATH
    # CLASSIFIER_PATH = args.CLASSIFIER_PATH

    Z_DIM = 256
    W_DIM = 256
    LEARNING_RATE = 0.001
    EPOCHS = 1
    BATCH_SIZE = 32
    DEVICE = "cpu"
    GENERATOR_PATH = "../StyleGAN/models/netG.pth"
    MAPPING_NETWORK_PATH = "../StyleGAN/models/netMappingNetwork.pth"
    CLASSIFIER_PATH = "../../../Classifier/scripts/fine_tuning/model.pth"
    TARGET_CLASS = 1
    LOG_RESOLUTION = 8

    current_dir = os.path.dirname(os.path.realpath(__file__))

    gen_model = load_gen_model(GENERATOR_PATH, MAPPING_NETWORK_PATH, DEVICE)

    biased_classifier = load_classifier(CLASSIFIER_PATH, DEVICE)

    bias_discoverer = BiasDiscoverer(Z_DIM)

    bias_discoverer.train()

    optimizer = optim.Adam(bias_discoverer.parameters(), lr=LEARNING_RATE)
    
    losses, biased_discoverer = optimize_hyperplane(bias_discoverer, biased_classifier, gen_model, optimizer, EPOCHS, BATCH_SIZE, LOG_RESOLUTION, W_DIM, TARGET_CLASS ,DEVICE)

    save_discoverer(biased_discoverer, current_dir)

    save_statistics(losses, current_dir)

    save_graphics(losses, current_dir)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Discover a biased attribute hyperplane.")
    # Add arguments
    # Add arguments

    parser.add_argument("--DEVICE", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="Device to run")
    
    parser.add_argument("--EPOCHS", type=int, default=300, help="Total number of training epochs")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Batch size for training")
    parser.add_argument("--Z_DIM", type=int, default=256, help="Dimension of Z")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args)
