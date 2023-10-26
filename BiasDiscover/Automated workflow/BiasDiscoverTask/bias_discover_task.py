from modules.discoverer import GenerativeModel, BiasDiscoverer
from StyleGAN.modules.stylegan import Generator, MappingNetwork
from modules.train import optimize_hyperplane
from modules.save import save_statistics, save_graphics
from torchvision import models
import torch
from torch import nn
from torch import optim
import os
import argparse

def load_gen_model(GENERATOR_PATH, MAPPING_NETWORK_PATH, DEVICE, LOG_RESOLUTION=8, Z_DIM=256, W_DIM=256):
    gen = Generator(LOG_RESOLUTION, W_DIM)
    gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=torch.device('cpu')))
    gen.to(DEVICE)
    gen.eval()

    mapping_network = MappingNetwork(Z_DIM, W_DIM)
    mapping_network.load_state_dict(torch.load(MAPPING_NETWORK_PATH, map_location=torch.device('cpu')))
    mapping_network.to(DEVICE)
    mapping_network.eval()
    return GenerativeModel(gen, mapping_network)


def load_classifier(PATH, DEVICE, NUM_CLASSES=39):
    classifier = models.vgg16_bn()
    num_ftrs = classifier.classifier[6].in_features
    classifier.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    classifier.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    classifier.to(DEVICE)
    classifier.eval()
    return classifier

def main():
    # Z_DIM = args.Z_DIM
    # LEARNING_RATE = args.LEARNING_RATE
    # EPOCHS = args.EPOCHS
    # BATCH_SIZE = args.BATCH_SIZE
    # DEVICE = args.DEVICE
    # GENERATOR_PATH = args.GENERATOR_PATH
    # CLASSIFIER_PATH = args.CLASSIFIER_PATH

    Z_DIM = 256
    LEARNING_RATE = 0.001
    EPOCHS = 1
    BATCH_SIZE = 32
    DEVICE = "cpu"
    GENERATOR_PATH = "../StyleGAN/trained_models/netG.pth"
    MAPPING_NETWORK_PATH = "../StyleGAN/trained_models/mappingNetwork.pth"
    CLASSIFIER_PATH = "../../../Classifier/models/model.pth"

    current_dir = os.path.dirname(os.path.realpath(__file__))


    mapping_network = 1

    gen_model = GenerativeModel(load_gen_model(GENERATOR_PATH, MAPPING_NETWORK_PATH, DEVICE), mapping_network)

    biased_classifier = load_classifier(CLASSIFIER_PATH, DEVICE)

    bias_discoverer = BiasDiscoverer(Z_DIM, gen_model, biased_classifier)

    optimizer = optim.Adam(bias_discoverer.parameters(), lr=LEARNING_RATE)
    
    losses = optimize_hyperplane(bias_discoverer, optimizer, EPOCHS, BATCH_SIZE, Z_DIM, DEVICE)

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