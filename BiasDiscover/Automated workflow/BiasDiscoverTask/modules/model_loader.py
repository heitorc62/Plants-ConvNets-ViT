from .discoverer import GenerativeModel
from .discoverer import GenerativeModel, BiasDiscoverer
import sys
sys.path.append('../')
from StyleGAN.modules.stylegan import Generator, MappingNetwork
from torchvision import models
import torch
from torch import nn

def load_gen_model(GENERATOR_PATH, MAPPING_NETWORK_PATH, DEVICE, LOG_RESOLUTION=8, Z_DIM=256, W_DIM=256):
    gen = Generator(LOG_RESOLUTION, W_DIM)
    gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=torch.device(DEVICE)))
    gen.to(DEVICE)
    gen.eval()

    mapping_network = MappingNetwork(Z_DIM, W_DIM)
    mapping_network.load_state_dict(torch.load(MAPPING_NETWORK_PATH, map_location=torch.device(DEVICE)))
    mapping_network.to(DEVICE)
    mapping_network.eval()
    return GenerativeModel(gen, mapping_network)


def load_classifier(PATH, DEVICE, NUM_CLASSES=39):
    classifier = models.vgg16_bn()
    num_ftrs = classifier.classifier[6].in_features
    classifier.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
    classifier.load_state_dict(torch.load(PATH, map_location=torch.device(DEVICE)))
    classifier.to(DEVICE)
    classifier.eval()
    return classifier


def load_discoverer(PATH, Z_DIM, DEVICE):
    discoverer = BiasDiscoverer(Z_DIM)
    discoverer.load_state_dict(torch.load(PATH, map_location=torch.device(DEVICE)))
    discoverer.to(DEVICE)
    discoverer.eval()
    return discoverer