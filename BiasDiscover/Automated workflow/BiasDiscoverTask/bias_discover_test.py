from modules.discoverer import BiasDiscoverer
from modules.model_loader import load_gen_model, load_classifier, load_discoverer
from modules.train import one_vs_all_inference, generate_traversal_images
from modules.save import save_images_with_scores
from StyleGAN.modules.utils import get_w
import os
import argparse

def get_images_and_probs(gen_model, biased_classifier, bias_discoverer, TARGET_CLASS, BATCH_SIZE, W_DIM, DEVICE, LOG_RESOLUTION=8):
    z_data_points = get_w(BATCH_SIZE, gen_model.mapping_network, W_DIM, DEVICE, LOG_RESOLUTION)
    latent_codes = bias_discoverer.generate_latent_codes(z_data_points)
    traversal_images = generate_traversal_images(gen_model, BATCH_SIZE, latent_codes, DEVICE)
    probs_predictions = one_vs_all_inference(biased_classifier, traversal_images, TARGET_CLASS)

    return traversal_images, probs_predictions


def main(args):
    W_DIM = 256
    Z_DIM = 256
    BATCH_SIZE = args.BATCH_SIZE
    DEVICE = args.DEVICE
    GENERATOR_PATH = "../StyleGAN/trained_models/netG.pth"
    MAPPING_NETWORK_PATH = "../StyleGAN/trained_models/netMappingNetwork.pth"
    CLASSIFIER_PATH = "../../../Classifier/scripts/fine_tuning/model.pth"
    TARGET_CLASS = args.TARGET_CLASS
    LOG_RESOLUTION = 8

    current_dir = os.path.dirname(os.path.realpath(__file__))

    gen_model = load_gen_model(GENERATOR_PATH, MAPPING_NETWORK_PATH, DEVICE)

    biased_classifier = load_classifier(CLASSIFIER_PATH, DEVICE)

    bias_discoverer = load_discoverer(Z_DIM, DEVICE)

    bias_discoverer.to(DEVICE)

    traversal_images, probs_predictions = get_images_and_probs(gen_model, biased_classifier, bias_discoverer, TARGET_CLASS, BATCH_SIZE, W_DIM, DEVICE, LOG_RESOLUTION)

    for i in range(BATCH_SIZE):
        save_images_with_scores(traversal_images[i], probs_predictions[i], current_dir)

    


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Discover a biased attribute hyperplane.")
    # Add arguments

    parser.add_argument("--DEVICE", type=str, default="cpu", help="Device to run")
    parser.add_argument("--EPOCHS", type=int, default=30, help="Total number of training epochs")
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="Batch size for training")
    parser.add_argument("--TARGET_CLASS", type=int, default=1, help="Target class to discover biases.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args)