import torch
from StyleGAN.modules.utils import get_w
from .discoverer import one_vs_all_inference


def TotalVARLoss(probs):
    return torch.log(1e-10 + torch.abs(probs[:, 1:] - probs[:, :-1]).mean())

def generate_traversal_images(gen_model, latent_codes, DEVICE, W_DIM, LOG_RESOLUTION):
    w = get_w(latent_codes, gen_model.mapping_network, W_DIM, DEVICE, LOG_RESOLUTION) 
    traversal_images = gen_model.generator(w, latent_codes)
    return traversal_images


def optimize_hyperplane(bias_discoverer, biased_classifier, gen_model, optimizer, EPOCHS, BATCH_SIZE, Z_DIM, TARGET_CLASS ,DEVICE):
    losses = []
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        z_data_points = torch.rand(BATCH_SIZE, Z_DIM).to(DEVICE)
        latent_codes = bias_discoverer.generate_latent_codes(z_data_points)
        traversal_images = generate_traversal_images(gen_model, latent_codes)
        prob_predictions = one_vs_all_inference(biased_classifier, traversal_images, TARGET_CLASS)
        loss = TotalVARLoss(prob_predictions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    return losses, bias_discoverer
        