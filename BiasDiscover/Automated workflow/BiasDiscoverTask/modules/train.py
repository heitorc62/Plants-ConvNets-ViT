import torch
from StyleGAN.modules.utils import get_w, get_noise
from .discoverer import one_vs_all_inference


def TotalVARLoss(probs_list):
    total_loss = 0
    for probs in probs_list:

        loss = - torch.log(1e-10 + torch.abs(probs[1:] - probs[:-1]).mean())

        total_loss += loss

    opa = total_loss / len(probs_list)
    return opa


def generate_traversal_images(gen_model, batch_size, latent_codes, DEVICE, W_DIM=256, LOG_RESOLUTION=8):
    traversal_images = []
    for alpha_batch in latent_codes:
        w = gen_model.mapping_network(alpha_batch)
        w = w[None, :, :].expand(LOG_RESOLUTION, -1, -1)
        noise = get_noise(batch_size, LOG_RESOLUTION, DEVICE)
        traversal_images.append(gen_model.generator(w, noise))
    return traversal_images


def optimize_hyperplane(bias_discoverer, biased_classifier, gen_model, optimizer, EPOCHS, BATCH_SIZE, LOG_RESOLUTION, W_DIM, TARGET_CLASS ,DEVICE):
    losses = []
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch, end=" ")
        z_data_points = torch.randn(BATCH_SIZE, W_DIM).to(DEVICE)
        latent_codes = bias_discoverer.generate_latent_codes(z_data_points)
        traversal_images = generate_traversal_images(gen_model, BATCH_SIZE, latent_codes, DEVICE)
        prob_predictions = one_vs_all_inference(biased_classifier, traversal_images, TARGET_CLASS)
        loss = TotalVARLoss(prob_predictions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        print("Loss: ", loss.item())

    return losses, bias_discoverer
        
