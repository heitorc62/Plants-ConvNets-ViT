import torch
from torchvision.utils import save_image
import os


def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def get_w(batch_size, mapping_network, W_DIM, DEVICE, LOG_RESOLUTION):

    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

def get_noise(batch_size, LOG_RESOLUTION, DEVICE):
    
        noise = []
        resolution = 4

        for i in range(LOG_RESOLUTION):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

            noise.append((n1, n2))

            resolution *= 2

        return noise


def generate_examples(gen, epoch, mapping_network, W_DIM, DEVICE, LOG_RESOLUTION, n=100):
    
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            w     = get_w(1, mapping_network, W_DIM, DEVICE, LOG_RESOLUTION)
            noise = get_noise(1, LOG_RESOLUTION, DEVICE)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    gen.train()