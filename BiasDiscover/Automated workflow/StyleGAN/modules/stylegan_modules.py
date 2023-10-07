from math import sqrt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn



class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb


class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):

        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])
    
class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):

        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])
    

class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):

        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):

        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)

    

class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale
    
    
class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = 0.):

        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)
    

class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):

        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)
    

class EqualizedWeight(nn.Module):

    def __init__(self, shape):

        super().__init__()

        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

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

    

class PathLengthPenalty(nn.Module):

    def __init__(self, beta):

        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):

        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss
    




