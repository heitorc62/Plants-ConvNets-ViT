import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from stylegan_modules import (
    EqualizedConv2d, EqualizedLinear, EqualizedWeight, 
    GradientPenalty, GeneratorBlock, DiscriminatorBlock, 
    ToRGB, StyleBlock, UpSample, MiniBatchStdDev
)

class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)
        return self.net(z)



class Generator(nn.Module):
    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        super().__init__()
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        self.style_block = StyleBlock(d_latent, features[0])
        self.to_rgb = ToRGB(d_latent, features[0])
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.up_samples = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = w.shape[1]
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new
        return rgb
    

class Discriminator(nn.Module):
    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(blocks)
        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)
    