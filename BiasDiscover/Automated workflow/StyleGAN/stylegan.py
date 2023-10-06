import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

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
    

    
