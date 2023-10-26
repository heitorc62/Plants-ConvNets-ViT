import torch
from torch import nn
from torch import linalg as LA


class GenerativeModel():
    def __init__(self, generator, mapping_network):
        self.generator = generator
        self.mapping_network = mapping_network


class BiasDiscoverer(nn.Module):
    def __init__(self, z_dim, generative_model, classifier, num_latent_codes=6, starting_alpha=-3, terminating_alpha=3):
        super(BiasDiscoverer, self).__init__()
        self.w = nn.Parameter(torch.randn(1, z_dim))
        self.b = nn.Parameter(torch.randn(1, 1))
        self.alphas = self.get_alphas(num_latent_codes, starting_alpha, terminating_alpha) 
    
    def get_alphas(self, num_alphas, starting_alpha, terminating_alpha):
        step = (terminating_alpha - starting_alpha)/num_alphas
        alphas = torch.arange(starting_alpha, terminating_alpha, step).unsqueeze(1).unsqueeze(2)
        return alphas
    
    def project_points(self, points, normal_vec, offset):
        projected_points = points - normal_vec * (points @ normal_vec.t() + offset / (normal_vec @ normal_vec.T))
        return projected_points

    def generate_latent_codes(self, z_points):
        #z_proj = z_points - ( ( (w.T @ z_points) + b ) / ( LA.vector_norm(w)**2 ) ) @ w
        z_proj = self.project_points(z_points, self.w, self.b)
        latent_codes = z_proj + ( self.alphas * ( self.w / LA.vector_norm(self.w) ) )
        return latent_codes