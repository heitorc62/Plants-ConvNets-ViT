import torch
from torch import nn
import numpy as np
from torch import linalg as LA

def one_vs_all_inference(model, images_list, TARGET_CLASS):
    results = []
    with torch.no_grad():
        for images in images_list:
            probs = model(images)
            probs = nn.functional.softmax(probs, dim=1)
            probs = probs[:, TARGET_CLASS]
            results.append(probs)
    return results


class GenerativeModel():
    def __init__(self, generator, mapping_network):
        self.generator = generator
        self.mapping_network = mapping_network


class BiasDiscoverer(nn.Module):
    def __init__(self, z_dim, num_latent_codes=6, starting_alpha=-3, terminating_alpha=3):
        super(BiasDiscoverer, self).__init__()
        self.w = nn.Parameter(torch.randn(1, z_dim))
        self.b = nn.Parameter(torch.randn(1, 1))
        self.alphas = self.get_alphas(num_latent_codes, starting_alpha, terminating_alpha) 
    
    def get_alphas(self, num_alphas, starting_alpha, terminating_alpha):
        step = (terminating_alpha - starting_alpha)/num_alphas
        alphas = np.arange(starting_alpha, terminating_alpha, step)
        return alphas
    
    def project_points(self, points, normal_vec, offset):
        projected_points = points - normal_vec * (points @ normal_vec.t() + offset / (normal_vec @ normal_vec.T))
        return projected_points

    def generate_latent_codes(self, z_points):
        latent_codes = []
        #print("z_points shape: ", z_points.shape)
        #z_proj = z_points - ( ( (w.T @ z_points) + b ) / ( LA.vector_norm(w)**2 ) ) @ w
        z_proj = self.project_points(z_points, self.w, self.b)
        #print("z_proj shape: ", z_proj.shape)

        #print("self.alphas shape: ", self.alphas.shape)

        for alpha in self.alphas:
            latent_codes.append(z_proj + ( alpha * ( self.w / LA.vector_norm(self.w) ) ))

        #print("latent_codes shape: ", latent_codes[0].shape)

        return latent_codes
