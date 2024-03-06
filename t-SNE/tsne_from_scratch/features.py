import os
import torch
import timeit
from datetime import timedelta
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG16_BN_Weights

activation = {}
    
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

def register_hooks(model):
    layers_of_interest = ['5', '12', '22', '32', '42']  # Convolutional layers before MaxPool
    for layer in layers_of_interest:
        model.features[int(layer)].register_forward_hook(get_activation(f'layer_{layer}'))


def get_model(weights_path, num_classes, device='cuda'):
    model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    print(f"num_classes = {num_classes}")
    num_ftrs = model.classifier[6].in_features
    print(f"num_ftrs = {num_ftrs}")
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        
    freeze_bn(model)
    
    # Load model weights if a path is provided
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(device))
    model.load_state_dict(checkpoint)

    register_hooks(model)  # Adjusted for VGG model
    
    return model


def compute_features(model, data_loader, device='cuda', interest_layer='layer_42'):
    global activation
    activation = {}   # Reset activation storage
    

    features = []
    path_images = []
    predictions = []
    true_labels = []
    
    model.to(device)  # Move model to the appropriate device  
    model.eval()      # Set the model to inference mode
    
    start = timeit.default_timer()  # Start timer

    with torch.no_grad():
        for data, labels, paths in data_loader:
            data = data.to(device)
            outputs = model(data)
            
            # Assuming 'layer_42' as the layer of interest based on VGG structure
            layer_features = torch.amax(activation[interest_layer], (2, 3))
            features.append(layer_features)

            # Process predictions
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds)
            true_labels.extend(labels)
            path_images.extend([os.path.basename(path) for path in paths])

    # Concatenate all features from batches
    features = torch.cat(features, dim=0).cpu().numpy()

    end = timeit.default_timer()  # End timer
    
    # Calculate accuracy
    print(f"true_labels:{true_labels}")
    print(f"predictions:{predictions}")
    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct_predictions / len(true_labels)

    print(f"Feature extraction and prediction completed in: {timedelta(seconds=end-start)}")
    print(f"Accuracy: {accuracy:.4f}")

    return features, path_images, predictions, true_labels
    