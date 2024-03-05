import os
import random

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vgg import VGG16_BN_Weights

from aux import defaults

import timeit
from datetime import timedelta

activation = {}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

class ILTDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.label_list = label_list
            
            

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        label = self.label_list[idx]
        img = PIL.Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        return img_transformed, img_path, label

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_model(load=False, num_classes=1000):
    model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    if load:
        # Replace the classifier's final layer to match the desired number of classes
        model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def register_hooks(model):
    layers_of_interest = ['5', '12', '22', '32', '42']  # Convolutional layers before MaxPool
    for layer in layers_of_interest:
        model.features[int(layer)].register_forward_hook(get_activation(f'layer_{layer}'))



# Input:
  # (1) images_folder: a string with the path to the folder with the images
  # (2) project_name: a string used to name the saved the files (optional, if
  #                   not provided, it uses the basename of the images_folder)
  # (3) weights_path: a string with the path to the weights to load (optional,
  #                   if not provided, loads weights from the ImageNet)
# Output:
def compute_features(images_folder, batch_id, model, weights_path, batch_df):
    global activation

    batch_size = 8
    device = 'cuda'

    seed = 0
    torch.backends.cudnn.benchmark = True

    seed_everything(seed)
    test_transform = get_transforms()

    freeze_bn(model)
    
    model.to(device)  # Move model to the appropriate device
    
    # Load model weights if a path is provided
    if weights_path != '':
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(device))
        model.load_state_dict(checkpoint)

    register_hooks(model)  # Adjusted for VGG model

    activation = {}  # Reset activation storage

    # Prepare dataset and loader
    inner_folder = os.path.join(images_folder, batch_id, defaults['inner_folder'])  # Adjust according to actual inner folder name
    file_list = [os.path.join(inner_folder, file) for file in os.listdir(inner_folder)]
    
    label_list = []
    for file in file_list:
        # Extract the basename of the file
        file_basename = os.path.basename(file)
        # Query the DataFrame for matching entries
        matching_labels = batch_df[(batch_df['names'] == file_basename) & (batch_df['batch'] == batch_id)]['labels'].values
        label_list.append(matching_labels[0])
        
    print(f"Label list: {label_list}")
    
    
    test_data = ILTDataset(file_list, label_list, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1)
    
    path_images = []
    predictions = []
    features = []
    true_labels = []

    start = timeit.default_timer()  # Start timer

    with torch.no_grad():
        for data, paths, labels in test_loader:
            data = data.to(device)
            output = model(data)
            
            # Assuming 'layer_42' as the layer of interest based on VGG structure
            layer_features = torch.amax(activation['layer_42'], (2, 3))
            features.append(layer_features)

            # Process predictions
            preds = output.argmax(dim=1).cpu().tolist()
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
