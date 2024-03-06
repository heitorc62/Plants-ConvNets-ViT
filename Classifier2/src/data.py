from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch


def load_data(data_dir, input_size, batch_size=8, train_percent=0.8):
    # Define your transform
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Load the dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # Determine the lengths of training and validation sets
    train_len = int(train_percent * len(dataset))  # 80% for training
    val_len = len(dataset) - train_len   # 20% for validation
    # Split the dataset
    train_set, val_set = random_split(dataset, [train_len, val_len])
    # Create a dictionary of the datasets
    image_datasets = {'train': train_set, 'val': val_set}
    # Create the dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'val']}
    return dataloaders_dict