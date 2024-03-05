import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # This method is called by DataLoader to fetch an item
        path, label = self.samples[index]  # Get the image path and label
        image = self.loader(path)  # Load the image
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, path  # Return image, label, and filepath


def get_batch_dataset(batches, batch_id, transform=None, batch_size=32, shuffle=False, num_workers=0):
    """
    Returns a DataLoader for the dataset in a specific batch.

    Args:
    - batches (dict or list): A collection containing the paths to the batch datasets.
    - batch_id (int or str): The identifier for the batch whose dataset is to be loaded.
    - transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default is None.
    - batch_size (int): How many samples per batch to load. Default is 32.
    - shuffle (bool): Whether to shuffle the dataset. Default is True.
    - num_workers (int): How many subprocesses to use for data loading. Default is 0, which means that the data will be loaded in the main process.

    Returns:
    - DataLoader: A DataLoader object for the specified batch dataset.
    """
    dataset_path = batches[batch_id]

    # Define your transformations here. This is just an example.
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    # Load the dataset using ImageFolder
    dataset = CustomImageFolder(root=dataset_path, transform=transform)

    # Create and return a DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return data_loader
    