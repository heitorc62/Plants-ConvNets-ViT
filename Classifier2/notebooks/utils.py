from torchvision import datasets, transforms
import numpy as np

def get_label_mappings(data_dir):
    # Use a dummy transform since we just need the class names and indices
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load the dataset to access the class_to_idx attribute
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Return the mapping
    # Return the reversed mapping
    return {v: k for k, v in dataset.class_to_idx.items()}



def predicted_labels(y_pred):
    return np.array(list(set(y_pred)))

def target_names_generator(labels_to_use, classes_mapping):
    target_names = []
    for label in labels_to_use:
        target_names.append(classes_mapping[label])

    return np.array(target_names)