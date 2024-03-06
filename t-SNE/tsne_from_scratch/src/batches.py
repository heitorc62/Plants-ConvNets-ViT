import os
import math
import shutil
from defaults import defaults
from collections import defaultdict

def create_batches_helper(output_path, num_batches, class_proportions, images_by_class, batch_size):
    batches = {}
    # Create batches
    batches_path = os.path.join(output_path, defaults['images'])
    for batch_num in range(1, num_batches + 1):
        batch_dir = os.path.join(batches_path, f'batch_{batch_num:04d}')
        batches[f'batch_{batch_num:04d}'] = batch_dir
        os.makedirs(batch_dir, exist_ok=True)
        
        for class_name, proportion in class_proportions.items():
            num_images_for_class = round(batch_size * proportion)
            class_dir = os.path.join(batch_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Adjust the number of images for this batch if it exceeds the number of images of this class
            if num_images_for_class > len(images_by_class[class_name]): 
                num_images_for_class = len(images_by_class[class_name])
                
            # Select and copy images for this class to the batch
            images_to_copy = images_by_class[class_name][:num_images_for_class]
            for img_path in images_to_copy:
                shutil.copy(img_path, class_dir)
            images_by_class[class_name] = images_by_class[class_name][num_images_for_class:]  # Update remaining images
            
    return batches
            

def create_batches(dataset_path, output_path, num_batches=8):
    """
    Organizes a flat dataset into batches, where each batch contains a specified number of images per class.

    Args:
    - dataset_path (str): Path to the original flat dataset directory.
    - output_path (str): Path to the directory where the organized dataset will be created.
    - images_per_class_per_batch (int): Number of images from each class to include in each batch.

    The organization follows the pattern:
    batch_0001
        class_1
            image_1.jpg
            image_2.jpg
            ...
        class_2
            image_1.jpg
            image_2.jpg
            ...
        ...
    batch_0002
        class_1
            image_1.jpg
            image_2.jpg
            ...
        class_2
            image_1.jpg
            image_2.jpg
            ...
        ...
    ...
    """
    #Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Assume dataset_path contains directories named after classes, each with their images
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

        
    # Calculate total number of images and class proportions
    total_images = 0
    images_by_class = defaultdict(list)
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images_by_class[class_name] = images
        total_images += len(images)
        
    batch_size = math.ceil(total_images / num_batches)
    class_proportions = {class_name: len(images)/total_images for class_name, images in images_by_class.items()}
    
    batches = create_batches_helper(output_path, num_batches, class_proportions, images_by_class, batch_size)
    
    return len(classes), batches  # Return the number of classes in the dataset as well as the number of batches created
    
