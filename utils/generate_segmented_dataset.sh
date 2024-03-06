#!/bin/bash

# The root directory of your dataset
DATASET_DIR="/home/heitor/USP/IC/FAPESP/code_dataset/dataset/Plant_leave_diseases_dataset_without_augmentation"

# The root directory for the output
OUTPUT_DIR="/home/heitor/USP/IC/FAPESP/code_dataset/dataset/Segmented_dataset"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate through each class directory in the dataset
for class_dir in "$DATASET_DIR"/*; do
    if [ -d "$class_dir" ]; then
        class_name=$(basename "$class_dir")
        
        # Create a corresponding directory in the output directory
        mkdir -p "$OUTPUT_DIR/$class_name"
        
        # Iterate through each image in the class directory
        for image_path in "$class_dir"/*; do
            image_name=$(basename "$image_path")
            output_path="$OUTPUT_DIR/$class_name"
            
            # Call your script here
            python leaf-image-segmentation/segment.py "$image_path" -d "$output_path"
        done
    fi
done

echo "Processing complete."
