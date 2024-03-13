#!/bin/bash
nohup python run.py --epochs 30 --feature_extract False --use_pretrained False --output_dir /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/regular_dataset &> regular_from_scratch.out &    # 30 epochs, From scratch
nohup python run.py --epochs 30 --dataset_dir /home/heitorc62/PlantsConv/dataset/Segmented_dataset --feature_extract False --use_pretrained False --output_dir /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/segmented_dataset &> segmented_from_scratch.out &    # 30 epochs, From scratch

