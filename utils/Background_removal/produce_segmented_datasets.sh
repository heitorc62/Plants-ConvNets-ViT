#!/bin/bash
nohup python rembg_dataset.py --input_dir /home/heitorc62/PlantsConv/dataset/Plant_leave_diseases_dataset_without_augmentation --output_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/segmented/RMBG >  RMBG.out &
nohup python rembg_dataset.py --input_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/EXP --output_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/segmented/EXP_RMBG >  EXP_RMBG.out &
nohup python rembg_dataset.py --input_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/EXP_WB --output_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/segmented/EXP_WB_RMBG >  EXP_WB_RMBG.out &
nohup python rembg_dataset.py --input_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/WB --output_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/segmented/WB_RMBG >  WB_RMBG.out &
nohup python rembg_dataset.py --input_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/WB_EXP --output_dir /home/heitorc62/PlantsConv/dataset/TCC_datasets/segmented/WB_EXP_RMBG >  WB_EXP_RMBG.out &
