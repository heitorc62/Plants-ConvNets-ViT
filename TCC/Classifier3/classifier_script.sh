#!/bin/bash

nohup python generalization.py \
    --epochs 1 \
    --device cuda:1 \
    --dataset_dir /home/heitorc62/PlantsConv/dataset/Plant_leave_diseases_dataset_without_augmentation \
    --testset_dir /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/TCC/test_datasets/IPM_dataset \
    --outpur_dir /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/TCC/Classifier3/results/REGULAR \
    &> out/REGULAR.out

#nohup python generalization.py \
#     --device cuda:1 \
#     --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/EXP/from_scratch/model.pth \
#     &> out/EXP.out
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/EXP_WB/from_scratch/model.pth \
#    &> out/EXP_WB.out 
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/WB/from_scratch/model.pth \
#    &> out/WB.out 
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/WB_EXP/from_scratch/model.pth \
#    &> out/WB_EXP.out 
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/EXP_RMBG/from_scratch/model.pth \
#    &> out/EXP_RMBG.out 
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/EXP_WB_RMBG/from_scratch/model.pth \
#    &> out/EXP_WB_RMBG.out 
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/RMBG/from_scratch/model.pth \
#    &> out/RMBG.out 
#nohup python generalization.py \
#    --device cuda:1 \cuda_
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/WB_EXP_RMBG/from_scratch/model.pth \
#    &> out/WB_EXP_RMBG.out 
#nohup python generalization.py \
#    --device cuda:1 \
#    --model_path /home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results/WB_RMBG/from_scratch/model.pth \
#    &> out/WB_RMBG.out