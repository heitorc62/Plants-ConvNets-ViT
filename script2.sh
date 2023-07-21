#!/bin/bash
nohup python VGG16/plantas_conv.py --epochs 100 --feature_extract False --use_pretrained False > from_scratch.out &    # 100 epochs, from scratch
nohup python VGG16/plantas_conv.py --epochs 30 --feature_extract True --use_pretrained True > feature_extract.out &       # 30 epochs, Feature extract, pretrained weights
nohup python VGG16/plantas_conv.py --epochs 30 --feature_extract False --use_pretrained True > fine_tuning.out &      # 30 epochs, Fine tuning, pretrained weights
