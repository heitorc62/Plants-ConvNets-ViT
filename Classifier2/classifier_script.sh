#!/bin/bash
nohup python run.py --epochs 30 --feature_extract False --use_pretrained False &> from_scratch.out &    # 30 epochs, From scratch
nohup python run.py --epochs 30 --feature_extract True --use_pretrained True &> feature_extract.out &   # 30 epochs, Feature extract, pretrained weights
nohup python run.py --epochs 30 --feature_extract False --use_pretrained True &> fine_tuning.out &      # 30 epochs, Fine tuning, pretrained weights
