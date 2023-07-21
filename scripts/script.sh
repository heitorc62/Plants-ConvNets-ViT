#!/bin/bash
nohup python VGG16/fine_tuning/plantas_conv_fine_tuning.py &
nohup python VGG16/last_layer/plantas_conv_last_layer.py &
nohup python VGG16/from_scratch/plantas_conv_from_scratch.py &
