#!/bin/bash

cd /home/pharrington/convmap_dcgan
source activate gpuTFandKeras
export CUDA_VISIBLE_DEVICES=1
python ./train.py LossTest 4

