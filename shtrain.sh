#!/bin/bash

cd /home/pharrington/convmap_dcgan
source activate gpuTF113
export CUDA_VISIBLE_DEVICES=0
python ./WGPtrain.py bigWGP_256 3

