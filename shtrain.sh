#!/bin/bash

cd /home/pharrington/convmap_dcgan
source activate gpuTFandKeras
export CUDA_VISIBLE_DEVICES=3
python ./JCtrain.py bigJC 3

