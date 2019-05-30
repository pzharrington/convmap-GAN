#!/bin/bash

cd /home/pharrington/convmap_dcgan
source activate gpuTF113
export CUDA_VISIBLE_DEVICES=2
python ./JCtrain.py bigJC_256 2

