#!/bin/bash

#block(name=warp, threads=5, memory=10000, subtasks=1, gpu=true, hours=666)
CUDA_VISIBLE_DEVICES=0 python -u giga_warping.py
# CUDA_VISIBLE_DEVICES=0 python -u warping.py
CUDA_VISIBLE_DEVICES=0 python -u slice.py
