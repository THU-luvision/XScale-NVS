#!/bin/bash

#block(name=syt, threads=5, memory=10000, subtasks=1, gpu=true, hours=666)
CUDA_VISIBLE_DEVICES=0,1 python -u base.py
