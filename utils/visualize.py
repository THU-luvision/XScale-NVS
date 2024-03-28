import sys

from tensorboardX import SummaryWriter
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn

from PIL import Image, ImageDraw

from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os

import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt

import time


def create_mask_image(mask):
    mask_image_final = torch.Tensor([]).to(mask.device)
    for i in range(mask.shape[0]):
        mask_image = torch.Tensor(np.zeros([mask.shape[1], mask.shape[2], mask.shape[3]])).to(mask.device)
        slice_mask = mask[i]
        idx0 = torch.where(slice_mask == 0)
        idx1 = torch.where(slice_mask == 1)
        mask_image[idx0] = 1.0
        mask_image[idx1] = 0.5
        mask_image = mask_image.unsqueeze(0)
        mask_image_final = torch.cat((mask_image_final, mask_image), 0)

    mask_image_final = mask_image_final.permute(0, 2, 3, 1)
    mask_image_final = mask_image_final.expand(mask_image_final.shape[0], mask_image_final.shape[1],
                                               mask_image_final.shape[2], 3).cpu().numpy()
    return mask_image_final


def create_alpha_channel(mask):
    alpha_channel_final = torch.Tensor([]).to(mask.device)
    for i in range(mask.shape[0]):
        alpha_channel = torch.Tensor(np.zeros([mask.shape[1], mask.shape[2], mask.shape[3]])).to(mask.device)
        slice_mask = mask[i]
        idx0 = torch.where(slice_mask == 0)
        idx1 = torch.where(slice_mask == 1)
        alpha_channel[idx0] = 0.0
        alpha_channel[idx1] = 1.0
        alpha_channel = alpha_channel.unsqueeze(0)
        alpha_channel_final = torch.cat((alpha_channel_final, alpha_channel), 0)

    alpha_channel_final = alpha_channel_final.permute(0, 2, 3, 1).cpu().numpy()
    return alpha_channel_final