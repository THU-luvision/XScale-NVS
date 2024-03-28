import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import numpy as np
# import pytorch3d

import sys
import os

import time

import pdb
from math import *
from ..models.pil import CNet
import matplotlib.pyplot as plt

class ImplicitShader(nn.Module):
    def __init__(self, params):
        super(ImplicitShader, self).__init__()
        self.params = params
        self.device = self.params.device
        self.render_net_c = CNet(self.params)
        self.render_net_c.to(self.device)

    def forward(self, c_embedding, l_embedding, x_embedding, d_embedding, xyz, view_dir, normal=None, center_global=None, scale_global=None):
        xyz = xyz.detach().clone()
        view_dir = view_dir.detach().clone()
        n = normal.detach().clone()
        
        z, dl, sl = self.render_net_c(l_emb=l_embedding, x_emb=x_embedding, s_feat=c_embedding, n=n, v=view_dir)
        
        self.diffuse = dl.detach().clone()
        self.specular = sl.detach().clone()
        return z, n
