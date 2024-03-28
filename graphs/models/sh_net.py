import sys
#sys.path.append(".../")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable
import math
from .siren import FiLMLayer, CustomMappingNetwork, frequency_init, first_layer_film_sine_init
from .sh_utils import eval_sh

class CNet(nn.Module):
    def __init__(self, params):
        super(CNet, self).__init__()
        self.params = params
        self.input_dim = self.params.z_length_s * self.params.hash_n_levels
        self.input_dim_dir = 3
        self.hidden_dim = self.params.descriptor_dim

        self.network = nn.ModuleList([
            FiLMLayer(self.input_dim, self.hidden_dim),
            FiLMLayer(self.hidden_dim, self.hidden_dim),
            # ----------------------------------------------
            # FiLMLayer(self.hidden_dim, self.hidden_dim),
            # ----------------------------------------------
        ])
        self.mapping_network = CustomMappingNetwork(
            z_dim=self.params.z_length_l, map_hidden_dim=self.params.descriptor_dim,
            map_output_dim=len(self.network) * self.hidden_dim * 2
        )
        
        if self.params.use_exp_emb:
            self.mapping_network_x = nn.Sequential(
                nn.Linear(self.params.z_length_x, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            )

        self.layer_sl = nn.Sequential(nn.Linear(self.hidden_dim, 3), nn.Sigmoid())
        
        self.layer_sl.apply(frequency_init(25))
        self.network.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, l_emb, x_emb, s_feat, n, v):
        # frequencies, phase_shifts = self.mapping_network(torch.cat((s_feat, n), dim=-1))
        frequencies, phase_shifts = self.mapping_network(l_emb)
        frequencies = frequencies * 15 + 30
        x = s_feat
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        # dl = self.layer_dl(x)
        d_feat = x

        if x_emb is not None and self.params.use_exp_emb:
            x_feat = self.mapping_network_x(x_emb)
            x_shift = x_feat[..., :self.hidden_dim]
            x_scale = torch.exp(x_feat[..., self.hidden_dim:])
            d_feat = x_scale * d_feat + x_shift
        
        c = self.layer_sl(d_feat)
        return c, torch.zeros_like(c), torch.zeros_like(c)