import sys

sys.path.append("../")
from graphs.render.render_implicit import ImplicitShader

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()

class HashTexel(nn.Module):
    def __init__(self, params, in_dim, n_levels, n_features_per_level, log2_hashmap_size, base_resolution, per_level_scale):
        super(HashTexel, self).__init__()
        '''
        Atlas Ultra Dense Feature Grid with HashEncoding
        grid_scale = exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
        '''
        self.params = params
        encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",  # "Linear"
            }
        self.spatial_enc = tcnn.Encoding(in_dim, encoding_config, dtype=torch.float)
        out_dim = n_features_per_level * n_levels
        assert self.spatial_enc.n_output_dims == out_dim
        
    def forward(self, x):
        Nv, Nr, Nm, C = x.shape
        x = x.reshape(-1, C)
        x = x * 0.5 + 0.5
        enc = self.spatial_enc(x).reshape(Nv, Nr, Nm, -1)
        return enc

class GlobalModel(nn.Module):
    def __init__(self, params):
        super(GlobalModel, self).__init__()
        self.params = params
        self.device = self.params.device

        # define all the modules in graph.
        self.shading_machine = ImplicitShader(params=self.params)

        self.spatial_encoding = HashTexel(
            params=self.params,
            in_dim=3,
            n_levels=self.params.hash_n_levels, 
            n_features_per_level=self.params.z_length_s, 
            log2_hashmap_size=self.params.log2_hashmap_size,
            base_resolution=self.params.hash_base_resol, 
            per_level_scale=self.params.hash_per_level_scale,
        )
        self.length: int = self.params.hash_n_levels * self.params.z_length_s
        self.mip_eps: float = 2 / (2 ** (self.params.hash_n_levels * np.log2(self.params.hash_per_level_scale)) * self.params.hash_base_resol)

        if self.params.use_exp_emb:
            # img-dependent exposure code.
            self.x_embeddings_list = nn.ModuleList()
            x_embeddings = torch.nn.Embedding(len(self.params.training_view_list), self.params.z_length_x,
                                            max_norm=1, sparse=self.params.use_sparse_embedding)
            torch.nn.init.normal_(x_embeddings.weight.data, 0.0, std=0.1)
            self.x_embeddings_list.append(x_embeddings)
        
        # time-vary lighting embeddings.
        self.l_embeddings_list = nn.ModuleList()
        l_embeddings = torch.nn.Embedding(self.params.num_lit, self.params.z_length_l,
                                          max_norm=1, sparse=self.params.use_sparse_embedding)
        torch.nn.init.normal_(l_embeddings.weight.data, 0.0, std=0.1)
        self.l_embeddings_list.append(l_embeddings)
        print('finished embedding initialization')

    def forward(self, xyz, view_dir, normal=None, l_embedding=None, x_embedding=None, center=None, scale=None, mode='train'):
        view_dir = view_dir.detach().clone()
        view_dir = view_dir.sum(dim=-2)
        view_dir = F.normalize(view_dir, p=2, dim=-1, eps=1e-8).contiguous()

        center_pad = center[:, None, None, :].detach().clone()
        scale_pad = scale[:, None, None, :].detach().clone()
        xyz -= center_pad
        xyz /= scale_pad
        xyz *= 2.0
        
        c_embedding = self.spatial_encoding(xyz) # [Nv, Nr, Nm, C]
        c_embedding = c_embedding.mean(dim=-2)
        c_embedding = torch.cat((c_embedding[..., :self.length], torch.zeros_like(c_embedding[..., self.length:])), dim=-1)
        
        d_embedding = None

        predict_rgb, predict_nrm = self.shading_machine(
            c_embedding=c_embedding,
            l_embedding=l_embedding,
            x_embedding=x_embedding,
            d_embedding=d_embedding,
            xyz=xyz,
            view_dir=view_dir,
            normal=normal,
        )
            
        predict_rgb = linear_to_srgb(predict_rgb).clip(0.0, 1.0)
        predict_rgb = predict_rgb * (1 + 2 * 1e-3) - 1e-3
        return predict_rgb, predict_nrm

def linear_to_srgb(linear):
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(torch.tensor([1e-7]).to(linear.device), linear)**(5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)

class FeatureTexel(nn.Module):
    def __init__(self, num_atlas, fdim, fsize):
        super(FeatureTexel, self).__init__()
        self.fs = fsize + 1
        self.fd = fdim
        self.ft = nn.Parameter(torch.randn(num_atlas, fdim, fsize + 1, fsize + 1))
        torch.nn.init.normal_(self.ft.data, 0.0, std=0.1)

    def forward(self, x, inds):
        ft_traced = self.ft[inds]  # (Nv,Nb,fd,fs,fs) / (Nb,fd,fs,fs)

        Nv, Nb, _ = x.shape
        assert ft_traced.shape == (Nv, Nb, self.fd, self.fs, self.fs)
        ft_traced = ft_traced.view(-1, self.fd, self.fs, self.fs)
        sample_coords = x.reshape(Nv * Nb, 1, 1, 2)
        sample = F.grid_sample(ft_traced, sample_coords,
                               align_corners=True, padding_mode='border')[..., 0, 0].reshape(Nv, Nb, -1)  # (Nv,Nb,C)
        return sample

if __name__ == "__main__":
    a = torch.linspace(0.0, 1.0, 5)[None, None, :, None]
    print(a)
    b = torch.Tensor([-0.9, -0.2, 0.3, 0.8])[None, None, :, None]
    b = torch.cat((torch.zeros_like(b), b), dim=-1)
    print(b.shape)
    c = F.grid_sample(a, b, mode='bilinear', align_corners=True)[..., 0, :].permute(0, 2, 1)
    print(c)