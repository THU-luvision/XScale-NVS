import sys
#sys.path.append(".../")
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()
    
class CNet(nn.Module):
    def __init__(self, params):
        super(CNet, self).__init__()
        self.params = params
        self.hidden_dim = self.params.descriptor_dim
        self.num_layers_d = 2
        self.num_layers_s = 2
        self.weight_norm = True
        
        self.input_dim = self.params.z_length_s * self.params.hash_n_levels
        config_encoding_dirs = {
                "otype": "SphericalHarmonics",
                "degree": 4
            }
        self.encoding_dir = tcnn.Encoding(3, config_encoding_dirs, dtype=torch.float)
        input_ch = self.encoding_dir.n_output_dims
        self.input_dim_dir = input_ch + self.hidden_dim
        
        self.layer_d = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer_d = nn.utils.weight_norm(self.layer_d)

        self.layer_sl = nn.Sequential(nn.utils.weight_norm(nn.Linear(self.input_dim_dir, 3)), nn.Sigmoid())
    
    def forward(self, l_emb, x_emb, s_feat, n, v):
        d_feat = self.layer_d(s_feat)
        
        reflected_dirs = v
        Nv, Nr, _ = reflected_dirs.shape
        reflected_dirs = (reflected_dirs.reshape(-1, 3) + 1.0) * 0.5
        reflected_dirs = self.encoding_dir(reflected_dirs)
        reflected_dirs = reflected_dirs.reshape(Nv, Nr, -1)
        
        c = self.layer_sl(torch.cat((d_feat, reflected_dirs), dim=-1))
        
        return c, torch.zeros_like(c), torch.zeros_like(c)