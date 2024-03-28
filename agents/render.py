"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import sys

sys.path.append("../")
from configs.parameter import Params
from dataset.data import MVSDataset, cameraPs2Ts
from dataset.frag import FragDataset
from agents.architecture import GlobalModel
from graphs.warping.warping import WarpingMachine

from tensorboardX import SummaryWriter
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr

from PIL import Image, ImageDraw
from tqdm import tqdm
from glob import glob

import random
import os

import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt
import time
from dataset.data import mvsnet_to_dr

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """
    def __init__(self, params):
        self.params = params
        self.logger = logging.getLogger("Agent")
        self.debug_flag = True

        self.t_begin = time.time()
        self.manual_seed = random.randint(1, 10000)
        print("seed: ", self.manual_seed)
        random.seed(self.manual_seed)

        use_cuda = self.params.use_cuda

        if (use_cuda and torch.cuda.is_available()):
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.manual_seed)
            print("Program will run on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            print("Program will run on *****CPU***** ")

        torch.multiprocessing.set_sharing_strategy('file_system')

        self.current_epoch = 0
        self.current_iteration = 0
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir, comment='biscale')

        self.AtlasDataset = FragDataset(params=self.params, mode="test")
        self.GlobalModel = GlobalModel(params=self.params).to(self.device)
        self.warping_machine = WarpingMachine(params=self.params)

        self.data_initialize()

        self.mse = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.lr = self.params.lr_net_2d

        self.load_checkpoint(filename = self.params.load_checkpoint_dir)

    def data_initialize(self):
        print('start mesh initialization')
        self.mesh = trimesh.load_mesh(self.params.atlas_load_path)
        
        self.v = torch.from_numpy(self.mesh.vertices).float().contiguous().to(self.device)
        self.f = torch.from_numpy(self.mesh.faces).int().contiguous().to(self.device)
        self.vn = torch.tensor(self.mesh.vertex_normals).float().contiguous().to(self.device)
        
        coord_max_global = torch.max(self.v, dim=0, keepdims=True)[0]
        coord_min_global = torch.min(self.v, dim=0, keepdims=True)[0]
        center_global = 0.5 * (coord_max_global + coord_min_global)
        # scale_global = (coord_max_global - coord_min_global).max(dim=-1, keepdims=True)[0]
        scale_global = (coord_max_global - coord_min_global + 1e-6)
        self.center = center_global[0, ...]
        self.scale = scale_global[0, ...]
        
        print('finish mesh initialization')

    def load_checkpoint(self, filename):
        if(filename is None):
            print('do not load checkpoint')
        else:
            try:
                print("Loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                self.GlobalModel.shading_machine.load_state_dict(checkpoint['shading_machine_dict'])
                self.GlobalModel.spatial_encoding.load_state_dict(checkpoint['spatial_encoding_dict'])
                self.GlobalModel.l_embeddings_list.load_state_dict(checkpoint['l_embeddings_list'])
                if self.params.use_exp_emb:
                    self.GlobalModel.x_embeddings_list.load_state_dict(checkpoint['x_embeddings_list'])

            except OSError as e:
                self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
                self.logger.info("**First time to train**")

    def test_render(self, ):
        glctx = dr.RasterizeGLContext()
        # glctx = dr.RasterizeCudaContext()
        
        if self.params.splitName is not None:
            render_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'nv')
            gt_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, self.params.splitName, 'gt')
        else:
            gt_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'gt')
            render_save_folder = os.path.join(self.params.root_file, 'point_exp', self.params.modelName, 'nv')
        os.makedirs(render_save_folder, exist_ok=True)
        os.makedirs(gt_save_folder, exist_ok=True)
        
        self.mvs_dataset = MVSDataset(params=self.params, load_id_list=self.params.test_view_list, load_img=True)
        images = self.mvs_dataset.imgs_all         # (N_v, H, W, 3)
        
        cameraPoses_interp = self.mvs_dataset.cameraPO4s
        cameraPositions_interp = self.mvs_dataset.cameraTs_new
        
        self.x_embedding = None
        if self.params.use_exp_emb:
            self.x_embedding = self.GlobalModel.x_embeddings_list[0]
        self.l_embedding = self.GlobalModel.l_embeddings_list[0]
        l_embb = self.l_embedding(torch.LongTensor([self.params.trajectory_lit_id]).to(self.device))
                
        N_v = cameraPoses_interp.shape[0]
        with torch.no_grad():
            for v in tqdm(range(N_v)):
                H, W, _ = images[v].shape
                mvp, campos = cameraPoses_interp[v: v+1].to(self.device), cameraPositions_interp[v: v+1].to(self.device)
                v_pos_clip = torch.matmul(torch.nn.functional.pad(self.v, pad=(0,1), mode='constant', value=1.0), torch.transpose(mvp, 1, 2))
                rast, rast_db = dr.rasterize(glctx, v_pos_clip, self.f, (H * self.params.ss_ratio, W * self.params.ss_ratio))  # shape: (N_v, H, W, 4)
                
                frg_xyz, _  = dr.interpolate(self.v.unsqueeze(0), rast, self.f)     # shape: (N_v, H, W, 3)
                frg_normal, _ = dr.interpolate(self.vn.unsqueeze(0), rast, self.f)  # shape: (N_v, H, W, 3)
                frg_normal = F.normalize(frg_normal, p=2, dim=-1, eps=1e-8).contiguous()
                frg_dir = frg_xyz - campos[:, None, None, :]                     # shape: (N_v, H, W, 3)
                frg_dir = F.normalize(frg_dir, p=2, dim=-1, eps=1e-8).contiguous()  # shape: (N_v, H, W, 3)
                inlier_mask = rast[..., 3:] > 0
                inlier_mask = F.interpolate(inlier_mask.permute(0,3,1,2).float(), scale_factor=1.0 / self.params.ss_ratio, mode='bilinear', align_corners=True).permute(0,2,3,1) > 0.0
                outlier_mask = ~inlier_mask.detach().cpu()
                
                x_embb_batch = None
                if self.params.use_exp_emb:
                    if self.params.trajectory_wb_id is None:
                        x_embs = []
                        for vi in range(len(self.params.training_view_list)):
                            x_embs.append(self.x_embedding(torch.LongTensor([vi]).to(self.device)))
                        x_embb_batch = torch.stack(x_embs, dim=0).mean(dim=0)[None, ...]
                    else:
                        x_embb_batch = self.x_embedding(torch.LongTensor([self.params.trajectory_wb_id]).to(self.device))[None, ...]

                xyz = []
                normal = []
                viewdr = []
                for i in range(self.params.ss_ratio):
                    for j in range(self.params.ss_ratio):
                        tmp_xyz = frg_xyz[0][i::self.params.ss_ratio, j::self.params.ss_ratio, ...]
                        tmp_nrm = frg_normal[0][i::self.params.ss_ratio, j::self.params.ss_ratio, ...]
                        tmp_vdr = frg_dir[0][i::self.params.ss_ratio, j::self.params.ss_ratio, ...]
                        xyz.append(tmp_xyz)
                        normal.append(tmp_nrm)
                        viewdr.append(tmp_vdr)
                        
                xyz = torch.stack(xyz, dim=-2).reshape(-1, int(self.params.ss_ratio ** 2), 3)
                normal = torch.stack(normal, dim=-2).reshape(-1, int(self.params.ss_ratio ** 2), 3)
                viewdr = torch.stack(viewdr, dim=-2).reshape(-1, int(self.params.ss_ratio ** 2), 3)
                    
                total_ray_num = xyz.shape[0]
                pred_colour = []
                pred_normal = []
                # start_time = time.time()
                for batch_j in range(0, total_ray_num, self.params.infer_batch_size):
                    xyz_batch = xyz[batch_j: batch_j + self.params.infer_batch_size]  # (Nb,3)
                    viewdr_batch = viewdr[batch_j: batch_j + self.params.infer_batch_size]  # (Nb,3)
                    normal_batch = normal[batch_j: batch_j + self.params.infer_batch_size]  # (Nb,3)
                    l_embb_batch = l_embb.expand(viewdr_batch.shape[0], -1)
                    
                    start_time = time.time()
                    predict_rgb, predict_nrm = self.GlobalModel(
                        xyz=xyz_batch[None, ...],
                        view_dir=viewdr_batch[None, ...],
                        normal=normal_batch[None, ...],
                        l_embedding=l_embb_batch[None, ...],
                        x_embedding=x_embb_batch,
                        # center=self.AtlasDataset.center[None, ...].to(self.device),
                        # scale=self.AtlasDataset.scale[None, ...].to(self.device),
                        center=self.center[None, ...].to(self.device),
                        scale=self.scale[None, ...].to(self.device),
                        mode='test',
                    )
                    
                    print('render time cost: ', time.time() - start_time)
                    
                    predict_nrm = predict_nrm[0]
                    predict_nrm = F.normalize(predict_nrm.sum(dim=-2), p=2, dim=-1, eps=1e-8).contiguous()
                    predict_rgb = predict_rgb[0]

                    pred_colour.append(predict_rgb)
                    pred_normal.append(predict_nrm)

                pred_colour = torch.cat(pred_colour, dim=0).reshape(1, H, W, -1)
                pred_normal = torch.cat(pred_normal, dim=0).reshape(1, H, W, -1)

                pred_colour = torch.clamp(pred_colour, min=0.0, max=1.0)
                pred_normal = (pred_normal + 1.0) * 0.5

                pred_colour[outlier_mask[..., 0]] = 1.0
                pred_normal[outlier_mask[..., 0]] = 1.0
                
                gt_colour = torch.from_numpy(images[v])
                gt_colour[outlier_mask[0, ..., 0]] = 1.0

                # cv2.imwrite(os.path.join(gt_save_folder, 'gt_{}.jpg'.format(self.params.test_view_list[v])), (gt_colour.numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                cv2.imwrite(os.path.join(render_save_folder, 'colour_{}.png'.format(self.params.test_view_list[v])), (pred_colour[0].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
                # cv2.imwrite(os.path.join(render_save_folder, 'normal_{}.jpg'.format(self.params.test_view_list[v])), (pred_normal[0].cpu().numpy() * 256).clip(0, 255).astype('uint8')[..., ::-1])
    
if __name__ == '__main__':
    params = Params()
    agent = BaseAgent(params)
    print('start render #@~#@#@#!@#@!#')

    agent.test_render()

