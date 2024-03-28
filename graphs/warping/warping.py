import smtpd
import sys

sys.path.append("../../")
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from tqdm import tqdm
import random
import os
import shutil
import numpy as np
import cv2
import scipy.misc
import math
import matplotlib.pyplot as plt
import time
import trimesh
import nvdiffrast.torch as dr
from configs.parameter import Params
from dataset.data import MVSDataset

class WarpingMachine(nn.Module):
    def __init__(self, params):
        super(WarpingMachine, self).__init__()
        self.params = params
        self.device = self.params.device

        self.cache_root_file_path = self.params.attribute_cache_path
        if not os.path.exists(self.cache_root_file_path):
            os.makedirs(self.cache_root_file_path)

    def forward(self):
        self.MVSDataset = MVSDataset(self.params, load_id_list=self.params.training_view_list)
        self.cameraPoses = self.MVSDataset.cameraPO4s  # (N_v, 4, 4)
        self.images_ori = self.MVSDataset.imgs_all  # (N_v, H, W, 3)
        self.cameraPositions = self.MVSDataset.cameraTs_new  # (N_v, 3)
        
        print("images: ", len(self.images_ori))
        print("cameraPoses: ", self.cameraPoses.shape)
        print("cameraPositions", self.cameraPositions.shape)
        assert len(self.images_ori) == len(self.params.training_view_list), "inconsistent num between params.training_view_list and # images"
        
        cache_save_folder = os.path.join(self.cache_root_file_path, 'cache_list')
        if not os.path.exists(cache_save_folder):
            os.makedirs(cache_save_folder)
        
        if self.params.splitName is not None:
            vis_save_folder = os.path.join(self.params.datasetFolder, self.params.modelName, self.params.splitName, "normal")
        else:
            vis_save_folder = os.path.join(self.params.datasetFolder, self.params.modelName, "normal")
        os.makedirs(vis_save_folder, exist_ok=True)
        
        glctx = dr.RasterizeGLContext()
        # glctx = dr.RasterizeCudaContext()
            
        with torch.no_grad():
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
            self.num_verts = self.v.shape[0]
            self.num_facet = self.f.shape[0]
            print('finish mesh initialization')
            
            batch_size = self.params.rasterize_batch_size
            assert batch_size == 1, "current rasterizer only supports batch_size 1 for input images with different resolution"

            for batch_i in tqdm(range(0, self.cameraPositions.shape[0], 1)):
                view_id_list = self.params.training_view_list[batch_i: batch_i + 1]
                mvp = self.cameraPoses[batch_i: batch_i + 1, ...].to(self.device)
                campos = self.cameraPositions[batch_i: batch_i + 1, ...].to(self.device)
                images_ori_batch = torch.from_numpy(self.images_ori[batch_i]).type(torch.FloatTensor)
                num_batch_view = mvp.shape[0]
                
                H, W, _ = images_ori_batch.shape
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

                for v in range(num_batch_view):
                    xyz = []
                    normal = []
                    viewdr = []
                    for i in range(self.params.ss_ratio):
                        for j in range(self.params.ss_ratio):
                            tmp_xyz = frg_xyz[v][i::self.params.ss_ratio, j::self.params.ss_ratio, ...]
                            tmp_nrm = frg_normal[v][i::self.params.ss_ratio, j::self.params.ss_ratio, ...]
                            tmp_vdr = frg_dir[v][i::self.params.ss_ratio, j::self.params.ss_ratio, ...]
                            xyz.append(tmp_xyz)
                            normal.append(tmp_nrm)
                            viewdr.append(tmp_vdr)
                    
                    xyz = torch.stack(xyz, dim=0)
                    normal = torch.stack(normal, dim=0)
                    viewdr = torch.stack(viewdr, dim=0)
                                        
                    if self.params.undistort_crop_rate_h * self.params.undistort_crop_rate_w > 0:
                        if self.params.undistort_crop_iter is not None and batch_i >= self.params.undistort_crop_iter:
                            temp_h, temp_w = images_ori_batch.shape[0], images_ori_batch.shape[1]
                            crop_h = int(temp_h * self.params.undistort_crop_rate_h)
                            crop_w = int(temp_w * self.params.undistort_crop_rate_w)
                            xyz = xyz[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                            normal = normal[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                            viewdr = viewdr[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                            images_ori_batch = images_ori_batch[crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                            inlier_mask = inlier_mask[:, crop_h: temp_h - crop_h, crop_w: temp_w - crop_w, ...]
                                                                
                    nrm_vis = normal[0].detach().cpu()
                    xyz = xyz[:, inlier_mask[v, ..., 0]].detach().cpu().permute(1,0,2)
                    normal = normal[:, inlier_mask[v, ..., 0]].detach().cpu().permute(1,0,2)
                    viewdr = viewdr[:, inlier_mask[v, ..., 0]].detach().cpu().permute(1,0,2)
                    colour = images_ori_batch[inlier_mask[v, ..., 0]].detach().cpu()
                    
                    nrm_vis[~inlier_mask[v, ..., 0]] == 0.0
                    vis_n = (nrm_vis.numpy()[..., ::-1] * 0.5 + 0.5)*255
                    vis_c = images_ori_batch.cpu().numpy()*255
                    vis = vis_n * 0.3 + vis_c * 0.7
                    cv2.imwrite(os.path.join(vis_save_folder, "nrm_{}_{}.jpg".format(self.MVSDataset.item_list[batch_i], view_id_list[v])), vis[..., ::-1])
                    
                    checker = (normal * -viewdr).sum(dim=-1) > 0.
                    print(checker.sum(), checker.shape[0] * checker.shape[1])

                    torch.save(xyz, os.path.join(cache_save_folder, "xyz_{}.pt".format(view_id_list[v])))
                    torch.save(normal, os.path.join(cache_save_folder, "normal_{}.pt".format(view_id_list[v])))
                    torch.save(colour, os.path.join(cache_save_folder, "colour_{}.pt".format(view_id_list[v])))
                    torch.save(viewdr, os.path.join(cache_save_folder, "viewdr_{}.pt".format(view_id_list[v])))

        torch.save(self.num_facet, os.path.join(cache_save_folder, "num_facet.pt"))
        torch.save(self.num_verts, os.path.join(cache_save_folder, "num_verts.pt"))
        torch.save(self.MVSDataset.lits_all, os.path.join(cache_save_folder, "lits_all.pt"))
        # torch.save(self.MVSDataset.evs_all, os.path.join(cache_save_folder, "evs_all.pt"))
        torch.save(self.center, os.path.join(cache_save_folder, "center.pt"))
        torch.save(self.scale, os.path.join(cache_save_folder, "scale.pt"))


if __name__ == "__main__":
    params = Params()
    warping_machine = WarpingMachine(params=params)
    warping_machine()
