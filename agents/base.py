"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import sys

sys.path.append("../")
from agents.architecture import GlobalModel
from configs.parameter import Params
from dataset.frag import FragDataset

from tensorboardX import SummaryWriter
import pdb
import time
import shutil
import logging
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset, Dataset
import random
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import time

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

        self.AtlasDataset = FragDataset(params=self.params)
        self.GlobalModel = GlobalModel(params=self.params).to(self.device)

        self.multi_gpu = False
        if torch.cuda.device_count() > 1:
            self.multi_gpu = True
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.GlobalModel = nn.DataParallel(self.GlobalModel)

        self.mse = nn.MSELoss()
        self.ssd = nn.MSELoss(reduction='sum')
        self.l1_loss = nn.L1Loss()
        self.relu = nn.ReLU()

        self.lr = self.params.lr_net_2d
        self.update_settings()
        self.load_checkpoint(filename = self.params.load_checkpoint_dir)

    def load_checkpoint(self, filename):
        if(filename is None):
            print('do not load checkpoint')
        else:
            try:
                print("Loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                if self.multi_gpu:
                    self.GlobalModel.module.shading_machine.load_state_dict(checkpoint['shading_machine_dict'])
                    self.GlobalModel.module.spatial_encoding.load_state_dict(checkpoint['spatial_encoding_dict'])
                    self.GlobalModel.module.l_embeddings_list.load_state_dict(checkpoint['l_embeddings_list'])
                    if self.params.use_exp_emb:
                        self.GlobalModel.module.x_embeddings_list.load_state_dict(checkpoint['x_embeddings_list'])
                else:
                    self.GlobalModel.shading_machine.load_state_dict(checkpoint['shading_machine_dict'])
                    self.GlobalModel.spatial_encoding.load_state_dict(checkpoint['spatial_encoding_dict'])
                    self.GlobalModel.l_embeddings_list.load_state_dict(checkpoint['l_embeddings_list'])
                    if self.params.use_exp_emb:
                        self.GlobalModel.x_embeddings_list.load_state_dict(checkpoint['x_embeddings_list'])

            except OSError as e:
                self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
                self.logger.info("**First time to train**")

    def save_checkpoint(self,  is_best=True):
        file_name = 'epoch:%s.pth.tar'%str(self.current_epoch).zfill(5)
        if self.multi_gpu:
            state = {
                'epoch': self.current_epoch,
                'lr': self.lr,
                'iteration': self.current_iteration,
                'shading_machine_dict': self.GlobalModel.module.shading_machine.state_dict(),
                'spatial_encoding_dict': self.GlobalModel.module.spatial_encoding.state_dict(),
                'l_embeddings_list': self.GlobalModel.module.l_embeddings_list.state_dict(),
            }
            if self.params.use_exp_emb:
                state.update({
                    'x_embeddings_list': self.GlobalModel.module.x_embeddings_list.state_dict(),
                })
        else:
            state = {
                'epoch': self.current_epoch,
                'lr': self.lr,
                'iteration': self.current_iteration,
                'shading_machine_dict': self.GlobalModel.shading_machine.state_dict(),
                'spatial_encoding_dict': self.GlobalModel.spatial_encoding.state_dict(),
                'l_embeddings_list': self.GlobalModel.l_embeddings_list.state_dict(),
            }
            if self.params.use_exp_emb:
                state.update({
                    'x_embeddings_list': self.GlobalModel.x_embeddings_list.state_dict(),
                })
                
        if not os.path.exists(os.path.join(self.params.checkpoint_dir, '')):
            os.makedirs(os.path.join(self.params.checkpoint_dir, ''))
        # Save the state
        torch.save(state, os.path.join(self.params.checkpoint_dir, file_name))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(os.path.join(self.params.checkpoint_dir, file_name),
                            os.path.join(self.params.checkpoint_dir, 'best.pth.tar'))

    def update_settings(self, load_state = False):
        # for para in self.GlobalModel.module.shading_machine.parameters():
        #     para.requires_grad = True
        if self.multi_gpu:
            self.optimizer_network = torch.optim.Adam(
                [
                    {
                        "params": filter(lambda p: p.requires_grad,
                                            self.GlobalModel.module.shading_machine.parameters()),
                        "lr": self.params.lr_net_2d
                    },
                ]
            )
            self.optimizer_encoding = torch.optim.Adam(
                [
                    {
                        "params": self.GlobalModel.module.spatial_encoding.parameters(),
                        "lr": self.params.lr_embeddings,
                    },
                ]
            )
            self.optimizer_l_embeddings = torch.optim.Adam(
                [
                    {
                        "params": self.GlobalModel.module.l_embeddings_list.parameters(),
                        "lr": self.params.lr_embeddings,
                    },
                ]
            )
            if self.params.use_exp_emb:
                self.optimizer_x_embeddings = torch.optim.Adam(
                    [
                        {
                            "params": self.GlobalModel.module.x_embeddings_list.parameters(),
                            "lr": self.params.lr_embeddings,
                        },
                    ]
                )
        else:
            self.optimizer_network = torch.optim.Adam(
                [
                    {
                        "params": filter(lambda p: p.requires_grad,
                                         self.GlobalModel.shading_machine.parameters()),
                        "lr": self.params.lr_net_2d
                    },
                ]
            )
            self.optimizer_encoding = torch.optim.Adam(
                [
                    {
                        "params": self.GlobalModel.spatial_encoding.parameters(),
                        "lr": self.params.lr_embeddings,
                    },
                ]
            )
            self.optimizer_l_embeddings = torch.optim.Adam(
                [
                    {
                        "params": self.GlobalModel.l_embeddings_list.parameters(),
                        "lr": self.params.lr_embeddings,
                    },
                ]
            )
            if self.params.use_exp_emb:
                self.optimizer_x_embeddings = torch.optim.Adam(
                    [
                        {
                            "params": self.GlobalModel.x_embeddings_list.parameters(),
                            "lr": self.params.lr_embeddings,
                        },
                    ]
                )
                
    def train_base(self):
        change_batch_key = 0
        # self.update_settings()
        start_time = time.time()
        for epoch in range(self.current_epoch, self.params.max_epoch):
            self.current_epoch = epoch
            print('-' * 50)
            print('start training %s epoch' % str(self.current_epoch))
            print('-' * 50)
            print('current time: ', time.time() - start_time)
            ###########################################
            if (epoch % self.params.shuffle_batches_iter == 0):
                self.AtlasDataset.cache_batch_inds()
            if (epoch % self.params.change_batches_iter == 0):
                # batch_idx = np.random.choice(self.params.ultra_cache_batch_size, 1)
                batch_idx = change_batch_key % self.params.ultra_cache_batch_size
                self.AtlasDataset.parse_frag_caches(batch_id=batch_idx)
                change_batch_key += 1
            if (epoch % self.params.save_checkpoint_iter == 0):
                self.save_checkpoint()
            if (epoch % self.params.batch_shuffle_iter == 0):
                # view_ind = np.random.choice(self.params.training_view_list)
                # print('Training on View{}.'.format(view_ind))
                self.AtlasDataset.create_ray_batch(epoch=epoch)
                Dataset = DataLoader(
                    self.AtlasDataset,
                    batch_size=self.params.random_view_batch_size,
                    shuffle=True, num_workers=4
                )
                
            if (epoch % self.params.validate_iter == 0):
                self.params.mode = "validation"
                # self.validate()
                self.params.mode = "train"
                print('validate finished')

            self.train_one_epoch(Dataset)

    def train_one_epoch(self, Dataset):
        for index_data, (view_inds, sample) in enumerate(Dataset, 0):
            print('Training on View: ', view_inds)
            print('Training on Lit: ', sample['lit'])

            colour = sample['colour']  # (Nv,Np,3)
            viewdr = sample['viewdr']  # (Nv,Np,3)
            xyz = sample['xyz']
            normal = sample['normal']  # (Nv,Np,3)
            
            x_embedding = None
            if self.multi_gpu:
                if self.params.use_exp_emb:
                    x_embedding = self.GlobalModel.module.x_embeddings_list[0](torch.LongTensor(view_inds).to(self.device)).unsqueeze(1).expand(-1, xyz.shape[1], -1).to(self.device)
                    
                l_embedding = self.GlobalModel.module.l_embeddings_list[0](
                    torch.LongTensor(sample['lit']).to(self.device)).unsqueeze(1).expand(-1, xyz.shape[1], -1).to(self.device)
            else:
                if self.params.use_exp_emb:
                    x_embedding = self.GlobalModel.x_embeddings_list[0](torch.LongTensor(view_inds).to(self.device)).unsqueeze(1).expand(-1, xyz.shape[1], -1).to(self.device)
                    
                l_embedding = self.GlobalModel.l_embeddings_list[0](
                    torch.LongTensor(sample['lit']).to(self.device)).unsqueeze(1).expand(-1, xyz.shape[1], -1).to(self.device)

            predict_rgb, predict_nrm = self.GlobalModel(
                xyz=xyz.to(self.device),
                view_dir=viewdr.to(self.device),
                normal=normal.to(self.device),
                l_embedding=l_embedding,
                x_embedding=x_embedding,
                center=self.AtlasDataset.center[None, ...].expand(viewdr.shape[0], -1),
                scale=self.AtlasDataset.scale[None, ...].expand(viewdr.shape[0], -1)
            )
            
            loss_rgb = self.l1_loss(predict_rgb, colour.to(self.device))
            # loss_rgb = self.mse(predict_rgb, colour.to(self.device))
            loss_nrm = self.mse(predict_nrm[..., 0, :], normal[..., 0, :].to(self.device))
            loss = self.params.loss_rgb_weight * loss_rgb + self.params.loss_nrm_weight * loss_nrm

            self.optimizer_network.zero_grad()
            self.optimizer_encoding.zero_grad()
            self.optimizer_l_embeddings.zero_grad()
            if self.params.use_exp_emb:
                self.optimizer_x_embeddings.zero_grad()

            loss.backward()

            self.optimizer_network.step()
            self.optimizer_encoding.step()
            self.optimizer_l_embeddings.step()
            if self.params.use_exp_emb:
                self.optimizer_x_embeddings.step()

            if (self.current_iteration % self.params.draw_cubic_iter) == 0:
                self.summary_writer.add_scalar("epoch/loss_rgb", loss_rgb, self.current_epoch)
                self.summary_writer.add_scalar("epoch/loss_nrm", loss_nrm, self.current_epoch)

            if (self.current_iteration % self.params.change_nrm_coef_iter) == 0:
                self.params.loss_nrm_weight *= 0.5
                print('change nrm weight: ', self.params.loss_nrm_weight)

            self.current_iteration += 1

if __name__ == '__main__':
    params = Params()
    agent = BaseAgent(params)
    if (params.mode == 'train'):
        print('start train #@~#@#@#!@#@!#')
        agent.train_base()
