import os
import pdb
from readline import insert_text
import sys
from tkinter.messagebox import NO
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
sys.path.append("../")
from configs.parameter import Params

class FragDataset:
    def __init__(self, params, mode="train"):
        super(FragDataset, self).__init__()
        self.params = params
        
        self.cache_file = os.path.join(self.params.attribute_cache_path, 'cache_list')
        if not os.path.exists(self.cache_file):
            raise ValueError('pre-cached data does not exist.')
        self.center = torch.load(os.path.join(self.cache_file, 'center.pt'))
        self.scale = torch.load(os.path.join(self.cache_file, 'scale.pt'))
        
        if mode == "train":
            self.lits_all = torch.load(os.path.join(self.cache_file, 'lits_all.pt'))
            self.batch_file = os.path.join(self.params.attribute_cache_path, 'batches')
            if not os.path.exists(self.batch_file):
                raise ValueError('pre-cached batch data does not exist.')
            self.num_rays = torch.load(os.path.join(self.batch_file, 'num_rays.pt'))
            
            self.ray_batch_lits = []
            for v, view_id in enumerate(self.params.training_view_list):
                self.ray_batch_lits.append(self.lits_all[v])
            self.ray_batch_lits = torch.stack(self.ray_batch_lits, dim=0)

    def __len__(self):
        return self.ray_batch_viewdr.shape[0]

    def __getitem__(self, index):
        """
        Multi-atlas, multi-view, multi-batch random training.
            Args:
                index: local atlas index, i.e. atlas index in the current cluster group, range from 0 to num_atlas.
        """
        sample = {}
        sample['xyz'] = self.ray_batch_xyz[index]
        sample['normal'] = self.ray_batch_normal[index]
        sample['colour'] = self.ray_batch_colour[index]
        sample['viewdr'] = self.ray_batch_viewdr[index]
        sample['lit'] = self.ray_batch_lits[index]
        return index, sample
    
    # def parse_frag_caches(self, batch_id):
    #     self.xyz, self.normal, self.colour, self.viewdr = [], [], [], []

    #     for v in tqdm(self.params.training_view_list):
    #         self.xyz.append(torch.load(os.path.join(self.batch_file, 'xyz_{}_b{}.pt'.format(v, batch_id))))
    #         self.normal.append(torch.load(os.path.join(self.batch_file, 'normal_{}_b{}.pt'.format(v, batch_id))))
    #         self.colour.append(torch.load(os.path.join(self.batch_file, 'colour_{}_b{}.pt'.format(v, batch_id))))
    #         self.viewdr.append(torch.load(os.path.join(self.batch_file, 'viewdr_{}_b{}.pt'.format(v, batch_id))))
    
    def parse_frag_caches(self, batch_id):
        self.xyz, self.normal, self.colour, self.viewdr = [], [], [], []
        
        def read_view_cache(v: int):
            xyz = torch.load(os.path.join(self.batch_file, 'xyz_{}_b{}.pt'.format(v, batch_id)))
            normal = torch.load(os.path.join(self.batch_file, 'normal_{}_b{}.pt'.format(v, batch_id)))
            colour = torch.load(os.path.join(self.batch_file, 'colour_{}_b{}.pt'.format(v, batch_id)))
            viewdr = torch.load(os.path.join(self.batch_file, 'viewdr_{}_b{}.pt'.format(v, batch_id)))
            return xyz, normal, colour, viewdr
        
        self.xyz, self.normal, self.colour, self.viewdr = zip(*thread_map(read_view_cache, self.params.training_view_list, desc='Loading ViewCaches'))
    
    def cache_batch_inds(self, ):
        self.mv_batch_inds = {}
        self.mv_batch_splits = {}
        for view_id in tqdm(self.params.training_view_list):
            num_rays = self.num_rays[view_id]
            # if num_rays < self.params.training_batch_size:
            #     print(num_rays, view_id)
            batch_indices = torch.randperm(num_rays)
            self.mv_batch_inds[view_id] = batch_indices
            self.mv_batch_splits[view_id] = int(num_rays / self.params.training_batch_size)
        
    def create_ray_batch(self, epoch):
        self.ray_batch_xyz = []
        self.ray_batch_normal = []
        self.ray_batch_colour = []
        self.ray_batch_viewdr = []
        for v, view_id in enumerate(self.params.training_view_list):
            split_id = epoch % self.mv_batch_splits[view_id]
            batch_indices = \
                self.mv_batch_inds[view_id][
                split_id * self.params.training_batch_size: (split_id + 1) * self.params.training_batch_size
                ]
            
            self.ray_batch_xyz.append(self.xyz[v][batch_indices])
            self.ray_batch_normal.append(self.normal[v][batch_indices])
            self.ray_batch_colour.append(self.colour[v][batch_indices])
            self.ray_batch_viewdr.append(self.viewdr[v][batch_indices])

        self.ray_batch_xyz = torch.stack(self.ray_batch_xyz, dim=0)
        self.ray_batch_normal = torch.stack(self.ray_batch_normal, dim=0)
        self.ray_batch_colour = torch.stack(self.ray_batch_colour, dim=0)
        self.ray_batch_viewdr = torch.stack(self.ray_batch_viewdr, dim=0)

    def split_to_blocks(self, size, num_blocks):
        '''
            Split the set of possible points into chuncks.
            Then permutes the indices to have random sampling.
        '''
        idxs = torch.randperm(size)
        block_size = int(float(idxs.size(0)) / float(num_blocks))
        blocks = []
        for i in range(num_blocks):
            blocks.append(idxs[block_size * i : block_size * (i + 1)])

        return blocks

if __name__ == '__main__':
    params = Params()