import os
import pdb
import sys

sys.path.append("../../")
import numpy as np
import torch
import time
from tqdm import tqdm
from configs.parameter import Params


class Slicer(object):
    def __init__(self, params):
        super(Slicer, self).__init__()
        self.params = params

    def split_to_blocks(self, size, num_blocks):
        '''
            Split the set of possible points into chuncks.
            Then permutes the indices to have random sampling.
            Potentially drop several pixels.
        '''
        idxs = torch.randperm(size)
        block_size = int(float(idxs.size(0)) / float(num_blocks))
        blocks = []
        for i in range(num_blocks):
            blocks.append(idxs[block_size * i: block_size * (i + 1)])
        return blocks

    def batchify_cache_file(self, num_batch=16):
        in_path = os.path.join(self.params.attribute_cache_path, 'cache_list')
        if not os.path.exists(in_path):
            raise ValueError('pre-cached data does not exist.')
        out_path = os.path.join(self.params.attribute_cache_path, 'batches')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        num_rays = {}
        for view in tqdm(self.params.training_view_list):
            xyz = torch.load(os.path.join(in_path, 'xyz_{}.pt'.format(view)))
            normal = torch.load(os.path.join(in_path, 'normal_{}.pt'.format(view)))
            colour = torch.load(os.path.join(in_path, 'colour_{}.pt'.format(view)))
            viewdr = torch.load(os.path.join(in_path, 'viewdr_{}.pt'.format(view)))

            total_ray_num = xyz.shape[0]
            blocks = self.split_to_blocks(size=total_ray_num, num_blocks=self.params.ultra_cache_batch_size)
            for bi, block in enumerate(blocks):
                xyz_batch = xyz[block]
                normal_batch = normal[block]
                colour_batch = colour[block]
                viewdr_batch = viewdr[block]

                torch.save(xyz_batch, os.path.join(out_path, "xyz_{}_b{}.pt".format(view, bi)))
                torch.save(normal_batch, os.path.join(out_path, "normal_{}_b{}.pt".format(view, bi)))
                torch.save(colour_batch, os.path.join(out_path, "colour_{}_b{}.pt".format(view, bi)))
                torch.save(viewdr_batch, os.path.join(out_path, "viewdr_{}_b{}.pt".format(view, bi)))
                # print(block)
                # print(block.min(), block.max())
                # print(block.shape)
            num_rays[view] = block.shape[0]
            print(block.shape[0])
        torch.save(num_rays, os.path.join(out_path, "num_rays.pt"))

    def compare_cache_file(self, ):
        root_folder_2 = os.path.join(self.params.root_file,
                                     'experiment/caches/{}/resol{}/Nf_{}'.format(self.params.modelName, 4, 3))
        f2 = os.path.join(root_folder_2, '{}'.format(self.params.input_mesh_resolution), 'cache_list')

        root_folder_1 = os.path.join(self.params.root_file,
                                     'experiment/caches/{}/resol{}/Nf_{}'.format(self.params.modelName, 1, 1))
        f1 = os.path.join(root_folder_1, 'cache_list')

        for v in self.params.training_view_list:
            uv_nrm = torch.load(os.path.join(f2, 'uv_nrm_{}.pt'.format(v)))
            uv = torch.load(os.path.join(f2, 'uv_{}.pt'.format(v)))
            xyz = torch.load(os.path.join(f2, 'xyz_{}.pt'.format(v)))
            normal = torch.load(os.path.join(f2, 'normal_{}.pt'.format(v)))
            colour = torch.load(os.path.join(f2, 'colour_{}.pt'.format(v)))
            viewdr = torch.load(os.path.join(f2, 'viewdr_{}.pt'.format(v)))
            inds = torch.load(os.path.join(f2, 'inds_{}.pt'.format(v)))
            coord_center = torch.load(os.path.join(f2, 'coord_center_{}.pt'.format(v)))
            scale_factor = torch.load(os.path.join(f2, 'scale_factor_{}.pt'.format(v)))
            if (self.params.faces_per_pixel > 1):
                masks = torch.load(os.path.join(f2, 'mask_{}.pt'.format(v)))
            a = uv_nrm.shape[0]
            print(uv_nrm.shape, uv.shape, xyz.shape, normal.shape, colour.shape, viewdr.shape, inds.shape,
                  coord_center.shape, scale_factor.shape)

            uv_nrm = torch.load(os.path.join(f1, 'uv_nrm_{}.pt'.format(v)))
            uv = torch.load(os.path.join(f1, 'uv_{}.pt'.format(v)))
            xyz = torch.load(os.path.join(f1, 'xyz_{}.pt'.format(v)))
            normal = torch.load(os.path.join(f1, 'normal_{}.pt'.format(v)))
            colour = torch.load(os.path.join(f1, 'colour_{}.pt'.format(v)))
            viewdr = torch.load(os.path.join(f1, 'viewdr_{}.pt'.format(v)))
            inds = torch.load(os.path.join(f1, 'inds_{}.pt'.format(v)))
            coord_center = torch.load(os.path.join(f1, 'coord_center_{}.pt'.format(v)))
            scale_factor = torch.load(os.path.join(f1, 'scale_factor_{}.pt'.format(v)))
            b = uv_nrm.shape[0]
            print(uv_nrm.shape, uv.shape, xyz.shape, normal.shape, colour.shape, viewdr.shape, inds.shape,
                  coord_center.shape, scale_factor.shape)
            print(b / a)
            pdb.set_trace()


if __name__ == "__main__":
    params = Params()
    slicer = Slicer(params=params)
    slicer.batchify_cache_file()