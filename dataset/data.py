import open3d as o3d
import pdb
import sys
import random
import os
import imageio
sys.path.append("../")
from configs.parameter import Params
import time
import numpy as np
import torch
import cv2
import math
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

class MVSDataset(Dataset):
    def __init__(self, params, load_id_list=None, load_img=True, attribute_flag=False):
        super(MVSDataset, self).__init__()
        self.params = params
        if attribute_flag:
            self.init_attr()
        else:
            self.init_data(load_id_list, load_img=load_img)
    
    def init_attr(self, ):
        imgPath = sorted(glob(self.params.imgNamePattern))
        item_list = [i.split('/')[-1].split('.')[0] for i in imgPath]
        
        self.lits_all = readLits(
            LitsFolder=self.params.litsFolder,
            item_list=item_list)
        
        self.evs_all = readEVs(
            EVsFolder=self.params.litsFolder,
            item_list=item_list)

    def init_data(self, load_id_list=None, load_img=True):
        self.imgs_all, self.sizes_all, self.item_list = readImages(
            imgNamePattern=self.params.imgNamePattern,
            load_id_list=load_id_list,
            load_img=load_img
        )
        
        if self.params.num_lit == 1:
            self.lits_all = torch.zeros(len(self.params.training_view_list)).type(torch.LongTensor)
        else:
            self.lits_all = readLits(
                LitsFolder=self.params.litsFolder,
                item_list=self.item_list)

        # ------------------------------------------------------------------------------------------------------------------------
        # read cameras in mvsnet / colmap convention
        self.cameraPOs, self.cameraPO4s, \
        self.cameraRTO4s, self.cameraKO4s = readCameraP0s_np_all(
            poseFolder=self.params.poseFolder,
            item_list=self.item_list)
        if self.params.ss_ratio > 1:
            self.cameraKO4s[:, :2] *= self.params.ss_ratio
            self.cameraPOs = np.matmul(self.cameraKO4s, self.cameraRTO4s)
            self.cameraPO4s = np.concatenate((self.cameraPOs, np.repeat(np.array([[[0., 0., 0., 1.]]]), repeats=self.cameraPOs.shape[0], axis=0)), axis=1)
        # ------------------------------------------------------------------------------------------------------------------------
        # convert cameras into open-gl convention used by nvdiffrast
        self.cameraPO4s, self.cameraTs_new = mvsnet_to_dr(self.cameraKO4s, self.cameraPOs, self.sizes_all, self.params.ss_ratio, self.params.z_near, self.params.z_far)
        # ------------------------------------------------------------------------------------------------------------------------

        # self.imgs_all, self.cameraPOs, self.cameraPO4s = resize_image_and_matrix(
        #     images=self.imgs_all, projection_M=self.cameraPOs, compress_ratio=self.params.compress_ratio_total
        # )

        # self.cameraTs_new = cameraPs2Ts(self.cameraPOs)
        self.cameraPO4s = torch.from_numpy(self.cameraPO4s).type(torch.FloatTensor)         # (N_v, 4, 4)
        # self.cameraRTO4s = torch.from_numpy(self.cameraRTO4s).type(torch.FloatTensor)       # (N_v, 3, 4)
        # self.cameraKO4s = torch.from_numpy(self.cameraKO4s).type(torch.FloatTensor)         # (N_v, 3, 3)
        self.cameraTs_new = torch.from_numpy(self.cameraTs_new).type(torch.FloatTensor)     # (N_v, 3)
        # self.imgs_all = torch.from_numpy(self.imgs_all).type(torch.FloatTensor)             # (N_v, H, W, 3)

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def cv_to_gl(cv):
    gl = cv @ flip_mat  # convert to GL convention used in iNGP
    return gl

def mvsnet_to_dr(cameraKO4s, cameraPOs, sizes, ss_ratio, zn=0.1, zf=1000.0):
    mvps = []
    cs = []
    for v in range(cameraPOs.shape[0]):
        # fl_x = cameraKO4s[v][0][0]
        fl_y = cameraKO4s[v][1][1]
        H, W = sizes[v] if len(sizes) > 1 else sizes[0]
        # fov_x = math.atan(W * ss_ratio / (fl_x * 2)) * 2
        fov_y = math.atan(H * ss_ratio / (fl_y * 2)) * 2
        y = np.tan(fov_y / 2)
        aspect = W / H
        proj = np.array([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(zf+zn)/(zf-zn), -(2*zf*zn)/(zf-zn)], 
                         [           0,    0,           -1,              0]], dtype=np.float32)
        out = cv2.decomposeProjectionMatrix(cameraPOs[v])
        R = out[1]
        t = out[2]
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.transpose()
        c2w[:3, 3] = (t[:3] / t[3])[:, 0]
        
        c2w_gl = cv_to_gl(c2w)
        mv = np.linalg.inv(c2w_gl)
        campos = c2w_gl[:3, 3]
        mvp = proj @ mv
        mvps.append(mvp)
        cs.append(campos)
    mvps = np.stack(mvps, axis=0)
    cs = np.stack(cs, axis=0)
    return mvps, cs

def readLits(LitsFolder, item_list):
    lits = []
    dict_lit = np.load(os.path.join(LitsFolder, "lit.npz"))
    for i_name in item_list:
        lits.append(torch.from_numpy(dict_lit[i_name]))
    return lits

def readEVs(EVsFolder, item_list):
    evs = []
    dict_ev = np.load(os.path.join(EVsFolder, "crf_coeff.npz"))
    for i_name in item_list:
        evs.append(torch.from_numpy(dict_ev[i_name]))
    return evs

def readCRFs(CRFsFolder, item_list):
    crfs = []
    dict_crf = np.load(os.path.join(CRFsFolder, "crf.npz"))
    for i_name in item_list:
        crfs.append(torch.from_numpy(dict_crf[i_name]))
    return crfs
    
def readImages(imgNamePattern, load_id_list, load_img=True):
    imgPath = sorted(glob(imgNamePattern))
    item_list = [i.split('/')[-1].split('.')[0] for i in imgPath]
    if load_id_list is None:
        img_list = imgPath
    else:
        img_list = [imgPath[i] for i in load_id_list]
        item_list = [item_list[i] for i in load_id_list]
        print("Loading: ", item_list)
    images_np = []
    sizes_np = []
    if load_img:
        for im_name in img_list:
            im = np.array(imageio.imread(im_name)) / 256.0
            images_np.append(im)
            sizes_np.append(im.shape[:2])
        # images_np = np.stack([np.array(imageio.imread(im_name)) for im_name in img_list])    # (N, H, W, 3)
    else:
        print("Does not load images.")
        im = np.array(imageio.imread(img_list[0])) / 256.0
        images_np.append(im)
        sizes_np.append(im.shape[:2])
    return images_np, sizes_np, item_list

def readCameraP0s_np_all(poseFolder, item_list):
    cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(poseFolder, item_list)
    ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
    cameraP0s = np.concatenate((cameraPOs, ones), axis=1)
    return (cameraPOs, cameraP0s, cameraRTOs, cameraKOs)

def readCameraPOs_as_np(poseFolder, item_list):
    cameraPOs = np.empty((len(item_list), 3, 4), dtype=np.float64)
    cameraRTOs = np.empty((len(item_list), 3, 4), dtype=np.float64)
    cameraKOs = np.empty((len(item_list), 3, 3), dtype=np.float64)
    
    for _i, i_name in enumerate(item_list):
        _cameraPO, _cameraRT, _cameraK = readCameraP0_as_np_tanks(
            cameraPO_file=os.path.join(poseFolder, '{}_cam.txt'.format(i_name))
        )
        cameraPOs[_i] = _cameraPO
        cameraRTOs[_i] = _cameraRT
        cameraKOs[_i] = _cameraK
    return cameraPOs, cameraRTOs, cameraKOs

def readCameraP0_as_np_tanks(cameraPO_file):
    with open(cameraPO_file) as f:
        lines = f.readlines()
    cameraRTO = np.empty((3, 4)).astype(np.float64)
    cameraRTO[0, :] = np.array(lines[1].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[1, :] = np.array(lines[2].rstrip().split(' ')[:4], dtype=np.float64)
    cameraRTO[2, :] = np.array(lines[3].rstrip().split(' ')[:4], dtype=np.float64)

    cameraKO = np.empty((3, 3)).astype(np.float64)
    cameraKO[0, :] = np.array(lines[7].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[1, :] = np.array(lines[8].rstrip().split(' ')[:3], dtype=np.float64)
    cameraKO[2, :] = np.array(lines[9].rstrip().split(' ')[:3], dtype=np.float64)

    cameraPO = np.dot(cameraKO, cameraRTO)
    return cameraPO, cameraRTO, cameraKO

def cameraPs2Ts(cameraPOs):
    """
    convert multiple POs to Ts.
    ----------
    input:
        cameraPOs: list / numpy
    output:
        cameraTs: list / numpy
    """
    if type(cameraPOs) is list:
        N = len(cameraPOs)
    else:
        N = cameraPOs.shape[0]
    cameraT_list = []
    for _cameraPO in cameraPOs:
        cameraT_list.append(__cameraP2T__(_cameraPO))

    return cameraT_list if type(cameraPOs) is list else np.stack(cameraT_list)

def __cameraP2T__(cameraPO):
    """
    cameraPO: (3,4)
    return camera center in the world coords: cameraT (3,0)
    >>> P = np.array([[798.693916, -2438.153488, 1568.674338, -542599.034996], \
                  [-44.838945, 1433.912029, 2576.399630, -1176685.647358], \
                  [-0.840873, -0.344537, 0.417405, 382.793511]])
    >>> t = np.array([555.64348632032, 191.10837560939, 360.02470478273])
    >>> np.allclose(__cameraP2T__(P), t)
    True
    """
    homo4D = np.array([np.linalg.det(cameraPO[:, [1, 2, 3]]), -1 * np.linalg.det(cameraPO[:, [0, 2, 3]]),
                       np.linalg.det(cameraPO[:, [0, 1, 3]]), -1 * np.linalg.det(cameraPO[:, [0, 1, 2]])])
    # print('homo4D', homo4D)
    cameraT = homo4D[:3] / homo4D[3]
    return cameraT

def resize_image_and_matrix(images,
                            projection_M,
                            return_list=False,
                            compress_ratio=1.0):
    '''
    compress image and garantee the camera position is not changing
    :param images:  all images of one model

    :param projection_M:  camera matrix
        shape: (N_views, 3, 4)
    :param return_list: bool
        if False return the numpy array
    '''
    resized_h = images[0].shape[0] // compress_ratio
    resized_w = images[0].shape[1] // compress_ratio

    compress_w_new, compress_h_new = compress_ratio, compress_ratio
    transform_matrix = np.array([[[1 / compress_w_new, 0, 0], [0, 1 / compress_h_new, 0], [0, 0, 1]]])
    projection_M_new = np.matmul(transform_matrix, projection_M)

    cameraTs = cameraPs2Ts(projection_M)
    cameraTs_new = cameraPs2Ts(projection_M_new)
    trans_vector = (cameraTs - cameraTs_new)[:, :, None]
    identical_matrix = np.repeat(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), cameraTs.shape[0], axis=0)
    bottom_matrix = np.repeat(np.array([[[0, 0, 0, 1]]]), cameraTs.shape[0], axis=0)
    transform_matrix2 = np.concatenate((identical_matrix, trans_vector), axis=2)
    transform_matrix2 = np.concatenate((transform_matrix2, bottom_matrix), axis=1)
    projection_M_new_f = np.concatenate((projection_M_new, bottom_matrix), axis=1)

    projection_M_new = np.matmul(transform_matrix2, projection_M_new_f)
    cameraPOs = projection_M_new[:, :3, :]
    ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=projection_M_new.shape[0], axis=0)
    cameraP04s = np.concatenate((cameraPOs, ones), axis=1)

    image_resized_list = []
    for i in range(images.shape[0]):
        # image_resized = scipy.misc.imresize(images[i], size=(resized_h, resized_w), interp='bicubic')
        image_resized = np.array(Image.fromarray(images[i].astype(np.uint8)).resize((resized_w, resized_h), Image.BICUBIC))
        image_resized = image_resized / 256.0
        image_resized_list.append(image_resized)
    images_resized = image_resized_list if return_list else np.stack(image_resized_list)
    return images_resized, cameraPOs, cameraP04s

if __name__ == "__main__":
    params = Params()
    mvs = MVSDataset(params=params)