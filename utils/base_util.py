import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles

import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append("../")
from utils.scene import save2ply
from utils.camera import cameraPs2Ts_all

def compute_inv(cameraR, cameraT):
    cameraR_inv = np.linalg.inv(cameraR)
    cameraT_inv = np.dot(-cameraR_inv, cameraT)
    return cameraR_inv, cameraT_inv

def zoomin_camera(cameraRT, cameraK, rate):
    # cameraRT[:3,3] = cameraRT[:3,3] * rate
    cameraR = cameraRT[:3,:3]
    cameraT = cameraRT[:3,3]

    cameraR_inv, cameraT_inv = compute_inv(cameraR, cameraT)
    cameraK_inv = np.linalg.inv(cameraK)

    centerPoint = [300,400,1]
    centerPoint_camera = np.dot(cameraK_inv, centerPoint)
    centerPoint_world = np.dot(cameraR_inv, centerPoint_camera) + cameraT_inv

    delta_di = centerPoint_world - cameraT_inv

    cameraT_inv = cameraT_inv + rate * delta_di
    cameraR_new, cameraT_new = compute_inv(cameraR_inv, cameraT_inv)

    cameraRT[:3,:3] = torch.Tensor(cameraR_new)
    cameraRT[:3,3] = torch.Tensor(cameraT_new)
    # cameraRT_new = np.zeros(cameraRT.shape, dtype = cameraRT.dtype)
    # cameraRT_new[:3,:3] = cameraR_new
    # cameraRT_new[:3,3] = cameraT_new

    return cameraRT

def expand_camera(cameraRT, cameraK, rate):
    cameraK[:2,:2] = cameraK[:2,:2] * rate
    return cameraK

def interpolate_cameras(cameraRTO4s_models, cameraKO4s_models, chooose_list, rate_list, interpolate_num = 4, direction = 1, zoomin_flag = False):
    if zoomin_flag:
        cameraRT_ch1 = torch.Tensor(np.array([[0.7404845135614985, 0.2325005573497136, 0.6305753991446678, -357.5776160575932],
                                            [-0.6113867548639025, 0.6226553427405186, 0.48837052366865125, -224.67534057994519],
                                            [-0.2790849063168464, -0.747156625483257, 0.6032149168942565, -314.37583021393465]]))
        cameraRT_ch2 = torch.Tensor(np.array([[0.7818430657776811, -0.5105525697800951, 0.3578509652459924, -236.34755864003523],
                                                [0.011400854998416882, 0.5855732302942881, 0.8105398357749803, -432.3902637782938],
                                                [-0.6233711434370777, -0.6296345988675118, 0.46364696910912734, -86.72248020694681]]))
        for i in range(len(chooose_list)):
            if chooose_list[i] == 1:
                cameraRT_new = zoomin_camera(cameraRTO4s_models[0,i], cameraKO4s_models[0,i], rate_list[i])
                cameraRTO4s_models[0,i] = cameraRT_new
        cameraRTO4s_models[0,2] = cameraRT_ch1
        cameraRTO4s_models[0,3] = cameraRT_ch2
    
    # cameraK_new = expand_camera(cameraRTO4s_models[0,1], cameraKO4s_models[0,1], 0.5)
    # cameraKO4s_models[0,1] = cameraK_new
    
    camera_angles = matrix_to_euler_angles(cameraRTO4s_models[:, :, :, :3],convention="XYZ")  # shape:(N_models, N_view, 3)
    camera_ts = cameraRTO4s_models[:, :, :, 3:4] # shape:(N_models, N_view, 3, 1)

    camera_angles_begin_expand = camera_angles[:,0:-1,:][:,:,None,:].expand(-1,-1,interpolate_num,-1) #shape:(N_models, N_view - 1, interpolate_num, 3)
    camera_angles_end_expand = camera_angles[:, 1:, :][:, :, None, :].expand(-1, -1, interpolate_num,-1)  # shape:(N_models, N_view - 1, interpolate_num, 3)

    camera_ts_begin_expand = camera_ts[:,0:-1,...][:,:,None,...].expand(-1,-1,interpolate_num,-1,-1) #shape:(N_models, N_view - 1, interpolate_num, 3, 1)
    camera_ts_end_expand = camera_ts[:, 1:, ...][:, :, None, ...].expand(-1, -1, interpolate_num, -1,-1)  # shape:(N_models, N_view - 1, interpolate_num, 3, 1)

    if(direction == 1):
        interpolate_alpha = torch.arange(0,1,1.0/interpolate_num) #shape : (interpolate_num)
    else:
        interpolate_alpha = 1 - 1.0 / interpolate_num - torch.arange(0, 1, 1.0 / interpolate_num)  # shape : (interpolate_num)
    camera_angles_new_expand = interpolate_alpha[None, None, :, None] * camera_angles_begin_expand + (1 - interpolate_alpha[None, None, :, None]) * camera_angles_end_expand #shape:(N_models, N_view - 1, interpolate_num, 3)
    #camera_angles_new_expand = camera_angles_begin_expand

    camera_ts_new_expand = interpolate_alpha[None, None, :, None, None] * camera_ts_begin_expand + (1 - interpolate_alpha[None, None, :, None, None]) * camera_ts_end_expand # shape:(N_models, N_view - 1, interpolate_num, 3, 1)
    #camera_ts_new_expand = camera_ts_begin_expand

    camera_rs_new_expand = euler_angles_to_matrix(camera_angles_new_expand, convention="XYZ") #shape:(N_models, N_view - 1, interpolate_num, 3, 3)
    camera_rt_new_expand = torch.cat((camera_rs_new_expand, camera_ts_new_expand), dim=4)
    camera_rt_new = camera_rt_new_expand.reshape(camera_rt_new_expand.shape[0], -1, 3, 4) #shape:(N_models, N_view_new, 3, 4)

    camera_ks_expand = cameraKO4s_models[:,0:-1,None,:,:].expand(-1,-1,interpolate_num,-1,-1) # shape:(N_models, N_view - 1, interpolate_num, 3, 3)
    camera_ks_new = camera_ks_expand.reshape(camera_rt_new_expand.shape[0], -1, 3, 3) #shape:(N_models, N_view_new, 3, 3)

    #pdb.set_trace()

    camera_p0s_new = torch.matmul(camera_ks_new, camera_rt_new) #shape:(N_models, N_view_new, 3, 4)

    camera_ts_new = torch.from_numpy(cameraPs2Ts_all(camera_p0s_new.numpy())).type(camera_p0s_new.dtype)

    ones = torch.tensor([0,0,0,1]).type(camera_rt_new.dtype)[None, None, None, :].expand(camera_rt_new.shape[0], camera_rt_new.shape[1], 1, -1) #shape:(N_models, N_view_new, 1, 4)
    camera_p04s_new = torch.cat((camera_p0s_new, ones), dim=2)  # shape:(N_models, N_view_new, 4, 4)
    # pdb.set_trace()
    # guangyu.
    #  return more camera parameters.
    return camera_p04s_new, camera_p0s_new, camera_ks_new, camera_rt_new, camera_ts_new

# def interpolate_cameras(cameraRTO4s_models, cameraKO4s_models, interpolate_num = 4, direction = 1):
#     camera_angles = matrix_to_euler_angles(cameraRTO4s_models[:, :, :, :3],convention="XYZ")  # shape:(N_models, N_view, 3)
#     camera_ts = cameraRTO4s_models[:, :, :, 3:4] # shape:(N_models, N_view, 3, 1)
#
#     camera_angles_begin_expand = camera_angles[:,0:-1,:][:,:,None,:].expand(-1,-1,interpolate_num,-1) #shape:(N_models, N_view - 1, interpolate_num, 3)
#     camera_angles_end_expand = camera_angles[:, 1:, :][:, :, None, :].expand(-1, -1, interpolate_num,-1)  # shape:(N_models, N_view - 1, interpolate_num, 3)
#
#     camera_ts_begin_expand = camera_ts[:,0:-1,...][:,:,None,...].expand(-1,-1,interpolate_num,-1,-1) #shape:(N_models, N_view - 1, interpolate_num, 3, 1)
#     camera_ts_end_expand = camera_ts[:, 1:, ...][:, :, None, ...].expand(-1, -1, interpolate_num, -1,-1)  # shape:(N_models, N_view - 1, interpolate_num, 3, 1)
#
#     if(direction == 1):
#         interpolate_alpha = torch.arange(0,1,1.0/interpolate_num) #shape : (interpolate_num)
#     else:
#         interpolate_alpha = 1 - 1.0 / interpolate_num - torch.arange(0, 1, 1.0 / interpolate_num)  # shape : (interpolate_num)
#     camera_angles_new_expand = interpolate_alpha[None, None, :, None] * camera_angles_begin_expand + (1 - interpolate_alpha[None, None, :, None]) * camera_angles_end_expand #shape:(N_models, N_view - 1, interpolate_num, 3)
#     #camera_angles_new_expand = camera_angles_begin_expand
#
#     camera_ts_new_expand = interpolate_alpha[None, None, :, None, None] * camera_ts_begin_expand + (1 - interpolate_alpha[None, None, :, None, None]) * camera_ts_end_expand # shape:(N_models, N_view - 1, interpolate_num, 3, 1)
#     #camera_ts_new_expand = camera_ts_begin_expand
#
#     camera_rs_new_expand = euler_angles_to_matrix(camera_angles_new_expand, convention="XYZ") #shape:(N_models, N_view - 1, interpolate_num, 3, 3)
#     camera_rt_new_expand = torch.cat((camera_rs_new_expand, camera_ts_new_expand), dim=4)
#     camera_rt_new = camera_rt_new_expand.reshape(camera_rt_new_expand.shape[0], -1, 3, 4) #shape:(N_models, N_view_new, 3, 4)
#
#     camera_ks_expand = cameraKO4s_models[:,0:-1,None,:,:].expand(-1,-1,interpolate_num,-1,-1) # shape:(N_models, N_view - 1, interpolate_num, 3, 3)
#     camera_ks_new = camera_ks_expand.reshape(camera_rt_new_expand.shape[0], -1, 3, 3) #shape:(N_models, N_view_new, 3, 3)
#
#     #pdb.set_trace()
#
#     camera_p0s_new = torch.matmul(camera_ks_new, camera_rt_new) #shape:(N_models, N_view_new, 3, 4)
#
#     camera_ts_new = torch.from_numpy(cameraPs2Ts_all(camera_p0s_new.numpy())).type(camera_p0s_new.dtype)
#
#     ones = torch.tensor([0,0,0,1]).type(camera_rt_new.dtype)[None, None, None, :].expand(camera_rt_new.shape[0], camera_rt_new.shape[1], 1, -1) #shape:(N_models, N_view_new, 1, 4)
#     camera_p0s_new = torch.cat((camera_p0s_new, ones), dim=2) #shape:(N_models, N_view_new, 4, 4)
#
#     #pdb.set_trace()
#
#     return camera_p0s_new, camera_ts_new

def norm_tensor(x):
    return x / (x.norm(dim = 1)[:,None, ...] + 1e-7)

def generate_new(meta_container, d_container, params, device):
    new_meta_container = {}
    delta_random = params.delta_length * (1 - 2 * torch.rand(meta_container[0]['camera_vector'][0].shape[0], 3)).to(device)
    for stage in range(params.image_compress_stage):
        new_meta_container[stage] = {}
        new_meta_container[stage]['camera_vector'] = (delta_random + meta_container[stage]['camera_vector'][0].to(device))[None, ...]
        new_meta_container[stage]['image_vector'] = norm_tensor(d_container[stage] * meta_container[stage]['image_vector'][0].to(device) - delta_random[...,None,None])[None, ...]
    return new_meta_container


def colour_depth(d_tensor, d_range = 1000):
    d_numpy = d_tensor.detach().cpu().numpy()
    fig, axes = plt.subplots(ncols=d_numpy.shape[0])
    cmap = plt.cm.get_cmap("winter")
    cmap.set_under("magenta")
    cmap.set_over("yellow")
    #print(d_numpy.shape)
    for i in range(d_numpy.shape[0]):
        cs = axes[i].imshow(d_numpy[i,0], cmap = cmap)
    fig.colorbar(cs)
    return fig


def image2sparse_cube(image, grid, mode = 'nearest'):
    #To avoid confusion in notation, let’s note that x corresponds to the width dimension IW, y corresponds to the height dimension IH and z corresponds to the depth dimension ID.
    #grid flow is (W,H,D)
    N_groups, C, img_h, img_w = image.size()

    grid = grid[0].transpose(1,2)[:,None,None, ...]
    grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] / img_w)
    grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] / img_h)

    grid = 2 * grid - 1
    grid = grid.detach()

    cvc = F.grid_sample(image[:, :, None, :, :],
                  grid,
                  mode=mode,
                  align_corners=True
                     )
    #print(grid)
    return cvc[:,:,0,0,:]
    #grid_adjust_h = torch


def image2cube(image, grid, mode = 'bilinear'):
    #To avoid confusion in notation, let’s note that x corresponds to the width dimension IW, y corresponds to the height dimension IH and z corresponds to the depth dimension ID.
    #grid flow is (W,H,D)
    N_groups, C, img_h, img_w = image.size()

    grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] / img_w)
    grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] / img_h)

    grid = 2 * grid - 1
    grid = grid.detach()

    cvc = F.grid_sample(image[:, :, None, :, :],
                  grid,
                  mode=mode,
                     )


    return cvc
    #grid_adjust_h = torch

def visialize_cube(cube):
    '''

    :param cube:  a 4D tensor with channel number of 3
    :param show_layer_num:
    :return:
    '''
    print(cube.size())
    cube_D = cube.size()[3]
    show_layer_num = cube_D//2
    cube_out = torch.zeros((show_layer_num, 3, cube_D, cube_D))
    for i in range(show_layer_num):
        cube_out[i] = cube[:, 2 * i , :, :]
    return cube_out


def construct_graph():
    return

def generate_shift(cube_D):
    x = np.arange(0, cube_D, 1.0)
    y = np.arange(0, cube_D, 1.0)
    z = np.arange(0, cube_D, 1.0)
    xx, yy, zz = np.meshgrid(x, y, z)
    XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).reshape(3, cube_D, cube_D, cube_D)
    #XYZ = np.moveaxis(XYZ, 0, 3)
    return XYZ

def draw_single_dense_cubic(xyz_min, cube_D, cube_resol, q, x = None, n = None, c = None, threshold = 0.5, file_path = '/test', file_name = 'output'):
    xx, yy, zz = np.meshgrid(np.linspace(0, cube_D - 1, cube_D),
                             np.linspace(0, cube_D - 1, cube_D),
                             np.linspace(0, cube_D - 1, cube_D))
    X = xx.flatten()[:, None]
    Y = yy.flatten()[:, None]
    Z = zz.flatten()[:, None]
    XYZ = np.concatenate((Y, X, Z), axis=1)
    XYZ = XYZ * cube_resol
    if(x is None):
        q = q.detach().cpu().numpy()
        q_out = (q > threshold).flatten()
        xyz_q = XYZ[q_out,:] + xyz_min
        q_path = os.path.join(file_path, '%s_q.ply'%file_name)
        save2ply(q_path, xyz_q)
    else:
        if(n is None):

            if(c is None):
                q = q.detach().cpu().numpy()
                x = x.detach().cpu().numpy()

                q_out = (q > threshold).flatten()
                xyz_q = XYZ[q_out, :] + xyz_min
                q_path = os.path.join(file_path, '%s_q.ply' % file_name)
                save2ply(q_path, xyz_q)

                q_out = (q > threshold).flatten()
                xyz_q = XYZ[q_out, :] + xyz_min
                q_path = os.path.join(file_path, '%s_q.ply' % file_name)
                save2ply(q_path, xyz_q)

                x_0 = x[:,0,...].flatten()[q_out][:,None]
                x_1 = x[:,1,...].flatten()[q_out][:,None]
                x_2 = x[:,2,...].flatten()[q_out][:,None]
                x_012 = np.concatenate((x_0, x_1, x_2), axis = 1)
                xyz_x = XYZ[q_out, :] + xyz_min + x_012 * cube_resol
                q_path = os.path.join(file_path, '%s_x.ply' % file_name)
                save2ply(q_path, xyz_x)

            else:
                q = q.detach().cpu().numpy()
                x = x.detach().cpu().numpy()
                c = c.detach().cpu().numpy()
                q_out = (q > threshold).flatten()
                xyz_q = XYZ[q_out, :] + xyz_min
                q_path = os.path.join(file_path, '%s_q.ply' % file_name)
                save2ply(q_path, xyz_q)

                x_0 = x[:, 0, ...].flatten()[q_out][:, None]
                x_1 = x[:, 1, ...].flatten()[q_out][:, None]
                x_2 = x[:, 2, ...].flatten()[q_out][:, None]
                x_012 = np.concatenate((x_0, x_1, x_2), axis=1)
                xyz_x = XYZ[q_out, :] + xyz_min + x_012 * cube_resol

                c_0 = c[:, 0, ...].flatten()[q_out][:, None]
                c_1 = c[:, 1, ...].flatten()[q_out][:, None]
                c_2 = c[:, 2, ...].flatten()[q_out][:, None]
                c_012 = np.concatenate((c_0, c_1, c_2), axis=1)

                q_path = os.path.join(file_path, '%s_x.ply' % file_name)
                save2ply(q_path, xyz_x, rgb_np = c_012)
        else:
            q = q.detach().cpu().numpy()
            x = x.detach().cpu().numpy()
            n = n.detach().cpu().numpy()

            q_out = (q > threshold).flatten()


            xyz_q = XYZ[q_out, :] + xyz_min
            q_path = os.path.join(file_path, '%s_q.ply' % file_name)
            save2ply(q_path, xyz_q)

            x_0 = x[:, 0, ...].flatten()[q_out][:, None]
            x_1 = x[:, 1, ...].flatten()[q_out][:, None]
            x_2 = x[:, 2, ...].flatten()[q_out][:, None]
            x_012 = np.concatenate((x_0, x_1, x_2), axis=1)

            n_0 = n[:, 0, ...].flatten()[q_out][:, None]
            n_1 = n[:, 1, ...].flatten()[q_out][:, None]
            n_2 = n[:, 2, ...].flatten()[q_out][:, None]
            n_012 = np.concatenate((n_0, n_1, n_2), axis=1)

            xyz_x = XYZ[q_out, :] + xyz_min + x_012 * cube_resol
            q_path = os.path.join(file_path, '%s_xn.ply' % file_name)
            save2ply(q_path, xyz_x, normal_np = n_012)



def draw_group_cubic(xyz_min, cube_resol, qxn_group, gt_group, threshold = 0.5, file = '/test', current_iteration = 0):

    resol_new = cube_resol.item()
    xyz_min_new = xyz_min.numpy()
    for i, qxn in enumerate(qxn_group):
        q,x,n = qxn
        cube_D = q.size()[2]
        file_path = os.path.join(file, 'iteration:%s'%str(current_iteration).zfill(5), 'stage:%s'%str(i))
        draw_single_dense_cubic(xyz_min_new, cube_D, resol_new, q, x, n, threshold = threshold, file_path = file_path, file_name = 'output')

        gt_x = gt_group[i + 1][0][:, 1:4, ...]
        gt_c = gt_group[i + 1][0][:, 4:7, ...]  * 256
        gt_q = gt_group[i + 1][0][:, 0, ...][:, None, :, :, :]
        draw_single_dense_cubic(xyz_min_new, cube_D, resol_new, q = gt_q, x = gt_x, n = None, c = gt_c, threshold=threshold, file_path=file_path,
                                file_name='gt')


        xyz_min_new += (resol_new / 2)
        resol_new *= 2


def summery_write():
    pass

def draw_sparse_cubic(sample, image_record, cube_record, file_path = '/test', iter = 0, draw_group = 2):
    file_path = os.path.join(file_path, 'iter:%s'%str(iter).zfill(6))

    a1 = 0
    a2 = -1
    for stage in range(len(sample['output'])):

        gt = sample['output'][stage]['xyz_global'][0,0].transpose(0,1)
        gt_trans = sample['output'][stage]['xyz_local'][0,0].transpose(0,1)
        gtc = 256 * sample['output'][stage]['rgb'][0,0].transpose(0,1)

        #print('sample[output][stage][sort_index][0]', sample['output'][stage]['sort_index'][0])
        gt_true = gt[sample['output'][stage]['q'][0, 0, 0].detach().byte().type(torch.bool), :][sample['output'][stage]['sort_index'][0],:][a1:a2, :]
        gt_trans_true = (2**stage) * sample['cubic_resol'] * \
                        gt_trans[sample['output'][stage]['q'][0, 0, 0].detach().byte().type(torch.bool), :][sample['output'][stage]['sort_index'][0],:][a1:a2, :]
        gtc_true = gtc[sample['output'][stage]['q'][0, 0, 0].detach().byte().type(torch.bool), :][sample['output'][stage]['sort_index'][0],:][a1:a2, :]

        gt = sample['output'][stage]['xyz_global'][0, 0].transpose(0, 1)[(8*a1):(8*a2), :]
        #gt_trans = sample['output'][stage]['xyz_local'][0, 0].transpose(0, 1)[(8*a1):(8*a2), :]
        gtc = 256 * sample['output'][stage]['rgb'][0, 0].transpose(0, 1)[(8*a1):(8*a2), :]


        file_gt = os.path.join(file_path, 'stage:%s' % str(stage), 'gt.ply')
        save2ply(file_gt, gt.cpu().numpy(), gtc.cpu().numpy())
        file_gt_true = os.path.join(file_path, 'stage:%s' % str(stage), 'gt_true.ply')
        save2ply(file_gt_true, gt_true.cpu().numpy(), gtc_true.cpu().numpy())
        file_gt_trans_true = os.path.join(file_path, 'stage:%s' % str(stage), 'gt_trans_true.ply')
        save2ply(file_gt_trans_true, gt_true.cpu().numpy() + gt_trans_true.cpu().numpy(), gtc_true.cpu().numpy())



    for i in range(draw_group):
        for stage in range(len(sample['output'])):
            gt = sample['output'][stage]['xyz_global'][0, 0].transpose(0, 1)

            for num in range(image_record[i][stage]['wrapping_cube'].shape[0]):
                file_wrapping_cube = os.path.join(file_path, 'stage:%s' % str(stage), 'wrapping_cube', '%s_%s.ply'%(str(i),str(num)))
                save2ply(file_wrapping_cube, gt.cpu().numpy(), 256 * (0.5 + image_record[i][stage]['wrapping_cube'][num, :,:].transpose(0,1).cpu().numpy()))

                file_P = os.path.join(file_path, 'stage:%s' % str(stage), 'MetaIn',
                                                  '%s_%s.ply'%(str(i),str(num)))
                save2ply(file_P, gt.cpu().numpy(),
                         256 * (0.5 + image_record[i][stage]['P'][num, :, :].transpose(0, 1)[:,:3].cpu().numpy()))

            file_Cube = os.path.join(file_path, 'stage:%s' % str(stage), 'MetaOutCube','%s.ply'%(str(i)))
            save2ply(file_Cube, gt.cpu().numpy(),
                     256 * (0.5 + image_record[i][stage]['Cube'][0, :, :].transpose(0, 1)[:, :3].cpu().numpy()))

    for stage in range(len(sample['output'])):
        gt = sample['output'][stage]['xyz_global'][0, 0].transpose(0, 1)

        for i in range(draw_group):
            file_CubeF = os.path.join(file_path, 'stage:%s' % str(stage), 'CubeF', '%s.ply' % (str(i)))
            save2ply(file_CubeF, gt.cpu().numpy(),
                     256 * (0.5 + cube_record[stage]['cube'][i, :, :].transpose(0, 1)[:, :3].cpu().numpy()))

            file_q_seperate = os.path.join(file_path, 'stage:%s' % str(stage), 'Q', '%s.ply' % (str(i)))
            save2ply(file_q_seperate,
                     gt.cpu().numpy(),
                     np.repeat(256 * (cube_record[stage]['q_seperate'][i, :, :].transpose(0, 1).cpu().numpy()), 3, axis = 1)
                     )
            index = (cube_record[stage]['q_seperate'][i, 0, :].cpu().numpy()) > 0.5
            file_q_seperate_true = os.path.join(file_path, 'stage:%s' % str(stage), 'Q', 'True_%s.ply' % (str(i)))
            save2ply(file_q_seperate_true,
                     gt.cpu().numpy()[index, :],
                     np.repeat(256 * (cube_record[stage]['q_seperate'][i, :, :].transpose(0, 1).cpu().numpy()), 3,
                               axis=1)[index, :]
                     )

        file_q = os.path.join(file_path, 'stage:%s' % str(stage), 'Q', 'total.ply' )
        save2ply(file_q,
                 gt.cpu().numpy(),
                 np.repeat(256 * (cube_record[stage]['q'][0, :, :].transpose(0, 1).cpu().numpy()), 3,
                           axis=1)
                 )

        index = (cube_record[stage]['q'][0, 0, :].cpu().numpy()) > 0.5
        file_q_true = os.path.join(file_path, 'stage:%s' % str(stage), 'Q', 'True_Total.ply')
        save2ply(file_q_true,
                 gt.cpu().numpy()[index, :],
                 np.repeat(256 * (cube_record[stage]['q'][0, :, :].transpose(0, 1).cpu().numpy()), 3,
                           axis=1)[index, :]
                 )
