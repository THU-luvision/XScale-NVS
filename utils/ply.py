from plyfile import PlyData, PlyElement
import os
import numpy as np
import open3d as o3d
import pdb
import torch
from pytorch3d.structures import Meshes

def load_ply_mesh(mesh_save_path,
                  mesh_name_pattern='#_surface_xyz_flow.ply',
                  mesh_resol=8.0,
                  device=torch.device("cpu"),
                  compute_mesh_normal=True):
    # mesh = pymesh.meshio.load_mesh(os.path.join(mesh_save_path, mesh_name_pattern.replace('#', str(mesh_resol))))
    # mesh = PlyData.read(os.path.join(mesh_save_path, mesh_name_pattern.replace('#', str(mesh_resol))))
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_save_path, mesh_name_pattern.replace('#', str(mesh_resol))))

    if (compute_mesh_normal):
        mesh.compute_vertex_normals()
        # print(np.asarray(mesh.triangle_normals).shape)
        # print(np.asarray(mesh.vertex_normals).shape)
        # pdb.set_trace()
        vertex_normals = np.asarray(mesh.vertex_normals)

    mesh_points = np.asarray(mesh.vertices)
    face_id = np.asarray(mesh.triangles)

    mesh = Meshes(verts=[torch.FloatTensor(mesh_points).to(device)],
                  faces=[torch.FloatTensor(face_id).to(device)],
                  # verts_normals=[torch.FloatTensor(vertex_normals).to(device)],
                  )

    return mesh

# def load_ply_mesh(mesh_save_path, mesh_resol, noise_level, mesh_name_pattern='$_mesh_noise_&.ply', device=torch.device("cpu")):
#     # mesh = pymesh.meshio.load_mesh(os.path.join(mesh_save_path, mesh_name_pattern.replace('#', str(mesh_resol))))
#     # mesh = PlyData.read(os.path.join(mesh_save_path, mesh_name_pattern.replace('#', str(mesh_resol))))
#     mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_save_path, mesh_name_pattern.\
#                                                   replace('$', str(mesh_resol)).\
#                                                   replace('&', str(noise_level))))
#     # print(np.asarray(mesh.vertices).shape)
#     # print(np.asarray(mesh.triangles).shape)
#     mesh_points = np.asarray(mesh.vertices)
#     face_id = np.asarray(mesh.triangles)
#     # pdb.set_trace()
#     mesh = Meshes(verts=[torch.FloatTensor(mesh_points).to(device)],
#                   faces=[torch.FloatTensor(face_id).to(device)],
#                   )
#     return mesh

def o3d_load(datasetName,
             datasetFolder,
             gtNamePattern,
             modelList,
             count_normal,
             edit_flag,
             ):
    gt_models = []
    for model in modelList:
        if (datasetName == 'DTU'):
            gt_name = gtNamePattern.replace('#', str(model))
            pcd_name = os.path.join(datasetFolder, gt_name)
            print(pcd_name)
            pcd = o3d.io.read_point_cloud(pcd_name)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            colors = np.asarray(pcd.colors)
            if edit_flag:
                slice_total_mask = np.load('/home/jinzhi//hdd10T/aMyLab/project_render/record/zhiwei/edit_ply/msak.npy')
                idx = np.where(slice_total_mask == 3)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 70*np.array([0.75148225, -0.18757119, -0.63252785])

                idx = np.where(slice_total_mask == 4)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 70*np.array([0.75148225, -0.18757119, -0.63252785])

                idx = np.where(slice_total_mask == 5)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 70*np.array([0.75148225, -0.18757119, -0.63252785])

                idx = np.where(slice_total_mask == 6)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 50*np.array([0.75148225, -0.18757119, -0.63252785])

                idx = np.where(slice_total_mask == 8)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 100*np.array([0.7, 0.5, -0.432])

                idx = np.where(slice_total_mask == 9)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 100*np.array([0.7, 0.5, -0.432])

                idx = np.where(slice_total_mask == 11)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 100*np.array([0.7, 0.5, -0.432])

                idx = np.where(slice_total_mask == 12)
                slice_mask = np.zeros(slice_total_mask.shape, dtype = np.bool_)
                slice_mask[idx] = True
                points[slice_mask] = points[slice_mask] + 100*np.array([0.7, 0.5, -0.432])
            if (normals.shape[0] == 0) or (count_normal):
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=50))  # 50 30
                o3d.io.write_point_cloud(pcd_name, pcd)
                pcd = o3d.io.read_point_cloud(pcd_name)
                points = np.asarray(pcd.points)
                normals = np.asarray(pcd.normals)
                colors = np.asarray(pcd.colors)
        elif (datasetName == 'tanks_COLMAP'):  # zhiwei
            gt_name = gtNamePattern.replace('#', str(model))
            pcd_name = os.path.join(datasetFolder, gt_name)
            pcd = o3d.io.read_point_cloud(pcd_name)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            colors = np.asarray(pcd.colors)
            if (normals.shape[0] == 0) or (count_normal):
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=50))  # 50 30
                o3d.io.write_point_cloud(pcd_name, pcd)
                pcd = o3d.io.read_point_cloud(pcd_name)
                points = np.asarray(pcd.points)
                normals = np.asarray(pcd.normals)
                colors = np.asarray(pcd.colors)
        elif (datasetName == 'blendedMVS'):  # zhiwei
            gt_name = gtNamePattern.replace('#', str(model))
            pcd_name = os.path.join(datasetFolder, gt_name)
            pcd = o3d.io.read_point_cloud(pcd_name)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            colors = np.asarray(pcd.colors)
            if (normals.shape[0] == 0) or (count_normal):
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=50))  # 50 30
                o3d.io.write_point_cloud(pcd_name, pcd)
                pcd = o3d.io.read_point_cloud(pcd_name)
                points = np.asarray(pcd.points)
                normals = np.asarray(pcd.normals)
                colors = np.asarray(pcd.colors)
        elif (datasetName == 'giga_ours'):  # zhiwei
            gt_name = gtNamePattern.replace('#', str(model))
            pcd_name = os.path.join(datasetFolder, gt_name)
            pcd = o3d.io.read_point_cloud(pcd_name)
            points = np.asarray(pcd.points)
            normals = np.asarray(pcd.normals)
            colors = np.asarray(pcd.colors)
            if (normals.shape[0] == 0) or (count_normal):
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=50))  # 50 30
                o3d.io.write_point_cloud(pcd_name, pcd)
                pcd = o3d.io.read_point_cloud(pcd_name)
                points = np.asarray(pcd.points)
                normals = np.asarray(pcd.normals)
                colors = np.asarray(pcd.colors)

        gt_models.append((points, normals, colors))

    return gt_models

def ply2array(datasetName,
              datasetFolder,
              gtNamePattern,
              modelList,
              ):

    gt_models = []
    for model in modelList:
        if (datasetName == 'DTU'):
            gt_name = gtNamePattern.replace('#', str(model).zfill(3))
            pcd_name = os.path.join(datasetFolder, gt_name)
            pcd = PlyData.read(pcd_name)
            count = pcd.elements[0].count

            pcd_xyz = np.c_[pcd['vertex']['x'], pcd['vertex']['y'], pcd['vertex']['z']]
            pcd_rgb = np.c_[pcd['vertex']['red'], pcd['vertex']['green'], pcd['vertex']['blue']]
        elif (_datasetName == 'MVS2d'):
            # pcd_name = os.path.join(self.datasetFolder, self.stl_name)
            pcd = np.load(self.stl_comb_name)

            count, _, _ = pcd[0].shape
            self.pcd_xyz = np.zeros((count, 3))
            self.pcd_xyz[:, 0] = pcd[0, :, 0, 1]
            self.pcd_xyz[:, 2] = pcd[0, :, 0, 0]

        gt_models.append((pcd_xyz, pcd_rgb))

    return gt_models

def zbuf_load(modelList,
              viewList,
              img_shape,
              params,
              ):
    (img_h, img_w) = img_shape
    for model_index, model_i in enumerate(modelList):


        file_root_zbuf_numpy = os.path.join(params.root_file, params.renderResultRoot_numpy,str(model_i), 'zbuf')
        # if not os.path.exists(file_root_zbuf_numpy):
        #     os.makedirs(file_root_zbuf_numpy)
        if params.load_zbuf:
            
            zbuf_list = np.zeros((len(modelList), len(viewList), img_h, img_w))
            for view_index, view_i in enumerate(viewList):
                zbuf = np.load(os.path.join(file_root_zbuf_numpy,'%s.npy' % str(view_i).zfill(8)))  # shape: (1,H,W)
                #pdb.set_trace()
                zbuf_list[model_index, view_index] = zbuf[0]
        else:
            zbuf_list = np.zeros((len(modelList), len(viewList), img_h, img_w))

    return zbuf_list

def mask_load(modelList,
              viewList,
              img_shape,
              params,
              ):
    (img_h, img_w) = img_shape
    for model_index, model_i in enumerate(modelList):


        file_root_mask_numpy = os.path.join(params.root_file, params.renderResultRoot_numpy,str(model_i), 'mask')
        # if not os.path.exists(file_root_zbuf_numpy):
        #     os.makedirs(file_root_zbuf_numpy)
        if params.load_mask:
            
            mask_list = np.zeros((len(modelList), len(viewList), img_h, img_w))
            for view_index, view_i in enumerate(viewList):
                mask = np.load(os.path.join(file_root_mask_numpy,'%s.npy' % str(view_i).zfill(8)))  # shape: (H,W)
                #pdb.set_trace()
                mask_list[model_index, view_index] = mask
        else:
            mask_list = np.zeros((len(modelList), len(viewList), img_h, img_w))

    return mask_list


def down_sample(datasetName,
              datasetFolder,
              gtNamePattern,
              gtFileNamePattern,
              gtRenderNamePattern,
              resol_gt_render,
              modelList,
              ):

    gt_models = []
    for model in modelList:
        if (datasetName == 'DTU'):
            gt_name = gtNamePattern.replace('#', str(model))
            datasetFolder_temper = '/home/jinzhi/hdd10T/mvs/result'
            pcd_name = os.path.join(datasetFolder_temper, gt_name)
            print(pcd_name)
            pcd = o3d.io.read_point_cloud(pcd_name)
            normals = np.asarray(pcd.normals)
            #if (normals.shape[0] == 0):
                #pcd.estimate_normals()    
            #o3d.io.write_point_cloud(pcd_name, pcd)
            #pcd = o3d.io.read_point_cloud(pcd_name)

            pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=resol_gt_render)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=resol_gt_render * 4, max_nn=40))  # 50 30
            print('estimate_normals')
            gt_render_name = gtRenderNamePattern.replace('#', str(model))
            pcd_render_name = os.path.join(datasetFolder, gt_render_name)
            pcd_render_name_file = os.path.join(datasetFolder, gtFileNamePattern.replace('#', str(model)))
            
            if not os.path.exists(pcd_render_name_file):
                os.makedirs(pcd_render_name_file)
            o3d.io.write_point_cloud(pcd_render_name, pcd)

if __name__ == '__main__':
    _input_data_rootFld = '/home/jinzhi/hdd10T/aMyLab/dataset'  # '/home/jinzhi/hdd10T/aMyLab/dataset' # '/home/frank/aMyLab/sgan/inputs'
    datasetFolder = os.path.join(_input_data_rootFld, 'DTU_MVS')

    for resol_gt_render in [1.0,2.0,4.0,8.0]:
        down_sample(datasetName = 'DTU',
                    datasetFolder = datasetFolder,
                    gtNamePattern = 'scan#_1.ply',
                    gtFileNamePattern = 'preprocess/ply_rmvs/#/',
                    gtRenderNamePattern = 'preprocess/ply_rmvs/#/$_surface_xyz_flow.ply'.replace('$', str(resol_gt_render)),
                    resol_gt_render = resol_gt_render,
                    modelList = [12,13,15,24,29,32,33,34,48,49,62,75,77,110,118]
                    )
