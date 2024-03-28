import open3d as o3d
import pdb
import numpy as np
import os
from plyfile import PlyData, PlyElement
import math
import copy

def initializeCubes(resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB):
    """
        generate {N_cubes} 3D overlapping cubes, each one has {N_cubeParams} embeddings
        for the cube with size of cube_D^3 the valid prediction region is the center part, say, cube_Dcenter^3
        E.g. cube_D=32, cube_Dcenter could be = 20. Because the border part of each cubes don't have accurate prediction because of ConvNet.

        ---------------
        inputs:
            resol: resolusion of each voxel in the CVC (mm)
            cube_D: size of the CVC (Colored Voxel Cube)
            cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
            cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
            BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
        outputs:
            cubes_param_np: (N_cubes, N_params) np.float32
            cube_D_mm: scalar

        ---------------
        usage:
        >>> cubes_param_np, cube_D_mm = initializeCubes(resol=1, cube_D=22, cube_Dcenter=10, cube_overlapping_ratio=0.5, BB=np.array([[3,88],[-11,99],[-110,-11]]))
        xyz bounding box of the reconstructed scene: [ 3 88], [-11  99], [-110  -11]
        >>> print cubes_param_np[:3]
        [([   3.,  -11., -110.], [0, 0, 0],  1.)
         ([   3.,  -11., -105.], [0, 0, 1],  1.)
         ([   3.,  -11., -100.], [0, 0, 2],  1.)]
        >>> print cubes_param_np['xyz'][18:22]
        [[   3.  -11.  -20.]
         [   3.  -11.  -15.]
         [   3.   -6. -110.]
         [   3.   -6. -105.]]
        >>> np.allclose(cubes_param_np['xyz'][18:22], cubes_param_np[18:22]['xyz'])
        True
        >>> print cube_D_mm
        22
        """

    cube_D_mm = resol * cube_D   # D size of each cube along each axis,
    cube_Center_D_mm = resol * cube_Dcenter   # D size of each cube's center that is finally remained
    cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio # the distance between adjacent cubes,
    safeMargin = (cube_D_mm - cube_Center_D_mm)/2

    print('xyz bounding box of the reconstructed scene: {}, {}, {}'.format(*BB))


    N_along_axis = lambda _min, _max, _resol: int(math.ceil((_max - _min) / _resol))
    N_along_xyz = [N_along_axis( (BB[_axis][0] - safeMargin), (BB[_axis][1] + safeMargin), cube_stride_mm) for _axis in range(3)]   # how many cubes along each axis
    # store the ijk indices of each cube, in order to localize the cube
    cubes_ijk = np.indices(tuple(N_along_xyz))
    N_cubes = int(cubes_ijk.size / 3)   # how many cubes
    print('total cubic number: ', N_cubes)
    cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), ('resol', np.float32)])    # attributes for each CVC (colored voxel cube)
    cubes_param_np['ijk'] = cubes_ijk.reshape([3,-1]).T  # i/j/k grid index
    cubes_xyz_min = cubes_param_np['ijk'] * cube_stride_mm + (BB[:,0][None,:] - safeMargin)
    cubes_param_np['xyz'] = cubes_xyz_min    # x/y/z coordinates (mm)
    cubes_param_np['resol'] = resol

    return cubes_param_np, cube_D_mm


def quantizePts2Cubes(pts_xyz, resol, cube_D, cube_Dcenter, cube_overlapping_ratio, BB=None, type_v='round'):
    """
    generate overlapping cubes covering a set of points which is denser, so that we need to quantize the pts' coords.

    --------
    inputs:
        pts_xyz: generate the cubes around these pts
        resol: resolusion of each voxel in the CVC (mm)
        cube_D: size of the CVC (Colored Voxel Cube)
        cube_Dcenter: only keep the center part of the CVC, because of the boundery effect of ConvNet.
        cube_overlapping_ratio: pertantage of the CVC are covered by the neighboring ones
        BB: bounding box, numpy array: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]

    --------
    outputs:
        cubes_param_np: (N_cubes, N_params) np.float32
        cube_D_mm: scalar

    --------
    examples:
    >>> pts_xyz = np.array([[-1, 2, 0], [0, 2, 0], [1, 2, 0], [0,1,0], [0,0,0], [1,0,0], [2.1,0,0]])
    >>> #TODO quantizePts2Cubes(pts_xyz, resol=2, cube_D=3, cube_Dcenter = 2, cube_overlapping_ratio = 0.5)
    """

    cube_D_mm = resol * cube_D  # D size of each cube along each axis,
    cube_Center_D_mm = resol * cube_Dcenter  # D size of each cube's center that is finally remained
    cube_stride_mm = cube_Center_D_mm * cube_overlapping_ratio  # the distance between adjacent cubes,
    cube_down_mm = cube_Center_D_mm * cube_overlapping_ratio

    # cube_stride_mm = cube_Center_D_mm * 0.4 # the distance between adjacent cubes,
    # cube_down_mm = cube_Center_D_mm * 0.05



    safeMargin = (cube_D_mm - cube_Center_D_mm) / 2
    #safeMargin = cube_D_mm / 2  # a little bit bigger than the BB
    inBB = np.array([np.logical_and(pts_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin),
                                    pts_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(
        axis=0)
    pts_xyz = pts_xyz[inBB]

    # pts_xyz = np.concatenate((pts_xyz,pts_xyz+np.array([-cube_stride_mm,0,0]),pts_xyz+np.array([0,-cube_stride_mm,0]),pts_xyz+np.array([0,0,-cube_stride_mm]),pts_xyz+np.array([cube_stride_mm,0,0]),pts_xyz+np.array([0,cube_stride_mm,0]),pts_xyz+np.array([0,0,cube_stride_mm])))
    # pts_xyz = np.concatenate((pts_xyz,pts_xyz+np.array([0,-cube_stride_mm,0]),pts_xyz+np.array([cube_stride_mm,0,0]),pts_xyz+np.array([0,0,cube_stride_mm])))

    # NMO
    if (True):
        print('pts_xyz.min',pts_xyz.min(axis=0)[
            None, ...])
        shift = pts_xyz.min(axis=0)[None, ...]  # (1, 3), make sure the cube_ijk is non-negative, and try to cover the pts in the middle of the cubes.

        if (type_v is 'round'):
            cubes_ijk = np.round((pts_xyz - shift) / cube_stride_mm)
            #print('cubes_ijk', cubes_ijk)
        else:
            cubes_ijk_floor = (pts_xyz - shift) // cube_stride_mm  # (N_pts, 3)
            cubes_ijk_ceil = ((pts_xyz - shift) // cube_stride_mm + 1)  # for each pt consider 2 neighboring cubesalong each axis.
            cubes_ijk = np.vstack([cubes_ijk_floor, cubes_ijk_ceil])  # (2*N_pts, 3)
        cubes_ijk_1d = cubes_ijk.view(dtype=cubes_ijk.dtype.descr * 3)
        cubes_ijk_unique = np.unique(cubes_ijk_1d).view(cubes_ijk.dtype).reshape((-1, 3))  # (N_cubes, 3)
        N_cubes = cubes_ijk_unique.shape[0]  # how many cubes

        print('total cubic number: ', N_cubes)
        cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), (
        'resol', np.float32)])  # attributes for each CVC (colored voxel cube)
        cubes_param_np['ijk'] = cubes_ijk_unique  # i/j/k grid index
        cubesCenter_xyz = cubes_param_np['ijk'] * cube_stride_mm + shift
        cubes_param_np['xyz'] = cubesCenter_xyz - cube_D_mm / 2  # (N_cubes, 3) min of x/y/z coordinates (mm)
        cubes_param_np['resol'] = resol

        #print('cubes_param_np',cubes_param_np['xyz'])
    else:
        print('use new quantizePts2Cubes')
        N_cubes = pts_xyz.shape[0]  # how many cubes

        cubes_param_np = np.empty((N_cubes,), dtype=[('xyz', np.float32, (3,)), ('ijk', np.uint32, (3,)), (
        'resol', np.float32)])  # attributes for each CVC (colored voxel cube)
        cubes_param_np['ijk'] = 0  # i/j/k grid index
        cubes_param_np['xyz'] = pts_xyz - cube_D_mm / 2  # (N_cubes, 3) min of x/y/z coordinates (mm)
        cubes_param_np['resol'] = resol

    return cubes_param_np, cube_D_mm


def gen_sparse_multi_ijkxyzrgbn(pts_xyz, pts_rgb, resol, stage_num, xyz_3D, cube_D, resol_gt = 0.2, ply_filePath = None, estimate_normal = False):

    '''

    get multiple quantize voxel of GT of a cube at different scaling
    :param pts_xyz:
        shape:(N_points, 3)
    :param pts_rgb: type:int
        shape:(N_points, 3)
    :param resol:
        float
    :param stage_num: int
    :param xyz_3D:
        shape:(3)
    :param cube_D:
        int
    :return:
        list of info at diffetent stage
            len(list) = stage_num
            element:
                 cubes_gt_np: the ijk index of GT points
                 length: N_points
                 type: dtype=[('ijk', np.uint32, (3,)), ('ijk_id', np.uint32),('ijk_id_s', np.uint32), ('xyz_global', np.float32, (3,)), ('xyz', np.float32, (3,)), ('rgb', np.float32, (3,)), ('normals', np.float32, (3,))])


    '''

    info_list = []

    resol_new = resol
    xyz_3D_new = copy.copy(xyz_3D)
    cube_D_new = cube_D
    cubes_gt_id_big = None

    for i in range(stage_num):
        if(ply_filePath)is not None:
            ply_filePath_new = os.path.join(ply_filePath,'stage:%s_'%str(i))
        else:
            ply_filePath_new = None

        cubes_gt_np = quantizeGt_sparse_2_ijkxyzrgbn(pts_xyz=pts_xyz,
                                                                  pts_rgb=pts_rgb,
                                                                  resol=resol_new,
                                                                  xyz_3D=xyz_3D_new,
                                                                  cube_D=int(cube_D_new),
                                                                  resol_gt=resol_gt,
                                                                  ply_filePath=ply_filePath_new,
                                                                  estimate_normal = estimate_normal
                                                                  )
        if(cubes_gt_id_big is not None):
            cubes_dense_np = np.zeros((int(cube_D_new * cube_D_new * cube_D_new),),
                                      dtype=[('ijk', np.uint32, (3,)), ('ijk_id', np.uint64),('ijk_id_s', np.uint64),
                                             ('xyz_global', np.float32, (3,)), ('xyz', np.float32, (3,)),
                                             ('rgb', np.float32, (3,)), ('normals', np.float32, (3,))])

            N = np.arange(int(cube_D_new * cube_D_new * cube_D_new))
            i_N = N // (cube_D_new * cube_D_new)
            j_N = (N - i_N * cube_D_new * cube_D_new) // (cube_D_new)
            k_N = (N - i_N * cube_D_new * cube_D_new - j_N * cube_D_new)
            cubes_dense_np['ijk_id'] = 8 * ((cube_D_new / 2) * (cube_D_new / 2) * (i_N // 2) + (cube_D_new / 2) * (j_N // 2) + k_N // 2) + (
                           4 * (i_N % 2) + 2 * (j_N % 2) + k_N % 2)
            #cubes_dense_np[cubes_gt_np['ijk_id_s']]['ijk_id']

            cubes_dense_np[cubes_gt_np['ijk_id_s']] = cubes_gt_np
            #pdb.set_trace()

            cubes_gt_np = cubes_dense_np[cubes_gt_id_big]

        cubes_gt_id_big = np.unique(cubes_gt_np['ijk_id']//8)
        info_list.append((cubes_gt_np))
        xyz_3D_new += (resol_new / 2)
        resol_new *= 2
        cube_D_new /= 2

    return info_list



def gen_multi_ijkxyzrgbn(pts_xyz, pts_rgb, resol, stage_num, xyz_3D, cube_D, resol_gt = 0.2, ply_filePath = None, estimate_normal = True):

    '''

    get multiple quantize voxel of GT of a cube at different scaling
    :param pts_xyz:
        shape:(N_points, 3)
    :param pts_rgb: type:int
        shape:(N_points, 3)
    :param resol:
        float
    :param stage_num: int
    :param xyz_3D:
        shape:(3)
    :param cube_D:
        int
    :return:
        list of info at diffetent stage
            len(list) = stage_num
            element:
                 cubes_gt_np: the ijk index of GT points
                 quanti_gt:voxelized results.  numpy array bool
                    shape: (cube_D,cube_D,cube_D,1)
                 info_gt: voxelized results and info of each voxel
                    shape: (cube_D,cube_D,cube_D,10)
                    each ele :
                        shape:(10)
                        contents:(voxilization, x,y,z,r,g,b,nx,ny,nz)
    '''

    info_list = []

    if (ply_filePath) is not None:
        ply_filePath_new = os.path.join(ply_filePath, 'stage:%s_' % str(-1))
    else:
        ply_filePath_new = None
    cubes_gt_np, quanti_gt, info_gt = quantizeGt_2_ijkxyzrgbn(pts_xyz=pts_xyz,
                                                              pts_rgb=pts_rgb,
                                                              resol=resol/2,
                                                              xyz_3D=xyz_3D-resol/4,
                                                              cube_D=int(cube_D*2),
                                                              resol_gt=resol_gt,
                                                              ply_filePath=ply_filePath_new,
                                                              estimate_normal=estimate_normal
                                                              )
    info_list.append((cubes_gt_np, quanti_gt, info_gt))
    resol_new = resol
    xyz_3D_new = copy.copy(xyz_3D)
    cube_D_new = cube_D

    for i in range(stage_num):
        if(ply_filePath)is not None:
            ply_filePath_new = os.path.join(ply_filePath,'stage:%s_'%str(i))
        else:
            ply_filePath_new = None
        cubes_gt_np, quanti_gt, info_gt = quantizeGt_2_ijkxyzrgbn(pts_xyz=pts_xyz,
                                                                  pts_rgb=pts_rgb,
                                                                  resol=resol_new,
                                                                  xyz_3D=xyz_3D_new,
                                                                  cube_D=int(cube_D_new),
                                                                  resol_gt=resol_gt,
                                                                  ply_filePath=ply_filePath_new,
                                                                  estimate_normal = estimate_normal
                                                                  )
        info_list.append((cubes_gt_np, quanti_gt, info_gt))
        xyz_3D_new += (resol_new / 2)
        resol_new *= 2
        cube_D_new /= 2

    return info_list


def quantizeGt_sparse_2_ijkxyzrgbn(pts_xyz, pts_rgb, resol, xyz_3D, cube_D, resol_gt = 0.2, ply_filePath = None, estimate_normal = True):
    '''
    get the sparse quantize voxel of GT of a cube
    :param pts_xyz:
        shape:(N_points, 3)
    :param pts_rgb: type:int
        shape:(N_points, 3)
    :param resol:
        float
    :param xyz_3D:
        shape:(3)
    :param cube_D:
        int
    :param estimate_normal
        bool
    :return:
         cubes_gt_np: the ijk index of GT points
         length: N_points
         type: dtype=[('ijk', np.uint32, (3,)), ('ijk_id', np.uint32),('ijk_id_s', np.uint32), ('xyz_global', np.float32, (3,)), ('xyz', np.float32, (3,)), ('rgb', np.float32, (3,)), ('normals', np.float32, (3,))])


    '''
    BB = np.zeros((3,2))
    BB[:,0] = xyz_3D
    BB[:,1] = xyz_3D + resol * (cube_D - 1)
    safeMargin = 0.5 * resol
    inBB = np.array([np.logical_and(pts_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin),
                                    pts_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(axis=0)
    pts_xyz = pts_xyz[inBB]
    pts_rgb = pts_rgb[inBB]

    pp = o3d.geometry.PointCloud()
    #print(pp.points)

    #pp.points = o3d.utility.Vector3dVector(pts_xyz)
    pp.points = o3d.Vector3dVector(pts_xyz.astype(np.float64))

    if(estimate_normal):
        if(resol>4 * resol_gt):
            o3d.estimate_normals(pp, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=resol / 2, max_nn=30))
            #pp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            #    radius=resol / 2, max_nn=30))
        else:
            o3d.estimate_normals(pp, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=resol_gt * 2, max_nn=30))
            #pp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            #    radius=resol_gt * 2, max_nn=30))
    else:
        pp.normals = o3d.Vector3dVector(np.zeros(np.shape(pts_xyz)))

    #pp.colors = o3d.utility.Vector3dVector(pts_rgb/256.0)
    pp.colors = o3d.Vector3dVector(pts_rgb / 256.0)
    #print(pts_rgb/256.0)
    #print(numpy.asarray(pp.colors))
    #o3d.visualization.draw_geometries([pp])

    #downpcd, ijk = pp.voxel_down_sample_and_trace(voxel_size=resol,
    #                                              min_bound=xyz_3D[:,None] - safeMargin,
    #                                              max_bound=xyz_3D[:,None] - safeMargin + resol * cube_D,
    #                                              approximate_class=False)
    downpcd, ijk = o3d.geometry.voxel_down_sample_and_trace(input = pp,
                                                   voxel_size=resol,
                                                  min_bound=(xyz_3D[:, None] - safeMargin).astype(np.float64),
                                                  max_bound=(xyz_3D[:, None] - safeMargin + resol * cube_D).astype(np.float64),
                                                            approximate_class = False
                                                   )
    #create_from_point_cloud_within_bounds(input, voxel_size, min_bound, max_bound)
    #downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #        radius=resol * 2, max_nn=30))
    pcd_xyz = (np.asarray(downpcd.points))
    pcd_rgb = (np.asarray(downpcd.colors))
    pcd_normals = (np.asarray(downpcd.normals))
    inBB = np.array([np.logical_and(pcd_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin),
                                    pcd_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(
        axis=0)
    pcd_xyz = pcd_xyz[inBB]
    pcd_rgb = pcd_rgb[inBB]
    pcd_normals = pcd_normals[inBB]

    #o3d.visualization.draw_geometries([downpcd])
    #print()
    shift = xyz_3D[None, ...]  # 0 coordinate
    cubes_ijk = np.round((pcd_xyz - shift) / resol)  # (N_pts, 3)
    inBB = np.array([np.logical_and(cubes_ijk[:, _axis] >= 0,
                                    cubes_ijk[:, _axis] < cube_D) for _axis in range(3)]).all(
        axis=0)
    cubes_ijk = cubes_ijk[inBB]
    cubes_ijk_1d = cubes_ijk.view(dtype=cubes_ijk.dtype.descr * 3)
    cubes_ijk_unique, cubes_ijk_idx = np.unique(cubes_ijk_1d, return_index=True)
    cubes_ijk_unique = cubes_ijk_unique.view(cubes_ijk.dtype).reshape((-1, 3))  # (N_cubes, 3)

    pcd_xyz = pcd_xyz[cubes_ijk_idx, :]
    pcd_rgb = pcd_rgb[cubes_ijk_idx, :]
    pcd_normals = pcd_normals[cubes_ijk_idx, :]

    if not(cubes_ijk_unique.shape[0] == pcd_xyz.shape[0]):
        print('cubes_ijk_unique.shape[0] == pcd_xyz.shape[0]')
        print('GT voxel sample went wrong')
        raise NameError

    N_cubes = cubes_ijk_unique.shape[0]  # how many cubes
    print('N_voxels', N_cubes)

    cubes_gt_np = np.empty((N_cubes,), dtype=[('ijk', np.uint32, (3,)), ('ijk_id', np.uint64),('ijk_id_s', np.uint64), ('xyz_global', np.float32, (3,)), ('xyz', np.float32, (3,)), ('rgb', np.float32, (3,)), ('normals', np.float32, (3,))])
    cubes_gt_np['ijk'] = cubes_ijk_unique  # i/j/k grid index
    ijk = cubes_ijk_unique
    id_l = 8 * ((cube_D/2) * (cube_D/2) * (ijk[:, 0] // 2) + (cube_D/2) * (ijk[:, 1] // 2) + ijk[:, 2] // 2) + (
                4 * (ijk[:, 0] % 2) + 2 * (ijk[:, 1] % 2) + ijk[:, 2] % 2)
    id_s = (cube_D * cube_D * ijk[:, 0] + cube_D * ijk[:, 1] + ijk[:, 2])
    cubes_gt_np['ijk_id'] = id_l
    cubes_gt_np['ijk_id_s'] = id_s
    cubes_gt_np['xyz'] = (pcd_xyz - shift - cubes_gt_np['ijk'] * resol)/resol
    cubes_gt_np['xyz_global'] = shift + cubes_gt_np['ijk'] * resol
    cubes_gt_np['rgb'] = pcd_rgb
    cubes_gt_np['normals'] = pcd_normals


    print('the percentage of points in quanti cubes: ', N_cubes/(cube_D*cube_D*cube_D))

    #if (ply_filePath) is not None:
    if False:
        ply_file_downppp = os.path.join(ply_filePath, 'downppp.ply')
        save2ply(ply_file_downppp, pcd_xyz, rgb_np=(pcd_rgb * 256), normal_np=pcd_normals)
        ply_file_ijk = os.path.join(ply_filePath, 'ijk.ply')
        save2ply(ply_file_ijk, shift + cubes_gt_np['ijk'] * resol)

    return cubes_gt_np



def quantizeGt_2_ijkxyzrgbn(pts_xyz, pts_rgb, resol, xyz_3D, cube_D, resol_gt = 0.2, ply_filePath = None, estimate_normal = True):
    '''
    get the quantize voxel of GT of a cube
    :param pts_xyz:
        shape:(N_points, 3)
    :param pts_rgb: type:int
        shape:(N_points, 3)
    :param resol:
        float
    :param xyz_3D:
        shape:(3)
    :param cube_D:
        int
    :param estimate_normal
        bool
    :return:
         cubes_gt_np: the ijk index of GT points
         quanti_gt:voxelized results.  numpy array bool
            shape: (cube_D,cube_D,cube_D,1)
         info_gt: voxelized results and info of each voxel
            shape: (cube_D,cube_D,cube_D,10)
            each ele :
                shape:(10)
                contents:(voxilization, x,y,z,r,g,b,nx,ny,nz)
    '''
    BB = np.zeros((3,2))
    BB[:,0] = xyz_3D
    BB[:,1] = xyz_3D + resol * (cube_D - 1)
    safeMargin = 0.5 * resol
    inBB = np.array([np.logical_and(pts_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin),
                                    pts_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(axis=0)
    pts_xyz = pts_xyz[inBB]
    pts_rgb = pts_rgb[inBB]

    pp = o3d.geometry.PointCloud()
    #print(pp.points)

    #pp.points = o3d.utility.Vector3dVector(pts_xyz)
    pp.points = o3d.Vector3dVector(pts_xyz.astype(np.float64))

    if(estimate_normal):
        if(resol>4 * resol_gt):
            o3d.estimate_normals(pp, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=resol / 2, max_nn=30))
            #pp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            #    radius=resol / 2, max_nn=30))
        else:
            o3d.estimate_normals(pp, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=resol_gt * 2, max_nn=30))
            #pp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            #    radius=resol_gt * 2, max_nn=30))
    else:
        pp.normals = o3d.Vector3dVector(np.zeros(np.shape(pts_xyz)))

    #pp.colors = o3d.utility.Vector3dVector(pts_rgb/256.0)
    pp.colors = o3d.Vector3dVector(pts_rgb / 256.0)
    #print(pts_rgb/256.0)
    #print(numpy.asarray(pp.colors))
    #o3d.visualization.draw_geometries([pp])

    #downpcd, ijk = pp.voxel_down_sample_and_trace(voxel_size=resol,
    #                                              min_bound=xyz_3D[:,None] - safeMargin,
    #                                              max_bound=xyz_3D[:,None] - safeMargin + resol * cube_D,
    #                                              approximate_class=False)
    downpcd, ijk = o3d.geometry.voxel_down_sample_and_trace(pp,
                                                   voxel_size=resol,
                                                  min_bound=(xyz_3D[:, None] - safeMargin).astype(np.float64),
                                                  max_bound=(xyz_3D[:, None] - safeMargin + resol * cube_D).astype(np.float64),

                                                   )
    #create_from_point_cloud_within_bounds(input, voxel_size, min_bound, max_bound)
    #downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #        radius=resol * 2, max_nn=30))
    pcd_xyz = (np.asarray(downpcd.points))
    pcd_rgb = (np.asarray(downpcd.colors))
    pcd_normals = (np.asarray(downpcd.normals))

    #o3d.visualization.draw_geometries([downpcd])
    #print()
    shift = xyz_3D[None, ...]  # 0 coordinate
    cubes_ijk = np.round((pcd_xyz - shift) / resol)  # (N_pts, 3)
    cubes_ijk_1d = cubes_ijk.view(dtype=cubes_ijk.dtype.descr * 3)
    cubes_ijk_unique, cubes_ijk_idx = np.unique(cubes_ijk_1d, return_index=True)
    cubes_ijk_unique = cubes_ijk_unique.view(cubes_ijk.dtype).reshape((-1, 3))  # (N_cubes, 3)

    pcd_xyz = pcd_xyz[cubes_ijk_idx, :]
    pcd_rgb = pcd_rgb[cubes_ijk_idx, :]
    pcd_normals = pcd_normals[cubes_ijk_idx, :]

    if not(cubes_ijk_unique.shape[0] == pcd_xyz.shape[0]):
        print('cubes_ijk_unique.shape[0] == pcd_xyz.shape[0]')
        print('GT voxel sample went wrong')
        raise NameError

    N_cubes = cubes_ijk_unique.shape[0]  # how many cubes
    print('N_voxels', N_cubes)

    cubes_gt_np = np.empty((N_cubes,), dtype=[('ijk', np.uint32, (3,)), ('xyz', np.float32, (3,)), ('rgb', np.float32, (3,)), ('normals', np.float32, (3,))])
    cubes_gt_np['ijk'] = cubes_ijk_unique  # i/j/k grid index
    cubes_gt_np['xyz'] = (pcd_xyz - shift - cubes_gt_np['ijk'] * resol)/resol

    cubes_gt_np['rgb'] = pcd_rgb
    cubes_gt_np['normals'] = pcd_normals
    quanti_gt = np.empty((cube_D, cube_D, cube_D, 1), dtype = np.bool) * False
    info_gt = np.zeros((cube_D, cube_D, cube_D, 10), dtype = np.float32)
    for ele in cubes_gt_np:
        ijk = ele['ijk']
        info_ele = np.concatenate((np.array([1.0]),ele['xyz'],ele['rgb'],ele['normals']))

        if((ijk[0] < cube_D) and (ijk[1] < cube_D) and (ijk[2] < cube_D) and (ijk[0] >=0) and (ijk[1] >=0) and (ijk[2] >=0)):
            info_gt[ijk[0], ijk[1], ijk[2],:] = info_ele

    for ijk in cubes_gt_np['ijk']:
        if ((ijk[0] < cube_D) and (ijk[1] < cube_D) and (ijk[2] < cube_D) and (ijk[0] >=0) and (ijk[1] >=0) and (ijk[2] >=0)):
            quanti_gt[ijk[0],ijk[1],ijk[2],:] = True

    print('the percentage of points in quanti cubes: ', (quanti_gt.sum()+0.0)/(cube_D*cube_D*cube_D))

    #if (ply_filePath) is not None:
    if False:
        ply_file_downppp = os.path.join(ply_filePath, 'downppp.ply')
        save2ply(ply_file_downppp, pcd_xyz, rgb_np=(pcd_rgb * 256), normal_np=pcd_normals)
        ply_file_ijk = os.path.join(ply_filePath, 'ijk.ply')
        save2ply(ply_file_ijk, shift + cubes_gt_np['ijk'] * resol)

    return cubes_gt_np, quanti_gt, info_gt


def quantizeGt_2_ijk(pts_xyz, resol, xyz_3D, cube_D):
    '''
    get the quantize voxel of GT of a cube
    :param pts_xyz:
        shape:(N_points, 3)
    :param resol:
        float
    :param xyz_3D:
        shape:(3)
    :param cube_D:
        int
    :return:
         cubes_gt_np: the ijk index of GT points
         quanti_gt:voxelized results.  numpy array bool
            shape: (cube_D,cube_D,cube_D,1)
    '''
    BB = np.zeros((3,2))
    BB[:,0] = xyz_3D
    BB[:,1] = xyz_3D + resol * (cube_D - 1)
    safeMargin = 0.5 * resol

    print('BB',BB)
    inBB = np.array([np.logical_and(pts_xyz[:, _axis] >= (BB[_axis, 0] - safeMargin),
                                    pts_xyz[:, _axis] <= (BB[_axis, 1] + safeMargin)) for _axis in range(3)]).all(axis=0)
    pts_xyz = pts_xyz[inBB]
    print('pts_xyz', pts_xyz.shape)


    shift = xyz_3D[None, ...]  # 0 coordinate
    cubes_ijk = np.round((pts_xyz - shift) / resol)  # (N_pts, 3)
    cubes_ijk_1d = cubes_ijk.view(dtype=cubes_ijk.dtype.descr * 3)
    print('cubes_ijk_1d', cubes_ijk_1d)
    cubes_ijk_unique = np.unique(cubes_ijk_1d).view(cubes_ijk.dtype).reshape((-1, 3))  # (N_cubes, 3)

    N_cubes = cubes_ijk_unique.shape[0]  # how many cubes
    print('N_cubes', N_cubes)


    cubes_gt_np = np.empty((N_cubes,), dtype=[('ijk', np.uint32, (3,))])
    cubes_gt_np['ijk'] = cubes_ijk_unique  # i/j/k grid index

    quanti_gt = np.empty((cube_D, cube_D, cube_D, 1), dtype = np.bool) * False
    for ijk in cubes_gt_np['ijk']:
        quanti_gt[ijk[0],ijk[1],ijk[2]] = True
    print('the percentage of points in quanti cubes: ', (quanti_gt.sum()+0.0)/(cube_D*cube_D*cube_D))
    return cubes_gt_np, quanti_gt

def readPointCloud_xyz(self, pointCloudFile='xx/xx.ply'):
    pcd = PlyData.read(pointCloudFile)  # pcd for Point Cloud Data
    pcd_xyz = np.c_[pcd['vertex']['x'], pcd['vertex']['y'], pcd['vertex']['z']]
    return pcd_xyz


def readBB_fromModel(self, objFile='xx/xx.obj'):
    mesh = mesh_util.load_obj(filename=objFile)
    BB = np.c_[mesh.v.min(axis=0), mesh.v.max(axis=0)]  # (3, 2)
    return BB

def save2ply(ply_filePath, xyz_np, rgb_np=None, normal_np=None):
    """
    save data to ply file, xyz (rgb, normal)

    ---------
    inputs:
        xyz_np: (N_voxels, 3)
        rgb_np: None / (N_voxels, 3)
        normal_np: None / (N_voxels, 3)

        ply_filePath: 'xxx.ply'
    outputs:
        save to .ply file
    """
    N_voxels = xyz_np.shape[0]
    atributes = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if normal_np is not None:
        atributes += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
    if rgb_np is not None:
        atributes += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    saved_pts = np.zeros(shape=(N_voxels,), dtype=np.dtype(atributes))

    saved_pts['x'], saved_pts['y'], saved_pts['z'] = xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2]
    if rgb_np is not None:
        # print('saveed', saved_pts)
        saved_pts['red'], saved_pts['green'], saved_pts['blue'] = rgb_np[:, 0], rgb_np[:, 1], rgb_np[:, 2]
    if normal_np is not None:
        saved_pts['nx'], saved_pts['ny'], saved_pts['nz'] = normal_np[:, 0], normal_np[:, 1], normal_np[:, 2]

    el_vertex = PlyElement.describe(saved_pts, 'vertex')
    outputFolder = os.path.dirname(ply_filePath)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    PlyData([el_vertex]).write(ply_filePath)
    # print('saved ply file: {}'.format(ply_filePath))
    return 1
