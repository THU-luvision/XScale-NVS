import pdb
import copy
import numpy as np
import os
import scipy
import math
import torch
from PIL import Image


def read_total_poses(cam_file_path):
    with open(cam_file_path) as f:
        lines = f.readlines()

    index_list = np.array(list(range(len(lines))))
    index_poses = np.where((index_list % 5) == 0)
    index = index_list[index_poses]
    total_poses = []
    for i in index:
        pose = np.empty([4, 4]).astype(np.float32)
        pose[0, :] = np.array(lines[i + 1].rstrip().split(' ')[:4], dtype=np.float32)
        pose[1, :] = np.array(lines[i + 2].rstrip().split(' ')[:4], dtype=np.float32)
        pose[2, :] = np.array(lines[i + 3].rstrip().split(' ')[:4], dtype=np.float32)
        pose[3, :] = np.array(lines[i + 4].rstrip().split(' ')[:4], dtype=np.float32)
        pose_new = pose[:3, :4]
        # pose_new = np.linalg.inv(pose)
        # pose_new = np.matmul(trans_mat_inv,pose_new)[:3,:4]
        total_poses.append(pose_new)

    return total_poses


def readCameraRTK_as_np_tanks(cameraPO_file, datasetName):
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

    if datasetName == 'DTU':
        cameraPO = np.dot(cameraKO, cameraRTO)
    elif datasetName == 'tanks_COLMAP':

        cameraPO = np.dot(cameraKO, cameraRTO)
    elif datasetName == 'blendedMVS':

        cameraPO = np.dot(cameraKO, cameraRTO)
    elif datasetName == 'giga_ours':

        cameraPO = np.dot(cameraKO, cameraRTO)

    return cameraRTO, cameraKO


def readCameraP0_as_np_tanks(cameraPO_file, datasetName, ):
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

    if datasetName == 'DTU':
        cameraPO = np.dot(cameraKO, cameraRTO)
    elif datasetName == 'tanks_COLMAP':

        cameraPO = np.dot(cameraKO, cameraRTO)
    elif datasetName == 'blendedMVS':

        cameraPO = np.dot(cameraKO, cameraRTO)
    elif datasetName == 'giga_ours':

        cameraPO = np.dot(cameraKO, cameraRTO)

    return cameraPO


def __readCameraPO_as_np_DTU__(cameraPO_file):
    """
    only load a camera PO in the file
    ------------
    inputs:
        cameraPO_file: the camera pose file of a specific view
    outputs:
        cameraPO: np.float64 (3,4)
    ------------
    usage:
    >>> p = __readCameraPO_as_np_DTU__(cameraPO_file = './test/cameraPO/pos_060.txt')
    >>> p # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[  1.67373847e+03,  -2.15171320e+03,   1.26963515e+03,
        ...
              6.58552305e+02]])
    """
    cameraPO = np.loadtxt(cameraPO_file, dtype=np.float64, delimiter=' ')
    return cameraPO


def __readCameraPOs_as_np_Middlebury__(cameraPO_file, viewList):
    """
    load camera POs of multiple views in one file
    ------------
    inputs:
        cameraPO_file: the camera pose file of a specific view
        viewList: view list
    outputs:
        cameraPO: np.float64 (N_views,3,4)
    ------------
    usage:
    >>> p = __readCameraPOs_as_np_Middlebury__(cameraPO_file = './test/cameraPO/dinoSR_par.txt', viewList=[3,8])
    >>> p # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        array([[[ -1.22933223e+03,   3.08329199e+03,   2.02784015e+02,
        ...
        6.41227584e-01]]])
    """
    with open(cameraPO_file) as f:
        lines = f.readlines()

    cameraPOs = np.empty((len(lines), 3, 4)).astype(np.float64)
    for _n, _l in enumerate(lines):
        if _n == 0:
            continue
        _params = np.array(_l.strip().split(' ')[1:], dtype=np.float64)
        _K = _params[:9].reshape((3, 3))
        _R = _params[9:18].reshape((3, 3))
        _t = _params[18:].reshape((3, 1))
        cameraPOs[_n] = np.dot(_K, np.c_[_R, _t])
    return cameraPOs[viewList]


def readCameraPOs_as_np(
        datasetFolder,
        datasetName,
        poseNamePattern,
        # model,
        viewList,
        model=None,
):
    """
    inputs:
      datasetFolder: 'x/x/x/middlebury'
      datasetName: 'DTU' / 'Middlebury'
      #model: 1..128 / 'dinoxx'
      viewList: [3,8,21,...]
    output:
      cameraPOs (N_views,3,4) np.flost64
    """
    cameraPOs = np.empty((len(viewList), 3, 4), dtype=np.float64)
    cameraRTOs = np.empty((len(viewList), 3, 4), dtype=np.float64)
    cameraKOs = np.empty((len(viewList), 3, 3), dtype=np.float64)

    if 'Middlebury' in datasetName:
        cameraPOs = self.__readCameraPOs_as_np_Middlebury__(
            cameraPO_file=os.path.join(datasetFolder, poseNamePattern), viewList=viewList)
    elif datasetName == 'tanks':
        for _i, _view in enumerate(viewList):
            _cameraPO = readCameraP0_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                            poseNamePattern.replace('#',
                                                                                                    '{:03}'.format(
                                                                                                        _view)).replace(
                                                                                '@', '{}'.format(_view))))
            # _cameraPO = readCameraP0_as_np_tanks(cameraPO_file = datasetFolder+poseNamePattern.replace('#', '{:03}'.format(_view)).replace('@', '{}'.format(_view)))

            cameraPOs[_i] = _cameraPO
            _cameraRT, _cameraK = readCameraRTK_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                                       poseNamePattern.replace('#',
                                                                                                               '{:03}'.format(
                                                                                                                   _view)).replace(
                                                                                           '@', '{}'.format(_view))))
            cameraRTOs[_i] = _cameraRT
            cameraKOs[_i] = _cameraK
    elif datasetName == 'tanks_COLMAP':  # zhiwei

        for _i, _view in enumerate(viewList):
            _cameraPO = readCameraP0_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                            poseNamePattern.replace('#',
                                                                                                    '{:03}'.format(
                                                                                                        _view))),
                                                 datasetName=datasetName)
            # _cameraPO = readCameraP0_as_np_tanks(cameraPO_file = datasetFolder+poseNamePattern.replace('#', '{:03}'.format(_view)).replace('@', '{}'.format(_view)))

            cameraPOs[_i] = _cameraPO
            _cameraRT, _cameraK = readCameraRTK_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                                       poseNamePattern.replace('#',
                                                                                                               '{:03}'.format(
                                                                                                                   _view))),
                                                            datasetName=datasetName)
            cameraRTOs[_i] = _cameraRT
            cameraKOs[_i] = _cameraK

    elif datasetName == 'blendedMVS':  # zhiwei

        for _i, _view in enumerate(viewList):
            _cameraPO = readCameraP0_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                            poseNamePattern.replace('#',
                                                                                                    '{:03}'.format(
                                                                                                        _view))),
                                                 datasetName=datasetName)
            # _cameraPO = readCameraP0_as_np_tanks(cameraPO_file = datasetFolder+poseNamePattern.replace('#', '{:03}'.format(_view)).replace('@', '{}'.format(_view)))

            cameraPOs[_i] = _cameraPO
            _cameraRT, _cameraK = readCameraRTK_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                                       poseNamePattern.replace('#',
                                                                                                               '{:03}'.format(
                                                                                                                   _view))),
                                                            datasetName=datasetName)
            cameraRTOs[_i] = _cameraRT
            cameraKOs[_i] = _cameraK

    elif datasetName == 'giga_ours':  # zhiwei

        for _i, _view in enumerate(viewList):
            _cameraPO = readCameraP0_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                            poseNamePattern.replace('#',
                                                                                                    '{:03}'.format(
                                                                                                        _view))),
                                                 datasetName=datasetName)
            # _cameraPO = readCameraP0_as_np_tanks(cameraPO_file = datasetFolder+poseNamePattern.replace('#', '{:03}'.format(_view)).replace('@', '{}'.format(_view)))

            cameraPOs[_i] = _cameraPO
            _cameraRT, _cameraK = readCameraRTK_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                                       poseNamePattern.replace('#',
                                                                                                               '{:03}'.format(
                                                                                                                   _view))),
                                                            datasetName=datasetName)
            cameraRTOs[_i] = _cameraRT
            cameraKOs[_i] = _cameraK

    else:  # cameraPOs are stored in different files
        # tran_mat_path = os.path.join(datasetFolder,transMatPattern)
        for _i, _view in enumerate(viewList):
            _cameraPO = readCameraP0_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                            poseNamePattern.replace('#',
                                                                                                    '{:03}'.format(
                                                                                                        _view - 1)).replace(
                                                                                '@', '{}'.format(_view - 1))),
                                                 datasetName=datasetName)
            cameraPOs[_i] = _cameraPO
            _cameraRT, _cameraK = readCameraRTK_as_np_tanks(cameraPO_file=os.path.join(datasetFolder,
                                                                                       poseNamePattern.replace('#',
                                                                                                               '{:03}'.format(
                                                                                                                   _view - 1)).replace(
                                                                                           '@',
                                                                                           '{}'.format(_view - 1))),
                                                            datasetName=datasetName)
            # _cameraPO = readCameraP0_as_np_tanks(cameraPO_file=poseNamePattern.replace('#', '{:03}'.format(_view - 1)),
            #                                      datasetName=datasetName)
            # cameraPOs[_i] = _cameraPO
            # _cameraRT, _cameraK = readCameraRTK_as_np_tanks(cameraPO_file=poseNamePattern.replace('#', '{:03}'.format(_view - 1)),
            #                                      datasetName=datasetName)
            cameraRTOs[_i] = _cameraRT
            cameraKOs[_i] = _cameraK
    # print('cameraPOs', cameraPOs)
    return cameraPOs, cameraRTOs, cameraKOs


def readCameraP0s_np_allModel(datasetFolder,
                              datasetName,
                              poseNamePatternModels,
                              modelList,
                              viewList,
                              transMatPattern=None
                              ):
    cameraPs = []
    cameraP4s = []
    cameraRTs = []
    cameraKs = []

    for i in modelList:
        if datasetName == 'tanks':
            ##########TODO###################

            cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(datasetFolder,
                                                                   datasetName,
                                                                   poseNamePattern,
                                                                   viewList, )
            ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
            cameraPOs, cameraRTOs, cameraKOs = np.concatenate((cameraPOs, ones), axis=1)

        elif datasetName == 'DTU':
            cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(datasetFolder,
                                                                   datasetName,
                                                                   poseNamePatternModels,
                                                                   viewList,
                                                                   )
            ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
            cameraP0s = np.concatenate((cameraPOs, ones), axis=1)
        elif datasetName == 'tanks_COLMAP':  # zhiwei
            cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(datasetFolder,
                                                                   datasetName,
                                                                   poseNamePatternModels.replace('$', str(i)),
                                                                   viewList,
                                                                   )
            ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
            cameraP0s = np.concatenate((cameraPOs, ones), axis=1)
        elif datasetName == 'blendedMVS':  # zhiwei
            cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(datasetFolder,
                                                                   datasetName,
                                                                   poseNamePatternModels.replace('$', str(i)),
                                                                   viewList,
                                                                   )
            ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
            cameraP0s = np.concatenate((cameraPOs, ones), axis=1)
        elif datasetName == 'giga_ours':  # zhiwei
            cameraPOs, cameraRTOs, cameraKOs = readCameraPOs_as_np(datasetFolder,
                                                                   datasetName,
                                                                   poseNamePatternModels.replace('$', str(i)),
                                                                   viewList,
                                                                   )
            ones = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=cameraPOs.shape[0], axis=0)
            cameraP0s = np.concatenate((cameraPOs, ones), axis=1)

        cameraPs.append(cameraPOs)
        cameraP4s.append(cameraP0s)
        cameraRTs.append(cameraRTOs)
        cameraKs.append(cameraKOs)

    return (cameraPs, np.array(cameraP4s), np.array(cameraRTs), np.array(cameraKs))

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


def cameraPs2Ts_all(cameraPOs_all):
    """

    """
    model_num = len(cameraPOs_all)
    # pdb.set_trace()
    cameraT_all = np.zeros((model_num, cameraPOs_all[0].shape[0], 3))
    for i in range(model_num):
        cameraT_all[i] = cameraPs2Ts(cameraPOs_all[i])

    return cameraT_all


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


def inverse_camera_matrix(cameraP0s):
    N_Ms = cameraP0s.shape[0]

    projection_new = np.zeros((N_Ms, 4, 4))
    projection_new[:, 0:3, :] = cameraP0s
    projection_new[:, 3, :] = np.array(([[0, 0, 0, 1]]))
    projection_new = np.linalg.inv(projection_new)
    return projection_new


def calculate_angle_p1_p2_p3(p1, p2, p3, return_angle=True, return_cosine=True):
    """
    calculate angle <p1,p2,p3>, which is the angle between the vectors p2p1 and p2p3

    Parameters
    ----------
    p1/p2/p3: numpy with shape (3,)
    return_angle: return the radian angle
    return_cosine: return the cosine value

    Returns
    -------
    angle, cosine

    Examples
    --------
    """
    unit_vector = lambda v: v / np.linalg.norm(v)
    angle = lambda v1, v2: np.arccos(np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0))
    cos_angle = lambda v1, v2: np.clip(np.dot(unit_vector(v1), unit_vector(v2)), -1.0, 1.0)

    vect_p2p1 = p1 - p2
    vect_p2p3 = p3 - p2
    return angle(vect_p2p1, vect_p2p3) if return_angle else None, \
           cos_angle(vect_p2p1, vect_p2p3) if return_cosine else None


def k_combination_np(iterable, k=2):
    """
    list all the k-combination along the output rows:
    input: [2,5,8], list 2-combination to a numpy array
    output: np.array([[2,5],[2,8],[5,8]])

    ----------
    usages:
    >>> k_combination_np([2,5,8])
    array([[2, 5],
           [2, 8],
           [5, 8]])
    >>> k_combination_np([2,5,8]).dtype
    dtype('int64')
    >>> k_combination_np([2.2,5.5,8.8,9.9], k=3)
    array([[ 2.2,  5.5,  8.8],
           [ 2.2,  5.5,  9.9],
           [ 2.2,  8.8,  9.9],
           [ 5.5,  8.8,  9.9]])
    """
    combinations = []
    for _combination in itertools.combinations(iterable, k):
        combinations.append(_combination)
    return np.asarray(combinations)


def viewPairAngles_wrt_pts(cameraTs, pts_xyz):
    """
    given a set of camera positions and a set of points coordinates, output the angle between camera pairs w.r.t. each 3D point.

    -----------
    inputs:
        cameraTs: (N_views, 3) camera positions
        pts_xyz: (N_pts, 3) 3D points' coordinates

    -----------
    outputs:
        viewPairAngle_wrt_pts: (N_pts, N_viewPairs) angle

    -----------
    usages:
    >>> pts_xyz = np.array([[0,0,0],[1,1,1]], dtype=np.float32)     # p1 / p2
    >>> cameraTs = np.array([[0,0,1], [0,1,1], [1,0,1]], dtype=np.float32)      # c1/2/3
    >>> viewPairAngles_wrt_pts(cameraTs, pts_xyz) * 180 / math.pi    # output[i]: [<c1,pi,c2>, <c1,pi,c3>, <c2,pi,c3>]
    array([[ 45.,  45.,  60.],
           [ 45.,  45.,  90.]], dtype=float32)
    """

    unitize_array = lambda array, axis: array / np.linalg.norm(array, axis=axis, ord=2, keepdims=True)
    calc_arccos = lambda cos_values: np.arccos(np.clip(cos_values, -1.0, 1.0))  # TODO does it need clip ?
    N_views = cameraTs.shape[0]
    vector_pts2cameras = pts_xyz[:, None, :] - cameraTs[
        None, ...]  # (N_pts, 1, 3) - (1, N_views, 3) ==> (N_pts, N_views, 3)
    unit_vector_pts2cameras = unitize_array(vector_pts2cameras,
                                            axis=-1)  # (N_pts, N_views, 3)  unit vector along axis=-1

    # do the matrix multiplication for the (N_pats,) tack of (N_views, 3) matrixs
    ## (N_pts, N_views, 3) * (N_pts, 3, N_views) ==> (N_pts, N_views, N_views)
    # viewPairCosine_wrt_pts = np.matmul(unit_vector_pts2cameras, unit_vector_pts2cameras.transpose((0,2,1)))
    viewPairs = self.k_combination_np(range(N_views), k=2)  # (N_combinations, 2)
    viewPairCosine_wrt_pts = np.sum(
        np.multiply(unit_vector_pts2cameras[:, viewPairs[:, 0]], unit_vector_pts2cameras[:, viewPairs[:, 1]]),
        axis=-1)  # (N_pts, N_combinations, 3) elementwise multiplication --> (N_pts, N_combinations) sum over the last axis
    viewPairAngle_wrt_pts = calc_arccos(viewPairCosine_wrt_pts)  # (N_pts, N_combinations)
    return viewPairAngle_wrt_pts


# def viewPairAngles_p0s_pts(self, projection_M, )

def viewPairAngles_wrt_groupView(cameraTs, group_cameraTs, xyz_3D):
    '''

    :param cameraTs:
        shape: (N_views,3)
    :param group_cameraTs:
        shape:(N_bool_views,3)
    :param xyz_3D:
        shape:(3)
    :return:
        angle_total: the angle of group T and camera T
        shape: (N_bool_views, N_views)

    '''

    cameraTs_array = (cameraTs - xyz_3D)[None, :, :, None]  # (N_views,3)->(1,N_views, 3,1)
    group_cameraTs_array = (group_cameraTs - xyz_3D)[:, None, None, :]  # (N_bool_views,3)->(N_bool_views,1,3,1)
    dot_two = np.matmul(group_cameraTs_array, cameraTs_array)[:, :, 0, 0]  # (N_bool_views, N_views)

    len_cameraTs = np.linalg.norm(cameraTs - xyz_3D, axis=1)[None, :]  # (1, N_views)
    len_group_cameraTs = np.linalg.norm(group_cameraTs - xyz_3D, axis=1)[:, None]  # (N_bool_views, 1)

    len_total = len_cameraTs * len_group_cameraTs  # (N_bool_views, N_views)

    cos_total = dot_two / (len_total + 1e-10)  # (N_bool_views, N_views)
    angle_total = np.arccos(np.clip(cos_total, -1.0, 1.0))
    return (angle_total)


def select_group_pairs(projection_M, cameraTs, group_cameraTs, xyz_3D, cube_length, image_shape, angle_thres,
                       group_pair_num_max, group_pair_num_min, group_pair_index):
    '''
    given group view number, select groupviews
    :param projection_M: the
        shape:(N_views, 3,4)
    :param cameraTs:
        shape:(N_views, 3)
    :param group_cameraTs:
        shape:(N_boole_views, 3)
    :param xyz_3D:
        shape:(3)
    :param cube_length:
        float: the length of the cube
    :param image_shape:
        (img_h, img_w)
    :param angle_thres:
        float ses params.in_group_angle
    :param group_pair_num_max/min:
        int see params.group_pair_num_max/min
    :param group_pair_index
        list of int pair: see params.group_pair_index
    :return:
        view_pair_list: list of view_pair index
         element in list:
            (group_left, group_right, (group_id_left, group_id_right))
                group_left/right:
                    numpy 1d array of view pair number
        e.g. [(array([ 6, 16,  4,  2,  6]), array([33, 24, 16, 14, 24]), (0, 2)), (array([ 3, 15, 20,  4, 33]), array([ 7, 36,  5, 19,  4]), (1, 3)), (array([33, 24, 16, 14, 24]), array([24, 15, 22, 34, 15]), (2, 4)), (array([ 7, 36,  5, 19,  4]), array([24, 43, 34, 42, 14]), (3, 5)), (array([24, 15, 22, 34, 15]), array([42, 34, 38, 18, 37]), (4, 6)), (array([24, 43, 34, 42, 14]), array([43, 42, 33, 15, 35]), (5, 7))]
    '''
    view_in_flag = judge_cubic_center_in_view(projection_M,
                                              xyz_3D,
                                              cube_length,
                                              image_shape,
                                              )
    angle_total = viewPairAngles_wrt_groupView(cameraTs, group_cameraTs, xyz_3D)

    group_pair_flag = view_in_flag[None, :] * (angle_total < angle_thres)
    # print('group_pair_flag', group_pair_flag.shape)
    view_list = np.repeat((np.arange(group_pair_flag.shape[1]))[None, :], axis=0, repeats=group_pair_flag.shape[0])
    # print(group_pair_flag)

    view_num_list = []
    for i in range(group_pair_flag.shape[0]):
        view_num_i = view_list[i, group_pair_flag[i, :]]
        if (view_num_i.shape[0] >= group_pair_num_max):
            view_num_i = np.random.choice(view_num_i, group_pair_num_max, replace=False)
        view_num_list.append(view_num_i)

    view_pair_list = []
    for (i, j) in (group_pair_index):
        if ((view_num_list[i].shape[0] >= group_pair_num_min) and (view_num_list[j].shape[0] >= group_pair_num_min)):
            view_pair_list.append((view_num_list[i], view_num_list[j], (i, j)))
    # print('view_pair_list',view_pair_list)
    return view_pair_list


def select_group(projection_M, cameraTs, group_cameraTs, xyz_3D, cube_length, image_shape, angle_thres,
                 group_pair_num_max, group_pair_num_min):
    '''
    given group view number, select groupviews
    :param projection_M: the
        shape:(N_views, 3,4)
    :param cameraTs:
        shape:(N_views, 3)
    :param group_cameraTs:
        shape:(N_boole_views, 3)
    :param xyz_3D:
        shape:(3)
    :param cube_length:
        float: the length of the cube
    :param image_shape:
        (img_h, img_w)
    :param angle_thres:
        float ses params.in_group_angle
    :param group_pair_num_max/min:
        int see params.group_pair_num_max/min

    :return:
        view_list: list of view index
         element in list:
                group:
                    numpy 1d array of view number
    '''
    # view_in_flag = judge_cubic_center_in_view(projection_M ,
    #                           xyz_3D ,
    #                           cube_length,
    #                          image_shape,
    #                          )
    view_in_flag = np.ones((projection_M.shape[0]), dtype=np.bool)
    angle_total = viewPairAngles_wrt_groupView(cameraTs, group_cameraTs, xyz_3D)

    group_pair_flag = view_in_flag[None, :] * (angle_total < angle_thres)
    # print('group_pair_flag', group_pair_flag.shape)
    view_list = np.repeat((np.arange(group_pair_flag.shape[1]))[None, :], axis=0, repeats=group_pair_flag.shape[0])
    # print(group_pair_flag)
    view_num_list = []
    for i in range(group_pair_flag.shape[0]):
        view_num_i = view_list[i, group_pair_flag[i, :]]
        if (view_num_i.shape[0] >= group_pair_num_max):
            view_num_i = np.sort(np.random.choice(view_num_i, group_pair_num_max, replace=False), axis=0)
            # view_num_i = (np.random.choice(view_num_i, group_pair_num_max, replace = False))

            # pdb.set_trace()
        if (view_num_i.shape[0] >= group_pair_num_min):
            view_num_list.append(view_num_i)

    return view_num_list


def perspectiveProj(
        projection_M,
        xyz_3D,
        return_int_hw=True,
        return_depth=False):
    """
    perform perspective projection from 3D points to 2D points given projection matrix(es)
            support multiple projection_matrixes and multiple 3D vectors
    notice: [matlabx,matlaby] = [width, height]

    ----------
    inputs:
    projection_M: numpy with shape (3,4) / (N_Ms, 3,4), during calculation (3,4) will --> (1,3,4)
    xyz_3D: numpy with shape (3,) / (N_pts, 3), during calculation (3,) will --> (1,3)
    return_int_hw: bool, round results to integer when True.

    ----------
    outputs:
    img_h, img_w: (N_pts,) / (N_Ms, N_pts)

    ----------
    usages:

    inputs: (N_Ms, 3,4) & (N_pts, 3), return_int_hw = False/True

    >>> np.random.seed(201611)
    >>> Ms = np.random.rand(2,3,4)
    >>> pts_3D = np.random.rand(2,3)
    >>> pts_2Dh, pts_2Dw = perspectiveProj(Ms, pts_3D, return_int_hw = False)
    >>> np.allclose(pts_2Dw, np.array([[ 1.35860185,  0.9878389 ],
    ...        [ 0.64522543,  0.76079278 ]]))
    True
    >>> pts_2Dh_int, pts_2Dw_int = perspectiveProj(Ms, pts_3D, return_int_hw = True)
    >>> np.allclose(pts_2Dw_int, np.array([[1, 1], [1, 1]]))
    True

    inputs: (3,4) & (3,)

    >>> np.allclose(
    ...         np.r_[perspectiveProj(Ms[1], pts_3D[0], return_int_hw = False)],
    ...         np.stack((pts_2Dh, pts_2Dw))[:,1,0])
    True
    """

    if projection_M.shape[-2:] != (3, 4):
        raise ValueError(
            "perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

    if xyz_3D.ndim == 1:
        xyz_3D = xyz_3D[None, :]

    if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
        raise ValueError(
            "perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))
    # perspective projection
    N_pts = xyz_3D.shape[0]
    xyz1 = np.c_[xyz_3D, np.ones((N_pts, 1))].astype(np.float64)  # (N_pts, 3) ==> (N_pts, 4)
    pts_3D = np.matmul(projection_M, xyz1.T)  # (3, 4)/(N_Ms, 3, 4) * (4, N_pts) ==> (3, N_pts)/(N_Ms,3,N_pts)
    # the result is vector: [w,h,1], w is the first dim!!! (matlab's x/y/1')
    pts_2D = pts_3D[..., :2, :]
    # self.pts_3D = pts_3D
    pts_2D /= pts_3D[..., 2:3, :]  # (2, N_pts) /= (1, N_pts) | (N_Ms, 2, N_pts) /= (N_Ms, 1, N_pts)
    # self.pts_2D = pts_2D
    # print(self.pts_2D)
    if return_int_hw:
        pts_2D = pts_2D.round().astype(np.int64)  # (2, N_pts) / (N_Ms, 2, N_pts)
    img_w, img_h = pts_2D[..., 0, :], pts_2D[..., 1, :]  # (N_pts,) / (N_Ms, N_pts)
    if return_depth:
        depth = pts_3D[..., 2, :]
        return img_h, img_w, depth
    return img_h, img_w


def perspectiveProj_cubesCorner(projection_M, cube_xyz_min, cube_D_mm, return_int_hw=True,
                                return_depth=False):
    """
    perform perspective projection from 3D points to 2D points given projection matrix(es)
            support multiple projection_matrixes and multiple 3D vectors
    notice: [matlabx,matlaby] = [width, height]

    ----------
    inputs:
    projection_M: numpy with shape (3,4) / (N_Ms, 3,4), during calculation (3,4) will --> (1,3,4)
    cube_xyz_min: numpy with shape (3,) / (N_pts, 3), during calculation (3,) will --> (1,3)
    cube_D_mm: cube with shape D^3
    return_int_hw: bool, round results to integer when True.
    return_depth: bool

    ----------
    outputs:
    img_h, img_w: (N_Ms, N_pts, 8)

    ----------
    usages:

    inputs: (N_Ms, 3, 4) & (N_pts, 3), return_int_hw = False/True, outputs (N_Ms, N_pts, 8)

    >>> np.random.seed(201611)
    >>> Ms = np.random.rand(2,3,4)
    >>> pts_3D = np.random.rand(2,3)
    >>> pts_2Dh, pts_2Dw = perspectiveProj_cubesCorner(Ms, pts_3D, cube_D_mm = 1, return_int_hw = False)
    >>> np.allclose(pts_2Dw[:,:,0], np.array([[ 1.35860185,  0.9878389 ],
    ...        [ 0.64522543,  0.76079278 ]]))
    True
    >>> pts_2Dh_int, pts_2Dw_int = perspectiveProj_cubesCorner(Ms, pts_3D, cube_D_mm = 1, return_int_hw = True)
    >>> np.allclose(pts_2Dw_int[:,:,0], np.array([[1, 1], [1, 1]]))
    True

    inputs: (3,4) & (3,), outputs (1,1,8)

    >>> np.allclose(
    ...         perspectiveProj_cubesCorner(Ms[1], pts_3D[0], cube_D_mm = 1, return_int_hw = False)[0],
    ...         pts_2Dh[1,0])        # (1,1,8)
    True
    """

    if projection_M.shape[-2:] != (3, 4):
        raise ValueError(
            "perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

    if cube_xyz_min.ndim == 1:
        cube_xyz_min = cube_xyz_min[None, :]  # (3,) --> (N_pts, 3)

    if cube_xyz_min.shape[1] != 3 or cube_xyz_min.ndim != 2:
        raise ValueError("perspectiveProj needs cube_xyz_min with shape (3,) or (N_pts, 3), however got {}".format(
            cube_xyz_min.shape))

    N_pts = cube_xyz_min.shape[0]
    cubeCorner_shift = np.indices((2, 2, 2)).reshape((3, -1)).T[None, :, :] * cube_D_mm  # (3,2,2,2) --> (1,8,3)
    cubeCorner = cube_xyz_min[:, None, :] + cubeCorner_shift  # (N_pts, 1, 3) + (1,8,3) --> (N_pts, 8, 3)
    img_h, img_w = perspectiveProj(projection_M=projection_M, xyz_3D=cubeCorner.reshape((N_pts * 8, 3)),
                                   return_int_hw=return_int_hw,
                                   return_depth=return_depth)  # img_w/h: (N_Ms, N_pts*8)
    img_w = img_w.reshape((-1, N_pts, 8))
    img_h = img_h.reshape((-1, N_pts, 8))
    return img_h, img_w


def image_compress_coef(projection_M,
                        cube_xyz_min,
                        cube_D_mm,
                        _cube_D_,
                        image_compress_multiple,
                        compress_ratio=1.0
                        ):
    img_h, img_w = perspectiveProj_cubesCorner(projection_M,
                                               cube_xyz_min,
                                               cube_D_mm,
                                               return_int_hw=True,
                                               return_depth=False)
    img_h_max = np.max(img_h, axis=2)  # (N_Ms, N_pts)
    img_w_max = np.max(img_w, axis=2)
    img_h_min = np.min(img_h, axis=2)
    img_w_min = np.min(img_w, axis=2)
    img_h_resol = (img_h_max - img_h_min + 0.0) / _cube_D_
    img_w_resol = (img_w_max - img_w_min + 0.0) / _cube_D_

    compress_h = compress_ratio * img_h_resol.mean() / image_compress_multiple
    compress_w = compress_ratio * img_w_resol.mean() / image_compress_multiple

    return ((compress_h), (compress_w))


# def resize_matrix(projection_M, compress_h_new, compress_w_new):
#     transform_matrix = np.array([[[1 / compress_w_new, 0, 0], [0, 1 / compress_h_new, 0], [0, 0, 1]]])
#     projection_M_new = np.matmul(transform_matrix, projection_M)
#
#     cameraTs = cameraPs2Ts(projection_M)
#     cameraTs_new = cameraPs2Ts(projection_M_new)
#     trans_vector = (cameraTs - cameraTs_new)[:, :, None]
#     identical_matrix = np.repeat(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), cameraTs.shape[0], axis=0)
#     bottom_matrix = np.repeat(np.array([[[0, 0, 0, 1]]]), cameraTs.shape[0], axis=0)
#     transform_matrix2 = np.concatenate((identical_matrix, trans_vector), axis=2)
#     transform_matrix2 = np.concatenate((transform_matrix2, bottom_matrix), axis=1)
#     projection_M_new_f = np.concatenate((projection_M_new, bottom_matrix), axis=1)
#
#     projection_M_new = np.matmul(transform_matrix2, projection_M_new_f)
#     projection_M_new = projection_M_new[:, :3, :]
#     return projection_M_new


def resize_image_and_matrix(images,
                            projection_M,
                            cube_xyz_min,
                            cube_D_mm,
                            _cube_D_,
                            image_compress_multiple,
                            return_list=False,
                            compress_ratio=1.0):
    '''
    compress image and garantee the camera position is not changing
    :param images:  all images of one model
        type:list or None
        if list
            list element: image array
                shape: (img_h,img_w, 3)

    :param projection_M:  camera matrix
        shape: (N_views, 3, 4)
    :param cube_xyz_min:  min xyz coordinate
        shape: (3,) / (N_pts, 3)  usually it is (3,) because we only sample one cubic to judge the resize term
    :param cube_D_mm:
        cubic length float
    :param _cube_D_:
        cubic size int
    :param image_compress_multiple:
        same as param.image_compress_multiple
    :param return_list: bool
        if False return the numpy array
    :param compress_ratio
        see self.params.compress_ratio
    :return:
        if image is not None
            images_resized:resized image
                shape:(N_view, img_h_new, img_w_new)resize_image_and_matrix
            projection_M_new: new cameraP
                shape:(N_view,3,4)
            (compress_h_new,compress_w_new):(float,float)
        elif image is None: only change the matrix
            projection_M_new: new cameraP
                shape:(N_view,3,4)
            (compress_h_new,compress_w_new):(float,float)
    '''
    (compress_h, compress_w) = image_compress_coef(projection_M,
                                                   cube_xyz_min,
                                                   cube_D_mm,
                                                   _cube_D_,
                                                   image_compress_multiple,
                                                   compress_ratio=compress_ratio)
    resized_h = int(image_compress_multiple * (images[0].shape[0] // (compress_h * image_compress_multiple)))
    resized_w = int(image_compress_multiple * (images[0].shape[1] // (compress_w * image_compress_multiple)))

    compress_h_new = images[0].shape[0] / (resized_h + 0.0)
    compress_w_new = images[0].shape[1] / (resized_w + 0.0)
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
    projection_M_new = projection_M_new[:, :3, :]
    image_resized_list = []
    if (images is not None):
        for image in images:
            image_resized = scipy.misc.imresize(image, size=(resized_h, resized_w), interp='bicubic')
            image_resized = image_resized / 256.0 - 0.5
            image_resized_list.append(image_resized)
        images_resized = image_resized_list if return_list else np.stack(image_resized_list)
        return (images_resized, projection_M_new, (compress_h_new, compress_w_new))
    else:
        return (None, projection_M_new, (compress_h_new, compress_w_new))


# def resize_multistage_image_and_matrix(images,
#                                        projection_M,
#                                        cube_xyz_min,
#                                        cube_D_mm,
#                                        _cube_D_,
#                                        image_compress_multiple,
#                                        image_compress_stage,
#                                        return_list=False,
#                                        compress_ratio=1.0):
#     '''
#     compress image and garantee the camera position is not changing
#     :param images:  all images of one model
#         type:list or None
#         if list
#             list element: image array
#                 shape: (img_h,img_w, 3)
#
#     :param projection_M:  camera matrix
#         shape: (N_views, 3, 4)
#     :param cube_xyz_min:  min xyz coordinate
#         shape: (3,) / (N_pts, 3)  usually it is (3,) because we only sample one cubic to judge the resize term
#     :param cube_D_mm:
#         cubic length float
#     :param _cube_D_:
#         cubic size int
#     :param image_compress_multiple:
#         same as param.image_compress_multiple
#     :param image_compress_stage
#         same as param.image_compress_stage
#     :param return_list: bool
#         if False return the numpy array
#     :param compress_ratio
#         see self.params.compress_ratio
#     :return:
#         if image is not None
#             image_resized_stage_list:multistage of resized image
#                 length : = image_compress_stage
#                 ele in each list:
#                     shape:(N_view, img_h_new//2**iter, img_w_new//2**iter)
#             projection_M_new: new cameraP
#                 shape:(N_view,3,4)
#             (compress_h_new,compress_w_new):(float,float)
#         elif image is None: only change the matrix
#             projection_M_new: new cameraP
#                 shape:(N_view,3,4)
#             (compress_h_new,compress_w_new):(float,float)
#     '''
#     # (compress_h, compress_w) = image_compress_coef(projection_M,
#     #                                                cube_xyz_min,
#     #                                                cube_D_mm,
#     #                                                _cube_D_,
#     #                                                1,
#     #                                               compress_ratio = compress_ratio)
#
#     # print('compress_h', compress_h, compress_w)
#     compress_h = compress_ratio
#     compress_w = compress_ratio
#     resized_h = int(image_compress_multiple * (images[0].shape[0] // (compress_h * image_compress_multiple)))
#     resized_w = int(image_compress_multiple * (images[0].shape[1] // (compress_w * image_compress_multiple)))
#
#     # pdb.set_trace()
#     compress_h_new = images[0].shape[0] / (resized_h + 0.0)
#     compress_w_new = images[0].shape[1] / (resized_w + 0.0)
#     transform_matrix = np.array([[[1 / compress_w_new, 0, 0], [0, 1 / compress_h_new, 0], [0, 0, 1]]])
#     projection_M_new = np.matmul(transform_matrix, projection_M)
#
#     cameraTs = cameraPs2Ts(projection_M)
#     cameraTs_new = cameraPs2Ts(projection_M_new)
#     trans_vector = (cameraTs - cameraTs_new)[:, :, None]
#     identical_matrix = np.repeat(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), cameraTs.shape[0], axis=0)
#     bottom_matrix = np.repeat(np.array([[[0, 0, 0, 1]]]), cameraTs.shape[0], axis=0)
#     transform_matrix2 = np.concatenate((identical_matrix, trans_vector), axis=2)
#     transform_matrix2 = np.concatenate((transform_matrix2, bottom_matrix), axis=1)
#     projection_M_new_f = np.concatenate((projection_M_new, bottom_matrix), axis=1)
#
#     projection_M_new = np.matmul(transform_matrix2, projection_M_new_f)
#     projection_M_new = projection_M_new[:, :3, :]
#
#     if (images is not None):
#         image_resized_stage_list = []
#         for iter in range(image_compress_stage):
#             image_resized_list = []
#             for image in images:
#                 # print('resized image shape',resized_h, resized_w)
#                 image_resized = scipy.misc.imresize(image,
#                                                     size=(int(resized_h // (2 ** iter)), int(resized_w // (2 ** iter))),
#                                                     interp='bicubic')
#                 image_resized = image_resized / 256.0 - 0.5
#                 image_resized_list.append(image_resized)
#             images_resized = image_resized_list if return_list else np.stack(image_resized_list)
#             image_resized_stage_list.append(images_resized)
#         return (image_resized_stage_list, projection_M_new, (compress_h_new, compress_w_new))
#     else:
#         return (None, projection_M_new, (compress_h_new, compress_w_new))


def judge_cubic_center_in_view(projection_M,
                               xyz_3D,
                               cube_length,
                               image_shape,
                               ):
    '''
    'the bool flag of each view can see the center of cubic:'
    :param projection_M:
        shape:(N_views, 3, 4)
    :param xyz_3D:
        shape:(3)
    :param cube_length
        float
    :param image_shape:
        (img_h,img_w)
    :return:
        view_in_flag: bool array
            shape: (N_views)
    '''

    img_h_new, img_w_new = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D,
    )
    img_h_100, img_w_100 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + np.array((cube_length, 0, 0)),
    )
    img_h_010, img_w_010 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + np.array((0, cube_length, 0)),
    )
    img_h_001, img_w_001 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + np.array((0, 0, cube_length)),
    )
    img_h_011, img_w_011 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + np.array((0, cube_length, cube_length)),
    )
    img_h_101, img_w_101 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + np.array((cube_length, 0, cube_length)),
    )
    img_h_110, img_w_110 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + np.array((cube_length, cube_length, 0)),
    )
    img_h_111, img_w_111 = perspectiveProj(
        projection_M=projection_M,
        xyz_3D=xyz_3D + cube_length,
    )

    img_h_bool = (img_h_new < image_shape[0]) * (img_h_new > 0)
    img_w_bool = (img_w_new < image_shape[1]) * (img_w_new > 0)
    img_h_bool_001 = (img_h_001 < image_shape[0]) * (img_h_001 > 0)
    img_w_bool_001 = (img_w_001 < image_shape[1]) * (img_w_001 > 0)
    img_h_bool_010 = (img_h_010 < image_shape[0]) * (img_h_010 > 0)
    img_w_bool_010 = (img_w_010 < image_shape[1]) * (img_w_010 > 0)
    img_h_bool_100 = (img_h_100 < image_shape[0]) * (img_h_100 > 0)
    img_w_bool_100 = (img_w_100 < image_shape[1]) * (img_w_100 > 0)
    img_h_bool_011 = (img_h_011 < image_shape[0]) * (img_h_011 > 0)
    img_w_bool_011 = (img_w_011 < image_shape[1]) * (img_w_011 > 0)
    img_h_bool_110 = (img_h_110 < image_shape[0]) * (img_h_110 > 0)
    img_w_bool_110 = (img_w_110 < image_shape[1]) * (img_w_110 > 0)
    img_h_bool_101 = (img_h_101 < image_shape[0]) * (img_h_101 > 0)
    img_w_bool_101 = (img_w_101 < image_shape[1]) * (img_w_101 > 0)
    img_h_bool_111 = (img_h_111 < image_shape[0]) * (img_h_111 > 0)
    img_w_bool_111 = (img_w_111 < image_shape[1]) * (img_w_111 > 0)
    view_in_flag = img_h_bool * img_w_bool * img_h_bool_001 * img_w_bool_001 * img_h_bool_010 * img_w_bool_010 * img_h_bool_100 * img_w_bool_100 * img_h_bool_110 * img_w_bool_110 * img_h_bool_101 * img_w_bool_101 * img_h_bool_011 * img_w_bool_011 * img_h_bool_111 * img_w_bool_111
    print('the bool flag of each view can see the center of cubic:', view_in_flag.sum())
    return view_in_flag[:, 0]


def count_gx_gy(projection_M, h_length=1, w_length=1):
    projection_M_inverse = inverse_camera_matrix(projection_M)
    N_view = projection_M_inverse.shape[0]
    vector_101 = np.array(([w_length, 0, 1, 1]))[None, :, None]
    vector_011 = np.array(([0, h_length, 1, 1]))[None, :, None]
    vector_001 = np.array(([0, 0, 1, 1]))[None, :, None]
    global_101 = np.matmul(projection_M_inverse, vector_101)[:, :3, 0]  # shape: (N_view, 4,1)->(N_view, 3)
    global_011 = np.matmul(projection_M_inverse, vector_011)[:, :3, 0]
    global_001 = np.matmul(projection_M_inverse, vector_001)[:, :3, 0]

    gx = np.linalg.norm(global_101 - global_001, axis=1)  # shape: (N_views)
    gy = np.linalg.norm(global_011 - global_001, axis=1)

    return (gx, gy)


def generateMetaVector_old(
        projection_M,
        cube_xyz_min,
        cameraTs,
        cube_D_resol,
        _cube_D_,
):
    '''

    :param projection_M:
        shape:(N_views, 3, 4)
    :param cube_xyz_min:
        shape:(,3)
    :param cameraTs:
        shape:(N_views, 3)
    :param cube_D_resol: resolution of each voxel
        float
    :param _cube_D_: length of cube
        int
    :return:
        meta_vector: the array of each vector represent camera position
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 10)
        wrapping_vector: the map from each voxel to image
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 3)
    '''
    x = np.arange(0, _cube_D_, 1.0)
    y = np.arange(0, _cube_D_, 1.0)
    z = np.arange(0, _cube_D_, 1.0)
    if not (x.shape[0] == _cube_D_):
        print('shape of Meta vector went wrong')
        raise TypeError
    xx, yy, zz = np.meshgrid(x, y, z)
    XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).reshape(3, _cube_D_, _cube_D_, _cube_D_)
    XYZ = np.moveaxis(XYZ, 0, 3)
    if not (list(XYZ[0, 1, 3, :]) == [0.0, 1.0, 3.0]):
        print('index of Meta vector went wrong')
        raise TypeError
    cube_xyz = cube_xyz_min[None, None, None, :] + XYZ * cube_D_resol  # shape:(_cube_D_, _cube_D_, _cube_D_, 3)
    ones = np.ones((_cube_D_, _cube_D_, _cube_D_, 1))
    cube_xyz_matmul = np.concatenate((cube_xyz, ones), axis=3)[None, :, :, :, :,
                      None]  # shape:(1, _cube_D_, _cube_D_, _cube_D_, 4, 1)
    projection_M_matmul = projection_M[:, None, None, None, :, :]  # shape:(N_view, 1, 1, 1, 3, 4)

    project_cube_xyz = np.matmul(projection_M_matmul,
                                 cube_xyz_matmul)  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3, 1)

    (gx, gy) = count_gx_gy(projection_M)

    Z = project_cube_xyz[:, :, :, :, 2,
        0]  # the depth of each cubic points  shape:(N_view, _cube_D_, _cube_D_, _cube_D_)

    alpha_x = (Z * gx[:, None, None, None] / cube_D_resol)[:, :, :, :,
              None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    alpha_y = (Z * gy[:, None, None, None] / cube_D_resol)[:, :, :, :,
              None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    print('the average pixel a cubic can get on x axis', alpha_x.mean())
    print('the average pixel a cubic can get on y axis', alpha_y.mean())

    tau = project_cube_xyz[:, :, :, :, :, 0] / np.linalg.norm(project_cube_xyz[:, :, :, :, :, 0], axis=4)[:, :, :, :,
                                               None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    vector_xyz = cube_xyz[None, :, :, :, :] - cameraTs[:, None, None, None,
                                              :]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)
    theta = vector_xyz / np.linalg.norm(vector_xyz, axis=4)[:, :, :, :,
                         None]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    YX = project_cube_xyz[:, :, :, :, :2, 0] / project_cube_xyz[:, :, :, :, 2, 0][:, :, :, :, None]
    H = YX[:, :, :, :, 1][:, :, :, :, None]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    W = YX[:, :, :, :, 0][:, :, :, :, None]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    D = np.zeros(np.shape(H))
    X = H - np.floor(H)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    Y = W - np.floor(W)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    meta_vector = np.concatenate((alpha_x, alpha_y, tau, theta, X, Y), axis=4)
    wrapping_vector = np.concatenate((D, H, W), axis=4)

    return (meta_vector, wrapping_vector)


def generateMetaVector(
        projection_M,
        compress,
        cube_xyz_min,
        cameraTs,
        cube_D_resol,
        _cube_D_,
):
    '''

    :param projection_M:
        shape:(N_views, 3, 4)
    :param compress
        turple: (compress_h, compress_w)
    :param cube_xyz_min:
        shape:(,3)
    :param cameraTs:
        shape:(N_views, 3)
    :param cube_D_resol: resolution of each voxel
        float
    :param _cube_D_: length of cube
        int
    :return:
        meta_vector: the array of each vector represent camera position
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 10)
        wrapping_vector: the map from each voxel to image
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 3)
    '''

    compress_h, compress_w = compress

    x = np.arange(0, _cube_D_, 1.0)
    y = np.arange(0, _cube_D_, 1.0)
    z = np.arange(0, _cube_D_, 1.0)
    if not (x.shape[0] == _cube_D_):
        print('shape of Meta vector went wrong')
        raise TypeError
    xx, yy, zz = np.meshgrid(x, y, z)
    XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).reshape(3, _cube_D_, _cube_D_, _cube_D_)
    XYZ = np.moveaxis(XYZ, 0, 3)
    if not (list(XYZ[0, 1, 3, :]) == [0.0, 1.0, 3.0]):
        print('index of Meta vector went wrong')
        raise TypeError
    cube_xyz = cube_xyz_min[None, None, None, :] + XYZ * cube_D_resol  # shape:(_cube_D_, _cube_D_, _cube_D_, 3)
    # print('cube_xyz_min[None, None, None, :]', cube_xyz_min[None, None, None, :])
    # print('@(*#@!#!@(*$&!@(*')
    # print('cube_xyz[2,3,1,:]', cube_xyz[2,3,1,:])
    # print('cube_xyz[2,3,2,:]', cube_xyz[2, 3, 2, :])
    # print('cube_xyz[2,4,1,:]', cube_xyz[2, 4, 1, :])

    ones = np.ones((_cube_D_, _cube_D_, _cube_D_, 1))
    cube_xyz_matmul = np.concatenate((cube_xyz, ones), axis=3)[None, :, :, :, :,
                      None]  # shape:(1, _cube_D_, _cube_D_, _cube_D_, 4, 1)
    projection_M_matmul = projection_M[:, None, None, None, :, :]  # shape:(N_view, 1, 1, 1, 3, 4)

    project_cube_xyz = np.matmul(projection_M_matmul,
                                 cube_xyz_matmul)  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3, 1)
    # print('@(*#@!#!@(*$&!@(*')
    # print(project_cube_xyz.shape)
    # print('project_cube_xyz[2,3,1,:]', project_cube_xyz[44, 2, 3, 1, :])
    # print('project_cube_xyz[2,3,2,:]', project_cube_xyz[44, 2, 3, 2, :])
    # print('project_cube_xyz[2,4,1,:]', project_cube_xyz[44, 2, 4, 1, :])

    (gx, gy) = count_gx_gy(projection_M, h_length=compress_h, w_length=compress_w)

    Z = project_cube_xyz[:, :, :, :, 2,
        0]  # the depth of each cubic points  shape:(N_view, _cube_D_, _cube_D_, _cube_D_)

    alpha_x = (Z * gx[:, None, None, None] / cube_D_resol)[:, :, :, :,
              None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    alpha_y = (Z * gy[:, None, None, None] / cube_D_resol)[:, :, :, :,
              None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    print('the average pixel a cubic can get on x axis', alpha_x.mean())
    print('the average pixel a cubic can get on y axis', alpha_y.mean())

    tau = project_cube_xyz[:, :, :, :, :, 0] / np.linalg.norm(project_cube_xyz[:, :, :, :, :, 0], axis=4)[:, :, :, :,
                                               None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    vector_xyz = cube_xyz[None, :, :, :, :] - cameraTs[:, None, None, None,
                                              :]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)
    theta = vector_xyz / np.linalg.norm(vector_xyz, axis=4)[:, :, :, :,
                         None]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    YX = project_cube_xyz[:, :, :, :, :2, 0] / project_cube_xyz[:, :, :, :, 2, 0][:, :, :, :, None]

    H = YX[:, :, :, :, 1][:, :, :, :, None] / compress_h  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    W = YX[:, :, :, :, 0][:, :, :, :, None] / compress_w  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    D = np.zeros(np.shape(H))
    X = H - np.floor(H)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    Y = W - np.floor(W)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    meta_vector = np.concatenate((alpha_x, alpha_y, tau, theta, X, Y), axis=4)
    wrapping_vector = np.concatenate((W, H, D),
                                     axis=4)  # To avoid confusion in notation, let’s note that x corresponds to the width dimension IW, y corresponds to the height dimension IH and z corresponds to the depth dimension ID.

    return (meta_vector, wrapping_vector)


def generate_sparseMetaVector(
        projection_M,
        compress,
        cube_xyz_min,
        stage_num,
        cameraTs,
        cube_D_resol,
        _cube_D_,
        info_list=None
):
    '''

    :param projection_M:
        shape:(N_views, 3, 4)
    :param compress
        turple: (compress_h, compress_w)
    :param cube_xyz_min:
        shape:(,3)
    :param stage_num
        int
    :param cameraTs:
        shape:(N_views, 3)
    :param cube_D_resol: resolution of each voxel
        float
    :param _cube_D_: length of cube
        int
    :param info_list
    :return:
        meta_list: list of meta\wrapping vector
            len: stage_num
            ele:
                (meta_vector, wrapping_vector)
        output_list: list of output vector
            len: stage_num
            ele:
                (q_final, xyz_final, rgb_final, n_final)
    '''

    meta_list = []
    input_list = []
    output_list = []

    resol_new = cube_D_resol
    xyz_3D_new = copy.copy(cube_xyz_min)
    cube_D_new = _cube_D_

    for i in range(stage_num):
        cubes_gt_np = info_list[i]
        if (i == (stage_num - 1)):
            use_dense = True
        else:
            use_dense = False

        (xyz_global_final, xyz_final, rgb_final, n_final, q_final, sort_index) = generate_sparse(
            cube_xyz_min=xyz_3D_new,
            cube_D_resol=resol_new,
            _cube_D_=cube_D_new,
            cubes_gt_np=cubes_gt_np,
            use_dense=use_dense
        )
        (meta_vector, wrapping_vector) = generateMeta_from_xyz(projection_M=projection_M,
                                                               compress=compress,
                                                               cameraTs=cameraTs,
                                                               cube_D_resol=resol_new,
                                                               _cube_D_=cube_D_new,
                                                               pts_xyz=xyz_global_final
                                                               )
        meta_list.append((meta_vector, wrapping_vector))
        output_list.append((q_final, xyz_final, rgb_final, n_final, xyz_global_final, sort_index))

        xyz_3D_new += (resol_new / 2)
        resol_new *= 2
        cube_D_new /= 2
        compress = (compress[0] * 2, compress[1] * 2)

    return meta_list, output_list


def generate_sparse(
        cube_xyz_min,
        cube_D_resol,
        _cube_D_,
        cubes_gt_np=None,
        use_dense=False
):
    '''

    :param cube_xyz_min:
        shape:(,3)
    :param cube_D_resol: resolution of each voxel
        float
    :param _cube_D_: length of cube
        int
    :param cubes_gt_np
    :return:
        xyz_global_final : the location of input voxel
            shape: (N,3)
        xyz_final: the relative location of output voxel
            shape: (N,3)
        rgb_final: output voxel
            shape: (N,3)
        n_final: output voxel
            shape: (N,3)
        q_final: output voxel
            shape: bool (N,1)
        sort_index: the sort index of the ground truth used for point up convolution
            shape: int (N_points,)
    '''

    # cubes_gt_np_sort = np.sort(cubes_gt_np, order = 'ijk_id')

    x = np.arange(0, _cube_D_, 1.0)
    y = np.arange(0, _cube_D_, 1.0)
    z = np.arange(0, _cube_D_, 1.0)
    if not (x.shape[0] == _cube_D_):
        print('shape of Meta vector went wrong')
        raise TypeError
    xx, yy, zz = np.meshgrid(x, y, z)
    XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).T
    XYZ_id = 8 * ((_cube_D_ / 2) * (_cube_D_ / 2) * (XYZ[:, 0] // 2) + (_cube_D_ / 2) * (XYZ[:, 1] // 2) + XYZ[:,
                                                                                                           2] // 2) + (
                     4 * (XYZ[:, 0] % 2) + 2 * (XYZ[:, 1] % 2) + XYZ[:, 2] % 2)
    XYZ_id_s = (_cube_D_ * _cube_D_ * XYZ[:, 0] + _cube_D_ * XYZ[:, 1] + XYZ[:, 2])
    XYZ_np = np.empty((XYZ.shape[0],), dtype=[('ijk', np.uint32, (3,)), ('ijk_id', np.uint32), ('ijk_id_s', np.uint32)])
    XYZ_np['ijk'] = XYZ
    XYZ_np['ijk_id'] = XYZ_id
    XYZ_np['ijk_id_s'] = XYZ_id_s
    XYZ_sort_np = np.sort(XYZ_np, order='ijk_id')
    XYZ_sort = XYZ_sort_np['ijk']

    # xyz_global = np.zeros((XYZ.shape[0], 3))
    xyz = np.zeros((XYZ.shape[0], 3))
    rgb = np.zeros((XYZ.shape[0], 3))
    n = np.zeros((XYZ.shape[0], 3))
    q = np.zeros((XYZ.shape[0], 1), dtype=np.bool)

    # xyz_global[cubes_gt_np['ijk_id'], :] = cubes_gt_np['xyz_global']
    xyz_global = XYZ_sort * cube_D_resol + cube_xyz_min

    xyz[cubes_gt_np['ijk_id'], :] = cubes_gt_np['xyz']
    rgb[cubes_gt_np['ijk_id'], :] = cubes_gt_np['rgb']
    n[cubes_gt_np['ijk_id'], :] = cubes_gt_np['normals']
    q[cubes_gt_np['ijk_id'], :] = True

    XYZ_big_num = int(XYZ.shape[0] // 8)
    xyz_global_new = xyz_global.reshape((XYZ_big_num, 8, 3))
    xyz_new = xyz.reshape((XYZ_big_num, 8, 3))
    rgb_new = rgb.reshape((XYZ_big_num, 8, 3))
    n_new = n.reshape((XYZ_big_num, 8, 3))
    q_new = q.reshape((XYZ_big_num, 8, 1))
    ijk_id_s_new = XYZ_sort_np['ijk_id_s'].reshape((XYZ_big_num, 8, 1))

    if (use_dense):
        xyz_global_final = xyz_global_new.reshape((-1, 3))
        xyz_final = xyz_new.reshape((-1, 3))
        rgb_final = rgb_new.reshape((-1, 3))
        n_final = n_new.reshape((-1, 3))
        q_final = q_new.reshape((-1, 1))
        ijk_id_s_final = ijk_id_s_new.reshape((-1))
    else:
        cubes_gt_id_big = np.unique(cubes_gt_np['ijk_id'] // 8)

        xyz_global_final = xyz_global_new[cubes_gt_id_big, :, :].reshape((-1, 3))
        xyz_final = xyz_new[cubes_gt_id_big, :, :].reshape((-1, 3))
        rgb_final = rgb_new[cubes_gt_id_big, :, :].reshape((-1, 3))
        n_final = n_new[cubes_gt_id_big, :, :].reshape((-1, 3))
        q_final = q_new[cubes_gt_id_big, :, :].reshape((-1, 1))
        ijk_id_s_final = ijk_id_s_new[cubes_gt_id_big, :, :].reshape((-1))

    sort_index = np.argsort(ijk_id_s_final[q_final[:, 0]])
    return (xyz_global_final, xyz_final, rgb_final, n_final, q_final, sort_index)


def generateMeta_from_xyz(projection_M,
                          compress,
                          cameraTs,
                          cube_D_resol,
                          _cube_D_,
                          pts_xyz
                          ):
    '''

        :param projection_M:
            shape:(N_views, 3, 4)
        :param compress
            turple: (compress_h, compress_w)
        :param cameraTs:
            shape:(N_views, 3)
        :param cube_D_resol: resolution of each voxel
            float
        :param _cube_D_: length of cube
            int
        :param pts_xyz: points of voxel
            shape: (N_points, 3)
        :return:
            meta_vector: the array of each vector represent camera position
                shape: (N_views, N_points, 10)
            wrapping_vector: the map from each voxel to image
                shape: (N_views, N_points, 3)
    '''
    compress_h, compress_w = compress

    N_points = pts_xyz.shape[0]
    ones = np.ones((N_points, 1))
    cube_xyz_matmul = np.concatenate((pts_xyz, ones), axis=1)[None, :, :, None]  # shape:(1, N_points, 4, 1)
    projection_M_matmul = projection_M[:, None, :, :]  # shape:(N_view, 1, 3, 4)
    project_cube_xyz = np.matmul(projection_M_matmul,
                                 cube_xyz_matmul)  # shape:(N_view, N_points, 3, 1)

    (gx, gy) = count_gx_gy(projection_M, h_length=compress_h, w_length=compress_w)

    Z = project_cube_xyz[:, :, 2,
        0]  # the depth of each cubic points  shape:(N_view, N_points,)

    alpha_x = (Z * gx[:, None] / cube_D_resol)[:, :, None]  # shape:(N_view, N_points, 1)
    alpha_y = (Z * gy[:, None] / cube_D_resol)[:, :, None]  # shape:(N_view, N_points, 1)

    print('the average pixel a cubic can get on x axis', alpha_x.mean())
    print('the average pixel a cubic can get on y axis', alpha_y.mean())

    tau = project_cube_xyz[:, :, :, 0] / np.linalg.norm(project_cube_xyz[:, :, :, 0], axis=2)[:, :,
                                         None]  # shape:(N_view, N_points, 3)

    vector_xyz = pts_xyz[None, :, :] - cameraTs[:, None, :]  # shape: (N_view, N_points, 3)
    theta = vector_xyz / np.linalg.norm(vector_xyz, axis=2)[:, :, None]  # shape: (N_view, N_points, 3)

    YX = project_cube_xyz[:, :, :2, 0] / project_cube_xyz[:, :, 2, 0][:, :, None]  # shape: (N_view, N_points, 2)
    H = YX[:, :, 1][:, :, None] / compress_h  # shape: (N_view, N_points, 1)
    W = YX[:, :, 0][:, :, None] / compress_w  # shape: (N_view, N_points, 1)
    D = np.zeros(np.shape(H))
    X = H - np.floor(H)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    Y = W - np.floor(W)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    meta_vector = np.concatenate((alpha_x, alpha_y, tau, theta, X, Y), axis=2)
    wrapping_vector = np.concatenate((W, H, D),
                                     axis=2)  # To avoid confusion in notation, let’s note that x corresponds to the width dimension IW, y corresponds to the height dimension IH and z corresponds to the depth dimension ID.
    return (meta_vector, wrapping_vector)


def generateMetaVector(
        projection_M,
        compress,
        cube_xyz_min,
        cameraTs,
        cube_D_resol,
        _cube_D_,
):
    '''

    :param projection_M:
        shape:(N_views, 3, 4)
    :param compress
        turple: (compress_h, compress_w)
    :param cube_xyz_min:
        shape:(,3)
    :param cameraTs:
        shape:(N_views, 3)
    :param cube_D_resol: resolution of each voxel
        float
    :param _cube_D_: length of cube
        int
    :return:
        meta_vector: the array of each vector represent camera position
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 10)
        wrapping_vector: the map from each voxel to image
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 3)
    '''

    compress_h, compress_w = compress

    x = np.arange(0, _cube_D_, 1.0)
    y = np.arange(0, _cube_D_, 1.0)
    z = np.arange(0, _cube_D_, 1.0)
    if not (x.shape[0] == _cube_D_):
        print('shape of Meta vector went wrong')
        raise TypeError
    xx, yy, zz = np.meshgrid(x, y, z)
    XYZ = np.array([yy.flatten(), xx.flatten(), zz.flatten()]).reshape(3, _cube_D_, _cube_D_, _cube_D_)
    XYZ = np.moveaxis(XYZ, 0, 3)
    if not (list(XYZ[0, 1, 3, :]) == [0.0, 1.0, 3.0]):
        print('index of Meta vector went wrong')
        raise TypeError
    cube_xyz = cube_xyz_min[None, None, None, :] + XYZ * cube_D_resol  # shape:(_cube_D_, _cube_D_, _cube_D_, 3)
    # print('cube_xyz_min[None, None, None, :]', cube_xyz_min[None, None, None, :])
    # print('@(*#@!#!@(*$&!@(*')
    # print('cube_xyz[2,3,1,:]', cube_xyz[2,3,1,:])
    # print('cube_xyz[2,3,2,:]', cube_xyz[2, 3, 2, :])
    # print('cube_xyz[2,4,1,:]', cube_xyz[2, 4, 1, :])

    ones = np.ones((_cube_D_, _cube_D_, _cube_D_, 1))
    cube_xyz_matmul = np.concatenate((cube_xyz, ones), axis=3)[None, :, :, :, :,
                      None]  # shape:(1, _cube_D_, _cube_D_, _cube_D_, 4, 1)
    projection_M_matmul = projection_M[:, None, None, None, :, :]  # shape:(N_view, 1, 1, 1, 3, 4)

    project_cube_xyz = np.matmul(projection_M_matmul,
                                 cube_xyz_matmul)  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3, 1)
    # print('@(*#@!#!@(*$&!@(*')
    # print(project_cube_xyz.shape)
    # print('project_cube_xyz[2,3,1,:]', project_cube_xyz[44, 2, 3, 1, :])
    # print('project_cube_xyz[2,3,2,:]', project_cube_xyz[44, 2, 3, 2, :])
    # print('project_cube_xyz[2,4,1,:]', project_cube_xyz[44, 2, 4, 1, :])

    (gx, gy) = count_gx_gy(projection_M, h_length=compress_h, w_length=compress_w)

    Z = project_cube_xyz[:, :, :, :, 2,
        0]  # the depth of each cubic points  shape:(N_view, _cube_D_, _cube_D_, _cube_D_)

    alpha_x = (Z * gx[:, None, None, None] / cube_D_resol)[:, :, :, :,
              None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    alpha_y = (Z * gy[:, None, None, None] / cube_D_resol)[:, :, :, :,
              None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    print('the average pixel a cubic can get on x axis', alpha_x.mean())
    print('the average pixel a cubic can get on y axis', alpha_y.mean())

    tau = project_cube_xyz[:, :, :, :, :, 0] / np.linalg.norm(project_cube_xyz[:, :, :, :, :, 0], axis=4)[:, :, :, :,
                                               None]  # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    vector_xyz = cube_xyz[None, :, :, :, :] - cameraTs[:, None, None, None,
                                              :]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)
    theta = vector_xyz / np.linalg.norm(vector_xyz, axis=4)[:, :, :, :,
                         None]  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    YX = project_cube_xyz[:, :, :, :, :2, 0] / project_cube_xyz[:, :, :, :, 2, 0][:, :, :, :, None]

    H = YX[:, :, :, :, 1][:, :, :, :, None] / compress_h  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    W = YX[:, :, :, :, 0][:, :, :, :, None] / compress_w  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    D = np.zeros(np.shape(H))
    X = H - np.floor(H)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    Y = W - np.floor(W)  # shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    meta_vector = np.concatenate((alpha_x, alpha_y, tau, theta, X, Y), axis=4)
    wrapping_vector = np.concatenate((W, H, D),
                                     axis=4)  # To avoid confusion in notation, let’s note that x corresponds to the width dimension IW, y corresponds to the height dimension IH and z corresponds to the depth dimension ID.

    return (meta_vector, wrapping_vector)


def generate_multiImageMetaVector(
        projection_M,
        compress,
        xyz_3D,
        stage_num,
        cameraTs,
        images_resized,
        angles,
        Ts,
):
    '''

    :param projection_M:
        shape:(N_views, 3, 4)
    :param compress
        turple: (compress_h, compress_w)

    :param stage_num
        int
    :param cameraTs:
        shape:(N_views, 3)
   :param images_resized:resized images
        list
    :return:
        meta_list: list of meta\wrapping vector
            len: stage_num
            ele:
                (vector_image, cameraTs)

    '''

    meta_list = []

    for i in range(stage_num):
        (vector_image) = generateImageMetaVector(
            projection_M,
            compress,
            cameraTs,
            image_size=images_resized[i].shape[1:3]
        )

        # direction_transfer = generateDirectionMetaVector(vector_image,
        #                                                    cameraTs,
        #                                                    xyz_3D,
        #                                                    angles,
        #                                                    Ts,
        #                                                    )

        meta_list.append(vector_image)
        # meta_list.append(direction_transfer)

        compress = (compress[0] * 2, compress[1] * 2)

    return meta_list


def generate_matrix(
        angles,
        ts
):
    (alpha, beta, gamma) = angles

    ratio = 180 / 3.14159
    alpha /= ratio
    beta /= ratio
    gamma /= ratio

    (t_x, t_y, t_z) = ts
    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
    R_rotate = np.matmul(R_x, np.matmul(R_y, R_z))

    t_total = np.array([[t_x], [t_y], [t_z]])
    RT = np.concatenate((R_rotate, t_total), axis=1)

    return R_rotate


def generateDirectionMetaVector(vector_image,
                                cameraTs,
                                BB_middle,
                                angles,
                                ts
                                ):
    R_rotate = generate_matrix(angles, ts)

    s = BB_middle[None, :] - cameraTs  # shape:(N_views,3)
    d_origin = 1

    v_length = (s[:, :, None, None] * vector_image).sum(axis=1)[:, None, :, :]  # shape:(N_views,1,img_w,img_h)
    direction = v_length * vector_image - s[:, :, None, None]  # shape:(N_views,3,img_w,img_h)

    # R_rotate_inverse = np.linalg.inv(R_rotate) #shape(3,3)
    direction_rotate = np.matmul(R_rotate[None, None, None, ...], np.moveaxis(direction, 1, -1)[..., None])
    direction_rotate = np.moveaxis(direction_rotate[:, :, :, :, 0], -1, 1)  # shape:(N_views,3,img_w,img_h)

    td_length = (np.array(ts)[None, :, None, None] * direction_rotate).sum(axis=1)[:, None, :, :]
    dd_length = (direction_rotate * direction_rotate).sum(axis=1)[:, None, :, :]

    direction_transfer = (1 + td_length / (dd_length + 1e-8)) * direction_rotate

    return direction_transfer
    # pdb.set_trace()


def generateImageMetaVector(
        projection_M,
        compress,
        cameraTs,
        image_size
):
    '''

    :param projection_M:
        shape:(N_views, 3, 4)
    :param compress
        turple: (compress_h, compress_w)

    :param cameraTs:
        shape:(N_views, 3)

    :param image_size
        turple
    :return:
        meta_vector: the array of each vector represent camera position
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 10)
        wrapping_vector: the map from each voxel to image
            shape: (N_views, _cube_D_, _cube_D_, _cube_D_, 3)
    '''

    compress_h, compress_w = compress
    img_h, img_w = image_size

    x = np.arange(0, img_w, 1.0) * compress_w
    y = np.arange(0, img_h, 1.0) * compress_h
    xx, yy = np.meshgrid(x, y)
    XY = np.array([yy.flatten(), xx.flatten()]).T.reshape(img_h, img_w, 2)
    XY = np.moveaxis(XY, 2, 0)
    Z = np.ones((1, img_h, img_w))
    XYZ = np.concatenate((XY, Z), axis=0)  # shape:(3,img_h, img_w)

    image_vector_inverse = inverseImageVector(projection_M, XYZ)

    vector_image = image_vector_inverse - cameraTs[..., None, None]
    vector_image = vector_image / (1e-8 + np.linalg.norm(vector_image, axis=1)[:, None, :, :])

    vector_image = np.repeat(XY[None, ...] / 1600, cameraTs.shape[0], axis=0)

    # return (vector_image, cameraTs)
    return (vector_image)

    # ones = np.ones((_cube_D_, _cube_D_, _cube_D_, 1))
    # cube_xyz_matmul = np.concatenate((cube_xyz, ones), axis = 3)[None, :,:,:,:,None]  #shape:(1, _cube_D_, _cube_D_, _cube_D_, 4, 1)
    # projection_M_matmul = projection_M[:,None,None,None,:,:] #shape:(N_view, 1, 1, 1, 3, 4)

    # project_cube_xyz = np.matmul(projection_M_matmul, cube_xyz_matmul) #shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3, 1)

    # (gx, gy) = count_gx_gy(projection_M, h_length = compress_h, w_length = compress_w)

    # Z = project_cube_xyz[:,:,:,:,2,0] #the depth of each cubic points  shape:(N_view, _cube_D_, _cube_D_, _cube_D_)

    # alpha_x = (Z * gx[:,None,None,None] / cube_D_resol)[:, :, :, :, None] # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    # alpha_y = (Z * gy[:,None,None,None] / cube_D_resol)[:, :, :, :, None] # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    # print('the average pixel a cubic can get on x axis', alpha_x.mean())
    # print('the average pixel a cubic can get on y axis', alpha_y.mean())

    # tau = project_cube_xyz[:, :, :, :, :, 0] / np.linalg.norm(project_cube_xyz[:, :, :, :, :, 0], axis = 4)[:, :, :, :, None] # shape:(N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    # vector_xyz = cube_xyz[None, :,:,:,:] - cameraTs[:,None,None,None,:] #shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)
    # theta = vector_xyz / np.linalg.norm(vector_xyz, axis = 4)[:, :, :, :, None] #shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 3)

    # YX = project_cube_xyz[:,:,:,:,:2,0] / project_cube_xyz[:,:,:,:,2,0][:,:,:,:,None]

    # H = YX[:,:,:,:,1][:,:,:,:,None] / compress_h #shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    # W = YX[:, :, :, :, 0][:,:,:,:,None] / compress_w #shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    # D = np.zeros(np.shape(H))
    # X = H - np.floor(H)   #shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)
    # Y = W - np.floor(W)   #shape: (N_view, _cube_D_, _cube_D_, _cube_D_, 1)

    # meta_vector = np.concatenate((alpha_x, alpha_y, tau, theta, X, Y), axis = 4)
    # wrapping_vector = np.concatenate((W, H, D), axis = 4) #To avoid confusion in notation, let’s note that x corresponds to the width dimension IW, y corresponds to the height dimension IH and z corresponds to the depth dimension ID.

    # return (meta_vector, wrapping_vector)


def rotateImageVector(self, pts_3D, image_vector):
    '''
    pts_3D:
        shape:(M_view * N_points * 3)/(N_points * 3)
    image_vector:
        shape:(M_view * N_points * 3 * img_h * img_w)/(N_points * 3 * img_h * img_w)
    ----------------------------------------------------
    pts_3D = np.random.rand(1,2,3)
    image_vector = np.zeros((1,2,3,4,5))
    image_vector[...,:,:] = pts_3D[...,None,None]
    camera_test = Camera()
    q = camera_test.rotateImageVector(pts_3D, image_vector)
    print(q.shape)
    print(q)
    '''
    if (len(image_vector.shape) == 4):
        image_vector = np.moveaxis(image_vector, 1, -1)
        N, img_h, img_w, _ = image_vector.shape
        matrix_x = np.zeros((N, img_h, img_w, 3, 3))
        matrix_yz = np.zeros((N, img_h, img_w, 3, 3))
    elif (len(image_vector.shape) == 5):
        image_vector = np.moveaxis(image_vector, 2, -1)
        M, N, img_h, img_w, _ = image_vector.shape
        matrix_x = np.zeros((M, N, img_h, img_w, 3, 3))
        matrix_yz = np.zeros((M, N, img_h, img_w, 3, 3))
    else:
        raise ValueError('inputs shape is wrong')
    a = pts_3D[..., 0]
    b = pts_3D[..., 1]
    c = pts_3D[..., 2]
    # print(pts_3D[...,1:])
    norm_bc = np.linalg.norm(pts_3D[..., 1:], axis=-1)
    norm_abc = np.linalg.norm(pts_3D[..., :], axis=-1)

    matrix_x[..., 0, 0] = 1.0
    matrix_x[..., 1, 1] = (c / norm_bc)[..., None, None]
    matrix_x[..., 1, 2] = (b / norm_bc)[..., None, None]
    matrix_x[..., 2, 1] = (-b / norm_bc)[..., None, None]
    matrix_x[..., 2, 2] = (c / norm_bc)[..., None, None]

    matrix_yz[..., 1, 1] = 1.0
    matrix_yz[..., 0, 0] = (norm_bc / norm_abc)[..., None, None]
    matrix_yz[..., 0, 2] = (a / norm_abc)[..., None, None]
    matrix_yz[..., 2, 0] = (-a / norm_abc)[..., None, None]
    matrix_yz[..., 2, 2] = (norm_bc / norm_abc)[..., None, None]

    self.matrix_R = np.matmul(matrix_x, matrix_yz)
    image_vector = np.matmul(image_vector[..., None, :], self.matrix_R)
    image_vector = image_vector[..., 0, :]
    image_vector = np.moveaxis(image_vector, -1, -3)
    return (image_vector)


def inverseImageVector(
        projection_M,
        image_vector):
    '''
     projection_M:
        shape:(N_views, 3, 4)
    image_vector:
        shape:(3 * img_h * img_w)

    :return
        image_vector_inverse
            shape:(N_views,3,img_h, img_w)

    ----------------------------------------------------
    '''
    image_vector = np.moveaxis(image_vector, 0, -1)
    N_Ms = projection_M.shape[0]
    img_h, img_w, _ = image_vector.shape
    image_vector_new = np.ones((1, img_h, img_w, 4))
    image_vector_new[..., 0:3] = image_vector

    projection_new = np.zeros((N_Ms, 4, 4))
    projection_new[:, 0:3, :] = projection_M
    projection_new[:, 3, :] = np.array(([[0, 0, 0, 1]]))
    projection_new = np.linalg.inv(projection_new)[:, None, None, :, :]  # shape:(N_views,img_h, img_w, 4, 4)

    image_vector_inverse = np.matmul(projection_new, image_vector_new[..., None])  # shape:(N_views,img_h, img_w, 4, 1)

    image_vector_inverse = image_vector_inverse[..., 0:3, 0]
    image_vector_inverse = np.moveaxis(image_vector_inverse, -1, -3)  # shape:(N_views,3,img_h, img_w)
    return (image_vector_inverse)


def generateVectorImage(self,
                        projection_M,
                        xyz_3D,
                        img_shape=(50, 50),
                        vector_type='world',  # world/camera/camera_rotate
                        ):
    '''
    np.random.seed(201611)
    #Ms = (np.random.rand(1,3,4) + ) / 10
    Ms = np.array([[[2607.429996,-3.844898,1498.178098,-533936.661373],
                    [-192.076910,2862.552532,681.798177,23434.686572],
                    [-0.241605,-0.030951,0.969881,22.540121]]])
    Ms = np.array([[[1,0,0,0],[0,1,0,0],[0,0,0.2,0]]])
    pts_3D = np.random.rand(2,3)
    pts_3D = np.array([[10.0,10.0,10.0]])

    camera_test = Camera()
    image_vector = camera_test.generateVectorImage(Ms, pts_3D)
    image_vector_show = image_vector/2 + 0.5
    np.moveaxis(image_vector_show[0,0],0,-1)
    import matplotlib.pyplot as plt
    plt.imshow(np.transpose(image_vector_show[0,0], (1, 2, 0)))
    plt.show()
    '''
    if projection_M.shape[-2:] != (3, 4):
        raise ValueError(
            "perspectiveProj needs projection_M with shape (3,4), however got {}".format(projection_M.shape))

    if xyz_3D.ndim == 1:
        xyz_3D = xyz_3D[None, :]

    if xyz_3D.shape[1] != 3 or xyz_3D.ndim != 2:
        raise ValueError(
            "perspectiveProj needs xyz_3D with shape (3,) or (N_pts, 3), however got {}".format(xyz_3D.shape))
    # perspective projection
    N_pts = xyz_3D.shape[0]
    self.N_pts = N_pts
    N_Ms = projection_M.shape[0]
    self.N_Ms = N_Ms
    xyz1 = np.c_[xyz_3D, np.ones((N_pts, 1))].astype(np.float64)  # (N_pts, 3) ==> (N_pts, 4)
    pts_3D = np.matmul(projection_M, xyz1.T)  # (3, 4)/(N_Ms, 3, 4) * (4, N_pts) ==> (3, N_pts)/(N_Ms,3,N_pts)
    # the result is vector: [w,h,1], w is the first dim!!! (matlab's x/y/1')
    pts_3D = pts_3D.swapaxes(1, 2)
    pts_3D_new = pts_3D[:, :, :, np.newaxis, np.newaxis]

    img_h, img_w = img_shape
    x = np.arange(0, img_w, 1.0)
    y = np.arange(0, img_h, 1.0)
    xx, yy = np.meshgrid(x, y)
    XY = np.array([yy.flatten(), xx.flatten()]).T.reshape(img_h, img_w, 2)
    XY = np.moveaxis(XY, 2, 0)
    Z = np.ones((1, img_h, img_w))
    XYZ = np.concatenate((XY, Z), axis=0)
    XYZ = XYZ[np.newaxis, np.newaxis, :, :, :]

    if (vector_type == 'camera_rotate'):
        image_vector = self.rotateImageVector(pts_3D, pts_3D_new - XYZ)
        image_vector_norm = np.linalg.norm(image_vector, axis=2)[:, :, np.newaxis, :, :]
        image_vector = image_vector / image_vector_norm
        # print(image_vector.shape)
        # print(image_vector[1,1,:,2,3])
        return image_vector
    elif (vector_type == 'world'):
        image_vector = self.inverseImageVector(projection_M, -pts_3D_new + XYZ)
        image_vector_norm = np.linalg.norm(image_vector, axis=2)[:, :, np.newaxis, :, :]
        image_vector = image_vector / image_vector_norm
        # print(image_vector.shape)
        # print(image_vector[1,1,:,2,3])
        return image_vector


def K_partition(cameraKO, compress_ratio_h=4.0, compress_ratio_w=4.0, img_size=(1200, 1600)):
    cx = cameraKO[0][2]
    cy = cameraKO[1][2]

    principal_coords_list = []
    partition_num = int(compress_ratio_h * compress_ratio_w)

    h = int(img_size[0] / compress_ratio_h)
    w = int(img_size[1] / compress_ratio_w)
    for i in range(compress_ratio_h):
        for j in range(compress_ratio_w):
            cxx = cx - j * w
            cyy = cy - i * h
            principal_coord = (cxx, cyy)
            principal_coords_list.append(principal_coord)

    cameraKOs = np.empty((partition_num, 3, 3), dtype=np.float64)
    for k in range(partition_num):
        cameraKOs[k] = cameraKO
        cameraKOs[k][0][2] = principal_coords_list[k][0]
        cameraKOs[k][1][2] = principal_coords_list[k][1]

    return cameraKOs

def partition_image_and_matrix(images,
                               cameraPO4s_model,
                               cameraRTO4s_model,
                               cameraKO4s_model,
                               # image_compress_stage,
                               # return_list=False,
                               compress_ratio_h=4.0,
                               compress_ratio_w=4.0):
    '''
    Args:
        images, list, (N_view, 3, img_h, img_w)
        cameraPO4s_model, (N_view, 4, 4)
        compress_ratio_h: compress_ratio for the H dimension.
        compress_ratio_w: compress_ratio for the W dimension.

    Outputs:
        parti_imgs, (N_view, N_partition, 3, parti_h, parti_w)
        _cameraP04s, (N_view, N_partition, 4, 4).

    '''
    num_view = len(images)
    num_partition = int(compress_ratio_w * compress_ratio_h)
    parti_h = int(images[0].shape[1] / compress_ratio_h)
    parti_w = int(images[0].shape[2] / compress_ratio_w)

    # parti_imgs = []
    parti_imgs = np.empty((num_view, num_partition, 3, parti_h, parti_w), dtype=np.float64)
    # cameraPO4s = np.empty((num_view, num_partition, 4, 4), dtype=np.float64)
    # print('images.shape: ', images[0].shape)

    for view_i in range(num_view):
        for partition_j in range(num_partition):
            start_h_idx = math.floor(partition_j / compress_ratio_w)
            start_w_idx = (partition_j % compress_ratio_w)

            start_h_pix = start_h_idx * parti_h
            start_w_pix = start_w_idx * parti_w

            final_h_pix = start_h_pix + parti_h
            final_w_pix = start_w_pix + parti_w

            # parti_imgs.append(images[view_i][start_h_pix: final_h_pix, start_w_pix: final_w_pix, :])
            parti_imgs[view_i, partition_j, :, :, :] = images[view_i][:, start_h_pix: final_h_pix,
                                                       start_w_pix: final_w_pix]
    # print('^^^^^^^^^^', parti_imgs.shape)

    _cameraP04s, _, _ = CameraPOs_as_torch_partitioned(cameraPO4s_model, cameraRTO4s_model, cameraKO4s_model,
                                                       compress_ratio_h=compress_ratio_h, compress_ratio_w=compress_ratio_w,
                                                       img_size=(images[0].shape[1], images[0].shape[2]))
    return parti_imgs, _cameraP04s

def CameraPOs_as_torch_partitioned(cameraPO4s_model, cameraRTO4s_model, cameraKO4s_model,
                                   compress_ratio_h=4.0, compress_ratio_w=4.0, img_size=(1200, 1600)):
    '''
    Args:
        cameraKO4s_models: (N_view, 3, 3)
    outputs:
        cameraP04s_: (N_view, N_partition, 4, 4).
    '''

    # num_model = len(cameraKO4s_models)
    num_view = cameraPO4s_model.shape[0]
    num_partition = int(compress_ratio_w * compress_ratio_h)

    # modify the dimension from (3,4) to (4,4)
    cameraPO4s = np.empty((num_view, num_partition, 3, 4), dtype=np.float64)
    cameraRTO4s = np.empty((num_view, num_partition, 3, 4), dtype=np.float64)
    cameraKO4s = np.empty((num_view, num_partition, 3, 3), dtype=np.float64)

    # for i in range(num_model):
    for j in range(num_view):
        cameraK0 = cameraKO4s_model[j]
        cameraK0s = K_partition(cameraK0, compress_ratio_h, compress_ratio_w, img_size)  # (num_partition, 3, 3)
        for k in range(num_partition):
            cameraKO4s[j][k] = cameraK0s[k]
            cameraRTO4s[j][k] = cameraRTO4s_model[j]
            cameraPO4s[j][k] = np.dot(cameraK0s[k], cameraRTO4s_model[j])

    #  concatenation for PO4, from (..., 3, 4) to (..., 4, 4).
    ones1 = np.repeat(np.array([[[0, 0, 0, 1]]]), repeats=num_partition, axis=0)
    ones2 = np.repeat(np.expand_dims(ones1, axis=0), repeats=num_view, axis=0)
    # ones3 = np.repeat(np.expand_dims(ones2, axis=0), repeats=num_model, axis=0)
    cameraP04s = np.concatenate((cameraPO4s, ones2), axis=2)

    #print('cameraP04s shape: ', cameraP04s.shape)

    # total_num = num_partition * num_view * num_model
    # ones = np.repeat(np.array([[[[[0, 0, 0, 1]]]]]), repeats=total_num, axis=0)
    # cameraP04s = np.concatenate((cameraPO4s, ones), axis=3)

    _cameraP04s = torch.from_numpy(cameraP04s).type(torch.FloatTensor)
    _cameraRTO4s = torch.from_numpy(cameraRTO4s).type(torch.FloatTensor)
    _cameraKO4s = torch.from_numpy(cameraKO4s).type(torch.FloatTensor)

    return _cameraP04s, _cameraRTO4s, _cameraKO4s

def resize_multistage_image_and_matrix(images,
                                       projection_M,
                                       intrinsic_K,
                                       cube_xyz_min,
                                       cube_D_mm,
                                       _cube_D_,
                                       image_compress_multiple,
                                       image_compress_stage,
                                       return_list=False,
                                       compress_ratio=1.0):
    # guangyu
    # input intrinsic_K (N_views, 3, 3).
    '''
    compress image and garantee the camera position is not changing
    :param images:  all images of one model
        type:list or None
        if list
            list element: image array
                shape: (img_h,img_w, 3)

    :param projection_M:  camera matrix
        shape: (N_views, 3, 4)
    :param intrinsic_K: intrinsic matrix
        shape: (N_views, 3, 3)
    :param extrinsic_RT: extrinsic matrix
        shape: (N_views, 3, 4)
    :param cube_xyz_min:  min xyz coordinate
        shape: (3,) / (N_pts, 3)  usually it is (3,) because we only sample one cubic to judge the resize term
    :param cube_D_mm:
        cubic length float
    :param _cube_D_:
        cubic size int
    :param image_compress_multiple:
        same as param.image_compress_multiple
    :param image_compress_stage
        same as param.image_compress_stage
    :param return_list: bool
        if False return the numpy array
    :param compress_ratio
        see self.params.compress_ratio
    :return:
        if image is not None
            image_resized_stage_list:multistage of resized image
                length : = image_compress_stage
                ele in each list:
                    shape:(N_view, img_h_new//2**iter, img_w_new//2**iter)
            projection_M_new: new cameraP
                shape:(N_view,3,4)
            (compress_h_new,compress_w_new):(float,float)
        elif image is None: only change the matrix
            projection_M_new: new cameraP
                shape:(N_view,3,4)
            (compress_h_new,compress_w_new):(float,float)
    '''

    compress_h = compress_ratio
    compress_w = compress_ratio
    resized_h = int(image_compress_multiple * (images[0].shape[0] // (compress_h * image_compress_multiple)))
    resized_w = int(image_compress_multiple * (images[0].shape[1] // (compress_w * image_compress_multiple)))

    # pdb.set_trace()
    compress_h_new = images[0].shape[0] / (resized_h + 0.0)
    compress_w_new = images[0].shape[1] / (resized_w + 0.0)
    transform_matrix = np.array([[[1 / compress_w_new, 0, 0], [0, 1 / compress_h_new, 0], [0, 0, 1]]])
    projection_M_new = np.matmul(transform_matrix, projection_M)
    # guangyu
    # calculate the K after resizing.
    intrinsic_K_new = np.matmul(transform_matrix, intrinsic_K)

    cameraTs = cameraPs2Ts(projection_M)
    cameraTs_new = cameraPs2Ts(projection_M_new)
    trans_vector = (cameraTs - cameraTs_new)[:, :, None]
    identical_matrix = np.repeat(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), cameraTs.shape[0], axis=0)
    bottom_matrix = np.repeat(np.array([[[0, 0, 0, 1]]]), cameraTs.shape[0], axis=0)
    transform_matrix2 = np.concatenate((identical_matrix, trans_vector), axis=2)
    transform_matrix2 = np.concatenate((transform_matrix2, bottom_matrix), axis=1)
    projection_M_new_f = np.concatenate((projection_M_new, bottom_matrix), axis=1)

    projection_M_new = np.matmul(transform_matrix2, projection_M_new_f)
    projection_M_new = projection_M_new[:, :3, :]

    if (images is not None):
        image_resized_stage_list = []
        for iter in range(image_compress_stage):
            image_resized_list = []
            for image in images:
                # print('resized image shape',resized_h, resized_w)
                image_resized = np.array(
                    Image.fromarray(image).resize(size=(int(resized_w // (2 ** iter)), int(resized_h // (2 ** iter))))
                )
                # image_resized = scipy.misc.imresize(image,
                #                                     size=(int(resized_h // (2 ** iter)), int(resized_w // (2 ** iter))),
                #                                     interp='bicubic')
                image_resized = image_resized / 256.0 - 0.5
                image_resized_list.append(image_resized)
            images_resized = image_resized_list if return_list else np.stack(image_resized_list)
            image_resized_stage_list.append(images_resized)
        return (image_resized_stage_list, projection_M_new, intrinsic_K_new, (compress_h_new, compress_w_new))
    else:
        return (None, projection_M_new, intrinsic_K_new, (compress_h_new, compress_w_new))

def resize_matrix(projection_M, intrinsic_K, compress_ratio_total):
    """
    input:
        projection_M, (N_view, 3, 4).
    output:
        projection_M_new: (N_view, 3, 4)
        intrinsic_K_new: (N_view, 3, 3).

    """
    compress_w_new = compress_ratio_total
    compress_h_new = compress_ratio_total
    transform_matrix = np.array([[[1 / compress_w_new, 0, 0], [0, 1 / compress_h_new, 0], [0, 0, 1]]])
    projection_M_new = np.matmul(transform_matrix, projection_M)
    # guangyu.
    # calculate the K after resizing.
    intrinsic_K_new = np.matmul(transform_matrix, intrinsic_K)

    cameraTs = cameraPs2Ts(projection_M)
    cameraTs_new = cameraPs2Ts(projection_M_new)
    trans_vector = (cameraTs - cameraTs_new)[:, :, None]
    identical_matrix = np.repeat(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]), cameraTs.shape[0], axis=0)
    bottom_matrix = np.repeat(np.array([[[0, 0, 0, 1]]]), cameraTs.shape[0], axis=0)
    transform_matrix2 = np.concatenate((identical_matrix, trans_vector), axis=2)
    transform_matrix2 = np.concatenate((transform_matrix2, bottom_matrix), axis=1)
    projection_M_new_f = np.concatenate((projection_M_new, bottom_matrix), axis=1)

    projection_M_new = np.matmul(transform_matrix2, projection_M_new_f)
    projection_M_new = projection_M_new[:, :3, :]
    return projection_M_new, intrinsic_K_new

