import scipy.io
import os
import numpy as np

import scipy.io
import os
import numpy as np


def get_BB_models(datasetFolder,
                  BBNamePattern,
                  modelList,
                  datasetName,
                  outer_bound_factor=0.1,):

    BB_models = []
    for model in modelList:
        if datasetName == 'DTU':
            BBName = BBNamePattern.replace('#', str(model).zfill(1))
            BB_filePath = os.path.join(datasetFolder, BBName)
            BB_matlab_var = scipy.io.loadmat(BB_filePath)  # matlab variable

            reconstr_sceneRange = BB_matlab_var['BB'].T
            size_reconstr_sceneRange = reconstr_sceneRange[:, 1] - reconstr_sceneRange[:, 0]
            reconstr_sceneRange_low = reconstr_sceneRange[:, 0] - outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange_up = reconstr_sceneRange[:, 1] + outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange = np.concatenate((reconstr_sceneRange_low[:, None], reconstr_sceneRange_up[:, None]),
                                                 axis=1)
            # [-73 129], [-197  183], [472 810]
            # self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            # self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
            BB = reconstr_sceneRange
            BB_models.append(BB)
        elif datasetName == 'tanks_COLMAP':
            # zhiwei
            BBName = BBNamePattern.replace('#', str(model))
            BB_filePath = os.path.join(datasetFolder, BBName)
            # BB_matlab_var = scipy.io.loadmat(BB_filePath)  # matlab variable
            # reconstr_sceneRange = BB_matlab_var['BB'].T
            reconstr_sceneRange = np.load(BB_filePath)
            size_reconstr_sceneRange = reconstr_sceneRange[:, 1] - reconstr_sceneRange[:, 0]
            reconstr_sceneRange_low = reconstr_sceneRange[:, 0] - outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange_up = reconstr_sceneRange[:, 1] + outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange = np.concatenate((reconstr_sceneRange_low[:, None], reconstr_sceneRange_up[:, None]),
                                                 axis=1)
            # [-73 129], [-197  183], [472 810]
            # self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            # self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
            BB = reconstr_sceneRange
            BB_models.append(BB)
        elif datasetName == 'blendedMVS':
            # zhiwei
            BBName = BBNamePattern.replace('#', str(model))
            BB_filePath = os.path.join(datasetFolder, BBName)
            # BB_matlab_var = scipy.io.loadmat(BB_filePath)  # matlab variable
            # reconstr_sceneRange = BB_matlab_var['BB'].T
            reconstr_sceneRange = np.load(BB_filePath)
            size_reconstr_sceneRange = reconstr_sceneRange[:, 1] - reconstr_sceneRange[:, 0]
            reconstr_sceneRange_low = reconstr_sceneRange[:, 0] - outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange_up = reconstr_sceneRange[:, 1] + outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange = np.concatenate((reconstr_sceneRange_low[:, None], reconstr_sceneRange_up[:, None]),
                                                 axis=1)
            # [-73 129], [-197  183], [472 810]
            # self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            # self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
            BB = reconstr_sceneRange
            BB_models.append(BB)
        elif datasetName == 'giga_ours':
            # zhiwei
            BBName = BBNamePattern.replace('#', str(model))
            BB_filePath = os.path.join(datasetFolder, BBName)
            # BB_matlab_var = scipy.io.loadmat(BB_filePath)  # matlab variable
            # reconstr_sceneRange = BB_matlab_var['BB'].T
            reconstr_sceneRange = np.load(BB_filePath)
            size_reconstr_sceneRange = reconstr_sceneRange[:, 1] - reconstr_sceneRange[:, 0]
            reconstr_sceneRange_low = reconstr_sceneRange[:, 0] - outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange_up = reconstr_sceneRange[:, 1] + outer_bound_factor * size_reconstr_sceneRange
            reconstr_sceneRange = np.concatenate((reconstr_sceneRange_low[:, None], reconstr_sceneRange_up[:, None]),
                                                 axis=1)
            # [-73 129], [-197  183], [472 810]
            # self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
            # self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
            BB = reconstr_sceneRange
            BB_models.append(BB)

    return BB_models

# def get_BB_models(
#                   datasetFolder,
#                   BBNamePattern,
#                   modelList,
#                   outer_bound_factor = 0.1,
#                   ):
#     BB_models = []
#     for model in modelList:
#
#         BBName = BBNamePattern.replace('#',str(model).zfill(1))
#         BB_filePath = os.path.join(datasetFolder, BBName)
#         BB_matlab_var = scipy.io.loadmat(BB_filePath)  # matlab variable
#
#         reconstr_sceneRange = BB_matlab_var['BB'].T
#         size_reconstr_sceneRange = reconstr_sceneRange[:, 1] - reconstr_sceneRange[:, 0]
#         reconstr_sceneRange_low = reconstr_sceneRange[:, 0] - outer_bound_factor * size_reconstr_sceneRange
#         reconstr_sceneRange_up = reconstr_sceneRange[:, 1] + outer_bound_factor * size_reconstr_sceneRange
#         reconstr_sceneRange = np.concatenate((reconstr_sceneRange_low[:, None], reconstr_sceneRange_up[:, None]),
#                                                   axis=1)
#         # [-73 129], [-197  183], [472 810]
#         # self.reconstr_sceneRange = np.asarray([(-20, 20), (100, 140), (640, 670)])
#         # self.BB = self.reconstr_sceneRange if self.debug_BB else self.BB_matlab_var['BB'].T   # np(3,2)
#         BB = reconstr_sceneRange
#         BB_models.append(BB)
#     return BB_models