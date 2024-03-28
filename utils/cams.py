import numpy as np

import os

import scipy.io
import scipy.misc
import random
import imageio



def blurImgs(imgs_list, kernelRadius):
    """

    -------
    inputs:
        imgs_list: list of images
        kernelRadius: kernel size is (kernelRadius*2+1)

    -------
    outputs:
        imgs_blured_list: list of blured imgs

    """

    imgs_blured_list = []
    for _img in imgs_list:
        _img_blured = scipy.ndimage.gaussian_filter(_img, sigma=[kernelRadius, kernelRadius, 0])
        imgs_blured_list.append(_img_blured)
    return imgs_blured_list

def readImages( datasetFolder,
               imgNamePattern,
               viewList,
               datasetName,
               return_list=True):
    """
    Only select the images of the views listed in the viewList.
    We assume that the view index is large or equal than 0
        &&
        the images' sizes are equal.

    ---------
    inputs:
        datasetFolder: where the dataset locates
        imgNamePattern: different dataset have different name patterns for images. Remember to include the subdirecteries, e.g. "x/x/xx.png"
                Replace '#' --> '{:03}'; '@' --> '{}'
        viewList: list the view index, such as [11, 1, 30, 6]
        return_list: True.  Return list if true else np.

    ---------
    outputs:
        imgs_list: list of the images
            or
        imgs_np: np array with shape of (len(viewList), img_h, img_w, 3)

    ---------
    usages:
    >>> imgs_np = readImages(".", "test/Lasagne0#.jpg", [6,6], return_list = False)     # doctest need to run in the local dir
    loaded img ./test/Lasagne0006.jpg
    loaded img ./test/Lasagne0006.jpg
    >>> imgs_np.shape
    (2, 225, 225, 3)
    """

    imgs_list = []

    for i, viewIndx in enumerate(viewList):
        # we assume the name pattern looks like 'x/x/*001*.xxx', if {:04}, add one 0 in the pattern: '*0#*.xxx'
        #zhiwei
        if datasetName == 'DTU':
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx)))
        elif datasetName == 'tanks_COLMAP':
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx)))
        elif datasetName == 'blendedMVS':
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx)))
        elif datasetName == 'giga_ours':
            imgPath = os.path.join(datasetFolder, imgNamePattern.replace('#', '{:03}'.format(viewIndx)))

        # img = scipy.misc.imread(imgPath)  # read as np array
        img = np.array(imageio.imread(imgPath))
        imgs_list.append(img)
        # print('loaded img ' + imgPath)

    return imgs_list if return_list else np.stack(imgs_list)

def readAllImages( datasetFolder,
               imgNamePattern,
               viewList,
               modelList,
               light_condition,
               return_list=False):
    '''

    :param datasetFolder:
    :param imgNamePattern:
    :param viewList:
    :param modelList:
    :param light_condition:
    :param return_list:
    :return:
        return the list of the list of the image
        this is a list of different models, each element of the list is the output of readImages()
        type:
            [[imgs_list],[imgs_list],...,[imgs_list]]
    '''

    imgsAllList = []

    for model in modelList:
        if light_condition is 'random':
            light_num = random.choice(['1', '2', '3', '4', '5', '6'])
        else:
            light_num = light_condition
        print('model', model)
        imgNamePattern_replace = imgNamePattern.replace('$', str(model)).replace('&', light_num)
        images_list = readImages(
            datasetFolder=datasetFolder,
            imgNamePattern=imgNamePattern_replace,
            viewList=viewList,
            return_list=True)
        imgsAllList.append(images_list)

    return imgsAllList


def readAllImagesAllLight( datasetFolder,
                           datasetName,
                           imgNamePattern,
                           viewList,
                           modelList,
                           light_condition_list,
                           return_list=False):
    '''

    :param datasetFolder:
    :param imgNamePattern:
    :param viewList:
    :param modelList:
    :param light_condition:
    :param return_list:
    :return:
        return the list of the list of the image
        this is a list of different models, each element of the list is the output of readImages()
        type:
            [[imgs_list],[imgs_list],...,[imgs_list]]
    '''


    imgsAll = {}

    for light_condition in light_condition_list:

        imgsAllList = []

        for model in modelList:

            light_num = light_condition
            print('model', model)
            #zhiwei
            if datasetName == 'DTU':
                imgNamePattern_replace = imgNamePattern.replace('$', str(model)).replace('&', light_num)
            elif datasetName == 'tanks_COLMAP':
                imgNamePattern_replace = imgNamePattern.replace('$', str(model))
            elif datasetName == 'blendedMVS':
                imgNamePattern_replace = imgNamePattern.replace('$', str(model))
            elif datasetName == 'giga_ours':
                imgNamePattern_replace = imgNamePattern.replace('$', str(model))

            images_list = readImages(
                datasetFolder=datasetFolder,
                imgNamePattern=imgNamePattern_replace,
                viewList=viewList,
                datasetName = datasetName,
                return_list=True)
            imgsAllList.append(images_list)
        imgsAll[light_condition] = imgsAllList

    return imgsAll

