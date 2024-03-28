import pdb
import numpy as np
import os
import cv2
import imageio
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from glob import glob

def intrinsics_from_xml(xml_file, f=None):
    root = ET.parse(xml_file).getroot()
    intrinsics = {}
    img_sizes = {}
    for e in root.findall('chunk/sensors')[0].findall('sensor'):
        sensor_id = e.get('id')
        calibration = e.find('calibration')
        resolution = e.find('resolution')
        width = float(resolution.get('width'))
        height = float(resolution.get('height'))
        # f = 7963.8462
        f = float(calibration.find('f').text)
        delta_cx = float(calibration.find('cx').text)
        delta_cy = float(calibration.find('cy').text)
        # delta_cx = 0.
        # delta_cy = 0.
        cx = width / 2 + delta_cx
        cy = height / 2 + delta_cy
        intrinsics[sensor_id] = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        img_sizes[sensor_id] = (width, height)
    return intrinsics, img_sizes

def extrinsics_from_xml(xml_file, verbose=False, group=None):
    if group is not None:
        root = ET.parse(xml_file).getroot()
        extrinsics = {}
        ext_sids = {}

        for g in root.findall('chunk/cameras')[0].findall('group'):
            label = g.get('label')
            if label == group:
                for e in g.findall('camera'):
                    label = e.get('label')
                    sensor_id = e.get('sensor_id')
                    try:
                        transforms = e.find('transform').text
                        extrinsic = np.array([float(x) for x in transforms.split()]).reshape(4, 4)
                        # extrinsic[:, 1:3] *= -1
                        extrinsics[label] = np.linalg.inv(extrinsic)
                        ext_sids[label] = sensor_id
                    except:
                        if verbose:
                            print('failed to align camera', label)

    else:
        root = ET.parse(xml_file).getroot()
        extrinsics = {}
        ext_sids = {}
        for e in root.findall('chunk/cameras')[0].findall('camera'):
            label = e.get('label')
            sensor_id = e.get('sensor_id')
            try:
                transforms = e.find('transform').text
                extrinsic = np.array([float(x) for x in transforms.split()]).reshape(4, 4)
                # extrinsic[:, 1:3] *= -1
                extrinsics[label] = np.linalg.inv(extrinsic)
                ext_sids[label] = sensor_id
            except:
                if verbose:
                    print('failed to align camera', label)
    return extrinsics, ext_sids

if __name__ == "__main__":
    dsp_factor = 1
    dsp_ratio = 1. / dsp_factor
    key = ''

    subject_file = "/media/womoer/Wdata/data/LY/GuLangYu"
    in_img_file = os.path.join(subject_file, "images")
    xml_file = os.path.join(subject_file, "cams.xml")

    intrinsics, img_size = intrinsics_from_xml(xml_file, f=None)
    extrinsics, ext_sids = extrinsics_from_xml(xml_file, group=None)

    txt_path = os.path.join(subject_file, 'cams_{}'.format(dsp_factor))
    img_path = os.path.join(subject_file, 'images_{}'.format(dsp_factor))
    os.makedirs(txt_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    # write
    # try:
    #     os.makedirs(txt_path)
    #     os.makedirs(img_path)
    # except os.error:
    #     print(txt_path + ' already exist.')
    for label, extrinsic in extrinsics.items():
        print(label)
        sid = ext_sids[label]
        # label = 'H' + label
        im_name = os.path.join(in_img_file, label+".JPG")
        if not os.path.exists(im_name):
            continue
            # raise ValueError('Current label not exists in undistorted')
        im = cv2.imread(im_name)
        w, h = img_size[sid]
        assert w == im.shape[1] and h == im.shape[0]
        im = cv2.resize(im, (int(w / dsp_factor), int(h / dsp_factor)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(img_path, key+label+".JPG"), im)

        intrinsic = intrinsics[sid].copy()
        # print(intrinsic)
        intrinsic[:2, :] *= dsp_ratio
        # print(intrinsic)
        with open(os.path.join(txt_path, '{}_cam.txt'.format(key+label)), 'w') as f:
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[j, k]) + ' ')
                f.write('\n')
        # exit(0)