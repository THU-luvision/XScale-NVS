# 💠XScale-NVS: Cross-Scale Novel View Synthesis with Hash Featurized Manifold

This repository is the official implementation of the CVPR'24 paper titled "XScale-NVS: Cross-Scale Novel View Synthesis with Hash Featurized Manifold".

[![Website](doc/badge-website.svg)](https://xscalenvs.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.02145)

Guangyu Wang,
Jinzhi Zhang,
Fan Wang,
[Ruqi Huang](https://rqhuang88.github.io/),
[Lu Fang](http://www.luvision.net/)

![qualitative_teaser_all](https://github.com/THU-luvision/XScale-NVS/assets/70793950/f1289420-f22f-4ae3-a757-ba8326bbe749)
We propose XScale-NVS for high-fidelity cross-scale novel view synthesis of real-world large-scale scenes. The core is to unleash the expressivity of hash-based featurization by explicitly prioritizing the sparse manifold. Our method demonstrates state-of-the-art results on various challenging real-world scenes, effectively representing highly detailed contents independent of the geometric resolution.

## 🌟 Motivation
![method](https://github.com/THU-luvision/XScale-NVS/assets/70793950/ff4e532d-0803-4fa1-8b7b-558ef4b77e05)
(a) UV-based featurizations tend to disorganize the feature distribution due to the inevitable distortions in surface parametrization. (b) Existing 3D-surface-based featurizations fail to express the sub-primitive-scale intricate details given the limited discretization resolution. (c) Volumetric featurizations inevitably yield a dispersed weight distribution during volume rendering, where many multi-view inconsistent yet highly weighted samples ambiguate surface colour and deteriorate surface features with inconsistent colour gradient. (d) Our method leverages hash encoding to unlock the dependence of featuremetric resolution on discretization resolution, while simultaneously utilizes rasterization to fully unleash the expressivity of volumetric hash encoding by propagating clean and multi-view consistent signals to surface features.

## 📢 News
2024-04-01: <a href="https://arxiv.org/abs/2312.02145"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" height="16"></a> Paper and code release. <br>
2024-02-27: Accepted to <b>CVPR 2024</b>. <br>

## 🛠️ Setup

The code was tested on:

- Ubuntu 20.04, Python 3.9.16, CUDA 11.4, GeForce RTX 3090

### 💻 Dependencies

Create the environment and install dependencies using **conda** and <b>pip</b>:

```bash
conda create -n xscalenvs --file environment.yml
conda activate xscalenvs
```

This implementation is built upon [pytorch](https://pytorch.org/), [tinycudann](https://github.com/NVlabs/tiny-cuda-nn), and [nvdiffrast](https://github.com/NVlabs/nvdiffrast).

## 🚀 Usage on custom scenes

### 📷 Prepare images

Carefully collect the multi-view images, since reconstruction quality is 100% tied to view sampling density. Compared to algorithmic improvements, denser coverage of the scene is always the most straight-forward yet effective way to boost the performance. Please refer to this [Guide](https://devstudio.dartmouth.edu/wordpress/metashape-guide/) for advanced capture tutorials.

### 🎮 Multi-view 3D reconstruction (A.K.A. Photogrammetry)

Estimate per-image camera parameters and reconstruct the dense geometry (in the form of triangle mesh) of the scene. Here we recommond to use off-the-shelf software **Agisoft Metashape** or **COLMAP** to finish this step:

- **Using Agisoft Metashape:**
 
    - Command-line Interface: specify the configs (including file names and parameters for reconstruction) and then run by ```python -u scripts/run_metashape.py```. The default parameters generally work well for most real-world scenes.
    - GUI: follow the *Basic Workflow* in [Metashape Guide - DEV Studio](https://devstudio.dartmouth.edu/wordpress/metashape-guide/)

- **Using COLMAP:** Please refer to the official documentation for [command-line interface](https://colmap.github.io/cli.html#) or [GUI](https://colmap.github.io/tutorial.html).

### ⚙️ File formats

After photogrammetry, export the undistorted images, camera parameters, and the reconstructed mesh model ```1.obj``` in the folder ```SCENE_NAME``` as:
```
SCENE_NAME                          
├── images_{dsp_factor}                 
│   ├── IMGNAME1.JPG       
│   ├── IMGNAME2.JPG       
│   └── ...                
├── cams_{dsp_factor}                   
│   ├── IMGNAME1_cam.txt   
│   ├── IMGNAME2_cam.txt   
│   └── ...                
└── 1.obj              
```

The camera convention strictly follows [MVSNet](https://github.com/YoYo000/MVSNet/tree/master), where the camera parameters are defined in the ```.txt``` file, with the extrinsic `E = [R|t]` and intrinsic `K` being expressed as:

```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22
```

### 🔄 Camera conversion

- For **Agisoft Metashape**, convert the resulting metashape camera file ```cams.xml``` to the ```cams``` folder using ```scripts/xml2txt.py```, where the following parameters are needed to be specified:

    - ```dsp_factor```: the down-sample rate, e.g., ```dsp_factor=4``` means down-sampling the resulting images and the related intrinsic parameters by a factor of 4.

    - ```subject_file```: the root path contains the exported image folder ```images``` and ```cams.xml```.

    The outputs are the two folders namely ```images_{dsp_factor}``` and ```cams_{dsp_factor}```.

- For **COLMAP**, please refer to [MVSNet/mvsnet/colmap2mvsnet.py](https://github.com/YoYo000/MVSNet/blob/master/mvsnet/colmap2mvsnet.py).

### 🛠️ Settings
All configs for the sebsequent neural rendering pipeline are stored in ```configs/parameter.py```. Make sure to properly set the following parameters before running the code:

- Root configs (`root_params`):
    - `exp_id`: The ID for the current run.
    - `root_file`: The root path to store the training logs, checkpoints, and also the render results.
    - `load_checkpoint_dir`: The absolute path to load the specified ckpt for inference or further training. Set as `None` when training from scratch.

- Model hyperparameters (`network_params`): 
    - The default values have been optimized for most real-world large scenes.

- Batch sizes (`cluster_params`):
    - `random_view_batch_size`: How many views to be sampled at once during training.
    - `training_batch_size`: How many rays to be sampled at once for each view during training. Depending on the memory, decrease it if the available memory is less than 24G.
    - `infer_batch_size`: How many rays to be sampled at once for rendering a single image. Depending on the memory, decrease it if the available memory is less than 24G, or increase it to #rays for the desired render resolution (e.g., 2073600=1080*1920 for 1080p rendering) when having enough memory.
    - Other parameters are related to the dynamic ray loading mechanism for training and have been optimized for the best results.

- Rasterization-related parameters (`render_params`):
    - The default values can always be fixed for good results.

- Data loading configs (`load_params`):
    - `datasetFolder`: Set as the root data path.
    - `modelName`: Set as the SCENE_NAME. The folder `images_{dsp_factor}` and `cams_{dsp_factor}`, and the mesh `1.obj` should be saved in `datasetFolder/modelName/..`
    - `all_view_list`: Specify the list of view_id (from 0 to the total number of the images / cameras) to be included from `images_{dsp_factor}` and `cams_{dsp_factor}`.
    - `test_view_list`: Specify the list of view_id to be held out for testing.

### ⬇ Ray caching
This functionality is developed to enable training on high-resolution (e.g., 8K) images, by pre-caching the sliced rasterization buffers in disk. Run by:
```bash
bash graphs/warping/warp.sh
```

### 🚀 Training
The optimization is done by iteratively sampling a random batch of cached rays and performing stochastic gradient descent with L1 photometric loss. Use the following script to start training:
```bash
bash agents/adap.sh
```

### 🖥️ Rendering
After training, render the test views by:
```bash
python -u agents/render.py
```
The current release only supports inference on the specified test views. Scripts for free-viewpoint rendering will be integrated soon.

## 📃 TODO list
- Add free-viewpoint rendering demo scripts, supported by Blender.
- Integration into [nerf-studio](https://github.com/nerfstudio-project/nerfstudio).
- Release of the GigaNVS Dataset.

## 🎓 Citation

Please cite our paper:

```bibtex

```