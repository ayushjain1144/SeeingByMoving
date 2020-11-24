# frustum_pointnets_pytorch
A pytorch version of [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 
(Not support Pointnet++ yet)

Adaptation of [frustum-pointnets-pytorch](https://github.com/simon3dv/frustum_pointnets_pytorch) for CARLA dataset

main function of f-pointnets now:
train/train_fpointnets.py, 
train/test_fpointnets.py,
train/provider_fpointnet.py,
models/frustum_pointnets_v1_old.py
model_util_old.py
kitti/*

## Requirements
Test on 
* Ubuntu-18.04
* CUDA-10.0
* Pytorch 1.3
* python 3.7


## Usage
### Installation(optional)
Some visulization demos need mayavi, it would be a little bit difficult to install it.
I install mayavi(python3) by:
(ref:http://docs.enthought.com/mayavi/mayavi/installation.html#installing-with-pip) 
```angular2
1. download vtk(https://vtk.org/download/) and compile:
unzip VTK
cd VTK
mkdir build
cd build
cmake ..
make
sudo make install 
2. install mayavi and PyQt5
pip install mayavi
pip install PyQt5
```
### Prepare Training Data
```angular2
frustum_pointnets_pytorch
├── dataset
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──train
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──val
│   │   │      ├──calib & velodyne & label_2 & image_2
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & label_2 & image_2
├── kitti
│   │   ├── image_sets
│   │   ├── rgb_detections
├── train
```
#### Carla
To visulize single sample in Kitti
```angular2
python kitti/prepare_data.py --demo
```
To prepare training data
```angular2
python kitti/prepare_data.py --gen_train --gen_val --gen_test
```
To visulize all gt boxes and prediction boxes:
```angular2
python kitti/kitti_object.py
```

## train
### CARLA
```angular2
CUDA_VISIBLE_DEVICES=0 python train/train_fpointnets.py --log_dir log
```


## Test
### CARLA
```angular2
CUDA_VISIBLE_DEVICES=0 python train/test_fpointnets.py --model_path <log/.../xx.pth> --output test_results --data_path <path to ground truth test set>
```

### Visulize
```
python kitti/kitti_object.py
```


### Acknowledgement
many codes are from:
* [frustum-pointnets-pytorch](https://github.com/simon3dv/frustum_pointnets_pytorch)
* [frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) 
* [frustum-convnet](https://github.com/zhixinwang/frustum-convnet)
* [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
* [NuScenes2KITTI](https://github.com/zcc31415926/NuScenes2KITTI)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
* [Det3D](https://github.com/poodarchu/Det3D)
