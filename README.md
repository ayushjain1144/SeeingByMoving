# SeeingByMoving


## Installation

Install pytorch, tensorflow, scikit-image, and opencv. These commands work on AWS:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge tensorflow moviepy scikit-image
pip install opencv-python
pip install connected-components-3d
```
To make tensorboard work with pytorch, install tensorboardX:
```
git clone https://github.com/zuoym15/tensorboardX
cd tensorboardX
python setup.py install
cd ..
```

## Quick start

Run one of the bash files, to run a particular mode of the repo. For example try,

`./carla_viewmine_go.sh`

Then, the following should happen:
- the bash file will set a temporary environment variable indicating the mode to run, e.g., `CARLA_EGO`
- `hyperparams.py` will set the default hyperaparameters;
- `exp_carla_ego.py` will overwrite the defaults; (edit this `exp` file to try different hyperparameter settings;)
- `hyperparams.py` will automatically generate a name for the experiment;
- `main.py` will create a logging directory
- `model_carla_ego.py` will call `backend/inputs.py` to load some data, then process it, then call the appropriate `nets/`, collect loss, and apply gradients.

If things go well, the code will just tell you that you are missing a dataset.

## Data

Data for `Carla` and `Replica` can be generated by following the instructions in the ReadME.md of `CarlaScripts` and `HabitatScripts` folder. 

Edit your exp file to indicate the location of the data: `dataset_location = "~/datasets"`

This should be the folder that contains the `.sh` file, `.txt` files and the folder of npzs.

Now, you should be able to retry the bash runner (`carla_viewmine_go.sh`) and see some results.

## Code

The main code is present in the `models` folder of this repository.

The code has these main parts:
- `model_*.py`: These files do most of the interesting work: they prepare the input tensors, call the networks, fires maskrcnn and propagate pseudo labels.
- `exp_*.py`: These files specify experiments settings, like what networks to run and what coefficients to use on the losses. There are more instructions on this below.
- `nets/`: These are all of the neural networks. The backbone for most 3D tasks is `feat3dnet`. 
- `archs/`: These are various 2D and 3D CNN architectures. 
- `utils/`: These files handle all the operations for which torch does not have native equivalents. Of particular interest here is `utils/geom.py` and `utils/vox.py`, for the geometry and voxel-related functions. 
- `backend/`: These files handle boring tasks like saving/loading checkpoints, and reading/batching data from the disk.


### Tensor shapes

We maintain consistent axis ordering across all tensors. In general, the ordering is `B x S x C x Z x Y x X`, where

- `B`: batch
- `S`: sequence (for temporal or multiview data)
- `C`: channels
- `Z`: depth
- `Y`: height
- `X`: width

This ordering stands even if a tensor is missing some dims. For example, plain images are `B x C x Y x X` (as is the pytorch standard).

### Axis directions

- Z: forward
- Y: down
- X: right

### Geometry conventions

We write pointclouds/tensors and transformations as follows:

- `p_a` is a point named `p` living in `a` coordinates.
- `a_T_b` is a transformation that takes points from coordinate system `b` to coordinate system `a`.

For example, `p_a = a_T_b * p_b`.

This convention lets us easily keep track of valid transformations, such as
`point_a = a_T_b * b_T_c * c_T_d * point_d`.

For example, an intrinsics matrix is `pix_T_cam`. An extrinsics matrix is `cam_T_world`. 

In this project's context, we often need something like this:
`xyz_cam0 = cam0_T_cam1 * cam1_T_velodyne * xyz_velodyne`

### Experiments

Experiment settings are defined hierarchically. Here is the hierarchy:

- experiments 
    - an experiment is a list of groups
- groups
    - a group is a list of hyperparameter settings
- hyperparameter settings
    - a hyperparameter setting is, for example, `occ_smooth_coeff = 0.1`
- mods
    - a mod marks a temporary change in the code. For example, if you change the code so that it only trains on frame15, you might write `mod = "frame15only"`. A mod should not last long in the code; it should either be undone or upgraded to a hyperparameter.

Experiments and groups are defined in `exp_whatever.py`. The `whatever` depends on the mode. Hyperparameters (and their default settings) are defined in `hyperparams.py`.

#### Autonaming

The names of directories for checkpoints and logs are generated automatically, based on the current hyperparameters and the current mod. For example, an automatically generated name looks like this:

`02_s2_m128x32x128_1e-4_F3_d32_G_2x11x2x1x2_r4_t1_d1_taqs100i2t_eg20`

To see how the names are generated (and to learn the shorthand for decoding them), see the bottom half of `hyperparams.py`. This particular name indicates: batch size 2, sequence length 2, resolution `128x32x128`, learning rate 0.0001, `feat3dnet` with feature dim 32, egonet with 2 scales and 11 rotations and a `2x1x2` voxel search region across 4 degrees, with a coefficient of 1.0 on each of its losses, running on the dataset `taqs100i2t`, with mod `eg20` (which should be defined manually in the exp file). 

#### Designing and running experiments

To run an experiment that has already been defined:

1. In `exp_whatever.py`, set the `current` variable to choose the current experiment. For example, set `current = 'emb_trainer'`.
2. If you want to mark something special about the experiment, set the `mod` variable, such as `mod = "special"'`
3. Execute the runner `whatever_go.sh`

To define a new experiment, either edit an existing group in `exp_whatever.py`, or create a new group. Inside the group, set all the hyperparameters you want.
