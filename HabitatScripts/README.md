# HabitatScripts

## Installation

First install [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab). 

Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt; 
python setup.py install --headless
```

Installing habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; 
pip install -e .
```

Install pytorch from https://pytorch.org/ according to your system configuration. The code is tested on pytorch v1.6.0. If you are using conda:
```
conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch 
```

Install Detectron2 MaskRCNN from https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md.
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```

## Data

Download Replica Dataset from https://github.com/facebookresearch/Replica-Dataset

Place the `HabitatScripts` Directory inside `habitat-lab` folder. 

## File Descriptions

`sim_automated_selfsup_randspawn.py`: Loads replica in habitat and generated multiview data for objects in replica while keeping the camera fixated at the center of the object. This script estimates the fixation point using the maskrcnn detections and depth information. This script is used in weakly-supervised experiments of the paper

`sim_automated_cirle.py`: Loads replica in habitat and generated multiview data for objects in replica while keeping the camera fixated at the center of the object. This script assumes access to one ground truth fixation point which is used in weakly-supervised experiments of the paper

## Generating Data

After installing all the dependencies described above, execute:

`python sim_automated_selfsup_randspawn.py` to run self-supervised data generation.

`python sim_automated_circle.py`: to run weakly-supervised data generation.
