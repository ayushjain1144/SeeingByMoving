# pip install pycocotools
import ipdb
st = ipdb.set_trace

import pycocotools

import torch, torchvision
from torchvision.utils import save_image
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import glob

import os
import ntpath
import numpy as np
import cv2
import random
import itertools
import urllib
import json
import PIL.Image as Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper
train_processed_dir = '/projects/katefgroup/viewpredseg/carla_supervised_train_processed'
val_processed_dir = '/projects/katefgroup/viewpredseg/carla_supervised_val_processed'
import scipy.misc
import imageio
import utils.improc
from LossEvalHook import LossEvalHook
import logging

from detectron2.data import detection_utils
import detectron2.data.transforms as T
import copy

write_data_to_disk = True


label_ratio = 0.1
train_files = glob.glob(os.path.join(train_processed_dir, '*.npz'))
train_files = train_files[:int(len(train_files)*label_ratio)]
val_files = glob.glob(os.path.join(val_processed_dir, '*.npz'))
val_files = val_files[:int(len(val_files)*label_ratio)]

'''
town5_idx = [2, 3, 4, 5, 7, 9, 20, 25, 30, 32, 38, 41, 42, 52, 54, 55, 62, 63, 68, 75, 87, 88, 92, 96, 97, 98, 104, 107, 108, 111, 119, 120, 121, 125, 126, 130, 131, 137, 139, 142, 143, 145, 148, 149, 150, 151, 156, 157, 159, 164, 165, 166, 170, 179, 187, 188, 189, 194, 195, 198, 199, 201, 204, 205, 207, 209]

train_files = []
for idx in town5_idx:
    for i in range(25):
        train_files.append("{0}/{1}.npz".format(train_processed_dir, (idx-1)*25+i))

val_files = glob.glob(os.path.join(val_processed_dir, '*.npz'))
val_files = val_files[:int(len(val_files)*label_ratio)]
'''

def train_dataset_function():

    dataset_dicts = []
    print("Loading train dataset...")

    for file in train_files:
        meta = np.load(file, allow_pickle=True)

        record = {}
        record["file_name"] = str(meta['file_name'])
        record["image_id"] = int(meta['image_id'])
        record["height"] = int(meta['height'])
        record["width"] = int(meta['width'])
        record["annotations"] = meta['annotations'].tolist()
        for i in range(len(record["annotations"])):
            record["annotations"][i]['category_id'] = 2

        dataset_dicts.append(record)

    print("Data loaded!")

    return dataset_dicts

def val_dataset_function():

    dataset_dicts = []
    print("Loading val dataset...")

    for file in val_files:
        meta = np.load(file, allow_pickle=True)

        record = {}
        record["file_name"] = str(meta['file_name'])
        record["image_id"] = int(meta['image_id'])
        record["height"] = int(meta['height'])
        record["width"] = int(meta['width'])
        record["annotations"] = meta['annotations'].tolist()
        for i in range(len(record["annotations"])):
            record["annotations"][i]['category_id'] = 2

        dataset_dicts.append(record)

    print("Data loaded!")

    return dataset_dicts

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


# Get coco dataset metadata
coco_meta = MetadataCatalog.get("coco_2017_train")

# register dataset, thing_classes same as coco thing_classes
d = "train"
DatasetCatalog.register("multiview_carla_gt_train", lambda d=d: train_dataset_function())
MetadataCatalog.get("multiview_carla_gt_train").thing_classes = coco_meta.get("thing_classes")

DatasetCatalog.register("multiview_carla_gt_val", lambda d=d: val_dataset_function())
MetadataCatalog.get("multiview_carla_gt_val").thing_classes = coco_meta.get("thing_classes")

multiview_carla_metadata = MetadataCatalog.get("multiview_carla_gt_train")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.OUTPUT_DIR = './logs_detectron/logs_carla_detectron_gt01'
cfg.DATASETS.TRAIN = ("multiview_carla_gt_train",) # add train set name
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = (
    100000
)  # we can adjust by looking at progress on tb

# Validation set (uncomment when we have validation set)
# set to train set for now
cfg.DATASETS.TEST = ("multiview_carla_gt_val",) 
cfg.TEST.EVAL_PERIOD = 5000


# visualise
dataset_dicts = train_dataset_function()
# j = 0
# for d in random.sample(dataset_dicts, 3):
#     j += 1
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=multiview_replica_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     print(os.getcwd())
#     cv2.imwrite(f'./im.png', out.get_image()[:, :, ::-1])
#     print("written image")

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    256
)  # not sure what matrix could handle 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# st()
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
        

