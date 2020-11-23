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
import scipy.misc
import imageio
import utils.improc
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import alphashape

# Settings
dataset_dir = '/projects/katefgroup/viewpredseg/carla_supervised_val'
new_dataset_dir = '/projects/katefgroup/viewpredseg/carla_supervised_val_processed'

if not os.path.exists(new_dataset_dir):
    os.makedirs(new_dataset_dir)

write_data_to_disk = True

# Loop through the dataset_dir
# Compute concave hull and save
dataset_dicts = []
for idx, filename in enumerate(os.listdir(dataset_dir)):
    if not filename.endswith(".npy"):
        continue
    print("Loading ", filename)
    image_name = filename.split('.')[0]
    if os.path.exists("{0}/{1}.npz".format(new_dataset_dir, image_name)):
        continue

    # Load saved dictionary, with fields:
    '''
        img: (1,3,H,W)
        mask_list: [np.array(H,W), ...]
        bbox_list: [np.array([ymin, xmin, ymax, xmax]), ...]
        catid_list: [catid, ...]
    '''
    full_file = os.path.join(dataset_dir, filename)
    info_dict = np.load(full_file, allow_pickle=True) 
    info_dict = info_dict.item()

    # Load fields
    img = torch.from_numpy(info_dict["img"]).squeeze(0)
    mask_list = info_dict["mask_list"]
    bbox_list = info_dict["bbox_list"]
    class_idx_list = info_dict["catid_list"]

    _, H, W = list(img.shape)
    # Convert bbox formatting, this is unnormalized
    # [ymin, xmin, ymax, xmax] --> [xmin, ymin, xmax, ymax]
    bbox_detectron2 = [[xmin, ymin, xmax, ymax] for [ymin, xmin, ymax, xmax] in bbox_list]

    # Image backtocolor + reshape to (H, W, 3)
    img = utils.improc.back2color(img)
    img = img.permute(1, 2, 0)
    img = img[:,:,[2,1,0]]

    # Process masks to contour point format
    mask_list_detectron2 = []
    bbox_list_detectron2 = []
    counter = -1
    for mask in mask_list:
        mask = np.array(mask).reshape(H, W, 1)
        counter += 1

        # Calculate convex/concave hull
        use_convex = False
        indices = np.where(mask!=0)
        indices = np.stack(indices[:-1], axis=1)
        if np.prod(indices.shape) == 0:
            continue
        if use_convex:
            hull = ConvexHull(indices)
            # Contour formatting
            hull_vertices = np.stack([indices[hull.vertices,1], indices[hull.vertices,0]], axis=1)
            keep_contours = [hull_vertices]
            segmentation = [hull_vertices.reshape(-1).tolist()]
        else:
            # use concave hull
            indices = indices.astype(np.float32)
            indices_min = indices.min(axis=0).reshape(1,2)
            indices_max = indices.max(axis=0).reshape(1,2)
            indices = (indices - indices_min) / (indices_max - indices_min)
            if np.isnan(indices).sum() > 0:
                continue
            alpha_shape = alphashape.alphashape(indices, 2.0)
            if alpha_shape.geom_type != "Polygon":
                continue
            xx, yy = alpha_shape.exterior.coords.xy
            xx = np.array(xx[:-1])
            yy = np.array(yy[:-1])
            vertices = np.stack([yy,xx], axis=1)
            vertices = vertices * (indices_max[:,[1,0]] - indices_min[:,[1,0]]) + indices_min[:,[1,0]]
            vertices = vertices.astype(int)
            keep_contours = [vertices]
            segmentation = [vertices.reshape(-1).tolist()]

        mask_list_detectron2.append(segmentation)
        bbox_list_detectron2.append(bbox_detectron2[counter])

    if write_data_to_disk:
        imageio.imwrite(f"{new_dataset_dir}/{image_name}.png", img.numpy())

        objs = []
        for i in range(len(mask_list_detectron2)):
            obj = {
                "bbox": bbox_list_detectron2[i],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": mask_list_detectron2[i],
                "category_id": class_idx_list[i],
                "is_crowd": 0
            }
            objs.append(obj)

        np.savez("{0}/{1}.npz".format(new_dataset_dir, image_name),
            file_name = f"{new_dataset_dir}/{image_name}.png",
            image_id = int(image_name),
            height = H, 
            width = W,
            annotations = objs)

print("Finished!")

