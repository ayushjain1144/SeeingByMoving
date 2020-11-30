# python pascalvoc.py -gt ../gt_pred/ -det ../maskrcnn_pred/ -gtformat 'xyrb' -detformat 'xyrb'
# train, val
# pseudo labal training, pseudo label val, gt train, validation
# finetuning 
# https://bitbucket.org/adamharley/pytorch_disco/src/72694058ba53188f1d259881c0e93e098bee8ab8/utils/finetune_detectron2_carla.py#lines-46:48
# eval 
# python pascalvoc.py -gt ../gt_pred/ -det ../maskrcnn_pred/ -gtformat 'xyrb' -detformat 'xyrb'
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import time
import detectron2
import ipdb
st = ipdb.set_trace

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from model_base import Model
from nets.crfnet import CrfNet, CrfflatNet
from backend import saverloader, inputs

from tensorboardX import SummaryWriter
import torch.nn.functional as F

import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc
import utils.track
import utils.box
import os
import scipy.ndimage
import torchvision
import cc3d
import cv2
import matplotlib
import pathlib
from scipy.io import savemat
from scipy import stats
from PIL import Image

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10
make_dataset = False
do_map_eval = False
do_visualize = False
# DATASET_PATH =  '/home/gsarch/datasets/processed/gt/replica_selfsup_gt_val'

make_pointnet_dataset = True

DATASET_PATH_POINTNET_BASE = "/home/gsarch/ayush/frustum_replica_gt_ayush/frustum_pointnets_pytorch" # already exists
DATASET_PATH_KITTI_BASE = '/home/gsarch/ayush/frustum_replica_gt_ayush/frustum_pointnets_pytorch/dataset/KITTI' 
DATASET_PATH_IMAGESETS = os.path.join(DATASET_PATH_KITTI_BASE, 'ImageSets')
DATASET_PATH_OBJECTS = os.path.join(DATASET_PATH_KITTI_BASE, 'object')
DATASET_PATH_TRAIN = os.path.join(DATASET_PATH_OBJECTS, 'train') # change to testing too
DATASET_PATH_KITTI_IMAGES = os.path.join(DATASET_PATH_TRAIN, 'image_2')
DATASET_PATH_KITTI_VELODYNE = os.path.join(DATASET_PATH_TRAIN, 'velodyne')
DATASET_PATH_KITTI_LABELS = os.path.join(DATASET_PATH_TRAIN, 'label_2')
DATASET_PATH_KITTI_CALIBS = os.path.join(DATASET_PATH_TRAIN, 'calib')

DATASET_PATH_SMALL_KITTI = "/home/gsarch/ayush/frustum_replica_gt_ayush/frustum_pointnets_pytorch/kitti"
DATASTET_PATH_IMAGE_SETS = os.path.join(DATASET_PATH_SMALL_KITTI, 'image_sets')
DATASET_PATH_RGB_DETECTIONS = os.path.join(DATASET_PATH_SMALL_KITTI, 'rgb_detections')

class CARLA_GT(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaGTModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)

class CarlaGTModel(nn.Module):
    def __init__(self):
        super(CarlaGTModel, self).__init__()
            
        # self.crop = (18,18,18)
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)

        if hyp.do_geodesic:
            self.geodesic3Dnet = Geodesic3DNet()

        if hyp.do_crf:
            self.crfnet = CrfNet()
            self.crfflatnet = CrfflatNet()

        # Initialize maskRCNN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg = cfg
        self.maskrcnn = DefaultPredictor(cfg)

        # Initialize vgg
        vgg16 = torchvision.models.vgg16(pretrained=True).double().cuda()
        vgg16.eval()
        self.vgg_feat_extractor = torch.nn.Sequential(*list(vgg16.features.children())[:1])
        self.vgg_mean = torch.from_numpy(np.array([0.485,0.456,0.406]).reshape(1,3,1,1))
        self.vgg_std = torch.from_numpy(np.array([0.229,0.224,0.225]).reshape(1,3,1,1))

        # if not hyp.shuffle_train:
        #     self.q = 0

        self.obj_counts = 0
        self.avg_iou = 0
        self.img_count = 0
        self.idx_count = 0

    def prepare_common_tensors_old(self, feed):
        # preparing tensorboard
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']
        if feed['data_ind'] < 487:   # iter where it fails
            return False

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        #initialisation
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.H4, self.W4 = int(self.H/4), int(self.W/4)
        self.PH, self.PW = hyp.PH, hyp.PW

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        elif self.set_name=='val':
            self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams_raw"].float()
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]

        self.camRs_T_origin = feed["camR_T_origin_raw"].float()
        self.origin_T_camRs = __u(utils.geom.safe_inverse(__p(self.camRs_T_origin)))
        self.origin_T_camXs = feed["origin_T_camXs_raw"].float()

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))

        self.camXs_T_origin = __u(
            utils.basic.matmul2(__p(self.camXs_T_camRs), __p(self.camRs_T_origin)))

        self.xyz_camXs = feed["xyz_camXs_raw"].float()
        # st()
        # for s in range(hyp.S):
        #     proj = utils.geom.Camera2Pixels(self.xyz_camXs[0, s].unsqueeze(0), self.pix_T_cams[0, s].unsqueeze(0))

        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        self.rgb_camXs = feed['rgb_camXs'][:,:,[0,1,2],:,:].float()
        #self.rgb_camXs = feed['rgb_camXs'].float()
        # self.feat_camXs = []
        if do_visualize:
            self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_camXs.unbind(1))

        # Filter only the five categories we care about
        '''
        class mapping between replica and maskRCNN
        class-name      replica ID      maskRCNN ID
        chair           20              56
        bed             7               59
        dining table    80              60
        toilet          84              61
        couch           76              57
        potted plant    44              58
        refrigerator    67              72
        tv(tv-screen)   87              62
        '''
        self.maskrcnn_to_catname = {56: "chair", 59: "bed", 61: "toilet", 57: "couch", 58: "indoor-plant", 
                            72: "refrigerator", 62: "tv", 60: "dining-table"}
        self.replica_to_maskrcnn = {20: 56, 7: 59, 84: 61, 76: 57, 44: 58, 67: 72, 87: 62, 80: 60}

        self.category_ids_camXs = feed['category_ids_camXs']
        self.object_category_names = feed['category_names_camXs']
        self.bbox_2d_camXs = feed['bbox_2d_camXs']
        self.mask_2d_camXs = feed['mask_2d_camXs']

        has_obj = False
        for s in list(range(self.S)):
            if len(self.bbox_2d_camXs[s]) > 0:
                has_obj = True
                break
        if not has_obj:
            return False

        self.bbox_2d_camXs = [torch.cat(bbox_2d_camX_i, dim=0) if len(bbox_2d_camX_i)>0 else [] for bbox_2d_camX_i in self.bbox_2d_camXs] # list of length S, each item = (N_obj,4)
        # self.object_bboxes_origin = torch.cat(self.object_bboxes_origin, dim=0) #(N_obj, 8, 3)

        
        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))

        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_)
        # st()
        if do_visualize:
            self.summ_writer.summ_oned('gt_depth/view_depth_camX0', self.depth_camXs[:,0]*self.valid_camXs[:,0], maxval=32.0)
            self.summ_writer.summ_oned('gt_depth/view_valid_camX0', self.valid_camXs[:,0], norm=False)
        self.dense_xyz_camXs = __u(self.dense_xyz_camXs_)

        self.dense_xyz_camX0 = self.dense_xyz_camXs[:,0]
        self.dense_xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.dense_xyz_camXs)))
        self.dense_xyz_camX0s_mult = self.dense_xyz_camX0s.reshape(self.B, self.S*self.dense_xyz_camX0s.shape[2], 3)

        self.boxlist2d_camXs = []
        self.masklist_camXs = []
        self.masklist_camXs_safe = []
        self.any_mask_list_camXs = []
        self.obj_id_list_camXs = []
        self.obj_catid_list_camXs = []
        self.obj_score_list_camXs = []

        self.obj_all_catid_list_camXs = []
        self.obj_all_score_list_camXs = []
        self.obj_all_box_list_camXs = []

        #Loop through all the views to get all the masks for all objects
        for s in list(range(self.S)):
            im = self.rgb_camXs[:, s]
            im = utils.improc.back2color(im)
            img_torch = im.clone().detach()

            # Run vgg
            '''
            img_torch = (img_torch - self.vgg_mean) / self.vgg_std
            with torch.no_grad():
                img_feat = self.vgg_feat_extractor(img_torch.cuda()) # B*C*H*W
            self.feat_camXs.append(img_feat.unsqueeze(0))
            '''

            # Run maskRCNN
            im = im[0]
            im = im.permute(1, 2, 0)
            im = im.detach().cpu().numpy()
            im = im[:, :, ::-1]
            outputs = self.maskrcnn(im)

            pred_masks = outputs['instances'].pred_masks
            pred_boxes = outputs['instances'].pred_boxes.tensor
            pred_classes = outputs['instances'].pred_classes
            pred_scores = outputs['instances'].scores
            
            # converts instance segmentation to individual masks and bbox
            # visualisations
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
            seg_im = out.get_image()
            if do_visualize:
                self.summ_writer.summ_rgb('maskrcnn_res/view{}_instance_mask'.format(s), torch.from_numpy(seg_im).permute(2, 0, 1).unsqueeze(0))

            # get just objects/"things" - theres prob an easier way to do this
            obj_ids = []
            obj_catids = []
            obj_scores = []
            obj_all_catids = []
            obj_all_scores = []
            obj_all_boxes = []

            for segs in range(len(pred_masks)):
                # old carla, keep only cars (remove bikes): if pred_classes[segs] > 1 and pred_classes[segs] <= 8 and pred_classes[segs] != 3:
                if pred_classes[segs].item() in self.maskrcnn_to_catname:
                    if pred_scores[segs] >= 0.90:
                        print(self.maskrcnn_to_catname[pred_classes[segs].item()], pred_scores[segs].item())
                        obj_ids.append(segs)
                        obj_catids.append(pred_classes[segs].item())
                        obj_scores.append(pred_scores[segs].item())

                    obj_all_catids.append(pred_classes[segs].item())
                    obj_all_scores.append(pred_scores[segs].item())
                    y, x = torch.where(pred_masks[segs])
                    pred_box = torch.Tensor([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                    obj_all_boxes.append(pred_box)

            self.obj_id_list_camXs.append(obj_ids)
            self.obj_catid_list_camXs.append(obj_catids)
            self.obj_score_list_camXs.append(obj_scores)

            self.obj_all_catid_list_camXs.append(obj_all_catids)
            self.obj_all_score_list_camXs.append(obj_all_scores)
            self.obj_all_box_list_camXs.append(obj_all_boxes)

            N, H, W = pred_masks.shape
            objs_anymask = torch.zeros((H,W))
            masklist = pred_masks.reshape(1, N, 1, self.H, self.W).float()
            self.masklist_camXs.append(masklist)
            weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
            for id in obj_ids:
                obj_mask = masklist[:,id]
                obj_mask = 1.0 - F.conv2d(1.0 - obj_mask, weights, padding=1).clamp(0, 1)
                #obj_mask = 1.0 - F.conv2d(1.0 - obj_mask, weights, padding=1).clamp(0, 1)
                obj_mask[obj_mask > 0] = 1
                masklist[:,id] = obj_mask
                objs_anymask[obj_mask[0,0]==1] = 1

            self.any_mask_list_camXs.append(objs_anymask.unsqueeze(0).unsqueeze(0))
            self.masklist_camXs_safe.append(masklist)

        # Andy: comment out the features for now
        # self.feat_camXs = torch.cat(self.feat_camXs, 1).float()

        for s in list(range(self.S)):
            if len(self.obj_id_list_camXs[s]) > 0:
                return True

        # return false if we found no objecs in all views
        print("No objects found....returning")
        return False

    def prepare_common_tensors(self, feed):
        # preparing tensorboard
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']
        if feed['data_ind'] < 0:
            return False

        self.B = feed["set_batch_size"]
        self.S = feed["set_seqlen"]
        self.set_name = feed['set_name']

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        #initialisation
        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.H4, self.W4 = int(self.H/4), int(self.W/4)
        self.PH, self.PW = hyp.PH, hyp.PW

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        elif self.set_name=='val':
            self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z1, self.Y1, self.X1 = int(self.Z/1), int(self.Y/1), int(self.X/1)
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed["pix_T_cams_raw"].float()
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]

        self.camRs_T_origin = feed["camR_T_origin_raw"].float()
        self.origin_T_camRs = __u(utils.geom.safe_inverse(__p(self.camRs_T_origin)))
        self.origin_T_camXs = feed["origin_T_camXs_raw"].float()

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))

        self.camXs_T_origin = __u(
            utils.basic.matmul2(__p(self.camXs_T_camRs), __p(self.camRs_T_origin)))

        self.xyz_camXs = feed["xyz_camXs_raw"].float()
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        self.rgb_camXs = feed['rgb_camXs'][:,:,[0,1,2],:,:].float()
        #self.rgb_camXs = feed['rgb_camXs'].float()
        # self.feat_camXs = []
        if do_visualize:
            self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_camXs.unbind(1))

        # Filter only the five categories we care about
        '''
        class mapping between replica and maskRCNN
        class-name      replica ID      maskRCNN ID
        chair           20              56
        bed             7               59
        dining table    80              60
        toilet          84              61
        couch           76              57
        potted plant    44              58
        refrigerator    67              72
        tv(tv-screen)   87              62
        '''
        self.maskrcnn_to_catname = {56: "chair", 59: "bed", 61: "toilet", 57: "couch", 58: "indoor-plant", 
                            72: "refrigerator", 62: "tv", 60: "dining-table"}
        self.replica_to_maskrcnn = {20: 56, 7: 59, 84: 61, 76: 57, 44: 58, 67: 72, 87: 62, 80: 60}

        self.category_ids_camXs = feed['category_ids_camXs']
        self.object_category_names = feed['category_names_camXs']
        self.bbox_2d_camXs = feed['bbox_2d_camXs']
        self.mask_2d_camXs = feed['mask_2d_camXs']
        self.box_3d_camXs = feed['bbox3d_camXs']

        has_obj = False
        for s in list(range(self.S)):
            if len(self.bbox_2d_camXs[s]) > 0:
                has_obj = True
                break
        if not has_obj:
            return False

        self.bbox_2d_camXs = [torch.cat(bbox_2d_camX_i, dim=0) if len(bbox_2d_camX_i)>0 else [] for bbox_2d_camX_i in self.bbox_2d_camXs] # list of length S, each item = (N_obj,4)

        # return false if we found no objecs in all views
        return True

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        # List of all objects' projections (dictionary, key=(scene,catid), value=list of list [[score1, mask1, bbox1], [score2, mask2, bbox2]])
        all_info_list = {}

        # List of classlist_g, boxlist_g
        self.classlist_g_s = []
        self.boxlist_g_s = []
        self.masklist_g_s = []
        self.boxlist_3d_g_s = []

        # loop through all views and all objects
        for s in list(range(self.S)):
            ####### Procuring GT 2D BOXS #########
            
            if torch.is_tensor(self.bbox_2d_camXs[s]):
                obj_bboxes_y_min, obj_bboxes_x_min, obj_bboxes_y_max, obj_bboxes_x_max = torch.unbind(self.bbox_2d_camXs[s], axis=1)
            else:
                self.classlist_g_s.append([])
                self.boxlist_g_s.append([])
                self.masklist_g_s.append([])
                print("boxlist_g empty....continuing")
                continue

            classlist_g = []
            boxlist_g = []
            boxlist_g_norm = []
            masklist_g = []
            boxlist_3d_g = []
            # st()
            for i in range(self.bbox_2d_camXs[s].shape[0]):
                ymin, xmin, ymax, xmax = obj_bboxes_y_min[i], obj_bboxes_x_min[i], obj_bboxes_y_max[i], obj_bboxes_x_max[i]
                if ymin == ymax or xmin == xmax or xmin>self.W or xmax<0 or ymin>self.H or ymax<0:
                    # remove empty boxes or boxes out of the current view
                    continue
                classlist_g.append(self.replica_to_maskrcnn[self.category_ids_camXs[s][i].item()])
                box = np.array([ymin, xmin, ymax, xmax])
                boxlist_g.append(box)
                # classlist_g.append(self.mask_2d[i])
                box_norm = np.array([ymin/self.H, xmin/self.W, ymax/self.H, xmax/self.W])
                boxlist_g_norm.append(box_norm)
                masklist_g.append(self.mask_2d_camXs[s][i])

                # append 3d box too
                corners_origin = self.box_3d_camXs[s][i].reshape(2, 3)
                vertices_origin = torch.stack([corners_origin[[1,1,1],[0,1,2]],
                    corners_origin[[1,1,0],[0,1,2]],
                    corners_origin[[0,1,0],[0,1,2]],
                    corners_origin[[0,1,1],[0,1,2]],
                    corners_origin[[1,0,1],[0,1,2]],
                    corners_origin[[1,0,0],[0,1,2]],
                    corners_origin[[0,0,0],[0,1,2]],
                    corners_origin[[0,0,1],[0,1,2]]]).unsqueeze(0)
                # st()
                vertices_origin = vertices_origin #- torch.Tensor([0, 1.5, 0]).reshape(1, 1, 3)
                vertices_camXs = utils.geom.apply_4x4_to_corners(self.camXs_T_origin[0, s].cuda(), vertices_origin.unsqueeze(0).cuda()).squeeze(0)

                box3d = vertices_camXs.squeeze(0).cpu().numpy()
                boxlist_3d_g.append(box3d)

            self.classlist_g_s.append(classlist_g)
            self.boxlist_g_s.append(boxlist_g)
            self.masklist_g_s.append(masklist_g)
            self.boxlist_3d_g_s.append(boxlist_3d_g)
            # st()
            if len(boxlist_g) == 0:
                print("boxlist_g empty....continuing")
                continue

            ###### Visualize input #########
            if do_visualize:
                self.summ_writer.summ_soft_seg('inputs/view{}_seg_camX0'.format(s), F.softmax(self.masklist_camXs[s].squeeze(2), dim=1))
                self.summ_writer.summ_rgb('inputs/view{}_rgb_camX0'.format(s), self.rgb_camXs[:,s])
                self.summ_writer.summ_oned('inputs/view{}_depth_camX0'.format(s), self.depth_camXs[:,s]*self.valid_camXs[:,s], maxval=32.0)
                self.summ_writer.summ_oned('inputs/view{}_valid_camX0'.format(s), self.valid_camXs[:,s], norm=False)

            # st()
            boxlist_g_norm = torch.from_numpy(np.array(boxlist_g_norm)).unsqueeze(0).clamp(0,1)
            if do_visualize:
                self.summ_writer.summ_boxlist2d('inputs/view{}_boxes2d'.format(s), self.rgb_camXs[:,s], boxlist_g_norm)

            # all_info_list: List of all objects' projections (dictionary, key=(scene,catid), value=list of list [[score1, mask1, bbox1], [score2, mask2, bbox2]])
            # For each view
        
            # Image
            img = self.rgb_camXs[:, s].cuda() # (1,3,H,W)

            # Write data
            keep_list = []

            ################ MAP evaluation #####################
            # if do_map_eval:
            # setup gt
            boxlist_g = self.boxlist_g_s[s]
            classlist_g = self.classlist_g_s[s]
            masklist_g = self.masklist_g_s[s]

            ################ save predictions to dataset ################
            # --> stored in /projects/katefgroup/viewpredseg_dataset

            
            if make_dataset:
                # st()
                if not os.path.exists(DATASET_PATH):
                    os.makedirs(DATASET_PATH)
                data_dict = {"img": img.cpu().numpy(),  
                                "mask_list": masklist_g,
                                "bbox_list": boxlist_g,
                                "catid_list": classlist_g}
                np.save(f'{DATASET_PATH}/{self.img_count}.npy', data_dict)
            
            # Increment image count
            self.img_count += 1

            ############# save pointnet++ dataset #########################
                    
            if make_pointnet_dataset:
                
                self.if_not_exists_makeit(DATASET_PATH_KITTI_BASE)
                self.if_not_exists_makeit(DATASET_PATH_IMAGESETS)
                self.if_not_exists_makeit(DATASET_PATH_OBJECTS)
                self.if_not_exists_makeit(DATASET_PATH_TRAIN)
                self.if_not_exists_makeit(DATASET_PATH_KITTI_IMAGES)
                self.if_not_exists_makeit(DATASET_PATH_KITTI_VELODYNE)
                self.if_not_exists_makeit(DATASET_PATH_KITTI_LABELS)
                self.if_not_exists_makeit(DATASET_PATH_KITTI_CALIBS)

                self.if_not_exists_makeit(DATASET_PATH_SMALL_KITTI)
                self.if_not_exists_makeit(DATASTET_PATH_IMAGE_SETS)
                self.if_not_exists_makeit(DATASET_PATH_RGB_DETECTIONS)
                # class_list_e_pseudo = catid_list # consider all objects, add constraint if we want later
                classlist_g = self.classlist_g_s[s]

                # # write image in DATASET_PATH_KITTI_IMAGE
                im = utils.improc.back2color(img).squeeze(0).permute(1, 2, 0).cpu().numpy()
                im = Image.fromarray(im)
                im.save(os.path.join(DATASET_PATH_KITTI_IMAGES, f'{self.idx_count}.png'))
                
                # write pointcloud
                # st()

                pc_xyz = self.xyz_camXs[0, s].cpu().numpy()
                np.save(os.path.join(DATASET_PATH_KITTI_VELODYNE, f'{self.idx_count}.npy'), pc_xyz)
                
                with open(os.path.join(DATASET_PATH_KITTI_VELODYNE, f'{self.idx_count}.npy'), 'w') as f:
                    np.save(f, pc_xyz) 

                # labels
                # box3d_list = self.boxlist_3d_g_s[s]
                # boxlist_g = boxlist_g.tolist()
                with open(os.path.join(DATASET_PATH_KITTI_LABELS, f'{self.idx_count}.txt'), 'w') as f:
                    for i in range(len(classlist_g)):
                        # st()
                        type_name = self.maskrcnn_to_catname[int(classlist_g[i])]
                        ymin, xmin, ymax, xmax = boxlist_g[i]
                        box3d_str = ' '.join(str(e) for e in boxlist_3d_g[i].reshape(-1))
                        # xc, yc, zc, wid, hei, dep, rx, ry, rz = box3d_list[i][0]
                        f.write(f"{type_name} 0 3 0 {xmin} {ymin} {xmax} {ymax} {box3d_str}\n")

                # # calibs
                with open(os.path.join(DATASET_PATH_KITTI_CALIBS, f'{self.idx_count}.txt'), 'w') as f:
                    # st()
                    pix_T_cam_save = self.pix_T_cams[0, s].reshape(-1).cpu().numpy()
                    pix_T_cam_str = ' '.join(str(e) for e in pix_T_cam_save)
                    f.write(f'pix_T_cam: {pix_T_cam_str}\n')

                    camX_T_origin_save = self.camXs_T_origin[0, s].reshape(-1).cpu().numpy()
                    camX_T_origin_str = ' '.join(str(e) for e in camX_T_origin_save)
                    f.write(f'camX_T_origin: {camX_T_origin_str}\n')

                    camX_T_camR_save = self.camXs_T_camRs[0, s].reshape(-1).cpu().numpy()
                    camX_T_camR_str = ' '.join(str(e) for e in camX_T_camR_save)
                    f.write(f'camX_T_camR: {camX_T_camR_str}')

                # write image sets
                with open(os.path.join(DATASTET_PATH_IMAGE_SETS, f'train.txt'), 'a+') as f:
                    f.write(f"{self.idx_count}\n")

                with open(os.path.join(DATASET_PATH_IMAGESETS, f'train.txt'), 'a+') as f:
                    f.write(f"{self.idx_count}\n")
                

                with open(os.path.join(DATASET_PATH_RGB_DETECTIONS, f'rgb_detections_test.txt'), 'a+') as f:
                    path_to_img = str(os.path.join(DATASET_PATH_KITTI_IMAGES, f'{self.idx_count}.png'))
                    for i in range(len(classlist_g)):
                        ymin, xmin, ymax, xmax = boxlist_g[i]
                        f.write(f"{path_to_img} 2 1 {xmin} {ymin} {xmax} {ymax}\n")               
                self.idx_count += 1


            ######################### POINTNET++ DATASET ENDS #########################################

        return total_loss, results, False
        

    def run_test(self, feed):
        pass

    def if_not_exists_makeit(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def forward(self, feed):
        
        data_ok = self.prepare_common_tensors_old(feed)

        if not data_ok:
            print("No objects detected in 2D, returning early")
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
            
        else:
            self.prepare_common_tensors(feed)
            if self.set_name == 'train':
                return self.run_train(feed)
            elif self.set_name == 'test':
                return self.run_test(feed)
            else:
                print('Not implemented this set name: ', self.set_name)
                assert(False)








