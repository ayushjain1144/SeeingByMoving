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
from nets.feat2dnet import Feat2dNet
from nets.feat3dnet import Feat3dNet
from nets.rgbnet import RgbNet
from nets.occnet import OccNet
from nets.rendernet import RenderNet
from nets.geodesic3Dnet import Geodesic3DNet
from nets.crfnet import CrfNet, CrfflatNet
from nets.box3dnet import Box3dNet
from backend import saverloader, inputs

import archs.pixelshuffle3d

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
do_map_eval = True
do_visualize = True
make_pointnet_dataset = False
DATASET_PATH =  '/projects/katefgroup/viewpredseg/carla_self_supervised_val'

DATASET_PATH_POINTNET_BASE = "/home/ayushj2/frustum_pointnets_pytorch" # already exists
DATASET_PATH_KITTI_BASE = '/home/ayushj2/frustum_pointnets_pytorch/dataset/KITTI' 
DATASET_PATH_IMAGESETS = os.path.join(DATASET_PATH_KITTI_BASE, 'ImageSets')
DATASET_PATH_OBJECTS = os.path.join(DATASET_PATH_KITTI_BASE, 'object')
DATASET_PATH_TRAIN = os.path.join(DATASET_PATH_OBJECTS, 'training') # change to testing too
DATASET_PATH_KITTI_IMAGES = os.path.join(DATASET_PATH_TRAIN, 'image_2')
DATASET_PATH_KITTI_VELODYNE = os.path.join(DATASET_PATH_TRAIN, 'velodyne')
DATASET_PATH_KITTI_LABELS = os.path.join(DATASET_PATH_TRAIN, 'label_2')
DATASET_PATH_KITTI_CALIBS = os.path.join(DATASET_PATH_TRAIN, 'calib')

DATASET_PATH_SMALL_KITTI = "/home/ayushj2/frustum_pointnets_pytorch/kitti"
DATASTET_PATH_IMAGE_SETS = os.path.join(DATASET_PATH_SMALL_KITTI, 'image_sets')
DATASET_PATH_RGB_DETECTIONS = os.path.join(DATASET_PATH_SMALL_KITTI, 'rgb_detections')


class CARLA_VIEWMINE(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaViewmineModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)

class CarlaViewmineModel(nn.Module):
    def __init__(self):
        super(CarlaViewmineModel, self).__init__()
            
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

        if not hyp.shuffle_train:
            self.q = 0

        self.obj_counts = 0
        self.avg_iou = 0
        self.img_count = 0
        self.idx_count = 0

    def prepare_common_tensors(self, feed):
        # preparing tensorboard
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']
        # st()
        if feed['data_ind'] < 149:   # iter where it fails
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
        self.pix_T_cams = feed["pix_T_cams"]
        # st()
        set_data_format = feed['set_data_format']
        self.S = feed["set_seqlen"]

        self.origin_T_camRs = feed["origin_T_camRs"]
        self.origin_T_camXs = feed["origin_T_camXs"]
        self.camXs_T_origin = utils.geom.safe_inverse(__p(self.origin_T_camXs))
        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))

        self.xyz_camXs = feed["xyz_camXs"]
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        self.rgb_camXs = feed['rgb_camXs']
        # self.feat_camXs = []

        if do_visualize:
            self.summ_writer.summ_rgbs('inputs/rgbs', self.rgb_camXs.unbind(1))
        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))

        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_)
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
                self.summ_writer.summ_rgb('input_rgb/view{}_instance_mask'.format(s), torch.from_numpy(seg_im).permute(2, 0, 1).unsqueeze(0))

            # get just objects/"things" - theres prob an easier way to do this
            seg_info_list = []
            all_info_list = []
            for segs in range(len(pred_masks)):
                # 1 and 3 are bikes. removing them
                if pred_classes[segs] > 1 and pred_classes[segs] <= 8 and pred_classes[segs] != 3:
                    y, x = torch.where(pred_masks[segs])
                    if len(y) == 0:
                        continue
                    pred_box = torch.Tensor([min(y), min(x), max(y), max(x)])
                    all_info_list.append([pred_classes[segs].item(), pred_scores[segs].item(), pred_box])

                    if pred_scores[segs] >= 0.90:
                        seg_info_list.append([segs, pred_classes[segs].item(), pred_scores[segs].item(), pred_box])

            obj_ids = []
            obj_catids = []
            obj_scores = []
            rem_info_list = sorted(seg_info_list, reverse=True, key=lambda x: x[2]) 
            while len(rem_info_list) > 0:
                obj_ids.append(rem_info_list[0][0])
                obj_catids.append(rem_info_list[0][1])
                obj_scores.append(rem_info_list[0][2])
                
                ymin1, xmin1, ymax1, xmax1 = rem_info_list[0][-1].cpu().numpy()
                area_conf = (ymax1 - ymin1) * (xmax1 - xmin1)

                rem_info_list_new = []

                for rrr in range(1, len(rem_info_list)):
                    ymin2, xmin2, ymax2, xmax2 = rem_info_list[rrr][-1].cpu().numpy()
                    area_cur = (ymax2 - ymin2) * (xmax2 - xmin2)
                    if not area_cur > 0:
                        continue

                    x_dist = (min(xmax1, xmax2) - max(xmin1, xmin2))
                    y_dist = (min(ymax1, ymax2) - max(ymin1, ymin2))

                    if x_dist > 0 and y_dist > 0:
                        area_overlap = x_dist * y_dist
                        if float(area_overlap) /float(area_conf + area_cur - area_overlap) > 0.5:
                            continue

                    rem_info_list_new.append(rem_info_list[rrr])

                rem_info_list = rem_info_list_new

            self.obj_id_list_camXs.append(obj_ids)
            self.obj_catid_list_camXs.append(obj_catids)
            self.obj_score_list_camXs.append(obj_scores)

            obj_all_catids = []
            obj_all_scores = []
            obj_all_boxes = []
            rem_info_list = sorted(all_info_list, reverse=True, key=lambda x: x[1])
            while len(rem_info_list) > 0:
                obj_all_catids.append(rem_info_list[0][0])
                obj_all_scores.append(rem_info_list[0][1])
                obj_all_boxes.append(rem_info_list[0][2])

                ymin1, xmin1, ymax1, xmax1 = rem_info_list[0][-1].cpu().numpy()
                area_conf = (ymax1 - ymin1) * (xmax1 - xmin1)

                rem_info_list_new = []

                for rrr in range(1, len(rem_info_list)):
                    ymin2, xmin2, ymax2, xmax2 = rem_info_list[rrr][-1].cpu().numpy()
                    area_cur = (ymax2 - ymin2) * (xmax2 - xmin2)
                    if not area_cur > 0:
                        continue

                    x_dist = (min(xmax1, xmax2) - max(xmin1, xmin2))
                    y_dist = (min(ymax1, ymax2) - max(ymin1, ymin2))

                    if x_dist > 0 and y_dist > 0:
                        area_overlap = x_dist * y_dist
                        if float(area_overlap) /float(area_conf + area_cur - area_overlap) > 0.5:
                            continue

                    rem_info_list_new.append(rem_info_list[rrr])

                rem_info_list = rem_info_list_new


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
                obj_mask = 1.0 - F.conv2d(1.0 - obj_mask, weights, padding=1).clamp(0, 1)
                obj_mask[obj_mask > 0] = 1
                masklist[:,id] = obj_mask
                objs_anymask[obj_mask[0,0]==1] = id

            self.any_mask_list_camXs.append(objs_anymask.unsqueeze(0).unsqueeze(0))
            self.masklist_camXs_safe.append(masklist)

        # Andy: comment out the features for now
        # self.feat_camXs = torch.cat(self.feat_camXs, 1).float()

        for s in list(range(self.S)):
            if len(self.obj_id_list_camXs[s]) > 0:
                return True

        # return false if we found no objecs in all views
        return False

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        # get full occupancy from full view
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 #hyp.YMIN
        scene_centroid_z = 18.0
        # for cater the table is y=0, so we move upward a bit
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()                                                                 
        self.vox_util = utils.vox.Vox_util(
            self.Z, self.Y, self.X, 
            self.set_name, scene_centroid=self.scene_centroid,
            assert_cube=True)

        #dense multiview pointcloud in frame of camera 0
        occ_memX0s_dense_scene = self.vox_util.voxelize_xyz(self.dense_xyz_camX0s_mult, self.Z, self.Y, self.X, assert_cube=False)
        if do_visualize:
            self.summ_writer.summ_occ('inputs/occ_memX0s_dense_scene', occ_memX0s_dense_scene)

        # unproject rgb to memory tensor at all views and convert to ref coord frame
        self.pixX_T_camX0s = __u(
            utils.basic.matmul2(__p(self.pix_T_cams), __p(self.camXs_T_camX0s)))

        # Get ground truth 3D boxes
        self.lrtlist_camRs = feed['lrtlist_camRs']
        self.full_tidlist_s = feed["tidlist_s"]
        self.full_scorelist_s = feed["scorelist_s"]
        self.lrtlist_camXs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camX0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        # List of all objects' projections (dictionary, key=(scene,catid), value=list of list [[score1, mask1, bbox1], [score2, mask2, bbox2]])
        all_info_list = {}

        # List of classlist_g, boxlist_g
        self.classlist_g_s = []
        self.boxlist_g_s = []
        self.box3dlist_g_s = []   # h, w, l, x, y, z, 0
        # loop through all views and all objects
        for s in list(range(self.S)):
            ####### Procuring GT 2D BOXS #########
            # Convert corners and centers to xyz 
            xyzlist_camXs = utils.geom.get_xyzlist_from_lrtlist(self.lrtlist_camXs[:,s])
            B2, N_lrt, D, E = list(xyzlist_camXs.shape)
            num_gt_obj = torch.sum(self.full_scorelist_s[:, s])

            scorelist_rescore = utils.misc.rescore_lrtlist_with_pointcloud(
                self.lrtlist_camRs[:, s], self.xyz_camRs[:, s], self.full_scorelist_s[:, s], thresh=2.0)
            self.full_scorelist_s[:, s] = self.full_scorelist_s[:, s] * scorelist_rescore

            # get box list
            corners_pix = utils.geom.get_boxlist2d_from_lrtlist(self.pix_T_cams[:,s], self.lrtlist_camXs[:, s], self.H, self.W)
            corners_pix = self.full_scorelist_s[:, s].unsqueeze(2) * corners_pix

            # deriving class
            lenlist_cam, _ = utils.geom.split_lrtlist(self.lrtlist_camX0s[:, s])
            classlist_g = []
            boxlist_g = []
            for i in range(N_lrt):
                
                if self.full_scorelist_s[:, s, i] > 0:
                    # st()    
                    lx, ly, lz = (torch.unbind(lenlist_cam, dim=1))[i][0]
                    if lx > 1.0:
                        classlist_g.append(1) #"car"
                        boxlist_g.append(corners_pix[0,i,:].cpu().numpy())

                    else:
                        classlist_g.append(0) #"bike"
                        # don;t append bikes in gt
                        # boxlist_g.append(corners_pix[0,i,:].cpu().numpy())

            self.classlist_g_s.append(classlist_g)
            self.boxlist_g_s.append(boxlist_g)

            ####### Visualize input #########
            if do_visualize:
                self.summ_writer.summ_soft_seg('inputs/view{}_seg_camX0'.format(s), F.softmax(self.masklist_camXs[s].squeeze(2), dim=1))
                self.summ_writer.summ_rgb('inputs/view{}_rgb_camX0'.format(s), self.rgb_camXs[:,s])
                self.summ_writer.summ_oned('inputs/view{}_depth_camX0'.format(s), self.depth_camXs[:,s]*self.valid_camXs[:,s], maxval=32.0)
                self.summ_writer.summ_oned('inputs/view{}_valid_camX0'.format(s), self.valid_camXs[:,s], norm=False)
            # self.summ_writer.summ_boxlist2d('inputs/view{}_boxes2d'.format(s), self.rgb_camXs[:,s], self.boxlist2d_camXs[s])

            # get the depth of the current view
            depth = self.depth_camXs[:,s]

            # pointcloud of depth by camXs
            dense_xyz_camX_s = self.dense_xyz_camXs[:,s]

            # masklist are the uneroded mask proposals
            N = len(self.obj_id_list_camXs[s])
            print("For view {0} we expect {1} confident objects".format(s, N))

            N_actual = 0

            for n in list(range(N)):
                obj_id = self.obj_id_list_camXs[s][n]
                obj_catid = self.obj_catid_list_camXs[s][n]
                obj_score = self.obj_score_list_camXs[s][n]

                summ_writer = self.summ_writer

                obj_mask = self.masklist_camXs_safe[s][:, obj_id]
                bkg_mask = 1.0 - self.masklist_camXs[s][:, obj_id]

                im = self.rgb_camXs[:, s].cuda()
                # self.summ_writer.summ_rgb('seg_res/bkg_mask_before'.format(s), im.cuda()*torch.tensor(bkg_mask).cuda())

                weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
                for _ in range(5):
                    bkg_mask = 1.0 - F.conv2d(1.0 - bkg_mask, weights, padding=1).clamp(0, 1)
                    bkg_mask[bkg_mask > 0] = 1

                # self.summ_writer.summ_rgb('seg_res/bkg_mask_after'.format(s), im.cuda()*torch.tensor(bkg_mask).cuda())

                weights = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
                obj_mask = 1.0 - F.conv2d(1.0 - obj_mask, weights, padding=1).clamp(0, 1)

                # give up if <16px available
                num_pts = torch.sum(obj_mask*self.valid_camXs[:,s])
                if num_pts < 32:
                    print('discarding', obj_id, 'because it is too small:', num_pts.cpu().item())
                    self.any_mask_list_camXs[s][self.any_mask_list_camXs[s] == obj_id] = 0
                    continue

                depth_obj = depth[obj_mask*self.valid_camXs[:,s] > 0]
                # give up if more than 32m away
                mean_depth = torch.mean(depth_obj)
                if mean_depth > 50.0:
                    print('discarding', obj_id , 'because it is too far:', mean_depth.cpu().item())
                    self.any_mask_list_camXs[s][self.any_mask_list_camXs[s] == obj_id] = 0
                    continue

                if summ_writer is not None:
                    pass

                assert(self.B==1) # simplify the indexing
                dense_xyz_camX_s = dense_xyz_camX_s.reshape(self.H*self.W, 3)
                dense_obj_camX_s = (obj_mask*self.valid_camXs[:,s]).reshape(self.H*self.W)
                obj_xyz_camX_s = dense_xyz_camX_s[dense_obj_camX_s > 0]

                dense_bkg_camX_s = (bkg_mask*self.valid_camXs[:,s]).reshape(self.H*self.W)
                bkg_xyz_camX_s = dense_xyz_camX_s[dense_bkg_camX_s > 0]

                # Andy: all the operations above are performed in the coord frame of the current view
                # Now we convert the occ's that we got by operating in the current view
                # So we can operate in the reference view after this
                obj_xyz_camX0 = utils.geom.apply_4x4(self.camX0s_T_camXs[:,s], obj_xyz_camX_s.unsqueeze(0))
                bkg_xyz_camX0 = utils.geom.apply_4x4(self.camX0s_T_camXs[:,s], bkg_xyz_camX_s.unsqueeze(0))

                # Prepare object safe mask
                obj_xyz_np = obj_xyz_camX0.squeeze(0).detach().cpu().numpy()
                obj_xyz_std = np.sqrt(np.var(obj_xyz_np, 0))
                obj_xyz_zero = obj_xyz_np - np.median(obj_xyz_np, 0)
                obj_xyz_norm = np.linalg.norm(obj_xyz_zero, axis=1)
                obj_xyz_norm_std = np.sqrt(np.var(obj_xyz_norm, 0))
                obj_xyz_safe = obj_xyz_np[obj_xyz_norm < obj_xyz_norm_std*2]
                if len(obj_xyz_safe) < 16:
                    obj_xyz_safe = obj_xyz_np
                obj_mid_safe = np.mean(obj_xyz_safe, 0)
                xyz_min = np.min(obj_xyz_safe, 0)
                xyz_max = np.max(obj_xyz_safe, 0)
                xyz_mid = (xyz_min + xyz_max)*0.5
                xyz_mid = torch.from_numpy(xyz_mid).float().cuda().reshape(1, 3)
                obj_xyz_safe = torch.from_numpy(obj_xyz_safe).float().cuda()
                obj_xyz_safe = obj_xyz_safe.reshape(1, -1, 3)

                # Prepare background mask
                bkg_xyz_np = bkg_xyz_camX0.squeeze(0).detach().cpu().numpy()
                bkg_xyz_safe = bkg_xyz_np.reshape(1, -1, 3)
                bkg_xyz_safe = torch.from_numpy(bkg_xyz_safe).float().cuda()

                high_res = True
                if high_res:
                    obj_xyz_np = obj_xyz_camX0.squeeze(0).detach().cpu().numpy()
                    obj_xyz_std = np.sqrt(np.var(obj_xyz_np, 0))

                    obj_xyz_zero = obj_xyz_np - np.median(obj_xyz_np, 0)
                    obj_xyz_norm = np.linalg.norm(obj_xyz_zero, axis=1)
                    obj_xyz_norm_std = np.sqrt(np.var(obj_xyz_norm, 0))
                    obj_xyz_np = obj_xyz_np[obj_xyz_norm < obj_xyz_norm_std*5]

                    xyz_min_all = np.min(obj_xyz_np, 0)
                    xyz_max_all = np.max(obj_xyz_np, 0)

                    max_dist = torch.max(torch.from_numpy(xyz_max_all - xyz_min_all))

                    # Maybe it's better if we just set max dist by hand
                    max_dist = torch.max(torch.from_numpy(np.array([10,10,10])))

                    bounds = (-max_dist, max_dist,
                              -max_dist, max_dist,
                              -max_dist, max_dist)

                    Z_dim, Y_dim, X_dim = self.Z//2, self.Y//2, self.X//2 #self.Z//4, self.Y//4, self.X//4 # try 80*80*80 first

                    # create object-centric vox util
                    obj_vox_util = utils.vox.Vox_util(
                        Z_dim, Y_dim, X_dim, 
                        self.set_name, scene_centroid=xyz_mid,
                        bounds=bounds, assert_cube=False
                    )

                    # voxelize object masks
                    occ_camX0_obj = obj_vox_util.voxelize_xyz(self.xyz_camX0s[:,s], Z_dim, Y_dim, X_dim, assert_cube=False)
                    occ_camX0_obj_masked = obj_vox_util.voxelize_xyz(obj_xyz_camX0, Z_dim, Y_dim, X_dim, assert_cube=False)
                    occ_camX0_obj_masked_safe = obj_vox_util.voxelize_xyz(obj_xyz_safe, Z_dim, Y_dim, X_dim, assert_cube=False)

                    # voxelize background mask
                    occ_camX0_bkg_masked_safe = obj_vox_util.voxelize_xyz(bkg_xyz_safe, Z_dim, Y_dim, X_dim, assert_cube=False)
                else:
                    # object centric full scene
                    # create object-centric vox util
                    Z_dim, Y_dim, X_dim = self.Z, self.Y, self.X # 160x160x160
                    obj_vox_util = utils.vox.Vox_util(Z_dim, Y_dim, X_dim, self.set_name, scene_centroid=xyz_mid, assert_cube=False)
                    obj_vox_util = self.vox_util
                    # voxelize object masks
                    occ_camX0_obj = obj_vox_util.voxelize_xyz(self.xyz_camX0s[:,s], Z_dim, Y_dim, X_dim, assert_cube=False)
                    occ_camX0_obj_masked = obj_vox_util.voxelize_xyz(obj_xyz_camX0, Z_dim, Y_dim, X_dim, assert_cube=False)
                    occ_camX0_obj_masked_safe = obj_vox_util.voxelize_xyz(obj_xyz_safe,Z_dim, Y_dim, X_dim, assert_cube=False)

                    # voxelize background mask
                    occ_camX0_bkg_masked_safe = obj_vox_util.voxelize_xyz(bkg_xyz_safe, Z_dim, Y_dim, X_dim, assert_cube=False)

                # Added - full occupancy
                # occupancy of obj on center of scene - used for plotting - may want to delete
                occ_camX0_scene = self.vox_util.voxelize_xyz(self.xyz_camX0s[:,s], self.Z, self.Y, self.X, assert_cube=False)
                occ_camX0_scene_masked = self.vox_util.voxelize_xyz(obj_xyz_camX0, self.Z, self.Y, self.X, assert_cube=False)
                occ_camX0_scene_masked_safe = self.vox_util.voxelize_xyz(obj_xyz_safe, self.Z, self.Y, self.X, assert_cube=False)
                
                if (occ_camX0_obj_masked.sum().item() == 0):
                    print("occupancy is empty for ", self.obj_id_list_camXs[s][n], " discarding...")
                    self.any_mask_list_camXs[s][self.any_mask_list_camXs[s] == obj_id] = 0
                    continue

                rgb_obj = self.vox_util.unproject_rgb_to_mem(
                    self.rgb_camXs[:,0], self.Z, self.Y, self.X, self.pix_T_cams[:,0])

                if summ_writer is not None and do_visualize:
                    # plot object centric occupancy
                    summ_writer.summ_occ('inputs/occ_camX0_obj', occ_camX0_obj)
                    summ_writer.summ_occ('inputs/occ_camX0_obj_masked', occ_camX0_obj_masked)
                    summ_writer.summ_occ('inputs/occ_camX0_obj_masked_safe', occ_camX0_obj_masked_safe)
                    summ_writer.summ_occ('inputs/occ_camX0_obj', occ_camX0_obj)
                    # summ_writer.summ_occ('inputs/occ_agg_obj', occ_agg_obj)
                    # summ_writer.summ_unps('inputs/rgb_agg_obj', [rgb_agg_obj], [occ_agg_obj])

                    # plot scene-centric occupancy
                    summ_writer.summ_occ('inputs/occ_camX0_scene', occ_camX0_scene)
                    summ_writer.summ_occ('inputs/occ_camX0_scene_masked', occ_camX0_scene_masked)
                    summ_writer.summ_occ('inputs/occ_camX0_scene_masked_safe', occ_camX0_scene_masked_safe)
                    summ_writer.summ_feat('inputs/rgb_obj', rgb_obj, pca=False)

                if hyp.do_crf:
                    # Move to pointcloud: color first, implement features later. rgb_camXs (1,S,3,256,768)
                    # get rgb and xyz
                    rgb_pc = self.rgb_camXs.permute(0,1,3,4,2).reshape(-1, self.rgb_camXs.shape[2]) # (1*S*256*768,3)
                    # feat_pc = self.feat_camXs.permute(0,1,3,4,2).reshape(-1, self.feat_camXs.shape[2]) # (1*S*256*768,3)
                    xyz_pc = self.dense_xyz_camX0s_mult.reshape(-1, 3) #(1*S*256*768,3)

                    # get fg pc mask
                    obj_pc_mask = torch.zeros((self.S, self.dense_xyz_camX0s.shape[2])).float().cuda()
                    obj_pc_mask[s] = (dense_obj_camX_s > 0).float()
                    obj_pc_mask = obj_pc_mask.reshape(-1) # (S*H*W,)

                    # get bg pc mask
                    bkg_pc_mask = torch.zeros((self.S, self.dense_xyz_camX0s.shape[2])).float().cuda()
                    bkg_pc_mask[s] = (dense_bkg_camX_s > 0).float()
                    bkg_pc_mask = bkg_pc_mask.reshape(-1) # (S*H*W,)

                    # filter only valid points
                    valid_pc = self.valid_camXs.reshape(-1)
                    rgb_pc = rgb_pc[valid_pc > 0]
                    # feat_pc = feat_pc[valid_pc > 0]
                    xyz_pc = xyz_pc[valid_pc > 0]
                    obj_pc_mask = obj_pc_mask[valid_pc > 0]
                    bkg_pc_mask = bkg_pc_mask[valid_pc > 0]

                    # filter only points around fg mask
                    fg = xyz_pc[obj_pc_mask > 0]
                    fg_avg = torch.mean(fg, dim=0)
                    fg_std = torch.std(fg, dim=0)
                    fg_norm = torch.norm(fg - fg_avg,dim=1)
                    fg_norm_std = torch.std(fg_norm)
                    in_range_idx = fg_norm < fg_norm_std*2
                    orig_idx = torch.where(obj_pc_mask > 0)[0]
                    obj_pc_mask[:] = 0
                    obj_pc_mask[orig_idx[in_range_idx]] = 1

                    # limit search space to speed up
                    limits = torch.tensor([6.0, 6.0, 6.0]).cuda()
                    in_bound_idx = ((xyz_pc < fg_avg + limits) * (xyz_pc > fg_avg - limits)).sum(-1) == 3
                    rgb_pc = rgb_pc[in_bound_idx]
                    # feat_pc = feat_pc[in_bound_idx]
                    xyz_pc = xyz_pc[in_bound_idx]
                    obj_pc_mask = obj_pc_mask[in_bound_idx]
                    bkg_pc_mask = bkg_pc_mask[in_bound_idx]

                    print("fg:", torch.sum(obj_pc_mask))
                    print("bg:", torch.sum(bkg_pc_mask))

                    if torch.sum(obj_pc_mask) == 0 or torch.sum(bkg_pc_mask) == 0:
                        print("ZZZZZZEROOOOOO")
                        continue

                    crf_seg, fg_svm, bg_svm, is_success = self.crfflatnet(
                        feat_pc = rgb_pc,
                        xyz_pc = xyz_pc,
                        fg_seed = obj_pc_mask,
                        bg_seed = bkg_pc_mask,
                        is_feature = False,
                        )

                    xyz_seged = xyz_pc[crf_seg > 0] # (N, 3)

                    # Visualize pointcloud segmentation in voxel grid
                    xyz_seged = xyz_seged.unsqueeze(0)
                    mask_grid = obj_vox_util.voxelize_xyz(xyz_seged, Z_dim, Y_dim, X_dim, assert_cube=False)
                    self.summ_writer.summ_occ('seg_res/occ_seg_result', mask_grid)

                    # Original occupancy  
                    occ_orig = obj_vox_util.voxelize_xyz(self.dense_xyz_camX0s_mult, Z_dim, Y_dim, X_dim, assert_cube=False)

                    if do_visualize:
                        self.summ_writer.summ_occ('seg_res/occ_seg_result', mask_grid)

                    # Original occupancy  
                    occ_orig = obj_vox_util.voxelize_xyz(self.dense_xyz_camX0s_mult, Z_dim, Y_dim, X_dim, assert_cube=False)
                    if do_visualize:
                        self.summ_writer.summ_occ('seg_res/occ_orig', occ_orig)

                    # Foreground mask
                    fg = xyz_pc[obj_pc_mask > 0]
                    fg = fg.unsqueeze(0)
                    occ_fg = obj_vox_util.voxelize_xyz(fg, Z_dim, Y_dim, X_dim, assert_cube=False)
                    if do_visualize:
                        self.summ_writer.summ_occ('seg_res/occ_fg', occ_fg)

                    # Background mask
                    bg = xyz_pc[bkg_pc_mask > 0]
                    bg = bg.unsqueeze(0)
                    occ_bg = obj_vox_util.voxelize_xyz(bg, Z_dim, Y_dim, X_dim, assert_cube=False)
                    if do_visualize:
                        self.summ_writer.summ_occ('seg_res/occ_bg', occ_bg)

                    # Foreground svm
                    fg_svm = torch.from_numpy(fg_svm).cuda().unsqueeze(0)
                    occ_fg_svm = obj_vox_util.voxelize_xyz(fg_svm, Z_dim, Y_dim, X_dim, assert_cube=False)
                    if do_visualize:
                        self.summ_writer.summ_occ('seg_res/occ_fg_svm', occ_fg_svm)

                    # Background svm
                    bg_svm = torch.from_numpy(bg_svm).cuda().unsqueeze(0)
                    occ_bg_svm = obj_vox_util.voxelize_xyz(bg_svm, Z_dim, Y_dim, X_dim, assert_cube=False)
                    if do_visualize:
                        self.summ_writer.summ_occ('seg_res/occ_bg_svm', occ_bg_svm)

                    # Project this mask to all the views
                    # remove 3d outliers
                    xyz_seged = xyz_pc[crf_seg > 0].cpu().numpy() # (N,3)
                    zscore = np.abs(stats.zscore(xyz_seged, axis=0)) # (N,3)
                    remove_idx = 1 - (zscore[:,0]<3) * (zscore[:,1]<3) * (zscore[:,2]<3)
                    crf_seg[remove_idx] = 0

                    # indices in inbound pc --> indices in valid pc
                    in_bound_idx = torch.where(in_bound_idx == 1)[0]
                    fg_idx_valid = in_bound_idx[crf_seg > 0]

                    # indices in valid pc --> indices in full pc
                    valid_idx = torch.where(self.valid_camXs.reshape(-1) == 1)[0]
                    fg_idx_amodal = valid_idx[fg_idx_valid]

                    # full pc
                    for ss in list(range(self.S)):
                        xyz_pc = utils.geom.apply_4x4(self.camXs_T_camX0s[:,ss], self.dense_xyz_camX0s_mult).reshape(-1, 3)

                        # Project the mask (modal)
                        front = ss * self.H * self.W
                        rear = (ss+1) * self.H * self.W
                        fg_idx_modal = fg_idx_amodal[(fg_idx_amodal >= front) * (fg_idx_amodal < rear)]

                        xyz_modal = xyz_pc[fg_idx_modal]
                        xyz_modal_seg = xyz_modal.unsqueeze(0)
                        
                        # estimating 3D box in kitti format in camX frame
                        mask_grid_s = obj_vox_util.voxelize_xyz(xyz_modal_seg, Z_dim, Y_dim, X_dim, assert_cube=False)
                        # st()
                        _, box3dlist, _, _, _ = utils.misc.get_boxes_from_flow_mag(mask_grid_s.squeeze(0), 1)
                        pred_lrtlist = utils.geom.convert_boxlist_to_lrtlist(box3dlist)
                        pred_lrtlist = obj_vox_util.apply_ref_T_mem_to_lrtlist(pred_lrtlist, Z_dim, Y_dim, X_dim)
                        pred_xyzlist = utils.geom.get_xyzlist_from_lrtlist(pred_lrtlist)
                        # pred_boxes = utils.geom.corners_to_box3D_single_py(pred_xyzlist)
                        box3d = pred_xyzlist[0][0]

                        mask_modal_xy = utils.geom.Camera2Pixels(xyz_modal.unsqueeze(0), self.pix_T_cams[:,ss])
                        mask_modal_x, mask_modal_y = torch.unbind(mask_modal_xy, dim=2)
                        mask_modal_x = mask_modal_x.reshape(-1).clamp(0, self.W-1)
                        mask_modal_y = mask_modal_y.reshape(-1).clamp(0, self.H-1)

                        # If mask not in view
                        if len(mask_modal_x) == 0:
                            continue
                            mask_modal = torch.zeros((self.H, self.W)).float()
                        elif torch.max(mask_modal_x) == 0 or torch.max(mask_modal_y) == 0 or torch.min(mask_modal_x)==self.W-1 or torch.min(mask_modal_y)==self.H-1:
                            continue
                            mask_modal = torch.zeros((self.H, self.W)).float()
                        else:
                            # fill in mask
                            mask_modal_x = mask_modal_x.long()
                            mask_modal_y = mask_modal_y.long()
                            mask_modal = torch.zeros((self.H, self.W)).float()
                            mask_modal[mask_modal_y, mask_modal_x] = 1

                        # Project the mask (amodal)
                        xyz_amodal = xyz_pc[fg_idx_amodal]
                        mask_amodal_xy = utils.geom.Camera2Pixels(xyz_amodal.unsqueeze(0), self.pix_T_cams[:,ss])
                        mask_amodal_x, mask_amodal_y = torch.unbind(mask_amodal_xy, dim=2)
                        mask_amodal_x = mask_amodal_x.reshape(-1).cpu().numpy()
                        mask_amodal_y = mask_amodal_y.reshape(-1).cpu().numpy()

                        # remove outliers
                        mask_amodal_xy = np.stack([mask_amodal_x, mask_amodal_y], axis=1) # (N, 2)
                        zscore = np.abs(stats.zscore(mask_amodal_xy, axis=0))
                        keep_idx = (zscore[:,0]<3) * (zscore[:,1]<3)
                        mask_amodal_x = mask_amodal_x[keep_idx]
                        mask_amodal_y = mask_amodal_y[keep_idx]

                        # clamp after removing outliers
                        mask_amodal_x = torch.from_numpy(mask_amodal_x).clamp(0, self.W-1).long()
                        mask_amodal_y = torch.from_numpy(mask_amodal_y).clamp(0, self.H-1).long()

                        # If mask not in view
                        if torch.max(mask_amodal_x) == 0 or torch.max(mask_amodal_y) == 0 or torch.min(mask_amodal_x)==self.W-1 or torch.min(mask_amodal_y)==self.H-1:
                            continue

                        # fill in mask
                        mask_amodal = torch.zeros((self.H, self.W)).float()
                        mask_amodal[mask_amodal_y, mask_amodal_x] = 1

                        imhere = self.rgb_camXs[:, ss].cuda()
                        #self.summ_writer.summ_rgb('check_mask/modal_mask_{0}_{1}_{2}'.format(s,n,ss), imhere.cuda()*mask_modal.unsqueeze(0).unsqueeze(0).cuda())
                        #self.summ_writer.summ_rgb('check_mask/amodal_mask_{0}_{1}_{2}'.format(s,n,ss), imhere.cuda()*mask_amodal.unsqueeze(0).unsqueeze(0).cuda())

                        # Get bbox from amodal mask
                        mask_modal = mask_modal.cpu().numpy()
                        mask_amodal = mask_amodal.cpu().numpy()
                        y_idx, x_idx = np.where(mask_amodal > 0)
                        bbox_amodal = np.array([np.min(y_idx), np.min(x_idx), np.max(y_idx), np.max(x_idx)])
                        ymin, xmin, ymax, xmax = bbox_amodal 
                        area_conf = float((xmax-xmin) * (ymax-ymin))
                        if area_conf == 0:
                            continue

                        # Condition modal mask on amodal mask (outlier removal is better with amodal mask)
                        mask_modal *= mask_amodal

                        # save to dictionary
                        key = (ss,obj_catid)
                        if key in all_info_list:
                            all_info_list[key].append([obj_score, mask_modal, bbox_amodal, mask_amodal, box3d])
                        else:
                            all_info_list[key] = [[obj_score, mask_modal, bbox_amodal, mask_amodal, box3d]]

                    N_actual += 1

            print("For view {0}, there are big objects: {1} / {2} out of confident objects".format(s, N_actual, N))     

        # Return early if no densecrf segmentation is done for all objects
        if len(all_info_list) is 0:
            print("No big confident mask found!")
            return total_loss, results, False

        # all_info_list: List of all objects' projections (dictionary, key=(scene,catid), value=list of list [[score1, mask1, bbox1], [score2, mask2, bbox2]])
        # For each view
        for s in list(range(self.S)):
            # Image
            img = self.rgb_camXs[:, s].cuda() # (1,3,H,W)

            # Write data
            keep_list = []

            # Intra-class NMS
            for catid in range(1,9):
                key = (s, catid)
                if key in all_info_list:
                    # in the current view and this category, sort masks in descending maskrcnn score order
                    view_cat_list = sorted(all_info_list[key], reverse=True, key=lambda x: x[0])
                    while len(view_cat_list) > 0:
                        # add in the most confident mask
                        keep_list.append([catid, view_cat_list[0][1], view_cat_list[0][2], view_cat_list[0][0], view_cat_list[0][4]])

                        mask_conf = view_cat_list[0][1]
                        area_conf = np.sum(mask_conf)

                        # suppress other masks with overlap (need to use amodal masks here, modal masks are too sparse)
                        view_cat_list_rem = []
                        for i in range(1, len(view_cat_list)):
                            mask_cur = view_cat_list[i][1]
                            area_cur = np.sum(mask_cur)

                            if area_cur == 0:
                                continue

                            intersection = np.sum(mask_conf * mask_cur)

                            if intersection/area_conf < 0.5 and intersection/area_cur < 0.5:
                                view_cat_list_rem.append(view_cat_list[i])

                        view_cat_list = view_cat_list_rem

            # Final lists to fill in 
            catid_list = []
            mask_list = []
            bbox_list = []
            score_list = []
            box3d_list = []
            # Inter-class NMS
            keep_list = sorted(keep_list, reverse=True, key=lambda x: x[3])
            while len(keep_list) > 0:
                catid_list.append(keep_list[0][0])
                mask_list.append(keep_list[0][1])
                bbox_list.append(keep_list[0][2])
                score_list.append(keep_list[0][3])
                box3d_list.append(keep_list[0][4])
                # mask_conf = keep_list[0][1]
                # area_conf = np.sum(mask_conf)
                # st()
                ymin1, xmin1, ymax1, xmax1 = keep_list[0][2]
                area_conf = (ymax1 - ymin1) * (xmax1 - xmin1)

                rem_info_list_new = []

                for rrr in range(1, len(keep_list)):
                    ymin2, xmin2, ymax2, xmax2 = keep_list[rrr][2]
                    area_cur = (ymax2 - ymin2) * (xmax2 - xmin2)
                    if not area_cur > 0:
                        continue

                    x_dist = (min(xmax1, xmax2) - max(xmin1, xmin2))
                    y_dist = (min(ymax1, ymax2) - max(ymin1, ymin2))
                    
                    if x_dist > 0 and y_dist > 0:
                        area_overlap = x_dist * y_dist
                        if float(area_overlap) /float(area_conf + area_cur - area_overlap) > 0.5:
                            continue

                    rem_info_list_new.append(keep_list[rrr])

                keep_list = rem_info_list_new
                # keep_list_rem = []
                # for i in range(1, len(keep_list)):
                #     mask_cur = keep_list[i][1]
                #     area_cur = np.sum(mask_cur)

                #     if area_cur == 0:
                #         continue

                #     intersection = np.sum(mask_conf * mask_cur)

                #     if intersection/area_conf < 0.5 and intersection/area_cur < 0.5:
                #         keep_list_rem.append(keep_list[i])

                # keep_list = keep_list_rem

            ################ MAP evaluation #####################
            if do_map_eval:
                # setup gt
                boxlist_g = self.boxlist_g_s[s]
                class_list_g = self.classlist_g_s[s]

                # Mask-rcnn
                boxlist_e_maskrcnn = [box.cpu().numpy() for box in self.obj_all_box_list_camXs[s]]
                for i in range(len(boxlist_e_maskrcnn)):
                    boxlist_e_maskrcnn[i][0] /= self.H
                    boxlist_e_maskrcnn[i][1] /= self.W
                    boxlist_e_maskrcnn[i][2] /= self.H
                    boxlist_e_maskrcnn[i][3] /= self.W
                boxlist_e_maskrcnn = torch.from_numpy(np.array(boxlist_e_maskrcnn)).unsqueeze(0)
                class_list_e_maskrcnn = [1 if cls==1 or cls==3 else 0 for cls in self.obj_all_catid_list_camXs[s]] # bike 1, car 0
                confidence_list_maskrcnn = self.obj_all_score_list_camXs[s]
                # mAP = utils.eval.get_mAP_with_classes(boxlist_e, boxlist_g, class_list_e, class_list_g, confidence_list, num_classes=2, mode="coco")

                # Pseudo-label
                boxlist_e_pseudo = [box.astype(np.float32) for box in bbox_list]
                for i in range(len(boxlist_e_pseudo)):
                    boxlist_e_pseudo[i][0] /= self.H
                    boxlist_e_pseudo[i][1] /= self.W
                    boxlist_e_pseudo[i][2] /= self.H
                    boxlist_e_pseudo[i][3] /= self.W
                boxlist_e_pseudo = torch.from_numpy(np.array(boxlist_e_pseudo)).unsqueeze(0)
                class_list_e_pseudo = [1 if cls==1 or cls==3 else 0 for cls in catid_list] # bike 1, car 0
                confidence_list_pseudo = score_list
                # mAP = utils.eval.get_mAP_with_classes(boxlist_e, boxlist_g, class_list_e, class_list_g, confidence_list, num_classes=2, mode="coco")

                boxlist_g = torch.from_numpy(np.array(boxlist_g)).unsqueeze(0).clamp(0,1)

                # print('{0} / {1} / {2}'.format(boxlist_g.shape, boxlist_e_maskrcnn.shape, boxlist_e_pseudo.shape))

                if do_visualize:
                    if boxlist_g.shape[1] > 0:
                        self.summ_writer.summ_boxlist2d('finals/boxes_{}_gt'.format(s), self.rgb_camXs[:,s], boxlist_g)
                    if boxlist_e_pseudo.shape[1] > 0:
                        print("pred visualized")
                        self.summ_writer.summ_boxlist2d('finals/boxes_{}_pred'.format(s), self.rgb_camXs[:,s], boxlist_e_pseudo)
                    if boxlist_e_maskrcnn.shape[1] > 0:
                        self.summ_writer.summ_boxlist2d('finals/boxes_{}_maskrcnn'.format(s), self.rgb_camXs[:,s], boxlist_e_maskrcnn)

                boxlist_g = boxlist_g.squeeze(0).cpu().numpy()
                boxlist_e_pseudo = boxlist_e_pseudo.squeeze(0).cpu().numpy()
                boxlist_e_maskrcnn = boxlist_e_maskrcnn.squeeze(0).cpu().numpy()

                # writing labels in folders and files
                output_gt_dir = "./gt_pred_train"
                output_maskrcnn_dir = "./maskrcnn_pred_train"
                output_pseudo_dir = "./pseudo_dir_train"

                if not os.path.exists(output_gt_dir):
                    os.makedirs(output_gt_dir)

                if not os.path.exists(output_maskrcnn_dir):
                    os.makedirs(output_maskrcnn_dir)
                
                if not os.path.exists(output_pseudo_dir):
                    os.makedirs(output_pseudo_dir)

                gt_file = open(f"{output_gt_dir}/{self.img_count}.txt", 'w')
                maskrcnn_file = open(f"{output_maskrcnn_dir}/{self.img_count}.txt", 'w')
                pseudo_file = open(f"{output_pseudo_dir}/{self.img_count}.txt", 'w')

                for i in range(len(boxlist_g)):
                    boxlist_g[i][0] *= self.H #ymin
                    boxlist_g[i][1] *= self.W #xmin
                    boxlist_g[i][2] *= self.H #ymax
                    boxlist_g[i][3] *= self.W #xmax
                    gt_file.write(f"'car' {round(boxlist_g[i][1])} {round(boxlist_g[i][0])} {round(boxlist_g[i][3])} {round(boxlist_g[i][2])}\n")
                gt_file.close()

                # getting class labels as text
                for i in range(len(boxlist_e_maskrcnn)):
                    boxlist_e_maskrcnn[i][0] *= self.H #ymin
                    boxlist_e_maskrcnn[i][1] *= self.W #xmin
                    boxlist_e_maskrcnn[i][2] *= self.H #ymax
                    boxlist_e_maskrcnn[i][3] *= self.W #xmax
                    maskrcnn_file.write(f"'car' 1 {round(boxlist_e_maskrcnn[i][1])} {round(boxlist_e_maskrcnn[i][0])} {round(boxlist_e_maskrcnn[i][3])} {round(boxlist_e_maskrcnn[i][2])}\n")
                maskrcnn_file.close()

                # getting class labels as text
                for i in range(len(boxlist_e_pseudo)):
                    boxlist_e_pseudo[i][0] *= self.H #ymin
                    boxlist_e_pseudo[i][1] *= self.W #xmin
                    boxlist_e_pseudo[i][2] *= self.H #ymax
                    boxlist_e_pseudo[i][3] *= self.W #xmax
                    pseudo_file.write(f"'car' 1 {round(boxlist_e_pseudo[i][1])} {round(boxlist_e_pseudo[i][0])} {round(boxlist_e_pseudo[i][3])} {round(boxlist_e_pseudo[i][2])}\n")
                pseudo_file.close()
           
            ################ save predictions to dataset ################
            # --> stored in /projects/katefgroup/viewpredseg_dataset
            if make_dataset:
                # st()
                data_dict = {"img": img.cpu().numpy(),  
                             "mask_list": mask_list,
                             "bbox_list": bbox_list,
                             "catid_list": catid_list}
                np.save(f'{DATASET_PATH}/{self.img_count}.npy', data_dict)
            
            self.img_count += 1


            ############# save pointnet++ dataset #########################
                    
            if make_pointnet_dataset:
                
                # self.if_not_exists_makeit(DATASET_PATH_POINTNET_BASE)
                # self.if_not_exists_makeit(DATASET_PATH_POINTNET_OBJECT_BASE)
                # self.if_not_exists_makeit(DATASET_PATH_POINTNET_OBJECT_TRAIN)
                # self.if_not_exists_makeit(DATASET_PATH_POINTNET_OBJECT_TEST)
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
                
                # write image in DATASET_PATH_KITTI_IMAGE
                im = utils.improc.back2color(img).squeeze(0).permute(1, 2, 0).cpu().numpy()
                im = Image.fromarray(im)
                im.save(os.path.join(DATASET_PATH_KITTI_IMAGES, f'{self.idx_count}.png'))
                
                # write pointcloud
                
                pc_xyz = self.xyz_camXs[0, s].cpu().numpy()
                np.save(os.path.join(DATASET_PATH_KITTI_VELODYNE, f'{self.idx_count}.npy'), pc_xyz)
                # with open(os.path.join(DATASET_PATH_KITTI_VELODYNE, f'{self.idx_count}.npy'), 'w') as f:
                    # np.save(f, pc_xyz) 

                # labels

                with open(os.path.join(DATASET_PATH_KITTI_LABELS, f'{self.idx_count}.txt'), 'w') as f:
                    for i in range(len(catid_list)):
                        # st()
                        type_name = 'Car'
                        ymin, xmin, ymax, xmax = bbox_list[i]
                        # st()
                        box3d_str = ' '.join(str(e) for e in box3d_list[i].reshape(-1).cpu().numpy())
                        # xc, yc, zc, wid, hei, dep, rx, ry, rz = box3d_list[i][0]
                        f.write(f"{type_name} 0 3 0 {xmin} {ymin} {xmax} {ymax} {box3d_str}\n")

                # calibs
                with open(os.path.join(DATASET_PATH_KITTI_CALIBS, f'{self.idx_count}.txt'), 'w') as f:
                    # st()
                    pix_T_cam_save = self.pix_T_cams[0, s].reshape(-1).cpu().numpy()
                    pix_T_cam_str = ' '.join(str(e) for e in pix_T_cam_save)
                    f.write(f'pix_T_cam: {pix_T_cam_str}\n')

                    camX_T_origin_save = self.camXs_T_origin[s].reshape(-1).cpu().numpy()
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

                with open(os.path.join(DATASET_PATH_RGB_DETECTIONS, f'rgb_detections_train.txt'), 'a+') as f:
                    path_to_img = str(os.path.join(DATASET_PATH_KITTI_IMAGES, f'{self.idx_count}.png'))
                    for i in range(len(catid_list)):
                        ymin, xmin, ymax, xmax = bbox_list[i]
                        f.write(f"{path_to_img} 2 1 {xmin} {ymin} {xmax} {ymax}\n")               

                self.idx_count += 1


            ######################### POINTNET++ DATASET ENDS #########################################
            
            # Below are all visualization
            pred_any_mask = np.zeros((self.H, self.W))
            for i in range(len(catid_list)):
                pred_any_mask = np.logical_or(pred_any_mask, mask_list[i])

            for catid in range(1,9):
                # view and category
                key = (s, catid)
                if key in all_info_list:
                    for info in all_info_list[key]:
                        # each object of this category in this view
                        pred_any_mask = np.logical_or(pred_any_mask, info[1])
            pred_any_mask = torch.from_numpy(pred_any_mask).float()

            if do_visualize:
                # (1) input rgb image
                self.summ_writer.summ_rgb('viewmine_output/view{0}_rgb_input'.format(s),img)
                # (2) input foreground mask, aka pretrained MaskRCNN output overlaid on image
                self.any_mask_list_camXs[s][self.any_mask_list_camXs[s] > 0] = 1
                self.summ_writer.summ_rgb('viewmine_output/view{0}_source_mask'.format(s), img.cuda()*self.any_mask_list_camXs[s].cuda())
                # (3) output modal mask overlaid on image
                self.summ_writer.summ_rgb('viewmine_output/view{0}_output_mask'.format(s), img.cuda()*pred_any_mask.unsqueeze(0).unsqueeze(0).cuda())
                # (4) output amodal bbox overlaid on image (code adapted from utils.improc.draw_boxlist2d_on_image_py())
                color_map = matplotlib.cm.get_cmap('tab10')
                color_map = color_map.colors
                img_box = utils.improc.back2color(img.detach().clone())
                img_box = img_box[0].cpu().numpy().copy()
                img_box = np.transpose(img_box, [1, 2, 0])
                img_box = cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR)
                for i, bbox in enumerate(bbox_list):
                    color_id = catid_list[i]
                    color = color_map[color_id]
                    color = np.array(color)*255.0

                    ymin, xmin, ymax, xmax = bbox
                    cv2.line(img_box, (xmin, ymin), (xmin, ymax), color, 2, cv2.LINE_AA)
                    cv2.line(img_box, (xmin, ymin), (xmax, ymin), color, 2, cv2.LINE_AA)
                    cv2.line(img_box, (xmax, ymin), (xmax, ymax), color, 2, cv2.LINE_AA)
                    cv2.line(img_box, (xmax, ymax), (xmin, ymax), color, 2, cv2.LINE_AA)
                img_box = cv2.cvtColor(img_box.astype(np.uint8), cv2.COLOR_BGR2RGB)
                img_box = torch.from_numpy(img_box).type(torch.ByteTensor).permute(2, 0, 1)
                img_box = torch.unsqueeze(img_box, dim=0)
                img_box = utils.improc.preprocess_color(img_box)
                img_box = torch.reshape(img_box, [1, 3, self.H, self.W])
                self.summ_writer.summ_rgb('viewmine_output/view{0}_amodal_box'.format(s), img_box.cuda())


        

        return total_loss, results, False
        

    def if_not_exists_makeit(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir) 
    
    def run_test(self, feed):
        pass

    def forward(self, feed):
        
        data_ok = self.prepare_common_tensors(feed)

        if not data_ok:
            print("No objects detected in 2D, returning early")
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
            
        else:
            if self.set_name == 'train':
                return self.run_train(feed)
            elif self.set_name == 'test':
                return self.run_test(feed)
            else:
                print('Not implemented this set name: ', self.set_name)
                assert(False)





