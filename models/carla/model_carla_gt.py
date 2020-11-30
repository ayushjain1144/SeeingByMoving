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
from backend import saverloader, inputs

# from nets.feat3dnet import Feat3dNet

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
do_visualise = False
make_pointnet_dataset = True
# writing labels in folders and files
DATASET_PATH =  '/projects/katefgroup/viewpredseg/carla_supervised_val'

ATASET_PATH_POINTNET_BASE = "/home/ayushj2/frustum_pointnets_pytorch" # already exists
DATASET_PATH_KITTI_BASE = '/home/ayushj2/frustum_pointnets_pytorch/dataset/KITTI_gt' 
DATASET_PATH_IMAGESETS = os.path.join(DATASET_PATH_KITTI_BASE, 'ImageSets')
DATASET_PATH_OBJECTS = os.path.join(DATASET_PATH_KITTI_BASE, 'object')
DATASET_PATH_TRAIN = os.path.join(DATASET_PATH_OBJECTS, 'val') # change to testing too
DATASET_PATH_KITTI_IMAGES = os.path.join(DATASET_PATH_TRAIN, 'image_2')
DATASET_PATH_KITTI_VELODYNE = os.path.join(DATASET_PATH_TRAIN, 'velodyne')
DATASET_PATH_KITTI_LABELS = os.path.join(DATASET_PATH_TRAIN, 'label_2')
DATASET_PATH_KITTI_CALIBS = os.path.join(DATASET_PATH_TRAIN, 'calib')

DATASET_PATH_SMALL_KITTI = "/home/ayushj2/frustum_pointnets_pytorch/kitti"
DATASTET_PATH_IMAGE_SETS = os.path.join(DATASET_PATH_SMALL_KITTI, 'image_sets_gt_val')
DATASET_PATH_RGB_DETECTIONS = os.path.join(DATASET_PATH_SMALL_KITTI, 'rgb_detections_gt_val')


class CARLA_GT(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = GTModel()

class GTModel(nn.Module):
    def __init__(self):
        super(GTModel, self).__init__()

        self.feat3dnet = Feat3dNet(in_dim=4)

        # # Initialize maskRCNN
        # cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # # cfg.MODEL.WEIGHTS = "/home/zhaoyuaf/pytorch_disco/logs_replica_detectron/model_0019999.pth"
        # self.cfg = cfg
        # self.maskrcnn = DefaultPredictor(cfg)

        # if not hyp.shuffle_train:
        #     self.q = 0

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
        self.masks_camXs = feed['masks_camXs']
        # self.feat_camXs = []

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

        self.obj_all_catid_list_camXs = []
        self.obj_all_score_list_camXs = []
        self.obj_all_box_list_camXs = []

        return True

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        # get full occupancy from full view
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 #hyp.YMIN
        scene_centroid_z = -1.0
        # for cater the table is y=0, so we move upward a bit
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()  
        # st()                                                               
        self.vox_util = utils.vox.Vox_util(
            self.Z, self.Y, self.X, 
            self.set_name, scene_centroid=self.scene_centroid,
            assert_cube=True)

        #dense multiview pointcloud in frame of camera 0
        occ_memX0s_dense_scene = self.vox_util.voxelize_xyz(self.dense_xyz_camX0s_mult, self.Z, self.Y, self.X, assert_cube=False)
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

        # List of classlist_g, boxlist_g
        self.classlist_g_s = []
        self.boxlist_g_s = []
        self.boxlist3d_g_s = []

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
            boxlist3d_g = []
            for i in range(N_lrt):
                
                if self.full_scorelist_s[:, s, i] > 0:
                    # st()    
                    lx, ly, lz = (torch.unbind(lenlist_cam, dim=1))[i][0]
                    if lx > 1.0:
                        classlist_g.append(1) #"car"
                        boxlist_g.append(corners_pix[0,i,:].cpu().numpy())
                        boxlist3d_g.append(xyzlist_camXs[0, i].cpu().numpy())

                    else:
                        continue
                        # classlist_g.append(0) #"bike"
                        # don;t append bikes in gt
                        # boxlist_g.append(corners_pix[0,i,:].cpu().numpy())

            self.classlist_g_s.append(classlist_g)
            self.boxlist_g_s.append(boxlist_g)
            self.boxlist3d_g_s.append(boxlist3d_g)

            if len(boxlist_g) == 0:
                print("boxlist_g empty....returning")
                continue  

            ################ MAP evaluation #####################
            if do_map_eval:
                # Image
                img = self.rgb_camXs[:, s].cuda()

                # setup gt
                boxlist_g = self.boxlist_g_s[s]
                classlist_g = self.classlist_g_s[s]

                # GT processing
                boxlist_g = torch.from_numpy(np.array(boxlist_g)).unsqueeze(0).clamp(0,1)

                # Visualize boxes
                if boxlist_g.shape[1] > 0:
                    self.summ_writer.summ_boxlist2d('finals/boxes_{}_gt'.format(s), self.rgb_camXs[:,s], boxlist_g)

                boxlist_g = boxlist_g.squeeze(0).cpu().numpy()

                # Mask processing
                masklist_g = self.masks_camXs[s].cpu().numpy()
                masklist_g = [masklist_g[0,i,:,:,0] for i in range(masklist_g.shape[1])]

                # boxes and masks
                bbox_list = []
                mask_list = []
                for i in range(len(boxlist_g)):
                    boxlist_g[i][0] *= self.H
                    boxlist_g[i][1] *= self.W
                    boxlist_g[i][2] *= self.H
                    boxlist_g[i][3] *= self.W

                    for j in range(len(masklist_g)):
                        in_box_pix = masklist_g[j][int(boxlist_g[i][0]):int(boxlist_g[i][2]), int(boxlist_g[i][1]):int(boxlist_g[i][3])].sum()
                        out_box_pix = masklist_g[j].sum() - in_box_pix
                        if in_box_pix > out_box_pix:
                            mask_list.append(masklist_g[j])
                            break
                            
                    bbox_list.append(boxlist_g[i])

                # catid_list
                catid_list = classlist_g

                # make_dataset = True
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
                box3d_list = self.boxlist3d_g_s[s]

                with open(os.path.join(DATASET_PATH_KITTI_LABELS, f'{self.idx_count}.txt'), 'w') as f:
                    for i in range(len(catid_list)):
                        # st()
                        type_name = 'Car'
                        ymin, xmin, ymax, xmax = bbox_list[i]
                        # st()
                        box3d_str = ' '.join(str(e) for e in box3d_list[i].reshape(-1))
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
                with open(os.path.join(DATASTET_PATH_IMAGE_SETS, f'val.txt'), 'a+') as f:
                    f.write(f"{self.idx_count}\n")

                with open(os.path.join(DATASET_PATH_IMAGESETS, f'val.txt'), 'a+') as f:
                    f.write(f"{self.idx_count}\n")

                with open(os.path.join(DATASET_PATH_RGB_DETECTIONS, f'rgb_detections_val.txt'), 'a+') as f:
                    path_to_img = str(os.path.join(DATASET_PATH_KITTI_IMAGES, f'{self.idx_count}.png'))
                    for i in range(len(catid_list)):
                        ymin, xmin, ymax, xmax = bbox_list[i]
                        f.write(f"{path_to_img} 2 1 {xmin} {ymin} {xmax} {ymax}\n")               

                self.idx_count += 1


            ######################### POINTNET++ DATASET ENDS #########################################
            
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


