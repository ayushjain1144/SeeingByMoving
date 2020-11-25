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

from nets.feat3dnet import Feat3dNet

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

np.set_printoptions(precision=2)
np.random.seed(0)
MAX_QUEUE = 10
make_dataset = False
do_map_eval = True

# writing labels in folders and files
output_gt_dir = "./gt_pred_train" #2:19999, 3:39999
output_maskrcnn_dir = "./maskrcnn_pred_train"

if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)

if not os.path.exists(output_maskrcnn_dir):
    os.makedirs(output_maskrcnn_dir)

class CARLA_EVAL(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = EvalModel()

class EvalModel(nn.Module):
    def __init__(self):
        super(EvalModel, self).__init__()

        self.feat3dnet = Feat3dNet(in_dim=4)

        # Initialize maskRCNN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        #cfg.MODEL.WEIGHTS = "/home/zhaoyuaf/pytorch_disco/logs_detectron/logs_carla_detectron_gt/model_0064999.pth"
        #cfg.MODEL.WEIGHTS = "/home/zhaoyuaf/pytorch_disco/logs_detectron/logs_carla_detectron_ss/model_0034999.pth"
        self.cfg = cfg
        self.maskrcnn = DefaultPredictor(cfg)

        # if not hyp.shuffle_train:
        #     self.q = 0

        self.obj_counts = 0
        self.avg_iou = 0
        self.img_count = 0

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
            #self.summ_writer.summ_rgb('input_rgb/view{}_instance_mask'.format(s), torch.from_numpy(seg_im).permute(2, 0, 1).unsqueeze(0))

            # get just objects/"things" - theres prob an easier way to do this
            all_info_list = []
            for segs in range(len(pred_masks)):
                # 1 and 3 are bikes. removing them for now
                if pred_classes[segs] > 1 and pred_classes[segs] <= 8 and pred_classes[segs] != 3: #and pred_scores[segs] > 0.95:
                    y, x = torch.where(pred_masks[segs])
                    if len(y) == 0:
                        continue 
                    #pred_box = torch.Tensor([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                    pred_box = pred_boxes[segs][[1,0,3,2]]
                    all_info_list.append([pred_box, pred_classes[segs].item(), pred_scores[segs].item()])

            obj_keep_catids = []
            obj_keep_scores = []
            obj_keep_boxes = []
            rem_info_list = sorted(all_info_list, reverse=True, key=lambda x: x[2])
            while len(rem_info_list) > 0:
                obj_keep_catids.append(rem_info_list[0][1])
                obj_keep_scores.append(rem_info_list[0][2])
                obj_keep_boxes.append(rem_info_list[0][0])

                ymin1, xmin1, ymax1, xmax1 = rem_info_list[0][0].cpu().numpy()
                area_conf = (ymax1 - ymin1) * (xmax1 - xmin1)

                rem_info_list_new = []

                for rrr in range(1, len(rem_info_list)):
                    ymin2, xmin2, ymax2, xmax2 = rem_info_list[rrr][0].cpu().numpy()
                    area_cur = (ymax2 - ymin2) * (xmax2 - xmin2)
                    if not area_cur > 0:
                        continue

                    x_dist = (min(xmax1, xmax2) - max(xmin1, xmin2))
                    y_dist = (min(ymax1, ymax2) - max(ymin1, ymin2))
                    
                    if x_dist > 0 and y_dist > 0:
                        area_overlap = x_dist * y_dist
                        if float(area_overlap) /float(area_conf + area_cur - area_overlap) > 0.5:
                            continue
                        if float(area_overlap) / float(area_cur) > 0.5 or float(area_overlap) / float(area_conf) > 0.5:
                            continue

                    rem_info_list_new.append(rem_info_list[rrr])

                rem_info_list = rem_info_list_new

            self.obj_all_catid_list_camXs.append(obj_keep_catids)
            self.obj_all_score_list_camXs.append(obj_keep_scores)
            self.obj_all_box_list_camXs.append(obj_keep_boxes)

        print("preped")

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

            ################ MAP evaluation #####################
            if do_map_eval:
                # setup gt
                boxlist_g = self.boxlist_g_s[s]
                classlist_g = self.classlist_g_s[s]

                # Mask-rcnn processing
                boxlist_e_maskrcnn = [box.cpu().numpy() for box in self.obj_all_box_list_camXs[s]]
                for i in range(len(boxlist_e_maskrcnn)):
                    boxlist_e_maskrcnn[i][0] /= self.H
                    boxlist_e_maskrcnn[i][1] /= self.W
                    boxlist_e_maskrcnn[i][2] /= self.H
                    boxlist_e_maskrcnn[i][3] /= self.W
                boxlist_e_maskrcnn = torch.from_numpy(np.array(boxlist_e_maskrcnn)).unsqueeze(0)
                class_list_e_maskrcnn = self.obj_all_catid_list_camXs[s] # consider all objects, add constraint if we want later
                confidence_list_maskrcnn = self.obj_all_score_list_camXs[s]

                # GT processing
                boxlist_g = torch.from_numpy(np.array(boxlist_g)).unsqueeze(0).clamp(0,1)

                # Visualize boxes
                '''
                if boxlist_g.shape[1] > 0:
                    self.summ_writer.summ_boxlist2d('finals/boxes_{}_gt'.format(s), self.rgb_camXs[:,s], boxlist_g)
                if boxlist_e_maskrcnn.shape[1] > 0:
                    self.summ_writer.summ_boxlist2d('finals/boxes_{}_maskrcnn'.format(s), self.rgb_camXs[:,s], boxlist_e_maskrcnn)
                '''

                # To numpy
                boxlist_g = boxlist_g.squeeze(0).cpu().numpy()
                boxlist_e_maskrcnn = boxlist_e_maskrcnn.squeeze(0).cpu().numpy()

                # Write to file
                gt_file = open(f"{output_gt_dir}/{self.img_count}.txt", 'w')
                maskrcnn_file = open(f"{output_maskrcnn_dir}/{self.img_count}.txt", 'w')

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

                self.img_count += 1

        return total_loss, results, False
        

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


