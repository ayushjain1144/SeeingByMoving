# python pascalvoc.py -gt ../gt_pred_finetune/ -det ../maskrcnn_pred_finetune/ -gtformat 'xyrb' -detformat 'xyrb'
# python pascalvoc.py -gt ../gt_pred_finetune/ -det ../maskrcnn_pred_finetune/ -gtformat 'xyrb' -detformat 'xyrb' -t 0.3
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
do_visualize = False

# writing labels in folders and files
output_gt_dir = "./gt_novel_pseudo"
output_maskrcnn_dir = "./maskrcnn_novel_pseudo"

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
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "/home/gsarch/repo/pytorch_disco/logs_novel/model_0019999.pth"

        # Get coco dataset metadata
        thing_classes = ['beanbag', 'cushion', 'nightstand', 'shelf']

        # register dataset, thing_classes same as coco thing_classes
        d = "train"
        DatasetCatalog.register("multiview_novel_selfsup_train", lambda d=d: train_dataset_function())
        MetadataCatalog.get("multiview_novel_selfsup_train").thing_classes = thing_classes

        DatasetCatalog.register("multiview_novel_selfsup_val", lambda d=d: val_dataset_function())
        MetadataCatalog.get("multiview_novel_selfsup_val").thing_classes = thing_classes
        cfg.DATASETS.TEST = ("multiview_novel_selfsup_val",) 

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
        if feed['data_ind'] < 0:   # iter where it fails
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
        # bottle          14              39
        # clock           22              74
        refrigerator    67              72
        tv(tv-screen)   87              62
        # vase            91              75
        '''
        self.maskrcnn_to_catname = {0: "beanbag", 1: "cushion", 2: "nightstand", 3: "shelf"}
        self.replica_to_maskrcnn = {6:0, 29:1, 54:2, 71:3}

        self.category_ids_camXs = feed['category_ids_camXs']
        self.object_category_names = feed['category_names_camXs']
        self.bbox_2d_camXs = feed['bbox_2d_camXs']
        self.mask_2d_camXs = feed['mask_2d_camXs']

        '''
        for idx in range(len(feed['object_category_ids'])):
            catid = feed['object_category_ids'][idx].item()
            if catid in self.replica_to_maskrcnn:
                # append object ids
                self.object_category_ids.append(self.replica_to_maskrcnn[catid])
                # get object bounding boxes. corners_origin (3,2), vertices_origin(1,8,3)
                
                # corners_origin = feed['bbox_origin'][0,idx].reshape(2,3) # 2=min&max, 3=xyz
                # vertices_origin = torch.stack([corners_origin[[0,0,0],[0,1,2]],
                #     corners_origin[[0,0,1],[0,1,2]],
                #     corners_origin[[0,1,0],[0,1,2]],
                #     corners_origin[[0,1,1],[0,1,2]],
                #     corners_origin[[1,0,0],[0,1,2]],
                #     corners_origin[[1,0,1],[0,1,2]],
                #     corners_origin[[1,1,0],[0,1,2]],
                #     corners_origin[[1,1,1],[0,1,2]]]).unsqueeze(0)
                # vertices_origin = vertices_origin - torch.Tensor([0, 1.5, 0]).reshape(1, 1, 3).cuda()

                self.object_bboxes_2d.append(feed['box2d'][idx])
                self.object_masks_2d.append(feed['mask_2d'][idx])
                # self.object_bboxes_origin.append(vertices_origin) # [(1,8,3)]
        '''

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

                    y, x = torch.where(pred_masks[segs])
                    if len(y) == 0:
                        continue
                    obj_all_catids.append(pred_classes[segs].item())
                    obj_all_scores.append(pred_scores[segs].item())
                    pred_box = torch.Tensor([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                    #pred_box = pred_boxes[segs][[1,0,3,2]]
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

        # List of classlist_g, boxlist_g
        self.classlist_g_s = []
        self.boxlist_g_s = []

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
            for i in range(self.bbox_2d_camXs[s].shape[0]):
                ymin, xmin, ymax, xmax = obj_bboxes_y_min[i], obj_bboxes_x_min[i], obj_bboxes_y_max[i], obj_bboxes_x_max[i]
                if ymin == ymax or xmin == xmax or xmin>self.W or xmax<0 or ymin>self.W or ymax<0:
                    # remove empty boxes or boxes out of the current view
                    continue

                classlist_g.append(self.replica_to_maskrcnn[self.category_ids_camXs[s][i].item()])
                box = np.array([ymin, xmin, ymax, xmax])
                boxlist_g.append(box)

                box_norm = np.array([ymin/self.H, xmin/self.W, ymax/self.H, xmax/self.W])
                boxlist_g_norm.append(box_norm)

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
                if boxlist_g.shape[1] > 0:
                    self.summ_writer.summ_boxlist2d('finals/boxes_{}_gt'.format(s), self.rgb_camXs[:,s], boxlist_g)
                if boxlist_e_maskrcnn.shape[1] > 0:
                    self.summ_writer.summ_boxlist2d('finals/boxes_{}_maskrcnn'.format(s), self.rgb_camXs[:,s], boxlist_e_maskrcnn)

                # To numpy
                boxlist_g = boxlist_g.squeeze(0).cpu().numpy()
                boxlist_e_maskrcnn = boxlist_e_maskrcnn.squeeze(0).cpu().numpy()

                gt_file = open(f"{output_gt_dir}/{self.img_count}.txt", 'w')
                maskrcnn_file = open(f"{output_maskrcnn_dir}/{self.img_count}.txt", 'w')

                for i in range(len(boxlist_g)):
                    boxlist_g[i][0] *= self.H #ymin
                    boxlist_g[i][1] *= self.W #xmin
                    boxlist_g[i][2] *= self.H #ymax
                    boxlist_g[i][3] *= self.W #xmax
                    gt_file.write(f"{self.maskrcnn_to_catname[classlist_g[i]]} {round(boxlist_g[i][1])} {round(boxlist_g[i][0])} {round(boxlist_g[i][3])} {round(boxlist_g[i][2])}\n")
                gt_file.close()

                # getting class labels as text
                for i in range(len(boxlist_e_maskrcnn)):
                    boxlist_e_maskrcnn[i][0] *= self.H #ymin
                    boxlist_e_maskrcnn[i][1] *= self.W #xmin
                    boxlist_e_maskrcnn[i][2] *= self.H #ymax
                    boxlist_e_maskrcnn[i][3] *= self.W #xmax
                    maskrcnn_file.write(f"{self.maskrcnn_to_catname[int(class_list_e_maskrcnn[i])]} 1 {round(boxlist_e_maskrcnn[i][1])} {round(boxlist_e_maskrcnn[i][0])} {round(boxlist_e_maskrcnn[i][3])} {round(boxlist_e_maskrcnn[i][2])}\n")
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

