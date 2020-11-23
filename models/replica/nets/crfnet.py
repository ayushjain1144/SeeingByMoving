import torch
import torch.nn as nn

import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder2d
import utils.geom
import utils.vox
import utils.misc
import utils.basic
import torch.nn.functional as F
import ipdb
st = ipdb.set_trace
import time

import numpy as np 
import scipy
import cc3d
import math

import pydensecrf.densecrf as dcrf 
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral
import cv2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.ensemble import BalancedBaggingClassifier

class CrfNet(nn.Module):
    def __init__(self):
        super().__init__()

    def get_prob(self, grid, fg_mask, bg_mask):
        '''
        Inputs:
            grid: (X,Y,Z,C) np array
            fg_mask: (X,Y,Z) np array, binary
            bg_mask: (X,Y,Z) np array, binary
        Returns:
            fg_prob: (X,Y,Z) np array, float in (0,1)
            bg_prob: (X,Y,Z) np array, float in (0,1)
        '''
        grid = grid / (1e-8+grid.sum(-1)[:, np.newaxis])

        fg_idx = np.nonzero(fg_mask)
        bg_idx = np.nonzero(bg_mask)
        X, Y, Z = grid.shape[:3]

        # compute average
        fg_avg = grid[fg_idx[0], fg_idx[1], fg_idx[2]].mean(0)
        # fg_avg = fg_avg / fg_avg.sum()
        bg_avg = grid[bg_idx[0], bg_idx[1], bg_idx[2]].mean(0)
        # bg_avg = bg_avg / bg_avg.sum()

        fg_prob = np.zeros((X,Y,Z))
        bg_prob = np.zeros((X,Y,Z))

        fg_prob = 0.5 + 0.5*np.dot(grid, fg_avg)/(np.linalg.norm(grid, axis=3)*np.linalg.norm(fg_avg))
        bg_prob = 0.5 + 0.5*np.dot(grid, bg_avg)/(np.linalg.norm(grid, axis=3)*np.linalg.norm(bg_avg))

        # fg_prob = np.exp(np.dot(grid, fg_avg)/(np.linalg.norm(grid, axis=3)*np.linalg.norm(fg_avg)))
        # bg_prob = np.exp(np.dot(grid, bg_avg)/(np.linalg.norm(grid, axis=3)*np.linalg.norm(bg_avg)))

        fg_prob[fg_idx[0], fg_idx[1], fg_idx[2]] = 1
        fg_prob[bg_idx[0], bg_idx[1], bg_idx[2]] = 0

        bg_prob[fg_idx[0], fg_idx[1], fg_idx[2]] = 0
        bg_prob[bg_idx[0], bg_idx[1], bg_idx[2]] = 1

        return fg_prob, bg_prob

    def forward(self, multiview_pc, occ_agg, fg_seed, bg_seed, summ_writer=None, is_feature=True):

        # simplify code for B =  1 assumption
        B, C, _, _, _ = multiview_pc.shape
        assert B == 1, 'batch size should be 1'
        
        # getting rid of batch dim
        multiview_pc = multiview_pc.squeeze(0)

        fg_seed = fg_seed.squeeze(0).squeeze(0)
        bg_seed = bg_seed.squeeze(0).squeeze(0)
        #print('fg_seed shape', fg_seed.shape) #  V X V X V
        
        feat_grid = multiview_pc.cpu().numpy()
        occ_agg =occ_agg.squeeze(0).squeeze(0).cpu().numpy()

        if is_feature:
            feat_grid = np.transpose(feat_grid, (1,2,3,0))
        else:
            feat_grid = np.transpose(feat_grid, (1,2,3,0)) + 0.5

        fg_mask_grid = fg_seed.cpu().numpy().astype(np.uint8)
        bg_mask_grid = bg_seed.cpu().numpy().astype(np.uint8)

        # Crop around pointcloud mean
        extent = 4
        fg_idx = np.array(np.nonzero(fg_mask_grid))
        #print(fg_idx.shape)
        box_center = np.round(np.mean(fg_idx,axis=1)).astype(int)
        box_width = np.round(np.std(fg_idx,axis=1)*extent).astype(int)

        xmin = box_center[0]-box_width[0]
        xmax = box_center[0]+box_width[0]
        ymin = box_center[1]-box_width[1]*2
        ymax = box_center[1]+box_width[1]*2
        zmin = box_center[2]-box_width[2]
        zmax = box_center[2]+box_width[2]
        
        xmin, ymin, zmin = max(xmin,0), max(ymin,0), max(zmin,0)
        xmax, ymax, zmax = min(xmax, feat_grid.shape[0]), min(ymax, feat_grid.shape[1]), min(zmax, feat_grid.shape[2])

        xmin, zmin = 0, 0
        xmax, zmax = feat_grid.shape[0], feat_grid.shape[2]
        
        # try:
        feat_grid_crop = feat_grid[xmin:xmax, ymin:ymax, zmin:zmax]
        fg_mask_grid_crop = fg_mask_grid[xmin:xmax, ymin:ymax, zmin:zmax]
        bg_mask_grid_crop = bg_mask_grid[xmin:xmax, ymin:ymax, zmin:zmax]
        occ_agg = occ_agg[xmin:xmax, ymin:ymax, zmin:zmax]

        # Keep only the connected component closest to median as the foreground mask
        # fg_idx_crop = np.array(np.nonzero(fg_mask_grid_crop))
        # median_idx = fg_idx_crop[:, fg_idx.shape[1]//2]
        # labels_out = cc3d.connected_components(fg_mask_grid_crop, connectivity=26)
        # min_dist = 1e8
        # min_label_id = -1
        # for label_id in range(1, np.max(labels_out)+1):
        #     fg_label_idx = np.array(np.where(labels_out == label_id))
        #     centroid_idx = np.mean(fg_label_idx, axis=-1)
        #     dist = np.linalg.norm(median_idx - centroid_idx)
        #     if dist < min_dist:
        #         min_dist = dist
        #         min_label_id = label_id
        # fg_mask_grid_crop *= (labels_out == min_label_id)

        # get fg and bg unaries
        fg_prob_grid_crop, bg_prob_grid_crop = self.get_prob(feat_grid_crop, fg_mask_grid_crop, bg_mask_grid_crop)
        unary = np.stack([fg_prob_grid_crop, bg_prob_grid_crop], 0).reshape(2, -1)
        unary /= unary.sum(0)
        unary = -np.log(unary+1e-8)

        # get pairwise potentials (gaussian and bilateral)
        p_gaussian = create_pairwise_gaussian([3,3,3], fg_prob_grid_crop.shape)
        p_bilateral = create_pairwise_bilateral([30,30,30], [13,13,13], feat_grid_crop, chdim=3)

        d = dcrf.DenseCRF(np.prod(fg_prob_grid_crop.shape), 2)
        d.setUnaryEnergy(unary)
        #d.addPairwiseEnergy(p_gaussian,3)
        #d.addPairwiseEnergy(p_bilateral,10)

        # Perform inference and condition output with occupancy
        Q = d.inference(10)
        xdim, ydim, zdim = fg_prob_grid_crop.shape
        MAP = 1 - np.argmax(Q, axis=0).reshape(xdim, ydim, zdim)
        MAP *= occ_agg.astype(int)

        MAP_raw = MAP.copy()

        # keep only the connected components connected to the fg mask
        labels_out = cc3d.connected_components(MAP, connectivity=26)
        for label_id in range(1, np.max(labels_out)+1):
            current_binary = (labels_out == label_id)
            if np.sum(np.logical_and(current_binary, fg_mask_grid_crop)) == 0:
                MAP[labels_out == label_id] = 0

        is_success = True

        bounds = (xmin, xmax,
                  ymin, ymax,
                  zmin, zmax)

        # except Exception as e:
        #     print("crfnet failed...")
        #     is_success = False
        #     return _, _, _, _, is_success

        return MAP_raw, MAP, bounds, bg_mask_grid_crop, fg_mask_grid_crop, is_success
        

class CrfflatNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))

    def get_prob(self, feat_pc, xyz_pc, fg_mask, bg_mask):
        '''
        Inputs:
            feat_pc: (N,C) np array
            xyz_pc: (N,3) np array
            fg_mask: (N) np array, binary
            bg_mask: (N) np array, binary
        Returns:
            unary: (2,N) np array
        '''
        # grid = grid / (1e-8+grid.sum(-1)[:, np.newaxis])

        np.random.seed(42)

        # Get fg bg rgb and xyz according to indices
        fg_idx = np.nonzero(fg_mask)
        bg_idx = np.nonzero(bg_mask) # (N_fg, C)
        fg_feat = feat_pc[fg_idx[0]]
        fg_xyz = xyz_pc[fg_idx[0]]
        bg_feat = feat_pc[bg_idx[0]]
        bg_xyz = xyz_pc[bg_idx[0]]

        # sort fg (descending) and bg (ascending) according to distance to fg center
        fg_center = np.mean(fg_xyz, axis=0)

        bg_sorted_idx = sorted(range(bg_feat.shape[0]), key=lambda k: np.sum((bg_xyz[k]-fg_center)**2))
        bg_feat = bg_feat[bg_sorted_idx]
        bg_xyz = bg_xyz[bg_sorted_idx]-fg_center

        fg_sorted_idx = sorted(range(fg_feat.shape[0]), reverse=True, key=lambda k: np.sum((fg_xyz[k]-fg_center)**2))
        fg_feat = fg_feat[fg_sorted_idx]
        fg_xyz = fg_xyz[fg_sorted_idx] - fg_center

        # Concate to get feature
        fg_feature = np.concatenate([fg_feat, fg_xyz], axis=1)
        bg_feature = np.concatenate([bg_feat, bg_xyz], axis=1)

        # SVM
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))

        # balance the two classes a little bit
        # also to speed up svm training
        fg_max_feat = 300
        if fg_feature.shape[0] > fg_max_feat:
            far_thresh = fg_feature.shape[0] // 5
            if far_thresh <= fg_max_feat // 2:
                random_indices_far = np.arange(0, far_thresh)
            else:
                random_indices_far = np.random.choice(far_thresh, fg_max_feat//2)
            random_indices_close = far_thresh + np.random.choice(fg_feature.shape[0] - far_thresh, fg_max_feat//2)
            random_indices = np.concatenate([random_indices_far, random_indices_close])
            fg_feature = fg_feature[random_indices,:]
            fg_xyz = fg_xyz[random_indices,:]
            # shuffler = np.random.permutation(fg_feature.shape[0])
            # fg_feature = fg_feature[shuffler]
            # random_indices = np.random.choice(fg_feature.shape[0], fg_max_feat)
            # fg_feature = fg_feature[random_indices,:]
            # fg_xyz = fg_xyz[random_indices,:]
        else:
            fg_max_feat = fg_feature.shape[0]

        bg_max_feat = fg_max_feat
        if bg_feature.shape[0] > bg_max_feat:
            close_thresh = bg_feature.shape[0] // 5 
            random_indices_close = np.random.choice(close_thresh, bg_max_feat//2)
            random_indices_far = close_thresh + np.random.choice(bg_feature.shape[0] - close_thresh, bg_max_feat//2)
            random_indices = np.concatenate([random_indices_close, random_indices_far])
            bg_feature = bg_feature[random_indices,:]
            bg_xyz = bg_xyz[random_indices,:]

        X = np.concatenate([fg_feature, bg_feature], axis=0)
        y = np.concatenate([np.ones(fg_feature.shape[0]), np.zeros(bg_feature.shape[0])])

        self.clf.fit(X,y)

        grid = np.concatenate([feat_pc, xyz_pc-fg_center], axis=1)
        prob = self.clf.predict_log_proba(grid)
        unary = -prob

        unary[fg_idx[0],:] = -np.log([0.01, 0.99])
        unary[bg_idx[0],:] = -np.log([0.99, 0.01])

        unary = unary.T

        return unary, fg_xyz+fg_center, bg_xyz+fg_center, fg_center

    def get_pairwise_gaussian(self, s, xyz_pc):   
        '''
        Inputs
            s: integer scaling factor
            xyz_pc: (N,3) xyz positions
        Returns:
            p_gaussian transpose: (3,N)
        '''
        p_gaussian = np.zeros((xyz_pc.shape[1], xyz_pc.shape[0]))
        xyz_pc_min, xyz_pc_max = np.min(xyz_pc, axis=0), np.max(xyz_pc, axis=0)
        pp = (xyz_pc - xyz_pc_min) / ((xyz_pc_max - xyz_pc_min) * s)
        pp = pp.transpose()

        for i in range(p_gaussian.shape[0]):
            p_gaussian[i, :] == pp[i,:]

        return p_gaussian

    def get_pairwise_bilateral(self, s1, s2, xyz_pc, feat_pc):
        '''
        Inputs
            s1: scaling factor for xyz
            s2: scaling factor for 
            xyz_pc: (N,3) xyz positions
            feat_pc: (N,C) feature
        Returns:
            p_bilateral: (3+C,N)
        '''
        xyz_pc_min, xyz_pc_max = np.min(xyz_pc, axis=0), np.max(xyz_pc, axis=0)
        p_bilateral = np.zeros((xyz_pc.shape[1]+feat_pc.shape[1], feat_pc.shape[0]))

        p_xyz = (xyz_pc - xyz_pc_min) / ((xyz_pc_max-xyz_pc_min) * s1)
        p_xyz = p_xyz.transpose() # (3,N)

        p_feat = feat_pc / s2
        p_feat = p_feat.transpose() #(C,N)

        p_bilateral[:p_xyz.shape[0]] = p_xyz
        p_bilateral[p_xyz.shape[0]:] = p_feat

        return p_bilateral

    def forward(self, feat_pc, xyz_pc, fg_seed, bg_seed, summ_writer=None, is_feature=True):
        '''
        Inputs:
            feat_pc: (N,C)
            xyz_pc: (N,3)
            fg_seed: (N,)
            bg_seed: (N,)
        '''
        
        feat_pc = feat_pc.cpu().numpy()
        xyz_pc = xyz_pc.cpu().numpy()

        if not is_feature:
            # rectify rgb
            feat_pc += 0.5

        fg_mask = fg_seed.cpu().numpy().astype(np.uint8)
        bg_mask = bg_seed.cpu().numpy().astype(np.uint8)

        # get unary
        t0 = time.time()
        unary, fg_xyz, bg_xyz, fg_center = self.get_prob(feat_pc, xyz_pc, fg_mask, bg_mask)
        unary, fg_xyz, bg_xyz = unary.astype(np.float32), fg_xyz.astype(np.float32), bg_xyz.astype(np.float32)
        print("unary time:", time.time() - t0)

        # get pairwise potentials (gaussian and bilateral)
        t0 = time.time()
        p_gaussian = self.get_pairwise_gaussian(1, xyz_pc-fg_center).astype(np.float32)
        if is_feature:
            p_bilateral = self.get_pairwise_bilateral(1, 10, xyz_pc, feat_pc).astype(np.float32)
        else:
            p_bilateral = self.get_pairwise_bilateral(1, 1, xyz_pc-fg_center, feat_pc).astype(np.float32)
        print("binary time:", time.time() - t0)

        t0 = time.time()
        d = dcrf.DenseCRF(unary.shape[1], 2)
        d.setUnaryEnergy(unary.copy(order='C'))
        d.addPairwiseEnergy(p_gaussian,3)
        # d.addPairwiseEnergy(p_bilateral,1)
        print("setting time:", time.time() - t0)

        # Perform inference and condition output with occupancy
        t0 = time.time()
        Q = d.inference(5)
        MAP = np.argmax(Q, axis=0).reshape(-1)
        print("drcf time:", time.time() - t0)


        is_success = True



        return MAP, fg_xyz, bg_xyz, is_success

