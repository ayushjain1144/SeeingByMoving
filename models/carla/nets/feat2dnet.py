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

EPS = 1e-4
class Feat2dNet(nn.Module):
    def __init__(self, in_dim=3):
        super(Feat2dNet, self).__init__()
        
        # self.net = archs.encoder2d.Net2d(in_dim, 64, hyp.feat2d_dim).cuda()
        self.net = archs.encoder2d.Encoder2d(in_dim, 64, hyp.feat2d_dim).cuda()
        
        print(self.net)

    def forward(self, rgb, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, H, W = list(rgb.shape)

        if summ_writer is not None:
            summ_writer.summ_rgb('feat2d/rgb', rgb)

        print('rgb', rgb.shape)
        feat = self.net(rgb)
        print('feat', feat.shape)

        # smooth loss
        dy, dx = utils.basic.gradient2d(feat, absolute=True)
        smooth_im = torch.mean(dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('feat2d/smooth_loss', smooth_im)
        smooth_loss = torch.mean(smooth_im)
        total_loss = utils.misc.add_loss('feat2d/smooth_loss', total_loss, smooth_loss, hyp.feat2d_smooth_coeff, summ_writer)

        feat = utils.basic.l2_normalize(feat, dim=1)
        
        if summ_writer is not None:
            summ_writer.summ_feat('feat2d/feat_output', feat, pca=True)
        
        return total_loss, feat

