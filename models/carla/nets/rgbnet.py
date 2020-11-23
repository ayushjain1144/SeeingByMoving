import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.pixelshuffle3d
import hyperparams as hyp
import utils.improc
import utils.misc
import utils.basic

class RgbNet(nn.Module):
    def __init__(self):
        super(RgbNet, self).__init__()

        print('RgbNet...')

        # self.conv3d = nn.Conv3d(in_channels=hyp.feat_dim, out_channels=8, kernel_size=1, stride=1, padding=0).cuda()
        # # self.unpack = nn.PixelShuffle(9)
        # self.unpack = archs.pixelshuffle3d.PixelShuffle3d(2)
        
        # self.conv3d = nn.Conv3d(in_channels=int(hyp.feat_dim/2), out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
        self.conv3d = nn.Conv3d(in_channels=hyp.feat3d_dim, out_channels=3, kernel_size=1, stride=1, padding=0).cuda()

        # self.conv3d = nn.ConvTranspose3d(hyp.feat_dim, 1, kernel_size=4, stride=2, padding=1, bias=False).cuda()
        
    def forward(self, feat, rgb_g=None, valid=None, occ_e=None, occ_g=None, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        rgb_e = self.conv3d(feat)
        # rgb_e = F.sigmoid(feat) - 0.5
        # rgb_e is B x 3 x Z x Y x X
        
        # smooth loss
        dz, dy, dx = utils.basic.gradient3d(rgb_e, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        if valid is not None:
            smooth_loss = utils.basic.reduce_masked_mean(smooth_vox, valid)
        else:
            smooth_loss = torch.mean(smooth_vox)
        total_loss = utils.misc.add_loss('rgb/smooth_loss%s' % suffix, total_loss, smooth_loss, hyp.rgb_smooth_coeff, summ_writer)
    
        if rgb_g is not None:
            loss_im = utils.basic.l1_on_axis(rgb_e-rgb_g, 1, keepdim=True)
            if valid is not None:
                rgb_loss = utils.basic.reduce_masked_mean(loss_im, valid)
            total_loss = utils.misc.add_loss('rgb/rgb_l1_loss', total_loss, rgb_loss, hyp.rgb_l1_coeff, summ_writer)

        if summ_writer is not None:
            if occ_e is not None:
                summ_writer.summ_unp('rgb/rgb_e', rgb_e, occ_e)
            if occ_g is not None:
                summ_writer.summ_unp('rgb/rgb_g', rgb_g, occ_g)
            summ_writer.summ_oned('rgb/smooth_loss%s' % suffix, torch.mean(smooth_vox, dim=3))

        return total_loss, rgb_e

