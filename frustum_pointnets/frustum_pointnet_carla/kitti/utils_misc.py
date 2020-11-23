import random
import tensorboardX
import torch
import torch.nn as nn
import numpy as np
import utils_vox
import utils_improc
import utils_geom
import utils_basic
import utils_samp
import imageio
import cv2
import hyperparams as hyp

def add_loss(name, total_loss, loss, coeff, summ_writer):
    # summ_writer should be Summ_writer object in utils_improc
    summ_writer.summ_scalar('unscaled_%s' % name, loss)
    summ_writer.summ_scalar('scaled_%s' % name, coeff*loss)

    total_loss = total_loss + coeff*loss
    return total_loss

# some code from: https://github.com/suruoxi/DistanceWeightedSampling
class MarginLoss(nn.Module):
    def __init__(self, margin=0.2, nu=0.0, weight=None, batch_axis=0, **kwargs):
        super(MarginLoss, self).__init__()
        self._margin = margin
        self._nu = nu

    def forward(self, anchors, positives, negatives, beta, a_indices=None):
        d_ap = torch.sqrt(torch.sum((positives - anchors)**2, dim=1) + 1e-8)
        d_an = torch.sqrt(torch.sum((negatives - anchors)**2, dim=1) + 1e-8)

        pos_loss = torch.clamp(d_ap - beta + self._margin, min=0.0)
        neg_loss = torch.clamp(beta - d_an + self._margin, min=0.0)

        pair_cnt = int(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)))

        loss = torch.sum(pos_loss + neg_loss) / (1e-4 + pair_cnt)
        return loss, pair_cnt

class DistanceWeightedSampling(nn.Module):
    '''
    parameters
    ----------
    batch_k: int
        number of images per class

    Inputs:
        data: input tensor with shape (batch_size, edbed_dim)
            Here we assume the consecutive batch_k examples are of the same class.
            For example, if batch_k = 5, the first 5 examples belong to the same class,
            6th-10th examples belong to another class, etc.
    Outputs:
        a_indices: indicess of anchors
        x[a_indices]
        x[p_indices]
        x[n_indices]
        xxx

    '''

    def __init__(self, batch_k, cutoff=0.5, nonzero_loss_cutoff=1.4, normalize=False, **kwargs):
        super(DistanceWeightedSampling,self).__init__()
        self.batch_k = batch_k
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.normalize = normalize
        
    def get_distance(self, x):
        square = torch.sum(x**2, dim=1, keepdims=True)
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        return torch.sqrt(distance_square + torch.eye(x.shape[0], device=torch.device('cuda')))

    def forward(self, x):
        k = self.batch_k
        n, d = x.shape

        debug = False
        # debug = True
        if debug:
            np.set_printoptions(precision=3, suppress=True)
            print(x[:,:5])
            print(x.shape)
        
        distance = self.get_distance(x)
        
        distance = torch.clamp(distance, min=self.cutoff)
        if debug:
            print('distance:')#, end=' ')
            print(distance.detach().cpu().numpy())

        log_weights = ((2.0 - float(d)) * torch.log(distance)
                       - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))

        if debug:
            print('log_weights:')#, end=' ')
            print(log_weights.detach().cpu().numpy())
        
        weights = torch.exp(log_weights - torch.max(log_weights))

        if debug:
            print('weights:')#, end=' ')
            print(weights.detach().cpu().numpy())

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones_like(weights)
        for i in list(range(0,n,k)):
            mask[i:i+k, i:i+k] = 0
            
        if debug:
            print('mask:')#, end=' ')
            print(mask.detach().cpu().numpy())
            print('dist < nonzero:')#, end=' ')
            print((distance < self.nonzero_loss_cutoff).float().detach().cpu().numpy())

        # let's eliminate nans and zeros immediately
        weights[torch.isnan(weights)] = 1.0
        weights[weights < 1e-2] = 1e-2

        weights = weights * mask * (distance < self.nonzero_loss_cutoff).float()
        if debug:
            print('masked weights:')#, end=' ')
            print(weights.detach().cpu().numpy())
        
        weights = weights.detach().cpu().numpy()

        if debug:
            print('np weights:')#, end=' ')
            print(weights)
        
        # weights[np.isnan(weights)] = 1.0
        # weights[weights < 1e-2] = 1e-2

        if debug:
            print('clean weights:')#, end=' ')
            print(weights)

        # careful divison here
        weights = weights / (1e-4 + np.sum(weights, axis=1, keepdims=True))
            
        if debug:
            print('new weights:')#, end=' ')
            # print(weights.detach().cpu().numpy())
            print(weights)
        
        a_indices = []
        p_indices = []
        n_indices = []

        # np_weights = weights.cpu().detach().numpy()
        np_weights = weights
        for i in list(range(n)):
            block_idx = i // k
            try:
                n_indices += np.random.choice(n, k-1, p=np_weights[i]).tolist()
            except:
                n_indices += np.random.choice(n, k-1).tolist()

            for j in list(range(block_idx * k, (block_idx + 1)*k)):
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return a_indices, x[a_indices], x[p_indices], x[n_indices], x

def shuffle_valid_and_sink_invalid_boxes(boxes, tids, scores):
    # put the good boxes shuffled at the top;
    # sink the bad boxes to the bottom.

    # boxes are B x N x D
    # tids are B x N
    # scores are B x N
    B, N, D = list(boxes.shape)

    boxes_new = torch.zeros_like(boxes)
    tids_new = -1*torch.ones_like(tids)
    scores_new = torch.zeros_like(scores)

    for b in list(range(B)):

        # for the sake of training,
        # we want to mix up the ordering
        inds = list(range(N))
        np.random.shuffle(inds)

        boxes[b] = boxes[b,inds]
        scores[b] = scores[b,inds]
        tids[b] = tids[b,inds]
        
        inds = np.argsort(-1.0*scores[b].cpu().detach().numpy()) # descending
        inds = np.squeeze(inds)

        print(inds)

        boxes_new[b] = boxes[b,inds]
        scores_new[b] = scores[b,inds]
        tids_new[b] = tids[b,inds]

        # print('ok, boxes old and new')
        # print(boxes[b])
        # print(boxes_new[b])
        # input()

    return boxes_new, tids_new, scores_new

def get_target_scored_box_single(target, boxlist, tidlist, scorelist, vislist):
    # boxlist is N x D
    # tidlist is N and int32
    # scorelist is N
    # vislist is N
    # here we retrieve one target box
    N, D = list(boxlist.shape)
    
    box_ = torch.ones(D) # use ones here, to avoid nans later
    score_ = torch.zeros(1)
    vis_ = torch.zeros(1)
    # print 'target = %d' % (target),

    count = 0
    for i in list(range(N)):
        box = boxlist[i]
        tid = tidlist[i]
        score = scorelist[i]
        vis = vislist[i]
        # print 'target = %d; tid = %d; score = %.2f' % (target, tid, score)
        # if score > 0.0 and tid==target:
        if tid==target:
            # print 'got it:',
            # print box,
            # print score
            return box, score, vis
    # did not find it; return empty stuff (with score=0, vis=0)
    return box_, score_, vis_

def get_target_traj(targets, boxlist_s, tidlist_s, scorelist_s, vislist_s):
    # targets is B
    # boxlist_s is B x S x N x D
    # tidlist_s is B x S x N
    # scorelist_s is B x S x N
    # scorelist_s is B x S x N
    
    B, S, N, D = list(boxlist_s.shape)
    # (no asserts on shape; boxlist could instead be lrtlist)

    # return box_traj for the target, sized B x S x D
    # and also the score_traj, sized B x S
    # (note the object may not live across all frames)

    box_traj = torch.zeros(B, S, D)
    score_traj = torch.zeros(B, S)
    vis_traj = torch.zeros(B, S)
    for b in list(range(B)):
        for s in list(range(S)):
            box_, score_, vis_ = get_target_scored_box_single(
                targets[b], boxlist_s[b,s], tidlist_s[b,s], scorelist_s[b,s], vislist_s[b,s])
            box_traj[b,s] = box_
            score_traj[b,s] = score_
            vis_traj[b,s] = vis_
    return box_traj.cuda(), score_traj.cuda(), vis_traj.cuda()

def collect_object_info(lrtlist_camRs, tidlist_s, scorelist_s, vislist_s, mod='', do_vis=True, summ_writer=None):
    # lrtlist_camRs is B x S x N x 19
    # tidlist_s is B x S x N
    # scorelist_s is B x S x N
    
    B, S, N, D = list(lrtlist_camRs.shape)
    
    # this returns a bunch of tensors that begin with dim N
    # these tensors are object-centric: along S is all the info for that particular obj
    # this is in contrast to something like boxes, which is frame-centric
    
    obj_lrt_traj = torch.zeros([N, B, S, 19], dtype=torch.float32, device=torch.device('cuda'))
    obj_score_traj = torch.zeros([N, B, S], dtype=torch.float32, device=torch.device('cuda'))
    obj_vis_traj = torch.zeros([N, B, S], dtype=torch.float32, device=torch.device('cuda'))

    # get the objects in random order
    inds = list(range(N))
    np.random.shuffle(inds)
    # print('inds order:', inds)
    
    count = 0
    for target_ind in inds:
        target_tid = tidlist_s[:,0,target_ind]
        tid_traj = torch.reshape(target_tid, [B, 1]).repeat(1, S)

        # extract its traj from the full tensors
        lrt_traj, score_traj, vis_traj = get_target_traj(target_tid, lrtlist_camRs, tidlist_s, scorelist_s, vislist_s)
        # lrt_traj is B x S x 19
        # score_traj is B x S
        # vis_traj is B x S

        if torch.sum(vis_traj) > 0:
            # print('ok, using tid %d in position %d' % (target_tid.detach().cpu().numpy(), count))
            obj_lrt_traj[count] = lrt_traj
            obj_score_traj[count] = score_traj
            obj_vis_traj[count] = vis_traj
            count = count + 1
            
    return obj_lrt_traj, obj_score_traj, obj_vis_traj

def rescore_boxlist_with_inbound(camX_T_camR, boxlist_camR, tidlist, Z, Y, X, only_cars=True, pad=2.0):
    # boxlist_camR is B x N x 9
    B, N, D = list(boxlist_camR.shape)
    assert(D==9)
    xyzlist = boxlist_camR[:,:,:3]
    # this is B x N x 3
    lenlist = boxlist_camR[:,:,3:7]
    # this is B x N x 3

    xyzlist = utils_geom.apply_4x4(camX_T_camR, xyzlist)
    
    validlist = 1.0-(torch.eq(tidlist, -1*torch.ones_like(tidlist))).float()
    # this is B x N
    
    if only_cars:
        biglist = (torch.norm(lenlist, dim=2) > 2.0).float()
        validlist = validlist * biglist

    xlist, ylist, zlist = torch.unbind(xyzlist, dim=2)
    inboundlist_0 = utils_vox.get_inbounds(torch.stack([xlist+pad, ylist, zlist], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_1 = utils_vox.get_inbounds(torch.stack([xlist-pad, ylist, zlist], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_2 = utils_vox.get_inbounds(torch.stack([xlist, ylist, zlist+pad], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist_3 = utils_vox.get_inbounds(torch.stack([xlist, ylist, zlist-pad], dim=2), Z, Y, X, already_mem=False).float()
    inboundlist = inboundlist_0*inboundlist_1*inboundlist_2*inboundlist_3
    scorelist = validlist * inboundlist
    return scorelist

def rescore_boxlist_with_pointcloud(camX_T_camR, boxlist_camR, xyz_camX, scorelist, tidlist, thresh=1.0):
    # boxlist_camR is B x N x 9
    B, N, D = list(boxlist_camR.shape)
    assert(D==9)
    xyzlist = boxlist_camR[:,:,:3]
    # this is B x N x 3
    lenlist = boxlist_camR[:,:,3:7]
    # this is B x N x 3


    xyzlist = utils_geom.apply_4x4(camX_T_camR, xyzlist)

    # xyz_camX is B x V x 3
    xyz_camX = xyz_camX[:,::10]
    xyz_camX = xyz_camX.unsqueeze(1)
    # xyz_camX is B x 1 x V x 3
    xyzlist = xyzlist.unsqueeze(2)
    # xyzlist is B x N x 1 x 3

    dists = torch.norm(xyz_camX - xyzlist, dim=3)
    # this is B x N x V

    mindists = torch.min(dists, 2)[0]
    ok = (mindists < thresh).float()
    scorelist = scorelist * ok
    return scorelist


def get_gt_flow(obj_lrtlist_camRs,
                obj_scorelist,
                camRs_T_camXs,
                Z, Y, X, 
                K=2,
                mod='',
                vis=True,
                summ_writer=None):
    # this constructs the flow field according to the given
    # box trajectories (obj_lrtlist_camRs) (collected from a moving camR)
    # and egomotion (encoded in camRs_T_camXs)
    # (so they do not take into account egomotion)
    # so, we first generate the flow for all the objects,
    # then in the background, put the ego flow
    N, B, S, D = list(obj_lrtlist_camRs.shape)
    assert(S==2) # as a flow util, this expects S=2

    flows = []
    masks = []
    for k in list(range(K)):
        obj_masklistR0 = utils_vox.assemble_padded_obj_masklist(
            obj_lrtlist_camRs[k,:,0:1],
            obj_scorelist[k,:,0:1],
            Z, Y, X,
            coeff=1.0)
        # this is B x 1(N) x 1(C) x Z x Y x Z
        # obj_masklistR0 = obj_masklistR0.squeeze(1)
        # this is B x 1 x Z x Y x X
        obj_mask0 = obj_masklistR0.squeeze(1)
        # this is B x 1 x Z x Y x X

        camR_T_cam0 = camRs_T_camXs[:,0]
        camR_T_cam1 = camRs_T_camXs[:,1]
        cam0_T_camR = utils_geom.safe_inverse(camR_T_cam0)
        cam1_T_camR = utils_geom.safe_inverse(camR_T_cam1)
        # camR0_T_camR1 = camR0_T_camRs[:,1]
        # camR1_T_camR0 = utils_geom.safe_inverse(camR0_T_camR1)

        # obj_masklistA1 = utils_vox.apply_4x4_to_vox(camR1_T_camR0, obj_masklistA0)
        # if vis and (summ_writer is not None):
        #     summ_writer.summ_occ('flow/obj%d_maskA0' % k, obj_masklistA0)
        #     summ_writer.summ_occ('flow/obj%d_maskA1' % k, obj_masklistA1)

        if vis and (summ_writer is not None):
            # summ_writer.summ_occ('flow/obj%d_mask0' % k, obj_mask0)
            summ_writer.summ_oned('flow/obj%d_mask0_%s' % (k, mod), torch.mean(obj_mask0, 3))
        
        _, ref_T_objs_list = utils_geom.split_lrtlist(obj_lrtlist_camRs[k])
        # this is B x S x 4 x 4
        ref_T_obj0 = ref_T_objs_list[:,0]
        ref_T_obj1 = ref_T_objs_list[:,1]
        obj0_T_ref = utils_geom.safe_inverse(ref_T_obj0)
        obj1_T_ref = utils_geom.safe_inverse(ref_T_obj1)
        # these are B x 4 x 4
        
        mem_T_ref = utils_vox.get_mem_T_ref(B, Z, Y, X)
        ref_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)

        ref1_T_ref0 = utils_basic.matmul2(ref_T_obj1, obj0_T_ref)
        cam1_T_cam0 = utils_basic.matmul3(cam1_T_camR, ref1_T_ref0, camR_T_cam0)
        mem1_T_mem0 = utils_basic.matmul3(mem_T_ref, cam1_T_cam0, ref_T_mem)

        xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
        xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)

        xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
        xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)

        # only use these displaced points within the obj mask
        # obj_mask03 = obj_mask0.view(B, Z, Y, X, 1).repeat(1, 1, 1, 1, 3)
        obj_mask0 = obj_mask0.view(B, Z, Y, X, 1)
        # # xyz_mem1[(obj_mask03 < 1.0).bool()] = xyz_mem0
        # cond = (obj_mask03 < 1.0).float()
        cond = (obj_mask0 > 0.0).float()
        xyz_mem1 = cond*xyz_mem1 + (1.0-cond)*xyz_mem0

        flow = xyz_mem1 - xyz_mem0
        flow = flow.permute(0, 4, 1, 2, 3)
        obj_mask0 = obj_mask0.permute(0, 4, 1, 2, 3)

        # if vis and k==0:
        if vis:
            summ_writer.summ_3D_flow('flow/gt_%d_%s' % (k, mod), flow, clip=4.0)

        masks.append(obj_mask0)
        flows.append(flow)

    camR_T_cam0 = camRs_T_camXs[:,0]
    camR_T_cam1 = camRs_T_camXs[:,1]
    cam0_T_camR = utils_geom.safe_inverse(camR_T_cam0)
    cam1_T_camR = utils_geom.safe_inverse(camR_T_cam1)

    mem_T_ref = utils_vox.get_mem_T_ref(B, Z, Y, X)
    ref_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)

    cam1_T_cam0 = utils_basic.matmul2(cam1_T_camR, camR_T_cam0)
    mem1_T_mem0 = utils_basic.matmul3(mem_T_ref, cam1_T_cam0, ref_T_mem)

    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
    xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)

    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)

    flow = xyz_mem1 - xyz_mem0
    flow = flow.permute(0, 4, 1, 2, 3)

    bkg_flow = flow

    # allow zero motion in the bkg
    any_mask = torch.max(torch.stack(masks, axis=0), axis=0)[0]
    masks.append(1.0-any_mask)
    flows.append(bkg_flow)

    flows = torch.stack(flows, axis=0)
    masks = torch.stack(masks, axis=0)
    masks = masks.repeat(1, 1, 3, 1, 1, 1)
    flow = utils_basic.reduce_masked_mean(flows, masks, dim=0)

    if vis:
        summ_writer.summ_3D_flow('flow/gt_complete', flow, clip=4.0)

    # flow is shaped B x 3 x D x H x W
    return flow

def get_synth_flow(occs,
                   unps,
                   summ_writer,
                   sometimes_zero=False,
                   do_vis=False):
    B,S,C,Z,Y,X = list(occs.shape)
    assert(S==2,C==1)

    # we do not sample any rotations here, to keep the distribution purely
    # uniform across all translations
    # (rotation ruins this, since the pivot point is at the camera)
    cam1_T_cam0 = [utils_geom.get_random_rt(B, r_amount=0.0, t_amount=1.0), # large motion
                   utils_geom.get_random_rt(B, r_amount=0.0, t_amount=0.1, # small motion
                                            sometimes_zero=sometimes_zero)]
    cam1_T_cam0 = random.sample(cam1_T_cam0, k=1)[0]

    occ0 = occs[:,0]
    unp0 = unps[:,0]
    occ1 = utils_vox.apply_4x4_to_vox(cam1_T_cam0, occ0, binary_feat=True)
    unp1 = utils_vox.apply_4x4_to_vox(cam1_T_cam0, unp0)
    occs = [occ0, occ1]
    unps = [unp0, unp1]

    if do_vis:
        summ_writer.summ_occs('synth/occs', occs)
        summ_writer.summ_unps('synth/unps', unps, occs)
        
    mem_T_cam = utils_vox.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)
    mem1_T_mem0 = utils_basic.matmul3(mem_T_cam, cam1_T_cam0, cam_T_mem)
    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
    xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)
    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)
    flow = xyz_mem1-xyz_mem0
    # this is B x Z x Y x X x 3
    flow = flow.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    if do_vis:
        summ_writer.summ_3D_flow('synth/flow', flow, clip=2.0)

    if do_vis:
        occ0_e = utils_samp.backwarp_using_3D_flow(occ1, flow, binary_feat=True)
        unp0_e = utils_samp.backwarp_using_3D_flow(unp1, flow)
        summ_writer.summ_occs('synth/occs_stab', [occ0, occ0_e])
        summ_writer.summ_unps('synth/unps_stab', [unp0, unp0_e], [occ0, occ0_e])

    occs = torch.stack(occs, dim=1)
    unps = torch.stack(unps, dim=1)

    return occs, unps, flow, cam1_T_cam0

def get_safe_samples(valid, dims, N_to_sample, mode='3D', tol=5.0):
    N, C = list(valid.shape)
    assert(C==1)
    assert(N==np.prod(dims))
    inds, locs, valids = get_safe_samples_py(valid, dims, N_to_sample, mode=mode, tol=tol)
    inds = torch.from_numpy(inds).to('cuda')
    locs = torch.from_numpy(locs).to('cuda')
    valids = torch.from_numpy(valids).to('cuda')
    
    inds = torch.reshape(inds, [N_to_sample, 1])
    inds = inds.long()
    if mode=='3D':
        locs = torch.reshape(locs, [N_to_sample, 3])
    elif mode=='2D':
        locs = torch.reshape(locs, [N_to_sample, 2])
    else:
        assert(False)# choose 3D or 2D please
    locs = locs.float()
    valids = torch.reshape(valids, [N_to_sample])
    valids = valids.float()
    return inds, locs, valids

def get_safe_samples_py(valid, dims, N_to_sample, mode='3D', tol=5.0):
    if mode=='3D':
        Z, Y, X = dims
    elif mode=='2D':
        Y, X = dims
    else:
        assert(False) # please choose 2D or 3D
    valid = valid.detach().cpu()
    valid = np.reshape(valid, [-1])
    N_total = len(valid)
    # assert(N_to_sample < N_total) # otw we need a padding step, and maybe a mask in the loss
    initial_tol = tol

    all_inds = np.arange(N_total)
    # reshape instead of squeeze, in case one or zero come
    valid_inds = all_inds[np.reshape((np.where(valid > 0)), [-1])]
    N_valid = len(valid_inds)
    # print('initial tol = %.2f' % tol)
    # print('N_valid = %d' % N_valid)
    # print('N_to_sample = %d' % N_to_sample)
    if N_to_sample < N_valid:
        # ok we can proceed

        if mode=='3D':
            xyz = utils_basic.gridcloud3D_py(Z, Y, X)
            locs = xyz[np.reshape((np.where(valid > 0)), [-1])]
        elif mode=='2D':
            xy = utils_basic.gridcloud2D_py(Y, X)
            locs = xy[np.reshape((np.where(valid > 0)), [-1])]

        samples_ok = False
        nTries = 0
        while (not samples_ok):
            # print('sample try %d...' % nTries)
            nTries += 1
            sample_inds = np.random.permutation(N_valid).astype(np.int32)[:N_to_sample]
            samples_try = valid_inds[sample_inds]
            locs_try = locs[sample_inds]
            nn_dists = np.zeros([N_to_sample], np.float32)
            samples_ok = True # ok this might work

            for i, loc in enumerate(locs_try):
                # exclude the current samp
                other_locs0 = locs_try[:i]
                other_locs1 = locs_try[i+1:]
                other_locs = np.concatenate([other_locs0, other_locs1], axis=0) 
                dists = np.linalg.norm(
                    np.expand_dims(loc, axis=0).astype(np.float32) - other_locs.astype(np.float32), axis=1)
                mindist = np.min(dists)
                nn_dists[i] = mindist
                if mindist < tol:
                    samples_ok = False
            # ensure we do not get stuck here: every 100 tries, subtract 1px to make it easier
            tol = tol - nTries*0.01
        # print(locs_try)
        if tol < (initial_tol/2.0):
            print('warning: initial_tol = %.2f; final_tol = %.2f' % (initial_tol, tol))
        # utils_basic.print_stats_py('nn_dists_%s' % mode, nn_dists)

        # print('these look ok:')
        # print(samples_try[:10])
        valid = np.ones(N_to_sample, np.float32)
    else:
        print('not enough valid samples! returning a few fakes')
        if mode=='3D':
            perm = np.random.permutation(Z*Y*X)
            samples_try = perm[:N_to_sample].astype(np.int32)
            locs_try = np.zeros((N_to_sample, 3), np.float32)
        elif mode=='2D':
            perm = np.random.permutation(Y*X)
            samples_try = perm[:N_to_sample].astype(np.int32)
            locs_try = np.zeros((N_to_sample, 2), np.float32)
        else:
            assert(False) # 2D or 3D please
        valid = np.zeros(N_to_sample, np.float32)
    return samples_try, locs_try, valid

def get_synth_flow_v2(xyz_cam0,
                      occ0,
                      unp0,
                      summ_writer,
                      sometimes_zero=False,
                      do_vis=False):
    # this version re-voxlizes occ1, rather than warp
    B,C,Z,Y,X = list(unp0.shape)
    assert(C==3)
    
    __p = lambda x: utils_basic.pack_seqdim(x, B)
    __u = lambda x: utils_basic.unpack_seqdim(x, B)

    # we do not sample any rotations here, to keep the distribution purely
    # uniform across all translations
    # (rotation ruins this, since the pivot point is at the camera)
    cam1_T_cam0 = [utils_geom.get_random_rt(B, r_amount=0.0, t_amount=3.0), # large motion
                   utils_geom.get_random_rt(B, r_amount=0.0, t_amount=0.1, # small motion
                                            sometimes_zero=sometimes_zero)]
    cam1_T_cam0 = random.sample(cam1_T_cam0, k=1)[0]

    xyz_cam1 = utils_geom.apply_4x4(cam1_T_cam0, xyz_cam0)
    occ1 = utils_vox.voxelize_xyz(xyz_cam1, Z, Y, X)
    unp1 = utils_vox.apply_4x4_to_vox(cam1_T_cam0, unp0)
    occs = [occ0, occ1]
    unps = [unp0, unp1]

    if do_vis:
        summ_writer.summ_occs('synth/occs', occs)
        summ_writer.summ_unps('synth/unps', unps, occs)
        
    mem_T_cam = utils_vox.get_mem_T_ref(B, Z, Y, X)
    cam_T_mem = utils_vox.get_ref_T_mem(B, Z, Y, X)
    mem1_T_mem0 = utils_basic.matmul3(mem_T_cam, cam1_T_cam0, cam_T_mem)
    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X)
    xyz_mem1 = utils_geom.apply_4x4(mem1_T_mem0, xyz_mem0)
    xyz_mem0 = xyz_mem0.reshape(B, Z, Y, X, 3)
    xyz_mem1 = xyz_mem1.reshape(B, Z, Y, X, 3)
    flow = xyz_mem1-xyz_mem0
    # this is B x Z x Y x X x 3
    flow = flow.permute(0, 4, 1, 2, 3)
    # this is B x 3 x Z x Y x X
    if do_vis:
        summ_writer.summ_3D_flow('synth/flow', flow, clip=2.0)

    if do_vis:
        occ0_e = utils_samp.backwarp_using_3D_flow(occ1, flow, binary_feat=True)
        unp0_e = utils_samp.backwarp_using_3D_flow(unp1, flow)
        summ_writer.summ_occs('synth/occs_stab', [occ0, occ0_e])
        summ_writer.summ_unps('synth/unps_stab', [unp0, unp0_e], [occ0, occ0_e])

    occs = torch.stack(occs, dim=1)
    unps = torch.stack(unps, dim=1)

    return occs, unps, flow, cam1_T_cam0

def get_boxes_from_flow_mag(flow_mag, N):
    B, Z, Y, X = list(flow_mag.shape)
    # flow_mag is B x Z x Y x X

    ## plan:
    # take a linspace of threhsolds between the min and max
    # for each thresh
    #   create a binary map
    #   turn this into labels with connected_components
    # vis all these

    assert(B==1) # later i will extend

    flow_mag = flow_mag[0]
    # flow_mag is Z x Y x X
    flow_mag = flow_mag.detach().cpu().numpy()

    from cc3d import connected_components

    # adjust for numerical errors
    flow_mag = flow_mag*100.0

    boxlist = np.zeros([N, 9], dtype=np.float32)
    scorelist = np.zeros([N], dtype=np.float32)
    connlist = np.zeros([N, Z, Y, X], dtype=np.float32)
    boxcount = 0

    mag = np.reshape(flow_mag, [Z, Y, X])
    mag_min, mag_max = np.min(mag), np.max(mag)
    # print('min, max = %.6f, %.6f' % (mag_min, mag_max))
    
    threshs = np.linspace(mag_min, mag_max, num=12)
    threshs = threshs[1:-1]
    # print('threshs:', threshs)
    
    zg, yg, xg = utils_basic.meshgrid3D_py(Z, Y, X, stack=False, norm=False)
    box3D_list = []

    flow_mag_vis = flow_mag - np.min(flow_mag)
    flow_mag_vis = flow_mag_vis / np.max(flow_mag_vis)
    # utils_basic.print_stats_py('flow_mag_vis', flow_mag_vis)
    image = (np.mean(flow_mag_vis, axis=1)*255.0).astype(np.uint8)
    image = np.stack([image, image, image], axis=2)

    # utils_basic.print_stats_py('image', image)

    # sx = float(X)/np.abs(float(utils_vox.XMAX-utils_vox.XMIN))
    # sy = float(Y)/np.abs(float(utils_vox.YMAX-utils_vox.YMIN))
    # sz = float(Z)/np.abs(float(utils_vox.ZMAX-utils_vox.ZMIN))
    # print('scalars:', sx, sy, sz)
    
    for ti, thresh in enumerate(threshs):
        # print('working on thresh %d: %.2f' % (ti, thresh))
        mask = (mag > thresh).astype(np.int32)
        if np.sum(mask) > 8: # if we have a few pixels to connect up 
            labels = connected_components(mask)
            segids = [ x for x in np.unique(labels) if x != 0 ]
            for si, segid in enumerate(segids):
                extracted_vox = (labels == segid)
                if np.sum(extracted_vox) > 8: # if we have a few pixels to box up 
                    # print('segid = %d' % segid)
                    # print('extracted vox has this shape:', extracted_vox.shape)
                    
                    z = zg[extracted_vox==1]
                    y = yg[extracted_vox==1]
                    x = xg[extracted_vox==1]

                    # find the oriented box in birdview
                    im = np.sum(extracted_vox, axis=1) # reduce on the Y dim
                    im = im.astype(np.uint8)
                    
                    contours, hier = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    if contours:
                        cnt = contours[0]
                        rect = cv2.minAreaRect(cnt)

                        # i want to clip at the index where YMAX dips under the ground
                        # and where YMIN reaches above some reasonable height

                        shift = hyp.YMIN
                        scale = float(Y)/np.abs(float(hyp.YMAX-hyp.YMIN))
                        ymin_ = (hyp.FLOOR-shift)*scale
                        ymax_ = (hyp.CEIL-shift)*scale

                        if ymin_ > ymax_:
                            # this is true if y points downards
                            ymax_, ymin_ = ymin_, ymax_
                            
                        # ymin = np.clip(np.min(y), ymin_, ymax_)
                        # ymax = np.clip(np.max(y), ymin_, ymax_)

                        ymin = np.min(y)
                        ymax = np.max(y)
                            
                        hei = ymax-ymin
                        yc = (ymax+ymin)/2.0

                        (xc,zc),(wid,dep),theta = rect
                        theta = -theta
                        
                        box = cv2.boxPoints(rect)
                        if dep < wid:
                            # dep goes along the long side of an oriented car
                            theta += 90.0
                            wid, dep = dep, wid
                        theta = utils_geom.deg2rad(theta)

                        if boxcount < N and (yc > ymin_) and (yc < ymax_):
                            # bx, by = np.split(box, axis=1)
                            # boxpoints[boxcount,:] = box

                            box3D = [xc, yc, zc, wid, hei, dep, 0, theta, 0]
                            box3D = np.array(box3D).astype(np.float32)

                            already_have = False
                            for box3D_ in box3D_list:
                                if np.all(box3D_==box3D):
                                    already_have = True
                            
                            if ((not already_have) and
                                # don't be empty (redundant now)
                                (hei > 0) and
                                (wid > 0) and
                                (dep > 0) and
                                # be less than huge
                                (hei < 10.0) and
                                (wid < 10.0) and
                                (dep < 10.0) and
                                # be bigger than 2 vox
                                (hei > 2.0) and
                                (wid > 2.0) and
                                (dep > 2.0)
                            ):
                                # print 'mean(y), min(y) max(y), ymin_, ymax_, ymin, ymax = %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (
                                #     np.mean(y), np.min(y), np.max(y), ymin_, ymax_, ymin, ymax)
                                # print 'xc, yc, zc = %.2f, %.2f, %.2f; wid, hei, dep = %.2f, %.2f, %.2f' % (
                                #     xc, yc, zc, wid, hei, dep)
                                
                                # print 'wid, hei, dep = %.2f, %.2f, %.2f' % (wid, hei, dep)
                                # # print 'theta = %.2f' % theta
                                
                                box = np.int0(box)
                                cv2.drawContours(image,[box],-1,(0,191,255),1)

                                boxlist[boxcount,:] = box3D
                                scorelist[boxcount] = np.random.uniform(0.1, 1.0)

                                conn_ = np.zeros([Z, Y, X], np.float32)
                                conn_[extracted_vox] = 1.0
                                connlist[boxcount] = conn_
                                
                                # imageio.imwrite('boxes_%02d.png' % (boxcount), image)
                                # imageio.imwrite('conn_%02d.png' % (boxcount), np.max(conn_, axis=1))
                                
                                boxcount += 1
                                box3D_list.append(box3D)
                            else:
                                # print('skipping a box that already exists')
                                pass
                        else:
                            # print('box overflow; found more than %d' % N)
                            pass


    image = np.transpose(image, [2, 0, 1]) # channels first
    image = torch.from_numpy(image).float().to('cuda').unsqueeze(0)
    boxlist = torch.from_numpy(boxlist).float().to('cuda').unsqueeze(0)
    scorelist = torch.from_numpy(scorelist).float().to('cuda').unsqueeze(0)
    connlist = torch.from_numpy(connlist).float().to('cuda').unsqueeze(0)

    tidlist = torch.linspace(1.0, N, N).long().to('cuda')
    tidlist = tidlist.unsqueeze(0)

    image = utils_improc.preprocess_color(image)
    
    return image, boxlist, scorelist, tidlist, connlist

def find_detections_corresponding_to_traj(obj_clist, obj_vlist, xyzlists, vislists):
    # obj_clist is B x S x 3
    # obj_vlist is B x S
    # xyzlists is B x S x N x 3
    # vislists is B x S x N x 3
    
    B, S, N, D = list(xyzlists.shape)
    assert(D==3) # this should be 3 values, for xyz

    # print('obj_clist', obj_clist.shape)
    # print('obj_vlist', obj_vlist.shape)
    # print('xyzlists', xyzlists.shape)
    # print('vislists', vislists.shape)

    # make life easier
    obj_foundlist = torch.zeros_like(obj_vlist)
    for b in range(B):
        obj_foundlist[b] = find_detections_corresponding_to_traj_single(
            obj_clist[b], obj_vlist[b], xyzlists[b], vislists[b])
    return obj_foundlist

def find_detections_corresponding_to_traj_single(obj_clist, obj_vlist, xyzlists, vislists):
    # obj_clist is S x 3
    # obj_vlist is S
    # xyzlists is S x N x 3
    # vislists is S x N x 3
    S, N, D = list(xyzlists.shape)
    assert(D==3) # this should be 3 values, for xyz
    
    obj_foundlist = torch.zeros_like(obj_vlist)

    # step through the trajectory, and
    # see if each location has a detection nearby
    for s in list(range(S)):
        obj_c = obj_clist[s]
        # this is 3

        # print('obj_c:', obj_c.detach().cpu().numpy())

        # look at the list of detections;
        # if there is one within some threshold of dist with this, then ok.

        xyzlist = xyzlists[s]
        # this is N x 3
        vislist = vislists[s]
        # this is N

        # print('xyzlist:', xyzlist.detach().cpu().numpy())
        

        distlist = torch.norm(obj_c.unsqueeze(0)-xyzlist, dim=1)
        # this is N

        # print('distlist:', distlist.detach().cpu().numpy())
        
        dist_thresh = 3.0 
        did_detect_something = 0.0
        for n in list(range(N)):
            if (vislist[n] > 0.5) and (distlist[n] < dist_thresh):
                did_detect_something = 1.0
        obj_foundlist[s] = did_detect_something
        
        # print('got it?', did_detect_something)
        # input()
    return obj_foundlist

