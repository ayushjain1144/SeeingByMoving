import torch
# import hyperparams as hyp
import numpy as np
import utils_geom
import utils_samp
import utils_improc
import torch.nn.functional as F
from utils_basic import *
import utils_basic

# if hyp.dataset_name == "mujoco_offline":
#     XMIN = -0.5 # right (neg is left)
#     XMAX = 0.5 # right
#     YMIN = -0.5 # down (neg is up)
#     YMAX = 0.5 # down
#     ZMIN = 0.3 # forward
#     ZMAX = 1.3 # forward
#     FLOOR = 0.0 # ground (parallel with refcam)
#     CEIL = (FLOOR-0.5) # 
# else:
#     # XMIN = -20.0 # right (neg is left)
#     # XMAX = 20.0 # right
#     XMIN = -16.0 # right (neg is left)
#     XMAX = 16.0 # right
    
#     # narrow performs better:
#     YMIN = -1.0 # down (neg is up)
#     YMAX = 3.0 # down
    
#     # YMIN = -2.0 # down (neg is up)
#     # YMAX = 6.0 # down
    
#     # ZMIN = 10.0 # forward
#     # ZMAX = 42.0 # forward
#     # ZMIN = 2.0 # forward
#     # ZMAX = 42.0 # forward
#     ZMIN = 2.0 # forward
#     ZMAX = 34.0 # forward

#     FLOOR = 2.65 # ground (2.65m downward from the cam)
#     # CEIL = (FLOOR-1.65) # 1.65m above ground
#     CEIL = (FLOOR-2.0) # 
    

def get_inbounds(xyz, Z, Y, X, already_mem=False):
    # xyz is B x N x 3
    if not already_mem:
        xyz = Ref2Mem(xyz, Z, Y, X)

    x = xyz[:,:,0]
    y = xyz[:,:,1]
    z = xyz[:,:,2]
    
    x_valid = (x>-0.5).byte() & (x<float(X-0.5)).byte()
    y_valid = (y>-0.5).byte() & (y<float(Y-0.5)).byte()
    z_valid = (z>-0.5).byte() & (z<float(Z-0.5)).byte()
    nonzero = (~(z==0.0)).byte()
    
    inbounds = x_valid & y_valid & z_valid & nonzero
    return inbounds.bool()

def get_inbounds_single(xyz, Z, Y, X, already_mem=False):
    # xyz is N x 3
    xyz = xyz.unsqueeze(0)
    inbounds = get_inbounds(xyz, Z, Y, X, already_mem=already_mem)
    inbounds = inbounds.squeeze(0)
    return inbounds

def voxelize_xyz(xyz_ref, Z, Y, X, already_mem=False):
    B, N, D = list(xyz_ref.shape)
    assert(D==3)
    if already_mem:
        xyz_mem = xyz_ref
    else:
        xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    vox = get_occupancy(xyz_mem, Z, Y, X)
    return vox

def voxelize_zoom(xyz_ref, lrt, Z, Y, X):
    B, N, D = list(xyz_ref.shape)
    assert(D==3)

    # xyz_ref = Zoom2Ref(xyz_zoom, lrt, Z2, Y2, X2, additive_pad=additive_pad)

    xyz_zoom = Ref2Zoom(xyz_ref, lrt, Z, Y, X, additive_pad=0.1)    
    # if already_mem:
    #     xyz_mem = xyz_ref
    # else:
    #     xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    vox = get_occupancy(xyz_zoom, Z, Y, X)
    return vox

def voxelize_near_xyz(xyz_ref, xyz, Z, Y, X):
    B, N, D = list(xyz_ref.shape)
    assert(D==3)

    # xyz_ref = Zoom2Ref(xyz_zoom, lrt, Z2, Y2, X2, additive_pad=additive_pad)

    xyzlist = xyz.unsqueeze(1) # B x 1 x 3
    lenlist = torch.ones_like(xyzlist)*16.0 # cube this size
    rotlist = torch.zeros_like(xyzlist) # no rot
    boxlist = torch.cat([xyzlist, lenlist, rotlist], dim=2)
    # boxlist is B x 1 x 9
    lrtlist = utils_geom.convert_boxlist_to_lrtlist(boxlist)
    lrt = lrtlist.squeeze(1)
    # lrt is B x 19

    xyz_zoom = Ref2Zoom(xyz_ref, lrt, Z, Y, X, additive_pad=0.1)    
    vox = get_occupancy(xyz_zoom, Z, Y, X)
    return vox

def get_occupancy_single(xyz, Z, Y, X):
    # xyz is N x 3 and in mem coords
    # we want to fill a voxel tensor with 1's at these inds

    # (we have a full parallelized version, but fill_ray_single needs this)

    inbounds = get_inbounds_single(xyz, Z, Y, X, already_mem=True)
    xyz = xyz[inbounds]
    # xyz is N x 3

    # this is more accurate than a cast/floor, but runs into issues when a dim==0
    xyz = torch.round(xyz).int()
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]

    vox_inds = sub2ind3D(Z, Y, X, z, y, x)
    vox_inds = vox_inds.flatten().long()
    voxels = torch.zeros(Z*Y*X, dtype=torch.float32)
    voxels[vox_inds] = 1.0
    voxels = voxels.reshape(1, Z, Y, X)
    # 1 x Z x Y x X
    return voxels

def get_occupancy(xyz, Z, Y, X):
    # xyz is B x N x 3 and in mem coords
    # we want to fill a voxel tensor with 1's at these inds
    B, N, C = list(xyz.shape)
    assert(C==3)

    # these papers say simple 1/0 occupancy is ok:
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
    # cont fusion says they do 8-neighbor interp
    # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

    inbounds = get_inbounds(xyz, Z, Y, X, already_mem=True)
    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
    mask = torch.zeros_like(x)
    mask[inbounds] = 1.0

    # set the invalid guys to zero
    # we then need to zero out 0,0,0
    # (this method seems a bit clumsy)
    x = x*mask
    y = y*mask
    z = z*mask

    x = torch.round(x)
    y = torch.round(y)
    z = torch.round(z)
    x = torch.clamp(x, 0, X-1).int()
    y = torch.clamp(y, 0, Y-1).int()
    z = torch.clamp(z, 0, Z-1).int()

    x = x.view(B*N)
    y = y.view(B*N)
    z = z.view(B*N)

    dim3 = X
    dim2 = X * Y
    dim1 = X * Y * Z

    base = torch.arange(0, B, dtype=torch.int32, device=torch.device('cuda'))*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

    vox_inds = base + z * dim2 + y * dim3 + x
    voxels = torch.zeros(B*Z*Y*X, device=torch.device('cuda')).float()
    voxels[vox_inds.long()] = 1.0
    # zero out the singularity
    voxels[base.long()] = 0.0
    voxels = voxels.reshape(B, 1, Z, Y, X)
    # B x 1 x Z x Y x X
    return voxels

def Ref2Mem(xyz, Z, Y, X, bounds='default'):
    # xyz is B x N x 3, in ref coordinates
    # transforms velo coordinates into mem coordinates
    B, N, C = list(xyz.shape)
    mem_T_ref = get_mem_T_ref(B, Z, Y, X, bounds=bounds)
    xyz = utils_geom.apply_4x4(mem_T_ref, xyz)
    return xyz

def Mem2Ref(xyz_mem, Z, Y, X, bounds='default'):
    # xyz is B x N x 3, in mem coordinates
    # transforms mem coordinates into ref coordinates
    B, N, C = list(xyz_mem.shape)
    ref_T_mem = get_ref_T_mem(B, Z, Y, X, bounds=bounds)
    xyz_ref = utils_geom.apply_4x4(ref_T_mem, xyz_mem)
    return xyz_ref

def get_ref_T_mem(B, Z, Y, X, bounds='default'):
    mem_T_ref = get_mem_T_ref(B, Z, Y, X, bounds=bounds)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = mem_T_ref.inverse()
    return ref_T_mem

def get_mem_T_ref(B, Z, Y, X, bounds='default'):
    # sometimes we want the mat itself
    # note this is not a rigid transform

    # if not bounds=='default':
    #     xmin, xmax, ymin, ymax, zmin, zmax = bounds
    #     # these can be shaped B or be scalars
    # else:
    #     xmin, xmax, ymin, ymax, zmin, zmax = XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX
        
    # for interpretability, let's construct this in two steps...
    
    # translation
    center_T_ref = utils_geom.eye_4x4(B)
    center_T_ref[:,0,3] = -hyp.XMIN
    center_T_ref[:,1,3] = -hyp.YMIN
    center_T_ref[:,2,3] = -hyp.ZMIN

    VOX_SIZE_X = (hyp.XMAX-hyp.XMIN)/float(X)
    VOX_SIZE_Y = (hyp.YMAX-hyp.YMIN)/float(Y)
    VOX_SIZE_Z = (hyp.ZMAX-hyp.ZMIN)/float(Z)
        
    # scaling
    mem_T_center = utils_geom.eye_4x4(B)
    mem_T_center[:,0,0] = 1./VOX_SIZE_X
    mem_T_center[:,1,1] = 1./VOX_SIZE_Y
    mem_T_center[:,2,2] = 1./VOX_SIZE_Z
    mem_T_ref = utils_basic.matmul2(mem_T_center, center_T_ref)
    
    return mem_T_ref

def unproject_rgb_to_mem(rgb_camB, Z, Y, X, pixB_T_camA):
    # rgb_camB is B x C x H x W
    # pixB_T_camA is B x 4 x 4

    # rgb lives in B pixel coords
    # we want everything in A memory coords

    # this puts each C-dim pixel in the rgb_camB
    # along a ray in the voxelgrid
    B, C, H, W = list(rgb_camB.shape)

    xyz_memA = gridcloud3D(B, Z, Y, X, norm=False)
    # grid_z, grid_y, grid_x = meshgrid3D(B, Z, Y, X)
    # # these are B x Z x Y x X
    # # these represent the mem grid coordinates

    # # we need to convert these to pixel coordinates
    # x = torch.reshape(grid_x, [B, -1])
    # y = torch.reshape(grid_y, [B, -1])
    # z = torch.reshape(grid_z, [B, -1])
    # # these are B x N
    # xyz_mem = torch.stack([x, y, z], dim=2)
    
    xyz_camA = Mem2Ref(xyz_memA, Z, Y, X)
    
    xyz_pixB = utils_geom.apply_4x4(pixB_T_camA, xyz_camA)
    normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
    EPS=1e-6
    xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)
    # this is B x N x 2
    # this is the (floating point) pixel coordinate of each voxel
    x_pixB, y_pixB = xy_pixB[:,:,0], xy_pixB[:,:,1]
    # these are B x N

    if (0):
        # handwritten version
        values = torch.zeros([B, C, Z*Y*X], dtype=torch.float32)
        for b in list(range(B)):
            values[b] = utils_samp.bilinear_sample_single(rgb_camB[b], x_pixB[b], y_pixB[b])
    else:
        # native pytorch version
        y_pixB, x_pixB = normalize_grid2D(y_pixB, x_pixB, H, W)
        # since we want a 3d output, we need 5d tensors
        z_pixB = torch.zeros_like(x_pixB)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
        rgb_camB = rgb_camB.unsqueeze(2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB, xyz_pixB)
        
    values = torch.reshape(values, (B, C, Z, Y, X))
    return values

def apply_pixX_T_memR_to_voxR(pix_T_camX, camX_T_camR, voxR, D, H, W):
    # mats are B x 4 x 4
    # voxR is B x C x Z x Y x X
    # H, W, D indicates how big to make the output 
    # returns B x C x D x H x W
    
    B, C, Z, Y, X = list(voxR.shape)
    z_near = hyp.ZMIN
    z_far = hyp.ZMAX

    grid_z = torch.linspace(z_near, z_far, steps=D, dtype=torch.float32, device=torch.device('cuda'))
    # grid_z = torch.exp(torch.linspace(np.log(z_near), np.log(z_far), steps=D, dtype=torch.float32, device=torch.device('cuda')))
    grid_z = torch.reshape(grid_z, [1, 1, D, 1, 1])
    grid_z = grid_z.repeat([B, 1, 1, H, W])
    grid_z = torch.reshape(grid_z, [B*D, 1, H, W])

    pix_T_camX__ = torch.unsqueeze(pix_T_camX, axis=1).repeat([1, D, 1, 1])
    pix_T_camX = torch.reshape(pix_T_camX__, [B*D, 4, 4])
    xyz_camX = utils_geom.depth2pointcloud(grid_z, pix_T_camX)

    camR_T_camX = utils_geom.safe_inverse(camX_T_camR)
    camR_T_camX_ = torch.unsqueeze(camR_T_camX, dim=1).repeat([1, D, 1, 1])
    camR_T_camX = torch.reshape(camR_T_camX_, [B*D, 4, 4])

    mem_T_cam = get_mem_T_ref(B*D, Z, Y, X)
    memR_T_camX = matmul2(mem_T_cam, camR_T_camX)

    xyz_memR = utils_geom.apply_4x4(memR_T_camX, xyz_camX)
    xyz_memR = torch.reshape(xyz_memR, [B, D*H*W, 3])
    
    samp = utils_samp.sample3D(voxR, xyz_memR, D, H, W)
    # samp is B x H x W x D x C
    return samp

def resample_to_target_views(occRs, camRs_T_camPs):
    # resample to the target view

    # occRs is B x S x Y x X x Z x 1
    # camRs_T_camPs is B x S x 4 x 4
    
    B, S, _, Z, Y, X = list(occRs.shape)

    # we want to construct a mat memR_T_memP
    
    cam_T_mem = get_ref_T_mem(B, Z, Y, X)
    mem_T_cam = get_mem_T_ref(B, Z, Y, X)
    cams_T_mems = cam_T_mem.unsqueeze(1).repeat(1, S, 1, 1)
    mems_T_cams = mem_T_cam.unsqueeze(1).repeat(1, S, 1, 1)

    cams_T_mems = torch.reshape(cams_T_mems, (B*S, 4, 4))
    mems_T_cams = torch.reshape(mems_T_cams, (B*S, 4, 4))
    camRs_T_camPs = torch.reshape(camRs_T_camPs, (B*S, 4, 4))
    
    memRs_T_memPs = torch.matmul(torch.matmul(mems_T_cams, camRs_T_camPs), cams_T_mems)
    memRs_T_memPs = torch.reshape(memRs_T_memPs, (B, S, 4, 4))

    occRs, valid = resample_to_view(occRs, memRs_T_memPs, multi=True)
    return occRs, valid

def resample_to_target_view(occRs, camR_T_camP):
    B, S, Z, Y, X, _ = list(occRs.shape)
    cam_T_mem = get_ref_T_mem(B, Z, Y, X)
    mem_T_cam = get_mem_T_ref(B, Z, Y, X)
    memR_T_memP = torch.matmul(torch.matmul(mem_T_cam, camR_T_camP), cam_T_mem)
    occRs, valid = resample_to_view(occRs, memR_T_memP, multi=False)
    return occRs, valid
    
def resample_to_view(feats, new_T_old, multi=False):
    # feats is B x S x c x Y x X x Z 
    # it represents some scene features in reference/canonical coordinates
    # we want to go from these coords to some target coords

    # new_T_old is B x 4 x 4
    # it represents a transformation between two "mem" systems
    # or if multi=True, it's B x S x 4 x 4

    B, S, C, Z, Y, X = list(feats.shape)

    # we want to sample for each location in the bird grid
    # xyz_mem = gridcloud3D(B, Z, Y, X)
    grid_y, grid_x, grid_z = meshgrid3D(B, Z, Y, X)
    # these are B x BY x BX x BZ
    # these represent the mem grid coordinates

    # we need to convert these to pixel coordinates
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N

    xyz_mem = torch.stack([x, y, z], dim=2)
    # this is B x N x 3

    xyz_mems = xyz_mem.unsqueeze(1).repeat(1, S, 1, 1)
    # this is B x S x N x 3

    xyz_mems_ = xyz_mems.view(B*S, Y*X*Z, 3)

    feats_ = feats.view(B*S, C, Z, Y, X)

    if multi:
        new_T_olds = new_T_old.clone()
    else:
        new_T_olds = new_T_old.unsqueeze(1).repeat(1, S, 1, 1)
    new_T_olds_ = new_T_olds.view(B*S, 4, 4)

    xyz_new_ = utils_geom.apply_4x4(new_T_olds_, xyz_mems_)
    # we want each voxel to replace its value
    # with whatever is at these new coordinates

    # i.e., we are back-warping from the "new" coords
    
    feats_, valid_ = utils_samp.resample3D(feats_, xyz_new_)
    feats = feats_.view(B, S, C, Z, Y, X)
    valid = valid_.view(B, S, 1, Z, Y, X)
    return feats, valid 

def convert_xyz_to_visibility(xyz, Z, Y, X, ray_add=0.0):
    # xyz is in camera coordinates
    # proto shows the size of the birdgrid
    B, N, C = list(xyz.shape)
    assert(C==3)
    voxels = torch.zeros(B, 1, Z, Y, X, dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        voxels[b,0] = fill_ray_single(xyz[b], Z, Y, X, ray_add=ray_add)
    return voxels

def fill_ray_single(xyz, Z, Y, X, ray_add=0.0):
    # xyz is N x 3, and in bird coords
    # we want to fill a voxel tensor with 1's at these inds,
    # and also at any ind along the ray before it

    xyz = torch.reshape(xyz, (-1, 3))
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    # these are N

    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    z = z.unsqueeze(1)

    # get the hypotenuses
    u = torch.sqrt(x**2+z**2) # flat to ground
    v = torch.sqrt(x**2+y**2+z**2)
    w = torch.sqrt(x**2+y**2)

    # the ray is along the v line
    # we want to find xyz locations along this line

    # get the angles
    EPS = 1e-6
    u = torch.clamp(u, min=EPS)
    v = torch.clamp(v, min=EPS)
    sin_theta = y/v # soh 
    cos_theta = u/v # cah
    sin_alpha = z/u # soh
    cos_alpha = x/u # cah

    samps = int(np.sqrt(Y**2 + Z**2))*2
    # for each proportional distance in [0.0, 1.0], generate a new hypotenuse
    dists = torch.linspace(0.0, 1.0, samps, device=torch.device('cuda'))
    dists = torch.reshape(dists, (1, samps))
    v_ = dists * v.repeat(1, samps)
    v_ = v_ + ray_add

    # now, for each of these v_, we want to generate the xyz
    y_ = sin_theta*v_
    u_ = torch.abs(cos_theta*v_)
    z_ = sin_alpha*u_
    x_ = cos_alpha*u_
    # these are the ref coordinates we want to fill
    x = x_.flatten()
    y = y_.flatten()
    z = z_.flatten()

    xyz = torch.stack([x,y,z], dim=1).unsqueeze(0)
    xyz = Ref2Mem(xyz, Z, Y, X)
    xyz = torch.squeeze(xyz, dim=0)
    # these are the mem coordinates we want to fill

    return get_occupancy_single(xyz, Z, Y, X)

def get_freespace(xyz, occ, ray_add=0.0):
    # xyz is B x N x 3
    # occ is B x H x W x D x 1
    B, C, Z, Y, X = list(occ.shape)
    assert(C==1)
    vis = convert_xyz_to_visibility(xyz, Z, Y, X, ray_add=ray_add)
    # visible space is all free unless it's occupied
    free = (1.0-(occ>0.0).float())*vis
    return free

def apply_4x4_to_vox(B_T_A, feat_A, already_mem=False, binary_feat=False, rigid=True):
    # B_T_A is B x 4 x 4
    # if already_mem=False, it is a transformation between cam systems
    # if already_mem=True, it is a transformation between mem systems

    # feat_A is B x C x Z x Y x X
    # it represents some scene features in reference/canonical coordinates
    # we want to go from these coords to some target coords

    # since this is a backwarp,
    # the question to ask is:
    # "WHERE in the tensor do you want to sample,
    # to replace each voxel's current value?"
    
    # the inverse of B_T_A represents this "where";
    # it transforms each coordinate in B
    # to the location we want to sample in A

    B, C, Z, Y, X = list(feat_A.shape)

    # we have B_T_A in input, since this follows the other utils_geom.apply_4x4
    # for an apply_4x4 func, but really we need A_T_B
    if rigid:
        A_T_B = utils_geom.safe_inverse(B_T_A)
    else:
        # this op is slower but more powerful
        A_T_B = B_T_A.inverse()
        

    if not already_mem:
        cam_T_mem = get_ref_T_mem(B, Z, Y, X)
        mem_T_cam = get_mem_T_ref(B, Z, Y, X)
        A_T_B = matmul3(mem_T_cam, A_T_B, cam_T_mem)

    # we want to sample for each location in the bird grid
    xyz_B = gridcloud3D(B, Z, Y, X)
    # this is B x N x 3

    # transform
    xyz_A = utils_geom.apply_4x4(A_T_B, xyz_B)
    # we want each voxel to take its value
    # from whatever is at these A coordinates
    # i.e., we are back-warping from the "A" coords

    # feat_B = F.grid_sample(feat_A, normalize_grid(xyz_A, Z, Y, X))
    feat_B = utils_samp.resample3D(feat_A, xyz_A, binary_feat=binary_feat)

    # feat_B, valid = utils_samp.resample3D(feat_A, xyz_A, binary_feat=binary_feat)
    # return feat_B, valid
    return feat_B

def apply_4x4s_to_voxs(Bs_T_As, feat_As, already_mem=False, binary_feat=False):
    # plural wrapper for apply_4x4_to_vox
    
    B, S, C, Z, Y, X = list(feat_As.shape)
    
    # utils for packing/unpacking along seq dim
    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)
    
    Bs_T_As_ = __p(Bs_T_As)
    feat_As_ = __p(feat_As)
    feat_Bs_ = apply_4x4_to_vox(Bs_T_As_, feat_As_, already_mem=already_mem, binary_feat=binary_feat)
    feat_Bs = __u(feat_Bs_)
    return feat_Bs

def prep_occs_supervision(camRs_T_camXs,
                          xyz_camXs,
                          Z, Y, X,
                          agg=False):
    B, S, N, D = list(xyz_camXs.size())
    assert(D==3)
    # occRs_half = __u(utils_vox.voxelize_xyz(__p(xyz_camRs), Z2, Y2, X2))

    # utils for packing/unpacking along seq dim
    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)
    
    camRs_T_camXs_ = __p(camRs_T_camXs)
    xyz_camXs_ = __p(xyz_camXs)
    xyz_camRs_ = utils_geom.apply_4x4(camRs_T_camXs_, xyz_camXs_)
    occXs_ = voxelize_xyz(xyz_camXs_, Z, Y, X)
    occRs_ = voxelize_xyz(xyz_camRs_, Z, Y, X)

    # note we must compute freespace in the given view,
    # then warp to the target view
    freeXs_ = get_freespace(xyz_camXs_, occXs_)
    freeRs_ = apply_4x4_to_vox(camRs_T_camXs_, freeXs_)

    occXs = __u(occXs_)
    occRs = __u(occRs_)
    freeXs = __u(freeXs_)
    freeRs = __u(freeRs_)
    # these are B x S x 1 x Z x Y x X
    
    if agg:
        # note we should only agg if we are in STATIC mode (time frozen)
        freeR = torch.max(freeRs, dim=1)[0]
        occR = torch.max(occRs, dim=1)[0]
        # these are B x 1 x Z x Y x X
        occR = (occR>0.5).float()
        freeR = (freeR>0.5).float()
        return occR, freeR, occXs, freeXs
    else:
        occRs = (occRs>0.5).float()
        freeRs = (freeRs>0.5).float()
        return occRs, freeRs, occXs, freeXs
        
def assemble_padded_obj_masklist(lrtlist, scorelist, Z, Y, X, coeff=1.0):
    # compute a binary mask in 3D for each object
    # we use this when computing the center-surround objectness score
    # lrtlist is B x N x 19
    # scorelist is B x N

    # returns masklist shaped B x N x 1 x Z x Y x X

    B, N, D = list(lrtlist.shape)
    assert(D==19)
    masks = torch.zeros(B, N, Z, Y, X)

    lenlist, ref_T_objlist = utils_geom.split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # ref_T_objlist is B x N x 4 x 4
    
    lenlist_ = lenlist.reshape(B*N, 3)
    ref_T_objlist_ = ref_T_objlist.reshape(B*N, 4, 4)
    obj_T_reflist_ = utils_geom.safe_inverse(ref_T_objlist_)

    # we want a value for each location in the mem grid
    xyz_mem_ = gridcloud3D(B*N, Z, Y, X)
    # this is B*N x V x 3, where V = Z*Y*X
    xyz_ref_ = Mem2Ref(xyz_mem_, Z, Y, X)
    # this is B*N x V x 3

    lx, ly, lz = torch.unbind(lenlist_, dim=1)
    # these are B*N
    
    # ref_T_obj = convert_box_to_ref_T_obj(boxes3D)
    # obj_T_ref = ref_T_obj.inverse()
    
    xyz_obj_ = utils_geom.apply_4x4(obj_T_reflist_, xyz_ref_)
    x, y, z = torch.unbind(xyz_obj_, dim=2)
    # these are B*N x V
    
    lx = lx.unsqueeze(1)*coeff
    ly = ly.unsqueeze(1)*coeff
    lz = lz.unsqueeze(1)*coeff
    # these are B*N x 1
    
    x_valid = (x > -lx/2.0).byte() & (x < lx/2.0).byte()
    y_valid = (y > -ly/2.0).byte() & (y < ly/2.0).byte()
    z_valid = (z > -lz/2.0).byte() & (z < lz/2.0).byte()
    inbounds = x_valid.byte() & y_valid.byte() & z_valid.byte()
    masklist = inbounds.float()
    # print(masklist.shape)
    masklist = masklist.reshape(B, N, 1, Z, Y, X)
    # print(masklist.shape)
    # print(scorelist.shape)
    masklist = masklist*scorelist.view(B, N, 1, 1, 1, 1)
    return masklist

def get_zoom_T_ref(lrt, Z, Y, X, additive_pad=0.0):
    # lrt is B x 19
    B, E = list(lrt.shape)
    assert(E==19)
    lens, ref_T_obj = utils_geom.split_lrt(lrt)
    lx, ly, lz = lens.unbind(1)

    debug = False

    if debug:
        print('lx, ly, lz')
        print(lx)
        print(ly)
        print(lz)
    
    obj_T_ref = utils_geom.safe_inverse(ref_T_obj)
    # this is B x 4 x 4

    if debug:
        print('ok, got obj_T_ref:')
        print(obj_T_ref)

    # we want a tiny bit of padding
    # additive helps avoid nans with invalid objects
    # mult helps expand big objects
    lx = lx + additive_pad
    ly = ly + additive_pad*0.5 # pad less in this dim, since it is usually pointless
    lz = lz + additive_pad
    # lx *= 1.1
    # ly *= 1.1
    # lz *= 1.1
    
    # translation
    center_T_obj_r = utils_geom.eye_3x3(B)
    center_T_obj_t = torch.stack([lx/2., ly/2., lz/2.], dim=1)
    if debug:
        print('merging these:')
        print(center_T_obj_r.shape)
        print(center_T_obj_t.shape)
    center_T_obj = utils_geom.merge_rt(center_T_obj_r, center_T_obj_t)

    if debug:
        print('ok, got center_T_obj:')
        print(center_T_obj)
    
    # scaling
    Z_VOX_SIZE_X = (lx)/float(X)
    Z_VOX_SIZE_Y = (ly)/float(Y)
    Z_VOX_SIZE_Z = (lz)/float(Z)
    diag = torch.stack([1./Z_VOX_SIZE_X,
                        1./Z_VOX_SIZE_Y,
                        1./Z_VOX_SIZE_Z,
                        torch.ones([B], device=torch.device('cuda'))],
                       axis=1).view(B, 4)
    if debug:
        print('diag:')
        print(diag)
        print(diag.shape)
    zoom_T_center = torch.diag_embed(diag)
    if debug:
        print('ok, got zoom_T_center:')
        print(zoom_T_center)
        print(zoom_T_center.shape)

    # compose these
    zoom_T_obj = utils_basic.matmul2(zoom_T_center, center_T_obj)

    if debug:
        print('ok, got zoom_T_obj:')
        print(zoom_T_obj)
        print(zoom_T_obj.shape)
    
    zoom_T_ref = utils_basic.matmul2(zoom_T_obj, obj_T_ref)

    if debug:
        print('ok, got zoom_T_ref:')
        print(zoom_T_ref)
    
    return zoom_T_ref

def get_ref_T_zoom(lrt, Z, Y, X, additive_pad=0.1):
    # lrt is B x 19
    zoom_T_ref = get_zoom_T_ref(lrt, Z, Y, X, additive_pad=additive_pad)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_zoom = zoom_T_ref.inverse()
    return ref_T_zoom

def Ref2Zoom(xyz_ref, lrt_ref, Z, Y, X, additive_pad=0.1):
    # xyz_ref is B x N x 3, in ref coordinates
    # lrt_ref is B x 19, specifying the box in ref coordinates
    # this transforms ref coordinates into zoom coordinates
    B, N, _ = list(xyz_ref.shape)
    zoom_T_ref = get_zoom_T_ref(lrt_ref, Z, Y, X, additive_pad=additive_pad)
    xyz_zoom = utils_geom.apply_4x4(zoom_T_ref, xyz_ref)
    return xyz_zoom

def Zoom2Ref(xyz_zoom, lrt_ref, Z, Y, X, additive_pad=0.1):
    # xyz_zoom is B x N x 3, in zoom coordinates
    # lrt_ref is B x 9, specifying the box in ref coordinates
    B, N, _ = list(xyz_zoom.shape)
    ref_T_zoom = get_ref_T_zoom(lrt_ref, Z, Y, X, additive_pad=additive_pad)
    xyz_ref = utils_geom.apply_4x4(ref_T_zoom, xyz_zoom)
    return xyz_ref

def crop_zoom_from_mem(mem, lrt, Z2, Y2, X2, additive_pad=0.1):
    # mem is B x C x Z x Y x X
    # lrt is B x 19
    
    B, C, Z, Y, X = list(mem.shape)
    B2, E = list(lrt.shape)

    assert(E==19)
    assert(B==B2)

    # for each voxel in the zoom grid, i want to
    # sample a voxel from the mem

    # this puts each C-dim pixel in the image
    # along a ray in the zoomed voxelgrid

    xyz_zoom = utils_basic.gridcloud3D(B, Z2, Y2, X2, norm=False)
    # these represent the zoom grid coordinates
    # we need to convert these to mem coordinates
    xyz_ref = Zoom2Ref(xyz_zoom, lrt, Z2, Y2, X2, additive_pad=additive_pad)
    xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)

    zoom = utils_samp.sample3D(mem, xyz_mem, Z2, Y2, X2)
    zoom = torch.reshape(zoom, [B, C, Z2, Y2, X2])
    return zoom

def center_mem_on_xyz(mem, xyz, Z2, Y2, X2):
    # mem is B x C x Z x Y x X
    # xyz is B x 3

    B, C, Z, Y, X = list(mem.shape)
    B2, D = list(xyz.shape)

    assert(D==3)
    assert(B==B2)

    # from the xyz i'll make a fat lrt
    # then call crop_zoom_from_mem

    xyzlist = xyz.unsqueeze(1) # B x 1 x 3
    lenlist = torch.ones_like(xyzlist)*10.0 # 10m cube
    rotlist = torch.zeros_like(xyzlist) # no rot
    
    boxlist = torch.cat([xyzlist, lenlist, rotlist], dim=2)
    # boxlist is B x 1 x 9
    
    lrtlist = utils_geom.convert_boxlist_to_lrtlist(boxlist)
    lrt = lrtlist.squeeze(1)
    # lrt is B x 19

    return crop_zoom_from_mem(mem, lrt, Z2, Y2, X2, additive_pad=0.0)

def assemble(bkg_feat0, obj_feat0, origin_T_camRs, camRs_T_zoom):
    # let's first assemble the seq of background tensors
    # this should effectively CREATE egomotion
    # i fully expect we can do this all in one shot

    # note it makes sense to create egomotion here, because
    # we want to predict each view

    B, C, Z, Y, X = list(bkg_feat0.shape)
    B2, C2, Z2, Y2, X2 = list(obj_feat0.shape)
    assert(B==B2)
    assert(C==C2)
    
    B, S, _, _ = list(origin_T_camRs.shape)
    # ok, we have everything we need
    # for each timestep, we want to warp the bkg to this timestep
    
    # utils for packing/unpacking along seq dim
    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)

    # we in fact have utils for this already
    cam0s_T_camRs = utils_geom.get_camM_T_camXs(origin_T_camRs, ind=0)
    camRs_T_cam0s = __u(utils_geom.safe_inverse(__p(cam0s_T_camRs)))

    bkg_feat0s = bkg_feat0.unsqueeze(1).repeat(1, S, 1, 1, 1, 1)
    bkg_featRs = apply_4x4s_to_voxs(camRs_T_cam0s, bkg_feat0s)

    # now for the objects
    
    # we want to sample for each location in the bird grid
    xyz_mems_ = utils_basic.gridcloud3D(B*S, Z, Y, X, norm=False)
    # this is B*S x Z*Y*X x 3
    xyz_camRs_ = Mem2Ref(xyz_mems_, Z, Y, X)
    camRs_T_zoom_ = __p(camRs_T_zoom)
    zoom_T_camRs_ = camRs_T_zoom_.inverse() # note this is not a rigid transform
    xyz_zooms_ = utils_geom.apply_4x4(zoom_T_camRs_, xyz_camRs_)

    # we will do the whole traj at once (per obj)
    # note we just have one feat for the whole traj, so we tile up
    obj_feats = obj_feat0.unsqueeze(1).repeat(1, S, 1, 1, 1, 1)
    obj_feats_ = __p(obj_feats)
    # this is B*S x Z x Y x X x C
    
    # to sample, we need feats_ in ZYX order
    obj_featRs_ = utils_samp.sample3D(obj_feats_, xyz_zooms_, Z, Y, X)
    obj_featRs = __u(obj_featRs_)

    # overweigh objects, so that we essentially overwrite
    # featRs = 0.05*bkg_featRs + 0.95*obj_featRs

    # overwrite the bkg at the object
    obj_mask = (bkg_featRs > 0).float()
    featRs = obj_featRs + (1.0-obj_mask)*bkg_featRs
    
    # note the normalization (next) will restore magnitudes for the bkg

    # # featRs = bkg_featRs
    # featRs = obj_featRs
                        
    # l2 normalize on chans
    featRs = l2_normalize(featRs, dim=2)

    validRs = 1.0 - (featRs==0).all(dim=2, keepdim=True).float().cuda()
                        
    return featRs, validRs, bkg_featRs, obj_featRs

def convert_boxlist_memR_to_camR(boxlist_memR, Z, Y, X):
    B, N, D = list(boxlist_memR.shape)
    assert(D==9)
    cornerlist_memR_legacy = utils_geom.transform_boxes_to_corners(boxlist_memR, legacy_format=True)
    ref_T_mem = get_ref_T_mem(B, Z, Y, X)
    cornerlist_camR_legacy = utils_geom.apply_4x4_to_corners(ref_T_mem, cornerlist_memR_legacy)
    boxlist_camR = utils_geom.corners_to_boxes(cornerlist_camR_legacy, legacy_format=True)
    return boxlist_camR

def convert_boxlist_camR_to_memR(boxlist_camR, Z, Y, X):
    B, N, D = list(boxlist_camR.shape)
    assert(D==9)
    cornerlist_camR_legacy = utils_geom.transform_boxes_to_corners(boxlist_camR, legacy_format=True)
    mem_T_ref = get_mem_T_ref(B, Z, Y, X)
    cornerlist_memR_legacy = utils_geom.apply_4x4_to_corners(mem_T_ref, cornerlist_camR_legacy)
    boxlist_memR = utils_geom.corners_to_boxes(cornerlist_memR_legacy, legacy_format=True)
    return boxlist_memR

