import torch
import numpy as np
import utils_geom
import utils_basic
np.set_printoptions(suppress=True, precision=6, threshold=2000)

def merge_rt_py(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def split_rt_py(rt):
    r = rt[:3,:3]
    t = rt[:3,3]
    r = np.reshape(r, [3, 3])
    t = np.reshape(t, [3, 1])
    return r, t

def apply_4x4_py(rt, xyz):
    # rt is 4 x 4
    # xyz is N x 3
    r, t = split_rt_py(rt)
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x 3 x N
    xyz = np.dot(r, xyz)
    # xyz is xyz1 x 3 x N
    xyz = np.transpose(xyz, [1, 0])
    # xyz is xyz1 x N x 3
    t = np.reshape(t, [1, 3])
    xyz = xyz + t
    return xyz

def rigid_transform_3D(xyz_cam0, xyz_cam1, do_ransac=True):
    xyz_cam0 = xyz_cam0.detach().cpu().numpy()
    xyz_cam1 = xyz_cam1.detach().cpu().numpy()
    cam1_T_cam0 = rigid_transform_3D_py(xyz_cam0, xyz_cam1, do_ransac=do_ransac)
    cam1_T_cam0 = torch.from_numpy(cam1_T_cam0).float().to('cuda')
    return cam1_T_cam0

def rigid_transform_3D_py_helper(xyz0, xyz1):
    assert len(xyz0) == len(xyz1)
    N = xyz0.shape[0] # total points
    if N > 3:
        centroid_xyz0 = np.mean(xyz0, axis=0)
        centroid_xyz1 = np.mean(xyz1, axis=0)
        # print('centroid_xyz0', centroid_xyz0)
        # print('centroid_xyz1', centroid_xyz1)

        # center the points
        xyz0 = xyz0 - np.tile(centroid_xyz0, (N, 1))
        xyz1 = xyz1 - np.tile(centroid_xyz1, (N, 1))

        H = np.dot(xyz0.T, xyz1)

        U, S, Vt = np.linalg.svd(H)

        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[2,:] *= -1
           R = np.dot(Vt.T, U.T)

        t = np.dot(-R, centroid_xyz0.T) + centroid_xyz1.T
        t = np.reshape(t, [3])
    else:
        print('too few points; returning identity')
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
    rt = merge_rt_py(R, t)
    return rt

def rigid_transform_3D_py(xyz0, xyz1, do_ransac=True):
    # xyz0 and xyz1 are each N x 3
    assert len(xyz0) == len(xyz1)

    N = xyz0.shape[0] # total points

    nPts = 8 # anything >3 is ok really
    if N <= nPts:
        print('N = %d; returning an identity mat' % N)
        R = np.eye(3, dtype=np.float32)
        t = np.zeros(3, dtype=np.float32)
        rt = merge_rt_py(R, t)
    elif not do_ransac:
        rt = rigid_transform_3D_py_helper(xyz0, xyz1)
    else:
        # print('N = %d' % N)
        # print('doing ransac')
        rts = []
        errs = []
        ransac_steps = 128
        for step in list(range(ransac_steps)):
            assert(N > nPts) 
            perm = np.random.permutation(N)
            cam1_T_cam0 = rigid_transform_3D_py_helper(xyz0[perm[:nPts]], xyz1[perm[:nPts]])
            # i got some errors in matmul when the arrays were too big, 
            # so let's just use 1k points for the error 
            perm = np.random.permutation(N)
            xyz1_prime = apply_4x4_py(cam1_T_cam0, xyz0[perm[:min([1000,N])]])
            xyz1_actual = xyz1[perm[:min([1000,N])]]
            err = np.mean(np.sum(np.abs(xyz1_prime-xyz1_actual), axis=1))
            rts.append(cam1_T_cam0)
            errs.append(err)
        ind = np.argmin(errs)
        rt = rts[ind]
    return rt

def compute_mem1_T_mem0_from_object_flow(flow_mem, mask_mem, occ_mem):
    B, C, Z, Y, X = list(flow_mem.shape)
    assert(C==3)
    mem1_T_mem0 = utils_geom.eye_4x4(B)

    xyz_mem0 = utils_basic.gridcloud3D(B, Z, Y, X, norm=False)
    
    for b in list(range(B)):
        # i think there is a way to parallelize the where/gather but it is beyond me right now
        occ = occ_mem[b]
        mask = mask_mem[b]
        flow = flow_mem[b]
        xyz0 = xyz_mem0[b]
        # cam_T_obj = camR_T_obj[b]
        # mem_T_cam = mem_T_ref[b]

        flow = flow.reshape(3, -1).permute(1, 0)
        # flow is -1 x 3
        inds = torch.where((occ*mask).reshape(-1) > 0.5)
        # inds is ?
        flow = flow[inds]

        xyz0 = xyz0[inds]
        xyz1 = xyz0 + flow

        mem1_T_mem0_ = rigid_transform_3D(xyz0, xyz1)
        # this is 4 x 4 
        mem1_T_mem0[b] = mem1_T_mem0_

    return mem1_T_mem0
