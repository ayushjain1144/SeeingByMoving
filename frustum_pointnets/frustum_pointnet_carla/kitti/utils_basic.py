import os
import numpy as np
from os.path import isfile
import torch
import torch.nn.functional as F
EPS = 1e-6

def assert_same_shape(t1, t2):
    for (x, y) in zip(list(t1.shape), list(t2.shape)):
        assert(x==y)

def print_stats_py(name, tensor):
    print('%s (%s) min = %.2f, mean = %.2f, max = %.2f' % (name, tensor.dtype, np.min(tensor), np.mean(tensor), np.max(tensor)))

def tensor2summ(tensor, permute_dim=False):
    # if permute_dim = True: 
    # for 2D tensor, assume input is torch format B x S x C x H x W, we want B x S x H x W x C
    # for 3D tensor, assume input is torch format B x S x C x H x W x D, we want B x S x H x W x C x D
    # and finally unbind the sequeence dimension and return a list of [B x H x W x C].
    assert(tensor.ndim == 5 or tensor.ndim == 6)
    assert(tensor.size()[1] == 2) #sequense length should be 2
    if permute_dim:
        if tensor.ndim == 6: #3D tensor
            tensor = tensor.permute(0, 1, 3, 4, 5, 2)
        elif tensor.ndim == 5: #2D tensor
            tensor = tensor.permute(0, 1, 3, 4, 2)

    tensor = torch.unbind(tensor, dim=1)
    return tensor

def normalize_single(d):
    # d is a whatever shape torch tensor
    dmin = torch.min(d)
    dmax = torch.max(d)
    d = (d-dmin)/(EPS+(dmax-dmin))
    return d

def normalize(d):
    # d is B x whatever. normalize within each element of the batch
    out = torch.zeros(d.size())
    if d.is_cuda:
        out = out.cuda()
    B = list(d.size())[0]
    for b in list(range(B)):
        out[b] = normalize_single(d[b])
    return out

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape
    # returns shape-1
    # axis can be a list of axes
    assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor

def gridcloud3D(B, Z, Y, X, norm=False):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3D(B, Z, Y, X, norm=norm)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

def gridcloud2D(B, Y, X, norm=False):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2D(B, Y, X, norm=norm)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    # these are B x N
    xy = torch.stack([x, y], dim=2)
    # this is B x N x 2
    return xy

def gridcloud3D_py(Z, Y, X):
    # we want to sample for each location in the grid
    grid_z, grid_y, grid_x = meshgrid3D_py(Z, Y, X)
    x = np.reshape(grid_x, [-1])
    y = np.reshape(grid_y, [-1])
    z = np.reshape(grid_z, [-1])
    # these are N
    xyz = np.stack([x, y, z], axis=1)
    # this is N x 3
    return xyz

def meshgrid2D_py(Y, X):
    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [Y, 1])
    grid_y = np.tile(grid_y, [1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, X])
    grid_x = np.tile(grid_x, [Y, 1])

    return grid_y, grid_x

def gridcloud2D_py(Y, X):
    # we want to sample for each location in the grid
    grid_y, grid_x = meshgrid2D_py(Y, X)
    x = np.reshape(grid_x, [-1])
    y = np.reshape(grid_y, [-1])
    # these are N
    xy = np.stack([x, y], axis=1)
    # this is N x 2
    return xy

def normalize_grid3D(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_z = 2.0*(grid_z / float(Z-1)) - 1.0
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
    
    return grid_z, grid_y, grid_x

def normalize_grid2D(grid_y, grid_x, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0
    
    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)
        
    return grid_y, grid_x

def normalize_gridcloud3D(xyz, Z, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xyz[...,0]
    y = xyz[...,1]
    z = xyz[...,2]
    
    z = 2.0*(z / float(Z-1)) - 1.0
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xyz = torch.stack([x,y,z], dim=-1)
    
    if clamp_extreme:
        xyz = torch.clamp(xyz, min=-2.0, max=2.0)
    return xyz

def normalize_gridcloud2D(xy, Y, X, clamp_extreme=True):
    # make things in [-1,1]
    x = xy[...,0]
    y = xy[...,1]
    
    y = 2.0*(y / float(Y-1)) - 1.0
    x = 2.0*(x / float(X-1)) - 1.0

    xy = torch.stack([x,y], dim=-1)
    
    if clamp_extreme:
        xy = torch.clamp(xy, min=-2.0, max=2.0)
    return xy

def meshgrid3D_yxz(B, Y, X, Z):
    # returns a meshgrid sized B x Y x X x Z
    # this ordering makes sense since usually Y=height, X=width, Z=depth

	grid_y = torch.linspace(0.0, Y-1, Y)
	grid_y = torch.reshape(grid_y, [1, Y, 1, 1])
	grid_y = grid_y.repeat(B, 1, X, Z)
	
	grid_x = torch.linspace(0.0, X-1, X)
	grid_x = torch.reshape(grid_x, [1, 1, X, 1])
	grid_x = grid_x.repeat(B, Y, 1, Z)

	grid_z = torch.linspace(0.0, Z-1, Z)
	grid_z = torch.reshape(grid_z, [1, 1, 1, Z])
	grid_z = grid_z.repeat(B, Y, X, 1)
	
	return grid_y, grid_x, grid_z

def meshgrid2D(B, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2D(
            grid_y, grid_x, Y, X)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
    
def meshgrid3D(B, Z, Y, X, stack=False, norm=False):
    # returns a meshgrid sized B x Z x Y x X

    grid_z = torch.linspace(0.0, Z-1, Z, device=torch.device('cuda'))
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device('cuda'))
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device('cuda'))
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3D(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x

def meshgrid3D_py(Z, Y, X, stack=False, norm=False):
    grid_z = np.linspace(0.0, Z-1, Z)
    grid_z = np.reshape(grid_z, [Z, 1, 1])
    grid_z = np.tile(grid_z, [1, Y, X])

    grid_y = np.linspace(0.0, Y-1, Y)
    grid_y = np.reshape(grid_y, [1, Y, 1])
    grid_y = np.tile(grid_y, [Z, 1, X])

    grid_x = np.linspace(0.0, X-1, X)
    grid_x = np.reshape(grid_x, [1, 1, X])
    grid_x = np.tile(grid_x, [Z, Y, 1])

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3D(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:
        # note we stack in xyz order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = np.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x

def sub2ind(height, width, y, x):
    return y*width + x

def sql2_on_axis(x, axis, keepdim=True):
    return torch.sum(x**2, axis, keepdim=keepdim)

def l2_on_axis(x, axis, keepdim=True):
    return torch.sqrt(EPS + sql2_on_axis(x, axis, keepdim=keepdim))

def l1_on_axis(x, axis, keepdim=True):
    return torch.sum(torch.abs(x), axis, keepdim=keepdim)

def sub2ind3D(depth, height, width, d, h, w):
    # when gathering/scattering with these inds, the tensor should be Z x Y x X
    return d*height*width + h*width + w

def gradient3D(x, absolute=False, square=False):
    # x should be B x C x D x H x W
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]

    # zeros = tf.zeros_like(x)
    zeros = torch.zeros_like(x)
    zero_z = zeros[:, :, 0:1, :, :]
    zero_y = zeros[:, :, :, 0:1, :]
    zero_x = zeros[:, :, :, :, 0:1]
    dz = torch.cat([dz, zero_z], axis=2)
    dy = torch.cat([dy, zero_y], axis=3)
    dx = torch.cat([dx, zero_x], axis=4)
    if absolute:
        dz = torch.abs(dz)
        dy = torch.abs(dy)
        dx = torch.abs(dx)
    if square:
        dz = dz ** 2
        dy = dy ** 2
        dx = dx ** 2
    return dz, dy, dx

def gradient2D(x, absolute=False, square=False):
    # x should be B x C x H x W
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]

    # zeros = tf.zeros_like(x)
    zeros = torch.zeros_like(x)
    zero_h = zeros[:, :, 0:1, :]
    zero_w = zeros[:, :, :, 0:1]
    dh = torch.cat([dh, zero_h], axis=2)
    dw = torch.cat([dw, zero_w], axis=3)
    if absolute:
        dh = torch.abs(dh)
        dw = torch.abs(dw)
    if square:
        dh = dh ** 2
        dw = dw ** 2
    return dh, dw

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def matmul3(mat1, mat2, mat3):
    return torch.matmul(mat1, torch.matmul(mat2, mat3))

def matmul4(mat1, mat2, mat3, mat4):
    return torch.matmul(torch.matmul(mat1, torch.matmul(mat2, mat3)), mat4)

def downsample(img, factor):
    down = torch.nn.AvgPool2d(factor)
    img = down(img)
    return img

def downsample3D(vox, factor):
    down = torch.nn.AvgPool3d(factor)
    vox = down(vox)
    return vox

def downsample3Dflow(flow, factor):
    down = torch.nn.AvgPool3d(factor)
    flow = down(flow) * 1./factor
    return flow

def l2_normalize(x, dim=1):
    # dim1 is the channel dim
    return F.normalize(x, p=2, dim=dim)

def hard_argmax3D(tensor):
    B, Z, Y, X = list(tensor.shape)

    flat_tensor = tensor.reshape(B, -1)
    argmax = torch.argmax(flat_tensor, dim=1)

    # convert the indices into 3D coordinates
    argmax_z = argmax // (Y*X)
    argmax_y = (argmax % (Y*X)) // X
    argmax_x = (argmax % (Y*X)) % X

    argmax_z = argmax_z.reshape(B)
    argmax_y = argmax_y.reshape(B)
    argmax_x = argmax_x.reshape(B)
    return argmax_z, argmax_y, argmax_x

def argmax3D(heat, hard=False):
    B, Z, Y, X = list(heat.shape)

    if hard:
        # hard argmax
        loc_z, loc_y, loc_x = hard_argmax3D(heat)
        loc_z = loc_z.float()
        loc_y = loc_y.float()
        loc_x = loc_x.float()
    else:
        heat = heat.reshape(B, Z*Y*X)
        prob = torch.nn.functional.softmax(heat, dim=1)

        grid_z, grid_y, grid_x = meshgrid3D(B, Z, Y, X)

        grid_z = grid_z.reshape(B, -1)
        grid_y = grid_y.reshape(B, -1)
        grid_x = grid_x.reshape(B, -1)
        
        loc_z = torch.sum(grid_z*prob, dim=1)
        loc_y = torch.sum(grid_y*prob, dim=1)
        loc_x = torch.sum(grid_x*prob, dim=1)
        # these are B
    return loc_z, loc_y, loc_x
