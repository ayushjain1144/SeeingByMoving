from exp_base import *

############## choose an experiment ##############

# current = 'builder'
current = 'trainer'

mod = "'sm00'"
mod = "'sm01'"
mod = "'sm02'"
mod = "'sm03'"
mod = "'sm04'" # weak supervision file
mod = "'sm_cushion'"
mod = "'sm_cushion_val'"
mod = "'sm_cushion_test'"
mod = "'sm_nightstand_val'"
mod = "'sm_nightstand_train'"
mod = "'sm_nightstand_test'"
mod = "'sm_shelf_test'"
mod = "'sm_shelf_train'"
mod = "'sm_shelf_val'"
mod = "'sm_beanbag_val'"
mod = "'sm_beanbag_train'"
mod = "'sm_beanbag_test'"
############## exps ##############

exps['builder'] = [
    'carla_viewmine', # mode
    #'carla_stat_stav_data', # dataset
    'carla_multiview_train_data', # dataset
    # 'carla_32-16-32_bounds_train',
    # '16-16-16_bounds_train',
    '57_iters',
    # 'lr5',
    'B1',
    #'no_shuf',
    # 'no_backprop',
    #'train_geodesic',
    'train_crf',
    'pretrained_feat3d',
    'pretrained_occ',
    'pretrained_rgb',
    # 'train_feat3d',
    # 'train_occ',
    # 'train_render',
    'train_box3d',
    'log1',
]
exps['trainer'] = [
    'carla_semi', # mode
    #'carla_multiview_train_data', # dataset
    # 'carla_multiview_25views_2cars',
    #'carla_multiview_90views',
    # 'replica_sta_data',
    'replica_fixation_data_novel',
    # '16-16-16_bounds_train',
    'replica_3_3_6_bounds',
    '30_iters',
    #'90_iters',
    'lr4',
    'B1',
    #'train_geodesic',
    'train_crf',
    #'pretrained_feat3d',
    #'pretrained_occ',
    #'pretrained_rgb',
    'no_backprop',
    # 'pretrained_feat3d',
    # 'train_feat3d',
    # 'train_occ',
    # 'train_rgb',
    # 'train_render',
    'no_shuf',
    # 'log11',
    'log1',
]

############## groups ##############

groups['carla_semi'] = ['do_carla_semi = True']
groups['do_test'] = ['do_test = True']

groups['train_feat2d'] = [
    'do_feat2d = True',
    'feat2d_dim = 64',
]
groups['train_up2d'] = [
    'do_up2d = True',
    'up2d_dim = 3',
    'up2d_l1_coeff = 10.0',
    'up2d_vgg_coeff = 1.0',
]
groups['train_feat3d'] = [
    'do_feat3d = True',
    'feat3d_dim = 32',
]
groups['train_occ'] = [
    'do_occ = True',
    'occ_coeff = 1.0',
    'occ_smooth_coeff = 0.1',
]
groups['train_rgb'] = [
    'do_rgb = True',
    'rgb_l1_coeff = 1.0',
    'rgb_smooth_coeff = 0.1',
]
groups['train_cut'] = [
    'do_cut = True',
    'cut_coeff = 1.0',
]

# can add some hyperparams later
groups['train_geodesic'] = [
    'do_geodesic = True',
]
groups['train_crf'] = [
    'do_crf = True',
]
groups['train_box3d'] = [
    'do_box3dnet = True',

]
groups['train_render'] = [
    'do_render = True',
    'render_depth = 64',
    'render_rgb_coeff = 10.0',
    # 'render_depth_coeff = 0.1',
    # 'render_smooth_coeff = 0.01',
]

############## datasets ##############

# # dims for mem
# SIZE = 32
# Z = int(SIZE*4)
# Y = int(SIZE*1)
# X = int(SIZE*4)

K = 2 # how many objects to consider
N = 25 # how many objects per npz
S = 25 # number of frames
S_test = 100
H = 256
W = 256
# H and W for proj stuff
PH = int(H/2.0)
PW = int(W/2.0)
# PH = int(H)
# PW = int(W)

SIZE = 16
SIZE_val = 16
SIZE_test = 16
SIZE_zoom = 16

SIZE = 20
SIZE_val = 20
SIZE_test = 20
SIZE_zoom = 20

dataset_location = "/projects/katefgroup/datasets/carla/processed/npzs"
# dataset_location = "/data/carla/processed/npzs"
# dataset_location = "/projects/katefgroup/datasets/carla_odometry/processed"


###################### CARLA DATASETS ####################################
dataset_mod = 'ah'
groups['carla_multiview_train1_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "m%ss7i3one"' % dataset_mod,
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train10_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "m%ss7i3ten"' % dataset_mod,
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_train_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "smabs5i8t"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_multiview_90views'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "viewseg_multiview_vs01_s51_i1"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]
groups['carla_multiview_25views'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "viewseg_multiview_vs03_s43_i1"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

groups['carla_multiview_25views_2cars'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "viewseg_multiview_vs04_s41_i1"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "%s"' % dataset_location,
    'dataset_filetype = "npz"'
]

#################################################################################

################### REPLICA DATASETS ###########################################
groups['replica_sta_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "cct"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/viewpredseg/datasets/replica_processed/npy"',
    'dataset_filetype = "npz"'
]

groups['replica_fixation_data'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "train"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/viewpredseg/replica_dome_processed/npy"',
    'dataset_filetype = "npz"'
]

groups['replica_fixation_data_1'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "train"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/home/ayushj2/datasets"',
    'dataset_filetype = "npz"'
]

groups['replica_fixation_data_2dgt'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "train"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/viewpredseg/replica_dome_obb_processed/npy"',
    'dataset_filetype = "npz"'
]

groups['replica_fixation_data_center'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "test"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/viewpredseg/replica_dome_center_processed/npy"',
    'dataset_filetype = "npz"'
]
groups['replica_fixation_data_selfsup'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "train"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/viewpredseg/processed/replica_selfsup_processed/npy"',
    'dataset_filetype = "npz"'
]

# train: 276
# val: 27
# test: 30

groups['replica_fixation_data_novel'] = [
    'dataset_name = "carla"',
    'H = %d' % H,
    'W = %d' % W,
    'trainset = "test"', #smabs5i8t
    'trainset_format = "multiview"', 
    'trainset_seqlen = %d' % S, 
    'dataset_location = "/projects/katefgroup/viewpredseg/replica_novel_categories_processed/npy"',
    'dataset_filetype = "npz"'
]
#####################################################################################


############## verify and execute ##############

def _verify_(s):
    varname, eq, val = s.split(' ')
    assert varname in globals()
    assert eq == '='
    assert type(s) is type('')

print(current)
assert current in exps
for group in exps[current]:
    print("  " + group)
    assert group in groups
    for s in groups[group]:
        print("    " + s)
        _verify_(s)
        exec(s)

s = "mod = " + mod
_verify_(s)

exec(s)

