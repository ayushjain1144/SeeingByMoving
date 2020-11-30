# import pretrained_nets_carla as pret_carla

exps = {}
groups = {}

############## dataset settings ##############

SIZE = 20
SIZE_val = 20
SIZE_test = 20

groups['carla_16-8-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['16-16-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*8)),
    'X = %d' % (int(SIZE*8)),
]
groups['32-16-32_bounds_train'] = [
    'XMIN = -32.0', # right (neg is left)
    'XMAX = 32.0', # right
    'YMIN = -16.0', # down (neg is up)
    'YMAX = 16.0', # down
    'ZMIN = -32.0', # forward
    'ZMAX = 32.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['16-8-16_bounds_train'] = [
    'XMIN = -16.0', # right (neg is left)
    'XMAX = 16.0', # right
    'YMIN = -8.0', # down (neg is up)
    'YMAX = 8.0', # down
    'ZMIN = -16.0', # forward
    'ZMAX = 16.0', # forward
    'Z = %d' % (int(SIZE*8)),
    'Y = %d' % (int(SIZE*4)),
    'X = %d' % (int(SIZE*8)),
]
groups['carla_16-8-16_bounds_val'] = [
    'XMIN_val = -16.0', # right (neg is left)
    'XMAX_val = 16.0', # right
    'YMIN_val = -8.0', # down (neg is up)
    'YMAX_val = 8.0', # down
    'ZMIN_val = -16.0', # forward
    'ZMAX_val = 16.0', # forward
    'Z_val = %d' % (int(SIZE_val*8)),
    'Y_val = %d' % (int(SIZE_val*4)),
    'X_val = %d' % (int(SIZE_val*8)),
]


############## preprocessing/shuffling ##############

############## modes ##############

groups['zoom'] = ['do_zoom = True']
groups['carla_mot'] = ['do_carla_mot = True']
groups['carla_static'] = ['do_carla_static = True']
groups['carla_flo'] = ['do_carla_flo = True']
groups['carla_reloc'] = ['do_carla_reloc = True']
groups['carla_obj'] = ['do_carla_obj = True']
groups['carla_focus'] = ['do_carla_focus = True']
groups['carla_track'] = ['do_carla_track = True']
groups['carla_siamese'] = ['do_carla_siamese = True']
groups['carla_genocc'] = ['do_carla_genocc = True']
groups['carla_gengray'] = ['do_carla_gengray = True']
groups['carla_vqrgb'] = ['do_carla_vqrgb = True']
groups['carla_vq3drgb'] = ['do_carla_vq3drgb = True']
groups['carla_precompute'] = ['do_carla_precompute = True']
groups['carla_propose'] = ['do_carla_propose = True']
groups['carla_det'] = ['do_carla_det = True']
groups['intphys_det'] = ['do_intphys_det = True']
groups['intphys_forecast'] = ['do_intphys_forecast = True']
groups['carla_forecast'] = ['do_carla_forecast = True']
groups['carla_pipe'] = ['do_carla_pipe = True']
groups['intphys_test'] = ['do_intphys_test = True']
groups['mujoco_offline'] = ['do_mujoco_offline = True']
groups['carla_pwc'] = ['do_carla_pwc = True']

############## extras ##############

groups['include_summs'] = [
    'do_include_summs = True',
]
groups['decay_lr'] = ['do_decay_lr = True']
groups['clip_grad'] = ['do_clip_grad = True']
# groups['quick_snap'] = ['snap_freq = 500']
# groups['quicker_snap'] = ['snap_freq = 50']
# groups['quickest_snap'] = ['snap_freq = 5']
groups['snap50'] = ['snap_freq = 50']
groups['snap100'] = ['snap_freq = 100']
groups['snap500'] = ['snap_freq = 500']
groups['snap1k'] = ['snap_freq = 1000']
groups['snap5k'] = ['snap_freq = 5000']

groups['no_shuf'] = ['shuffle_train = False',
                     'shuffle_val = False',
                     'shuffle_test = False',
]
groups['time_flip'] = ['do_time_flip = True']
groups['no_backprop'] = ['backprop_on_train = False',
                         'backprop_on_val = False',
                         'backprop_on_test = False',
]
groups['train_on_trainval'] = ['backprop_on_train = True',
                               'backprop_on_val = True',
                               'backprop_on_test = False',
]
groups['gt_ego'] = ['ego_use_gt = True']
groups['precomputed_ego'] = ['ego_use_precomputed = True']
groups['aug3d'] = ['do_aug3d = True']
groups['aug2D'] = ['do_aug2D = True']

groups['sparsify_pointcloud_10k'] = ['do_sparsify_pointcloud = 10000']
groups['sparsify_pointcloud_1k'] = ['do_sparsify_pointcloud = 1000']

groups['horz_flip'] = ['do_horz_flip = True']
groups['synth_rt'] = ['do_synth_rt = True']
groups['piecewise_rt'] = ['do_piecewise_rt = True']
groups['synth_nomotion'] = ['do_synth_nomotion = True']
groups['aug_color'] = ['do_aug_color = True']
# groups['eval'] = ['do_eval = True']
groups['eval_recall'] = ['do_eval_recall = True']
groups['eval_map'] = ['do_eval_map = True']
groups['no_eval_recall'] = ['do_eval_recall = False']
groups['save_embs'] = ['do_save_embs = True']
groups['save_ego'] = ['do_save_ego = True']
groups['save_vis'] = ['do_save_vis = True']
groups['save_outputs'] = ['do_save_outputs = True']

groups['profile'] = ['do_profile = True',
                     'log_freq_train = 100000000',
                     'log_freq_val = 100000000',
                     'log_freq_test = 100000000',
                     'max_iters = 20']

groups['B1'] = ['trainset_batch_size = 1']
groups['B2'] = ['trainset_batch_size = 2']
groups['B4'] = ['trainset_batch_size = 4']
groups['B6'] = ['trainset_batch_size = 6']
groups['B8'] = ['trainset_batch_size = 8']
groups['B10'] = ['trainset_batch_size = 10']
groups['B12'] = ['trainset_batch_size = 12']
groups['B16'] = ['trainset_batch_size = 16']
groups['B24'] = ['trainset_batch_size = 24']
groups['B32'] = ['trainset_batch_size = 32']
groups['B64'] = ['trainset_batch_size = 64']
groups['B128'] = ['trainset_batch_size = 128']
groups['vB1'] = ['valset_batch_size = 1']
groups['vB2'] = ['valset_batch_size = 2']
groups['vB4'] = ['valset_batch_size = 4']
groups['vB8'] = ['valset_batch_size = 8']
groups['lr0'] = ['lr = 0.0']
groups['lr1'] = ['lr = 1e-1']
groups['lr2'] = ['lr = 1e-2']
groups['lr3'] = ['lr = 1e-3']
groups['2lr4'] = ['lr = 2e-4']
groups['5lr4'] = ['lr = 5e-4']
groups['lr4'] = ['lr = 1e-4']
groups['lr5'] = ['lr = 1e-5']
groups['lr6'] = ['lr = 1e-6']
groups['lr7'] = ['lr = 1e-7']
groups['lr8'] = ['lr = 1e-8']
groups['lr9'] = ['lr = 1e-9']
groups['lr12'] = ['lr = 1e-12']
groups['1_iters'] = ['max_iters = 1']
groups['2_iters'] = ['max_iters = 2']
groups['3_iters'] = ['max_iters = 3']
groups['5_iters'] = ['max_iters = 5']
groups['6_iters'] = ['max_iters = 6']
groups['9_iters'] = ['max_iters = 9']
groups['21_iters'] = ['max_iters = 21']
groups['7_iters'] = ['max_iters = 7']
groups['10_iters'] = ['max_iters = 10']
groups['15_iters'] = ['max_iters = 15']
groups['18_iters'] = ['max_iters = 18']
groups['19_iters'] = ['max_iters = 19']
groups['20_iters'] = ['max_iters = 20']
groups['25_iters'] = ['max_iters = 25']
groups['30_iters'] = ['max_iters = 30']
groups['50_iters'] = ['max_iters = 50']
groups['71_iters'] = ['max_iters = 71']
groups['72_iters'] = ['max_iters = 72']
groups['73_iters'] = ['max_iters = 73']
groups['75_iters'] = ['max_iters = 75']
groups['100_iters'] = ['max_iters = 100']
groups['150_iters'] = ['max_iters = 150']
groups['200_iters'] = ['max_iters = 200']
groups['210_iters'] = ['max_iters = 210']
groups['250_iters'] = ['max_iters = 250']
groups['300_iters'] = ['max_iters = 300']
groups['397_iters'] = ['max_iters = 397']
groups['400_iters'] = ['max_iters = 400']
groups['447_iters'] = ['max_iters = 447']
groups['500_iters'] = ['max_iters = 500']
groups['850_iters'] = ['max_iters = 850']
groups['1000_iters'] = ['max_iters = 1000']
groups['2000_iters'] = ['max_iters = 2000']
groups['2445_iters'] = ['max_iters = 2445']
groups['3000_iters'] = ['max_iters = 3000']
groups['4000_iters'] = ['max_iters = 4000']
groups['4433_iters'] = ['max_iters = 4433']
groups['5000_iters'] = ['max_iters = 5000']
groups['10000_iters'] = ['max_iters = 10000']
groups['1k_iters'] = ['max_iters = 1000']
groups['2k_iters'] = ['max_iters = 2000']
groups['5k_iters'] = ['max_iters = 5000']
groups['10k_iters'] = ['max_iters = 10000']
groups['20k_iters'] = ['max_iters = 20000']
groups['30k_iters'] = ['max_iters = 30000']
groups['40k_iters'] = ['max_iters = 40000']
groups['50k_iters'] = ['max_iters = 50000']
groups['60k_iters'] = ['max_iters = 60000']
groups['80k_iters'] = ['max_iters = 80000']
groups['100k_iters'] = ['max_iters = 100000']
groups['100k10_iters'] = ['max_iters = 100010']
groups['200k_iters'] = ['max_iters = 200000']
groups['300k_iters'] = ['max_iters = 300000']
groups['400k_iters'] = ['max_iters = 400000']
groups['500k_iters'] = ['max_iters = 500000']

groups['resume'] = ['do_resume = True']
groups['reset_iter'] = ['reset_iter = True']

groups['log1'] = [
    'log_freq_train = 1',
    'log_freq_val = 1',
    'log_freq_test = 1',
]
groups['log5'] = [
    'log_freq_train = 5',
    'log_freq_val = 5',
    'log_freq_test = 5',
]
groups['log10'] = [
    'log_freq_train = 10',
    'log_freq_val = 10',
    'log_freq_test = 10',
]
groups['log50'] = [
    'log_freq_train = 50',
    'log_freq_val = 50',
    'log_freq_test = 50',
]
groups['log11'] = [
    'log_freq_train = 11',
    'log_freq_val = 11',
    'log_freq_test = 11',
]
groups['log53'] = [
    'log_freq_train = 53',
    'log_freq_val = 53',
    'log_freq_test = 53',
]
groups['log100'] = [
    'log_freq_train = 100',
    'log_freq_val = 100',
    'log_freq_test = 100',
]
groups['log500'] = [
    'log_freq_train = 500',
    'log_freq_val = 500',
    'log_freq_test = 500',
]
groups['log5000'] = [
    'log_freq_train = 5000',
    'log_freq_val = 5000',
    'log_freq_test = 5000',
]



groups['no_logging'] = [
    'log_freq_train = 100000000000',
    'log_freq_val = 100000000000',
    'log_freq_test = 100000000000',
]

