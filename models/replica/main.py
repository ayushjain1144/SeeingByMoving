from model_carla_viewmine_gt import CARLA_GT
from model_carla_viewmine import CARLA_VIEWMINE
from model_carla_proposal import CARLA_PROPOSAL
from model_nuscenes_explain import NUSCENES_EXPLAIN
from model_carla_compose import CARLA_COMPOSE
from model_nuscenes_compose import NUSCENES_COMPOSE
from model_carla_eval import CARLA_EVAL
from model_carla_semi import CARLA_SEMI
import hyperparams as hyp
import os
import cProfile
import logging
import ipdb
st = ipdb.set_trace

logger = logging.Logger('catch_all')

def main():
    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)
    
    if hyp.do_carla_static:
        log_dir_ = os.path.join("logs_carla_static", hyp.name)
    elif hyp.do_carla_flo:
        log_dir_ = os.path.join("logs_carla_flo", hyp.name)
    elif hyp.do_carla_time:
        log_dir_ = os.path.join("logs_carla_time", hyp.name)
    elif hyp.do_carla_reloc:
        log_dir_ = os.path.join("logs_carla_reloc", hyp.name)
    elif hyp.do_carla_sub:
        log_dir_ = os.path.join("logs_carla_sub", hyp.name)
    elif hyp.do_carla_sob:
        log_dir_ = os.path.join("logs_carla_sob", hyp.name)
    elif hyp.do_carla_explain:
        log_dir_ = os.path.join("logs_carla_explain", hyp.name)
    elif hyp.do_carla_ego:
        log_dir_ = os.path.join("logs_carla_ego", hyp.name)
    elif hyp.do_kitti_ego:
        log_dir_ = os.path.join("logs_kitti_ego", hyp.name)
    elif hyp.do_kitti_entity:
        log_dir_ = os.path.join("logs_kitti_entity", hyp.name)
    elif hyp.do_carla_entity:
        log_dir_ = os.path.join("logs_carla_entity", hyp.name)
    elif hyp.do_carla_slot2d:
        log_dir_ = os.path.join("logs_carla_slot2d", hyp.name)
    elif hyp.do_carla_slot3d:
        log_dir_ = os.path.join("logs_carla_slot3d", hyp.name)
    elif hyp.do_carla_proposal:
        log_dir_ = os.path.join("logs_carla_proposal", hyp.name)
    elif hyp.do_nuscenes_slot3d:
        log_dir_ = os.path.join("logs_nuscenes_slot3d", hyp.name)
    elif hyp.do_carla_render:
        log_dir_ = os.path.join("logs_carla_render", hyp.name)
    elif hyp.do_carla_resolve:
        log_dir_ = os.path.join("logs_carla_resolve", hyp.name)
    elif hyp.do_carla_minko:
        log_dir_ = os.path.join("logs_carla_minko", hyp.name)
    elif hyp.do_carla_proto:
        log_dir_ = os.path.join("logs_carla_proto", hyp.name)
    elif hyp.do_carla_goodvar:
        log_dir_ = os.path.join("logs_carla_goodvar", hyp.name)
    elif hyp.do_kitti_explain:
        log_dir_ = os.path.join("logs_kitti_explain", hyp.name)
    elif hyp.do_nuscenes_explain:
        log_dir_ = os.path.join("logs_nuscenes_explain", hyp.name)
    elif hyp.do_carla_free:
        log_dir_ = os.path.join("logs_carla_free", hyp.name)
    elif hyp.do_kitti_free:
        log_dir_ = os.path.join("logs_kitti_free", hyp.name)
    elif hyp.do_carla_obj:
        log_dir_ = os.path.join("logs_carla_obj", hyp.name)
        log_dir_ = os.path.join("logs_carla_obj2", hyp.name)
    elif hyp.do_carla_focus:
        log_dir_ = os.path.join("logs_carla_focus", hyp.name)
    elif hyp.do_carla_track:
        log_dir_ = os.path.join("logs_carla_track", hyp.name)
    elif hyp.do_carla_siamese:
        # log_dir_ = os.path.join("logs_carla_siamese", hyp.name)
        log_dir_ = os.path.join("logs_carla_siamese_rob", hyp.name)
    elif hyp.do_carla_vsiamese:
        log_dir_ = os.path.join("logs_carla_vsiamese", hyp.name)
    elif hyp.do_carla_rsiamese:
        log_dir_ = os.path.join("logs_carla_rsiamese", hyp.name)
    elif hyp.do_carla_msiamese:
        log_dir_ = os.path.join("logs_carla_msiamese", hyp.name)
    elif hyp.do_carla_ssiamese:
        log_dir_ = os.path.join("logs_carla_ssiamese", hyp.name)
    elif hyp.do_carla_csiamese:
        log_dir_ = os.path.join("logs_carla_csiamese", hyp.name)
    elif hyp.do_carla_genocc:
        log_dir_ = os.path.join("logs_carla_genocc", hyp.name)
    elif hyp.do_carla_gengray:
        log_dir_ = os.path.join("logs_carla_gengray", hyp.name)
    elif hyp.do_carla_vqrgb:
        log_dir_ = os.path.join("logs_carla_vqrgb", hyp.name)
    elif hyp.do_carla_moc:
        log_dir_ = os.path.join("logs_carla_moc", hyp.name)
    elif hyp.do_carla_don:
        log_dir_ = os.path.join("logs_carla_don", hyp.name)
    elif hyp.do_kitti_don:
        log_dir_ = os.path.join("logs_kitti_don", hyp.name)
    elif hyp.do_kitti_moc:
        log_dir_ = os.path.join("logs_kitti_moc", hyp.name)
    elif hyp.do_carla_zoom:
        log_dir_ = os.path.join("logs_carla_zoom", hyp.name)
    elif hyp.do_kitti_zoom:
        log_dir_ = os.path.join("logs_kitti_zoom", hyp.name)
    elif hyp.do_kitti_siamese:
        log_dir_ = os.path.join("logs_kitti_siamese", hyp.name)
    elif hyp.do_carla_ml:
        log_dir_ = os.path.join("logs_carla_ml", hyp.name)
    elif hyp.do_carla_bottle:
        log_dir_ = os.path.join("logs_carla_bottle", hyp.name)
    elif hyp.do_carla_pretty:
        log_dir_ = os.path.join("logs_carla_pretty", hyp.name)
    elif hyp.do_carla_compose:
        log_dir_ = os.path.join("logs_carla_compose", hyp.name)
    elif hyp.do_nuscenes_compose:
        log_dir_ = os.path.join("logs_nuscenes_compose", hyp.name)
    elif hyp.do_carla_occ:
        log_dir_ = os.path.join("logs_carla_occ", hyp.name)
    elif hyp.do_carla_bench:
        log_dir_ = os.path.join("logs_carla_bench", hyp.name)
    elif hyp.do_carla_reject:
        log_dir_ = os.path.join("logs_carla_reject", hyp.name)
    elif hyp.do_carla_auto:
        log_dir_ = os.path.join("logs_carla_auto", hyp.name)
    elif hyp.do_carla_ret:
        log_dir_ = os.path.join("logs_carla_ret", hyp.name)
    elif hyp.do_carla_vq3drgb:
        log_dir_ = os.path.join("logs_carla_vq3drgb", hyp.name)
    elif hyp.do_clevr_vq3drgb:
        log_dir_ = os.path.join("logs_clevr_vq3drgb", hyp.name)
    elif hyp.do_clevr_gen3dvq:
        log_dir_ = os.path.join("logs_clevr_gen3dvq", hyp.name)
    elif hyp.do_carla_gen3dvq:
        log_dir_ = os.path.join("logs_carla_gen3dvq", hyp.name)
    elif hyp.do_carla_precompute:
        log_dir_ = os.path.join("logs_carla_precompute", hyp.name)
    elif hyp.do_carla_propose:
        
        # ## this dir is trying to learn some ok detnets:
        # log_dir_ = os.path.join("logs_carla_propose", hyp.name)

        ## this dir is for doing the 5-crit eval
        # log_dir_ = os.path.join("logs_carla_propose2", hyp.name)

        ## aws
        log_dir_ = os.path.join("logs_carla_propose", hyp.name)
        
    elif hyp.do_carla_det:
        log_dir_ = os.path.join("logs_carla_det", hyp.name)
    elif hyp.do_intphys_det:
        log_dir_ = os.path.join("logs_intphys_det", hyp.name)
    elif hyp.do_intphys_forecast:
        log_dir_ = os.path.join("logs_intphys_forecast", hyp.name)
    elif hyp.do_carla_forecast:
        log_dir_ = os.path.join("logs_carla_forecast", hyp.name)
    elif hyp.do_carla_pipe:
        log_dir_ = os.path.join("logs_carla_pipe", hyp.name)
    elif hyp.do_intphys_test:
        log_dir_ = os.path.join("logs_intphys_test", hyp.name)
    elif hyp.do_carla_pwc:
        log_dir_ = os.path.join("logs_carla_pwc", hyp.name)
    elif hyp.do_carla_viewmine:
        log_dir_ = os.path.join("logs_carla_viewmine", hyp.name)
    elif hyp.do_carla_eval:
        log_dir_ = os.path.join("logs_carla_eval", hyp.name)
    elif hyp.do_carla_gt:
        log_dir_ = os.path.join("logs_carla_gt", hyp.name)
    elif hyp.do_carla_semi:
        log_dir_ = os.path.join("logs_carla_semi", hyp.name)
    else:
        assert(False) # what mode is this?

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    try:
        if hyp.do_carla_static:
            model = CARLA_STATIC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_flo:
            model = CARLA_FLO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_time:
            model = CARLA_TIME(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_reloc:
            model = CARLA_RELOC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_sub:
            model = CARLA_SUB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_gt:
            model = CARLA_GT(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_sob:
            model = CARLA_SOB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_explain:
            model = CARLA_EXPLAIN(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ego:
            model = CARLA_EGO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_ego:
            model = KITTI_EGO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_entity:
            model = KITTI_ENTITY(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_entity:
            model = CARLA_ENTITY(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_slot2d:
            model = CARLA_SLOT2D(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_slot3d:
            # st()
            model = CARLA_SLOT3D(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_proposal:
            model = CARLA_PROPOSAL(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_nuscenes_slot3d:
            model = NUSCENES_SLOT3D(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_render:
            model = CARLA_RENDER(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_resolve:
            model = CARLA_RESOLVE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_minko:
            model = CARLA_MINKO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_proto:
            model = CARLA_PROTO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_goodvar:
            model = CARLA_GOODVAR(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_explain:
            model = KITTI_EXPLAIN(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_nuscenes_explain:
            model = NUSCENES_EXPLAIN(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_free:
            model = CARLA_FREE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_free:
            model = KITTI_FREE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_obj:
            model = CARLA_OBJ(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_focus:
            model = CARLA_FOCUS(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_track:
            model = CARLA_TRACK(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_siamese:
            model = CARLA_SIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_vsiamese:
            model = CARLA_VSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_rsiamese:
            model = CARLA_RSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_msiamese:
            model = CARLA_MSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ssiamese:
            model = CARLA_SSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_csiamese:
            model = CARLA_CSIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_genocc:
            model = CARLA_GENOCC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_gengray:
            model = CARLA_GENGRAY(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_vqrgb:
            model = CARLA_VQRGB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_vq3drgb:
            model = CARLA_VQ3DRGB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_moc:
            model = CARLA_MOC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_don:
            model = CARLA_DON(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_don:
            model = KITTI_DON(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_moc:
            model = KITTI_MOC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_zoom:
            model = CARLA_ZOOM(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_zoom:
            model = KITTI_ZOOM(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_kitti_siamese:
            model = KITTI_SIAMESE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ml:
            model = CARLA_ML(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_bottle:
            model = CARLA_BOTTLE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_pretty:
            model = CARLA_PRETTY(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_compose:
            model = CARLA_COMPOSE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_nuscenes_compose:
            model = NUSCENES_COMPOSE(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_occ:
            model = CARLA_OCC(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_bench:
            model = CARLA_BENCH(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_reject:
            model = CARLA_REJECT(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_auto:
            model = CARLA_AUTO(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_ret:
            model = CARLA_RET(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_clevr_vq3drgb:
            model = CLEVR_VQ3DRGB(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_clevr_gen3dvq:
            model = CLEVR_GEN3DVQ(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_gen3dvq:
            model = CARLA_GEN3DVQ(
                checkpoint_dir=checkpoint_dir_,
                log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_precompute:
            model = CARLA_PRECOMPUTE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_propose:
            model = CARLA_PROPOSE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_det:
            model = CARLA_DET(checkpoint_dir=checkpoint_dir_, log_dir=log_dir_)
            model.go()
        elif hyp.do_intphys_det:
            model = INTPHYS_DET(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_intphys_forecast:
            model = INTPHYS_FORECAST(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_forecast:
            model = CARLA_FORECAST(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_pipe:
            model = CARLA_PIPE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_intphys_test:
            model = INTPHYS_TEST(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_pwc:
            model = CARLA_PWC(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_viewmine:
            # st()
            model = CARLA_VIEWMINE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_eval:
            model = CARLA_EVAL(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_semi:
            model = CARLA_SEMI(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        else:
            assert(False) # what mode is this?

    except (Exception, KeyboardInterrupt) as ex:
        logger.error(ex, exc_info=True)
        log_cleanup(log_dir_)

def log_cleanup(log_dir_):
    log_dirs = []
    for set_name in hyp.set_names:
        log_dirs.append(log_dir_ + '/' + set_name)

    for log_dir in log_dirs:
        for r, d, f in os.walk(log_dir):
            for file_dir in f:
                file_dir = os.path.join(log_dir, file_dir)
                file_size = os.stat(file_dir).st_size
                if file_size == 0:
                    os.remove(file_dir)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')

