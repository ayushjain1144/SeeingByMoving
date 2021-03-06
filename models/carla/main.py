from model_carla_viewmine import CARLA_VIEWMINE
from model_carla_eval import CARLA_EVAL
from model_carla_gt import CARLA_GT
import hyperparams as hyp
import os
import cProfile
import logging
import ipdb
st = ipdb.set_trace

logger = logging.Logger('catch_all')

def main():
    checkpoint_dir_ = os.path.join("checkpoints", hyp.name)
    
    if hyp.do_carla_viewmine:
        log_dir_ = os.path.join("logs_carla_viewmine", hyp.name)
    elif hyp.do_carla_eval:
        log_dir_ = os.path.join("logs_carla_eval", hyp.name)
    elif hyp.do_carla_gt:
        log_dir_ = os.path.join("logs_carla_gt", hyp.name)
    else:
        assert(False) # what mode is this?

    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(log_dir_):
        os.makedirs(log_dir_)

    try:

        if hyp.do_carla_viewmine:
            model = CARLA_VIEWMINE(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_eval:
            model = CARLA_EVAL(checkpoint_dir=checkpoint_dir_,
                    log_dir=log_dir_)
            model.go()
        elif hyp.do_carla_gt:
            model = CARLA_GT(checkpoint_dir=checkpoint_dir_,
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

