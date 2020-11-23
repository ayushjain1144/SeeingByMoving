import os
import numpy as np
import torch
import pickle
import sys
DATASET_DIR = '/projects/katefgroup/viewpredseg/replica_novel_categories_processed/npy'

train_file = open(os.path.join(DATASET_DIR, 'train.txt'), 'w')
val_file = open(os.path.join(DATASET_DIR, 'val.txt'), 'w')
test_file = open(os.path.join(DATASET_DIR, 'test.txt'), 'w')

VAL_APARTMENTS = ['frl_apartment_1_13']
TEST_APARTMENTS = ['frl_apartment_4_12', 'room_1_6']

# SKIP_APARTMENTS = ["frl_apartment_1_37"]
MAIN_DATASET_RELATIVE = 'aa'
MAIN_DATASET_ABSOLUTE = os.path.join(DATASET_DIR, MAIN_DATASET_RELATIVE)

for apartment in os.listdir(MAIN_DATASET_ABSOLUTE):
    print(apartment)

    APARTMENT_RELATIVE = os.path.join(MAIN_DATASET_RELATIVE, apartment)
    APARTMENT_ABSOLUTE = os.path.join(DATASET_DIR, APARTMENT_RELATIVE)

    print(APARTMENT_ABSOLUTE)

    for pickle_file in os.listdir(APARTMENT_ABSOLUTE):
        print(pickle_file)
        
        assert(pickle_file.endswith('.p'))

        PICKLE_FILE_RELATIVE  = os.path.join(APARTMENT_RELATIVE, pickle_file)
        PICKLE_FILE_ABSOLUTE = os.path.join(DATASET_DIR, PICKLE_FILE_RELATIVE)

        print(PICKLE_FILE_RELATIVE)

        if apartment in TEST_APARTMENTS:
            test_file.write(f"{PICKLE_FILE_RELATIVE}\n")
        elif apartment in VAL_APARTMENTS:
            val_file.write(f"{PICKLE_FILE_RELATIVE}\n")
        else:
            train_file.write(f"{PICKLE_FILE_RELATIVE}\n")

        # # for fixing pickle files -- commented out
        # d = np.load(PICKLE_FILE_ABSOLUTE, allow_pickle=True)
        # bbox_origin = d['bbox_origin']
        # # print(d['bbox_origin'])
        # for i, t in enumerate(bbox_origin):
        #     try:
        #         bbox_origin[i] = t.cpu().detach().numpy()
        #     except Exception as e:
        #         print(e)
        # d['bbox_origin'] = bbox_origin
        # # print(d['bbox_origin'])
        # # st
        # f = open(PICKLE_FILE_ABSOLUTE, 'wb')
        # pickle.dump(d, f)
        # # sys.exit()

train_file.close()
val_file.close()
test_file.close()





