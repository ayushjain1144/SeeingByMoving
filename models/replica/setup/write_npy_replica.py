import os
import os
import sys
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.append(os.getcwd())
sys.path.append('/hdd/ayushj/pytorch_disco')
os.environ["MODE"] = "CLEVR_STA"
os.environ["exp_name"] = "trainer_quantize_object_no_detach_rotate"
os.environ["run_name"] = "check"
import numpy as np 
import socket
# import open3d as o3d
import pickle
import h5py 
from time import time
import quaternion
from PIL import Image
# from lib_classes import Nel_Utils as nlu
import cv2
import matplotlib

import matplotlib.pyplot as plt
import torch
import utils_geom
import utils_pointcloud
import utils_improc
from scipy.misc import imresize
import ipdb
import random
import math
st = ipdb.set_trace
hostname = socket.gethostname()
'''
Mod info
aa - Normal with no thresholding on bboxes
bb - added area and occlusion threshold. Removed some objects (ignore classes). Each object should be in 5 views to be considered.
'''

base_dir = "/projects/katefgroup/viewpredseg/replica_novel_categories"
file_dir = "/projects/katefgroup/viewpredseg/replica_novel_categories_processed/npy"
dumpmod = "aa"

dump_dir = os.path.join(file_dir, dumpmod)

if os.path.exists(os.path.join(file_dir, dumpmod)):
    print("This datamod already exists. Terminating")
    exit(1)

'''
origin -> habitatCamX -> camX
habitatCamX has z and y opposite of pdisco camX i.e. 180 degree rotated along x axis (pitch)
'''
class write_habitat_to_npy():
    def __init__(self):
        self.H = 256
        self.W = 256
        # self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.scene_dirs = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        self.visualize = False
        self.device = torch.device("cuda")
        # https://github.com/facebookresearch/habitat-sim/issues/80
        self.fov = 90
        self.num_camR_candidates = 3
        self.bbox_area_thresh = 1100
        self.occlusion_thresh = 0.1
        # self.ignore_classes = ['bathtub', 'bed', 'book', 'bottle', 'bowl', 'candle', 'chopping-board', 'computer-keyboard', 'door', 'faucet', 'kitchen-utensil', 'knife-block', 'major-appliance', 'mouse', 'pan', 'pillow', 'plate', 'remote-control', 'shoe', 'small-appliance', 'stair', 'tablet', 'toothbrush', 'utensil-holder', 'wall']
        self.ignore_classes = []
        #quaternion(0.707106781186548, -0.707106781186547, -0, -0)

    def get_habitat_pix_T_camX(self):
        hfov = float(self.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        return pix_T_camX
    
    def get_pix_T_camX(self):
        pix_T_camX = self.get_habitat_pix_T_camX()
        pix_T_camX[0,2] = self.W/2.
        pix_T_camX[1,2] = self.H/2.
        return pix_T_camX

    def get_habitatCamX_T_camX(self):
        # Rotate 180 degrees along x axis
        rot_pi = np.eye(4)
        rot_pi[1,1] = -1
        rot_pi[2,2] = -1
        habitatCamX_T_camX = rot_pi
        return habitatCamX_T_camX

    def get_origin_T_camX(self, pos, rot):
        rotation_0 = quaternion.as_rotation_matrix(rot)
        origin_T_habitatCamX = np.eye(4)
        origin_T_habitatCamX[0:3,0:3] = rotation_0
        origin_T_habitatCamX[0:3,3] = pos
        
        return origin_T_habitatCamX @ self.get_habitatCamX_T_camX()
    
    def generate_xyz_habitatCamXs(self, depth_camXs, rgb_camXs, pix_T_camXs):
        xyz_camXs = []
        for i in range(depth_camXs.shape[0]):

            K = pix_T_camXs[i]
            xs, ys = np.meshgrid(np.linspace(-1*self.W/2.,1*self.W/2.,self.W), np.linspace(1*self.W/2.,-1*self.W/2.,self.W))
            depth = depth_camXs[i].reshape(1,self.W,self.W)
            xs = xs.reshape(1,self.W,self.W)
            ys = ys.reshape(1,self.W,self.W)

            xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
            xys = xys.reshape(4, -1)
            xy_c0 = np.matmul(np.linalg.inv(K), xys)
            xyz_camX = xy_c0.T[:,:3]
            xyz_camXs.append(xyz_camX)
            if self.visualize and False:
                pcd = nlu.make_pcd(xyz_camX)
                o3d.visualization.draw_geometries([pcd, self.mesh_frame])
        return np.stack(xyz_camXs)

        

    def test_bbox_projection(self, xyz_camXs_origin_agg, origin_T_camXs, pix_T_camXs, rgb_camX, xyz_camXs, f):
        rgb_camX = rgb_camX.astype(np.float32)
        
        objs_info = f['objects_info']
        for obj_info in objs_info:
            if obj_info['category_name'] == "chair":

                bbox_center = obj_info['bbox_center']
                bbox_size = obj_info['bbox_size']
                print("bbox center and size are: ", bbox_center, bbox_size)

        xmin, xmax = bbox_center[0]-bbox_size[0]/2., bbox_center[0]+bbox_size[0]/2.
        ymin, ymax = bbox_center[1]-bbox_size[1]/2., bbox_center[1]+bbox_size[1]/2.
        zmin, zmax = bbox_center[2]-bbox_size[2]/2., bbox_center[2]+bbox_size[2]/2.

        bbox_origin_ends = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
        bbox_origin_theta = nlu.get_alignedboxes2thetaformat(torch.tensor(bbox_origin_ends).reshape(1, 1, 2, 3).float())
        bbox_origin_corners = utils_geom.transform_boxes_to_corners(bbox_origin_theta)

        nlu.only_visualize(nlu.make_pcd(xyz_camXs_origin_agg.numpy()), bbox_origin_ends.reshape(1,-1))

        print("Ends of bbox in origin are: ", bbox_origin_ends)

        camX_T_origin = utils_geom.safe_inverse(torch.tensor(origin_T_camXs).unsqueeze(0)).float()
        bbox_corners_camX = utils_geom.apply_4x4(camX_T_origin, bbox_origin_corners.squeeze(0).float())
        bbox_ends_camX = nlu.get_ends_of_corner(bbox_corners_camX.permute(0,2,1)).permute(0,2,1)
        
        ends_camX = bbox_ends_camX.reshape(1,-1).numpy()
        print("ends in camX are: ", ends_camX)

        nlu.only_visualize(nlu.make_pcd(xyz_camXs), ends_camX)
        plt.imshow(rgb_camX)
        plt.show(block=True)
        utils_pointcloud.draw_boxes_on_rgb(rgb_camX, pix_T_camXs, ends_camX, visualize=True)

    def get_object_info(self, f, rgb_camX, pix_T_camX, origin_T_camX):
        # st()
        objs_info = f['objects_info']
        object_dict = {}
        if self.visualize:
            plt.imshow(rgb_camX[...,:3])
            plt.show(block=True)
        for obj_info in objs_info:
            classname = obj_info['category_name']
            # if classname in self.ignore_classes:
            #     continue
            category = obj_info['category_id']
            instance_id = obj_info['instance_id']
            bbox_center = obj_info['bbox_center']
            bbox_size = obj_info['bbox_size']
            #bbox_world_T_local = obj_info['world_T_local']
            mask_2d = obj_info['mask_2d']
            box_2d = obj_info['box_2d']

            # st()


            # pix_T_camX = torch.tensor(pix_T_camX).float()
            # origin_T_camX = torch.tensor(origin_T_camX).float()
            
            
            # bbox_size = torch.tensor(bbox_size).reshape(1, 1, 3)
            # bbox_world_T_local = torch.tensor(bbox_world_T_local).reshape(1, 1, 4, 4)
            
            # lx, ly, lz = torch.unbind(bbox_size, axis=2)
            # xs = torch.stack([bbox_center[0]-lx/2., bbox_center[0]-lx/2., bbox_center[0]-lx/2., bbox_center[0]-lx/2., bbox_center[0]+lx/2., bbox_center[0]+lx/2., bbox_center[0]+lx/2., bbox_center[0]+lx/2.], axis=2)
            # ys = torch.stack([bbox_center[1]-ly/2., bbox_center[1]-ly/2., bbox_center[1]+ly/2., bbox_center[1]+ly/2., bbox_center[1]-ly/2., bbox_center[1]-ly/2., bbox_center[1]+ly/2., bbox_center[1]+ly/2.], axis=2)
            # zs = torch.stack([bbox_center[2]-lz/2., bbox_center[2]+lz/2., bbox_center[2]-lz/2., bbox_center[2]+lz/2., bbox_center[2]-lz/2., bbox_center[2]+lz/2., bbox_center[2]-lz/2., bbox_center[2]+lz/2.], axis=2)
            # # these are B x N x 8
            # xyzlist = torch.stack([xs, ys, zs], axis=3)
            # rtlist_ = bbox_world_T_local.reshape(1, 4, 4)
            # xyzlist_obj_ = xyzlist.reshape(1, 8, 3)
            # xyzlist_origin_ = utils_geom.apply_4x4(rtlist_, xyzlist_obj_)
            # xyzlist_origin = xyzlist_origin_.reshape(1, 1, 8, 3)
            # xyz_box = xyzlist_origin.reshape(8, 3)
            # camX_T_origin = utils_geom.safe_inverse(origin_T_camX.unsqueeze(0))
            # xyzlist_camX = utils_geom.apply_4x4_to_corners(camX_T_origin, xyzlist_origin).reshape(8, 3)

            # lrt_box = utils_geom.merge_lrtlist(bbox_size, bbox_world_T_local)
            # camX_T_origin = utils_geom.safe_inverse(origin_T_camX.unsqueeze(0))
            # lrt_box_camX = utils_geom.apply_4x4_to_lrtlist(camX_T_origin, lrt_box)

            # xyz_box = utils_geom.get_xyzlist_from_lrtlist(lrt_box).reshape(8, 3)
            # xmin, ymin, zmin = torch.min(xyz_box, axis=0)[0]
            # xmax, ymax, zmax = torch.max(xyz_box, axis=0)[0]
            # bbox_origin_ends = torch.tensor([xmin, ymin, zmin, xmax, ymax, zmax]).reshape(1, 1, 2, 3).float()

            # visualise
            # st()
            # corners_pix = utils_geom.get_boxlist2d_from_lrtlist(pix_T_camX.unsqueeze(0), lrt_box_camX, 256, 256)[0][0]
            # obj_bboxes_xy_s = utils_geom.Camera2Pixels(xyzlist_camX.unsqueeze(0), pix_T_camX.unsqueeze(0))
            # obj_bboxes_x_s, obj_bboxes_y_s = torch.unbind(obj_bboxes_xy_s, dim=2)
            # obj_bboxes_x_s = obj_bboxes_x_s.reshape(1, 8).cpu().numpy()
            # obj_bboxes_y_s = obj_bboxes_y_s.reshape(1, 8).cpu().numpy()

            # obj_bboxes_x_min = np.min(obj_bboxes_x_s, axis=1)
            # obj_bboxes_x_max = np.max(obj_bboxes_x_s, axis=1)
            # obj_bboxes_y_min = np.min(obj_bboxes_y_s, axis=1)
            # obj_bboxes_y_max = np.max(obj_bboxes_y_s, axis=1)

            # ymin, xmin, ymax, xmax = obj_bboxes_y_min[0], obj_bboxes_x_min[0], obj_bboxes_y_max[0], obj_bboxes_x_max[0]
            if self.visualize:
                ymin, xmin, ymax, xmax = box_2d
                H = 256
                W = 256
                # ymin, ymax = ymin*H, ymax*H
                # xmin, xmax = xmin*W, xmax*W
                # st()
                # rgb = np.transpose(rgb_camX, [1, 2, 0]) # put channels last
                rgb = cv2.cvtColor(rgb_camX[...,:3], cv2.COLOR_RGB2BGR)

                color_map = matplotlib.cm.get_cmap('tab20')
                color_map = color_map.colors
                color = color_map[1]
                xmin = np.clip(int(xmin), 0,  W-1)
                xmax = np.clip(int(xmax), 0,  W-1)
                ymin = np.clip(int(ymin), 0,  H-1)
                ymax = np.clip(int(ymax), 0,  H-1)
                thickness = 1
                # st()
                cv2.line(rgb, (xmin, ymin), (xmin, ymax), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmin, ymin), (xmax, ymin), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmax, ymin), (xmax, ymax), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmax, ymax), (xmin, ymax), color, thickness, cv2.LINE_AA) 


                rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
                # st()
                
                cv2.imwrite(f'/hdd/ayushj/image_dump/{classname}_box.png', rgb)
                # masked_rgb = cv2.bitwise_and(rgb, np.uint8(mask_2d))
                # cv2.imwrite(f'/hdd/ayushj/image_dump/{classname}_mask.png', masked_rgb)
                # pixels = np.array(rgb)
                plt.imshow(rgb)
                plt.show(block=True)
                cv2.waitKey(0)
            xmin, xmax = bbox_center[0]-bbox_size[0]/2., bbox_center[0]+bbox_size[0]/2.
            ymin, ymax = bbox_center[1]-bbox_size[1]/2., bbox_center[1]+bbox_size[1]/2.
            zmin, zmax = bbox_center[2]-bbox_size[2]/2., bbox_center[2]+bbox_size[2]/2.
            bbox_volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
            
            bbox_origin_ends = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
            # bbox_origin_ends = torch.tensor(bbox_origin_ends).reshape(1,1,2,3).float()
            # bbox_origin_theta = nlu.get_alignedboxes2thetaformat(bbox_origin_ends)

            # bbox_origin_corners = utils_geom.transform_boxes_to_corners(bbox_origin_theta).float()
            # camX_T_origin = utils_geom.safe_inverse(torch.tensor(origin_T_camX).unsqueeze(0)).float().cuda()
            # bbox_corners_camX = utils_geom.apply_4x4(camX_T_origin.float(), bbox_origin_corners.squeeze(0).float())
            # bbox_corners_pixX = utils_geom.apply_pix_T_cam(torch.tensor(pix_T_camX).unsqueeze(0).float().cuda(), bbox_corners_camX.cuda())
            # bbox_ends_pixX = nlu.get_ends_of_corner(bbox_corners_pixX.permute(0,2,1)).permute(0,2,1)
           
            
            # bbox_ends_pixX_np = torch.clamp(bbox_ends_pixX.squeeze(0), 0, rgb_camX.shape[1]).cpu().numpy().astype(int)
            # bbox_area = (bbox_ends_pixX_np[1,1] - bbox_ends_pixX_np[0,1])*(bbox_ends_pixX_np[1,0] - bbox_ends_pixX_np[0,0])
            # print("Volume and area occupied by class {} is {} and {}".format(classname, bbox_volume, bbox_area))

            
            # semantic = f['semantic_camX']
            # instance_id_pixel_cnt = np.where(semantic == instance_id)[0].shape
            # object_to_bbox_ratio = instance_id_pixel_cnt/bbox_area
            # print("Num pixels in semantic map {}. Ratio of pixels to bbox area{}. Ratio of pixels to bbox volume {}. ".format(instance_id_pixel_cnt, object_to_bbox_ratio, instance_id_pixel_cnt/bbox_volume))
            # if self.visualize:
            #     # print("bbox ends are: ", bbox_ends_pixX_np)
            #     cropped_rgb = rgb_camX[bbox_ends_pixX_np[0,1]:bbox_ends_pixX_np[1,1], bbox_ends_pixX_np[0,0]:bbox_ends_pixX_np[1,0], :3]
            #     plt.imshow(cropped_rgb)
            #     plt.show(block=True)
            # if bbox_area < self.bbox_area_thresh:
            #     continue
            
            # if object_to_bbox_ratio < self.occlusion_thresh:
            #     continue
            object_dict[instance_id] = (classname, category, instance_id, bbox_origin_ends, box_2d, mask_2d)

        return object_dict
    
    def select_frequently_occuring_objects(self, scene_object_dict):
        scene_bbox_ends = []
        scene_category_ids = []
        scene_category_names = []
        scene_instance_ids = []
        scene_box2d = []
        scene_mask2d = []
        for key in scene_object_dict:
            # if len(scene_object_dict[key]) < 5:
            #     continue
            scene_instance_ids.append(key)
            scene_category_names.append(scene_object_dict[key][0][0])
            scene_category_ids.append(scene_object_dict[key][0][1])
            scene_bbox_ends.append(scene_object_dict[key][0][3])
            scene_box2d.append(scene_object_dict[key][0][4])
            scene_mask2d.append(scene_object_dict[key][0][5])
        print("Number of objects retained in scene: ", len(scene_category_ids))
        return scene_category_ids, scene_instance_ids, scene_category_names, scene_bbox_ends, scene_box2d, scene_mask2d             


    def process(self):
        fnames = []
        # st()
        for apartments in self.scene_dirs:
            apartment_name = os.path.basename(os.path.normpath(apartments))
            print(apartment_name)
            apartment_dirs = [os.path.join(apartments, f) for f in os.listdir(apartments) if os.path.isdir(os.path.join(apartments, f))]
            for scene_cnt, scene_dir in enumerate(apartment_dirs):
                # st()
                print("Processing scene {}. Scene number {}".format(scene_dir, scene_cnt))
                
                rgb_camXs = []
                depth_camXs = []
                pix_T_camXs = []
                origin_T_camXs = []
                xyz_camXs = []
                habitat_pix_T_camXs = []
                object_info_s_list = []
                pickle_files = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if f.endswith('.p')]
                pickle_files = sorted(pickle_files)
                for cnt, pickle_file in enumerate(pickle_files):
                    print("count of pickle file is: ", cnt)
                    f = pickle.load(open(pickle_file, "rb"))
                    if cnt==0:
                        sample_f = f

                    rgb_camXs.append(f['rgb_camX'])
                    depth_camXs.append(f['depth_camX'])
                    pix_T_camXs.append(self.get_pix_T_camX())
                    habitat_pix_T_camXs.append(self.get_habitat_pix_T_camX())
                    origin_T_camXs.append(self.get_origin_T_camX(f['sensor_pos'], f['sensor_rot']))
                    
                    object_info = self.get_object_info(f, rgb_camXs[-1], pix_T_camXs[-1], origin_T_camXs[-1])

                    object_info_s_list.append(object_info)

                habitat_pix_T_camXs = np.stack(habitat_pix_T_camXs)
                rgb_camXs_to_save = np.stack(rgb_camXs)[:,:,:,:3]
                rgb_camXs = np.stack(rgb_camXs)[:,:,:,:3].astype(np.float32)/255.
                origin_T_camXs = np.stack(origin_T_camXs)
                depth_camXs = np.stack(depth_camXs)
                pix_T_camXs = np.stack(pix_T_camXs)
                xyz_habitatCamXs = self.generate_xyz_habitatCamXs(depth_camXs, rgb_camXs, habitat_pix_T_camXs)

                if self.visualize:
                    print("Showing pointclouds in habitat_camXs coordinate ref frame")
                    for xyz_habitatCamX in xyz_habitatCamXs:
                        pcd = nlu.make_pcd(xyz_habitatCamX)
                        o3d.visualization.draw_geometries([pcd, self.mesh_frame])

                # Get xyz_camXs in pydisco coordinate frame.
                # Since its 180 deg rotation, habitatCamX_T_camX and it's inverse will be same. Therefore, not taking inv.
                xyz_camXs = utils_geom.apply_4x4(torch.tensor(self.get_habitatCamX_T_camX()).repeat(xyz_habitatCamXs.shape[0], 1, 1), torch.tensor(xyz_habitatCamXs)).numpy()

                if self.visualize:
                    print("Showing pointclouds in pydisco_camXs coordinate ref frame")
                    for xyz_camX, rgb_camX in zip(xyz_camXs, rgb_camXs):
                        pcd = nlu.make_pcd(xyz_camX)
                        o3d.visualization.draw_geometries([pcd, self.mesh_frame])

                        pix_T_camX = pix_T_camXs[0]
                        depth, _ = utils_geom.create_depth_image(torch.tensor(pix_T_camX).unsqueeze(0).float(), torch.tensor(xyz_camX).unsqueeze(0).float(), self.H, self.W)
                        depth[torch.where(depth>10)]=0
                        utils_pointcloud.visualize_colored_pcd(depth.squeeze(0).squeeze(0).numpy(), rgb_camX, pix_T_camX)
                        
                xyz_camXs_origin = utils_geom.apply_4x4(torch.tensor(origin_T_camXs), torch.tensor(xyz_camXs))
                xyz_camXs_origin_agg = xyz_camXs_origin.reshape(-1,3)

                # Visualize aggregated pointcloud
                if self.visualize:
                    print("Showing aggregated pointclouds")
                    pcd_list = [self.mesh_frame]
                    for xyz_camX_origin in xyz_camXs_origin:
                        pcd_list.append(nlu.make_pcd(xyz_camX_origin))
                    o3d.visualization.draw_geometries(pcd_list)
                
                if self.visualize:
                    self.test_bbox_projection(xyz_camXs_origin_agg, origin_T_camXs[0], pix_T_camXs[0], rgb_camXs[0], xyz_camXs[0], sample_f)
                
                # object_data = self.get_objects_in_scene()
                # First num_camR_candidates views will be our camR candidates
                for num_save in range(self.num_camR_candidates):
                    camX1_T_origin = utils_geom.safe_inverse(torch.tensor(origin_T_camXs[num_save]).unsqueeze(0)).float().repeat(origin_T_camXs.shape[0], 1, 1).numpy()
                    data_to_save = {"camR_index": num_save, "pix_T_cams_raw": pix_T_camXs, "camR_T_origin_raw": camX1_T_origin,
                                       "xyz_camXs_raw": xyz_camXs, "origin_T_camXs_raw": origin_T_camXs, 'rgb_camXs_raw': rgb_camXs_to_save,
                                       "object_info_s_list": object_info_s_list}
                    cur_epoch = str(time()).replace(".","")
                    pickle_fname = cur_epoch + ".p"
                    fnames.append(pickle_fname)
                    if not os.path.exists(os.path.join(dump_dir, apartment_name)):
                        os.makedirs(os.path.join(dump_dir, apartment_name))
                    # st()
                    with open(os.path.join(dump_dir, apartment_name, pickle_fname), 'wb') as f:
                        pickle.dump(data_to_save, f)
        #st()
        return fnames
            

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)   

if __name__ == "__main__":
    mkdir(dump_dir)
    datawriter = write_habitat_to_npy()
    fnames = datawriter.process()
    random.shuffle(fnames)
    train_len = int(len(fnames)*0.75)
    test_len = len(fnames) - train_len

    
    trainfile = open(os.path.join(file_dir, dumpmod+"t.txt"),"w")
    valfile = open(os.path.join(file_dir, dumpmod+"v.txt"),"w")

    for i in range(len(fnames)):
        if i < train_len:
            trainfile.write(os.path.join(dumpmod, fnames[i]))
            trainfile.write("\n")
        else:
            valfile.write(os.path.join(dumpmod, fnames[i]))
            valfile.write("\n")
    trainfile.close()
    valfile.close()



