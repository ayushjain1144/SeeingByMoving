import ipdb
st = ipdb.set_trace
# st()
import habitat_sim
import habitat
print("HJ")
from habitat.config.default import get_config
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_angle_axis
import cv2
import math
import random
#%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(12 ,8))
import time
import numpy as np
import quaternion
import ipdb

import os 
import sys
import pickle
import json
from habitat_sim.utils import common as utils
# from habitat_sim.utils import viz_utils as vut
from scipy.spatial.transform import Rotation
EPSILON = 1e-8


class AutomatedMultiview():
    def __init__(self):   
        self.visualize = False
        self.verbose = False
        # st()
        self.mapnames = os.listdir('/home/nel/gsarch/Replica-Dataset/out/')
        # self.mapnames = os.listdir('/hdd/replica/Replica-Dataset/out/')
        self.num_episodes = len(self.mapnames)
        # self.num_episodes = 1 # temporary
        #self.ignore_classes = ['book','base-cabinet','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
        
        # self.include_classes = ['chair', 'bed', 'toilet', 'sofa', 'indoor-plant', 'refrigerator', 'tv-screen', 'table']
        # self.small_classes = ['indoor-plant', 'toilet']
        self.include_classes = ['beanbag', 'cushion', 'nightstand', 'shelf']
        self.small_classes = []
        self.rot_interval = 5.0
        self.radius_max = 2
        self.radius_min = 1
        self.num_flat_views = 3
        self.num_any_views = 7
        self.num_views = 25
        # self.env = habitat.Env(config=config, dataset=None)
        # st()
        # self.test_navigable_points()
        self.run_episodes()

    def run_episodes(self):

        for episode in range(self.num_episodes):
            print("STARTING EPISODE ", episode)
            # mapname = np.random.choice(self.mapnames)
            mapname = self.mapnames[episode] # KEEP THIS
            #mapname = 'apartment_0'
            #mapname = 'frl_apartment_4'
            self.test_scene = "/home/nel/gsarch/Replica-Dataset/out/{}/habitat/mesh_semantic.ply".format(mapname)
            self.object_json = "/home/nel/gsarch/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)
            # self.test_scene = "/hdd/replica/Replica-Dataset/out/{}/habitat/mesh_semantic.ply".format(mapname)
            # self.object_json = "/hdd/replica/Replica-Dataset/out/{}/habitat/info_semantic.json".format(mapname)
            self.sim_settings = {
                "width": 256,  # Spatial resolution of the observations
                "height": 256,
                "scene": self.test_scene,  # Scene path
                "default_agent": 0,
                "sensor_height": 1.5,  # Height of sensors in meters
                "color_sensor": True,  # RGB sensor
                "semantic_sensor": True,  # Semantic sensor
                "depth_sensor": True,  # Depth sensor
                "seed": 1,
            }

            self.basepath = f"/home/nel/gsarch/replica_novel_categories/{mapname}_{episode}"
            # self.basepath = f"/hdd/ayushj/habitat_data/{mapname}_{episode}"
            if not os.path.exists(self.basepath):
                os.mkdir(self.basepath)

            self.cfg, self.sim_cfg = self.make_cfg(self.sim_settings)
            self.sim = habitat_sim.Simulator(self.cfg)
            random.seed(self.sim_settings["seed"])
            self.sim.seed(self.sim_settings["seed"])
            self.set_agent(self.sim_settings)
            self.nav_pts = self.get_navigable_points()

            config = get_config()
            config.defrost()
            config.TASK.SENSORS = []
            config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
            config.freeze()

            self.run()
            
            self.sim.close()
            time.sleep(3)


    def set_agent(self, sim_settings):
        agent = self.sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([1.5, 1.072447, 0.0])
        #agent_state.position = np.array([1.0, 3.0, 1.0])
        agent.set_state(agent_state)
        agent_state = agent.get_state()
        if self.verbose:
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        self.agent = agent

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings["scene"]

        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "do_nothing": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.)
            ),
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.rot_interval)
            ),
            "look_up":habitat_sim.ActionSpec(
                "look_up", habitat_sim.ActuationSpec(amount=self.rot_interval)
            ),
            "look_down":habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=self.rot_interval)
            ),
            "look_down_init":habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=100.0)
            )
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg]), sim_cfg 
    

    def display_sample(self, rgb_obs, semantic_obs, depth_obs, mainobj=None, visualize=False):
        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        # st()
        
        
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

        display_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)
        #print(display_img.shape)

        # mask_image = False
        # if mask_image and mainobj is not None:
        #     main_id = int(mainobj.id[1:])
        #     print("MAINID ", main_id)
        #     # semantic = observations["semantic_sensor"]
        #     display_img[semantic_obs == main_id] = [1, 0, 1]
            # st()

        #display_img = cv2
        plt.imshow(display_img)
        plt.show()
        # cv2.imshow('img',display_img)
        if visualize:
            arr = [rgb_img, semantic_img, depth_img]
            titles = ['rgb', 'semantic', 'depth']
            plt.figure(figsize=(12 ,8))
            for i, data in enumerate(arr):
                ax = plt.subplot(1, 3, i+1)
                ax.axis('off')
                ax.set_title(titles[i])
                plt.imshow(data)
                # plt.pause()
            plt.show()
            # plt.pause(0.5)
            # cv2.imshow()
            # plt.close()


    def save_datapoint(self, agent, observations, data_path, viewnum, mainobj_id, flat_view):
        if self.verbose:
            print("Print Sensor States.", self.agent.state.sensor_states)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        
        # Extract objects from instance segmentation
        object_list = []
        obj_ids = np.unique(semantic)
        if self.verbose:
            print("Unique semantic ids: ", obj_ids)

        # st()
        for obj_id in obj_ids:
            if obj_id < 1 or obj_id > len(self.sim.semantic_scene.objects):
                continue
            if self.sim.semantic_scene.objects[obj_id] == None:
                continue
            if self.sim.semantic_scene.objects[obj_id].category == None:
                continue
            try:
                class_name = self.sim.semantic_scene.objects[obj_id].category.name()
                
                if self.verbose:
                    print("Class name is : ", class_name)
            except Exception as e:
                print(e)
                print("done")
            #if class_name not in self.ignore_classes:
            if class_name in self.include_classes:
                obj_instance = self.sim.semantic_scene.objects[obj_id]
                # st()
                mask = np.zeros_like(semantic)
                mask[semantic == obj_id] = 1
                y, x = np.where(mask)
                pred_box = np.array([min(y), min(x), max(y), max(x)]) # ymin, xmin, ymax, xmax
                # print("Object name {}, Object category id {}, Object instance id {}".format(class_name, obj_instance['id'], obj_instance['class_id']))
                # st()
                obj_data = {'instance_id': obj_id, 'category_id': obj_instance.category.index(), 'category_name': obj_instance.category.name(),
                                 'bbox_center': obj_instance.obb.center, 'bbox_size': obj_instance.obb.sizes,
                                  'local_T_world': obj_instance.obb.local_to_world, 'mask_2d': mask, 'box_2d': pred_box}
                # object_list.append(obj_instance)
                object_list.append(obj_data)

        # st()
        depth = observations["depth_sensor"]
        # self.display_sample(rgb, semantic, depth, visualize=True)
        agent_pos = observations["positions"] #agent.state.position
        agent_rot = observations["rotations"]
        # Assuming all sensors have same extrinsics
        color_sensor_pos = observations["positions"] #agent.state.sensor_states['color_sensor'].position
        color_sensor_rot = observations["rotations"] #agent.state.sensor_states['color_sensor'].rotation
        print("POS ", agent_pos)
        print("ROT ", color_sensor_rot)

        save_data = {'flat_view': flat_view, 'mainobj_id': mainobj_id, 'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
        
        with open(os.path.join(data_path, str(viewnum) + ".p"), 'wb') as f:
            pickle.dump(save_data, f)
        f.close()

    def is_valid_datapoint(self, observations, mainobj):
        main_id = int(mainobj.id[1:])
        semantic = observations["semantic_sensor"]
        # st()
        num_occ_pixels = np.where(semantic == main_id)[0].shape[0]
        #print(semantic.shape)
        #print("Number of pixels: ", num_occ_pixels)
        small_objects = []
        if mainobj.category.name() in self.small_classes:
            if  num_occ_pixels < 0.9*256*256:
                return True
        else:
            if num_occ_pixels < 0.9*256*256: 
                return True
        return False

    def quaternion_from_two_vectors(self, v0: np.array, v1: np.array) -> np.quaternion:
        r"""Computes the quaternion representation of v1 using v0 as the origin."""
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        c = v0.dot(v1)
        # Epsilon prevents issues at poles.
        if c < (-1 + EPSILON):
            c = max(c, -1)
            m = np.stack([v0, v1], 0)
            _, _, vh = np.linalg.svd(m, full_matrices=True)
            axis = vh.T[:, 2]
            w2 = (1 + c) * 0.5
            w = np.sqrt(w2)
            axis = axis * np.sqrt(1 - w2)
            return np.quaternion(w, *axis)

        axis = np.cross(v0, v1)
        s = np.sqrt((1 + c) * 2)
        return np.quaternion(s * 0.5, *(axis / s))

    def run(self):

        scene = self.sim.semantic_scene
        objects = scene.objects
        for obj in objects:
            if obj == None or obj.category == None or obj.category.name() not in self.include_classes:
                continue
            # st()
            if self.verbose:
                print(f"Object name is: {obj.category.name()}")
            # Calculate distance to object center
            obj_center = obj.obb.to_aabb().center
            #print(obj_center)
            obj_center = np.expand_dims(obj_center, axis=0)
            #print(obj_center)
            distances = np.sqrt(np.sum((self.nav_pts - obj_center)**2, axis=1))

            # Get points with r_min < dist < r_max
            valid_pts = self.nav_pts[np.where((distances > self.radius_min)*(distances<self.radius_max))]
            # if not valid_pts:
                # continue

            # plot valid points that we happen to select
            # self.plot_navigable_points(valid_pts)

            # Bin points based on angles [vertical_angle (10 deg/bin), horizontal_angle (10 deg/bin)]
            valid_pts_shift = valid_pts - obj_center

            dz = valid_pts_shift[:,2]
            dx = valid_pts_shift[:,0]
            dy = valid_pts_shift[:,1]

            # Get yaw for binning 
            valid_yaw = np.degrees(np.arctan2(dz,dx))

            # pitch calculation 
            dxdz_norm = np.sqrt((dx * dx) + (dz * dz))
            valid_pitch = np.degrees(np.arctan2(dy,dxdz_norm))

            # binning yaw around object
            nbins = 18
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size

            spawns_per_bin = int(self.num_views / num_valid_bins) + 2
            print(f'spawns_per_bin: {spawns_per_bin}')

            if self.visualize:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))
                    points = valid_pts[cur_bi]
                    x_sample = points[:,0]
                    z_sample = points[:,2]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                plt.plot(obj_center[:,2], obj_center[:,0], 'x', color = 'black')
                plt.show()

            
            action = "do_nothing"
            episodes = []
            valid_pts_selected = []
            cnt = 0
            for b in range(nbins):
                
                # get all angle indices in the current bin range
                # st()
                inds_bin_cur = np.where(bin_yaw==(b+1)) # bins start 1 so need +1
                if inds_bin_cur[0].size == 0:
                    continue

                for s in range(spawns_per_bin):
                    # st()
                    s_ind = np.random.choice(inds_bin_cur[0])
                    #s_ind = inds_bin_cur[0][0]
                    pos_s = valid_pts[s_ind]
                    valid_pts_selected.append(pos_s)
                    agent_state = habitat_sim.AgentState()
                    agent_state.position = pos_s + np.array([0, 1.5, 0])


                    # YAW calculation - rotate to object
                    agent_to_obj = np.squeeze(obj_center) - agent_state.position
                    agent_local_forward = np.array([0, 0, -1.0]) # y, z, x
                    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                    flat_to_obj /= flat_dist_to_obj

                    det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
                    quat_yaw = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))

                    # Set agent yaw rotation to look at object
                    agent_state.rotation = quat_yaw
                    
                    # change sensor state to default 
                    # need to move the sensors too
                    print(self.agent.state.sensor_states)
                    for sensor in self.agent.state.sensor_states:
                        # st()
                        self.agent.state.sensor_states[sensor].rotation = agent_state.rotation
                        self.agent.state.sensor_states[sensor].position = agent_state.position # + np.array([0, 1.5, 0]) # ADDED IN UP TOP
                        # print("PRINT", self.agent.state.sensor_states[sensor].rotation)

                    # Calculate Pitch from head to object
                    turn_pitch = np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))
                    num_turns = np.abs(np.floor(turn_pitch/self.rot_interval)).astype(int) # compute number of times to move head up or down by rot_interval
                    print("MOVING HEAD ", num_turns, " TIMES")
                    movement = "look_up" if turn_pitch>0 else "look_down"

                    # initiate agent
                    self.agent.set_state(agent_state)
                    self.sim.step(action)

                    # Rotate "head" of agent up or down based on calculated pitch angle to object to center it in view
                    if num_turns == 0:
                        pass
                    else: 
                        for turn in range(num_turns):
                            # st()
                            self.sim.step(movement)
                            if self.verbose:
                                for sensor in self.agent.state.sensor_states:
                                    print(self.agent.state.sensor_states[sensor].rotation)
                    
                    # get observations after centiering
                    observations = self.sim.step(action)
                    
                    # Assuming all sensors have same rotation and position
                    observations["rotations"] = self.agent.state.sensor_states['color_sensor'].rotation #agent_state.rotation
                    observations["positions"] = self.agent.state.sensor_states['color_sensor'].position
                    
                    if self.is_valid_datapoint(observations, obj):
                        if self.verbose:
                            print("episode is valid......")
                        episodes.append(observations)
                        if self.visualize:
                            rgb = observations["color_sensor"]
                            semantic = observations["semantic_sensor"]
                            depth = observations["depth_sensor"]
                            self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)
                    # if self.visualize:
                    #         rgb = observations["color_sensor"]
                    #         semantic = observations["semantic_sensor"]
                    #         depth = observations["depth_sensor"]
                    #         self.display_sample(rgb, semantic, depth, mainobj=obj, visualize=False)

                    #print("agent_state: position", self.agent.state.position, "rotation", self.agent.state.rotation)

                    cnt +=1
                        

            if len(episodes) >= self.num_views:
                print(f'num episodes: {len(episodes)}')
                data_folder = obj.category.name() + '_' + obj.id
                data_path = os.path.join(self.basepath, data_folder)
                print("Saving to ", data_path)
                os.mkdir(data_path)
                flat_obs = np.random.choice(episodes, self.num_views, replace=False)
                viewnum = 0
                for obs in flat_obs:
                    self.save_datapoint(self.agent, obs, data_path, viewnum, obj.id, True)
                    viewnum += 1
            else:
                print(f"Not enough episodes: f{len(episodes)}")

            if self.visualize:
                valid_pts_selected = np.vstack(valid_pts_selected)
                self.plot_navigable_points(valid_pts_selected)



    def get_navigable_points(self):
        navigable_points = np.array([0,0,0])
        for i in range(20000):
            navigable_points = np.vstack((navigable_points,self.sim.pathfinder.get_random_navigable_point()))
        return navigable_points
    
    def plot_navigable_points(self, points):
        # print(points)
        x_sample = points[:,0]
        z_sample = points[:,2]
        plt.plot(z_sample, x_sample, 'o', color = 'red')
        
        plt.show()


if __name__ == '__main__':
    AutomatedMultiview()