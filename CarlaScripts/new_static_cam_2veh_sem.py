import time
import random
import glob
import argparse
import logging
import sys
import os
from PIL import Image
import uuid
import copy
import numpy as np
import carla_utils
import pickle
import dynamic_weather as weather
import datetime
import csv
try:
    import queue
except ImportError:
    import Queue as queue
# import ipdb
# st = ipdb.set_trace
try:
    sys.path.append(glob.glob('/hdd/carla97/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import carla_utils
from carla_utils import ClientSideBoundingBoxes
import image_converter
from carla import ColorConverter as cc

from carla_sync_mode import CarlaSyncMode

def save_npy(data, filename):
    np.save(filename, data)

# Built on carla_two_unique.py
# can be used to spawn 2-3 vehicles
'''
camR ranges: 8-14, -5,5, 1-3
'''
class CarlaMultiviewRunner():
    def __init__(self, start_episode, end_episode, mod, save_dir):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

        self.base_path = os.path.join(save_dir, "surveil_{}".format(mod))
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.randomize_each_camera = True

        self.start_frame = 0
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.frames_per_episode = 1
        self.num_vehicles = 2
        self.today = str(datetime.date.today())
        self.episode_path_format = '{:s}_episode_{:0>4d}_vehicles_{:0>3d}'
        self.out_filename_format = '{:s}_episode_{:0>4d}_vehicles_{:0>3d}/{:s}/{:0>6d}'

        self.nearest_x = 2.0
        self.dis_to_center = 15.0#5.0
        self.height = 0.65
        self.d_height = 1.0

        # Camera specific params
        self.image_size_x = 768
        self.image_size_y = 256
        self.fov = 90
        self.focal = self.image_size_x/(2.0*np.tan(self.fov*np.pi/360.0))
        self.fps = 10
        self.half_sqrt2 = np.sqrt(2)/2

        self.host = "127.0.0.1"
        self.port = 2000
        self.filterv = "vehicle.*"
        self.sensor_types = ['sensor.camera.rgb', 'sensor.camera.depth', 'sensor.camera.semantic_segmentation']
        self.sensor_names_format = ['CameraRGB{}', 'CameraDepth{}', 'CameraSemantic{}']

        self.category = {'vehicle': 0}

        self.calculate_camera_locations()

        self.visualize = True
        self.visual_dir = 'visual'
        if self.visualize and not os.path.exists(self.visual_dir):
            os.mkdir(self.visual_dir)

    def calculate_camera_locations(self):
        self.center = np.array([self.nearest_x+self.dis_to_center, 0.0, self.height])
        self.positions = []
        self.rotations = []
        #hori_rs = [self.dis_to_center, self.dis_to_center, self.dis_to_center]
        verti_hs = [self.height+self.d_height, self.height, self.height+2*self.d_height]
        pitches = [-0.08/np.pi*180 for i in range(3)]
        #for i in [0]: #range(len(hori_rs)):
        #hori_r = hori_rs[0]
        h = 1.65 # verti_hs[1]
        p_angle = pitches[0]
        #theta = 15.0
        # for angle in np.arange(5.0, 355.0, 20.0):#np.arange(0.0, 359.0, 30.0):
        #     a_rad = angle/180.0*np.pi
        #     scale = 1. 
        #     trans = np.zeros(3)
        #     #pos = self.center + scale*np.array([-hori_r*np.cos(a_rad), hori_r*np.sin(a_rad), h]) + trans
        #     pos = scale*np.array([-hori_r*np.cos(a_rad), hori_r*np.sin(a_rad), h]) + trans
        #     self.positions.append(pos)
        #     self.rotations.append([p_angle, -angle, 0])

        # specify angles to rotate around
        bin_angle_size = 20.0
        num_views_per_bin = 2
        angles = np.arange(0, 360.0, bin_angle_size)
        angles = np.tile(angles, num_views_per_bin)

        #hori_rs = [10.0, 15.0, 20.0]

        # pick radii for each angle
        hori_rs = np.random.uniform(low=3.0, high=15.0, size=angles.shape[0])

        ind = 0
        for angle in angles: #np.arange(0.0, 359.0, 30.0):
            a_rad = angle/180.0*np.pi
            scale = 1. 
            trans = np.zeros(3)
            hori_r = hori_rs[ind]
            #pos = self.center + scale*np.array([-hori_r*np.cos(a_rad), hori_r*np.sin(a_rad), h]) + trans
            pos = scale*np.array([-hori_r*np.cos(a_rad), hori_r*np.sin(a_rad), h]) + trans
            self.positions.append(pos)
            self.rotations.append([p_angle, -angle, 0])
            ind += 1

        self.K = np.identity(3)
        self.K[0, 2] = self.image_size_x / 2.0
        self.K[1, 2] = self.image_size_y / 2.0
        # use -focal to be the same as the previous version
        self.K[0, 0] = self.K[1, 1] = self.image_size_x / \
            (2.0 * np.tan(self.fov * np.pi / 360.0))
    
    def destroy_actors(self):
        print('Destroying actors.')
        for actor in self.vehicles_list:
            if actor.is_alive:
                actor.destroy()
        for actor in self.sensors_list:
            if actor.is_alive:
                actor.destroy()

        print("Destroyed all actors")


    def run_carla_client(self):
        # print(self.positions)
        # print(self.rotations)
        # print(self.start_episode, self.end_episode)
        client = carla.Client(self.host, self.port)
        client.set_timeout(20.0)

        try:
            self.available_maps = client.get_available_maps()
            print('loaded available maps')

            logging.info('listening to server %s:%s', self.host, self.port)

            num_batches = 2

            for episode in range(self.start_episode, self.end_episode):
                
                print("Starting episode number %d" % episode)
                            
                uuid_run = str(uuid.uuid1())

                cur_map = random.choice(self.available_maps)
                print("About to load the map %s" %cur_map)
                map_name = cur_map.split('/')[-1]

                episode_path = os.path.join(self.base_path, self.episode_path_format.format(map_name, episode, self.num_vehicles))
                if not os.path.exists(episode_path):
                    os.mkdir(episode_path)

                self.world = client.load_world(cur_map)
                
                self.world.tick()

                # Initialize the actor lists
                self.vehicles_list = []
                # self.sensors_list = []
                # self.sensors_name = []
                self.vehicle_dir_paths = []
                self.actors = []
                # self.position_list = []
                # self.rotation_list = []
                self.camR_sensor_indices = []
                self.vehicle_bbox_list = []
                self.vehicle_extrinsics = []
                self.vehicle_names = []

                world_T_agents_f = open(os.path.join(episode_path, 'world_T_agent.txt'), 'w')
                world_T_agents_f.write('roll,pitch,yaw\n')
                fcsv = open(os.path.join(episode_path, 'bboxes.csv'), 'w')
                csv_writer = csv.writer(fcsv)
                csv_writer.writerow(['episode', 'frame', 'obj_id', 'category','x','y','z','l','w','h','r','x1','y1','x2','y2','depth', 'occluded'])
                
                # Get all the blueprints
                blueprints = self.world.get_blueprint_library().filter(self.filterv)

                # remove bikes
                cars = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
                blueprints = cars
                #for blueprint in blueprints:
                    #print(blueprint)

                # for blueprint in blueprints:
                #     print(blueprint.id)
                #     for attr in blueprint:
                #         print('  - {}'.format(attr))

                weather = carla.WeatherParameters(
                        cloudiness=np.random.randint(0, 70),
                        precipitation=np.random.randint(0, 75),
                        sun_altitude_angle=np.random.randint(30, 90))
                self.world.set_weather(weather)
                self.world.tick()
                
                

                # Get all the spawn points
                spawn_points = self.world.get_map().get_spawn_points()
                random.shuffle(spawn_points)

                self.spawn_vehicles(blueprints, episode_path, spawn_points)

                batches = list(np.floor(np.linspace(0, len(self.positions), num_batches)).astype(int))
                print("BATCH INDICES: ", batches)
                self.idx = 0 # for naming
                self.positions_ = [] # actual rotations of agents
                self.rotations_ = [] # actual rotations of agents
                self.first_batch = True
                for batch in range(len(batches)-1):

                    print("BATCH #", batches[batch], " to ", batches[batch+1])

                    self.sensors_list = []
                    self.sensors_name = []
                    self.position_list = []
                    self.rotation_list = []
                
                    # if batch == len(batches)-2:
                    #     positions_batch = self.positions[batches[batch]:]
                    #     rotations_batch = self.rotations[batches[batch]:]
                    # else:
                    positions_batch = self.positions[batches[batch]:batches[batch+1]]
                    rotations_batch = self.rotations[batches[batch]:batches[batch+1]]

                    print(positions_batch)
                    print(rotations_batch)
                    # print("SPAWN POINT: ", spawn_points[0])

                    self.spawn_sensors(positions_batch, rotations_batch, spawn_points[0], episode_path) # set the position relative to the first spawn point

                    print("Done with actor creation")
                    print("Total number of sensors are: ", len(self.sensors_list))
                
                    self.world.tick()
                    
                    last_saved_frame = 0
                    # Create a synchronous mode context.
                    with CarlaSyncMode(self.world, self.sensors_list, fps=self.fps) as sync_mode:
                        cnt = 0
                        for v in self.vehicles_list:
                            # print("Bounding box for this vehicle is: ", v.bounding_box.location, v.bounding_box.extent)
                            bbox_loc, bbox_ext = v.bounding_box.location, v.bounding_box.extent
                            bbox = [bbox_loc.x - bbox_ext.x, bbox_loc.y - bbox_ext.y, bbox_loc.z - bbox_ext.z, bbox_loc.x + bbox_ext.x, bbox_loc.y + bbox_ext.y, bbox_loc.z + bbox_ext.z]
                            self.vehicle_bbox_list.append(bbox)
                            print("bbox coords are: ", bbox)
                        #     v.set_autopilot(False)
                            v.set_simulate_physics(False)

                        print("VEHICLES: ", self.vehicles_list)
                            
                        # print("All vehicles put to autopilot")
                        # set weather
                        
                        while True:
                            print(cnt)

                            if cnt == self.frames_per_episode + self.start_frame:
                                print("Done with episode %d." %episode)
                                time.sleep(5)
                                break

                            # self.world.tick()
                            # print("Getting the data")
                            # Advance the simulation and wait for the data.
                            data, frame = sync_mode.tick(timeout=12.0)
                            # print('Got the data :))')
                            if self.first_batch:
                                frame_a = frame
                            #frame += batches[batch]

                            print("FRAME ", frame_a)
                    
                            if cnt >= self.start_frame:
                                data = data[1:] # Remove the world tick datapoint
                                valid = self.save_data(data, episode, frame_a, episode_path, world_T_agents_f, csv_writer)
                                if not valid:
                                    print("throwing out view ", cnt, " due to depth")
                            cnt += 1

                    for actor in self.sensors_list:
                        if actor.is_alive:
                            # print("ALIVE! ", actor)
                            actor.destroy()
                    
                    self.first_batch = False
                    
                # these are the translation relative to the first spawn point
                # print("ALL POSITIONS", self.positions_)
                # print("LENGTH ", len(self.positions_))
                # print("ALL ROTATIONS", self.rotations_)
                # print("LENGTH ", len(self.rotations_))
                save_npy(np.array(self.positions_), os.path.join(episode_path, 'all_cam_positions'))
                save_npy(np.array(self.rotations_), os.path.join(episode_path, 'all_cam_rotations'))

        except Exception as e: print(e)

        

        finally:
            #self.world.destroy()
            # for actor in self.world.get_actors().filter("vehicle*"):
            #     if actor.is_alive:
            #         #print("2", actor)
            #         #sys.stdout.write('2')
            #         actor.destroy()
            # for actor in self.sensors_list:
            #     if actor.is_alive:
            #         #print("5", actor)
            #         #sys.stdout.write('5')
            #         actor.destroy()
            client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter("vehicle*")])
            client.apply_batch([carla.command.DestroyActor(x) for x in self.sensors_list])
        #     self.destroy_actors()


    def save_data(self, data, episode, framenum, episode_path, world_T_agents_f, csv_writer):
        # Check if too many occlusions (e.g. in a wall)
        # for idx, name in enumerate(self.sensors_name):
        #     if "Depth" in name:
        #         data_instance = data[idx]
        #         processed_data = image_converter.depth_in_meters(data_instance)
        #         valid = self.check_depth(processed_data)
        #         if valid == 0:
        #             return False

        # Save data
        #print("SENSOR LENGTH:", len(self.sensors_name))
        #print("SENSOR NAMES:", self.sensors_name)
        for idx, name in enumerate(self.sensors_name):
            camera_path = os.path.join(episode_path, name)
            if not os.path.exists(camera_path):
                os.mkdir(camera_path)
            if not os.path.exists(os.path.join(episode_path, 'boxes')):
                os.mkdir(os.path.join(episode_path, 'boxes'))
            data_instance = data[idx]
            if "RGB" in name:
                processed_data = image_converter.to_rgb_array(data_instance)
            elif "Depth" in name:
                processed_data = image_converter.depth_in_meters(data_instance)
                #valid = self.get_valid_view(processed_data)
                # print("DEPTH")
                # print(processed_data.shape)
                #print(processed_data)
            elif "Semantic" in name:
                # TODO: handle R channel properly here.
                print("SEMANTIC")
                # print(data_instance.shape)
                # print(data_instance)
                #print(np.unique(data_instance, axis=0))
                processed_data = image_converter.to_rgb_array(data_instance)
                print(processed_data.shape)
                print(np.unique(processed_data[:,:,0]))
                # data_instance.convert(cc.CityScapesPalette)
                # #data_instance.save_to_disk(os.path.join(self.visual_dir, 'img_e{:0>4d}_{:s}_f{:0>4d}.png'.format(0, name, framenum)),carla.ColorConverter.CityScapesPalette)
                # #processed_data = image_converter.to_rgb_array(data_instance)
                # print(processed_data.shape)
                # print(np.unique(processed_data[:,:,0]))

                # get segmentation
                sensor = self.sensors_list[idx]
                corners_cam, corners_cam_unproject, w2c, v2w = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles_list, sensor, self.K)
                c2w = np.linalg.inv(w2c)
                v2w = np.concatenate(v2w, axis=0)

                masks = []
                for obj_ind in range(len(corners_cam)):

                    # get 2D bounding box
                    xmin = int(np.min(corners_cam[obj_ind][:,0]))
                    xmax = int(np.max(corners_cam[obj_ind][:,0]))
                    ymin = int(np.min(corners_cam[obj_ind][:,1]))
                    ymax = int(np.max(corners_cam[obj_ind][:,1]))
                    if ymin < 0:
                        ymin = 0
                    elif ymax > self.image_size_y:
                        ymax = self.image_size_y
                    if xmin < 0:
                        xmin = 0
                    elif xmax > self.image_size_x:
                        xmax = self.image_size_x

                    veh_id = 10 # vehicle segmentation ID for semantic sensor

                    # get segmentation mask
                    img = processed_data.copy()
                    img2 = processed_data.copy()
                    img2 = img2[ymin:ymax, xmin:xmax, :]
                    #print(np.unique(img2[:,:,0]))
                    img2[img2!=veh_id] = 0.0
                    #print(np.sum(img2[:,:,:]==10))
                    img2[img2[:,:,0]==veh_id, :] = 1.0
                    #print(np.sum(img2==255))
                    img[:,:,:] = 0.0
                    img[ymin:ymax, xmin:xmax, :] = img2

                    if np.sum(img) != 0:
                        masks.append(img)

                    if self.visualize:
                        img = img * 255 # make mask white
                        img = Image.fromarray(img)
                        #img = ClientSideBoundingBoxes.draw_bounding_boxes(img, corners_cam)
                        #data_instance.save_to_disk(os.path.join(self.visual_dir, 'img_e{:0>4d}_{:s}_f{:0>4d}.png'.format(0, name, framenum)),carla.ColorConverter.CityScapesPalette)
                        img.save(os.path.join(self.visual_dir, 'img_e{:0>4d}_{:s}_f{:0>4d}_obj{}.png'.format(0, name, framenum, obj_ind)))

                masks = np.array(masks)

                processed_data = masks

                print("MASKS", masks.shape)
                    
                

            
            else:
                print("Invalid sensor type %s. Quitting" %j)
                exit(1)
            save_npy(processed_data, os.path.join(camera_path, '{:0>6d}'.format(framenum)))

            if 'RGB' in name:#frame % 1 == 0: 
                #output non-player agents
                img = processed_data.copy()
                sensor = self.sensors_list[idx]
                corners_cam, corners_cam_unproject, w2c, v2w = ClientSideBoundingBoxes.get_bounding_boxes(self.vehicles_list, sensor, self.K)
                c2w = np.linalg.inv(w2c)
                v2w = np.concatenate(v2w, axis=0)

                sensor_to_world = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
                save_npy(c2w, os.path.join(camera_path, '{:0>6d}_c2w'.format(framenum)))

                if self.visualize:
                    img = Image.fromarray(processed_data)
                    img = ClientSideBoundingBoxes.draw_bounding_boxes(img, corners_cam)
                    img.save(os.path.join(self.visual_dir, 'img_e{:0>4d}_{:s}_f{:0>4d}.png'.format(0, name, framenum)))
                if 'RGB0' in name:
                    # boxes = np.array(corners_cam_unproject, dtype=np.float32)
                    save_npy(v2w, os.path.join(episode_path, 'boxes', '{:0>6d}'.format(framenum)))
            # if 'Semantic' in name:#frame % 1 == 0: 

            #     if self.visualize:
            #         # print("1", corners_cam)
            #         # print(corners_cam[0].shape)
            #         # print("2", corners_cam_unproject)
            #         # print("3", w2c)
            #         # print("4", v2w)
            #         xmin = int(np.min(corners_cam[0][:,0]))
            #         xmax = int(np.max(corners_cam[0][:,0]))
            #         ymin = int(np.min(corners_cam[0][:,1]))
            #         ymax = int(np.max(corners_cam[0][:,1]))
            #         if ymin < 0:
            #             ymin = 0
            #         elif ymax > self.image_size_y:
            #             ymax = self.image_size_y
            #         if xmin < 0:
            #             xmin = 0
            #         elif xmax > self.image_size_x:
            #             xmax = self.image_size_x
            #         print(xmin)
            #         print(xmax)
            #         print(ymin)
            #         print(ymax)
            #         print(processed_data.flags)
            #         img = processed_data.copy()
            #         img2 = processed_data.copy()
            #         img2 = img2[ymin:ymax, xmin:xmax, :]
            #         print(np.unique(img2[:,:,0]))
            #         img2[img2!=10] = 0.0
            #         print(np.sum(img2[:,:,:]==10))
            #         img2[img2[:,:,0]==10, :] = 255
            #         print(np.sum(img2==255))
            #         #processed_data = image_converter.to_rgb_array(data_instance)
            #         #img = np.zeros(img2.shape)
            #         img[:,:,:] = 0.0
            #         img[ymin:ymax, xmin:xmax, :] = img2
            #         img = Image.fromarray(img)
            #         img.save(os.path.join(self.visual_dir, 'img_e{:0>4d}_{:s}_f{:0>4d}.png'.format(0, name, framenum)))
        return True

    def get_valid_view(self, data):
        for idx, name in enumerate(self.sensors_name):
            if "Depth" in name:
                data_instance = data[idx]
                processed_data = image_converter.depth_in_meters(data_instance)
                valid = self.check_depth(processed_data)
                if valid == 0:
                    return False
        return True

    def check_depth(self, depth):
        #all_ok = True
        # num_views = len(depths)

        # # determine how many objects present in each view
        # obj_in_views = np.zeros(num_views)
        # for n in list(range(MAX_OBJS)):
        #         # print('ok, working on object %d' % n)
        #     score = scorelist_s[0][n]
        #     tid = tidlist_s[0][n]
        #     lrt = lrtlist_camRs[0][n]
        #     if score == 1.0:
        #         vis_multiview = get_target_multiview_vis(tid, lrt, origin_T_camXs, origin_T_camRs[0], xyz_camXs)
        #         obj_in_views = np.add(vis_multiview, obj_in_views)
        #         # print(vis_multiview)
        #         #print(obj_in_views)
        #         # print(S)

        # # print(obj_in_views)
        # # print(obj_in_views.shape)


        #valid_views = []
        # num_bad = 0
        # for view in range(num_views):
        # mindepth = np.min(depth)
        #depth = depths[view]
        not_ok = depth < 2.0
        yes_ok = depth > 2.0
        A = np.sum(yes_ok)
        B = np.sum(not_ok)

        #print(scorelist_s[view])
        if A < B*3: # or (obj_in_views[view]==0): # view does not have space in front of it or no objects throw out
        #if (obj_in_views[view]==0): # view does not have space in front of it or no objects throw out
            print("NOT VALID")
            print('yes = %d; no = %d;' % (A, B))
            #print(obj_in_views[view])
            print("min depth: ", np.min(depth))
            #valid_views.append([0])
            #num_bad += 1
            valid_view = 0
            
            #if obj_in_views[view]==0:
                #print("No objects found in view ", view)

        else: # ok view
            #valid_views.append([1])
            print("VALID")
            # print("min depth: ", np.min(depth))
            print('yes = %d; no = %d;' % (A, B))
        #print("Number of invalid views: ", num_bad)
            valid_view = 1
        #valid_views = np.array(valid_views).astype(np.float32)
        #valid_views = valid_views.reshape((len(valid_views)))
        return valid_view

    def pts_dist(self, locsi, locsj):
        dist = (locsi.x - locsj.x)**2 + (locsi.y - locsj.y)**2 + (locsi.z - locsj.z)**2
        return np.sqrt(dist)
    '''
    Spawns the vehicles and attaches sensors to it.
    '''
    def spawn_vehicles(self, blueprints, episode_path, spawn_points):
        #print(spawn_points)
        # print(spawn_points[:min(self.num_vehicles, len(spawn_points))])
        # print(spawn_points[0])
        #for spawn_point in spawn_points: #spawn_points[:min(self.num_vehicles, len(spawn_points))]:
            # setup blueprint

        # spawn first vehicle at center of rotation
        spawn_point = spawn_points[0]
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
        blueprint.set_attribute('role_name', 'autopilot')

        # position = carla.Location(x=0, y=0, z=0) + spawn_point.location
        # transform = carla.Transform(position, carla.Rotation(pitch=0, yaw=0, roll=0))
        vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        #vehicle = self.world.try_spawn_actor(blueprint, transform)
        self.vehicle_names.append(blueprint.id)
        self.actors.append(vehicle)
        self.vehicles_list.append(vehicle)

        # Spawn second vehicle closest to first vehicle
        #dist_best = 100000000
        spts_closest = []
        center_loc = spawn_point.location
        for spts in spawn_points:
            loc = spts.location
            #print(loc)
            #print(loc.shape)
            #print(loc.x)
            #print(loc.y)
            dist = np.sqrt((loc.x - center_loc.x)**2 + (loc.y - center_loc.y)**2 + (loc.z - center_loc.z)**2)
            spts_closest.append(dist)
            # print(dist)
            # if dist < dist_best:
            #     dist_best = dist
            #     spts_closest = spts_closest
            #     print(dist_best)
            #     print(spts.location)
            #     print(spawn_point)
        spts_closest = np.array(spts_closest)
        spts_closest_inds = np.argsort(spts_closest)
        spts_closest_sort = np.sort(spts_closest)
        spts_closest_inds = spts_closest_inds[1:] # remove first object
        spts_closest_sort = spts_closest_sort[1:] # remove first object


        dist_threshold = 5 # must be atleast this away from first object
        while True:
            if spts_closest_sort[0] < dist_threshold:
                spts_closest_sort = spts_closest_sort[1:]
                spts_closest_inds = spts_closest_inds[1:]
            else:
                break

        # print(spts_closest)
        print(spts_closest_inds)
        print(spts_closest_sort)
        

            

        if self.num_vehicles==1:
            return
        
        for i in range(self.num_vehicles-1):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            # offset = np.array([self.nearest_x+self.dis_to_center, 0.0, 0.0]) # -1.0+10*np.random.rand(*(np.array(positions).shape))
            # position = carla.Location(x=offset[0], y=offset[1], z=offset[2]) + spawn_point.location
            # trans = -1.0+2*np.random.rand(3)
            # transform = carla.Transform(position, carla.Rotation(pitch=trans[0], yaw=trans[1], roll=trans[2]))
            spawn_point_close = spawn_points[spts_closest_inds[i]]
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point_close)
            self.vehicle_names.append(blueprint.id)
            self.actors.append(vehicle)
            self.vehicles_list.append(vehicle)


        print('finish spawning {} vehicles'.format(len(self.vehicles_list)))

    def spawn_sensors(self, positions, rotations, spawn_point, episode_path):
        
        #camera_indices = [0] + camera_indices[0:3] + [len(positions) - 1]
        if self.first_batch:
            camera_indices = list(range(1, len(positions) - 1))
            random.shuffle(camera_indices)
            camera_indices = [0] + camera_indices
        else:
            camera_indices = list(range(0, len(positions) - 1))
            random.shuffle(camera_indices)
        # camera_indices = [0] # for traj data, we only need one camera

        #trans = -1.0+2*np.random.rand(*(np.array(positions).shape))
        positions_ = positions #+ trans
        self.positions_.extend(positions_) # for saving out positions 
        #trans = -1.0+2*np.random.rand(*(np.array(positions).shape))
        rotations_ = rotations #+ trans
        self.rotations_.extend(rotations_) # for saving out rotations

        # # these are the translation relative to the first spawn point
        # save_npy(np.array(positions_), os.path.join(episode_path, 'all_cam_positions'))
        # save_npy(np.array(rotations_), os.path.join(episode_path, 'all_cam_rotations'))

        
        nsp = 0
        for idx in camera_indices:
            position, rotation = positions_[idx], rotations_[idx]
            for sensor_type, sensor_name in zip(self.sensor_types, self.sensor_names_format):
                print("DOING ", sensor_name.format(self.idx))
                sensor = self.get_sensor(spawn_point, position, rotation, sensor_type, sensor_name.format(idx))
                self.sensors_list.append(sensor)
                self.sensors_name.append(sensor_name.format(self.idx))
                #time.sleep(1)
                print("FINISHED ", nsp)
                nsp += 1
            self.idx += 1

        print('finish spawning {} sensors'.format(len(self.sensors_list)))
    
    def get_sensor(self, relative_to, position, rotation, sensor_type, sensor_id):
        blueprint = self.world.get_blueprint_library().find(sensor_type)

        blueprint.set_attribute('image_size_x', str(self.image_size_x))
        blueprint.set_attribute('image_size_y', str(self.image_size_y))
        blueprint.set_attribute('fov', str(self.fov))
        blueprint.set_attribute('role_name', sensor_id)
         
        # Set the time in seconds between sensor captures
        blueprint.set_attribute('sensor_tick', '0.0')
        # print("RELATIVE TO: ", relative_to.location)
        # print("POSITON: ", position)
        
        # Provide the position of the sensor relative to the vehicle.
        position = carla.Location(x=position[0], y=position[1], z=position[2]) + relative_to.location
        transform = carla.Transform(position, carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]))
        
        self.position_list.append(position)
        self.rotation_list.append(rotation)
        # use the commented line to attach cameras to the vehicle actor.
        # sensor = self.world.spawn_actor(blueprint, transform, attach_to=vehicle)
        # fixed cameras
        sensor = self.world.spawn_actor(blueprint, transform)
        
        return sensor


    def initiate(self):
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        
        while True:
            # try:
            self.run_carla_client()
            return

            # except Exception as error:
            #     print("Error occured while interacting with carla.")
            #     print(error)
            #     logging.error(error)
            #     time.sleep(1)

if __name__ == '__main__':
    episode = int(sys.argv[1])
    mod = sys.argv[2]
    save_dir = sys.argv[3]
    print(mod)
    print(save_dir)
    cmr = CarlaMultiviewRunner(episode, episode + 1, mod, save_dir)
    cmr.initiate()