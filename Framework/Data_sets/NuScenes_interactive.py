import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
from scipy import interpolate as interp
from pyquaternion import Quaternion

# from nuscenes.map_expansion.map_api import NuScenesMap, locations
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, locations
from NuScenes.vec_map import VectorMap
from NuScenes.nusc_utils import populate_vector_map

class NuScenes_interactive(data_set_template):
    def get_name(self=None):
        names = {'print':'NuScenes',
                 'file':'NuScenes_I',
                 'latex': '\emph{NuS}'}
        return names
    
    def future_input(self=None):
        return False
    
    def includes_images(self=None):
        return True
    
    def set_scenario(self):
        self.scenario = scenario_none()

    def create_path_samples(self):
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []

        # prepare images
        image_path = self.path + os.sep + 'Data_sets' + os.sep + 'NuScenes' + os.sep + 'data'
        image_path_full = image_path + os.sep + 'maps' + os.sep + 'expansion' + os.sep 
        self.Images = pd.DataFrame(np.zeros((0, 2), object), columns = ['Image', 'Target_MeterPerPx'])

        map_files = os.listdir(image_path_full)
        px_per_meter = 1
        
        max_width = 0
        max_height = 0
        for map_file in map_files:
            if map_file.endswith('.json'):
                map_name = map_file[:-5]
                map_image = NuScenesMap(dataroot = image_path, map_name = map_name)
                vector_map = VectorMap('placeholder:' + map_name)
                populate_vector_map(vector_map, map_image)
                bit_map = vector_map.rasterize(resolution = px_per_meter)
                self.Images.loc[map_name] = [bit_map, 1 / px_per_meter]
                max_width = max(bit_map.shape[1], max_width)
                max_height = max(bit_map.shape[0], max_height)
        
        # pad images
        for loc_id in self.Images.index:
            img = self.Images.loc[loc_id].Image
            img_pad = np.pad(img, ((0, max_height - img.shape[0]),
                                   (0, max_width  - img.shape[1]),
                                   (0,0)), 'constant', constant_values=0)
            self.Images.Image.loc[loc_id] = img_pad 

        nuscenes_dt = 0.5
        file_path  = self.path + os.sep + 'Data_sets' + os.sep + 'NuScenes' + os.sep + 'data'

        data_obj = NuScenes(version = 'v1.0-trainval', dataroot = file_path, verbose = True)

        for data_idx, scene_record in enumerate(data_obj.scene):
            scene_name = scene_record['name']
            scene_desc = scene_record['description']
            scene_location = data_obj.get('log', scene_record['log_token'])['location']
            scene_length = scene_record['nbr_samples']

            print('Scene ' + str(data_idx + 1) + ' of ' + str(len(data_obj.scene)) +
                  ': ' + scene_name + ' (' + scene_desc + ')')
            
            all_frames = []
            frame_idx_dict = {}
            curr_scene_token = scene_record["first_sample_token"]
            frame_idx = 0
            while curr_scene_token:
                frame = data_obj.get("sample", curr_scene_token)

                all_frames.append(frame)
                frame_idx_dict[frame["token"]] = frame_idx 

                curr_scene_token = frame["next"]
                frame_idx += 1


            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])

            t = np.arange(scene_length) * nuscenes_dt

            all_agents_token = []
            agent_id = 1

            ego_translation_list = []
            for frame_idx, frame in enumerate(all_frames):
                # get the ego agent
                cam_front_data = data_obj.get('sample_data', frame['data']['CAM_FRONT'])
                ego_pose = data_obj.get('ego_pose', cam_front_data['ego_pose_token']) 
                ego_translation_list.append(np.array([ego_pose['translation']]))

                # go through all other agents in the scene
                for agent_token in frame['anns']:
                    agent_data = data_obj.get('sample_annotation', agent_token)

                    agent_category = agent_data['category_name']

                    if not (agent_category.startswith('vehicle') or agent_category.startswith('human')):
                        continue
                    if agent_data['next'] == '':
                        continue
                    if agent_data['instance_token'] in all_agents_token:
                        continue

                    translation_list = [np.array(agent_data["translation"][:3])[np.newaxis]]

                    prev_idx = frame_idx 
                    curr_sample_ann_token = agent_data["next"]
                    while curr_sample_ann_token:
                        agent_data_new = data_obj.get('sample_annotation', curr_sample_ann_token)
                        translation = np.array(agent_data_new["translation"][:3])
                        
                        curr_idx = frame_idx_dict[agent_data_new["sample_token"]]
                        # check for missing frames
                        if curr_idx > prev_idx + 1:
                            fill_time = np.arange(prev_idx + 1, curr_idx)
                            xs = np.interp(fill_time, [prev_idx, curr_idx], [translation_list[-1][0,0], translation[0]])
                            ys = np.interp(fill_time, [prev_idx, curr_idx], [translation_list[-1][0,1], translation[1]])
                            zs = np.interp(fill_time, [prev_idx, curr_idx], [translation_list[-1][0,2], translation[2]])
                            translation_list.append(np.stack((xs, ys, zs), axis = 1))

                        translation_list.append(translation[np.newaxis])
                        
                        prev_idx = curr_idx
                        curr_sample_ann_token = agent_data_new["next"]

                    translations = np.concatenate(translation_list, axis = 0)

                    agent_name = 'v_' + str(agent_id)
                    if agent_category.startswith('vehicle'):
                        agent_types[agent_name] = 'V'
                        if agent_category.startswith('vehicle.bicycle'):
                            agent_types[agent_name] = 'B'
                        elif agent_category.startswith('vehicle.motorcycle'):
                            agent_types[agent_name] = 'M'
                    elif agent_category.startswith('human'):    
                        agent_types[agent_name] = 'P'


                    agent_traj = np.ones((scene_length, 2)) * np.nan
                    agent_traj[frame_idx:frame_idx + len(translations),:] = translations[:,:2]

                    path[agent_name] = agent_traj * np.array([[1, -1]]) # Align with Images
                    agent_id += 1

            ego_translations = np.concatenate(ego_translation_list, axis = 0)
            path['tar'] = ego_translations[:,:2] * np.array([[1, -1]]) # Align with Images
            agent_types['tar'] = 'V'

            domain = pd.Series(np.zeros(3, object), index = ['location', 'scene', 'image_id'])
            domain.location = scene_location
            domain.scene = scene_name 
            domain.image_id = scene_location

            self.num_samples += 1
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)


    def calculate_distance(self, path, t, domain):
        return None
    
    def evaluate_scenario(self, path, D_class, domain):
        return None
        
    def calculate_additional_distances(self, path, t, domain):
        return None
    
    def fill_empty_path(self, path, t, domain, agent_types):
        for agent in path.index:
            if isinstance(path[agent], float):
                assert str(path[agent]) == 'nan'
            else:
                x = path[agent][:,0]
                y = path[agent][:,1]
                
                rewrite = np.isnan(x)
                if not rewrite.any():
                    continue
                useful = np.invert(rewrite)
                if agent_types[agent] == 'P':
                    x = np.interp(t,t[useful],x[useful])
                    y = np.interp(t,t[useful],y[useful])
                else:
                    x = interp.interp1d(t[useful], x[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
                    y = interp.interp1d(t[useful], y[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
            
                path[agent] = np.stack([x, y], axis = -1)
        
        return path, agent_types
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_dashed = []
        return lines_solid, lines_dashed