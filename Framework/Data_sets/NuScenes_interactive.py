import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
import json
from scipy import interpolate as interp
from pyquaternion import Quaternion



class NuScenes_interactive(data_set_template):
    '''
    The NuScenes dataset is one of the most common real-word datasets used in testing 
    general trajectory prediction models. It encompasses many situations that might 
    be encounterd inside cities.
    
    If one wants to do the NuScenes prediction challenge (only including vehicles), then
    one would have to set the exact following arguments in Datasets:
    - 't0_type': 'all'
    
    The following settings would have to be set in Data_Params:
    - 'dt': 0.5
    - 'num_timesteps_in': 4
    - 'num_timesteps_out': 12
    
    The dataset meanwhile is available at https://www.nuscenes.org/nuscenes, and the 
    following citation can be used:
        
    Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., ... & Beijbom, O. (2020). 
    nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference 
    on computer vision and pattern recognition (pp. 11621-11631).
    
    '''
    def get_name(self=None):
        names = {'print':'NuScenes',
                 'file':'NuScenes_I',
                 'latex': '\emph{NuS}'}
        return names
    
    def future_input(self=None):
        return False
    
    def includes_images(self=None):
        return True 
    
    def includes_sceneGraphs(self = None):
        return True
    
    def set_scenario(self):
        self.scenario = scenario_none()
    
    def path_data_info(self = None):
        return ['x', 'y']

    def create_path_samples(self):
        # from nuscenes.map_expansion.map_api import NuScenesMap, locations
        from nuscenes.nuscenes import NuScenes
        from NuScenes.data.train_val_split import train, val
        
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.Size_old = []
        self.T = []
        self.Domain_old = []

        # prepare file path
        file_path = self.path + os.sep + 'Data_sets' + os.sep + 'NuScenes' + os.sep + 'data'

        # extract map data
        # Get images
        image_path_full = file_path + os.sep + 'maps' + os.sep + 'expansion' + os.sep 
        self.Images = pd.DataFrame(np.zeros((0, 2), object), columns = ['Image', 'Target_MeterPerPx'])

        # Get scenegraphs
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                            'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), columns = sceneGraph_columns)

        map_files = os.listdir(image_path_full)
        map_files.reverse()
        px_per_meter = 6
        
        for map_file in map_files:
            if map_file.endswith('.json'):
                from nuscenes.map_expansion.map_api import NuScenesMap

                map_name = map_file[:-5]

                print(' Extracting map data for ' + map_name)

                map_image = NuScenesMap(dataroot = file_path, map_name = map_name)

                # Get the sceneGraph
                graph = self.getSceneGraphNuScenes(map_image)
                self.SceneGraphs.loc[map_name] = graph

                # Get the bit map 
                test_file = file_path + os.sep + 'Map_arrays' + os.sep + map_name + '.npy'
                if os.path.isfile(test_file):
                    bit_map = np.load(test_file, allow_pickle = True)
                else:
                    from NuScenes.vec_map import VectorMap
                    from NuScenes.nusc_utils import populate_vector_map
                    vector_map = VectorMap('placeholder:' + map_name)
                    populate_vector_map(vector_map, map_image)
                    bit_map = vector_map.rasterize(resolution = px_per_meter)
                    
                    # Get less memory intensive saving form
                    bit_map *= 255
                    bit_map = bit_map.astype(np.uint8)
                    
                    # Save bit_map
                    os.makedirs(os.path.dirname(test_file), exist_ok=True)
                    np.save(test_file, bit_map)
                    
                self.Images.loc[map_name] = [bit_map, 1 / px_per_meter]


        # Get trajectories
        data_obj = NuScenes(version = 'v1.0-trainval', dataroot = file_path, verbose = True)
        
        nuscenes_dt = 0.5

        pred_file = file_path + os.sep + 'maps' + os.sep + 'prediction' + os.sep + 'prediction_scenes.json'
        with open(pred_file, 'r') as f:
            pred_data = json.load(f)

        for data_idx, scene_record in enumerate(data_obj.scene):
            scene_name = scene_record['name']
            scene_desc = scene_record['description']
            scene_location = data_obj.get('log', scene_record['log_token'])['location']
            scene_length = scene_record['nbr_samples']

            print('Scene ' + str(data_idx + 1) + ' of ' + str(len(data_obj.scene)) +
                  ': ' + scene_name + ' (' + scene_desc + ')')
            
            try:
                scene_preds = pred_data[scene_name]
            except:
                scene_preds = []

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
            agent_sizes = pd.Series(np.zeros(0, np.ndarray), index = [])
            pred_points = pd.Series(np.zeros(0, np.ndarray), index = [])

            t = np.arange(scene_length) * nuscenes_dt

            all_agents_token = []
            agent_id = 1

            ego_translation_list = []
            ego_prediction_list = []
            for frame_idx, frame in enumerate(all_frames):
                # get the ego agent
                cam_front_data = data_obj.get('sample_data', frame['data']['CAM_FRONT'])
                ego_pose = data_obj.get('ego_pose', cam_front_data['ego_pose_token']) 
                ego_translation_list.append(np.array([ego_pose['translation']]))
                ego_pred_token = ego_pose['token'] + frame['token']
                ego_prediction_list.append(ego_pred_token in scene_preds)
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

                    all_agents_token.append(agent_data['instance_token'])
                    assert frame['token'] == agent_data['sample_token']

                    translation_list = [np.array(agent_data["translation"][:3])[np.newaxis]]
                    prev_idx = frame_idx 
                    curr_sample_ann_token = agent_data["next"]
                    pred_token = agent_data['instance_token'] + '_' + agent_data['sample_token'] 
                    Prediction = [pred_token in scene_preds]
                    while curr_sample_ann_token:
                        agent_data_new = data_obj.get('sample_annotation', curr_sample_ann_token)
                        translation = np.array(agent_data_new["translation"][:3])
                        
                        pred_token_new = agent_data_new['instance_token'] + '_' + agent_data_new['sample_token']

                        curr_idx = frame_idx_dict[agent_data_new["sample_token"]]
                        # check for missing frames
                        if curr_idx > prev_idx + 1:
                            fill_time = np.arange(prev_idx + 1, curr_idx)
                            xs = np.interp(fill_time, [prev_idx, curr_idx], [translation_list[-1][0,0], translation[0]])
                            ys = np.interp(fill_time, [prev_idx, curr_idx], [translation_list[-1][0,1], translation[1]])
                            zs = np.interp(fill_time, [prev_idx, curr_idx], [translation_list[-1][0,2], translation[2]])
                            translation_list.append(np.stack((xs, ys, zs), axis = 1))
                            Prediction.extend([False] * len(fill_time))

                        translation_list.append(translation[np.newaxis])
                        Prediction.append(pred_token_new in scene_preds)
                        
                        prev_idx = curr_idx
                        curr_sample_ann_token = agent_data_new["next"]

                    translations = np.concatenate(translation_list, axis = 0)
                    Prediction = np.array(Prediction)

                    agent_name = 'v_' + str(agent_id)
                    if agent_category.startswith('vehicle'):
                        agent_types[agent_name] = 'V'
                        if agent_category.startswith('vehicle.bicycle'):
                            agent_types[agent_name] = 'B'
                        elif agent_category.startswith('vehicle.motorcycle'):
                            agent_types[agent_name] = 'M'
                    elif agent_category.startswith('human'):    
                        agent_types[agent_name] = 'P'

                    agent_sizes[agent_name] = np.array([agent_data['size'][1], agent_data['size'][0]]) # length, width

                    agent_traj = np.ones((scene_length, 2)) * np.nan
                    agent_traj[frame_idx:frame_idx + len(translations),:] = translations[:,:2]

                    agent_pred = np.zeros((scene_length,), bool)
                    agent_pred[frame_idx:frame_idx + len(Prediction)] = Prediction

                    path[agent_name] = agent_traj * np.array([[1, -1]]) # Align with Images
                    pred_points[agent_name] = agent_pred
                    agent_id += 1
            
            ego_translations = np.concatenate(ego_translation_list, axis = 0)
            ego_predictions = np.array(ego_prediction_list)
            path['tar'] = ego_translations[:,:2] * np.array([[1, -1]]) # Align with Images
            pred_points['tar'] = ego_predictions
            agent_types['tar'] = 'V'
            agent_sizes['tar'] = np.array([5.0, 2.0]) # length, width

            domain = pd.Series(np.zeros(7, object), index = ['location', 'scene', 'image_id', 'graph_id',
                                                             'pred_agents', 'pred_timepoints', 'splitting'])
            domain.location = scene_location
            domain.scene = scene_name 
            domain.image_id = scene_location
            domain.graph_id = scene_location
            domain.pred_agents = pred_points
            domain.pred_timepoints = t
            
            if scene_name in train:
                domain.splitting = 'train'
            elif scene_name in val:
                domain.splitting = 'test'
            else:
                print('This Scene is not sorted into train or testing split.')
                continue
            
            print('Number of agents: ' + str(len(path)))
            print('Number of frames: ' + str(len(t)))
            print('Number of predictions: ' + str(np.stack(pred_points.to_numpy()).sum()) + '/' + str(len(scene_preds)))
            self.num_samples += 1
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.Size_old.append(agent_sizes)
            self.T.append(t)
            self.Domain_old.append(domain)
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
        self.Size_old = pd.DataFrame(self.Size_old)
        
    

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
                if agent_types[agent] == 'P':
                    path[agent] = self.extrapolate_path(path[agent], t, mode = 'pos')
                else:
                    path[agent] = self.extrapolate_path(path[agent], t, mode = 'vel')
        
        return path, agent_types
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_dashed = []
        return lines_solid, lines_dashed
