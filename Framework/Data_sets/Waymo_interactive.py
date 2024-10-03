import numpy as np
import pandas as pd
from scipy import interpolate as interp
from pathlib import Path

import os
import shutil

from data_set_template import data_set_template
from scenario_none import scenario_none

from trajdata import UnifiedDataset, MapAPI
from trajdata.data_structures.agent import AgentType
from trajdata.caching.df_cache import DataFrameCache


class Waymo_interactive(data_set_template):
    '''
    The Lyft dataset is recored by an AV driving around a city, including gneral
    human bahavior in various situations.

    This dataset is imported using the trajflow libary, which require that numpy >= 1.20
    and numpy <= 1.23 is installed.
    
    The data can be found at https://woven-planet.github.io/l5kit/dataset.html
    amd the following citation can be used:
        
    Houston, J., Zuidhof, G., Bergamini, L., Ye, Y., Chen, L., Jain, A., ... & Ondruska, P. 
    (2021, October). One thousand and one hours: Self-driving motion prediction dataset. 
    In Conference on Robot Learning (pp. 409-418). PMLR.
    '''
    
    def get_name(self = None):
        names = {'print': 'Waymo Open Motion',
                 'file': 'Waymo_data',
                 'latex': '\emph{Waymo}'}
        return names
    
    def future_input(self = None):
        return False
    
    def includes_images(self = None):
        return True 
    
    def includes_sceneGraphs(self = None):
        return False
    
    def set_scenario(self):
        self.scenario = scenario_none()
    
    def path_data_info(self = None):
        return ['x', 'y', 'v_x', 'v_y', 'theta']
        
    def create_path_samples(self):
        # Prepare output
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.Size_old = []
        self.T = []
        self.Domain_old = []

        # Get images 
        self.Images = pd.DataFrame(np.zeros((0, 2), object), index = [], columns = ['Image', 'Target_MeterPerPx'])
        px_per_meter = 2

        # Get scenegraphs
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                            'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), columns = sceneGraph_columns)
        
        # The framed code should be the only dataset specific part whenusing trajdata
        ################################################################################
        data_path = self.path + os.sep + 'Data_sets' + os.sep + 'Waymo' + os.sep + 'data' + os.sep
        
        cache_path  = data_path + '.unified_data_cache'
        
        dataset = UnifiedDataset(desired_data = ["waymo_train", "waymo_val", "waymo_test"],
                                 data_dirs = {"waymo_train": data_path,
                                              "waymo_val":   data_path,
                                              "waymo_test":  data_path},
                                 cache_location = cache_path,
                                 verbose = True)
        # [waymo_train-train, val-waymo_val, waymo_test-test]
        testing_env_names = ["waymo_test"]
        ################################################################################
        
        map_api = MapAPI(Path(cache_path))

        # Go over scenes to collect maps
        for i, scene in enumerate(dataset.scenes()):
            # Get map
            map_id = scene.env_name + ':' + scene.location
            map_api.get_map(map_id)

        # Save images
        for map_key in list(map_api.maps.keys()):
            if not map_key in self.Images.index:
                map_image = map_api.maps[map_key]

                # Get the sceneGraph
                graph = self.getSceneGraphTrajdata(map_image)
                self.SceneGraphs.loc[map_key] = graph


                img = map_image.rasterize(px_per_meter)
                
                # Get less memory intensive saving form
                if (img.dtype != np.uint8) and (img.max() <= 1.0): 
                    img *= 255.0
                        
                self.Images.loc[map_key] = [img.astype(np.uint8), 1 / px_per_meter] 

        
        # Get allready saved samples
        num_samples_saved = self.get_number_of_saved_samples()
        
        # Go over scenes
        for i, scene in enumerate(dataset.scenes()):
            if i < num_samples_saved:
                continue
            
            print('')
            print('Scene ' + str(i + 1) + ': ' + scene.name)
            print('Number of frames: ' + str(scene.length_timesteps) + ' (' + str(scene.length_seconds()) + 's)')
            
            # Get map
            map_id = scene.env_name + ':' + scene.location
            map_api.get_map(map_id)
            
            # Get map offset
            min_x, min_y, _, _, _, _ = map_api.maps[map_id].extent 
            
            Cache = DataFrameCache(cache_path = dataset.cache_path,
                                   scene = scene)
            
            scene_agents = np.array([[agent.name, agent.type.name] for agent in scene.agents if agent.type != AgentType.UNKNOWN])
            
            # Extract position data
            scene_data = Cache.scene_data_df[['x', 'y', 'vx', 'vy', 'heading', 'length', 'width']]
            scene_data = scene_data.loc[scene_agents[:,0]]
            
            # Set indices
            sort_index = np.argsort(scene_agents[:,0])
            agent_index = scene_data.index.get_level_values(0).to_numpy()
            agent_index = sort_index[np.searchsorted(scene_agents[sort_index,0], agent_index)]
            times_index = scene_data.index.get_level_values(1).to_numpy()
            
            # Set trajectories
            trajectories = np.ones((len(scene_agents),scene.length_timesteps, 2), dtype = np.float32) * np.nan
            trajectories[agent_index, times_index] = scene_data.to_numpy()

            # Get aveage sizes
            sizes = np.ones((len(scene_agents), scene.length_timesteps, 2), dtype = np.float32) * np.nan
            sizes[agent_index, times_index] = scene_data[['length', 'width']].to_numpy()
            sizes = np.nanmax(sizes, axis = 1)
            
            # Adjust to map
            trajectories -= np.array([[[min_x, min_y, 0, 0, 0]]])
            trajectories[...,1] *= -1 # mirror y_position
            trajectories[...,3] *= -1 # mirror y_velocity
            trajectories[...,4] *= -1 # mirror heading

            # Get agent names
            assert scene_agents[0,0] == 'ego'
            Index = ['tar'] + ['v_' + str(i) for i in range(1, len(scene_agents))]
            
            # Set path and agent types
            path = pd.Series(list(trajectories), dtype = object, index = Index)
            agent_types = pd.Series(scene_agents[:,1].astype('<U1'), index = Index)
            agent_sizes = pd.Series(list(sizes), index = Index)
            
            # Get timesteps
            t = np.arange(scene.length_timesteps) * scene.dt
            
            # Set domain
            domain = pd.Series(np.zeros(4, object), index = ['location', 'scene', 'image_id', 'splitting'])
            domain.location = scene.env_name
            domain.scene = scene.name 
            domain.image_id = map_id
            
            # Get sample purpose
            if scene.env_name in testing_env_names:
                domain.splitting = 'test'
            else:
                domain.splitting = 'train'
            
            print('Number of agents: ' + str(len(path)))
            self.num_samples += 1
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.Size_old.append(agent_sizes)
            self.T.append(t)
            self.Domain_old.append(domain)
            
            # Chcek if data can be saved
            self.check_created_paths_for_saving()
    
        self.check_created_paths_for_saving(last = True) 

        # deletet cached data
        shutil.rmtree(cache_path)


        
        
        
        
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