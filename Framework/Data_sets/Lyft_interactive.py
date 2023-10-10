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


class Lyft_interactive(data_set_template):
    '''
    The Lyft dataset is recored by an AV driving around a city, including gneral
    human bahavior in various situations.

    This dataset is imported using the trajflow libary, which require that numpy >= 1.20.0
    and numpy <= 1.20.3 is installed.
    
    The data can be found at https://woven-planet.github.io/l5kit/dataset.html
    amd the following citation can be used:
        
    Houston, J., Zuidhof, G., Bergamini, L., Ye, Y., Chen, L., Jain, A., ... & Ondruska, P. 
    (2021, October). One thousand and one hours: Self-driving motion prediction dataset. 
    In Conference on Robot Learning (pp. 409-418). PMLR.
    '''
    
    def get_name(self = None):
        names = {'print': 'Lyft Level-5',
                 'file': 'LyftLevel5',
                 'latex': '\emph{Lyft}'}
        return names
    
    def future_input(self = None):
        return False
    
    def includes_images(self = None):
        return True
    
    def set_scenario(self):
        self.scenario = scenario_none()
        
    def create_path_samples(self):
        # Prepare output
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        # The framed code should be the only dataset specific part whenusing trajdata
        ################################################################################
        data_path = self.path + os.sep + 'Data_sets' + os.sep + 'Lyft' + os.sep + 'data' + os.sep
        
        cache_path  = data_path + '.unified_data_cache'
        scenes_path = data_path + 'scenes' + os.sep
        
        dataset = UnifiedDataset(desired_data = ["lyft_sample"], #, "lyft_train", "lyft_val", "lyft_train_full"],
                                 data_dirs = {"lyft_sample":     scenes_path + 'sample.zarr',
                                              "lyft_train":      scenes_path + 'train.zarr',
                                              "lyft_val":        scenes_path + 'validate.zarr',
                                              "lyft_train_full": data_path + 'train_full.zarr'},
                                 cache_location = cache_path,
                                 verbose = True)
        ################################################################################
        map_api = MapAPI(Path(cache_path))
        # Go over scenes
        for i, scene in enumerate(dataset.scenes()):
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
            scene_data = Cache.scene_data_df[['x', 'y']]
            scene_data = scene_data.loc[scene_agents[:,0]]
            
            # Set indices
            sort_index = np.argsort(scene_agents[:,0])
            agent_index = scene_data.index.get_level_values(0).to_numpy()
            agent_index = sort_index[np.searchsorted(scene_agents[sort_index,0], agent_index)]
            times_index = scene_data.index.get_level_values(1).to_numpy()
            
            # Set trajectories
            trajectories = np.ones((len(scene_agents),scene.length_timesteps, 2), dtype = np.float32) * np.nan
            trajectories[agent_index, times_index] = scene_data.to_numpy()
            
            # Adjust to map
            trajectories -= np.array([[[min_x, min_y]]])
            trajectories[...,1] *= -1
            
            # Get agent names
            assert scene_agents[0,0] == 'ego'
            Index = ['tar'] + ['v_' + str(i) for i in range(1, len(scene_agents))]
            
            # Set path and agent types
            path = pd.Series(list(trajectories), dtype = object, index = Index)
            agent_types = pd.Series(scene_agents[:,1].astype('<U1'), index = Index)
            
            # Get timesteps
            t = np.arange(scene.length_timesteps) * scene.dt
            
            # Set domain
            domain = pd.Series(np.zeros(3, object), index = ['location', 'scene', 'image_id'])
            domain.location = scene.env_name
            domain.scene = scene.name 
            domain.image_id = map_id
            
            print('Number of agents: ' + str(len(path)))
            self.num_samples += 1
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
            
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)  
        
        # Get images 
        self.Images = pd.DataFrame(np.zeros((len(map_api.maps), 2), object), 
                                   index = list(map_api.maps.keys()),
                                   columns = ['Image', 'Target_MeterPerPx'])
        
        # set image resolution
        px_per_meter = 2
        self.Images.Target_MeterPerPx = 1 / px_per_meter
        
        # cycle through images
        max_height = 0
        max_width = 0
        for map_key in self.Images.index:
            img = map_api.maps[map_key].rasterize(px_per_meter)
            self.Images.Image.loc[map_key] = img
            max_width = max(img.shape[1], max_width)
            max_height = max(img.shape[0], max_height)
                
        # pad images
        for loc_id in self.Images.index:
            img = self.Images.Image.loc[loc_id]
            img_pad = np.pad(img, ((0, max_height - img.shape[0]),
                                   (0, max_width  - img.shape[1]),
                                   (0,0)), 'constant', constant_values=0)
            self.Images.Image.loc[loc_id] = img_pad        

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