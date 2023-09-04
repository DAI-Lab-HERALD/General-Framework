import numpy as np
import pandas as pd
from scipy import interpolate as interp
from pathlib import Path

import os
import shutil

from data_set_template import data_set_template
from scenario_none import scenario_none

from trajdata import UnifiedDataset, MapAPI
from trajdata.simulation import SimulationScene
from trajdata.data_structures.agent import AgentType


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
        
        dataset = UnifiedDataset(desired_data = ["lyft_sample", "lyft_train", "lyft_val"], #, "lyft_train_full"],
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
            
            # Prepare new scene data
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            
            # Get timesteps
            t = np.arange(scene.length_timesteps) * scene.dt
            
            # Get map
            map_id = scene.env_name + ':' + scene.location
            map_api.get_map(map_id)
            
            # Get map offset
            min_x, min_y, _, _, _, _ = map_api.maps[map_id].extent 
            
            # Get useful agents 
            useful_agent = np.array([agent.type != AgentType.UNKNOWN for agent in scene.agents])
            
            # Get agent presence matrix to determine advantageous time indices
            agent_present = pd.DataFrame(np.zeros((scene.length_timesteps, len(scene.agents)), bool),
                                         columns = scene.agents)
            
            for t_ind in agent_present.index:
                agent_present.loc[t_ind][scene.agent_presence[t_ind]] = True
                
            agent_present_array = agent_present.to_numpy()[:,useful_agent]
            
            # Get greedy algortihms for minimal set cover problem
            needed_rows = []
            while agent_present_array.shape[1] > 0:
                new_row = agent_present_array.sum(1).argmax()
                
                needed_rows.append(new_row)
                agent_present_array = agent_present_array[:,~agent_present_array[new_row]]
            
            needed_rows = np.sort(needed_rows)
            
            # Go through all needed agents
            for t_ind in needed_rows:
                sim_scene = SimulationScene(env_name = scene.env_name,
                                            scene_name = scene.name,
                                            scene = scene,
                                            dataset = dataset,
                                            init_timestep = t_ind,
                                            )
                scene_data = sim_scene.cache.scene_data_df[['x', 'y']]
                
                for agent in scene.agent_presence[t_ind]:
                    # Check if agent type is allowable
                    if agent.type == AgentType.UNKNOWN:
                        continue
                    
                    if agent.name == 'ego':
                        path_name = 'tar'
                    else:
                        path_name = 'v_' + str(agent.name)
                        
                    # Check if agent is already included
                    if path_name in path.index:
                        continue
    
                    traj = np.ones((scene.length_timesteps, 2), dtype = np.float32) * np.nan
                    
                    # Get observed trajectories
                    traj_observed = scene_data.loc[agent.name].to_numpy().astype(np.float32)
                    
                    # Adjust trajectories to map
                    traj_observed = traj_observed - np.array([[min_x, min_y]])
                    traj_observed = traj_observed * np.array([[1.0, -1.0]])
                    
                    traj[agent.first_timestep:agent.last_timestep + 1] = scene_data.loc[agent.name].to_numpy()
                    path[path_name] = traj
                    if agent.type == AgentType.VEHICLE:
                        agent_types[path_name] = 'V'
                    elif agent.type == AgentType.PEDESTRIAN:
                        agent_types[path_name] = 'P'
                    elif agent.type == AgentType.MOTORCYCLE:
                        agent_types[path_name] = 'M'
                    elif agent.type == AgentType.BICYCLE:
                        agent_types[path_name] = 'B'
                    else:   
                        raise ValueError('Unknown agent type: ' + str(agent.type))
                        
            domain = pd.Series(np.zeros(3, object), index = ['location', 'scene', 'image_id'])
            domain.location = scene.env_name
            domain.scene = scene.name 
            domain.image_id = map_id
            
            # Reset path names
            num_index = len(path.index)
            assert path.index[0] == 'tar'
            
            Index = ['tar'] + ['v_' + str(i) for i in range(1, num_index)]
            path.index = Index
            
            
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