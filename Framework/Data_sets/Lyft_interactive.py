import numpy as np
import pandas as pd
from scipy import interpolate as interp

import os

from data_set_template import data_set_template
from scenario_none import scenario_none

from trajdata import UnifiedDataset
from trajdata.data_structures.scene_metadata import Scene
from trajdata.simulation import SimulationScene
from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.scene import SceneTimeAgent
# from trajdata import AgentBatch, UnifiedDataset, MapAPI, VectorMap


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
    
    def include_images(self = None):
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
        
        
        data_path = self.path + os.sep + 'Data_sets' + os.sep + 'Lyft' + os.sep + 'data'
        zarr_path = data_path + os.sep + 'scenes' + os.sep + 'sample.zarr'
        
        dataset = UnifiedDataset(desired_data=["lyft_sample"],
                                 data_dirs={"lyft_sample": zarr_path})
        # Go over scenes
        for i, scene in enumerate(dataset.scenes()):
            print('Scene ' + str(i + 1) + ': ' + scene.name)
            print('Number of frames: ' + str(scene.length_timesteps) + ' (' + str(scene.length_seconds()) + 's)')
            
            # Prepare new scene data
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])

            scene_data_path = dataset._scene_index[i]

            
            sim_scene = SimulationScene(env_name = scene.env_name,
                                        scene_name = scene.name,
                                        scene = scene,
                                        dataset = dataset,
                                        init_timestep = 0,
                                        )
            scene_data = sim_scene.cache.scene_data_df[['x', 'y']]
            # Get timesteps
            t = np.arange(scene.length_timesteps) * scene.dt
            
            for agent in scene.agents:
                if agent.type == AgentType.UNKNOWN:
                    continue
                print('Agent ' + str(agent.name) + ': ' + str(agent.type))
                
                if agent.name == 'ego':
                    path_name = 'tar'
                else:
                    path_name = 'v_' + str(agent.name)

                traj = np.ones((scene.length_timesteps, 2)) * np.nan
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