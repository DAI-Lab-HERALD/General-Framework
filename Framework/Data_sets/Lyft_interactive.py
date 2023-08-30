import numpy as np
import pandas as pd
from scipy import interpolate as interp

import os

from data_set_template import data_set_template
from scenario_none import scenario_none

import trajdata
from trajdata import UnifiedDataset
# from trajdata import AgentBatch, UnifiedDataset, MapAPI, VectorMap


class Lyft_interactive(data_set_template):
    '''
    The Lyft dataset is recored by an AV driving around a city, including gneral
    human bahavior in various situations.
    
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
    
    def get_scenario(self):
        self.scenario = scenario_none()
        
    def create_path_samples(self):
        # Prepare output
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        
        data_path = self.path + os.sep + 'Lyft' + os.sep + 'data' + os.sep
        
        dataset = UnifiedDataset(desired_data=["lyft_sample"],
                                 data_dirs={"lyft_sample": data_path})
        
        assert False

        
        
        
        
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