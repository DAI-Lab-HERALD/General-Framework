import cv2
import numpy as np
import os
import pandas as pd

from data_set_template import data_set_template
from Scenarios.scenario_none import scenario_none

class THOR(data_set_template):
    def get_name(self=None):
        names = {'print': 'THOR',
                 'file': 'THOR',
                 'latex': r'<\emph{Dataset} name>'}
        
        return names
    
    def future_input(self=None):
        return False
    
    def includes_images(self=None):
        return True
    
    def set_scenario(self):
        self.set_scenario = scenario_none()

        return self.set_scenario()
    
    
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'THOR' + os.sep + 'THOR_processed.pkl')
        # analize raw data
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []

        self.Images = pd.DataFrame(np.zeros((1, 1), object), 
                            index = self.Data.index[:1], columns = ['Image'])
        
        self.Images['Target_MeterPerPx'] = 0.0
        
        for loc_i, location in enumerate(np.unique(self.Data.scenario)):
            Loc_data = self.Data[self.Data.scenario == location].copy()
            first_frame_all = np.min(Loc_data['First frame'])
            last_frame_all  = np.max(Loc_data['Last frame'])
            
            t = np.arange(first_frame_all, last_frame_all + 1) * 0.01
            
            Loc_data['First frame'] -= first_frame_all
            Loc_data['Last frame']  -= first_frame_all
            num_timesteps = len(t)
            
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])

            if location == 'Experiment_1' or location == 'Experiment_2':
                img_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                        'THOR' + os.sep + 'data' + os.sep + 
                        'orebro_map.png')
            else:
                img_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                        'THOR' + os.sep + 'data' + os.sep + 
                        'orebro_map_exp3.png')
                
            Meta_data = pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
                                'THOR' + os.sep + 'data' + os.sep + 
                                location + '.csv')
            
            for i in range(len(Loc_data)):
                path_i = Loc_data.iloc[i]
                
                traj = np.ones((num_timesteps, 2), float) * np.nan
                
                traj_exist = path_i.path[['x', 'y']].to_numpy()
                
                traj[path_i['First frame'] : path_i['Last frame'] + 1] = traj_exist
                
                name = 'v_' + str(i)
                
                path[name] = traj
                agent_types[name] = 'P'

                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                self.Images.Image.loc[i] = np.array(img)
                self.Images.Target_MeterPerPx.loc[i] = Meta_data['MeterPerPx'][0]
                self.Images.rename(index={i: location}, inplace=True)

            
            domain = pd.Series(np.zeros(2, object), index = ['location', 'name'])
            domain.location = location
            domain.name     = str(loc_i)
            domain.image_id = location
            domain.x_center = Meta_data['x_center'][0]
            domain.y_center = Meta_data['y_center'][0]
            domain.rot_angle = Meta_data['rot_angle'][0]
            
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
            self.num_samples = self.num_samples + 1
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)


    def calculate_distance(self):
        return None
    
    def evaluate_scenario(self):
        return None
    
    def calculate_additional_distances(self):
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
                x = np.interp(t,t[useful],x[useful])
                y = np.interp(t,t[useful],y[useful])
            
                path[agent] = np.stack([x, y], axis = -1)
        
        return path, agent_types
    

    
    def provide_map_drawing(self):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed