import cv2
import glob
import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
import scipy
import torch

from PIL import Image
from scipy import interpolate as interp

pd.set_option('mode.chained_assignment',None)

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track

class CommonRoad(data_set_template):
    '''
    TODO: Add docstring
    '''
    def set_scenario(self):
        self.scenario = scenario_none()

        # extension = 'csv'
        # os.chdir(self.path + os.sep + 'Data_sets' + os.sep + 
        #                            'CommonRoad' + os.sep + 'data')
        # result = glob.glob('*.{}'.format(extension))
        
        # os.chdir(self.path)

        # Loc_data_pix = pd.DataFrame(np.zeros((len(result),4),float), columns = ['xCenter', 'yCenter', 'r'])
        # self.Loc_scale = {}
        # self.Loc_data = {}

        # for i in range(len(result)):
        #     Meta_data=pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
        #                            'CommonRoad' + os.sep + 'data' + os.sep + result[i] + '.csv')
        #     Loc_data_pix.xCenter.iloc[i] = Meta_data['x_center']
        #     Loc_data_pix.yCenter.iloc[i] = Meta_data['y_center']
        #     Loc_data_pix.r.iloc[i] = Meta_data['rot_angle']

        #     self.Loc_scale[i] = Meta_data['MeterToPx']
        #     self.Loc_data.iloc[i] = Loc_data_pix.iloc[i] * (1/self.Loc_scale[i])

        
        
    def create_path_samples(self): 
        # Load raw data
        print('in create_path_samples')
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'CommonRoad' + os.sep + 'CommonRoad_processed.pkl')
        
        print('Got data')
        # analize raw dara 
        num_tars = len(self.Data)
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []

        self.Images = pd.DataFrame(np.zeros((1, 1), object), 
                            index = self.Data.index[:1], columns = ['Image'])
        
        print('Got images')
        
        print('Got images')
        
        self.Images['Target_MeterPerPx'] = 0.0
        # extract raw samples
        max_width = 0
        max_height = 0

        unique_scenarios = np.unique(self.Data.scenario)

        i_csv = np.where(unique_scenarios == 'ZAM_Tjunction-1_396_T-1')[0][0]

        unique_scenarios = np.concatenate((unique_scenarios[[i_csv]], 
                                           unique_scenarios[np.arange(len(unique_scenarios)) != i_csv]))
        for i_scenario, scenario in enumerate(unique_scenarios):
            data_scenario = self.Data[self.Data.scenario == scenario]

            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            for i in range(len(data_scenario)):
                print('scenario  {}/{}: agent {}/{}'.format(i_scenario + 1, len(unique_scenarios), i + 1, len(data_scenario)))
                data_i = data_scenario.iloc[i]
                
                # img = Image.open(img_file)
                if i_scenario == 0 and i==0:
                    img_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                                'CommonRoad' + os.sep + 'data' + os.sep + 
                                data_i.scenario + '.png')
                    
                    Meta_data = pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
                                'CommonRoad' + os.sep + 'data' + os.sep + 
                                data_i.scenario + '.csv')

                    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    self.Images.Image.loc[i] = np.array(img)
                    self.Images.Target_MeterPerPx.loc[i] = Meta_data['MeterToPx'][0]
                    self.Images.rename(index={i: data_i.scenario}, inplace=True)


                max_width = max(img.shape[0], max_width)
                max_height = max(img.shape[1], max_height)
                    
        
                # find crossing point
                track_all = data_i.path.copy(deep = True)
                if i == 0:
                    agent_id = 'tar'
                    t = track_all.t.to_numpy()
                else:
                    agent_id = 'v_{}'.format(i)
                    assert (t == track_all.t.to_numpy()).all(), 'timesteps do not match'

                path[agent_id] = np.stack([track_all.x.to_numpy(), track_all.y.to_numpy()], axis = -1)
                agent_types[agent_id] = data_i.type

                if i == 0:
                    domain = pd.Series(np.zeros(8, object), 
                                    index = ['location', 'neighbors', 'name', 'type', 'image_id', 'rot_angle', 'x_center', 'y_center'])
                    domain.location = data_i.scenario
                    domain.image_id = self.Data.iloc[0].scenario
                    domain.type = data_i.type
                    domain.x_center = Meta_data['x_center'][0]
                    domain.y_center = Meta_data['y_center'][0]
                    domain.rot_angle = Meta_data['rot_angle'][0]
                else:
                    # Check if Domain is consistent
                    assert domain.location == data_i.scenario, 'Domain is not consistent'
                    assert domain.image_id == self.Data.iloc[0].scenario, 'Domain is not consistent'
                    assert domain.type == data_i.type, 'Domain is not consistent'
                    assert domain.x_center == Meta_data['x_center'][0], 'Domain is not consistent'
                    assert domain.y_center == Meta_data['y_center'][0], 'Domain is not consistent'
                    assert domain.rot_angle == Meta_data['rot_angle'][0], 'Domain is not consistent'


            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
            self.num_samples = self.num_samples + 1

        # Remove duplicate image indices
        self.Images = self.Images[~self.Images.index.duplicated(keep='first')]
        print('Removed duplicate images')

        # pad images to max size
        for i in range(1):
            data_i = self.Data.iloc[i]
            img = self.Images.loc[data_i.scenario].Image
            img_pad = np.pad(img, ((0, max_height - img.shape[0]),
                                   (0, max_width  - img.shape[1]),
                                   (0,0)), 'constant', constant_values=0)
            self.Images.Image.loc[i] = img_pad 

        print('padded images')
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)

        print('done')
    
    def calculate_distance(self, path, t, domain):
        r'''
        This function calculates the abridged distance of the relevant agents in a scenarion
        for each of the possible classification type. If the classification is not yet reached,
        thos distances are positive, while them being negative means that a certain scenario has
        been reached.
    
        Parameters
        ----------
        path : pandas.Series
            A pandas series of :math:`(2 N_{agents})` dimensions,
            where each entry is itself a numpy array of lenght :math:`\{n \times |T|\}`, the number of recorded timesteps.
        t : numpy.ndarray
            A numpy array of lenght :math:`|T|`, recording the corresponding timesteps.
    
        Returns
        -------
        Dist : pandas.Series
            This is a :math:`N_{classes}` dimensional Series.
            For each column, it returns an array of lenght :math:`|T|` with the distance to the classification marker.
        '''
        return None
    
    def evaluate_scenario(self, path, D_class, domain):
        r'''
        This function says weither the agents are in a position at which they fullfill their assigned roles.
    
        Parameters
        ----------
        path : pandas.Series
            A pandas series of :math:`(2 N_{agents})` dimensions,
            where each entry is itself a numpy array of lenght :math:`|T|`, the number of recorded timesteps.
        t : numpy.ndarray
            A numpy array of lenght :math:`|T|`, recording the corresponding timesteps.
    
        Returns
        -------
        in_position : numpy.ndarray
            This is a :math:`|T|` dimensioanl boolean array, which is true if all agents are
            in a position where the classification is possible.
        '''
        return None
        
    def calculate_additional_distances(self, path, t, domain):
        r'''
        This function calculates other distances of the relevant agents needed for the 2D->1D transformation 
        of the input data. The returned distances must not be nan, so a method has to be designed
        which fills in those distances if they are unavailable
    
        Parameters
        ----------
        path : pandas.Series
            A pandas series of :math:`(2 N_{agents})` dimensions,
            where each entry is itself a numpy array of lenght :math:`|T|`, the number of recorded timesteps.
        t : numpy.ndarray
            A numpy array of lenght :math:`|T|`, recording the corresponding timesteps.
    
        Returns
        -------
        Dist : pandas.Series
            This is a :math:`N_{other dist}` dimensional Series.
            For each column, it returns an array of lenght :math:`|T|` with the distance to the classification marker..
            
            If self.can_provide_general_input() == False, this will be None.
        '''
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
                x = interp.interp1d(t[useful], x[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
                y = interp.interp1d(t[useful], y[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
            
                path[agent] = np.stack([x, y], axis = -1)
        
        return path, agent_types
            
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'CommonRoad',
                 'file': 'CommonRoad',
                 'latex': r'\emph{CR}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return True
