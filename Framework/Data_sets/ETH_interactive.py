import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
from scipy import interpolate as interp

pd.set_option('mode.chained_assignment',None)

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class ETH_interactive(data_set_template):
    '''
    The ETH/UCY dataset is one of the most commonly used pedestrian dataset
    for general trajectroy prediction models. It consists out of five parts
    recorded at the following locations (for the location splitting, it would
    be in the following order):
    - 'eth_hotel'
    - 'eth_univ'
    - 'ucy_univ'
    - 'ucy_zara01'
    - 'ucy_zara02'
    
    The raw data can be found at 
    https://github.com/cschoeller/constant_velocity_pedestrian_motion/tree/master/data
    and the following two citations can be used:
        
    Pellegrini, S., Ess, A., Schindler, K., & Van Gool, L. (2009, September). You'll 
    never walk alone: Modeling social behavior for multi-target tracking. In 2009 IEEE 
    12th international conference on computer vision (pp. 261-268). IEEE.
    
    Lerner, A., Chrysanthou, Y., & Lischinski, D. (2007, September). Crowds by example. 
    In Computer graphics forum (Vol. 26, No. 3, pp. 655-664). Oxford, UK: Blackwell 
    Publishing Ltd.
    '''
    def set_scenario(self):
        self.scenario = scenario_none()
        
        def eth_classifying_agents():
            return []
        
        self.scenario.classifying_agents = eth_classifying_agents
    
    def path_data_info(self = None):
        return ['x', 'y']
        
        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'ETH_pedestrians' + os.sep + 'ETH_processed.pkl')
        # analize raw data
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        for loc_i, location in enumerate(np.unique(self.Data.scenario)):
            Loc_data = self.Data[self.Data.scenario == location].copy()
            first_frame_all = np.min(Loc_data['First frame'])
            last_frame_all  = np.max(Loc_data['Last frame'])
            
            t = np.arange(first_frame_all, last_frame_all + 1) * 0.4
            
            Loc_data['First frame'] -= first_frame_all
            Loc_data['Last frame']  -= first_frame_all
            num_timesteps = len(t)
            
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            
            for i in range(len(Loc_data)):
                path_i = Loc_data.iloc[i]
                
                traj = np.ones((num_timesteps, 2), float) * np.nan
                
                traj_exist = path_i.path[['x', 'y']].to_numpy()
                
                traj[path_i['First frame'] : path_i['Last frame'] + 1] = traj_exist
                
                name = 'v_' + str(i)
                
                path[name] = traj
                agent_types[name] = 'P'
            
            domain = pd.Series(np.zeros(2, object), index = ['location', 'name'])
            domain.location = location
            domain.name     = str(loc_i)
            
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
            self.num_samples = self.num_samples + 1
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
    
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
                path[agent] = self.extrapolate_path(path[agent], t, mode = 'pos')
        
        return path, agent_types
            
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'ETH (pedestrians)',
                 'file': 'ETH_ped_ia',
                 'latex': r'\emph{ETH/UCY}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return False 
    
    def includes_sceneGraphs(self = None):
        return False