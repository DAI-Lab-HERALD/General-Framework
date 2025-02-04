import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
import pickle

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class Toy_bi_modal(data_set_template):   
    
    def set_scenario(self):
        self.scenario = scenario_none()
    
    def path_data_info(self = None):
        return ['x', 'y']
        
        
    def create_path_samples(self): 
        # Load raw data
        Data_out = pickle.load(open(self.path + os.sep + 'Data_sets' + os.sep + 
                                    'Toy_bi_modal' + os.sep + 'data' + os.sep + 'aug_future_trajectories', 'rb'))
        Data_in  = pickle.load(open(self.path + os.sep + 'Data_sets' + os.sep + 
                                    'Toy_bi_modal' + os.sep + 'data' + os.sep + 'aug_past_trajectories', 'rb'))
        # analize raw dara 
        num_traj = len(Data_out)
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        # extract raw samples
        for i in range(num_traj):
            data_in = Data_in[i]
            data_out = Data_out[i]
            
            data = np.concatenate((data_in, data_out), axis = 0)
            
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])

            path['tar'] = data
            agent_types['tar'] = 'P'
            
            t = np.arange(len(data)) * 0.25
            
            domain = pd.Series(np.zeros(1, object), index = ['location'])
            domain.location = 'normal'
            
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
        return path, agent_types
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'Toy problem (Bi-modal)',
                 'file': 'BM_toy_pro',
                 'latex': r'\emph{Toy}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return False 
    
    def includes_sceneGraphs(self = None):
        return False
    
    def has_repeated_inputs(self):
        return True