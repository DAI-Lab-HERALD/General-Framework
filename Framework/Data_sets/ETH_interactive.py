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
    def set_scenario(self):
        self.scenario = scenario_none()
        
        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'ETH_pedestrians' + os.sep + 'ETH_processed.pkl')
        # analize raw dara 
        num_tars = len(self.Data)
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        # extract raw samples
        max_number_other = 0
        for i in range(num_tars):
            data_i = self.Data.iloc[i]
            
            # Get other agents
            other_agents_bool = ((self.Data.index != data_i.name) & 
                                 (self.Data['First frame'] <= data_i['Last frame']) & 
                                 (self.Data['Last frame'] >= data_i['First frame']) &
                                 (self.Data.scenario == data_i.scenario))
            
            
            other_agents = self.Data.loc[other_agents_bool]
            
            self.Data.iloc[i].path['CN'] = np.empty((len(data_i.path), 0)).tolist()
            
            for j, frame in enumerate(data_i.path.index):
                useful = (other_agents['Last frame'] >= frame) & (other_agents['First frame'] <= frame)
                self.Data.iloc[i].path.CN.iloc[j] = list(other_agents.index[useful])
                max_number_other = max(max_number_other, useful.sum())
                
            
            # find crossing point
            track_all = data_i.path.copy(deep = True)
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            
            path['tar'] = np.stack([track_all.x.to_numpy(), track_all.y.to_numpy()], axis = -1)
            agent_types['tar'] = 'P'
            
            t = track_all.t.to_numpy()
            
            domain = pd.Series(np.zeros(3, object), index = ['location', 'neighbors', 'name'])
            domain.location = data_i.scenario
            domain.name = data_i.name
            track_all = track_all.set_index('t')
            domain.neighbors = track_all.CN
            
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
        I_t = t + domain.t_0
        
        n_I = self.num_timesteps_in_real
        
        Neighbor = domain.neighbors.copy()
        N_U = (Neighbor.index >= I_t[0]) & (Neighbor.index <= I_t[n_I])
        N_ID = np.unique(np.concatenate(Neighbor.iloc[N_U].to_numpy())).astype(int)
        Own_pos = path.tar[np.newaxis]
        
        Pos = np.zeros((len(N_ID), len(I_t),2))
        for j, nid in enumerate(N_ID):
            t = self.T[nid]
            pos = self.Path.iloc[nid,0]
            for dim in range(2):
                Pos[j, :, dim] = interp.interp1d(np.array(t), pos[:,dim], 
                                                 fill_value = 'extrapolate', assume_sorted = True)(I_t)
        
        D = np.sqrt(((Pos[:,:n_I] - Own_pos[:,:n_I]) ** 2).sum(-1)).min(-1)
        
        Pos = Pos[np.argsort(D)]
        if self.max_num_addable_agents is not None:
            Pos = Pos[:self.max_num_addable_agents]
        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i+1)
            path[name] = pos
            agent_types[name] = 'P'
        
        return path, agent_types
            
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'ETH (interactive pedestrians)',
                 'file': 'ETH_ped_ia',
                 'latex': r'\emph{ETH/UCY}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return False