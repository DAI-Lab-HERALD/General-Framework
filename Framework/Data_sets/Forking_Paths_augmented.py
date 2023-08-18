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


class Forking_Paths_augmented(data_set_template):
    def set_scenario(self):
        self.scenario = scenario_none()
        
        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'Forking_Paths' + os.sep + 'FP_processed.pkl')
        # analize raw dara 
        num_tars = len(self.Data)
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        Path_init = []
        T_init = []
        Domain_init = []
        
        # extract raw samples
        max_number_other = 0
        for i in range(num_tars):
            data_i = self.Data.iloc[i]
             
            scene, moment, tar_id, dest, subj, _ = data_i.scenario.split('_')
            if int(data_i.Id) != int(tar_id):
                continue 
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
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            
            track_all = data_i.path.copy(deep = True)
            path['tar'] = np.stack([track_all.x.to_numpy(), track_all.y.to_numpy()], axis = -1)
            
            t = track_all.t.to_numpy()
            
            domain = pd.Series(np.zeros(5, object), index = ['location', 'scene', 'scene_full', 'neighbors', 'name'])
            domain.location = scene
            domain.scene = "_".join([scene, moment, tar_id])
            domain.scene_full = data_i.scenario
            domain.name = data_i.name
            track_all = track_all.set_index('t')
            domain.neighbors = track_all.CN
            
            Path_init.append(path)
            T_init.append(t)
            Domain_init.append(domain)
        
        Path_init = pd.DataFrame(Path_init)
        T_init = np.array(T_init+[()], np.ndarray)[:-1]
        Domain_init = pd.DataFrame(Domain_init)
        
        for i in range(len(Path_init)):
            path_init   = Path_init.iloc[i].tar
            t_init      = T_init[i]
            domain_init = Domain_init.iloc[i]
        
            other_samples_bool = (Domain_init.scene == domain_init.scene).to_numpy()
            num_T = path_init.shape[0]
            Paths_other = np.zeros((other_samples_bool.sum(), num_T, 2), np.float32)
            Paths_other_df = Path_init.iloc[other_samples_bool]
            for j in range(other_samples_bool.sum()):
                path_other = Paths_other_df.iloc[j]
                num_T_other = path_other.tar.shape[0]
                num_t = min(num_T, num_T_other)
                Paths_other[j, :num_t] = path_other.tar[:num_t]
    
            Check = Paths_other == path_init[np.newaxis]
            if Check.any():
                in_position = ~Check.all(axis = (0, 2))
                ind_split = np.argmax(in_position) - 1
            else:
                ind_split = 0
            
            Factors = [1.0]
            
            for factor in Factors:
                path = pd.Series(np.zeros(0, np.ndarray), index = [])
                agent_types = pd.Series(np.zeros(0, str), index = [])
                
                traj = path_init.copy()
                
                traj[ind_split + 1:] = (traj[ind_split + 1:] - traj[[ind_split]]) * factor + traj[[ind_split]] 
                
                path['tar'] = traj
                agent_types['tar'] = 'P'
                
                domain = domain_init.copy()
                domain['t_split'] = t_init[ind_split]
                
                self.Path.append(path)
                self.Type_old.append(agent_types)
                self.T.append(t_init)
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
        other_samples_bool = (self.Domain_old.scene == domain.scene).to_numpy()
        num_T = path.tar.shape[0]
        Paths_other = np.zeros((other_samples_bool.sum(), num_T, 2), np.float32)
        for i in range(other_samples_bool.sum()):
            path_other = self.Path.iloc[other_samples_bool].iloc[i]
            num_T_other = path_other.tar.shape[0]
            num_t = min(num_T, num_T_other)
            Paths_other[i, :num_t] = path_other.tar[:num_t]

        Check = Paths_other == path.tar[np.newaxis]
        in_position = ~Check.all(axis = (0, 2))
        ind_first = np.argmax(in_position)
        if ind_first > 0:
            in_position[ind_first - 1] = True
              
        return in_position
        
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
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'Forking_Paths' + os.sep + 'FP_processed.pkl')
            self.Data = self.Data.reset_index(drop = True)

        I_t = t + domain.t_0
        
        n_I = self.num_timesteps_in_real
        
        Neighbor = domain.neighbors.copy()
        N_U = (Neighbor.index >= I_t[0]) & (Neighbor.index <= I_t[n_I])
        N_ID = np.unique(np.concatenate(Neighbor.iloc[N_U].to_numpy())).astype(int)
        Own_pos = path.tar[np.newaxis]
        
        Pos = np.zeros((len(N_ID), len(I_t), 2))
        for j, nid in enumerate(N_ID):
            data_id = self.Data[(self.Data.index == nid) & (self.Data.scenario == domain.scene_full)].iloc[0]

            t = data_id.path.t.to_numpy()
            pos = np.stack([data_id.path.x.to_numpy(), data_id.path.y.to_numpy()], axis = -1)
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
        names = {'print': 'Forking Paths (interactive pedestrians - augmented)',
                 'file': 'Fork_P_Aug',
                 'latex': r'\emph{FP}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return False
