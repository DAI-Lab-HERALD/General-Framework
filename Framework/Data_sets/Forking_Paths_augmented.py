import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
import scipy
import torch
from scipy import interpolate as interp

pd.set_option('mode.chained_assignment',None)

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class Forking_Paths_augmented(data_set_template):
    '''
    The forking paths dataset is an adaption of the ETH/UCY pedestrian dataset.
    In this case, in specific trajectories, test subjects where ask to use a 
    controler to continue the observed path of a specific agent in a scene towards
    a specific and (per subject varying goal), while other agents simply followed 
    their originally observed trjectories.
    
    However, by slightly scaling and rotating those subject generated trajectories, 
    the dataset is enlarged significantly.
    
    The data can be found at https://github.com/JunweiLiang/Multiverse#the-forking-paths-dataset
    and the following citation can bes used:
        
    Liang, J., Jiang, L., Murphy, K., Yu, T., & Hauptmann, A. (2020). The garden of 
    forking paths: Towards multi-future trajectory prediction. In Proceedings of the 
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10508-10518).
    '''
    def set_scenario(self):
        self.scenario = scenario_none()
    
    def path_data_info(self = None):
        return ['x', 'y']
        
        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'Forking_Paths_complete' + os.sep + 'FP_comp_processed.pkl')
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
            
            domain = pd.Series(np.zeros(6, object), index = ['location', 'scene', 'scene_full', 'neighbors', 'name', 'type'])
            domain.location = scene
            domain.scene = "_".join([scene, moment, tar_id])
            domain.scene_full = data_i.scenario
            domain.name = data_i.name
            track_all = track_all.set_index('t')
            domain.neighbors = track_all.CN
            domain.type = data_i.type
            
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
            
            if other_samples_bool.sum() < 2:
                continue
            
            num_T = path_init.shape[0]
            Paths_other = np.zeros((other_samples_bool.sum(), num_T, 2), np.float32)
            Paths_other_df = Path_init.iloc[other_samples_bool]
            for j in range(other_samples_bool.sum()):
                path_other = Paths_other_df.iloc[j]
                num_T_other = path_other.tar.shape[0]
                num_t = min(num_T, num_T_other)
                Paths_other[j, :num_t] = path_other.tar[:num_t]
    
            Dist = np.abs(Paths_other - path_init[np.newaxis])
            Dist = np.nanmax(Dist, axis = (0,2))
            ind_split = max(0, np.argmax(Dist > 1e-3) - 1)

            I_t = t_init[ind_split+1:]
            
            Neighbor = domain_init.neighbors.copy()
            N_U = (Neighbor.index >= I_t[0])
            N_ID = np.unique(np.concatenate(Neighbor.iloc[N_U].to_numpy())).astype(int)
            
            Pos = np.zeros((len(N_ID), len(I_t), 2))
            for j, nid in enumerate(N_ID):
                data_id = self.Data[(self.Data.index == nid) & (self.Data.scenario == domain_init.scene_full)].iloc[0]

                t = data_id.path.t.to_numpy()
                pos = np.stack([data_id.path.x.to_numpy(), data_id.path.y.to_numpy()], axis = -1)
                    
                if len(t) > 1:
                    for dim in range(2):
                        Pos[j, :, dim] = interp.interp1d(np.array(t), pos[:,dim], 
                                                        fill_value = 'extrapolate', assume_sorted = True)(I_t)
                        
                else:
                    Pos[j, :, :] = pos.repeat(len(I_t), axis = 0)[np.newaxis]
            
            # Prepare parameters for sampling
            s_min = 0.9
            s_max = 1.1
            s_std = 0.05
            
            s_min_ang = -np.pi/72
            s_max_ang = np.pi/72
            s_std_ang = np.pi/144
            
            num_samples_test = 1000
            num_samples = 100
            
            # Prepare samples trajectories
            # Traj_new = np.zeros((0, *path_init.shape), float)
            Traj_new = path_init[np.newaxis]

            P2s = Pos[np.newaxis, :, :-1]
            P2e = Pos[np.newaxis, :, 1:]

            P1s = Traj_new[:, np.newaxis, ind_split+1:-1] 
            P1e = Traj_new[:, np.newaxis, ind_split+2:] 
            
            # Get dp
            dP1 = P1e - P1s
            dP2 = P2e - P2s
            
            # Get the factors 
            A = P1s - P2s
            B = dP1 - dP2
            
            # The distance d(t) can be calculated in the form:
            # d(t) = ||A + t * B|| = a * t ^ 2 + b * t + c
            a = (B ** 2).sum(-1)
            b = 2 * (A * B).sum(-1)
            c = (A ** 2).sum(-1)
            
            # We know that a >= 0, so we can calculate t_min with:
            # d'(t_min) = 2 * a * t_min + b = 0
            t_min = - b / 2 * (a + 1e-6)
            
            # Constrict t_min to interval between 0 and 1,
            # As we only compare lines between the points
            t_min = np.clip(t_min, 0.0, 1.0)
            
            # Calculate d(t_min)
            D_min = a * t_min ** 2 + b * t_min + c
            
            # Get D_min over all other agents and timesteps
            d_min = D_min.min(axis = (1, 2))
            
            if d_min.min() < 0.5:
                d_collision = d_min.min() - 0.01
            else:
                d_collision = 0.5

            

            col_cnt = 0
            while len(Traj_new) < num_samples:
                # Create new trajectories
                Traj_test = np.tile(path_init[np.newaxis], (num_samples_test, 1, 1))
                
                # Sample random factors
                s_mean = 1.0
                Factors = scipy.stats.truncnorm.rvs((s_min - s_mean) / s_std, (s_max - s_mean) / s_std, 
                                                    loc = s_mean, scale = s_std, 
                                                    size = num_samples_test)   
    
                Angles = scipy.stats.truncnorm.rvs(s_min_ang / s_std_ang, s_max_ang / s_std_ang, 
                                                   loc = 0.0, scale = s_std_ang, 
                                                   size = num_samples_test)
                
                # Prepare for vectorized operations
                Angles = Angles[:,np.newaxis]
                Factors = Factors[:,np.newaxis,np.newaxis]
                
                # Rotate and stretch trajectories
                c, s = np.cos(Angles), np.sin(Angles)
                Traj_centered = (Traj_test[:,ind_split + 1:] - Traj_test[:,[ind_split]])
                x_vals, y_vals = Traj_centered[...,0], Traj_centered[...,1]
                new_x_vals = c * x_vals - s * y_vals # _rotate x
                new_y_vals = s * x_vals + c * y_vals # _rotate y
                Traj_centered[...,0] = new_x_vals
                Traj_centered[...,1] = new_y_vals
                
                Traj_test[:, ind_split + 1:] = Traj_centered * Factors + Traj_test[:, [ind_split]]
                
                # Get starting and end points
                P1s = Traj_test[:, np.newaxis, ind_split+1:-1] 
                P1e = Traj_test[:, np.newaxis, ind_split+2:] 
                
                # Get dp
                dP1 = P1e - P1s
                dP2 = P2e - P2s
                
                # Get the factors 
                A = P1s - P2s
                B = dP1 - dP2
                
                # The distance d(t) can be calculated in the form:
                # d(t) = ||A + t * B|| = a * t ^ 2 + b * t + c
                a = (B ** 2).sum(-1)
                b = 2 * (A * B).sum(-1)
                c = (A ** 2).sum(-1)
                
                # We know that a >= 0, so we can calculate t_min with:
                # d'(t_min) = 2 * a * t_min + b = 0
                t_min = - b / 2 * (a + 1e-6)
                
                # Constrict t_min to interval between 0 and 1,
                # As we only compare lines between the points
                t_min = np.clip(t_min, 0.0, 1.0)
                
                # Calculate d(t_min)
                D_min = a * t_min ** 2 + b * t_min + c
                
                # Get D_min over all other agents and timesteps
                d_min = D_min.min(axis = (1, 2))
                
                # Check if there are collisions
                no_collision = d_min > d_collision
                
                if ~no_collision.all():
                    col_cnt += 1
                    print(col_cnt)
                # Get collision free trajectories
                Traj_good = Traj_test[no_collision]
                
                # Add good collisions to collection
                Traj_new = np.concatenate((Traj_new, Traj_good), axis = 0)
            
            # Only get required number of samples
            Traj_new = Traj_new[:num_samples]
            
            # Save new trajectories
            for traj in Traj_new:
                path = pd.Series(np.zeros(0, np.ndarray), index = [])
                agent_types = pd.Series(np.zeros(0, str), index = [])
                
                path['tar'] = traj
                # should be done based on actual agent types
                agent_types['tar'] = domain.type
                
                domain = domain_init.copy()
                domain['t_split'] = t_init[ind_split]
                domain['ind_split'] = ind_split
                
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
        in_position = np.arange(len(path.tar)) >= domain.ind_split
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
                                   'Forking_Paths_complete' + os.sep + 'FP_comp_processed.pkl')
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
                
            if len(t) > 1:
                for dim in range(2):
                    Pos[j, :, dim] = interp.interp1d(np.array(t), pos[:,dim], 
                                                    fill_value = 'extrapolate', assume_sorted = True)(I_t)
                    
            else:
                Pos[j, :, :] = pos.repeat(len(I_t), axis = 0)[np.newaxis]
        
        D = np.sqrt(((Pos[:,:n_I] - Own_pos[:,:n_I]) ** 2).sum(-1)).min(-1)
        
        Pos = Pos[np.argsort(D)]
        if self.max_num_addable_agents is not None:
            Pos = Pos[:self.max_num_addable_agents]
        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i+1)
            path[name] = pos
            agent_types[name] = domain.type#'P'
        
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
    
    def includes_sceneGraphs(self = None):
        return False
    
    def has_repeated_inputs(self):
        return True

