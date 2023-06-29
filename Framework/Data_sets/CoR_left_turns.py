import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
import os

class CoR_left_turns(data_set_template):
    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'CoR_left_turns' + os.sep + 'CoR_processed.pkl')
        # analize raw dara 
        self.num_samples = len(self.Data)
        self.Path = []
        self.T = []
        self.Domain_old = []
        agents = ['V_ego', 'V_tar']
        # extract raw samples
        for i in range(self.num_samples):
            path = pd.Series(np.empty(len(agents), np.ndarray), index = agents)
            
            t_index = self.Data.bot_track.iloc[i].index
            t = np.array(self.Data.bot_track.iloc[i].t[t_index])
            V_ego = np.stack([self.Data.bot_track.iloc[i].x[t_index], self.Data.bot_track.iloc[i].y[t_index]], -1)
            V_tar = np.stack([self.Data.ego_track.iloc[i].x[t_index], self.Data.ego_track.iloc[i].y[t_index]], -1)
            
            path.V_ego = V_ego
            path.V_tar = V_tar
            
            domain = pd.Series(np.ones(1, int) * self.Data.subj_id.iloc[i], index = ['Subj_ID'])
            
            self.Path.append(path)
            self.T.append(t)
            self.Domain_old.append(domain)
        
        self.Path = pd.DataFrame(self.Path)
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
        
        ego_x = path.V_ego[...,0]
        tar_x = path.V_tar[...,0]
        tar_y = path.V_tar[...,1]
        
        lane_width = 3.5
        vehicle_length = 5
        
        # Get Dc and Le
        Dc = (-ego_x) - (lane_width + 0.5 * vehicle_length)
        
        
        # get Da and Lt
        X = np.concatenate((tar_x[:,:,np.newaxis], tar_y[:,:,np.newaxis]), -1)
        
        X0 = X[:,:-5]
        X1 = X[:,5:]
        
        Xm = 0.5 * (X0 + X1)
        DX = (X1 - X0)
        DX[:, :, 0] = np.sign(DX[:, :, 0]) * np.maximum(np.abs(DX[:, :, 0]), 0.01)
        DX = DX / (np.linalg.norm(DX, axis = -1, keepdims = True) + 1e-6)
        DX[np.linalg.norm(DX, axis = -1) < 0.1, 0] = -1
        
        N = Xm[:,:,1] / (DX[:,:,1] + 1e-5 + 3 * 1e-5 * np.sign(DX[:,:,1]))
        Dx = N * DX[:,:,0] 
        Dx = np.concatenate(((np.ones((1,5)) * Dx[:,[0]]), Dx), axis = -1)
        
        ind_n, ind_t = np.where((Dx < 0) & (tar_y > 0)) 
        Dx[ind_n, ind_t] = np.max(Dx, axis = -1)[ind_n]
        
        Dx[tar_y > 0] = np.minimum(Dx[tar_y > 0], (tar_x[tar_y > 0] + 0.5 * lane_width))
        Da = np.sign(tar_y) * np.sqrt(Dx ** 2 + tar_y ** 2)
        
        Dist = pd.Series([Da, Dc], index = ['accepted', 'rejected'])
        return Dist
    
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
        
        tar_x = path.V_tar[...,0]
        
        in_position = tar_x > - 3
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
        ego_x = path.V_ego[...,0]
        
        lane_width = 3.5
        
        Le = np.ones_like(ego_x) * 2 * lane_width
        
        Lt = np.ones_like(ego_x) * lane_width
        
        D1 = np.ones_like(ego_x) * 500
        
        D2 = np.ones_like(ego_x) * 500
        
        D3 = np.ones_like(ego_x) * 500
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
        
    
    def fill_empty_input_path(self, path, t, domain):
        return path
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_solid.append(np.array([[-300, -4],[-4, -4],[-4, -300]]))
        lines_solid.append(np.array([[300, -4],[4, -4],[4, -300]]))
        lines_solid.append(np.array([[-300, 4],[-4, 4],[-4, 300]]))
        lines_solid.append(np.array([[300, 4],[4, 4],[4, 300]]))
        
        lines_solid.append(np.array([[-4, -4],[-4, 0],[-6, 0]]))
        lines_solid.append(np.array([[4, 4],[4, 0],[6, 0]]))
        lines_solid.append(np.array([[4, -4],[0, -4],[0, -6]]))
        lines_solid.append(np.array([[-4, 4],[0, 4],[0, 6]]))
        
        lines_dashed = []
        lines_dashed.append(np.array([[0, 6],[0, 300]]))
        lines_dashed.append(np.array([[0, -6],[0, -300]]))
        lines_dashed.append(np.array([[6, 0],[300, 0]]))
        lines_dashed.append(np.array([[-6, 0],[-300, 0]]))
        
        
        return lines_solid, lines_dashed
    
    
    def get_name(self = None):
        names = {'print': 'L-GAP (left turns)',
                 'file': 'Lgap_lturn',
                 'latex': r'\emph{L-GAP}'}
        return names
    
    def future_input(self = None):
            return True
    
    def includes_images(self = None):
        return False

