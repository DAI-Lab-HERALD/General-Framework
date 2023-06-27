import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
import os

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class Commotions_crossing(data_set_template):
    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
        
        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'Commotions_crossing' + os.sep + 'Commotions_processed.pkl')
        # analize raw dara 
        num_tars = len(self.Data)
        self.num_samples = 0 
        self.Path = []
        self.T = []
        self.Domain_old = []
        # extract raw samples
        # If false, set every y value to zero
        Driving_left = True
        
        for i in range(num_tars):
            data_i = self.Data.iloc[i]
            # find crossing point
            tar_track_all = data_i.participant_track.copy(deep = True)
            # target vehicle is to come along x axis, towards origin
            # find crossing point
            drones = np.sort(self.Data.columns[2:])
            drones = [drone for drone in drones if type(self.Data.iloc[i][drone]) == type(self.Data)]
            
            if len(drones) > 0:
                test_track = self.Data.iloc[i][drones[0]].copy(deep = True)
                
                tar_arr  = tar_track_all.to_numpy()[np.arange(0,len(tar_track_all), 12), 1:3]
                test_arr = test_track.to_numpy()[np.arange(0,len(test_track), 12), 1:3]
                
                dist = np.sum((tar_arr[:,np.newaxis,:] - test_arr[np.newaxis,:,:]) ** 2, -1)
                _, test_ind = np.unravel_index(np.argmin(dist), dist.shape)
                
                test_dx = test_arr[test_ind] - test_arr[test_ind - 1]
                
                test_i = test_ind * 12
                
                angle = np.angle(test_dx[0] + 1j * test_dx[1])
                angle_des = np.pi / 2
                
                rot_angle = (angle_des - angle)
                
                tar_track_all  = rotate_track(tar_track_all, - rot_angle, np.zeros((1,2)))
                test_track = rotate_track(test_track, - rot_angle, np.zeros((1,2)))
                
                x_center = test_track.iloc[test_i].x + np.abs(test_track.offset.iloc[test_i])
                
                ind = test_track.index[test_i]
                
                y_center = tar_track_all.loc[ind].y + np.abs(tar_track_all.offset.loc[ind])
                
                center = np.array([[x_center, y_center]])
                
                tar_track_all = rotate_track(tar_track_all, 0, center)
                
                
                # lane_width = 3.65 m
                
                for j, drone in enumerate(drones):
                    ego_track = self.Data.iloc[i][drone].copy(deep = True)
                    ego_track = rotate_track(ego_track, -rot_angle, np.zeros((1,2)))
                    ego_track = rotate_track(ego_track, 0, center)
                    
                    tar_track = tar_track_all.loc[ego_track.index].copy(deep = True)
                    t = np.array(tar_track.t)
                    path = pd.Series(np.empty(0, np.ndarray), index = [])
                    
                    
                    if not ego_track.leaderID.iloc[0] == -1:
                        v_1_name = drones[ego_track.leaderID.iloc[0]] 
                        v_1_track = self.Data.iloc[i][v_1_name].copy(deep = True)
                        v_1_track = rotate_track(v_1_track, -rot_angle, np.zeros((1,2)))
                        v_1_track = rotate_track(v_1_track, 0, center)
                        
                        v_1_x = np.ones(len(ego_track.index)) * np.nan
                        v_1_y = np.ones(len(ego_track.index)) * np.nan
                        
                        frame_ego_min = ego_track.index[0]
                        frame_ego_max = ego_track.index[-1]
                        
                        frame_v_1_min = v_1_track.index[0]
                        frame_v_1_max = v_1_track.index[-1]
                        
                        v_1_x[max(0, frame_v_1_min - frame_ego_min) : 
                              min(len(v_1_x), 1 + frame_v_1_max - frame_ego_min)] = v_1_track.x.loc[max(frame_ego_min, frame_v_1_min):
                                                                                                    min(frame_ego_max, frame_v_1_max)]
                        v_1_y[max(0, frame_v_1_min - frame_ego_min) : 
                              min(len(v_1_x), 1 + frame_v_1_max - frame_ego_min)] = v_1_track.y.loc[max(frame_ego_min, frame_v_1_min):
                                                                                                    min(frame_ego_max, frame_v_1_max)]
                        
                        if Driving_left:                                                                            
                            path['V_v_1'] = np.stack([v_1_x, v_1_y], axis = -1)
                        else:                                  
                            path['V_v_1'] = np.stack([v_1_x, -v_1_y], axis = -1)
                        
                        
                    if not ego_track.followerID.iloc[0] == -1:
                        v_2_name = drones[ego_track.followerID.iloc[0]] 
                        v_2_track = self.Data.iloc[i][v_2_name].copy(deep = True)
                        v_2_track = rotate_track(v_2_track, -rot_angle, np.zeros((1,2)))
                        v_2_track = rotate_track(v_2_track, 0, center)
                        
                        v_2_x = np.ones(len(ego_track.index)) * np.nan
                        v_2_y = np.ones(len(ego_track.index)) * np.nan
                        
                        frame_ego_min = ego_track.index[0]
                        frame_ego_max = ego_track.index[-1]
                        
                        frame_v_2_min = v_2_track.index[0]
                        frame_v_2_max = v_2_track.index[-1]
                        
                        v_2_x[max(0, frame_v_2_min - frame_ego_min) : 
                              min(len(v_2_x), 1 + frame_v_2_max - frame_ego_min)] = v_2_track.x.loc[max(frame_ego_min, frame_v_2_min):
                                                                                                    min(frame_ego_max, frame_v_2_max)]
                        v_2_y[max(0, frame_v_2_min - frame_ego_min) : 
                              min(len(v_2_x), 1 + frame_v_2_max - frame_ego_min)] = v_2_track.y.loc[max(frame_ego_min, frame_v_2_min):
                                                                                                    min(frame_ego_max, frame_v_2_max)]
                        
                        if Driving_left:                                                                            
                            path['V_v_2'] = np.stack([v_2_x, v_2_y], axis = -1)
                        else:                                  
                            path['V_v_2'] = np.stack([v_2_x, -v_2_y], axis = -1)
                    
                    if Driving_left:                                                                            
                        path['V_ego'] = np.stack([ego_track.x, ego_track.y], axis = -1)                                   
                        path['V_tar'] = np.stack([tar_track.x, tar_track.y], axis = -1)
                        domain = pd.Series(np.array([data_i.participant, 'left']), index = ['Subj_ID', 'Driving'])
                    else:                                                                             
                        path['V_ego'] = np.stack([ego_track.x, -ego_track.y], axis = -1)                                   
                        path['V_tar'] = np.stack([tar_track.x, -tar_track.y], axis = -1)
                        domain = pd.Series(np.array([data_i.participant, 'right']), index = ['Subj_ID', 'Driving'])
                    
                    
                    self.Path.append(path)
                    self.T.append(t)
                    self.Domain_old.append(domain)
                    self.num_samples = self.num_samples + 1
        
        self.Path = pd.DataFrame(self.Path)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
    
    
    def calculate_distance(self, path, domain):
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
        tar_x = path.V_tar[...,0]
        
        if domain.Driving == 'left':
            ego_y = path.V_ego[...,1]
        else:
            ego_y = - path.V_ego[...,1]
        
        lane_width = 3.65
        vehicle_length = 5
        
        Dc = (-ego_y) - (lane_width + 1 + 0.5 * vehicle_length) # 1 m for traffic island
        Da = tar_x - 0.5 * vehicle_length
        
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
        if domain.Driving == 'left':
            ego_y = path.V_ego[...,1]
            tar_y = path.V_tar[...,1]
        else:
            ego_y = - path.V_ego[...,1]
            tar_y = - path.V_tar[...,1]
        
        lane_width = 3.65
        vehicle_length = 5
        
        if isinstance(path.V_v_1, float):
            assert str(path.V_v_1) == 'nan'
            D1 = np.ones(len(ego_y)) * 1000
        
        else:
            if domain.Driving == 'left':
                v_1_y = path.V_v_1[...,1]
            else:
                v_1_y = - path.V_v_1[...,1]
            
            D1 = v_1_y - ego_y - vehicle_length
            D1_good = np.isfinite(D1)
            if not all(D1_good):
                index = np.arange(len(D1))
                D1 = np.interp(index, index[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])
        
        Le = np.ones_like(ego_y) * (lane_width + 1)
        
        in_position = (-(lane_width + 1) < tar_y) & (tar_y < 0) & (D1 > D_class['rejected'] + Le) 
        
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
        if domain.Driving == 'left':
            ego_y = path.V_ego[...,1]
        else:
            ego_y = - path.V_ego[...,1]
        
        lane_width = 3.65
        vehicle_length = 5
        
        
        if isinstance(path.V_v_1, float):
            assert str(path.V_v_1) == 'nan'
            D1 = np.ones(len(ego_y)) * 1000
        
        else:
            if domain.Driving == 'left':
                v_1_y = path.V_v_1[...,1]
            else:
                v_1_y = - path.V_v_1[...,1]
        
            D1 = v_1_y - ego_y - vehicle_length
            D1_good = np.isfinite(D1)
            if not all(D1_good):
                index = np.arange(len(D1))
                D1 = np.interp(index, index[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])
        
        if isinstance(path.V_v_2, float):
            assert str(path.V_v_2) == 'nan'
            D2 = np.ones(len(ego_y)) * 1000
        
        else:
            if domain.Driving == 'left':
                v_2_y = path.V_v_2[...,1]
            else:
                v_2_y = - path.V_v_2[...,1]
        
            D2 = ego_y - v_2_y - vehicle_length
            D2_good = np.isfinite(D2)
            if not all(D2_good):
                index = np.arange(len(D2))
                D2 = np.interp(index, index[D2_good], D2[D2_good], left = D2[D2_good][0], right = D2[D2_good][-1])
        
        D3 = np.ones(len(ego_y)) * 1000
        
        Le = np.ones_like(ego_y) * (lane_width + 1)
        Lt = np.ones_like(ego_y) * lane_width
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
    
    
    def fill_empty_input_path(self, path, t, domain):
        # check vehicle v_1 (in front of ego)
        if isinstance(path.V_v_1, float):
            assert str(path.V_v_1) == 'nan'
        else:
            v_1_x = path.V_v_1[...,0]
            v_1_good = np.isfinite(v_1_x)
            if not all(v_1_good):
                D = path.V_v_1[...,1] - path.V_ego[...,1]
                index = np.arange(len(D))
                D = np.interp(index, index[v_1_good], D[v_1_good], left = D[v_1_good][0], right = D[v_1_good][-1])
                v_1_y = path.V_ego[...,1] + D
                v_1_x = np.interp(index, index[v_1_good], v_1_x[v_1_good], left = v_1_x[v_1_good][0], right = v_1_x[v_1_good][-1])
                
                path.V_v_1 = np.stack([v_1_x, v_1_y], axis = -1)
                
        if isinstance(path.V_v_2, float):
            assert str(path.V_v_2) == 'nan'
        else:
            v_2_x = path.V_v_2[...,0]
            v_2_good = np.isfinite(v_2_x)
            if not all(v_2_good):
                D = path.V_v_2[...,1] - path.V_ego[...,1]
                index = np.arange(len(D))
                D = np.interp(index, index[v_2_good], D[v_2_good], left = D[v_2_good][0], right = D[v_2_good][-1])
                v_2_y = path.V_ego[...,1] + D
                v_2_x = np.interp(index, index[v_2_good], v_2_x[v_2_good], left = v_2_x[v_2_good][0], right = v_2_x[v_2_good][-1])
                
                path.V_v_2 = np.stack([v_2_x, v_2_y], axis = -1)     
        return path
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        lines_solid.append(np.array([[-300, -4],[-4, -4],[-4, -300]]))
        lines_solid.append(np.array([[300, -4],[4, -4],[4, -300]]))
        lines_solid.append(np.array([[-300, 4],[-4, 4],[-4, 300]]))
        lines_solid.append(np.array([[300, 4],[4, 4],[4, 300]]))
        
        lines_dashed = []
        lines_dashed.append(np.array([[0, -300],[0, 300]]))
        lines_dashed.append(np.array([[4, -4],[4, 0],[300, 0]]))
        lines_dashed.append(np.array([[-4, 4],[-4, 0],[-300,0]]))
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'Leeds (intersection)',
                 'file': 'Leedscross',
                 'latex': r'\emph{Leeds}'}
        return names
    
    def future_input(self = None):
        return True
    
    def includes_images(self = None):
        return False