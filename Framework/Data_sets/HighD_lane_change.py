import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
import os
from PIL import Image
from scipy import interpolate as interp
from scipy.signal import savgol_filter

class HighD_lane_change(data_set_template):
    '''
    The highD dataset is extracted from drone recordings of real world traffic over
    german highways. This specific instance focuses on cases where a vehicle wants
    to switch to a faster lane, along which vehicles already driving in that lane
    have priority. Predicting if the vehicle will change lanes in front or behind such 
    vehicles can therefore be seen as a gap acceptance problem.
    
    The dataset can be found at https://www.highd-dataset.com/ and the following 
    citation can be used:
        
    Krajewski, R., Bock, J., Kloeker, L., & Eckstein, L. (2018, November). The highd 
    dataset: A drone dataset of naturalistic vehicle trajectories on german highways 
    for validation of highly automated driving systems. In 2018 21st international 
    conference on intelligent transportation systems (ITSC) (pp. 2118-2125). IEEE.
    '''
    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
        
    def _create_path_sample(self, data_i, tar_track, ego_track, pre_event_frame):
        # Find overlapping frames
        frame_min = max(tar_track.index.min(), ego_track.index.min())
        frame_max = min(tar_track.index.max(), ego_track.index.max())
        
        tar_track = tar_track.loc[frame_min:frame_max]
        ego_track = ego_track.loc[frame_min:frame_max]
        
        frames = tar_track.index
        
        # Find the right lane alignment
        right_pre = ((tar_track.laneId == tar_track.laneId.loc[pre_event_frame]) &
                     (ego_track.laneId == ego_track.laneId.loc[pre_event_frame]) & 
                     (frames <= pre_event_frame))
        
        if not right_pre.loc[:pre_event_frame].all():
            frame_min = frames[np.where(np.array(~right_pre) & 
                                        np.array(frames <= pre_event_frame))[0][-1] + 1] 
            # Last frame update
            tar_track = tar_track.loc[frame_min:frame_max]
            ego_track = ego_track.loc[frame_min:frame_max]
        
            frames = tar_track.index      
        
        # Save those paths
        path = pd.Series(np.zeros(0, np.ndarray), index = [])
        agent_types = pd.Series(np.zeros(0, str), index = [])
        
        path['tar'] = np.stack([tar_track.x.to_numpy(), tar_track.y.to_numpy()], axis = -1)
        path['ego'] = np.stack([ego_track.x.to_numpy(), ego_track.y.to_numpy()], axis = -1)
        
        agent_types['tar'] = 'V'
        agent_types['ego'] = 'V'
        
        # Get corresponding time points
        t = np.array(ego_track.index / 25)
        
        # get remaining important vehicles
        v_1_id = ego_track.precedingId.loc[pre_event_frame]
        if v_1_id != 0:
            v_1_ind = np.where(self.Data.id == v_1_id)[0][0]
            v_1_track = self.Data.iloc[v_1_ind].track[['frame', 'x', 'y', 'laneId', 'followingId']].copy(deep = True).set_index('frame')
            v_1_track = v_1_track.reindex(np.arange(frame_min, frame_max + 1))
            path['v_1'] = np.stack([v_1_track.x.to_numpy(), v_1_track.y.to_numpy()], axis = -1)
            agent_types['v_1'] = 'V'
        
        v_2_id = ego_track.followingId.loc[pre_event_frame]    
        if v_2_id != 0:
            v_2_ind = np.where(self.Data.id == v_2_id)[0][0]
            v_2_track = self.Data.iloc[v_2_ind].track[['frame', 'x', 'y']].copy(deep = True).set_index('frame')
            v_2_track = v_2_track.reindex(np.arange(frame_min, frame_max + 1))
            path['v_2'] = np.stack([v_2_track.x.to_numpy(), v_2_track.y.to_numpy()], axis = -1)
            agent_types['v_2'] = 'V'
        
        v_3_id = tar_track.precedingId.loc[pre_event_frame]
        if v_3_id != 0:
            v_3_ind = np.where(self.Data.id == v_3_id)[0][0]
            v_3_track = self.Data.iloc[v_3_ind].track[['frame', 'x', 'y']].copy(deep = True).set_index('frame')
            v_3_track = v_3_track.reindex(np.arange(frame_min, frame_max + 1))
            path['v_3'] = np.stack([v_3_track.x.to_numpy(), v_3_track.y.to_numpy()], axis = -1)
            agent_types['v_3'] = 'V'
        
        domain = pd.Series(np.zeros(4, object), index = ['location', 'image_id', 'drivingDirection', 'laneMarkings'])
        domain.location         = data_i.locationId
        domain.image_id         = data_i.recordingId
        domain.drivingDirection = data_i.drivingDirection
        domain.laneMarkings     = data_i.laneMarkings
        
        self.Path.append(path)
        self.Type_old.append(agent_types)
        self.T.append(t)
        self.Domain_old.append(domain)
        self.num_samples = self.num_samples + 1   
        
        
    def create_path_samples(self):     
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'HighD_highways' + os.sep + 'HighD_processed.pkl')
        
        self.Data = self.Data.reset_index(drop = True)
        # analize raw dara 
        num_samples_max = len(self.Data)
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        
        # ego is the vehicle offering the gap
        # v_1 is the vehicle in front of the ego vehicle
        # v_2 is the vehicle following the ego vehicle
        # v_3 is the vehicle the target vehicle is following before the lane change
        # other vehicle start as v_4
        
        # Create Images
        self.Images = pd.DataFrame(np.zeros((0, 1), np.ndarray), columns = ['Image'])
        self.Target_MeterPerPx = 0.5
        
        max_width = 0
        max_height = 0
        data_path = self.path + os.sep + 'Data_sets' + os.sep + 'HighD_highways' + os.sep + 'data' + os.sep
        potential_image_files = os.listdir(data_path)
        
        # Treat recording as location due to slight misalignments
        image_files = [file for file in potential_image_files if file[-4:] == '.png']
        for img_file in image_files:
            img_str = img_file.split('_')[0]
            img_id  = int(img_str)
            
            img_path = data_path + img_file
            img = Image.open(img_path)
            
            # This is an approximation
            MeterPerPx = 420 / img.width
            
            img_scaleing = MeterPerPx / self.Target_MeterPerPx
            
            height_new = int(img.height * img_scaleing)
            width_new  = int(img.width * img_scaleing)
            
            img_new = img.resize((width_new, height_new), Image.LANCZOS)
            
            self.Images.loc[img_id] = list(np.array(img_new)[np.newaxis])
            max_width = max(width_new, max_width)
            max_height = max(height_new, max_height)
            
        # pad images to max size
        for loc_id in self.Images.index:
            img = self.Images.loc[loc_id].Image
            img_pad = np.pad(img, ((0, max_height - img.shape[0]),
                                   (0, max_width  - img.shape[1]),
                                   (0,0)), 'constant', constant_values=0)
            self.Images.loc[loc_id].Image = img_pad           
        
        # extract raw samples
        self.num_samples = 0
        for i in range(num_samples_max):
            # to keep track:
            if np.mod(i, 1000) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
            
            data_i = self.Data.iloc[i]
            track_i = data_i.track[['frame','x', 'xVelocity', 
                                    'y', 'laneId',
                                    'followingId',
                                    'precedingId',
                                    'leftFollowingId',
                                    'leftAlongsideId',
                                    'leftPrecedingId']].copy(deep = True)

            # Check for rejected gaps
            overtaken = np.where((np.array(track_i.leftFollowingId.iloc[:-1]) == np.array(track_i.leftAlongsideId.iloc[1:])) &
                                 (np.array(track_i.leftFollowingId.iloc[:-1]) != 0))[0]
            
            lane_array = np.array(track_i.laneId)
            
            for pre_event_ind in overtaken:
                pre_event_frame = track_i.frame.iloc[pre_event_ind]
                tar_track = track_i.set_index('frame')
                
                # Get ego vehicle
                ego_id = tar_track.leftFollowingId.loc[pre_event_frame]
                ego_ind = np.where(self.Data.id == ego_id)[0][0]
                ego_track = self.Data.iloc[ego_ind].track[['frame', 'x', 'y', 'xVelocity', 'laneId', 
                                                           'followingId', 'precedingId']].copy(deep = True).set_index('frame') 
                
                has_intention = False
                # check if tar has lane change intention (i.e., we observe a later lane change)
                fut_lane_array = lane_array[pre_event_ind:]
                later_lane_change = np.where(fut_lane_array == fut_lane_array[0] + 1)[0]
                if len(later_lane_change) > 0:
                    if (fut_lane_array[:later_lane_change[0] - 1] == fut_lane_array[0]).all():
                        has_intention = True
                
                # check if the preceeding vehicle is significantly slower
                if not has_intention:
                    v_1_id = ego_track.precedingId.loc[pre_event_frame]
                    if v_1_id == 0:
                        frame_start = max(tar_track.index.min(), ego_track.index.min())
                    else:
                        start_inds = np.where((tar_track.leftAlongsideId.loc[:pre_event_frame] == v_1_id) & 
                                              (tar_track.laneId[:pre_event_frame] == lane_array[pre_event_ind]))[0]
                        if len(start_inds) > 0:
                            frame_start = max(tar_track.index[start_inds[-1]] + 1, ego_track.index.min())
                        else:
                            frame_start = max(tar_track.index.min(), ego_track.index.min())
                        
                    v_3_id = tar_track.precedingId.loc[pre_event_frame]
                    if v_3_id != 0:
                        v_3_ind = np.where(self.Data.id == v_3_id)[0][0]
                        v_3_track = self.Data.iloc[v_3_ind].track[['frame', 'x', 'xVelocity', 'y']].copy(deep = True).set_index('frame')
                        
                        vehicle_lengths_3 = 0.5 * (self.Data.iloc[v_3_ind].width + data_i.width)
                        frame_start = max(frame_start, v_3_track.index.min())
                        Dx3 = (v_3_track.x - tar_track.x).loc[frame_start]
                        Dv3 = (tar_track.xVelocity - v_3_track.xVelocity).loc[frame_start] * np.sign(Dx3)
                        Dx3 = np.abs(Dx3) - vehicle_lengths_3
                        
                        DT3 = Dx3 / max(1e-6, Dv3)
                        
                        assert np.isfinite(DT3)
                        
                        vehicle_lengths_e = 0.5 * (self.Data.iloc[ego_ind].width + data_i.width)
                        
                        Dxe = (tar_track.x - ego_track.x).loc[frame_start]
                        Dve = (ego_track.xVelocity - tar_track.xVelocity).loc[frame_start] * np.sign(Dxe)
                        Dxe = np.abs(Dxe) - vehicle_lengths_e
                        
                        DTe = Dxe / max(1e-6, Dve)
                        
                        assert np.isfinite(DTe) 
                        
                        if DT3 < min(10, 0.5 * DTe):
                            has_intention = True
                
                if not has_intention:
                    continue
                    
                self._create_path_sample(data_i, tar_track, ego_track, pre_event_frame)   
                
                
            # Check for accepted gaps
            interesting_lane_change = np.where(lane_array[1:] == lane_array[:-1] + 1)[0]
            
            for pre_event_ind in interesting_lane_change:
                # Find ego vehicle offering the gap (left)
                ego_id = track_i.leftFollowingId.iloc[pre_event_ind]
                if ego_id == 0:
                    # No ego vehicle is there
                    continue
                
                if not ego_id == track_i.followingId.iloc[pre_event_ind + 1]:
                    continue
                
                ego_ind = np.where(self.Data.id == ego_id)[0][0]
                ego_track = self.Data.iloc[ego_ind].track[['frame', 'x', 'y', 'laneId', 
                                                           'followingId', 'precedingId']].copy(deep = True).set_index('frame')
                
                pre_event_frame = track_i.frame.iloc[pre_event_ind]
                tar_track = track_i.set_index('frame')
                
                # check that the gap was not to big
                # assume that this is the distance at which vehicles notice each other
                if np.abs(tar_track.loc[pre_event_frame].x - ego_track.loc[pre_event_frame].x) > 200:
                    continue
                
                # Find overlapping frames
                self._create_path_sample(data_i, tar_track, ego_track, pre_event_frame)
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], tuple)[:-1]
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
        vehicle_length = 5 
        
        ego_x = path.ego[...,0]
        tar_x = path.tar[...,0]
        tar_y = path.tar[...,1]
        
        drivingDirection = domain.drivingDirection
        laneMarkings     = domain.laneMarkings
        
        
        if drivingDirection == 1: # going to the right
            ego_x        = - ego_x
            tar_x        = - tar_x
            tar_y        = - tar_y
            laneMarkings = - laneMarkings
            
        Dc = tar_x - ego_x - vehicle_length
        
        lane_mark = laneMarkings[np.argmax(laneMarkings[np.newaxis] > tar_y[:,[0]], axis = 1)]
        
        # 0.75m are the assumed minimal vehicle 
        DA = lane_mark[:,np.newaxis] - tar_y - 0.5
        angle = np.angle(tar_x[:,1:] - tar_x[:,:-1] + 1j * (tar_y[:,1:] - tar_y[:,:-1]))
        angle = np.concatenate((angle[:,[0]], angle), axis = 1)
        angle = np.clip(angle, 1e-3, 0.5 * np.pi)
        if angle.max() > 1e-3:
            angle = np.clip(angle, max(1e-3, 0.99 * np.unique(angle)[1]), None)
        
        mean_dt = np.mean(t[1:] - t[:-1])
        n_dt = max(3, int(self.dt / mean_dt))
        
        for i in range(len(angle)):
            angle[i] = savgol_filter(angle[i], n_dt, 1)
        
        Da = DA / np.sin(angle)
        for i in range(len(angle)):
            Da[i] = savgol_filter(Da[i], n_dt, 1)
        
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
        
        tar_x = path.tar[...,0]
        
        drivingDirection = domain.drivingDirection
        
        if drivingDirection == 1: 
            tar_x = - tar_x
            
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
            in_position = np.ones(tar_x.shape, bool)
        else:
            v_1_x = path.v_1[...,0] 
            if drivingDirection == 1:
                v_1_x = - v_1_x
                
            in_position = (v_1_x >= tar_x + 5)
        
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
        vehicle_length = 5 
        
        ego_x = path.ego[...,0]
        tar_x = path.tar[...,0]
        
        drivingDirection = domain.drivingDirection
        
        if drivingDirection == 1: # going to the right
            ego_x = - ego_x
            tar_x = - tar_x
        
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
            D1 = np.ones(tar_x.shape, float) * 1000
        else:
            v_1_x = path.v_1[...,0]  
            if drivingDirection == 1:
                v_1_x = - v_1_x
                
            D1 = v_1_x - ego_x - vehicle_length
            
            D1_good = np.isfinite(D1)
            if not all(D1_good):
                D1 = np.interp(t, t[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])    
        
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
            D2 = np.ones(tar_x.shape, float) * 1000
        else:
            v_2_x = path.v_2[...,0]  
            if drivingDirection == 2:
                v_2_x = - v_2_x
                
            D2 = ego_x - v_2_x - vehicle_length
            
            D2_good = np.isfinite(D2)
            if not all(D2_good):
                D2 = np.interp(t, t[D2_good], D2[D2_good], left = D2[D2_good][0], right = D2[D2_good][-1])    
        
        if isinstance(path.v_3, float):
            assert str(path.v_3) == 'nan'
            D3 = np.ones(tar_x.shape, float) * 1000
        else:
            v_3_x = path.v_3[...,0]  
            if drivingDirection == 3:
                v_3_x = - v_3_x
                
            D3 = v_3_x - tar_x - vehicle_length
            
            D3_good = np.isfinite(D3)
            if not all(D3_good):
                D3 = np.interp(t, t[D3_good], D3[D3_good], left = D3[D3_good][0], right = D3[D3_good][-1]) 
        
        mean_dt = np.mean(t[1:] - t[:-1])
        n_dt = max(3, int(self.dt / mean_dt))
        
        Dx = tar_x - ego_x - vehicle_length
        Dv = np.interp(t, t[n_dt:], (Dx[n_dt:] - Dx[:-n_dt]) / (t[n_dt:] - t[:-n_dt]))
        Dv = np.maximum(-Dv, 1e-6)
        acc = 2 # assume acceleration of up to 2m/s^2
        Te = Dv / acc 
        Lt = 0.5 * acc * Te ** 2
        Lt = np.clip(Lt, vehicle_length, None)
        
        Le = np.ones_like(ego_x) * vehicle_length
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
    
    
    def _fill_round_about_path(self, pos, t, domain):
        v_x = pos[:,0]
        v_y = pos[:,1]
        
        v_rewrite = np.isnan(v_x)
        if v_rewrite.any():
            useful = np.invert(v_rewrite)
            if useful.sum() > 2:
                v_x = interp.interp1d(t[useful], v_x[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
                v_y = interp.interp1d(t[useful], v_y[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
            else:   
                v_x = np.interp(t, t[useful], v_x[useful], left = v_x[useful][0], right = v_x[useful][-1])
                v_y = np.interp(t, t[useful], v_y[useful], left = v_y[useful][0], right = v_y[useful][-1])
                
            assert not np.isnan(v_x).any()
        return np.stack([v_x, v_y], axis = -1)
    
    
    def fill_empty_path(self, path, t, domain, agent_types):
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
        else:
            path.v_1 = self._fill_round_about_path(path.v_1, t, domain)
            
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
        else:
            path.v_2 = self._fill_round_about_path(path.v_2, t, domain)
            
        if isinstance(path.v_3, float):
            assert str(path.v_3) == 'nan'
        else:
            path.v_3 = self._fill_round_about_path(path.v_3, t, domain)
        
        
        n_I = self.num_timesteps_in_real

        tar_pos = path.tar[np.newaxis]
        
        help_pos = []
        for agent in path.index:
            if isinstance(path[agent], float):
                assert str(path[agent]) == 'nan'
                continue
            help_pos.append(path[agent])
            
        help_pos = np.stack(help_pos, axis = 0)
        
        tar_frames = 25 * (t + domain.t_0)
        
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'HighD_highways' + os.sep + 'HighD_processed.pkl')
            self.Data = self.Data.reset_index(drop = True) 
            
        frames_help = np.concatenate([[tar_frames[0] - 1], tar_frames])
        # search for vehicles
        Neighbor = np.where((self.Data.frame_min < tar_frames[-1]) &
                            (self.Data.frame_max > tar_frames[0]) & 
                            (self.Data.drivingDirection == domain.drivingDirection))[0]
        
        Pos = np.ones((len(Neighbor), len(tar_frames) + 1,2)) * np.nan
        for i, n in enumerate(Neighbor):
            track_n = self.Data.iloc[n].track
            Pos[i,:,0] = np.interp(frames_help, np.array(track_n.frame), track_n.x, left = np.nan, right = np.nan)
            Pos[i,:,1] = np.interp(frames_help, np.array(track_n.frame), track_n.y, left = np.nan, right = np.nan)
        
        Pos = Pos[np.isfinite(Pos[:,1:n_I + 1]).any((1,2))]
        D_help = np.nanmin(np.sqrt(((Pos[np.newaxis, :,1:n_I + 1] - help_pos[:,np.newaxis,:n_I]) ** 2).sum(-1)), -1).min(0)
        
        
        
        Pos = Pos[(D_help > 0.5) & (D_help < 100)]
        
        D = np.nanmin(((Pos[:,1:n_I + 1] - tar_pos[:,:n_I]) ** 2).sum(-1), -1)
        Pos = Pos[np.argsort(D)]
        
        if self.max_num_addable_agents is not None:
            Pos = Pos[:self.max_num_addable_agents]
            
        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i + 4)
            u = np.isfinite(pos[:,0])
            if u.sum() > 1:
                if u.all():
                    path[name] = pos
                else:
                    frames = frames_help[u]
                    p = pos[u].T
                    path[name] = np.stack([interp.interp1d(frames, p[0], fill_value = 'extrapolate', assume_sorted = True)(tar_frames),
                                           interp.interp1d(frames, p[1], fill_value = 'extrapolate', assume_sorted = True)(tar_frames)], axis = -1)
                    
                agent_types[name] = 'V'    
                    
        return path, agent_types 
    
    
    def provide_map_drawing(self, domain):
        # TODO: overwrite if actual lane markings are available
        lines_solid = []
        laneMarkings = domain.laneMarkings 
        
        lines_solid.append(np.array([[-10, laneMarkings[0]],[460, laneMarkings[0]]]))
        lines_solid.append(np.array([[-10, laneMarkings[1]],[460, laneMarkings[1]]]))
        
        lines_dashed = []
        for lane_mark in laneMarkings[1:-1]:
            lines_dashed.append(np.array([[-10, lane_mark],[460, lane_mark]]))
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'HighD (Lane Change)',
                 'file': 'HighD_LCGA',
                 'latex': r'\emph{HighD (gap)}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return True