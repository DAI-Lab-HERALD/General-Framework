import numpy as np
import pandas as pd
import torch
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
    
    def path_data_info(self = None):
        return ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
        
    def _create_path_sample(self, data_i, tar_track, ego_track, tar_id, ego_id, pre_event_frame):
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
        sizes = pd.Series(np.zeros(0, np.ndarray), index = [])
        
        path['tar'] = tar_track.to_numpy()[...,:6]
        path['ego'] = ego_track.to_numpy()[...,:6]
        
        agent_types['tar'] = 'V'
        agent_types['ego'] = 'V'

        tar_ind = np.where(self.Data.id == tar_id)[0][0]
        ego_ind = np.where(self.Data.id == ego_id)[0][0]

        sizes['tar'] = np.array([self.Data.iloc[tar_ind].width, self.Data.iloc[tar_ind].height])
        sizes['ego'] = np.array([self.Data.iloc[ego_ind].width, self.Data.iloc[ego_ind].height])
        
        # Get corresponding time points
        t = np.array(ego_track.index / 25)
        
        # get remaining important vehicles
        v_1_id = ego_track.precedingId.loc[pre_event_frame]
        if v_1_id != 0:
            v_1_ind = np.where(self.Data.id == v_1_id)[0][0]
            v_1_track = self.Data.iloc[v_1_ind].track[['frame', 'x', 'y', 'xVelocity', 'yVelocity', 
                                                       'xAcceleration', 'yAcceleration']].copy(deep = True).set_index('frame')
            v_1_track = v_1_track.reindex(np.arange(frame_min, frame_max + 1))
            path['v_1'] = v_1_track.to_numpy()
            agent_types['v_1'] = 'V'
            sizes['v_1'] = np.array([self.Data.iloc[v_1_ind].width, self.Data.iloc[v_1_ind].height])
        
        v_2_id = ego_track.followingId.loc[pre_event_frame]    
        if v_2_id != 0:
            v_2_ind = np.where(self.Data.id == v_2_id)[0][0]
            v_2_track = self.Data.iloc[v_2_ind].track[['frame', 'x', 'y', 'xVelocity', 'yVelocity', 
                                                       'xAcceleration', 'yAcceleration']].copy(deep = True).set_index('frame')
            v_2_track = v_2_track.reindex(np.arange(frame_min, frame_max + 1))
            path['v_2'] = v_2_track.to_numpy()
            agent_types['v_2'] = 'V'
            sizes['v_2'] = np.array([self.Data.iloc[v_2_ind].width, self.Data.iloc[v_2_ind].height])
        
        v_3_id = tar_track.precedingId.loc[pre_event_frame]
        if v_3_id != 0:
            v_3_ind = np.where(self.Data.id == v_3_id)[0][0]
            v_3_track = self.Data.iloc[v_3_ind].track[['frame', 'x', 'y', 'xVelocity', 'yVelocity', 
                                                       'xAcceleration', 'yAcceleration']].copy(deep = True).set_index('frame')
            v_3_track = v_3_track.reindex(np.arange(frame_min, frame_max + 1))
            path['v_3'] = v_3_track.to_numpy()
            agent_types['v_3'] = 'V'
            sizes['v_3'] = np.array([self.Data.iloc[v_3_ind].width, self.Data.iloc[v_3_ind].height])
        
        domain = pd.Series(np.zeros(5, object), index = ['location', 'image_id', 'graph_id', 'drivingDirection', 'laneMarkings'])
        domain.location         = data_i.locationId
        domain.image_id         = data_i.recordingId
        domain.graph_id         = data_i.recordingId
        domain.drivingDirection = data_i.drivingDirection
        domain.laneMarkings     = data_i.laneMarkings
        
        self.Path.append(path)
        self.Type_old.append(agent_types)
        self.Size_old.append(sizes)
        self.T.append(t)
        self.Domain_old.append(domain)
        self.num_samples = self.num_samples + 1   
        
     
    def get_sceneGraph(self, recording_id):
        Data_record = self.Data[self.Data.recordingId == recording_id].copy()
        
        # get the combined lane markings
        lane_markings = []
        lane_markings_token = []
        for j in range(len(Data_record)):
            lane_marking = Data_record.iloc[j].laneMarkings
            lane_token = np.mean(lane_marking)

            lane_markings.append(lane_marking)
            lane_markings_token.append(lane_token)
        
        # Get unique lane_tokens
        unique_tokens, unique_index = np.unique(lane_markings_token, return_index = True)
        lane_markings = [lane_markings[i] for i in unique_index]

        # Check for each unique token if the outer lane is a merging lane
        has_merge_lane = []
        for _, lane_token in enumerate(unique_tokens):
            use_id = np.where(lane_markings_token == lane_token)[0]
            has_merge_lane.append(Data_record.iloc[use_id].numMerges.sum() > 0)


        # prepare the sceneGraph
        num_nodes = 0
        lane_idcs = []
        pre_pairs = np.zeros((0,2), int)
        suc_pairs = np.zeros((0,2), int)
        left_pairs = np.zeros((0,2), int)
        right_pairs = np.zeros((0,2), int)

        left_boundaries = []
        right_boundaries = []
        centerlines = []

        lane_type = []

        segment_id = 0
        # Go throught the different lane markers
        for k, lane_marking in enumerate(lane_markings):
            has_merge = has_merge_lane[k]
            # Lanes go from right to left (from drivers perspective)
            diff_sign = np.sign(np.diff(lane_marking))

            # diff sign shoul dbe identical
            assert len(np.unique(diff_sign)) == 1
            diff_sign = diff_sign[0]

            # If diff_sign is positive, the lane goes from left to right
            if diff_sign > 0:
                x_start = -10
                x_end = 460
            else:
                x_start = 460
                x_end = -10
            
            # get number of splits needed for lane segments a little under 50 m long
            num_splits = int(np.ceil((np.abs(x_end - x_start) / 50)))
            len_splits = (x_end - x_start) / num_splits
            num_points = 1 + int(np.ceil(np.abs(len_splits) / 1.1))

            num_lanes = len(lane_marking) - 1

            right_y  = lane_marking[:-1]
            left_y   = lane_marking[1:]
            center_y = 0.5 * (right_y + left_y)

            # get lane_segments
            for i in range(num_splits):
                x_start_i = x_start + i * len_splits
                x_end_i = x_start + (i + 1) * len_splits

                y_base = np.ones((num_lanes, num_points))
                x_base = np.tile(np.linspace(x_start_i, x_end_i, num_points)[np.newaxis], (num_lanes, 1))

                left_pts   = np.stack([x_base, y_base * left_y[:,np.newaxis]], axis = -1)
                center_pts = np.stack([x_base, y_base * center_y[:,np.newaxis]], axis = -1)
                right_pts  = np.stack([x_base, y_base * right_y[:,np.newaxis]], axis = -1)

                # Add the lane segments
                for j in range(num_lanes):
                    centerlines.append(center_pts[j])
                    left_boundaries.append(left_pts[j])
                    right_boundaries.append(right_pts[j])

                    lane_type.append(('VEHICLE', False))

                    # Append lane_idc
                    lane_idcs += [segment_id] * (len(center_pts[j]) - 1)
                    num_nodes += len(center_pts[j]) - 1

                    # Get the connections:
                    # left (j, j+1)
                    # right (j, j-1)
                    # suc (i, i+1)
                    # pre (i, i-1)
                    if j < num_lanes - 1:
                        left_pairs = np.vstack([left_pairs, [segment_id, segment_id + 1]])
                    if j > 0:
                        # Exclude merge lanes, where vehicles should not drive into
                        if not (has_merge and j == 1):
                            right_pairs = np.vstack([right_pairs, [segment_id, segment_id - 1]])
                    
                    if i < num_splits - 1:
                        suc_pairs = np.vstack([suc_pairs, [segment_id, segment_id + num_lanes]])
                    if i > 0:
                        pre_pairs = np.vstack([pre_pairs, [segment_id, segment_id - num_lanes]])



                    segment_id += 1

        graph = pd.Series([], dtype = object)
        graph['num_nodes'] = num_nodes
        graph['lane_idcs'] = np.array(lane_idcs)
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        graph['left_boundaries']  = left_boundaries 
        graph['right_boundaries'] = right_boundaries 
        graph['centerlines'] = centerlines
        graph['lane_type'] = lane_type


        # Get available gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph = self.add_node_connections(graph, device = device)

        return graph
    

    def create_path_samples(self):     
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'HighD_highways' + os.sep + 'highD_processed.pkl')
        
        self.Data = self.Data.reset_index(drop = True)
        # analize raw dara 
        num_samples_max = len(self.Data)
        self.Path = []
        self.Type_old = []
        self.Size_old = []
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

        # set up the scenegraph
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                              'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), columns = sceneGraph_columns)
        
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

            # Add the scenegraph
            graph = self.get_sceneGraph(img_id)
            self.SceneGraphs.loc[img_id] = graph


        
        # extract raw samples
        self.num_samples = 0
        for i in range(num_samples_max):
            # to keep track:
            if np.mod(i, 1000) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
            
            data_i = self.Data.iloc[i]
            track_i = data_i.track[['frame', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 
                                    'laneId', 'followingId', 'precedingId',
                                    'leftFollowingId', 'leftAlongsideId', 'leftPrecedingId']].copy(deep = True)

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
                ego_track = self.Data.iloc[ego_ind].track[['frame', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 
                                                           'laneId', 'followingId', 'precedingId']].copy(deep = True).set_index('frame') 
                
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
                    
                self._create_path_sample(data_i, tar_track, ego_track, data_i.id, ego_id, pre_event_frame)   
                
                
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
                ego_track = self.Data.iloc[ego_ind].track[['frame', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration', 
                                                           'laneId', 'followingId', 'precedingId']].copy(deep = True).set_index('frame')
                
                pre_event_frame = track_i.frame.iloc[pre_event_ind]
                tar_track = track_i.set_index('frame')
                
                # check that the gap was not to big
                # assume that this is the distance at which vehicles notice each other
                if np.abs(tar_track.loc[pre_event_frame].x - ego_track.loc[pre_event_frame].x) > 200:
                    continue
                
                # Find overlapping frames
                self._create_path_sample(data_i, tar_track, ego_track, data_i.id, ego_id, pre_event_frame)
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.Size_old = pd.DataFrame(self.Size_old)
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
            A pandas series of :math:`(N_{agents})` dimensions,
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
        
        lane_mark = laneMarkings[np.argmax(laneMarkings[np.newaxis] > tar_y[...,[0]], axis = -1)]
        
        # 0.75m are the assumed minimal vehicle 
        DA = lane_mark[...,np.newaxis] - tar_y - 0.5
        angle = np.angle(tar_x[...,1:] - tar_x[...,:-1] + 1j * (tar_y[...,1:] - tar_y[...,:-1]))
        angle = np.concatenate((angle[...,[0]], angle), axis = -1)
        angle = np.clip(angle, 1e-3, 0.5 * np.pi)
        if angle.max() > np.float32(1e-3):
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
                if D1_good.any():
                    D1 = np.interp(t, t[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])
                else:    
                    D1 = np.ones(tar_x.shape, float) * 1000
        
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
                if D2_good.any():
                    D2 = np.interp(t, t[D2_good], D2[D2_good], left = D2[D2_good][0], right = D2[D2_good][-1])
                else:
                    D2 = np.ones(tar_x.shape, float) * 1000

        
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
                if D3_good.any():
                    D3 = np.interp(t, t[D3_good], D3[D3_good], left = D3[D3_good][0], right = D3[D3_good][-1]) 
                else:
                    D3 = np.ones(tar_x.shape, float) * 1000

        
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
    
    
    def fill_empty_path(self, path, t, domain, agent_types, size):
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
        else:
            path.v_1 = self.extrapolate_path(path.v_1, t, mode = 'vel')
            
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
        else:
            path.v_2 = self.extrapolate_path(path.v_2, t, mode = 'vel')
            
        if isinstance(path.v_3, float):
            assert str(path.v_3) == 'nan'
        else:
            path.v_3 = self.extrapolate_path(path.v_3, t, mode = 'vel')
        
        
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
                                       'HighD_highways' + os.sep + 'highD_processed.pkl')
            self.Data = self.Data.reset_index(drop = True) 
            
        frames_help = np.concatenate([[tar_frames[0] - 1], tar_frames]).astype(int)
        # search for vehicles
        Neighbor = np.where((self.Data.frame_min < tar_frames[-1]) &
                            (self.Data.frame_max > tar_frames[0]) & 
                            (self.Data.drivingDirection == domain.drivingDirection))[0]
        
        num_data = len(self.path_data_info())
        Pos = np.ones((len(Neighbor), len(tar_frames) + 1,num_data)) * np.nan
        Sizes = np.ones((len(Neighbor), 2)) * np.nan
        for i, n in enumerate(Neighbor):
            track_n = self.Data.iloc[n].track.set_index('frame')
            track_n.index = track_n.index.astype(int)
            Pos[i] = track_n.reindex(frames_help)[['x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']].to_numpy()
            Sizes[i,0] = self.Data.iloc[n].width
            Sizes[i,1] = self.Data.iloc[n].height
        
        # Past available 
        Past_available = np.isfinite(Pos[:,1:n_I + 1]).any((1,2))
        Pos = Pos[Past_available]
        Sizes = Sizes[Past_available]

        # Distance to the existing agents
        D_help = np.nanmin(np.sqrt(((Pos[np.newaxis, :,1:n_I + 1,...,:2] - help_pos[:,np.newaxis,:n_I,...,:2]) ** 2).sum(-1)), -1).min(0)
        Close_enough = (D_help > 0.5) & (D_help < 100)
        Pos = Pos[Close_enough]
        Sizes = Sizes[Close_enough]
        
        # Distance to target agent
        D = np.nanmin(((Pos[:,1:n_I + 1,...,:2] - tar_pos[:,:n_I,...,:2]) ** 2).sum(-1), -1)
        D_argsort = np.argsort(D)
        Pos = Pos[D_argsort]
        Sizes = Sizes[D_argsort]
        
        # Cut of furthest agents
        if self.max_num_addable_agents is not None:
            Pos = Pos[:self.max_num_addable_agents]
            Sizes = Sizes[:self.max_num_addable_agents]

        # Remove extra timestep
        Pos = Pos[:,1:]
        
        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i + 4)
            u = np.isfinite(pos[:,0])
            if u.sum() > 1:
                path[name] = self.extrapolate_path(pos, t, mode='vel')
                agent_types[name] = 'V'    
                size[name] = Sizes[i]
                    
        return path, agent_types, size 
    
    
    def provide_map_drawing(self, domain):
        # TODO: overwrite if actual lane markings are available
        lines_solid = []
        laneMarkings = domain.laneMarkings 
        
        lines_solid.append(np.array([[-10, laneMarkings[0]],[460, laneMarkings[0]]]))
        lines_solid.append(np.array([[-10, laneMarkings[-1]],[460, laneMarkings[-1]]]))
        
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
    
    def includes_sceneGraphs(self = None):
        return True