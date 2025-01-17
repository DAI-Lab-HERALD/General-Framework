import numpy as np
import pandas as pd
import torch
from data_set_template import data_set_template
from scenario_direction_highway import scenario_direction_highway
import os
from PIL import Image
from scipy import interpolate as interp
from scipy.signal import savgol_filter

class HighD_direction(data_set_template):
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
        self.scenario = scenario_direction_highway()
    
    def path_data_info(self = None):
        return ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
        
        
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
                    lane_idcs += [segment_id] * (len(center_pts) - 1)
                    num_nodes += len(center_pts) - 1

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

        graph = pd.Series([])
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
            track_i = data_i.track[['frame','x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration','laneId']].copy(deep = True)

            # get lane change
            lane_change = np.where(track_i.laneId.diff() != 0)[0]

            track_start = np.concatenate([[0], lane_change[0] + 1, [len(track_i)]])
            
            for i, i_start in enumerate(track_start[:-1]):
                # Check if the current lane holds for at least 1s (25 frames)
                if track_start[i + 1] - i_start < 25:
                    continue

                # Remove eberything that is more than 10s (250 frames) after the next lane change
                min_frame = track_i.frame.iloc[i_start]
                max_frame = track_i.frame.iloc[track_start[i + 1] - 1] + 250
                max_frame = min(max_frame, track_i.frame.iloc[-1]) 

                tar_track = track_i.set_index('frame').loc[min_frame:max_frame]
                
                # Save those paths
                path = pd.Series(np.zeros(0, np.ndarray), index = [])
                agent_types = pd.Series(np.zeros(0, str), index = [])
                sizes = pd.Series(np.zeros(0, np.ndarray), index = [])
                
                path['tar'] = tar_track.to_numpy()[...,:6]
                agent_types['tar'] = 'V'
                sizes['tar'] = np.array([data_i.width, data_i.height])
                
                # Get corresponding time points
                t = np.array(tar_track.index / 25)
                
                # get remaining important vehicles
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
                
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.Size_old = pd.DataFrame(self.Size_old)
        self.T = np.array(self.T+[()], tuple)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
        
        
    def calculate_distance(self, path, t, domain):
        r'''
        This function calculates the abridged distance of the relevant agents in a scenarion
        for each of the possible classification type. If the classification is not yet reached,
        those distances are positive, while them being negative means that a certain scenario has
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
        tar_y = path.tar[...,1]
        drivingDirection = domain.drivingDirection
        laneMarkings     = domain.laneMarkings
        
        if drivingDirection == 1: # flip agents to move to the right
            ego_x        = - ego_x
            tar_x        = - tar_x
            tar_y        = - tar_y
            laneMarkings = - laneMarkings
        
        # get left and right lane mark
        higher_lane = np.argmax(laneMarkings[np.newaxis] > tar_y[...,[0]], axis = -1)

        lane_mark_left = laneMarkings[higher_lane]
        lane_mark_right = laneMarkings[higher_lane - 1]

        D_left = lane_mark_left - tar_y
        D_right = tar_y - lane_mark_right
        D_straight = np.ones_like(D_left) * 1000

        Dist = pd.Series([D_right, D_straight, D_left], index = ['right', 'straight', 'left'])
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
        
        tar_y = path.tar[...,1]
        lane_markings = domain.laneMarkings
        in_position = (tar_y >= np.min(lane_markings)) & (tar_y <= np.max(lane_markings[-1]))
        
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
            
            If *self.can_provide_general_input() == False* or *self.can_provide_general_input() == False*  , this will be None.
        '''

        return None
    
    
    def fill_empty_path(self, path, t, domain, agent_types, size):
        n_I = self.num_timesteps_in_real

        tar_pos = path.tar[np.newaxis]
        tar_frames = 25 * (t + domain.t_0)
        
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'HighD_highways' + os.sep + 'HighD_processed.pkl')
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
        D = np.nanmin(((Pos[:,1:n_I + 1,...,:2] - tar_pos[:,:n_I,...,:2]) ** 2).sum(-1), -1)
        Close_enough = (D > 0.5) & (D < 100)
        Pos = Pos[Close_enough]
        Sizes = Sizes[Close_enough]
        D = D[Close_enough]
        
        # Distance to target agent
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
        lines_solid.append(np.array([[-10, laneMarkings[1]],[460, laneMarkings[1]]]))
        
        lines_dashed = []
        for lane_mark in laneMarkings[1:-1]:
            lines_dashed.append(np.array([[-10, lane_mark],[460, lane_mark]]))
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'HighD (Direction)',
                 'file': 'HighD_Dirc',
                 'latex': r'\emph{HighD (Dir)}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return True 
    
    def includes_sceneGraphs(self = None):
        return True