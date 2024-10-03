import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
from PIL import Image
import torch
from scipy import interpolate as interp

class HighD_interactive(data_set_template):
    '''
    The highD dataset is extracted from drone recordings of real world traffic over
    german highways. TThis is the full dataset and contains instances of car
    following, lane changes and merging.
    
    The dataset can be found at https://www.highd-dataset.com/ and the following 
    citation can be used:
        
    Krajewski, R., Bock, J., Kloeker, L., & Eckstein, L. (2018, November). The highd 
    dataset: A drone dataset of naturalistic vehicle trajectories on german highways 
    for validation of highly automated driving systems. In 2018 21st international 
    conference on intelligent transportation systems (ITSC) (pp. 2118-2125). IEEE.
    '''
    def set_scenario(self):
        self.scenario = scenario_none()
        
        def highD_classifying_agents():
            return []
        
        self.scenario.classifying_agents = highD_classifying_agents
    
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
        graph['left_boundaries'] = right_boundaries # The use of * -1 mirrors everything, switching sides
        graph['right_boundaries'] = left_boundaries # The use of * -1 mirrors everything, switching sides
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
        self.Path = []
        self.Type_old = []
        self.Size_old = []
        self.T = []
        self.Domain_old = []
        
        # Create Images
        self.Images = pd.DataFrame(np.zeros((0, 1), np.ndarray), columns = ['Image'])
        self.Target_MeterPerPx = 0.5

        # set up the scenegraph
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                              'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        self.SceneGraphs = pd.DataFrame(np.zeros((1, len(sceneGraph_columns)), object), columns = sceneGraph_columns)
        
        data_path = self.path + os.sep + 'Data_sets' + os.sep + 'HighD_highways' + os.sep + 'data' + os.sep
        potential_image_files = os.listdir(data_path)
             
        
        # Get allready saved samples
        num_samples_saved = self.get_number_of_saved_samples()

        # extract raw samples
        unique_recording_ids = np.unique(self.Data.recordingId)
        self.num_samples = 0
        for rec_i, recording_id in enumerate(unique_recording_ids):
            Data_record = self.Data[self.Data.recordingId == recording_id]
            print('Recording ' + str(rec_i + 1).rjust(len(str(len(unique_recording_ids)))) + '/{} analized'.format(len(unique_recording_ids)))
            unique_directions = np.unique(Data_record.drivingDirection)
            for dir_i, direction in enumerate(unique_directions):
                i = rec_i * len(unique_directions) + dir_i
                if i < num_samples_saved:
                    continue

                # to keep track:
                Data_direction = Data_record[Data_record.drivingDirection == direction]

                path = pd.Series(np.zeros(0, np.ndarray), index = [])
                agent_types = pd.Series(np.zeros(0, str), index = [])
                sizes = pd.Series(np.zeros(0, np.ndarray), index = [])

                # Get included frames
                min_frame = np.min(Data_direction.frame_min)
                max_frame = np.max(Data_direction.frame_max)

                # Go thorugh all agents
                lane_markings = []
                for j in range(len(Data_direction)):
                    data_j = Data_direction.iloc[j]
                    track_i = data_j.track[['frame','x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']].copy(deep = True)
                    
                    # get name
                    name = 'v_' + str(j)

                    path_j = np.full((max_frame - min_frame + 1, 6), np.nan, dtype = np.float32)
                    path_j[track_i.frame - min_frame] = track_i[['x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']].to_numpy()
                    
                    path[name] = path_j
                    agent_types[name] = 'V'
                    sizes[name] = np.array([data_j.width, data_j.height])
                    lane_markings.append(data_j.laneMarkings)
                
                t = np.arange(min_frame, max_frame + 1) / 25
            
                domain = pd.Series(np.zeros(4, object), index = ['location', 'image_id', 'graph_id', 'laneMarkings'])
                domain.location         = data_j.locationId
                domain.image_id         = data_j.recordingId
                domain.graph_id         = data_j.recordingId
                domain.laneMarkings     = np.unique(np.concatenate(lane_markings))
                
                self.num_samples += 1 
                self.Path.append(path)
                self.Type_old.append(agent_types)
                self.Size_old.append(sizes)
                self.T.append(t)
                self.Domain_old.append(domain) 

                # Chcek if data can be saved
                if dir_i < len(unique_directions) - 1:
                    self.check_created_paths_for_saving()
                else:
                    self.check_created_paths_for_saving(force_save = True)

    
        self.check_created_paths_for_saving(last = True) 
            
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
    
    
    def _fill_round_about_path(self, pos, t, domain):
        v_x = pos[...,0]
        v_y = pos[...,1]
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
    
    def fill_empty_path(self, path, t, domain, agent_types, size = None):
        for agent in path.index:
            path[agent] = self._fill_round_about_path(path[agent], t, domain)
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
    
    def includes_sceneGraphs(self = None):
        return True
    
