import numpy as np
import pandas as pd
import torch
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
import os

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    tar_tr = np.dot(Rot_matrix, (tar_tr).T).T
    track[['x','y']] = tar_tr - center
    return track


class Commotions_crossing(data_set_template):
    '''
    The part of the COMMOTIONS Urban Interactive Driving Simulator Study dataset 
    included here focueses on intersection in left-hand traffic. There, a number
    of vehicle with priority appear from the left side, and the vehicle driven by 
    huma controler has to decided when to pass through this oncoming traffic and 
    continue driving straight on.
    
    Due to this limited focus, one can describe this situation as a gap acceptance
    scenario.
    
    The data is published at https://osf.io/eazg5/?view_only= and the following
    citation can be used:
        
    Srinivasan, A. R., Schumann, J., Wang, Y., Lin, Y. S., Daly, M., 
    Solernou, A., ... & Markkula, G. (2023). The COMMOTIONS Urban Interactions 
    Driving Simulator Study Dataset. arXiv preprint arXiv:2305.11909.
    '''
    
    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
    
    def path_data_info(self = None):
        return ['x', 'y']
        

    def extract_drones(self, name, data_i, ego_track, angle, center, Driving_left):
        track = data_i[name].copy(deep = True)
        track = rotate_track(track, angle, center)
        
        trajectory = np.ones((len(ego_track.index), 2)) * np.nan
        
        frame_ego_min = ego_track.index[0]
        frame_ego_max = ego_track.index[-1]
        
        frame_min = track.index[0]
        frame_max = track.index[-1]

        start_ind = max(0, frame_min - frame_ego_min)
        end_ind = min(len(trajectory), 1 + frame_max - frame_ego_min)
        
        trajectory[start_ind : end_ind] = track[['x', 'y']].loc[max(frame_ego_min, frame_min) : min(frame_ego_max, frame_max)].to_numpy()
        
        if not Driving_left:                                                                            
            trajectory[:, -1] *= -1
        
        return trajectory
    

    def add_rotated_lanes(self, centerlines, left_boundaries, right_boundaries, heading, center_pts, right_pts = None, left_pts = None):
        # Rotate the lanes
        rot_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
        centerlines.append(np.dot(center_pts, rot_matrix))
        if right_pts is not None:
            right_boundaries.append(np.dot(right_pts, rot_matrix))
        else:
            right_boundaries.append(np.zeros((0,2)))
        if left_pts is not None:
            left_boundaries.append(np.dot(left_pts, rot_matrix))
        else:
            left_boundaries.append(np.zeros((0,2)))
        return centerlines, left_boundaries, right_boundaries
    

    def get_sceneGraph(self, lane_width):
        # set up the scenegraph
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                              'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        self.SceneGraphs = pd.DataFrame(np.zeros((1, len(sceneGraph_columns)), object), columns = sceneGraph_columns)

        # Get the end points and headings of lanes
        dict_lane = {'left':  (30, np.pi),
                     'up':    (30, -np.pi/2),
                     'right': (30, 0),
                     'down':  (50, np.pi/2)}
        
        # Prepare empty graph
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
        
        # Define the lanes leading to the intersection
        for i, lane in enumerate(dict_lane.values()):
            road_length = lane[0]
            heading = lane[1]

            in_id = 2 * i
            out_id = 2 * i + 1

            centerlines_x = np.arange(0, road_length - lane_width, 1.25) + lane_width
            centerlines_y = np.ones_like(centerlines_x) * lane_width / 2

            # Get the incoming lanes
            center_in_pts = np.stack((np.flip(centerlines_x), - centerlines_y), axis = -1)
            left_in_pts  = center_in_pts + np.ones_like(center_in_pts) * np.array([[0, - lane_width/2]])
            right_in_pts = center_in_pts + np.ones_like(center_in_pts) * np.array([[0, lane_width/2]])

            # Append to the graph
            num_nodes += len(center_in_pts) - 1
            lane_idcs += [in_id] * (len(center_in_pts) - 1)

            centerlines, left_boundaries, right_boundaries = self.add_rotated_lanes(centerlines, left_boundaries, right_boundaries, heading,
                                                                                    center_in_pts, right_in_pts, left_in_pts)
            
            lane_type.append(('VEHILCE', False))

            # Get the outgoing lanes
            center_out_pts = np.stack((centerlines_x, centerlines_y), axis = -1)

            # Get the left and right boundaries
            left_out_pts  = center_out_pts + np.ones_like(center_out_pts) * np.array([[0, lane_width/2]])
            right_out_pts = center_out_pts + np.ones_like(center_out_pts) * np.array([[0, - lane_width/2]])

            # Append to the graph
            num_nodes += len(center_out_pts) - 1
            lane_idcs += [out_id] * (len(center_out_pts) - 1)

            centerlines, left_boundaries, right_boundaries = self.add_rotated_lanes(centerlines, left_boundaries, right_boundaries, heading, 
                                                                                    center_out_pts, right_out_pts, left_out_pts)
            
            lane_type.append(('VEHILCE', False))
            

        new_id = 8
        # Create the connections to the other three lanes
        # Here, we have: radius, anggle_start, angle_in, center_x, center_y
        turns = {2: (0.5 * lane_width, -np.pi/2, -np.pi, lane_width, lane_width),
                 0: (1.5 * lane_width, np.pi/2, np.pi, lane_width, -lane_width)}
        # Get connections
        for i, lane in enumerate(dict_lane.values()):
            in_id = 2 * i
            heading = lane[1]
            # Draw curves
            for j, data in turns.items():
                radius, angle_start, angle_in, center_x, center_y = data
                # j = 0: one step clockwise, j = 1: two steps clockwise, j = 2: three steps clockwise
                # Get the out target id
                out_target_i = np.mod(i + j + 1, 4)
                out_target_id = 2 * out_target_i + 1

                connect_id = new_id + 0
                new_id += 1

                # Draw right turn (circle with radius 0.5 * lane_width from (lane_width, 0.5 * lane_width) to (0.5 * lane_width, lane_width))
                # This is an angle from -pi/2 to -pi
                arc_length = radius * np.abs((angle_in - angle_start))
                angle_steps = max(5, int(np.ceil(arc_length / 1.25)))

                angle = np.linspace(angle_start, angle_in, angle_steps)

                # Get corresponding points
                center_x = center_x + radius * np.cos(angle)
                center_y = center_y + radius * np.sin(angle)

                center_pts = np.stack((center_x, center_y), axis = -1)

                # Append to the graph
                num_nodes += len(center_pts) - 1
                lane_idcs += [connect_id] * (len(center_pts) - 1)

                centerlines, left_boundaries, right_boundaries = self.add_rotated_lanes(centerlines, left_boundaries, right_boundaries, heading, center_pts)

                lane_type.append(('VEHILCE', True))

                # Add the connections
                pre_pairs = np.concatenate((pre_pairs, np.array([[connect_id, in_id]])))
                suc_pairs = np.concatenate((suc_pairs, np.array([[in_id, connect_id]])))

                pre_pairs = np.concatenate((pre_pairs, np.array([[out_target_id, connect_id]])))
                suc_pairs = np.concatenate((suc_pairs, np.array([[connect_id, out_target_id]])))
            
            # Draw straight connection
            connect_id = new_id + 0
            new_id += 1
            j = 1
            out_target_i = np.mod(i + j + 1, 4)
            out_target_id = 2 * out_target_i + 1
            arc_length = lane_width * 2

            center_x = np.linspace(lane_width, -lane_width, max(3, int(np.ceil(arc_length / 1.25))))
            center_y = np.ones_like(center_x) * lane_width / 2

            center_pts = np.stack((center_x, center_y), axis = -1)

            # Append to the graph
            num_nodes += len(center_pts) - 1
            lane_idcs += [connect_id] * (len(center_pts) - 1)

            centerlines, left_boundaries, right_boundaries = self.add_rotated_lanes(centerlines, left_boundaries, right_boundaries, heading, center_pts)

            lane_type.append(('VEHILCE', False))

            # Add the connections
            pre_pairs = np.concatenate((pre_pairs, np.array([[connect_id, in_id]])))
            suc_pairs = np.concatenate((suc_pairs, np.array([[in_id, connect_id]])))

            pre_pairs = np.concatenate((pre_pairs, np.array([[out_target_id, connect_id]])))
            suc_pairs = np.concatenate((suc_pairs, np.array([[connect_id, out_target_id]])))

        # Build graph
        graph = pd.Series([], dtype=object)
        graph['num_nodes'] = num_nodes
        graph['lane_idcs'] = np.array(lane_idcs)
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        graph['left_boundaries'] = left_boundaries 
        graph['right_boundaries'] = right_boundaries
        graph['centerlines'] = centerlines
        graph['lane_type'] = lane_type


        # Get available gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph = self.add_node_connections(graph, device = device)
        self.SceneGraphs.loc[self.SceneGraphs.index[0]] = graph

        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'Commotions_crossing' + os.sep + 'Commotions_processed.pkl')
        # analize raw dara 
        num_tars = len(self.Data)
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        # extract raw samples
        # If false, set every y value to zero
        Driving_left = True


        # Get scenegraphs
        lane_width = 3.65
        self.get_sceneGraph(lane_width)
        
        for i in range(num_tars):
            data_i = self.Data.iloc[i]
            # find crossing point
            tar_track_all = data_i.participant_track.copy(deep = True)
            # target vehicle is to come along x axis, towards origin
            # find crossing point
            drones = np.sort(self.Data.columns[2:])
            drones = [drone for drone in drones if isinstance(data_i[drone], pd.DataFrame)]
            
            if len(drones) > 0:
                # get first drone to determine rotation adjustment
                test_track = data_i[drones[0]].copy(deep = True)
                
                # Find the crossing point
                tar_arr  = tar_track_all[['x', 'y']].to_numpy()[np.arange(0,len(tar_track_all), 12)] # Use 1s steps
                test_arr = test_track[['x', 'y']].to_numpy()[np.arange(0,len(test_track), 12)] # Use 1s steps
                
                dist = np.sum((tar_arr[:,np.newaxis,:] - test_arr[np.newaxis,:,:]) ** 2, -1)
                tar_ind, test_ind = np.unravel_index(np.argmin(dist), dist.shape)
                test_i = test_ind * 12
                tar_i = tar_ind * 12
                
                # Find the angle of the test agent at the corrsing point
                test_delta = test_arr[test_ind] - test_arr[test_ind - 1]
                angle = np.arctan2(test_delta[1], test_delta[0])
                
                # Rotate the tracks to get to the correct angle
                tar_track_all = rotate_track(tar_track_all, angle, np.zeros((1,2)))
                test_track    = rotate_track(test_track, angle, np.zeros((1,2)))
                
                # Get the vertical center, assuming the drone vehicle goes to the right in the left lane
                y_center = test_track.iloc[test_i].y - 0.5 * lane_width
                
                # Get the horizontal center, assuming the target vehicle goes upwards in the left lane
                # Get x positions of target agent while its y_position is +/- 25m of y_center
                x_values = tar_track_all[(tar_track_all.y > y_center - 25) & (tar_track_all.y < y_center + 25)].x
                x_center = x_values.mean() + 0.5 * lane_width
                
                center = np.array([[x_center, y_center]])
                
                # Adjust teh track positions
                tar_track_all = rotate_track(tar_track_all, 0, center)
                test_track    = rotate_track(test_track, 0, center)
                
                for j, drone in enumerate(drones):
                    ego_track = data_i[drone].copy(deep = True)
                    ego_track = rotate_track(ego_track, angle, center)
                    
                    tar_track = tar_track_all.loc[ego_track.index].copy(deep = True)
                    t = np.array(tar_track.t)
                    path = pd.Series(np.empty(0, np.ndarray), index = [])
                    agent_types = pd.Series(np.empty(0, str), index = [])
                    
                    if not ego_track.leaderID.iloc[0] == -1:
                        v_1_name = drones[ego_track.leaderID.iloc[0]]                                  
                        path['v_1'] = self.extract_drones(v_1_name, data_i, ego_track, angle, center, Driving_left)
                        agent_types['v_1'] = 'V'
                        
                    if not ego_track.followerID.iloc[0] == -1:
                        v_2_name = drones[ego_track.followerID.iloc[0]]                                 
                        path['v_2'] = self.extract_drones(v_2_name, data_i, ego_track, angle, center, Driving_left)
                        agent_types['v_2'] = 'V'

                    
                    if Driving_left:                                                                            
                        path['ego'] = np.stack([ego_track.x, ego_track.y], axis = -1)                                   
                        path['tar'] = np.stack([tar_track.x, tar_track.y], axis = -1)
                        domain = pd.Series(np.array([data_i.participant, 'left', 0]), index = ['Subj_ID', 'Driving', 'graph_id'])
                    else:                                                                             
                        path['ego'] = np.stack([ego_track.x, -ego_track.y], axis = -1)                                   
                        path['tar'] = np.stack([tar_track.x, -tar_track.y], axis = -1)
                        domain = pd.Series(np.array([data_i.participant, 'right', 0]), index = ['Subj_ID', 'Driving', 'graph_id'])
                    
                    domain.graph_id = self.SceneGraphs.index[0]

                    agent_types['ego'] = 'V'
                    agent_types['tar'] = 'V'
                    
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
        ego_x = path.ego[...,0]
        
        if domain.Driving == 'left':
            tar_y = path.tar[...,1]
        else:
            tar_y = - path.tar[...,1]
        
        lane_width = 3.65
        vehicle_length = 5
        
        Da = (- tar_y) - 0.5 * vehicle_length
        Dc = (- ego_x) - (lane_width + 0.5 * vehicle_length) # 1 m for traffic island
        
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
        ego_x = path.ego[...,0]
        tar_x = path.tar[...,0]

        lane_width = 3.65
        vehicle_length = 5
        
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
            D1 = np.ones(len(ego_x)) * 1000
        
        else:
            v_1_x = path.v_1[...,0]
            
            D1 = v_1_x - ego_x - vehicle_length
            D1_good = np.isfinite(D1)
            if not all(D1_good):
                index = np.arange(len(D1))
                D1 = np.interp(index, index[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])
        
        Le = np.ones_like(ego_x) * (lane_width + 1)
        
        in_position = (-(lane_width + 0.5) < tar_x) & (tar_x < 0.5) & (D1 > D_class['rejected'] + Le) 
        
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
        ego_x = path.ego[...,0]
        lane_width = 3.65
        vehicle_length = 5
        
        
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
            D1 = np.ones(len(ego_x)) * 1000
        
        else:
            v_1_x = path.v_1[...,0]
        
            D1 = v_1_x - ego_x - vehicle_length
            D1_good = np.isfinite(D1)
            if not all(D1_good):
                index = np.arange(len(D1))
                D1 = np.interp(index, index[D1_good], D1[D1_good], left = D1[D1_good][0], right = D1[D1_good][-1])
        
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
            D2 = np.ones(len(ego_x)) * 1000
        
        else:
            v_2_x = path.v_2[...,0]
        
            D2 = ego_x - v_2_x - vehicle_length
            D2_good = np.isfinite(D2)
            if not all(D2_good):
                index = np.arange(len(D2))
                D2 = np.interp(index, index[D2_good], D2[D2_good], left = D2[D2_good][0], right = D2[D2_good][-1])
        
        D3 = np.ones(len(ego_x)) * 1000
        
        Le = np.ones_like(ego_x) * (lane_width + 1)
        Lt = np.ones_like(ego_x) * lane_width
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
    
    
    def fill_empty_path(self, path, t, domain, agent_types):
        # check vehicle v_1 (in front of ego)
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
        else:
            path.v_1 = self.extrapolate_path(path.v_1, t, mode = 'vel')
                
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
        else:
            path.v_2 = self.extrapolate_path(path.v_2, t, mode = 'vel')    
        return path, agent_types
    
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
    
    def includes_sceneGraphs(self = None):
        return True