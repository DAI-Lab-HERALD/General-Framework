import numpy as np
import pandas as pd
import torch
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
from scipy.signal import savgol_filter
import os
from PIL import Image

class CoR_left_turns(data_set_template):
    '''
    The L-GAP simulator study focuses on a two agent scenario, where on computer
    controlloed vehicle drives straight across an intersection, whle the human driving
    vehicle, approaching the intersection from the opposite direction with the intention
    to turn left, has to decided if it is to cross either in front of or behind the AV.
    
    This is consequently a exaple for a gap acceptance scenario, the code of which can be found at 
    https://data.4tu.nl/file/80e7f503-2471-4f06-be03-9d620a2a5495/0a77d88b-15df-47aa-92c9-d141baf6a2b1
    and the following citation can be used:
        
    Zgonnikov, A., Abbink, D., & Markkula, G. (2022). Should I stay or should I go? 
    Cognitive modeling of left-turn gap acceptance decisions in human drivers. 
    Human factors, 00187208221144561.
    '''
    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
    
    def path_data_info(self = None):
        return ['x', 'y', 'v_x', 'v_y', 'theta']
    
    def get_image(self):
        # analize raw dara 
        self.num_samples = len(self.Data)
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []

        x_center = 163.6666
        y_center = -23.6666
        rot_angle = 0
        px_per_meter = 6

        self.Images = pd.DataFrame(np.zeros((1, 2), object), columns = ['Image', 'Target_MeterPerPx'])

        image_path = self.path + os.sep + 'Data_sets' + os.sep + 'CoR_left_turns' + os.sep + 'image' + os.sep + 'L-GAP_image_6_px_per_meter.png'
        image = Image.open(image_path)
        rgb_image = image.convert('RGB')
        img = np.array(rgb_image, dtype=np.uint8)

        self.Images.loc[0].Image = img
        self.Images.loc[0].Target_MeterPerPx = 1/px_per_meter
        return x_center, y_center, rot_angle
    

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


    def get_sceneGraph(self):
        # set up the scenegraph
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                              'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        self.SceneGraphs = pd.DataFrame(np.zeros((1, len(sceneGraph_columns)), object), columns = sceneGraph_columns)

        # Get the lane width
        lane_width = 3.5

        # Get the end points and headings of lanes
        dict_lane = {'left':  (200, np.pi),
                     'up':    (25, -np.pi/2),
                     'right': (25, 0),
                     'down':  (25, np.pi/2)}
        
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
            center_in_pts = np.stack((np.flip(centerlines_x), centerlines_y), axis = -1)
            left_in_pts  = center_in_pts + np.ones_like(center_in_pts) * np.array([[0, - lane_width/2]])
            right_in_pts = center_in_pts + np.ones_like(center_in_pts) * np.array([[0, lane_width/2]])

            # Append to the graph
            num_nodes += len(center_in_pts) - 1
            lane_idcs += [in_id] * (len(center_in_pts) - 1)

            centerlines, left_boundaries, right_boundaries = self.add_rotated_lanes(centerlines, left_boundaries, right_boundaries, heading,
                                                                                    center_in_pts, right_in_pts, left_in_pts)
            
            lane_type.append(('VEHILCE', False))

            # Get the outgoing lanes
            center_out_pts = np.stack((centerlines_x, - centerlines_y), axis = -1)

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
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 'CoR_left_turns' + os.sep + 'CoR_processed.pkl')

        # Define agents
        agents = ['ego', 'tar']

        # Get images
        x_center, y_center, rot_angle = self.get_image()

        # Get scenegraphs
        self.get_sceneGraph()

        # extract raw samples
        for i in range(self.num_samples):
            path = pd.Series(np.empty(len(agents), np.ndarray), index = agents)
            agent_type = pd.Series(np.full(len(agents), 'V', dtype = str), index = agents)
            
            t_index = self.Data.bot_track.iloc[i].index
            t = np.array(self.Data.bot_track.iloc[i].t[t_index])
            ego_raw = self.Data.iloc[i].bot_track[['x', 'y', 'vx', 'vy', 'heading']]
            tar_raw = self.Data.iloc[i].ego_track[['x', 'y', 'vx', 'vy', 'heading']]
            
            ego = []
            tar = []
            for column in ego_raw.columns:
                ego_value = ego_raw[column][t_index]
                tar_value = tar_raw[column][t_index]
                # Condider wrapping for angels
                if column == 'heading':
                    # Consider wrap around
                    ego_real = np.cos(ego_value)
                    ego_imag = np.sin(ego_value)

                    ego_real_smooth = savgol_filter(ego_real,50,3)
                    ego_imag_smooth = savgol_filter(ego_imag,50,3)

                    ego.append(np.arctan2(ego_imag_smooth, ego_real_smooth))

                    tar_real = np.cos(tar_value)
                    tar_imag = np.sin(tar_value)

                    tar_real_smooth = savgol_filter(tar_real,50,3)
                    tar_imag_smooth = savgol_filter(tar_imag,50,3)

                    tar.append(np.arctan2(tar_imag_smooth, tar_real_smooth))
                else:
                    ego.append(savgol_filter(ego_value,50,3))
                    tar.append(savgol_filter(tar_value,50,3))

            ego = np.stack(ego, axis = -1)
            tar = np.stack(tar, axis = -1)

            # Limit headings
            ego[...,4] = np.mod(ego[...,4] + np.pi, 2 * np.pi) - np.pi
            tar[...,4] = np.mod(tar[...,4] + np.pi, 2 * np.pi) - np.pi
            
            path.ego = ego
            path.tar = tar
            
            domain = pd.Series([int(self.Data.subj_id.iloc[i]),
                                x_center,
                                y_center,
                                rot_angle,
                                self.Images.index[0],
                                self.SceneGraphs.index[0]
                                ], index = ['Subj_ID','x_center','y_center','rot_angle', 'image_id', 'graph_id'])
            
            # domain = pd.Series(np.ones(1, int) * self.Data.subj_id.iloc[i], index = ['Subj_ID'])
            
            self.Path.append(path)
            self.Type_old.append(agent_type)
            self.T.append(t)
            self.Domain_old.append(domain)
        
        # print(self.Path)
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
        tar_x = path.tar[...,0]
        tar_y = path.tar[...,1]
        
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
        
        tar_x = path.tar[...,0]
        
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
        ego_x = path.ego[...,0]
        
        lane_width = 3.5
        
        Le = np.ones_like(ego_x) * 2 * lane_width
        
        Lt = np.ones_like(ego_x) * lane_width
        
        D1 = np.ones_like(ego_x) * 500
        
        D2 = np.ones_like(ego_x) * 500
        
        D3 = np.ones_like(ego_x) * 500
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
        
    
    def fill_empty_path(self, path, t, domain, agent_types):
        return path, agent_types
    
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
        return True
    
    def includes_sceneGraphs(self = None):
        return True

