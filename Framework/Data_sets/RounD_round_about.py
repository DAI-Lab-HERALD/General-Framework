import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
import os
from PIL import Image

# Map import
import copy
import torch







def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T

    if 'v_x' in track.columns:
        tar_vel = track[['v_x','v_y']].to_numpy()
        track[['v_x','v_y']] = np.dot(Rot_matrix, tar_vel.T).T

    if 'a_x' in track.columns:
        tar_acc = track[['a_x','a_y']].to_numpy()
        track[['a_x','a_y']] = np.dot(Rot_matrix, tar_acc.T).T
    return track

def rotate_track_array(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[:,0:2]
    track[:,0:2] = np.dot(Rot_matrix,(tar_tr - center).T).T

    return track


class RounD_round_about(data_set_template): 
    '''
    The rounD dataset is extracted from drone recordings of real world traffic over
    three german round abouts. This specific instance focuses on the behavior of 
    vehicles entering a roundabout, and if the tend to cut of or let pass the vehicles
    already in the round about, which is one examply of a gap acceptance scenario.
    
    The dataset can be found at https://www.round-dataset.com/ and the following 
    citation can be used:
        
    Krajewski, R., Moers, T., Bock, J., Vater, L., & Eckstein, L. (2020, September). 
    The round dataset: A drone dataset of road user trajectories at roundabouts in 
    germany. In 2020 IEEE 23rd International Conference on Intelligent Transportation 
    Systems (ITSC) (pp. 1-6). IEEE.
    '''

    def get_lane_graph(self, map_path, UtmOrigin):
        import lanelet2
        from av2.geometry.interpolate import compute_midpoint_line
        from lanelet2.projection import UtmProjector
        from pyproj import Proj, transform

        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                    lanelet2.traffic_rules.Participants.Vehicle)
        # Transfer UTm Origin into lattiude and longitude
        utm_zone = 32
        utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', south= False)
        wgs84_proj = Proj(proj='latlong', datum='WGS84')
        longitude, latitude = transform(utm_proj, wgs84_proj, UtmOrigin[0,0], UtmOrigin[0,1])

        # Load map
        projector = UtmProjector(lanelet2.io.Origin(latitude, longitude))
        map = lanelet2.io.load(map_path, projector)
        routing_graph = lanelet2.routing.RoutingGraph(map, traffic_rules)

        # build node features
        lane_ids = []
        centerlines, left_boundaries, right_boundaries = [], [], []
        lane_type = []
        for ll in map.laneletLayer:
            # Check lanelet type
            if ll.attributes['subtype'] == 'road' or ll.attributes['subtype'] == 'highway':
                lane_type_val = 'VEHICLE'
                is_intersection = False
            elif ll.attributes['subtype'] == 'crosswalk':
                lane_type_val = 'PEDESTRIAN'
                is_intersection = True
            elif ll.attributes['subtype'] == 'walkway':
                lane_type_val = 'PEDESTRIAN'
                is_intersection = False
            elif ll.attributes['subtype'] == 'bicycle_lane':
                lane_type_val = 'BIKE'
                is_intersection = False
            elif ll.attributes['subtype'] == 'emergency_lane':
                continue
            else:
                print('Unknown lanelet type: {}'.format(ll.attributes['subtype']))
                continue
            
            # Get boundaries
            left_boundary = np.zeros((len(ll.leftBound), 2))
            right_boundary  = np.zeros((len(ll.rightBound), 2))

            for i in range(len(ll.leftBound)):
                left_boundary[i][0] = copy.deepcopy(ll.leftBound[i].x)
                left_boundary[i][1] = copy.deepcopy(ll.leftBound[i].y)

            for i in range(len(ll.rightBound)):
                right_boundary[i][0] = copy.deepcopy(ll.rightBound[i].x)
                right_boundary[i][1] = copy.deepcopy(ll.rightBound[i].y)
            
            # computes centerline with min(max(M,N), 10) data points per lanelet
            distance = min(np.linalg.norm(left_boundary[0] - left_boundary[-1]), np.linalg.norm(right_boundary[0] - right_boundary[-1]))
            num_points = max(int(distance) + 1, left_boundary.shape[0], right_boundary.shape[0])
            centerline, _ = compute_midpoint_line(left_boundary, right_boundary, num_points)
            centerline = copy.deepcopy(centerline)  



            # Get the lane marker types
            lane_type.append((lane_type_val, is_intersection))
            
            # num_segs = len(centerline) - 1
            lane_ids.append(ll.id)
            centerlines.append(centerline)
            left_boundaries.append(left_boundary)
            right_boundaries.append(right_boundary)
        

        # node indices (when nodes are concatenated into one array)
        lane_idcs = []
        num_nodes = 0 
        for i, lane_id in enumerate(lane_ids):
            num_nodes_i = len(centerlines[i]) - 1
            num_nodes += num_nodes_i
            lane_idcs += [i] * num_nodes_i

        # Get connections
        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = map.laneletLayer[lane_id]

            # compute lane_id pairs of predecessor [u,v]
            if len(routing_graph.previous(lane)) > 0:
                for prev_lane in routing_graph.previous(lane):
                    if prev_lane.id in lane_ids:
                        j = lane_ids.index(prev_lane.id)
                        pre_pairs.append([i, j])

            # compute lane_id pairs of successor [u,v]
            if len(routing_graph.following(lane)) > 0:
                for foll_lane in routing_graph.following(lane):
                    if foll_lane.id in lane_ids:
                        j = lane_ids.index(foll_lane.id)
                        suc_pairs.append([i, j])

            # compute lane_id pairs of left [u,v]
            if routing_graph.left(lane) is not None:
                if routing_graph.left(lane).id in lane_ids:
                    j = lane_ids.index(routing_graph.left(lane).id)
                    left_pairs.append([i, j])

            # compute lane_id pairs of right [u,v]
            if routing_graph.right(lane) is not None:
                if routing_graph.right(lane).id in lane_ids:
                    j = lane_ids.index(routing_graph.right(lane).id)
                    right_pairs.append([i, j])
        
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)

        if len(pre_pairs) == 0:
            pre_pairs = np.zeros((0, 2), np.int64)
        
        if len(suc_pairs) == 0:
            suc_pairs = np.zeros((0, 2), np.int64)

        if len(left_pairs) == 0:
            left_pairs = np.zeros((0, 2), np.int64)

        if len(right_pairs) == 0:
            right_pairs = np.zeros((0, 2), np.int64)

        graph = pd.Series([])
        graph['num_nodes'] = num_nodes
        graph['lane_idcs'] = np.array(lane_idcs)
        graph['centerlines'] = centerlines
        graph['left_boundaries'] = left_boundaries
        graph['right_boundaries'] = right_boundaries
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        graph['lane_type'] = lane_type

        # Get available gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph = self.add_node_connections(graph, device = device)
        return graph



    def analyze_maps(self):
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                    'RounD_round_about' + os.sep + 'RounD_processed.pkl')
            
        unique_rec = np.unique(self.Data[['recordingId','locationId']], axis = 0)
        self.Rec_loc = pd.Series(unique_rec[:,1], index = unique_rec[:,0])
        
        unique_location = [0, 1, 2]
        
        Loc_data_pix = pd.DataFrame(np.zeros((len(unique_location),4),float), columns = ['xCenter', 'yCenter', 'r', 'R'])
        
        Loc_data_pix.xCenter.iloc[0] = 801.0
        Loc_data_pix.yCenter.iloc[0] = -465.0
        Loc_data_pix.R.iloc[0] = 247.5
        Loc_data_pix.r.iloc[0] = 154.0
        
        Loc_data_pix.xCenter.iloc[1] = 781.0
        Loc_data_pix.yCenter.iloc[1] = -481.0
        Loc_data_pix.R.iloc[1] = 98.1
        Loc_data_pix.r.iloc[1] = 45.1
        
        Loc_data_pix.xCenter.iloc[2] = 1011.0
        Loc_data_pix.yCenter.iloc[2] = -449.0
        Loc_data_pix.R.iloc[2] = 100.0
        Loc_data_pix.r.iloc[2] = 48.0
        
        
        self.Loc_scale = {}
        self.Loc_data = pd.DataFrame(np.zeros((len(self.Rec_loc), 4),float), index = self.Rec_loc.index, columns = Loc_data_pix.columns)
        for rec_Id in self.Rec_loc.index:
            loc_Id = self.Rec_loc[rec_Id]
            Meta_data=pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
                                  'RounD_round_about' + os.sep + 
                                  'data' + os.sep + '{}_recordingMeta.csv'.format(str(rec_Id).zfill(2)))
            self.Loc_scale[rec_Id] = Meta_data['orthoPxToMeter'][0] * 10
            self.Loc_data.loc[rec_Id] = Loc_data_pix.iloc[loc_Id] * self.Loc_scale[rec_Id] 


    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
        self.analyze_maps()

    
    def path_data_info(self = None):
        return ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
    
    def _create_path_sample(self, tar_track, ego_track, other_agents, frame_min, frame_max, 
                            ego_id, v_1_id, v_2_id, v_3_id, v_4_id,
                            original_angle, Rot_center, data_i, SceneGraphs_unturned, path_id):
        
        tar_track_l = tar_track.loc[frame_min:frame_max].copy(deep = True)
        ego_track_l = ego_track.loc[frame_min:frame_max].copy(deep = True)
        
        path = pd.Series(np.empty(0, np.ndarray), index = [])
        agent_types = pd.Series(np.zeros(0, str), index = [])
        sizes = pd.Series(np.empty(0, np.ndarray), index = [])
        
        path['tar'] = tar_track_l.to_numpy()[:,:6]
        path['ego'] = ego_track_l.to_numpy()[:,:6]
    
        agent_types['tar'] = 'V'
        agent_types['ego'] = 'V'
        
        sizes['tar'] = np.array([data_i.length, data_i.width])
        sizes['ego'] = np.array([self.Data.loc[ego_id].length, self.Data.loc[ego_id].width])
        
        if v_1_id >= 0:
            v_1_track = other_agents.loc[v_1_id].track.loc[frame_min:frame_max]
            
            if len(v_1_track) > 0:
                frame_min_v1 = v_1_track.index.min()
                frame_max_v1 = v_1_track.index.max()
                
                v_1_track_l = v_1_track.loc[frame_min_v1:frame_max_v1]
                
                path['v_1'] = v_1_track_l.to_numpy()[:,:6]
                agent_types['v_1'] = 'V'
                sizes['v_1'] = np.array([self.Data.loc[v_1_id].length, self.Data.loc[v_1_id].width])
            
        if v_2_id >= 0:
            v_2_track = other_agents.loc[v_2_id].track.loc[frame_min:frame_max]
            
            if len(v_2_track) > 0:
                frame_min_v2 = v_2_track.index.min()
                frame_max_v2 = v_2_track.index.max()
                
                v_2_track_l = v_2_track.loc[frame_min_v2:frame_max_v2]
                
                path['v_2'] = v_2_track_l.to_numpy()[:,:6]
                agent_types['v_2'] = 'V'
                sizes['v_2'] = np.array([self.Data.loc[v_2_id].length, self.Data.loc[v_2_id].width])
            
        if v_3_id >= 0:
            v_3_track = other_agents.loc[v_3_id].track.loc[frame_min:frame_max]
            
            if len(v_3_track) > 0:
                frame_min_v3 = v_3_track.index.min()
                frame_max_v3 = v_3_track.index.max()

                v_3_track_l = v_3_track.loc[frame_min_v3:frame_max_v3]
                
                path['v_3'] = v_3_track_l.to_numpy()[:,:6]
                agent_types['v_3'] = 'V'
                sizes['v_3'] = np.array([self.Data.loc[v_3_id].length, self.Data.loc[v_3_id].width])
    
        if v_4_id >= 0:
            v_4_track = other_agents.loc[v_4_id].track.loc[frame_min:frame_max]
            
            if len(v_4_track) > 0:
                frame_min_v4 = v_4_track.index.min()
                frame_max_v4 = v_4_track.index.max()
                
                v_4_track_l = v_4_track.loc[frame_min_v4:frame_max_v4]
                
                path['v_4'] = v_4_track_l.to_numpy()[:,:6]
                agent_types['v_4'] = 'P'
                sizes['v_4'] = np.array([0.5, 0.5])

        t = np.array(tar_track_l.index / 25)
        
        domain = pd.Series(np.zeros(9, object), index = ['location', 'splitting', 'image_id', 'graph_id', 'track_id', 'rot_angle', 'x_center', 'y_center', 'class'])
        domain.location  = data_i.locationId
        if np.mod(path_id, 5) == 0:
            domain.splitting = 'test'
        else:
            domain.splitting = 'train'
        domain.image_id  = data_i.recordingId
        domain.graph_id  = len(self.Domain_old)
        domain.track_id  = data_i.trackId
        domain.rot_angle = original_angle
        domain.x_center  = Rot_center[0,0]
        domain.y_center  = Rot_center[0,1]
        domain['class']  = data_i['class']

        # Get local SceneGraph, by rotating the respective positions and boundaries
        graph = SceneGraphs_unturned.loc[domain.location].copy(deep = True)
        centerlines_new = []
        left_boundaries_new = []
        right_boundaries_new = []
        for i in range(len(graph.centerlines)):
            c = graph.centerlines[i].copy()
            l = graph.left_boundaries[i].copy()
            r = graph.right_boundaries[i].copy()
            centerlines_new.append(rotate_track_array(c, original_angle, Rot_center))
            left_boundaries_new.append(rotate_track_array(l, original_angle, Rot_center))
            right_boundaries_new.append(rotate_track_array(r, original_angle, Rot_center)) 
    
        graph.centerlines = centerlines_new
        graph.left_boundaries = left_boundaries_new
        graph.right_boundaries = right_boundaries_new
        self.SceneGraphs.loc[domain.graph_id] = graph
        
        self.Path.append(path)
        self.Type_old.append(agent_types)
        self.Size_old.append(sizes)
        self.T.append(t)
        self.Domain_old.append(domain)
        self.num_samples = self.num_samples + 1
    
        
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'RounD_round_about' + os.sep + 'RounD_processed.pkl')
        # analize raw dara 
        self.Data = self.Data.reset_index(drop = True)
        num_samples_max = len(self.Data)
        self.Path = []
        self.Type_old = []
        self.Size_old = []
        self.T = []
        self.Domain_old = []
        
        # Get images
        self.Images = pd.DataFrame(np.zeros((len(self.Loc_data), 1), object), 
                                   index = self.Loc_data.index, columns = ['Image'])
        
        # Get scenegraphs
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                              'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']  
        SceneGraphs_unturned = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), columns = sceneGraph_columns)
        self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), columns = sceneGraph_columns)
        
        
        self.Target_MeterPerPx = 0.5 # TODO: Maybe change this?
        for rec_Id in self.Loc_data.index:
            ## Get Image
            img_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                        'RounD_round_about' + os.sep + 'data' + os.sep + 
                        str(rec_Id).zfill(2) + '_background.png')
            
            # open image
            img = Image.open(img_file)

            # rescale image
            img_scaleing = self.Loc_scale[rec_Id] / self.Target_MeterPerPx
            
            height_new = int(img.height * img_scaleing)
            width_new  = int(img.width * img_scaleing)
            
            img_new = img.resize((width_new, height_new), Image.LANCZOS)
            
            # save image
            self.Images.loc[rec_Id].Image = np.array(img_new)


        location_to_lanelet_file = {0: '0_neuweiler', 
                                    1: '1_kackertstrasse', 
                                    2: '2_thiergarten'}
        
        # Go though unique locations
        unique_loc = np.unique(self.Rec_loc)
        for loc_Id in unique_loc:
            ## Get SceneGraph
            lanelet_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                            'RounD_round_about' + os.sep + 'maps' + os.sep + 
                            'lanelets' + os.sep + location_to_lanelet_file[loc_Id] + os.sep +
                            'location' + str(loc_Id) + '.osm')
            rec_Id = self.Rec_loc[self.Rec_loc == loc_Id].index[0]

            # Get the UTM origin
            meta_file = (self.path + os.sep + 'Data_sets' + os.sep +
                         'RounD_round_about' + os.sep + 'data' + os.sep + 
                         str(rec_Id).zfill(2) + '_recordingMeta.csv')
            meta_data = pd.read_csv(meta_file).iloc[0]
            origin = np.array([[meta_data.xUtmOrigin, meta_data.yUtmOrigin]])

            # load lanelet map
            SceneGraphs_unturned.loc[loc_Id] = self.get_lane_graph(lanelet_file, origin)
            
        # extract raw samples
        self.num_samples = 0
        for i in range(num_samples_max):

            # to keep track:
            if np.mod(i,100) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
                print('found cases: ' + str(self.num_samples))
                print('')
            data_i = self.Data.iloc[i]
            # Get location information
            loc_data = self.Loc_data.iloc[data_i.recordingId]

            # assume i is the tar vehicle, which has to be a motor vehicle
            if data_i['class'] in ['bicycle', 'pedestrian', 'trailer']:
                continue
            
            tar_track = data_i.track[['frame','xCenter','yCenter','xVelocity','yVelocity','xAcceleration','yAcceleration']]
            tar_track = tar_track.rename(columns={"xCenter": "x", "yCenter": "y",
                                                  "xVelocity" : "v_x", "yVelocity" : "v_y",
                                                  "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            Rot_center = np.array([[loc_data.xCenter, loc_data.yCenter]])
            
            tar_track['r'] = np.sqrt((tar_track.x - loc_data.xCenter) ** 2 + 
                                     (tar_track.y - loc_data.yCenter) ** 2)
            
            # exclude trajectory driving over the middle
            if any(tar_track['r'] < loc_data.r):
                continue
            
            # check if tar_track goes through round_about or use shortcut around it
            if not any(tar_track['r'] <= loc_data.R):
                continue
            
            # Exclude vehicles that already startinside the round about
            if tar_track['r'].iloc[0] <= loc_data.R + 10:
                continue
            
            # frame where target vehicle approaches roundd about
            frame_entry = np.where(tar_track['r'] < loc_data.R + 10)[0][0]
            
            tar_frame_A = tar_track['frame'].iloc[np.where(tar_track['r'] < loc_data.R)[0][0]]

            # angle along this route
            original_angle = np.angle((tar_track.x.iloc[0] - tar_track.x.iloc[frame_entry]) + 
                                      (tar_track.y.iloc[0] - tar_track.y.iloc[frame_entry]) * 1j, deg = False)
            
            
            tar_track = rotate_track(tar_track, original_angle, Rot_center)
            
            tar_track['angle'] = np.angle(tar_track.x + tar_track.y * 1j)
            
            tar_track = tar_track.set_index('frame')
            
            other_agents = self.Data[['trackId','class','track']].iloc[data_i.otherVehicles].copy(deep = True)
            
            for j in range(len(other_agents)):
                track_i = other_agents['track'].iloc[j] 
                
                track_i = track_i[['frame','xCenter','yCenter','xVelocity','yVelocity','xAcceleration','yAcceleration']]
                track_i = track_i.rename(columns={"xCenter": "x", "yCenter": "y",
                                                  "xVelocity" : "v_x", "yVelocity" : "v_y",
                                                  "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
                
                track_i = rotate_track(track_i, original_angle, Rot_center)
                
                track_i = track_i.set_index('frame').reindex(np.arange(tar_track.index[0], tar_track.index[-1] + 1))

                track_i['r']     = np.sqrt(track_i.x ** 2 + track_i.y ** 2)
                track_i['angle'] = np.angle(track_i.x + track_i.y * 1j)
                
                other_agents['track'].iloc[j] = track_i
            
            # Looking for ego_vehicle. Two conditions:
            # - Actually cross (if ego vehicle leaves before, it has no need for predictions)
            # - Gap is either rejected
            
            rejected_Ego = []
            rejected_frame_C = []
            accepted_Ego = []
            accepted_frame_C = []
            
            # check status of other agents
            for j in range(len(other_agents)):
                tr_j = other_agents['track'].iloc[j] 
                
                if other_agents['class'].iloc[j] in ['bicycle', 'pedestrian', 'trailer']:
                    continue
                # Check if ego vehicle is there to have offered accepted gap
                if tr_j.index[0] > tar_frame_A:
                    continue
                
                contested = ((tr_j.r.to_numpy() <= loc_data.R) &
                             (tr_j.angle.to_numpy() > 0) & 
                             (tr_j.angle.to_numpy() < np.pi / 6))
                K = np.where(contested[1:] & (contested[:-1] == False))[0] + 1
                
                for k in K:
                    frame_C = tr_j.index[0] + k
                    if tr_j.r.to_numpy()[k - 1] > loc_data.R:
                        continue
                    
                    # Check if target vehicle was there to have closed the gap
                    if tar_track.index[0] > frame_C:
                        continue
                    
                    if frame_C <= tar_frame_A:
                        rejected_Ego.append(other_agents.trackId.iloc[j])
                        rejected_frame_C.append(frame_C)
                    else:
                        accepted_Ego.append(other_agents.trackId.iloc[j])
                        accepted_frame_C.append(frame_C)
            
            rejected_order = np.argsort(rejected_frame_C)
            rejected_Ego = np.array(rejected_Ego)[rejected_order]  
            rejected_frame_C = np.array(rejected_frame_C)[rejected_order] 
            
            other_ped = other_agents[np.logical_or(other_agents['class'] == 'pedestrian', other_agents['class'] == 'bicycle')]
            
            entered_RA = []
            in_RA = []
            
            for j in range(len(other_agents)):
                tr_j = other_agents['track'].iloc[j] 
                interested = ((tr_j.r.to_numpy() <= loc_data.R) &
                              (tr_j.angle.to_numpy() > - np.pi / 6) & 
                              (tr_j.angle.to_numpy() < np.pi / 6))
                K = np.where(interested[1:] & (interested[:-1] == False))[0] + 1
                if interested[0]:
                    K = np.concatenate(([0], K))
                
                for k in K:
                    if k == 0:
                        frame_E = tr_j.index[0] + k
                        # Decide if vehicle came into roundabout
                        dx = tr_j.x.iloc[min(5, len(tr_j.x) - 1)] - tr_j.x.iloc[0]
                        dy = tr_j.y.iloc[min(5, len(tr_j.x) - 1)] - tr_j.y.iloc[0]
                        
                        dangle = np.angle(dx + dy * 1j) - 0.5 * np.pi - tr_j.angle.iloc[0]
              
                        if dangle > 0.2 and tr_j.angle.iloc[0] > 0:
                            entered_RA.append([other_agents.trackId.iloc[j], frame_E])
                        else:
                            in_RA.append([other_agents.trackId.iloc[j], frame_E])
                    else:
                        if tr_j.r.to_numpy()[k - 1] > loc_data.R and tr_j.angle.to_numpy()[k - 1] > 0:
                            # moved into round about
                            frame_A = tr_j.index[0] + k
                            entered_RA.append([other_agents.trackId.iloc[j], frame_A])
                        else:
                            # time of exiting the round about
                            frame_E = tr_j.index[0] + k
                            
                            in_RA.append([other_agents.trackId.iloc[j], frame_E])
                
            entered_RA = np.array(entered_RA)
            in_RA = np.array(in_RA)
            
            if len(in_RA) > 1:
                in_order = np.argsort(in_RA[:,1])
                in_RA = in_RA[in_order,:] 
            
            # Assume accepted gap
            if len(accepted_Ego) > 0:
                ego_id = accepted_Ego[np.argmin(accepted_frame_C)]
                frame_C = np.min(accepted_frame_C)
                
                ego_track = other_agents.loc[ego_id].track.copy(deep = True)
                
                frame_min = max(ego_track.index[0], tar_track.index[0])
                frame_max = min(ego_track.index[-1], tar_track.index[-1])
                
                # find v_1: in_RA directly before ego vehicle
                in_ego = np.where(in_RA[:,0] == ego_id)[0][0]
                
                if in_ego == 0:
                    v_1_id = -1
                else:
                    v_1_id = in_RA[in_ego - 1, 0]
                    
                # find v_2: in_RA directly after ego vehicle
                try:
                    v_2_id = in_RA[in_ego + 1, 0]
                except:
                    v_2_id = -1
                
                # find v_3: entered_RA with the largest tar_frame_A that is still smaller than tar_frame_A
                if len(entered_RA) > 0:
                    feasible = entered_RA[:,1] < tar_frame_A
                else:
                    feasible = [False]
                if np.any(feasible):
                    v_3_id = entered_RA[feasible,0][np.argmax(entered_RA[feasible,1])] 
                else: 
                    v_3_id = -1
                # check for pedestrian 
                if len(other_ped) > 0:
                    distance = []
                    for j in range(len(other_ped)):
                        track_p = other_ped.iloc[j].track.loc[frame_min:frame_max]
                        tar_track_help = tar_track.copy(deep = True)
                        distance_to_cross = np.sqrt((track_p.x - loc_data.R - 5) ** 2 +
                                                    (track_p.y - tar_track_help.iloc[frame_entry].y) ** 2)
                        distance_to_tar = np.sqrt((track_p.x - tar_track_help.x) ** 2 + (track_p.y - tar_track_help.y) ** 2)
                        
                        if np.min(distance_to_cross) < 10:
                            usable = distance_to_cross < 10
                            distance.append(np.min(distance_to_tar.loc[usable.index[usable]]))
                        else:
                            distance.append(np.min(distance_to_cross) + 1000)
                    v_4_id = other_ped.index[np.argmin(distance)]              
                else:
                    v_4_id = -1
            
            
                # Collect path data
                self._create_path_sample(tar_track, ego_track, other_agents, frame_min, frame_max, 
                                         ego_id, v_1_id, v_2_id, v_3_id, v_4_id,
                                         original_angle, Rot_center, data_i, SceneGraphs_unturned, i)
                
                
            # Assume rejected gap
            for [ego_id, frame_C] in zip(rejected_Ego,rejected_frame_C):
                ego_track = other_agents.loc[ego_id].track.copy(deep = True)
                
                frame_min = max(ego_track.index[0], tar_track.index[0])
                frame_max = min(ego_track.index[-1], tar_track.index[-1])
                
                # find v_1: in_RA directly before ego vehicle
                in_ego = np.where(in_RA[:,0] == ego_id)[0][0]
                
                if in_ego == 0:
                    v_1_id = -1
                else:
                    v_1_id = in_RA[in_ego - 1, 0]
                    
                # find v_2: in_RA directly after ego vehicle
                try:
                    v_2_id = in_RA[in_ego + 1, 0]
                except:
                    v_2_id = -1
                
                # find v_3: entered_RA with the largest tar_frame_A that is still smaller than tar_frame_A
                if len(entered_RA) > 0:
                    feasible = entered_RA[:,1] < tar_frame_A
                else:
                    feasible = [False]
                if np.any(feasible):
                    v_3_id = entered_RA[feasible,0][np.argmax(entered_RA[feasible,1])] 
                else: 
                    v_3_id = -1
                # check for pedestrian 
                if len(other_ped) > 0:
                    distance = []
                    for j in range(len(other_ped)):
                        track_p = other_ped.iloc[j].track.loc[frame_min:frame_max]
                        tar_track_help = tar_track.copy(deep = True)
                        distance_to_cross = np.sqrt((track_p.x - loc_data.R - 5) ** 2 +
                                                    (track_p.y - tar_track_help.iloc[frame_entry].y) ** 2)
                        distance_to_tar = np.sqrt((track_p.x - tar_track_help.x) ** 2 + (track_p.y - tar_track_help.y) ** 2)
                        
                        if np.min(distance_to_cross) < 10:
                            usable = distance_to_cross < 10
                            distance.append(np.min(distance_to_tar.loc[usable.index[usable]]))
                        else:
                            distance.append(np.min(distance_to_cross) + 1000)
                    v_4_id = other_ped.index[np.argmin(distance)] 
                else:
                    v_4_id = -1
            
                # Collect path data
                self._create_path_sample(tar_track, ego_track, other_agents, frame_min, frame_max, 
                                         ego_id, v_1_id, v_2_id, v_3_id, v_4_id,
                                         original_angle, Rot_center, data_i, SceneGraphs_unturned, i)
        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.Size_old = pd.DataFrame(self.Size_old)
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
        lane_width = 3.5
        vehicle_length = 5 
        
        ego_x = path.ego[...,0]
        ego_y = path.ego[...,1]
        tar_x = path.tar[...,0]
        tar_y = path.tar[...,1]
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        tar_r = np.sqrt(tar_x ** 2 + tar_y ** 2)
        tar_a = np.angle(tar_x + tar_y * 1j)
        
        # From location data
        R = self.Loc_data.R[domain.image_id]
        
        ego_frame_0 = np.nanargmin(np.abs(ego_a) + (ego_r > R) * 2 * np.pi, axis = 1)
        ego_a_change_n, ego_a_change_t = np.where((ego_a[:,1:] < np.pi * 0.5) & 
                                                  (ego_a[:,:-1] > np.pi * 0.5))
        
        for i, t_ind in enumerate(ego_a_change_t + 1):
            n_ind = ego_a_change_n[i]
            if t_ind < ego_frame_0[n_ind]:
                ego_a[n_ind,:t_ind] -= 2 * np.pi
            else:
                ego_a[n_ind,t_ind:] += 2 * np.pi

        Rl = R - 0.5 * lane_width
        
        D_ego = np.log(1 / (1 + np.exp(ego_r - Rl))) * np.tanh(5 * ego_a) - Rl * ego_a
        
        Dc = D_ego - 0.5 * vehicle_length
        
        tar_a_change = np.where((tar_a[1:] < np.pi * 0.5) & (tar_a[:-1] > np.pi * 0.5))[0] + 1
        for i_change in tar_a_change:
            tar_a[i_change:] += 2 * np.pi
        
        DR_tar = np.log((1 + np.exp(tar_r - R))) * np.tanh(5 * (np.pi * 0.25 - tar_a)) - np.log(2)
        Da = DR_tar - Rl * (tar_a - lane_width / R) * (1 - np.exp(np.minimum(0,5 * DR_tar))) ** 2
        
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
        lane_width = 3.5
        vehicle_length = 5 
        
        ego_x = path.ego[...,0]
        ego_y = path.ego[...,1]
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        
        # From location data
        R = self.Loc_data.R[domain.image_id]
        
        ego_frame_0 = np.nanargmin(np.abs(ego_a) + (ego_r > R) * 2 * np.pi)
        ego_a_change = np.where((ego_a[1:] < np.pi * 0.5) & (ego_a[:-1] > np.pi * 0.5))[0] + 1
        for i_change in ego_a_change:
            if i_change < ego_frame_0:
                ego_a[:i_change] -= 2 * np.pi
            else:
                ego_a[i_change:] += 2 * np.pi
                
           
        # This is an error flag designed to catch vehicles, whose roll is wrongly classified
        # Example. What could be v_3 is already inside the roundabout and then classified as 
        # either v_1 or v_2, which is an mistake     
        some_error = False
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
        else:
            v_1_x = path.v_1[...,0]
            v_1_y = path.v_1[...,1]
            v_1_r = np.sqrt(v_1_x ** 2 + v_1_y ** 2)
            v_1_a = np.angle(v_1_x + v_1_y * 1j)
            
            v_1_frame_0 = np.nanargmin(np.abs(v_1_a) + (v_1_r > R) * 2 * np.pi)
            v_1_a_change = np.where((v_1_a[1:] < np.pi * 0.5) & (v_1_a[:-1] > np.pi * 0.5))[0] + 1
            for i_change in v_1_a_change:
                if i_change < v_1_frame_0:
                    v_1_a[:i_change] -= 2 * np.pi
                else:
                    v_1_a[i_change:] += 2 * np.pi
             
            if not v_1_frame_0 < ego_frame_0:
                some_error = True
            
            
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
        else:
            v_2_x = path.v_2[...,0]
            v_2_y = path.v_2[...,1]
            v_2_r = np.sqrt(v_2_x ** 2 + v_2_y ** 2)
            v_2_a = np.angle(v_2_x + v_2_y * 1j)
            
            v_2_frame_0 = np.nanargmin(np.abs(v_2_a) + (v_2_r > R) * 2 * np.pi)
            if not v_2_frame_0 > ego_frame_0:
                some_error = True
            

        if some_error:
            in_position = np.zeros(len(ego_x), bool) 
        else:
            Rl = R - 0.5 * lane_width
            if isinstance(path.v_1, float):
                in_position = np.invert((ego_r > R) | (ego_r < R - 2 * lane_width))
            else:
                D_ego = np.log(1 / (1 + np.exp(ego_r - Rl))) * np.tanh(5 * ego_a) - Rl * ego_a
                D_v_1 = np.log(1 / (1 + np.exp(v_1_r - Rl))) * np.tanh(5 * v_1_a) - Rl * v_1_a
                
                D1 = D_ego - D_v_1 - vehicle_length
                
                Le = np.ones_like(D1) * lane_width
                
                out_of_position = ((ego_r > R) | (ego_r < R - 2 * lane_width)  |
                                   (v_1_a < 0) | 
                                   ((v_1_a < 0.5 * np.pi) & (v_1_y < lane_width + 0.5 * vehicle_length)))
                
                in_position = np.invert(out_of_position) & (D1 > D_class['rejected'] + Le)
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
        lane_width = 3.5
        vehicle_length = 5 
        
        
        ego_x = path.ego[...,0]
        ego_y = path.ego[...,1]
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        
        tar_x = path.tar[...,0]
        tar_y = path.tar[...,1]
        
        tar_r = np.sqrt(tar_x ** 2 + tar_y ** 2)
        tar_a = np.angle(tar_x + tar_y * 1j)
        
        
        # From location data
        R = self.Loc_data.R[domain.image_id]
        
        ego_frame_0 = np.nanargmin(np.abs(ego_a) + (ego_r > R) * 2 * np.pi)
        ego_a_change = np.where((ego_a[1:] < np.pi * 0.5) & (ego_a[:-1] > np.pi * 0.5))[0] + 1
        for i_change in ego_a_change:
            if i_change < ego_frame_0:
                ego_a[:i_change] -= 2 * np.pi
            else:
                ego_a[i_change:] += 2 * np.pi
        
        Rl = R - 0.5 * lane_width
        
        D_ego = np.log(1 / (1 + np.exp(ego_r - Rl))) * np.tanh(5 * ego_a) - Rl * ego_a
        
        tar_a_change = np.where((tar_a[1:] < np.pi * 0.5) & (tar_a[:-1] > np.pi * 0.5))[0] + 1
        for i_change in tar_a_change:
            tar_a[i_change:] += 2 * np.pi
            
        DR_tar = np.log((1 + np.exp(tar_r - R))) * np.tanh(5 * (np.pi * 0.25 - tar_a)) - np.log(2)
        D_tar = DR_tar - Rl * (tar_a - lane_width / R) * (1 - np.exp(np.minimum(0,5 * DR_tar))) ** 2
        
        Dc = D_ego - 0.5 * vehicle_length
        Da = D_tar
        
        # This is an error flag designed to catch vehicles, whose roll is wrongly classified
        # Example. What could be v_3 is already inside the roundabout and then classified as 
        # either v_1 or v_2, which is an mistake
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
            D1 = 1000 * np.ones_like(Dc)
        else:
            v_1_x = path.v_1[...,0]
            v_1_y = path.v_1[...,1]
            
            v_1_r = np.sqrt(v_1_x ** 2 + v_1_y ** 2)
            v_1_a = np.angle(v_1_x + v_1_y * 1j)
            
            if not np.isnan(v_1_a).all():
                v_1_frame_0 = np.nanargmin(np.abs(v_1_a) + (v_1_r > R) * 2 * np.pi)
                v_1_a_change = np.where((v_1_a[1:] < np.pi * 0.5) & (v_1_a[:-1] > np.pi * 0.5))[0] + 1
                for i_change in v_1_a_change:
                    if i_change < v_1_frame_0:
                        v_1_a[:i_change] -= 2 * np.pi
                    else:
                        v_1_a[i_change:] += 2 * np.pi
        
            D_v_1 = np.log(1 / (1 + np.exp(v_1_r - Rl))) * np.tanh(5 * v_1_a) - Rl * v_1_a
            D1 = D_ego - D_v_1 - vehicle_length
        
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
            D2 = 1000 * np.ones_like(Dc)
        else:
            v_2_x = path.v_2[...,0]
            v_2_y = path.v_2[...,1]
            
            v_2_r = np.sqrt(v_2_x ** 2 + v_2_y ** 2)
            v_2_a = np.angle(v_2_x + v_2_y * 1j)
            
            if not np.isnan(v_2_a).all():
                v_2_frame_0 = np.nanargmin(np.abs(v_2_a) + (v_2_r > R) * 2 * np.pi)
                v_2_a_change = np.where((v_2_a[1:] < np.pi * 0.5) & (v_2_a[:-1] > np.pi * 0.5))[0] + 1
                for i_change in v_2_a_change:
                    if i_change < v_2_frame_0:
                        v_2_a[:i_change] -= 2 * np.pi
                    else:
                        v_2_a[i_change:] += 2 * np.pi
                        
            D_v_2 = np.log(1 / (1 + np.exp(v_2_r - Rl))) * np.tanh(5 * v_2_a) - Rl * v_2_a
            D2 = D_v_2 - D_ego - vehicle_length
        
        
        if isinstance(path.v_3, float):
            assert str(path.v_3) == 'nan'
            D3 = 1000 * np.ones_like(Dc)
        else:
            v_3_x = path.v_3[...,0]
            v_3_y = path.v_3[...,1]
        
            v_3_r = np.sqrt(v_3_x ** 2 + v_3_y ** 2)
            v_3_a = np.angle(v_3_x + v_3_y * 1j)
        
            if not np.isnan(v_3_a).all():
                v_3_a_change = np.where((v_3_a[1:] < np.pi * 0.5) & (v_3_a[:-1] > np.pi * 0.5))[0] + 1
                for i_change in v_3_a_change:
                    v_3_a[i_change:] += 2 * np.pi
            
            
            DR_v_3 = np.log((1 + np.exp(v_3_r - R))) * np.tanh(5 * (np.pi * 0.25 - v_3_a)) - np.log(2)
            D_v_3 = DR_v_3 - Rl * (v_3_a - lane_width / R) * (1 - np.exp(np.minimum(0,5 * DR_v_3))) ** 2
        
            D3 = D_tar - D_v_3 - vehicle_length
        
        Dadt = np.interp(t, t[5:], (Da[5:] - Da[:-5]) / (t[5:] - t[:-5]))
        Dcdt = np.interp(t, t[5:], (Dc[5:] - Dc[:-5]) / (t[5:] - t[:-5]))
        
        Dv = np.maximum(Dadt - Dcdt, 0)
        Te = Dv / 2 # assume acceleration of up to 2m/s^2
        Lt = 0.5 * 2 * Te ** 2 - Dadt * Te # assume again acceleration of up to 2m/s^2
        Lt = np.clip(Lt, lane_width, None)
        Le = np.ones_like(Dc) * lane_width
        
        # repair 
        if np.isnan(D1).any():
            if np.isfinite(D1).any():
                D1 = np.interp(t, t[np.isfinite(D1)], D1[np.isfinite(D1)])
            else:
                D1 = 1000 * np.ones_like(Dc)
        if np.isnan(D2).any():
            if np.isfinite(D2).any():
                D2 = np.interp(t, t[np.isfinite(D2)], D2[np.isfinite(D2)])
            else:
                D2 = 1000 * np.ones_like(Dc)
        if np.isnan(D3).any():
            if np.isfinite(D3).any():
                D3 = np.interp(t, t[np.isfinite(D3)], D3[np.isfinite(D3)])
            else:
                D3 = 1000 * np.ones_like(Dc)
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
    
    
    def fill_empty_path(self, path, t, domain, agent_types, size):
        R = self.Loc_data.R[domain.image_id]
        # check vehicle v_1 (in front of ego)
        if isinstance(path.v_1, float):
            assert str(path.v_1) == 'nan'
        else:
            path.v_1 = self.extrapolate_path(path.v_1, t, mode='vel')
            
        if isinstance(path.v_2, float):
            assert str(path.v_2) == 'nan'
        else:
            path.v_2 = self.extrapolate_path(path.v_2, t, mode='vel')
            
        if isinstance(path.v_3, float):
            assert str(path.v_3) == 'nan'
        else:
            path.v_3 = self.extrapolate_path(path.v_3, t, mode='vel')
            
        
        # check vehicle v_4 (pedestrian)
        if isinstance(path.v_4, float):
            assert str(path.v_4) == 'nan'
        else:
            v_4_rewrite = np.isnan(path.v_4[:,0])
            if v_4_rewrite.any():
                path.v_4 = self.extrapolate_path(path.v_4, t, mode = 'pos')
                
        
        # look for other participants
        n_I = self.num_timesteps_in_real

        tar_pos = path.tar[np.newaxis, :, :2] # shape (1, n_I, 2)
        
        help_pos = []
        for agent in path.index:
            if isinstance(path[agent], float):
                assert str(path[agent]) == 'nan'
                continue
            if agent[2:] == 'tar':
                continue
            help_pos.append(path[agent])
            
        help_pos = np.stack(help_pos, axis = 0)[..., :2] # shape (n_non_tar, n_I, 2)
        
        tar_frames = 25 * (t + domain.t_0)
        
        # Load data if necessary
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'RounD_round_about' + os.sep + 'RounD_processed.pkl')
            self.Data = self.Data.reset_index(drop = True) 
        
        
        Neighbor = self.Data.iloc[domain.track_id].otherVehicles
        Neighbor_type = np.array(self.Data.iloc[Neighbor]['class'])
        frames_help = np.concatenate([[tar_frames[0] - 1], tar_frames]).astype(int)

        num_data = len(self.path_data_info())
        # search for vehicles
        Pos = np.ones((len(Neighbor), len(frames_help), num_data)) * np.nan
        Size = np.ones((len(Neighbor), 2)) * np.nan
        for i, n in enumerate(Neighbor):
            track_n = self.Data.iloc[n].track.set_index('frame')
            track_n.index = track_n.index.astype(int)
            track_n = track_n.reindex(frames_help)[['xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]
            track_n = track_n.rename(columns={"xCenter": "x", "yCenter": "y",
                                              "xVelocity" : "v_x", "yVelocity" : "v_y",
                                              "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            track_n = rotate_track(track_n, domain.rot_angle, np.array([[domain.x_center, domain.y_center]]))
            Pos[i] = track_n.to_numpy()[:,:6]

            if Neighbor_type[i] == 'pedestrian':
                Size[i] = 0.5
            else:
                Size[i] = [self.Data.iloc[n].length, self.Data.iloc[n].width]
        
        # GET EXISTING VEHICLES
        actually_there = np.isfinite(Pos[:,1:n_I + 1]).any((1,2))
        Neighbor_type = Neighbor_type[actually_there]
        Pos           = Pos[actually_there]
        Size          = Size[actually_there]

        # Filter out vehicles to far away
        D_help = np.nanmin(np.sqrt(((Pos[np.newaxis, :,1:n_I + 1,:2] - help_pos[:,np.newaxis,:n_I]) ** 2).sum(-1)), -1).min(0)
        actually_interesting = (D_help > 0.1) & (D_help < 100)
        Neighbor_type = Neighbor_type[actually_interesting]
        Pos           = Pos[actually_interesting]
        Size          = Size[actually_interesting]
        
        # filter out nonmoving vehicles
        actually_moving = (np.nanmax(Pos, 1) - np.nanmin(Pos, 1))[...,:2].max(1) > 0.1
        Neighbor_type = Neighbor_type[actually_moving]
        Pos           = Pos[actually_moving]
        Size          = Size[actually_moving]
        
        # Find cars that could influence tar vehicle
        D = np.nanmin(np.sqrt(((Pos[:,1:n_I + 1,:2] - tar_pos[:,:n_I]) ** 2).sum(-1)), -1)
        Neighbor_type = Neighbor_type[D < 75]
        Pos           = Pos[D < 75]
        Size          = Size[D < 75]
        D             = D[D < 75]
        
        # sort by closest vehicle
        Pos           = Pos[np.argsort(D)]
        Neighbor_type = Neighbor_type[np.argsort(D)]
        Size          = Size[np.argsort(D)]
        
        if self.max_num_addable_agents is not None:
            Pos           = Pos[:self.max_num_addable_agents]
            Neighbor_type = Neighbor_type[:self.max_num_addable_agents]
            Size          = Size[:self.max_num_addable_agents]

        # Remove extra timestep
        Pos = Pos[:,1:]
        
        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i + 5)
                
            u = np.isfinite(pos[:,0])
            if u.sum() > 1:
                path[name] = self.extrapolate_path(pos, t, mode='vel')
                size[name] = Size[i]
                
                if Neighbor_type[i] == 'pedestrian':
                    agent_types[name] = 'P'
                elif Neighbor_type[i] == 'bicycle':
                    agent_types[name] = 'B'
                elif Neighbor_type[i] == 'motorcycle':
                    agent_types[name] = 'M'
                else:
                    agent_types[name] = 'V'
            
            
        return path, agent_types, size
            
    
    def provide_map_drawing(self, domain):
        R = self.Loc_data.R[domain.image_id]
        r = self.Loc_data.r[domain.image_id]
        
        x = np.arange(-1,1,501)[:,np.newaxis]
        unicircle_upper = np.concatenate((x, np.sqrt(1 - x ** 2)), axis = 1)
        unicircle_lower = np.concatenate((- x, - np.sqrt(1 - x ** 2)), axis = 1)
        
        unicircle = np.concatenate((unicircle_upper, unicircle_lower[1:, :]))
        
        lines_solid = []
        lines_solid.append(unicircle * r)
        
        lines_dashed = []
        lines_dashed.append(unicircle * R)
        
        return lines_solid, lines_dashed
    
    
    def get_name(self = None):
        names = {'print': 'RounD (roundabout)',
                 'file': 'Roundabout',
                 'latex': r'\emph{RounD}'}
        return names
    
    def future_input(self = None):
        return False
    
    def includes_images(self = None):
        return True 
    
    def includes_sceneGraphs(self = None):
        return True
