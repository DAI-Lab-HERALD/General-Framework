import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_direction import scenario_direction
import os
from PIL import Image

# Map import
import copy
import torch
import lanelet2
from av2.geometry.interpolate import compute_midpoint_line
from lanelet2.projection import UtmProjector
from pyproj import Proj, transform


projector = UtmProjector(lanelet2.io.Origin(0, 0))
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                              lanelet2.traffic_rules.Participants.Vehicle)

pd.options.mode.chained_assignment = None

class InD_direction(data_set_template):
    '''
    The inD dataset is extracted from drone recordings of real world traffic over
    four german intersections. This specific instance focuses on the behavior of
    vehicles when entering the intersection, and the challange lies in predicting
    the direction which they are about to take (left, straight, right or staying).
    
    The dataset can be found at https://www.ind-dataset.com/format and the following
    citation can be used:
        
    Bock, J., Krajewski, R., Moers, T., Runde, S., Vater, L., & Eckstein, L. (2020, October). 
    The ind dataset: A drone dataset of naturalistic road user trajectories at german 
    intersections. In 2020 IEEE Intelligent Vehicles Symposium (IV) (pp. 1929-1934). IEEE.
    '''

    def get_lane_graph(self, map_path, UtmOrigin):
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
            if 'subtype' not in ll.attributes:
                continue

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
            elif ll.attributes['subtype'] == 'bus_lane': 
                lane_type_val = 'BUS'
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
                                    'InD_direction' + os.sep + 'InD_processed.pkl')
            
        unique_rec = np.unique(self.Data[['recordingId','locationId']], axis = 0)
        self.Rec_loc = pd.Series(unique_rec[:,1], index = unique_rec[:,0])

        unique_map = [1, 2, 3, 4] # TODO: Change this
        self.Loc_rec = {1: 7, 2: 18, 3: 30, 4: 0}
        
        Loc_data_pix = pd.DataFrame(np.zeros((len(unique_map),4),float), columns = ['locationId', 'center_x', 'center_y', 'streets'])
        # Center is choosen, so that it lies at teh crossing between the dividers of incoming and outcoming lanes
        # at the entrance of the crossing
        
        Loc_data_pix.locationId = unique_map
        Loc_data_pix = Loc_data_pix.set_index('locationId')
        
        # Location 1
        Loc_data_pix.center_x.loc[1], Loc_data_pix.center_y.loc[1] = 564, -335
        streets_1 = pd.DataFrame(np.zeros((4,7), float), columns = ['with_priority', 
                                                                    'entry_x', 'entry_y', 
                                                                    'at_cross_x', 'at_cross_y',
                                                                    'target_x', 'target_y'])
        streets_1.with_priority = streets_1.with_priority.astype(object)
        # start at 0 heading, go counterclockwise
        streets_1.with_priority.iloc[0] = [1,3]
        streets_1.entry_x.iloc[0],      streets_1.entry_y.iloc[0]       = 703, -179
        streets_1.at_cross_x.iloc[0],   streets_1.at_cross_y.iloc[0]    = 590, -282
        streets_1.target_x.iloc[0],     streets_1.target_y.iloc[0]       = 657, -361
        streets_1.with_priority.iloc[1] = []
        streets_1.entry_x.iloc[1],      streets_1.entry_y.iloc[1]       = 337,  -59
        streets_1.at_cross_x.iloc[1],   streets_1.at_cross_y.iloc[1]    = 529, -272
        streets_1.target_x.iloc[1],     streets_1.target_y.iloc[1]       = 555, -243
        streets_1.with_priority.iloc[2] = [1,3]
        streets_1.entry_x.iloc[2],      streets_1.entry_y.iloc[2]       = 425, -493
        streets_1.at_cross_x.iloc[2],   streets_1.at_cross_y.iloc[2]    = 541, -376
        streets_1.target_x.iloc[2],     streets_1.target_y.iloc[2]       = 481, -310
        streets_1.with_priority.iloc[3] = []
        streets_1.entry_x.iloc[3],      streets_1.entry_y.iloc[3]       = 809, -620
        streets_1.at_cross_x.iloc[3],   streets_1.at_cross_y.iloc[3]    = 603, -396
        streets_1.target_x.iloc[3],     streets_1.target_y.iloc[3]       = 579, -421
        Loc_data_pix.streets.iloc[0] = streets_1
        
        Loc_data_pix.center_x.loc[2], Loc_data_pix.center_y.loc[2] = 483, -306
        streets_2 = pd.DataFrame(np.zeros((4,7), float), columns = ['with_priority', 
                                                                    'entry_x', 'entry_y', 
                                                                    'at_cross_x', 'at_cross_y',
                                                                    'target_x', 'target_y'])
        streets_2.with_priority = streets_2.with_priority.astype('str')
        # start at 0 heading, go counterclockwise
        streets_2.with_priority.iloc[0] = [1]
        streets_2.entry_x.iloc[0],      streets_2.entry_y.iloc[0]       = 990, -220
        streets_2.at_cross_x.iloc[0],   streets_2.at_cross_y.iloc[0]    = 585, -291
        streets_2.target_x.iloc[0],     streets_2.target_y.iloc[0]       = 592, -325
        streets_2.with_priority.iloc[1] = [2]
        streets_2.entry_x.iloc[1],      streets_2.entry_y.iloc[1]       = 390, -124
        streets_2.at_cross_x.iloc[1],   streets_2.at_cross_y.iloc[1]    = 502, -238
        streets_2.target_x.iloc[1],     streets_2.target_y.iloc[1]       = 548, -228
        streets_2.with_priority.iloc[2] = [3]
        streets_2.entry_x.iloc[2],      streets_2.entry_y.iloc[2]       =  98, -394
        streets_2.at_cross_x.iloc[2],   streets_2.at_cross_y.iloc[2]    = 394, -321
        streets_2.target_x.iloc[2],     streets_2.target_y.iloc[2]       = 383, -296
        streets_2.with_priority.iloc[3] = [0]
        streets_2.entry_x.iloc[3],      streets_2.entry_y.iloc[3]       = 563, -569
        streets_2.at_cross_x.iloc[3],   streets_2.at_cross_y.iloc[3]    = 476, -345
        streets_2.target_x.iloc[3],     streets_2.target_y.iloc[3]       = 545, -337
        Loc_data_pix.streets.iloc[1] = streets_2
        
        Loc_data_pix.center_x.loc[3], Loc_data_pix.center_y.loc[3] = 430, -262
        streets_3 = pd.DataFrame(np.zeros((4,7), float), columns = ['with_priority', 
                                                                    'entry_x', 'entry_y', 
                                                                    'at_cross_x', 'at_cross_y',
                                                                    'target_x', 'target_y'])
        streets_3.with_priority = streets_3.with_priority.astype('str')
        # start at 0 heading, go counterclockwise
        streets_3.with_priority.iloc[0] = [1,3]
        streets_3.entry_x.iloc[0],      streets_3.entry_y.iloc[0]       = 751, -129
        streets_3.at_cross_x.iloc[0],   streets_3.at_cross_y.iloc[0]    = 454, -223
        streets_3.target_x.iloc[0],     streets_3.target_y.iloc[0]       = 625, -352
        streets_3.with_priority.iloc[1] = []
        streets_3.entry_x.iloc[1],      streets_3.entry_y.iloc[1]       = 105,  -20
        streets_3.at_cross_x.iloc[1],   streets_3.at_cross_y.iloc[1]    = 375, -221
        streets_3.target_x.iloc[1],     streets_3.target_y.iloc[1]       = 402, -180
        # streets_3[2] is imagined
        streets_3.with_priority.iloc[3] = []
        streets_3.entry_x.iloc[3],      streets_3.entry_y.iloc[3]       = 851, -578
        streets_3.at_cross_x.iloc[3],   streets_3.at_cross_y.iloc[3]    = 608, -395
        streets_3.target_x.iloc[3],     streets_3.target_y.iloc[3]       = 566, -457
        Loc_data_pix.streets.iloc[2] = streets_3
        
        Loc_data_pix.center_x.loc[4], Loc_data_pix.center_y.loc[4] = 928, -390
        streets_4 = pd.DataFrame(np.zeros((4,7), float), columns = ['with_priority', 
                                                                    'entry_x', 'entry_y', 
                                                                    'at_cross_x', 'at_cross_y',
                                                                    'target_x', 'target_y'])
        streets_4.with_priority = streets_4.with_priority.astype('str')
        # start at 0 heading, go counterclockwise
        streets_4.with_priority.iloc[0] = []
        streets_4.entry_x.iloc[0],      streets_4.entry_y.iloc[0]       = 1192, -272
        streets_4.at_cross_x.iloc[0],   streets_4.at_cross_y.iloc[0]    = 1079, -313
        streets_4.target_x.iloc[0],     streets_4.target_y.iloc[0]       = 1102, -377
        streets_4.with_priority.iloc[1] = [0,2]
        streets_4.entry_x.iloc[1],      streets_4.entry_y.iloc[1]       = 922,   -92
        streets_4.at_cross_x.iloc[1],   streets_4.at_cross_y.iloc[1]    = 900,  -341
        streets_4.target_x.iloc[1],     streets_4.target_y.iloc[1]       = 1096, -258
        streets_4.with_priority.iloc[2] = []
        streets_4.entry_x.iloc[2],      streets_4.entry_y.iloc[2]       = 365,  -690
        streets_4.at_cross_x.iloc[2],   streets_4.at_cross_y.iloc[2]    = 826,  -448
        streets_4.target_x.iloc[2],     streets_4.target_y.iloc[2]       = 793,  -382
        # streets_3[3] is imagined
        Loc_data_pix.streets.iloc[3] = streets_4
        
        # Attention: No deep copy of the pandas dataframe in streets, so be careful
        self.Loc_data = pd.DataFrame(np.zeros((len(self.Rec_loc), 3),object), index = self.Rec_loc.index, columns = Loc_data_pix.columns)
        
        # You cannot rely on the values given for orthoPxToMeter
        self.Loc_scale = {}
        for rec_Id in self.Rec_loc.index:
            loc_Id = self.Rec_loc[rec_Id]
            Meta_data=pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
                                  'InD_direction' + os.sep + 
                                  'data' + os.sep + '{}_recordingMeta.csv'.format(str(rec_Id).zfill(2)))
            
            self.Loc_scale[rec_Id] = Meta_data['orthoPxToMeter'][0] * 12
        
            streets_pix = Loc_data_pix.streets.loc[loc_Id]
            streets = pd.DataFrame(np.zeros(streets_pix.shape, object), columns = streets_pix.columns, index = streets_pix.index)
            streets.iloc[:,1:] = streets_pix.iloc[:,1:] * self.Loc_scale[rec_Id]
            streets.iloc[:,0]  = streets_pix.iloc[:,0] 
            self.Loc_data.loc[rec_Id].streets = streets
            self.Loc_data.center_x.loc[rec_Id] = Loc_data_pix.center_x.loc[loc_Id] * self.Loc_scale[rec_Id]
            self.Loc_data.center_y.loc[rec_Id] = Loc_data_pix.center_y.loc[loc_Id] * self.Loc_scale[rec_Id]
            
    def set_scenario(self):
        self.scenario = scenario_direction()
        self.analyze_maps()
    
    def path_data_info(self = None):
        return ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
        
   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'InD_direction' + os.sep + 'InD_processed.pkl')
        # analize raw dara 
        self.Data = self.Data.reset_index(drop = True)
        num_samples_max = len(self.Data)
        self.Path = []
        self.Type_old = []
        self.Size_old = []
        self.T = []
        self.Domain_old = []
    
        # Create Images
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
                        'InD_direction' + os.sep + 'data' + os.sep + 
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
        

        location_to_lanelet_file = {1: '01_bendplatz', 
                                    2: '02_frankenburg', 
                                    3: '03_heckstrasse',
                                    4: '04_aseag'}
        
        # Go though unique locations
        unique_loc = np.unique(self.Rec_loc)
        for loc_Id in unique_loc:
            ## Get SceneGraph
            lanelet_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                            'InD_direction' + os.sep + 'maps' + os.sep + 
                            'lanelets' + os.sep + location_to_lanelet_file[loc_Id] + os.sep +
                            'location' + str(loc_Id) + '.osm')
            rec_Id = self.Rec_loc[self.Rec_loc == loc_Id].index[0]

            # Get the UTM origin
            meta_file = (self.path + os.sep + 'Data_sets' + os.sep +
                         'InD_direction' + os.sep + 'data' + os.sep + 
                         str(rec_Id).zfill(2) + '_recordingMeta.csv')
            meta_data = pd.read_csv(meta_file).iloc[0]
            origin = np.array([[meta_data.xUtmOrigin, meta_data.yUtmOrigin]])

            # load lanelet map
            SceneGraphs_unturned.loc[loc_Id] = self.get_lane_graph(lanelet_file, origin)


        # extract raw samples
        self.num_samples = 0
        self.Data['street_in'] = 0
        self.Data['street_out'] = 0
        self.Data['behavior'] = '0'
        
        for i in range(num_samples_max):
            # to keep track:
            if np.mod(i,100) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
                print('found cases: ' + str(self.num_samples))
                print('')

            agent_i = self.Data.iloc[i]
            
            # assume i is the tar vehicle, which has to be a motor vehicle
            if agent_i['class'] in ['bicycle', 'pedestrian']:
                continue
            
            track_i = agent_i.track[['frame','xCenter','yCenter','xVelocity','yVelocity','xAcceleration','yAcceleration','heading']]
            track_i = track_i.rename(columns={"xCenter": "x", "yCenter": "y",
                                              "xVelocity" : "v_x", "yVelocity" : "v_y",
                                              "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            
            streets_i = self.Loc_data.streets.loc[agent_i.locationId]
            
            # Determine the action of the vehicle
            if self.Data.iloc[i].behavior == '0':
                agent_i.street_in, agent_i.street_out, agent_i.behavior = determine_streets(track_i, streets_i)
                self.Data.street_in.iloc[i], self.Data.street_out.iloc[i], self.Data.behavior.iloc[i] = agent_i.street_in, agent_i.street_out, agent_i.behavior
            
            entry = streets_i.loc[agent_i.street_in][['entry_x', 'entry_y']].to_numpy()
            center = streets_i.loc[agent_i.street_in][['at_cross_x', 'at_cross_y']].to_numpy()
            
            diff = entry - center
            
            angle = np.angle(diff[0] + 1j * diff[1])     
            Rot_center = center[np.newaxis]
            track_i = rotate_track(track_i, angle, Rot_center)
            
            # Check if the vehicle is already leaving
            if not (135 < track_i.heading.iloc[0] < 225):
                continue
            
            path = pd.Series(np.zeros(0, object), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            sizes = pd.Series(np.zeros(0, object), index = [])
            
            path['tar'] = track_i.to_numpy()[:,1:7]
            agent_types['tar'] = 'V'
            sizes['tar'] = np.array([agent_i.length, agent_i.width])
            
            t = np.array(track_i.frame / 25)
            
            # collect domain data
            domain = pd.Series(np.zeros(11, object), index = ['location', 'splitting', 'image_id', 'graph_id', 'rot_angle', 'x_center', 'y_center', 
                                                             'behavior', 'class', 'neighbor_veh', 'neighbor_ped'])
            domain.location    = agent_i.locationId
            if np.mod(i,5) == 0:
                domain.splitting = 'test'
            else:
                domain.splitting = 'train'
            domain.image_id    = agent_i.recordingId
            domain.graph_id    = len(self.Domain_old)
            domain.rot_angle   = angle
            domain.x_center    = center[0]
            domain.y_center    = center[1]
            domain['class']    = agent_i['class']
            domain['behavior'] = agent_i['behavior']
            domain['old_id']   = agent_i.trackId
            
            # Divide neighbors by class
            neighbor_id    = agent_i.otherVehicles
            neighbor_class = self.Data.loc[neighbor_id]['class']
            
            domain['neighbor_ped'] = neighbor_id[(neighbor_class == 'pedestrian')]      
            domain['neighbor_byc'] = neighbor_id[(neighbor_class == 'bicycle')]     
            domain['neighbor_mtc'] = neighbor_id[(neighbor_class == 'motorcycle')]
            domain['neighbor_veh'] = neighbor_id[(neighbor_class != 'pedestrian') & (neighbor_class != 'bicycle') & (neighbor_class != 'motorcycle')]

            # Get local SceneGraph, by rotating the respective positions and boundaries
            graph = SceneGraphs_unturned.loc[domain.location].copy(deep = True)
            centerlines_new = []
            left_boundaries_new = []
            right_boundaries_new = []
            for i in range(len(graph.centerlines)):
                c = graph.centerlines[i].copy()
                l = graph.left_boundaries[i].copy()
                r = graph.right_boundaries[i].copy()
                centerlines_new.append(rotate_track_array(c, angle, Rot_center))
                left_boundaries_new.append(rotate_track_array(l, angle, Rot_center))
                right_boundaries_new.append(rotate_track_array(r, angle, Rot_center)) 
        
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
        pos = path.tar[...,:2]
        
        # Get streets
        streets_old = self.Loc_data.loc[domain.location].streets.iloc[:,1:].to_numpy().reshape(4,3,2).astype(float)
        Rot_matrix = np.array([[np.cos(domain.rot_angle), -np.sin(domain.rot_angle)],
                               [np.sin(domain.rot_angle), np.cos(domain.rot_angle)]])
        center = np.array([[[domain.x_center, domain.y_center]]])
        
        streets = np.dot((streets_old - center), Rot_matrix)
        street_in = np.argmin((streets[:,1] ** 2).sum(-1))
        
        # Reorder streets:
        streets = streets[np.mod(np.arange(len(streets)) + street_in, len(streets))]
        useless_streets = (streets[:,0] == streets[:,1]).all(1)
        useless_streets[0] = True 
        # Distance to staying is not required as weel
        
        Dt = (streets[:,2] - streets[:,1])[np.newaxis,np.newaxis]
        Dt /= np.linalg.norm(Dt, axis = -1, keepdims = True) + 1e-6
        Dtt = np.stack([Dt[:,:,:,1], -Dt[:,:,:,0]], -1)
        
        De = (streets[:,0] - streets[:,1])[np.newaxis,np.newaxis]
        Dp = pos[:,:,np.newaxis] - streets[np.newaxis,np.newaxis,:,1,:]
        
        sides = np.sign((De * Dtt).sum(-1)) # 1 if both xE and XP are on the same side of the line
        
        D = - sides * (Dtt * Dp).sum(-1)
        D = D.transpose(2,0,1)
        
        D[useless_streets] = 500
        
        Dist = pd.Series(list(D), index = self.Behaviors)
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
        pos = path.tar[...,:2]
        
        # Get streets
        streets_old = self.Loc_data.loc[domain.location].streets.iloc[:,1:].to_numpy().reshape(4,3,2).astype(float)
        Rot_matrix = np.array([[np.cos(domain.rot_angle), -np.sin(domain.rot_angle)],
                               [np.sin(domain.rot_angle), np.cos(domain.rot_angle)]])
        center = np.array([[[domain.x_center, domain.y_center]]])
        
        streets = np.dot((streets_old - center), Rot_matrix)
        street_in = np.argmin((streets[:,1] ** 2).sum(-1))
        
        # Reorder streets:
        streets = streets[np.mod(np.arange(len(streets)) + street_in, len(streets))]
        useless_streets = (streets[:,0] == streets[:,1]).all(1)
        useless_streets[0] = True 
        # Distance to staying is not required as weel
        
        Dt = (streets[:,2] - streets[:,1])[np.newaxis]
        Dt /= np.linalg.norm(Dt, axis = -1, keepdims = True) + 1e-6
        Dtt = np.stack([Dt[:,:,1], -Dt[:,:,0]], -1)
        
        De = (streets[:,0] - streets[:,1])[np.newaxis]
        Dp = pos[:,np.newaxis] - streets[np.newaxis,:,1,:]
        
        sides = np.sign((De * Dtt).sum(-1)) # 1 if both xE and XP are on the same side of the line
        
        D = - sides * (Dtt * Dp).sum(-1)
        
        in_position = ~((D[:,~useless_streets] < 0) & (D[:,[0]] < 0)).any(-1)
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
        pos = path.tar[...,:2]
        
        # Get streets
        streets_old = self.Loc_data.loc[domain.location].streets.iloc[:,1:].to_numpy().reshape(4,3,2).astype(float)
        Rot_matrix = np.array([[np.cos(domain.rot_angle), -np.sin(domain.rot_angle)],
                               [np.sin(domain.rot_angle), np.cos(domain.rot_angle)]])
        center = np.array([[[domain.x_center, domain.y_center]]])
        
        streets = np.dot((streets_old - center), Rot_matrix)
        street_in = np.argmin((streets[:,1] ** 2).sum(-1))
        
        # Reorder streets:
        street = streets[street_in]
        # Distance to staying is not required as weel
        
        Dt = (street[2] - street[1])[np.newaxis]
        Dt /= np.linalg.norm(Dt, axis = -1, keepdims = True) + 1e-6
        Dtt = np.stack([Dt[:,1], -Dt[:,0]], -1)
        
        De = (street[0] - street[1])[np.newaxis]
        Dp = pos - street[[1],:]
        
        sides = np.sign((De * Dtt).sum(-1)) # 1 if both xE and XP are on the same side of the line
        
        D = - sides * (Dtt * Dp).sum(-1)

        # repair 
        if np.isnan(D).any():
            if np.isfinite(D).any():
                D = np.interp(t, t[np.isfinite(D)], D1[np.isfinite(D)])
            else:
                D = 1000 * np.ones_like(D)

        Dist = pd.Series([-D], index = ['D_decision'])
        return Dist
    
    
    def fill_empty_path(self, path, t, domain, agent_types, size):
        # Get old data, so that removed pedestrians can be accessed
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'InD_direction' + os.sep + 'InD_processed.pkl')
            self.Data = self.Data.reset_index(drop = True)
        
        n_I = self.num_timesteps_in_real

        tar_pos = path.tar[np.newaxis,:,:2] # shape (1, n_I, 2)
        I_t = t + domain.t_0
        t_help = np.concatenate([[I_t[0] - self.dt], I_t]) # shape (n_I + 1,)
        frames_help = (t_help * 25).astype(int)

        num_data = len(self.path_data_info())
        # search for vehicles
        Neighbor_veh = domain.neighbor_veh.copy()
        Pos_veh = np.ones((len(Neighbor_veh), len(I_t) + 1, num_data)) * np.nan # shape (n_veh, n_I + 1, 6)
        Size_veh = np.ones((len(Neighbor_veh), 2)) * np.nan
        for i, n in enumerate(Neighbor_veh):
            track_n = self.Data.loc[n].track.set_index('frame')
            track_n.index = track_n.index.astype(int)
            track_n = track_n.reindex(frames_help)[['xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]
            # Rename columns
            track_n = track_n.rename(columns={"xCenter": "x", "yCenter": "y",
                                              "xVelocity" : "v_x", "yVelocity" : "v_y",
                                              "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            
            # Rotate track 
            track_n = rotate_track(track_n, domain.rot_angle, np.array([domain.x_center, domain.y_center]))
            Pos_veh[i] = track_n.to_numpy()

            Size_veh[i] = [self.Data.loc[n].length, self.Data.loc[n].width]
        
        # get vehicles with actually recorded steps
        finite_vehicle = np.isfinite(Pos_veh[:,1:n_I + 1]).any((1,2))
        Pos_veh = Pos_veh[finite_vehicle] # shape (n_veh, n_I + 1, 6)
        Size_veh = Size_veh[finite_vehicle] # shape (n_veh, 2)
        
        # filter our parking vehicles
        actually_moving = (np.nanmax(Pos_veh, 1) - np.nanmin(Pos_veh, 1))[...,:2].max(1) > 0.1
        Pos_veh  = Pos_veh[actually_moving]
        Size_veh = Size_veh[actually_moving]
        
        # search for pedestrians
        Neighbor_ped = domain.neighbor_ped.copy()
        Pos_ped = np.ones((len(Neighbor_ped), len(I_t) + 1, num_data)) * np.nan
        for i, n in enumerate(Neighbor_ped):
            track_n = self.Data.loc[n].track.set_index('frame')
            track_n.index = track_n.index.astype(int)
            track_n = track_n.reindex(frames_help)[['xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]

            # Rename columns
            track_n = track_n.rename(columns={"xCenter": "x", "yCenter": "y",
                                              "xVelocity" : "v_x", "yVelocity" : "v_y",
                                              "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            
            # Rotate track 
            track_n = rotate_track(track_n, domain.rot_angle, np.array([domain.x_center, domain.y_center]))
            Pos_ped[i] = track_n.to_numpy()
        
        Pos_ped = Pos_ped[np.isfinite(Pos_ped[:,1:n_I + 1]).any((1,2))]
        Size_ped = np.ones((len(Pos_ped), 2)) * 0.5
        
        # search for bicycles
        Neighbor_byc = domain.neighbor_byc.copy()
        Pos_byc = np.ones((len(Neighbor_byc), len(I_t + 1), num_data)) * np.nan
        Size_byc = np.ones((len(Neighbor_byc), 2)) * np.nan
        for i, n in enumerate(Neighbor_byc):
            track_n = self.Data.loc[n].track.set_index('frame')
            track_n.index = track_n.index.astype(int)
            track_n = track_n.reindex(frames_help)[['xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]

            # Rename columns
            track_n = track_n.rename(columns={"xCenter": "x", "yCenter": "y",
                                              "xVelocity" : "v_x", "yVelocity" : "v_y",
                                              "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            
            # Rotate track 
            track_n = rotate_track(track_n, domain.rot_angle, np.array([domain.x_center, domain.y_center]))
            Pos_byc[i] = track_n.to_numpy()
            
            Size_byc[i] = [self.Data.loc[n].length, self.Data.loc[n].width]
        
        use_byc = np.isfinite(Pos_byc[:,1:n_I + 1]).any((1,2))
        Pos_byc  = Pos_byc[use_byc]
        Size_byc = Size_byc[use_byc]

        # Search for motorcyclists
        Neighbor_mtc = domain.neighbor_mtc.copy()
        Pos_mtc = np.ones((len(Neighbor_mtc), len(I_t + 1), num_data)) * np.nan
        Size_mtc = np.ones((len(Neighbor_mtc), 2)) * np.nan
        for i, n in enumerate(Neighbor_mtc):
            track_n = self.Data.loc[n].track.set_index('frame')
            track_n.index = track_n.index.astype(int)
            track_n = track_n.reindex(frames_help)[['xCenter', 'yCenter', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration']]

            # Rename columns
            track_n = track_n.rename(columns={"xCenter": "x", "yCenter": "y",
                                              "xVelocity" : "v_x", "yVelocity" : "v_y",
                                              "xAcceleration" : "a_x", "yAcceleration" : "a_y"}).copy(deep = True)
            
            # Rotate track 
            track_n = rotate_track(track_n, domain.rot_angle, np.array([domain.x_center, domain.y_center]))
            Pos_mtc[i] = track_n.to_numpy()
            
            Size_mtc[i] = [self.Data.loc[n].length, self.Data.loc[n].width]

        use_mtc = np.isfinite(Pos_mtc[:,1:n_I + 1]).any((1,2))
        Pos_mtc  = Pos_mtc[use_mtc]
        Size_mtc = Size_mtc[use_mtc]
        
        Pos = np.concatenate((Pos_veh, Pos_ped, Pos_byc, Pos_mtc), axis = 0)
        Size = np.concatenate((Size_veh, Size_ped, Size_byc, Size_mtc), axis = 0)
        Type = np.zeros(len(Pos))
        Type[:len(Pos_veh)] = 1
        Type[len(Pos_veh):len(Pos_veh) + len(Pos_ped)] = 2
        Type[len(Pos_veh) + len(Pos_ped):len(Pos_veh) + len(Pos_ped) + len(Pos_byc)] = 3
        
        # Get closest agents
        D = np.nanmin((np.sqrt((Pos[:,1:n_I + 1,:2] - tar_pos[:,:n_I]) ** 2).sum(-1)), -1)
        Pos  = Pos[np.argsort(D)]
        Size = Size[np.argsort(D)]
        Type = Type[np.argsort(D)]
        
        if self.max_num_addable_agents is not None:
            Pos  = Pos[:self.max_num_addable_agents] # shape (n_agents, n_I + 1, 6)
            Size = Size[:self.max_num_addable_agents] # shape (n_agents, 2)
            Type = Type[:self.max_num_addable_agents] # shape (n_agents,)

        # Remove extra timestep
        Pos = Pos[:,1:]

        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i + 1)
            u = np.isfinite(pos[:,0])
            if u.sum() > 1:
                path[name] = self.extrapolate_path(pos, t, mode='vel')
                if Type[i] == 1:
                    agent_types[name] = 'V'
                elif Type[i] == 2:
                    agent_types[name] = 'P'
                elif Type[i] == 3:
                    agent_types[name] = 'B'
                else:
                    agent_types[name] = 'M'
                size[name] = Size[i]
                    
        return path, agent_types, size
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'InD (directions)',
                 'file': 'InD_direct',
                 'latex': r'\emph{InD}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return True 
    
    def includes_sceneGraphs(self = None):
        return True
    
    
def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T

    if 'v_x' in track.columns:
        tar_vel = track[['v_x','v_y']].to_numpy()
        track[['v_x','v_y']] = np.dot(Rot_matrix, tar_vel.T).T

    if 'a_x' in track.columns:
        tar_acc = track[['a_x','a_y']].to_numpy()
        track[['a_x','a_y']] = np.dot(Rot_matrix, tar_acc.T).T

    if hasattr(track, 'heading'):
        track.heading = np.mod(track.heading - angle * 180 / np.pi, 360)
    return track


def rotate_track_array(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[:,0:2]
    track[:,0:2] = np.dot(Rot_matrix,(tar_tr - center).T).T

    return track


def determine_streets(track, streets):
    pos = track[['x','y']].to_numpy()
    streets = streets.to_numpy()[:,1:7].reshape(len(streets), 3, 2).astype(float)
    
    useless_streets = (streets[:,0] == streets[:,1]).all(1)
    # Distance to staying is not required as weel
    
    Dt = (streets[:,2] - streets[:,1])[np.newaxis]
    Dt /= np.linalg.norm(Dt, axis = -1, keepdims = True) + 1e-6
    Dtt = np.stack([Dt[:,:,1], -Dt[:,:,0]], -1)
    
    De = (streets[:,0] - streets[:,1])[np.newaxis]
    Dp = pos[:,np.newaxis] - streets[np.newaxis,:,1,:]
    
    sides = np.sign((De * Dtt).sum(-1)) # 1 if both xE and XP are on the same side of the line
    
    D = - sides * (Dtt * Dp).sum(-1)
    D = D.transpose(1,0)
    
    D = D[~useless_streets]
    Use_ind = np.where(~useless_streets)[0]
    
    if np.min(D[:,0]) >= 0:
        dist_in = ((pos[[0]] - streets[:,1]) ** 2).sum(-1)
        street_in = np.argmin(dist_in)
    else:
        street_in = Use_ind[np.argmin(D[:,0])]
        
    if np.min(D[:,-1]) >= 0:
        dist_out = ((pos[[-1]] - streets[:,1]) ** 2).sum(-1)
        street_out = np.argmin(dist_out)
    else:
        street_out = Use_ind[np.argmin(D[:,-1])]
    
    if street_in == street_out:
        # check if heading changed
        heading_in = track.heading.iloc[0]
        heading_out = track.heading.iloc[-1]
        heading_change = abs(heading_in - heading_out)
        # account for circle
        heading_change = min(heading_change, 360 - heading_change)
        if heading_change > 90:
            behavior = 'turned_around'
        else: 
            behavior = 'went_straight'
        
    elif street_in == np.mod(street_out + 1, len(streets)):
        behavior = 'turned_left'
    elif street_in == np.mod(street_out - 1, len(streets)):
        behavior = 'turned_right'
    else:
        behavior = 'went_straight'
    return street_in, street_out, behavior