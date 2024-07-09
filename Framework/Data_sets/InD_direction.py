import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_direction import scenario_direction
import os
from PIL import Image
from scipy import interpolate as interp

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
    def analyze_maps(self):
        unique_map = [1, 2, 3, 4]
        
        self.Loc_data_pix = pd.DataFrame(np.zeros((len(unique_map),3),float), columns = ['locationId', 'center_x', 'center_y'])
        # Center is choosen, so that it lies at teh crossing between the dividers of incoming and outcoming lanes
        # at the entrance of the crossing
        
        self.Loc_data_pix.locationId = [1,2,3,4]
        self.Loc_data_pix['streets'] = '0'
        self.Loc_data_pix = self.Loc_data_pix.set_index('locationId')
        
        # Location 1
        self.Loc_data_pix.center_x.loc[1], self.Loc_data_pix.center_y.loc[1] = 564, -335
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
        self.Loc_data_pix.streets.iloc[0] = streets_1
        
        self.Loc_data_pix.center_x.loc[2], self.Loc_data_pix.center_y.loc[2] = 483, -306
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
        self.Loc_data_pix.streets.iloc[1] = streets_2
        
        self.Loc_data_pix.center_x.loc[3], self.Loc_data_pix.center_y.loc[3] = 430, -262
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
        self.Loc_data_pix.streets.iloc[2] = streets_3
        
        self.Loc_data_pix.center_x.loc[4], self.Loc_data_pix.center_y.loc[4] = 928, -390
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
        self.Loc_data_pix.streets.iloc[3] = streets_4
        
        # Attention: No deep copy of the pandas dataframe in streets, so be careful
        self.Loc_data = pd.DataFrame(np.empty(self.Loc_data_pix.shape, object), 
                                     columns = self.Loc_data_pix.columns, 
                                     index = self.Loc_data_pix.index)
        
        # You cannot rely on the values given for orthoPxToMeter
        
        self.Loc_rec = {1: 7, 2: 18, 3: 30, 4: 0}
        
        self.Loc_scale = {}
        for locId in unique_map:
            rec_id = self.Loc_rec[locId]
            Meta_data=pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
                                  'InD_direction' + os.sep + 
                                  'data' + os.sep + '{}_recordingMeta.csv'.format(str(rec_id).zfill(2)))
            
            self.Loc_scale[locId] = Meta_data['orthoPxToMeter'][0] * 12
        
        for locId in unique_map:
            MeterPerPx = self.Loc_scale[locId]
            streets_pix = self.Loc_data_pix.streets.loc[locId]
            streets = pd.DataFrame(np.zeros(streets_pix.shape, object), 
                                   columns = streets_pix.columns, 
                                   index = streets_pix.index)
            streets.iloc[:,1:] = streets_pix.iloc[:,1:] * MeterPerPx
            streets.iloc[:,0]  = streets_pix.iloc[:,0] 
            self.Loc_data.loc[locId].streets = streets
            self.Loc_data.center_x.loc[locId] = self.Loc_data_pix.center_x.loc[locId] * MeterPerPx
            self.Loc_data.center_y.loc[locId] = self.Loc_data_pix.center_y.loc[locId] * MeterPerPx
            
    def set_scenario(self):
        self.scenario = scenario_direction()
        self.analyze_maps()
    
    def path_data_info(self = None):
        return ['x', 'y']
        
   
    def create_path_samples(self): 
        # Load raw data
        self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                   'InD_direction' + os.sep + 'InD_processed.pkl')
        # analize raw dara 
        self.Data = self.Data.reset_index(drop = True)
        num_samples_max = len(self.Data)
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
    
        # Create Images
        self.Images = pd.DataFrame(np.zeros((len(self.Loc_data), 1), object), 
                                   index = self.Loc_data.index, columns = ['Image'])
        
        max_width = 0
        max_height = 0
        
        self.Target_MeterPerPx = 0.5 # TODO: Maybe change this?
        for loc_id in self.Loc_data.index:
            rec_id = self.Loc_rec[loc_id]
            img_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                        'InD_direction' + os.sep + 'data' + os.sep + 
                        str(rec_id).zfill(2) + '_background.png')
            
            img = Image.open(img_file)
            
            img_scaleing = self.Loc_scale[loc_id] / self.Target_MeterPerPx
            
            height_new = int(img.height * img_scaleing)
            width_new  = int(img.width * img_scaleing)
            
            img_new = img.resize((width_new, height_new), Image.LANCZOS)
            
            self.Images.loc[loc_id].Image = np.array(img_new)
            max_width = max(width_new, max_width)
            max_height = max(height_new, max_height)
            
        # pad images to max size
        for loc_id in self.Loc_data.index:
            img = self.Images.loc[loc_id].Image
            img_pad = np.pad(img, ((0, max_height - img.shape[0]),
                                   (0, max_width  - img.shape[1]),
                                   (0,0)), 'constant', constant_values=0)
            self.Images.loc[loc_id].Image = img_pad            
        
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
            
            track_i = agent_i.track[['frame','xCenter','yCenter', 'heading']].rename(columns={"xCenter": "x", "yCenter": "y"}).copy(deep = True)
            
            streets_i = self.Loc_data.streets.loc[agent_i.locationId]
            
            # Determine the action of the vehicle
            if self.Data.iloc[i].behavior == '0':
                agent_i.street_in, agent_i.street_out, agent_i.behavior = determine_streets(track_i, streets_i)
                self.Data.street_in.iloc[i], self.Data.street_out.iloc[i], self.Data.behavior.iloc[i] = agent_i.street_in, agent_i.street_out, agent_i.behavior
            
            entry = streets_i.loc[agent_i.street_in][['entry_x', 'entry_y']].to_numpy()
            center = streets_i.loc[agent_i.street_in][['at_cross_x', 'at_cross_y']].to_numpy()
            
            diff = entry - center
            
            angle = np.angle(diff[0] + 1j * diff[1])
            track_i = rotate_track(track_i, angle, center)
            
            # Check if the vehicle is already leaving
            if not (135 < track_i.heading.iloc[0] < 225):
                continue
            
            path = pd.Series(np.zeros(0, object), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            
            path['tar'] = np.stack([track_i.x.to_numpy(), track_i.y.to_numpy()], axis = -1)
            agent_types['tar'] = 'V'
            
            t = np.array(track_i.frame / 25)
            
            # collect domain data
            domain = pd.Series(np.zeros(8, object), index = ['location', 'image_id', 'rot_angle', 'x_center', 'y_center', 
                                                             'behavior', 'class', 'neighbor_veh', 'neighbor_ped'])
            domain.location    = agent_i.locationId
            domain.image_id    = agent_i.locationId
            domain.rot_angle   = angle
            domain.x_center    = center[0]
            domain.y_center    = center[1]
            domain['class']    = agent_i['class']
            domain['behavior'] = agent_i['behavior']
            domain['old_id']   = agent_i.trackId
            
            # Divide neighbors by class
            neighbor_id    = agent_i.otherVehicles
            neighbor_class = self.Data.loc[neighbor_id]['class']
            
            domain['neighbor_veh'] = neighbor_id[(neighbor_class == 'car') | (neighbor_class == 'truck_bus')]
            domain['neighbor_ped'] = neighbor_id[(neighbor_class == 'pedestrian')]               
            
            
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
        pos = path.tar
        
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
        pos = path.tar
        
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
        pos = path.tar
        
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
        
        Dist = pd.Series([-D], index = ['D_decision'])
        return Dist
    
    
    def fill_empty_path(self, path, t, domain, agent_types):
        # Get old data, so that removed pedestrians can be accessed
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'InD_direction' + os.sep + 'InD_processed.pkl')
            self.Data = self.Data.reset_index(drop = True)
        
        n_I = self.num_timesteps_in_real

        tar_pos = path.tar[np.newaxis]
        I_t = t + domain.t_0
        t_help = np.concatenate([[I_t[0] - self.dt], I_t])
        # search for vehicles
        Neighbor_veh = domain.neighbor_veh.copy()
        Pos_veh = np.ones((len(Neighbor_veh), len(I_t) + 1,2)) * np.nan
        for i, n in enumerate(Neighbor_veh):
            track_n = self.Data.loc[n].track
            t = track_n.frame / 25
            # exclude stationary cars
            if t.max() - t.min() > 100:
                continue
            Pos_veh[i,:,0] = np.interp(t_help, np.array(t), track_n.xCenter, left = np.nan, right = np.nan)
            Pos_veh[i,:,1] = np.interp(t_help, np.array(t), track_n.yCenter, left = np.nan, right = np.nan)
        
        Pos_veh = Pos_veh[np.isfinite(Pos_veh[:,1:n_I + 1]).any((1,2))]
        
        # filter our parking vehicles
        actually_moving = (np.nanmax(Pos_veh, 1) - np.nanmin(Pos_veh, 1)).max(1) > 0.1
        Pos_veh = Pos_veh[actually_moving]
        
        # search for pedestrians
        Neighbor_ped = domain.neighbor_ped.copy()
        Pos_ped = np.ones((len(Neighbor_ped), len(I_t) + 1,2)) * np.nan
        for i, n in enumerate(Neighbor_ped):
            track_n = self.Data.loc[n].track
            t = track_n.frame / 25
            Pos_ped[i,:,0] = np.interp(t_help, np.array(t), track_n.xCenter, left = np.nan, right = np.nan)
            Pos_ped[i,:,1] = np.interp(t_help, np.array(t), track_n.yCenter, left = np.nan, right = np.nan)
        
        Pos_ped = Pos_ped[np.isfinite(Pos_ped[:,1:n_I + 1]).any((1,2))]
        
        Pos = np.concatenate((Pos_veh, Pos_ped), axis = 0)
        Type = np.zeros(len(Pos))
        Type[:len(Pos_veh)] = 1
        
        D = np.nanmin((np.sqrt((Pos[:,1:n_I + 1] - tar_pos[:,:n_I]) ** 2).sum(-1)), -1)
        Pos  = Pos[np.argsort(D)]
        Type = Type[np.argsort(D)]
        
        if self.max_num_addable_agents is not None:
            Pos  = Pos[:self.max_num_addable_agents]
            Type = Type[:self.max_num_addable_agents]
            

        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i + 1)
                
            u = np.isfinite(pos[:,0])
            if u.sum() > 1:
                if u.all():
                    path[name] = pos
                else:
                    t = t_help[u]
                    p = pos[u].T
                    path[name] = np.stack([interp.interp1d(t, p[0], fill_value = 'extrapolate', assume_sorted = True)(I_t),
                                           interp.interp1d(t, p[1], fill_value = 'extrapolate', assume_sorted = True)(I_t)], axis = -1)
                
                if Type[i] == 1:
                    agent_types[name] = 'V'
                else:
                    agent_types[name] = 'P'
                    
        return path, agent_types
    
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
    
    
def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],
                           [-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    if hasattr(track, 'heading'):
        track.heading = np.mod(track.heading - angle * 180 / np.pi, 360)
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