import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from data_set_template import data_set_template
from scenario_gap_acceptance import scenario_gap_acceptance
import os
from PIL import Image
from scipy import interpolate as interp


def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class RounD_round_about(data_set_template):   
    def set_scenario(self):
        self.scenario = scenario_gap_acceptance()
        
        unique_map = [0, 1, 2]
        
        Loc_data_pix = pd.DataFrame(np.zeros((len(unique_map),4),float), columns = ['xCenter', 'yCenter', 'r', 'R'])
        
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
        
        
        self.Loc_rec = {0: 2, 1: 0, 2: 1}
        self.Loc_scale = {}
        self.Loc_data = Loc_data_pix.copy(deep=True)
        for locId in unique_map:
            rec_id = self.Loc_rec[locId]
            Meta_data=pd.read_csv(self.path + os.sep + 'Data_sets' + os.sep + 
                                  'RounD_round_about' + os.sep + 
                                  'data' + os.sep + '{}_recordingMeta.csv'.format(str(rec_id).zfill(2)))
            self.Loc_scale[locId] = Meta_data['orthoPxToMeter'][0] * 10
            self.Loc_data.iloc[locId] = Loc_data_pix.iloc[locId] * self.Loc_scale[locId] 
    
    def _create_path_sample(self, tar_track, ego_track, other_agents, frame_min, frame_max, 
                            v_1_id, v_2_id, v_3_id, v_4_id,
                            original_angle, Rot_center, data_i):
        
        tar_track_l = tar_track.loc[frame_min:frame_max].copy(deep = True)
        ego_track_l = ego_track.loc[frame_min:frame_max].copy(deep = True)
        
        path = pd.Series(np.empty(0, np.ndarray), index = [])
        path['V_ego'] = np.stack([ego_track_l.x, ego_track_l.y], axis = -1)
        path['V_tar'] = np.stack([tar_track_l.x, tar_track_l.y], axis = -1)
    
        if v_1_id >= 0:
            v_1_track = other_agents.loc[v_1_id].track.loc[frame_min:frame_max]
            
            if len(v_1_track) > 0:
                frame_min_v1 = v_1_track.index.min()
                frame_max_v1 = v_1_track.index.max()
                
                v1x = np.ones(frame_max + 1 - frame_min) * np.nan
                v1x[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.x
                
                v1y = np.ones(frame_max + 1 - frame_min) * np.nan
                v1y[frame_min_v1 - frame_min : frame_max_v1 + 1 - frame_min] = v_1_track.y
                
                path['V_v_1'] = np.stack([v1x, v1y], axis = -1)
            
        if v_2_id >= 0:
            v_2_track = other_agents.loc[v_2_id].track.loc[frame_min:frame_max]
            
            if len(v_2_track) > 0:
                frame_min_v2 = v_2_track.index.min()
                frame_max_v2 = v_2_track.index.max()
                
                v2x = np.ones(frame_max + 1 - frame_min) * np.nan
                v2x[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.x
                
                v2y = np.ones(frame_max + 1 - frame_min) * np.nan
                v2y[frame_min_v2 - frame_min : frame_max_v2 + 1 - frame_min] = v_2_track.y
                
                path['V_v_2'] = np.stack([v2x, v2y], axis = -1)
            
        if v_3_id >= 0:
            v_3_track = other_agents.loc[v_3_id].track.loc[frame_min:frame_max]
            
            if len(v_3_track) > 0:
                frame_min_v3 = v_3_track.index.min()
                frame_max_v3 = v_3_track.index.max()
                
                v3x = np.ones(frame_max + 1 - frame_min) * np.nan
                v3x[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.x
                
                v3y = np.ones(frame_max + 1 - frame_min) * np.nan
                v3y[frame_min_v3 - frame_min : frame_max_v3 + 1 - frame_min] = v_3_track.y
                
                path['V_v_3'] = np.stack([v3x, v3y], axis = -1)
    
        if v_4_id >= 0:
            v_4_track = other_agents.loc[v_4_id].track.loc[frame_min:frame_max]
            
            if len(v_4_track) > 0:
                frame_min_v4 = v_4_track.index.min()
                frame_max_v4 = v_4_track.index.max()
                
                v4x = np.ones(frame_max + 1 - frame_min) * np.nan
                v4x[frame_min_v4 - frame_min : frame_max_v4 + 1 - frame_min] = v_4_track.x
                
                v4y = np.ones(frame_max + 1 - frame_min) * np.nan
                v4y[frame_min_v4 - frame_min : frame_max_v4 + 1 - frame_min] = v_4_track.y
                
                path['P_v_4'] = np.stack([v4x, v4y], axis = -1)

        t = np.array(tar_track_l.index / 25)
        
        domain = pd.Series(np.zeros(7, object), index = ['location', 'image_id', 'track_id', 'rot_angle', 'x_center', 'y_center', 'class'])
        domain.location  = data_i.locationId
        domain.image_id  = data_i.locationId
        domain.track_id  = data_i.trackId
        domain.rot_angle = original_angle
        domain.x_center  = Rot_center[0,0]
        domain.y_center  = Rot_center[0,1]
        domain['class']  = data_i['class']
        
        self.Path.append(path)
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
        self.T = []
        self.Domain_old = []
        
        self.Images = pd.DataFrame(np.zeros((len(self.Loc_data), 1), object), 
                                   index = self.Loc_data.index, columns = ['Image'])
        
        max_width = 0
        max_height = 0
        
        self.Target_MeterPerPx = 0.5 # TODO: Maybe change this?
        for loc_id in self.Loc_data.index:
            rec_id = self.Loc_rec[loc_id]
            img_file = (self.path + os.sep + 'Data_sets' + os.sep + 
                        'RounD_round_about' + os.sep + 'data' + os.sep + 
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
        for i in range(num_samples_max):
            # to keep track:
            if np.mod(i,100) == 0:
                print('trajectory ' + str(i).rjust(len(str(num_samples_max))) + '/{} analized'.format(num_samples_max))
                print('found cases: ' + str(self.num_samples))
                print('')
            data_i = self.Data.iloc[i]
            # assume i is the tar vehicle, which has to be a motor vehicle
            if data_i['class'] in ['bicycle', 'pedestrian', 'trailer']:
                continue
            
            tar_track = data_i.track[['frame','xCenter','yCenter']].rename(columns={"xCenter": "x", "yCenter": "y"}).copy(deep = True)
            Rot_center = np.array([[self.Loc_data.iloc[data_i.locationId].xCenter, self.Loc_data.iloc[data_i.locationId].yCenter]])
            
            tar_track['r'] = np.sqrt((tar_track.x - self.Loc_data.iloc[data_i.locationId].xCenter) ** 2 + 
                                     (tar_track.y - self.Loc_data.iloc[data_i.locationId].yCenter) ** 2)
            
            # exclude trajectory driving over the middle
            if any(tar_track['r'] < self.Loc_data.iloc[data_i.locationId].r):
                continue
            
            # check if tar_track goes through round_about or use shortcut around it
            if not any(tar_track['r'] <= self.Loc_data.iloc[data_i.locationId].R):
                continue
            
            # Exclude vehicles that already startinside the round about
            if tar_track['r'].iloc[0] <= self.Loc_data.iloc[data_i.locationId].R + 10:
                continue
            
            # frame where target vehicle approaches roundd about
            frame_entry = np.where(tar_track['r'] < self.Loc_data.iloc[data_i.locationId].R + 10)[0][0]
            
            tar_frame_A = tar_track['frame'].iloc[np.where(tar_track['r'] < self.Loc_data.iloc[data_i.locationId].R)[0][0]]

            # angle along this route
            original_angle = np.angle((tar_track.x.iloc[0] - tar_track.x.iloc[frame_entry]) + 
                                       (tar_track.y.iloc[0] - tar_track.y.iloc[frame_entry]) * 1j,deg = False)
            
            
            tar_track = rotate_track(tar_track, original_angle, Rot_center)
            
            tar_track['angle'] = np.angle(tar_track.x + tar_track.y * 1j)
            
            tar_track = tar_track.set_index('frame')
            
            other_agents = self.Data[['trackId','class','track']].iloc[data_i.otherVehicles].copy(deep = True)
            
            for j in range(len(other_agents)):
                track_i = other_agents['track'].iloc[j] 
                
                track_i = track_i[['frame','xCenter','yCenter']].rename(columns={"xCenter": "x", "yCenter": "y"}).copy(deep = True)
                
                track_i = rotate_track(track_i, original_angle, Rot_center)
                
                track_i = track_i.set_index('frame').loc[tar_track.index[0]: tar_track.index[-1]]
                
                other_agents['track'].iloc[j] = track_i
                
                other_agents['track'].iloc[j]['r'] = np.sqrt(track_i.x ** 2 + track_i.y ** 2)
                
                other_agents['track'].iloc[j]['angle'] = np.angle(track_i.x + track_i.y * 1j)
            
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
                
                contested = ((tr_j.r.to_numpy() <= self.Loc_data.iloc[data_i.locationId].R) &
                             (tr_j.angle.to_numpy() > 0) & 
                             (tr_j.angle.to_numpy() < np.pi / 6))
                K = np.where(contested[1:] & (contested[:-1] == False))[0] + 1
                
                for k in K:
                    frame_C = tr_j.index[0] + k
                    if tr_j.r.to_numpy()[k - 1] > self.Loc_data.iloc[data_i.locationId].R:
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
                interested = ((tr_j.r.to_numpy() <= self.Loc_data.iloc[data_i.locationId].R) &
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
                        if tr_j.r.to_numpy()[k - 1] > self.Loc_data.iloc[data_i.locationId].R and tr_j.angle.to_numpy()[k - 1] > 0:
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
                        distance_to_cross = np.sqrt((track_p.x - self.Loc_data.iloc[data_i.locationId].R - 5) ** 2 +
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
                                         v_1_id, v_2_id, v_3_id, v_4_id,
                                         original_angle, Rot_center, data_i)
                
                
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
                        distance_to_cross = np.sqrt((track_p.x - self.Loc_data.iloc[data_i.locationId].R - 5) ** 2 +
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
                                         v_1_id, v_2_id, v_3_id, v_4_id,
                                         original_angle, Rot_center, data_i)
        
        self.Path = pd.DataFrame(self.Path)
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
        
        ego_x = path.V_ego[...,0]
        ego_y = path.V_ego[...,1]
        tar_x = path.V_tar[...,0]
        tar_y = path.V_tar[...,1]
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        tar_r = np.sqrt(tar_x ** 2 + tar_y ** 2)
        tar_a = np.angle(tar_x + tar_y * 1j)
        
        # From location data
        R = self.Loc_data.R[domain.location]
        
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
        
        ego_x = path.V_ego[...,0]
        ego_y = path.V_ego[...,1]
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        
        # From location data
        R = self.Loc_data.R[domain.location]
        
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
        if isinstance(path.V_v_1, float):
            assert str(path.V_v_1) == 'nan'
        else:
            v_1_x = path.V_v_1[...,0]
            v_1_y = path.V_v_1[...,1]
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
            
            
        if isinstance(path.V_v_2, float):
            assert str(path.V_v_2) == 'nan'
        else:
            v_2_x = path.V_v_2[...,0]
            v_2_y = path.V_v_2[...,1]
            v_2_r = np.sqrt(v_2_x ** 2 + v_2_y ** 2)
            v_2_a = np.angle(v_2_x + v_2_y * 1j)
            
            v_2_frame_0 = np.nanargmin(np.abs(v_2_a) + (v_2_r > R) * 2 * np.pi)
            if not v_2_frame_0 > ego_frame_0:
                some_error = True
            

        if some_error:
            in_position = np.zeros(len(ego_x), bool) 
        else:
            Rl = R - 0.5 * lane_width
            if isinstance(path.V_v_1, float):
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
        
        
        ego_x = path.V_ego[...,0]
        ego_y = path.V_ego[...,1]
        
        ego_r = np.sqrt(ego_x ** 2 + ego_y ** 2)
        ego_a = np.angle(ego_x + ego_y * 1j)
        
        tar_x = path.V_tar[...,0]
        tar_y = path.V_tar[...,1]
        
        tar_r = np.sqrt(tar_x ** 2 + tar_y ** 2)
        tar_a = np.angle(tar_x + tar_y * 1j)
        
        
        # From location data
        R = self.Loc_data.R[domain.location]
        
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
        if isinstance(path.V_v_1, float):
            assert str(path.V_v_1) == 'nan'
            D1 = 1000 * np.ones_like(Dc)
        else:
            v_1_x = path.V_v_1[...,0]
            v_1_y = path.V_v_1[...,1]
            
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
        
        if isinstance(path.V_v_2, float):
            assert str(path.V_v_2) == 'nan'
            D2 = 1000 * np.ones_like(Dc)
        else:
            v_2_x = path.V_v_2[...,0]
            v_2_y = path.V_v_2[...,1]
            
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
        
        
        if isinstance(path.V_v_3, float):
            assert str(path.V_v_3) == 'nan'
            D3 = 1000 * np.ones_like(Dc)
        else:
            v_3_x = path.V_v_3[...,0]
            v_3_y = path.V_v_3[...,1]
        
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
            D1 = np.interp(t, t[np.isfinite(D1)], D1[np.isfinite(D1)])
        if np.isnan(D2).any():
            D2 = np.interp(t, t[np.isfinite(D2)], D2[np.isfinite(D2)])
        if np.isnan(D3).any():
            D3 = np.interp(t, t[np.isfinite(D3)], D3[np.isfinite(D3)])
        
        Dist = pd.Series([D1, D2, D3, Le, Lt], index = ['D_1', 'D_2', 'D_3', 'L_e', 'L_t'])
        return Dist
    
    def _fill_round_about_path(self, pos, t, domain, R):
        v_x = pos[:,0]
        v_y = pos[:,1]
        
        v_rewrite = np.isnan(v_x)
        if v_rewrite.any():
            v_r = np.sqrt(v_x ** 2 + v_y ** 2)
            v_a = np.angle(v_x + v_y * 1j)
            
            useful = np.invert(v_rewrite)
            # assume whole missing stuff is only outside the roundabout
            useful_r = useful.copy()
            if useful.sum() == 1:
                if useful[0]:
                    v_r[1] = v_r[0] + 10 * self.dt
                    useful_r[1] = True
                elif useful[-1]:
                    v_r[-2] = v_r[-1] + 10 * self.dt
                    useful_r[-2] = True
                else:
                    raise TypeError("Vehicle is way, way, way to fast")
            v_r = interp.interp1d(t[useful_r], v_r[useful_r], fill_value = 'extrapolate', assume_sorted = True)(t)
            v_a = np.interp(t, t[useful], v_a[useful], left = v_a[useful][0], right = v_a[useful][-1])
                
            v_x = np.cos(v_a) * v_r
            v_y = np.sin(v_a) * v_r
            
            assert not np.isnan(v_x).any()
        return np.stack([v_x, v_y], axis = -1)
    
    def fill_empty_input_path(self, path, t, domain):
        R = self.Loc_data.R[domain.location]
        # check vehicle v_1 (in front of ego)
        if isinstance(path.V_v_1, float):
            assert str(path.V_v_1) == 'nan'
        else:
            path.V_v_1 = self._fill_round_about_path(path.V_v_1, t, domain, R)
            
        if isinstance(path.V_v_2, float):
            assert str(path.V_v_2) == 'nan'
        else:
            path.V_v_2 = self._fill_round_about_path(path.V_v_2, t, domain, R)
            
        if isinstance(path.V_v_3, float):
            assert str(path.V_v_3) == 'nan'
        else:
            path.V_v_3 = self._fill_round_about_path(path.V_v_3, t, domain, R)
            
        
        # check vehicle v_4 (pedestrian)
        if isinstance(path.P_v_4, float):
            assert str(path.P_v_4) == 'nan'
        else:
            v_4_x = path.P_v_4[:,0]
            v_4_y = path.P_v_4[:,1]
            
            v_4_rewrite = np.isnan(v_4_x)
            if v_4_rewrite.any():
                v_4_x = np.interp(t,t[np.invert(v_4_rewrite)],v_4_x[np.invert(v_4_rewrite)])
                v_4_y = np.interp(t,t[np.invert(v_4_rewrite)],v_4_y[np.invert(v_4_rewrite)])
                path.P_v_4 = np.stack([v_4_x, v_4_y], axis = -1)
                
        
        # look for other participants
        n_I = self.num_timesteps_in_real

        tar_pos = path.V_tar[np.newaxis]
        
        help_pos = []
        for agent in path.index:
            if isinstance(path[agent], float):
                assert str(path[agent]) == 'nan'
                continue
            if agent[2:] == 'tar':
                continue
            help_pos.append(path[agent])
            
        help_pos = np.stack(help_pos, axis = 0)
        
        tar_frames = 25 * (t + domain.t_0)
        
        
        if not hasattr(self, 'Data'):
            self.Data = pd.read_pickle(self.path + os.sep + 'Data_sets' + os.sep + 
                                       'RounD_round_about' + os.sep + 'RounD_processed.pkl')
            self.Data = self.Data.reset_index(drop = True) 
        
        
        Neighbor = self.Data.iloc[domain.track_id].otherVehicles
        Neighbor_type = np.array(self.Data.iloc[Neighbor]['class'])
        frames_help = np.concatenate([[tar_frames[0] - 1], tar_frames])
        # search for vehicles
        Pos = np.ones((len(Neighbor), len(frames_help), 2)) * np.nan
        for i, n in enumerate(Neighbor):
            track_n = self.Data.iloc[n].track.rename(columns={"xCenter": "x", "yCenter": "y"}).copy(deep = True)
            track_n = rotate_track(track_n, domain.rot_angle, 
                                   np.array([[domain.x_center, domain.y_center]]))
            Pos[i,:,0] = np.interp(frames_help, np.array(track_n.frame), track_n.x, left = np.nan, right = np.nan)
            Pos[i,:,1] = np.interp(frames_help, np.array(track_n.frame), track_n.y, left = np.nan, right = np.nan)
        
        
        actually_there = np.isfinite(Pos[:,1:n_I + 1]).any((1,2))
        Neighbor_type = Neighbor_type[actually_there]
        Pos           = Pos[actually_there]

        D_help = np.nanmin(np.sqrt(((Pos[np.newaxis, :,1:n_I + 1] - help_pos[:,np.newaxis,:n_I]) ** 2)).sum(-1), -1).min(0)
        actually_interesting = (D_help > 0.1) & (D_help < 100)
        Neighbor_type = Neighbor_type[actually_interesting]
        Pos           = Pos[actually_interesting]
        
        # filter out nonmoving vehicles
        actually_moving = (np.nanmax(Pos, 1) - np.nanmin(Pos, 1)).max(1) > 0.1
        Neighbor_type = Neighbor_type[actually_moving]
        Pos           = Pos[actually_moving]
        
        # Find cars that could influence tar vehicle
        D = np.nanmin(np.sqrt(((Pos[:,1:n_I + 1] - tar_pos[:,:n_I]) ** 2).sum(-1)), -1)
        Neighbor_type = Neighbor_type[D < 75]
        Pos           = Pos[D < 75]
        D             = D[D < 75]
        
        # sort by closest vehicle
        Pos           = Pos[np.argsort(D)]
        Neighbor_type = Neighbor_type[np.argsort(D)]
        
        ind_veh = 0 
        ind_ped = 0
        for i, pos in enumerate(Pos):
            agent_type = Neighbor_type[i]
            if agent_type == 'pedestrian':
                ind_ped += 1
                name = 'P_v_{}'.format(ind_ped + 999)
            else:
                ind_veh += 1
                name = 'V_v_{}'.format(ind_veh + 4)
                
            u = np.isfinite(pos[:,0])
            if u.sum() > 1:
                if u.all():
                    path[name] = pos
                else:
                    frames = frames_help[u]
                    p = pos[u].T
                    path[name] = np.stack([interp.interp1d(frames, p[0], fill_value = 'extrapolate', assume_sorted = True)(tar_frames),
                                           interp.interp1d(frames, p[1], fill_value = 'extrapolate', assume_sorted = True)(tar_frames)], axis = -1)
                
        return path
            
    
    def provide_map_drawing(self, domain):
        R = self.Loc_data.R[domain.location]
        r = self.Loc_data.r[domain.location]
        
        x = np.arange(-1,1,501)[:,np.newaxis]
        unicircle_upper = np.concatenate((x, np.sqrt(1 - x ** 2)), axis = 1)
        unicircle_lower = np.concatenate((- x, - np.sqrt(1 - x ** 2)), axis = 1)
        
        unicircle = np.concatenate((unicircle_upper, unicircle_lower[1:, :]))
        
        lines_solid = []
        lines_solid.append(unicircle * r)
        
        lines_dashed = []
        lines_dashed.append(np.array([[R, 0],[300, 0]]))
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