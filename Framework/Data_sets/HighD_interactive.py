import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
from PIL import Image
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
    
    def path_data_info(self = None):
        return ['x', 'y']
        
        
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
            track_i = data_i.track[['frame','x', 'y']].copy(deep = True)
            
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            
            path['tar'] = np.stack([track_i.x.to_numpy(), track_i.y.to_numpy()], axis = -1)
            agent_types['tar'] = 'V'
            
            t = np.array(track_i.frame / 25)
        
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
    
    
    def fill_empty_path(self, path, t, domain, agent_types):
        n_I = self.num_timesteps_in_real

        tar_pos = path.tar[np.newaxis]
        
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
        # Get ditance to extra vehicles
        D = np.nanmin(np.sqrt(((Pos[:,1:n_I + 1] - tar_pos[:,:n_I]) ** 2).sum(-1)), -1)
        
        # D < 0.5 => Vehicle is tar vehicle => Exclude
        Pos = Pos[(D > 0.5) & (D < 100)]
        Pos = Pos[np.argsort(D)]
        
        if self.max_num_addable_agents is not None:
            Pos = Pos[:self.max_num_addable_agents]
            
        for i, pos in enumerate(Pos):
            name = 'v_{}'.format(i + 1)
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
    
    def includes_sceneGraphs(self = None):
        return False
    
