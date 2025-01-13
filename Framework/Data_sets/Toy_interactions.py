import numpy as np
import pandas as pd
from data_set_template import data_set_template
from scenario_none import scenario_none
import os
import pickle

def rotate_track(track, angle, center):
    Rot_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    tar_tr = track[['x','y']].to_numpy()
    track[['x','y']] = np.dot(Rot_matrix,(tar_tr - center).T).T
    return track


class Toy_interactions(data_set_template):   
    
    
    def set_scenario(self):
        self.scenario = scenario_none()
        
        def toy_classifying_agents():
            return []
        
        self.scenario.classifying_agents = toy_classifying_agents
        
        
    def create_path_samples(self): 
        # Load raw data
        Data = pickle.load(open(self.path + os.sep + 'Data_sets' + os.sep + 
                                    'Toy_interactions' + os.sep + 'data' + os.sep + 'simulated_data_2agents_0-5000_lessNoise', 'rb'))
        
        # Data = pd.concat([Data]*3).reset_index(drop=True)
        # scenarios_adapted = np.repeat(np.arange(0, 5000*3, 1),2)
        # Data['Scenario'] = scenarios_adapted

        # analize raw dara 
        self.num_samples = 0 
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []

        unique_scenarios = np.unique(Data.Scenario)
        
        # extract raw samples
        for i, scenario in enumerate(unique_scenarios):
            
            data = Data[Data.Scenario == scenario]
            unique_ids = np.unique(data.Id)
            
            path = pd.Series(np.zeros(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])
            
            for j, id in enumerate(unique_ids):
                path_j = data[data.Id == id]

                if path_j.iloc[0].Path[0,0] == -2:
                    past = np.concatenate([np.arange(-2-0.2*12, -2.1, 0.2)[:,np.newaxis], np.zeros(12)[:, np.newaxis]], axis=1)
                else:
                    past = np.concatenate([np.flip(np.arange(2.2, 2+0.2*12+0.1, 0.2))[:,np.newaxis], np.zeros(12)[:, np.newaxis]], axis=1)
                traj = np.concatenate([past, path_j.iloc[0].Path[:25,:2]]) # only take the first 25 timesteps in which the bimodality should occur
                
                name = 'v_' + str(j)
                
                path[name] = traj
                agent_types[name] = 'P'
            
            domain = pd.Series(np.zeros(1, object), index = ['location'])
            domain.location = '2agentSim' # make general if including other simulations with more agents
            
            
            t = np.arange(len(path[name])) * 0.2
            
            
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
    
    def path_data_info(self = None):
        return ['x', 'y']
    
    
    def fill_empty_path(self, path, t, domain, agent_types):
        return path, agent_types
    
    def provide_map_drawing(self, domain):
        lines_solid = []
        
        lines_dashed = []
        
        return lines_solid, lines_dashed

    
    def get_name(self = None):
        names = {'print': 'Toy problem (Interactions)',
                 'file': 'Inter_toy_pro',
                 'latex': r'\emph{Toy}'}
        return names
    
    def future_input(self = None):
        return False
    
    
    def includes_images(self = None):
        return False
    

    def includes_sceneGraphs(self = None):
        return False