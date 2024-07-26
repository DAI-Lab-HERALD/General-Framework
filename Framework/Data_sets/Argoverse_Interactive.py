import av2
import numpy as np
import os
import pandas as pd

from Argoverse.argoverse_utils import read_argoverse2_data, get_lane_graph
from scipy import interpolate as interp
from tqdm import tqdm

from data_set_template import data_set_template
from scenario_none import scenario_none

class Argoverse_Interactive(data_set_template):
    object_type_dict = {
        0: 'V',
        1: 'P',
        2: 'M',
        3: 'B',
        4: 'V', # bus
        5: 'other', # static
        6: 'other', # background
        7: 'other', # construction
        8: 'other', # riderless_bicycle
        9: 'other' # unknown
    }

    agent_cat_dict = {
        0: 'fragmented',
        1: 'unscored',
        2: 'scored',
        3: 'focal'
    }

    def get_name(self = None):

        names = {'print': 'Argoverse',
                    'file': 'Argoverse',
                    'latex': r'<\emph{Argoverse}>'}

        return names

    def future_input(self = None):
        r'''
        return True: The future data of the pov agent can be used as input.
        This is especially feasible if the ego agent was controlled by an algorithm in a simulation,
        making the recorded future data similar to the ego agent's planned path at each point in time.
            
        return False: This usage of future ego agent's trajectories as model input is prevented. This is
        especially advisable if the behavior of the vehicle might include too many clues for a prediction
        model to use.
            
        Returns
        -------
        future_input_decision : bool
            
        '''
        return True
    
    def includes_images(self = None):
        return False
    

    def includes_sceneGraphs(self = None):
        return True
    

    def set_scenario(self):
        self.scenario = scenario_none()

    
    def get_focal_track(self, data):
        focal_id = data['focal_id']

        for i in range(len(data['trajs'])):
            if data['track_ids'][i][0] == focal_id:
                # if path is shorter than 110 timesteps fill rest of the path with nan 
                data['trajs'][i] = np.concatenate([data['trajs'][i], np.full((110-len(data['trajs'][i]), 2), np.nan)])
                return data['trajs'][i], self.object_type_dict[data['agenttypes'][i][0,0]], i
            
    def sort_tracks(self, data, path, agent_types, categories):
        other_agent = 0
        scored_agent = 0

        for i in range(len(data['trajs'])-1):
            if data['agenttypes'][i][0,0] in [0, 1, 2, 3, 4]:
                data['trajs'][i] = np.concatenate([data['trajs'][i], np.full((110-len(data['trajs'][i]), 2), np.nan)])
                data['vels'][i] = np.concatenate([data['vels'][i], np.full((110-len(data['vels'][i]), 2), np.nan)])
                data['psirads'][i] = np.concatenate([data['psirads'][i], np.full((110-len(data['psirads'][i]), 1), np.nan)])

                if data['agentcategories'][i][0,0] == 2:
                    path['scored_agent_' + str(scored_agent)] = np.concatenate([data['trajs'][i], data['vels'][i], data['psirads'][i]], axis = 1)
                    agent_types['scored_agent_' + str(scored_agent)] = self.object_type_dict[data['agenttypes'][i][0,0]]
                    # agent_categories['scored_agent_' + str(scored_agent)] = data['agentcategories'][i][0,0]

                    categories.append(data['agentcategories'][i][0,0])
                    
                    scored_agent += 1

                elif data['agentcategories'][i][0,0] == 3:
                    path['tar'] = np.concatenate([data['trajs'][i], data['vels'][i], data['psirads'][i]], axis = 1)
                    agent_types['tar'] = self.object_type_dict[data['agenttypes'][i][0,0]]
                    # agent_categories['tar' ] = data['agentcategories'][i][0,0]

                    categories.append(data['agentcategories'][i][0,0])

                elif data['agentcategories'][i][0,0] == 1:
                    path['other_agent_' + str(other_agent)] = np.concatenate([data['trajs'][i], data['vels'][i], data['psirads'][i]], axis = 1)
                    agent_types['other_agent_' + str(other_agent)] = self.object_type_dict[data['agenttypes'][i][0,0]]
                    # agent_categories['other_agent_' + str(other_agent)] = data['agentcategories'][i][0,0]

                    categories.append(data['agentcategories'][i][0,0])

                    other_agent += 1

        # agent_id = 0
        # for i in range(len(data['trajs'])-1):
        #     if i != focal_id and data['agenttypes'][i][0,0] in [0, 1, 2, 3, 4]:
        #         # if path is shorter than 110 timesteps fill rest of the path with nan 
        #         data['trajs'][i] = np.concatenate([data['trajs'][i], np.full((110-len(data['trajs'][i]), 2), np.nan)])
        #         path['agent_' + str(agent_id)] = data['trajs'][i]
        #         agent_types['agent_' + str(agent_id)] = self.object_type_dict[data['agenttypes'][i][0,0]]
        #         agent_id += 1

        return path, agent_types, categories

    def create_path_samples(self):

        # TODO: Implement
        self.num_samples = 0
        self.Path = []
        self.Type_old = []
        self.T = []
        self.Domain_old = []
        self.SceneGraphs = []

        file_path = self.path + os.sep + 'Data_sets' + os.sep + 'Argoverse' + os.sep + 'data'

        graph_id = 0
        for _, name in tqdm(enumerate(os.listdir(file_path + '/train_small'))):
            data_path = file_path + '/train_small/' + name
            data_collection = read_argoverse2_data(data_path)

            lanegraph = get_lane_graph(data_path)

            domain = pd.Series(np.zeros(5, object), index = ['graph_id', 'focal_id', 'location', 'splitting', 'category'])
            
            domain.graph_id = int(graph_id)
            domain.focal_id = data_collection['focal_id']
            domain.location = data_collection['city_name']
            domain.splitting = 'train'

            # categories = [i[0,0] for i in data_collection['agentcategories'] if i[0,0] in [1, 2, 3]]
            categories = []

            # focal_track, focal_agent_type, focal_track_id = self.get_focal_track(data_collection)

            path = pd.Series(np.empty(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])            

            path, agent_types, categories = self.sort_tracks(data_collection, path, agent_types, categories)

            assert 0 not in categories


            path['AV'] = np.concatenate([data_collection['trajs'][-1], data_collection['vels'][-1], data_collection['psirads'][-1]], axis = 1)
            agent_types['AV'] = 'V'
            categories.append(1)
            domain.category = pd.Series(categories, index = agent_types.index)

            t = np.arange(0, 11, 0.1)

            lanegraph_df = pd.DataFrame.from_dict(lanegraph, orient='index', dtype=object)
            lanegraph_df.columns = [int(graph_id)]


            print('Number of agents: ' + str(len(path)))
            print('Number of frames: ' + str(len(t)))
            self.num_samples += 1
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
            self.SceneGraphs.append(lanegraph_df.iloc[:,0])

            graph_id += 1

        
        for idx, name in tqdm(enumerate(os.listdir(file_path + '/val_small'))):
            data_path = file_path + '/val_small/' + name
            data_collection = read_argoverse2_data(data_path)

            lanegraph = get_lane_graph(data_path)

            domain = pd.Series(np.zeros(5, object), index = ['graph_id', 'focal_id', 'location', 'splitting', 'category'])
            
            domain.graph_id = int(graph_id)
            domain.focal_id = data_collection['focal_id']
            domain.location = data_collection['city_name']
            domain.splitting = 'test'

            # categories = [i[0,0] for i in data_collection['agentcategories'] if i[0,0] in [1, 2, 3]]
            categories = []
            # focal_track, focal_agent_type, focal_track_id = self.get_focal_track(data_collection)

            path = pd.Series(np.empty(0, np.ndarray), index = [])
            agent_types = pd.Series(np.zeros(0, str), index = [])


            path, agent_types, categories = self.sort_tracks(data_collection, path, agent_types, categories)

            assert 0 not in categories

            path['AV'] = np.concatenate([data_collection['trajs'][-1], data_collection['vels'][-1], data_collection['psirads'][-1]], axis = 1)
            agent_types['AV'] = 'V'  
            categories.append(1)
            domain.category = pd.Series(categories, index = agent_types.index)

            # path['tar'] = focal_track
            # agent_types['tar'] = focal_agent_type
            
            # path, agent_types = self.get_other_tracks(data_collection, focal_track_id, path, agent_types)

            t = np.arange(0, 11, 0.1)

            lanegraph_df = pd.DataFrame.from_dict(lanegraph, orient='index', dtype=object)
            lanegraph_df.columns = [int(graph_id)]

            print('Number of agents: ' + str(len(path)))
            print('Number of frames: ' + str(len(t)))
            self.num_samples += 1
            self.Path.append(path)
            self.Type_old.append(agent_types)
            self.T.append(t)
            self.Domain_old.append(domain)
            self.SceneGraphs.append(lanegraph_df.iloc[:,0])

            graph_id += 1

        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
        self.SceneGraphs = pd.DataFrame(self.SceneGraphs)
        


    def fill_empty_path(self, path, t, domain, agent_types):

        for agent in path.index:
            if isinstance(path[agent], float):
                assert str(path[agent]) == 'nan'
            else:
                x = path[agent][:,0]
                y = path[agent][:,1]
                
                rewrite = np.isnan(x)
                if not rewrite.any():
                    continue
                useful = np.invert(rewrite)
                if agent_types[agent] == 'P':
                    x = np.interp(t,t[useful],x[useful])
                    y = np.interp(t,t[useful],y[useful])
                else:
                    x = interp.interp1d(t[useful], x[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
                    y = interp.interp1d(t[useful], y[useful], fill_value = 'extrapolate', assume_sorted = True)(t)
            
                path[agent] = np.stack([x, y], axis = -1)
        
        return path, agent_types
    

    def path_data_info(self = None):
        r'''
        This returns the datatype that is saved in the **self.Path** attribute.

        Returns
        -------
        path_data_type : list
            This is a list of strings, with each string indicating what type of data 
            is saved along the last dimension of the numpy arrays in **self.Path**.
            The following strings are right now admissible:
            - 'x':          The :math:`x`-coordinate of the agent's position.
            - 'y':          The :math:`y`-coordinate of the agent's position.
            - 'v_x':        The :math:`x`-component of the agent's velocity, 
                            i.e., :math:`v_x`.
            - 'v_y':        The :math:`y`-component of the agent's velocity, 
                            i.e., :math:`v_y`.
            - 'a_x':        The :math:`x`-component of the agent's acceleration, 
                            i.e., :math:`a_x`.
            - 'a_y':        The :math:`y`-component of the agent's acceleration, 
                            i.e., :math:`a_y`.
            - 'v':          The magnitude of the agent's velocity. It is calculated 
                            as :math:`\sqrt{v_x^2 + v_y^2}`. 
            - 'theta':      The angle of the agent's orientation. It is calculated as 
                            :math:`\arctan2(v_y / v_x)`.
            - 'a':          The magnitude of the agent's acceleration. It is calculated 
                            as :math:`\sqrt{a_x^2 + a_y^2}`.
            - 'd_theta':    The angle of the agent's acceleration. It is calculated as
                            :math:`(a_x v_y - a_y v_x) / (v_x^2 + v_y^2)`. 
        '''
        return ['x', 'y', 'v_x', 'v_y', 'theta']
    
    
    def provide_map_drawing(self, domain):
        r'''
        For the visualization feature of the framework, a background picture is desirable. However, such an
        image might not be available, or it might be beneficial to highlight certain features. In that case,
        one can provide additional lines (either dashed or solid) to be drawn (if needed on top of images),
        that allow greater context for the depicted scenario.
            
        Parameters
        ----------
        domain : pandas.Series
        A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
        sample. Its entries contain at least all the columns of **self.Domain_old**. 

        Returns
        -------
        lines_solid : list
        This is a list of numpy arrays, where each numpy array represents on line to be drawn. 
        Each array is of the shape :math:`\{N_{points} \times 2 \}`, where the positions of the 
        points are given in the same coordinate frame as the positions in **self.Path**. The lines
        connecting those points will be solid.
                
        lines_dashed : list
        This is identical in its form to **lines_solid**, however, the depicted points will be 
        connected by dashed lines.
                
        '''

        # TODO: Implement

        lines_solid = []
        lines_dashed = []
        return lines_solid, lines_dashed


    def evaluate_scenario(self, path, D_class, domain):
        return None
