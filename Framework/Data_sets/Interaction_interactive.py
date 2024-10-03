import numpy as np
import os
import pandas as pd

from Interaction.interaction_utils import get_lane_graph, read_interaction_data
from scipy import interpolate as interp
from tqdm import tqdm

from data_set_template import data_set_template
from scenario_none import scenario_none


class Interaction_interactive(data_set_template):
    def get_name(self = None):
        r'''
        Provides a dictionary with the different names of the dataset
            
        Returns
        -------
        names : dict
            The first key of names ('print')  will be primarily used to refer to the dataset in console outputs. 
                
            The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
            (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. 
                
            The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
            latex commands - such as using '$$' for math notation.
            
        '''

        names = {'print': 'Interaction',
                    'file': 'Interaction',
                    'latex': r'<\emph{Interaction}>'}

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

    def create_path_samples(self):
        r'''
        Loads the original trajectory data from wherever it is saved.
        Then, this function has to extract for each potential test case in the data set 
        some required information. This information has to be collected in the following attributes, 
        which do not have to be returned, but only defined in this function:

        **self.Path** : pandas.DataFrame          
        A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. 
        Here, each row :math:`i` represents one recorded sample, while each column includes the 
        trajectory of an agent (as a numpy array of shape :math:`\{\vert T_i \vert{\times} 2\}`. 
        It has to be noted that :math:`N_{agents}` is the maximum number of agents considered in one
        sample over all recorded samples. If the number of agents in a sample is lower than :math:`N_{agents}`
        the subsequent corresponding fields of the missing agents are filled with np.nan instead of the
        aforementioned numpy array. It is also possible that positional data for an agent is only available
        at parts of the required time points, in which cases, the missing positions should be filled up with
        (np.nan, np.nan).
                    
        The name of each column corresponds to the name of the corresponding
        agent whose trajectory is covered. The name of such agents is relevant, as the selected scenario requires 
        some agents with a specific name to be present. The names of those relevant agents can be found in 
        self.scenario.pov_agent() and self.scenario.classifying_agents().
                    
        **self.Type_old** : pandas.DataFrame  
        A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. Its column names are
        identical to the column names of **self.Path**. Each corresponding entry contains the type of the agent
        whose path is recorded at the same location in *self.Path**.
    
        Currently, four types of agents are implemented:
            - 'V': Vehicles like cars and trucks
            - 'M': Motorcycles
            - 'B': Bicycles
            - 'P': Pedestrians
                
        **self.T** : np.ndarray
        A numpy array (dtype = object) of length :math:`N_{samples}`. Each row :math:`i` contains the timepoints 
        of the data collected in **self.Path** in a tensor of length :math:`\vert T_i \vert`.
                    
        **self.Domain_old** : pandas.DataFrame  
        A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} (N_{info})\}`.
        In this DataFrame, one can collect any ancillary metadata that might be needed
        in the future. An example might be the location at which a sample was recorded
        or the subject id involved, which might be needed later to construct the training
        and testing set. Another useful idea might be to record the place in the raw data the sample
        originated from, as might be used later to extract surrounding agents from this raw data.
                    
        **self.num_samples** : int
        A scalar integer value, which gives the number of samples :math:`N_{samples}`. It should be noted 
        that :math:`self.num_Samples = len(self.Path) = len(self.T) = len(self.Domain_old) = N_{samples}`.
            
        It might be possible that the selected dataset can provide images. In this case, it is
        paramount that **self.Domain_old** contains a column named 'image_id', so that images can
        be assigned to each sample with only having to save one image for each location instead for
        each sample:

        **self.Images** : pandas.DataFrame  
        A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} 2\}`.
        In the first column, named 'Image', the images for each location are saved. It is paramount that the 
        indices of this DataFrame are equivalent to the unique values found in **self.Domain_old**['image_id']. 
        The entry for each cell of the column meanwhile should be a numpy array of dtype np.uint8 and shape
        :math:`\{H {\times} W \times 3\}`. It is assumed that a position :math:`(0,0)` that is recorded
        in the trajectories in **self.Path** corresponds to the upper left corner (that is self.Images.*.Image[0, 0])
        of the image, while the position :math:`(s \cdot W, - s \cdot H)` would be the lower right corner
        (that is self.Images.*.Image[H - 1, W - 1]).
                    
        If this is not the case, due to some translation and subsequent rotation 
        of the recorded positions, the corresponding information has to be recorded in columns of 
        **self.Domain_old**, with the column names 'x_center' and 'y_center'. When we take a trajectory saved in
        self.Path_old, then rotate it counterclockwise by 'rot_angle', and then add 'x_center' and
        'y_center' to the rotated trajectory, the resulting trajectory would then be in the described coordinate
        system where (0,0) would be on the upper left corner of the corresponding image.

        Given a value :math:`\Delta x` for 'x_center' and :math:`\Delta y` for 'y_center',
        and :math:`\theta` for 'rot_angle', the relationship between a position :math:`(x,y)` in the trajectory
        included in **self.Path_old** and the same point :math:`(\hat{x}, \hat{y})` in the coordinate system aligned
        with the image would be the following.
        
        .. math::
            \begin{pmatrix} \hat{x} \\ \hat{y} \end{pmatrix} = \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix} +
            \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} 
            \begin{pmatrix} x \\ y \end{pmatrix}

        NOTE: if any one of the values 'x_center', 'y_center', or 'rot_angle' is set, then the other two values also 
        have to be set. Otherwise, a missing attribute error will be thrown.

        The second column of the DataFrame, named 'Target_MeterPerPx', contains a scalar float value
        :math:`s` that gives us the scaling of the images in the unit :math:`m /` Px. 

        '''

        # TODO: Implement

        self.num_samples = 0
        self.Path = []
        self.Type_old = []
        self.Size_old = []
        self.T = []
        self.Domain_old = []

        file_path = self.path + os.sep + 'Data_sets' + os.sep + 'Interaction' + os.sep + 'data'

        graph_id = 0
        num_saved_sampled = self.get_number_of_saved_samples()

        # Prepare the SceneGraph
        self.map_split_save = True
        sceneGraph_columns = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                              'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type', 'pre', 'suc', 'left', 'right']   
        self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), index = [], columns = sceneGraph_columns)

        for _, name in tqdm(enumerate(os.listdir(file_path + '/train'))):

            data_path = file_path + '/train/' + name
            map_path = file_path + '/maps/' + name.split('.')[0][:-6] + '.osm'
            data_collection = read_interaction_data(data_path)


                # categories = [i[0,0] for i in data_collection['agentcategories'] if i[0,0] in [1, 2, 3]]
                # categories = []

                # focal_track, focal_agent_type, focal_track_id = self.get_focal_track(data_collection)
            for scene in range(len(data_collection['trajs'])):
                self.num_samples += 1
                if self.num_samples > num_saved_sampled:
                    path = pd.Series(np.empty(0, np.ndarray), index = [])
                    agent_types = pd.Series(np.zeros(0, str), index = [])  
                    agent_sizes = pd.Series(np.empty(0, np.ndarray), index = [])
                    agent_id = 0   

                    # min_nan_count = float('inf')
                    # best_id = None
                    for agent in range(data_collection['trajs'].shape[1]):
                        if ~np.isnan(data_collection['trajs'][scene,agent]).all():
                            if np.isnan(data_collection['trajs'][scene,agent]).any() or 'tar' in path:
                                if np.isnan(data_collection['psirads'][scene, agent]).all():
                                    # Get psi_rad from velocity
                                    v = data_collection['vels'][scene, agent]
                                    psi_rad = np.arctan2(v[:,1], v[:,0])
                                    data_collection['psirads'][scene, agent] = psi_rad[:,None]

                                path['agent_'+str(agent_id)] = np.concatenate([data_collection['trajs'][scene, agent], 
                                                                        data_collection['vels'][scene, agent], 
                                                                        data_collection['psirads'][scene, agent]], axis = 1)
                                if (data_collection['agenttypes'][scene, agent] == 1).all():
                                    agent_types['agent_'+str(agent_id)] = 'V'
                                else:
                                    agent_types['agent_'+str(agent_id)] = 'P'

                                agent_sizes['agent_'+str(agent_id)] = data_collection['shapes'][scene, agent][0]

                                
                                # nan_count = np.isnan(path['v_'+str(agent_id)][:,0]).sum()
                                # if nan_count < min_nan_count and nan_count < 30:
                                #     min_nan_count = nan_count
                                #     best_id = agent_id
                                agent_id += 1
                                
                            else:
                                if np.isnan(data_collection['psirads'][scene, agent]).all():
                                    # Get psi_rad from velocity
                                    v = data_collection['vels'][scene, agent]
                                    psi_rad = np.arctan2(v[:,1], v[:,0])
                                    data_collection['psirads'][scene, agent] = psi_rad[:,None]

                                path['tar'] = np.concatenate([data_collection['trajs'][scene, agent], 
                                                            data_collection['vels'][scene, agent], 
                                                            data_collection['psirads'][scene, agent]], axis = 1)
                                if (data_collection['agenttypes'][scene, agent] == 1).all():
                                    agent_types['tar'] = 'V'
                                else:
                                    agent_types['tar'] = 'P'

                                
                                agent_sizes['tar'] = data_collection['shapes'][scene, agent][0]
                        else:
                            continue   

                    if not 'tar' in path:
                        # if best_id is None:
                        continue # remove instances with only one agent who has a complete past
                        # else:
                        #     path['tar'] = path['v_'+str(best_id)]
                        #     agent_types['tar'] = 'V'
                        #     path = path.drop('v_'+str(best_id))
                        #     agent_types = agent_types.drop('v_'+str(best_id))


                    # path, agent_types, categories = self.sort_tracks(data_collection, path, agent_types, categories)

                    # assert 0 not in categories

                    # categories.insert(0,1)
                    # domain.category = pd.Series(categories, index = agent_types.index)

                    t = np.arange(0, 4, 0.1)


                    domain = pd.Series(np.zeros(3, object), index = ['graph_id', 'location', 'splitting'])#, 'category'])
                    
                    domain.graph_id = int(graph_id)
                    domain.location = data_collection['city']
                    domain.splitting = 'train'

                    self.Path.append(path)
                    self.Type_old.append(agent_types)
                    self.Size_old.append(agent_sizes)
                    self.T.append(t)
                    self.Domain_old.append(domain)

                    # Get the scene graph
                    lanegraph = get_lane_graph(map_path)
                    lanegraph_df = pd.DataFrame.from_dict(lanegraph, orient='index', dtype=object)
                    lanegraph_df.columns = [int(graph_id)]
                    self.SceneGraphs.loc[int(graph_id)] = lanegraph_df.iloc[:,0]

                    if self.num_samples % 5000 == 0:
                        self.check_created_paths_for_saving(force_save=True) 
                    else:
                        self.check_created_paths_for_saving(force_save=False)

                graph_id += 1
 

        
        for idx, name in tqdm(enumerate(os.listdir(file_path + '/val'))):

            data_path = file_path + '/val/' + name
            map_path = file_path + '/maps/' + name.split('.')[0][:-4] + '.osm'

            data_collection = read_interaction_data(data_path)


                # categories = [i[0,0] for i in data_collection['agentcategories'] if i[0,0] in [1, 2, 3]]
                # categories = []

                # focal_track, focal_agent_type, focal_track_id = self.get_focal_track(data_collection)
            for scene in range(len(data_collection['trajs'])):
                self.num_samples += 1
                if self.num_samples > num_saved_sampled:
                    path = pd.Series(np.empty(0, np.ndarray), index = [])
                    agent_types = pd.Series(np.zeros(0, str), index = []) 
                    agent_sizes = pd.Series(np.empty(0, np.ndarray), index = [])
                    agent_id = 0   

                    # min_nan_count = float('inf')
                    # best_id = None
                    for agent in range(data_collection['trajs'].shape[1]):
                        if ~np.isnan(data_collection['trajs'][scene,agent]).all():
                            if np.isnan(data_collection['trajs'][scene,agent]).any() or 'tar' in path:
                                if np.isnan(data_collection['psirads'][scene, agent]).all():
                                    # Get psi_rad from velocity
                                    v = data_collection['vels'][scene, agent]
                                    psi_rad = np.arctan2(v[:,1], v[:,0])
                                    data_collection['psirads'][scene, agent] = psi_rad[:,None]

                                path['agent_'+str(agent_id)] = np.concatenate([data_collection['trajs'][scene, agent], 
                                                                        data_collection['vels'][scene, agent], 
                                                                        data_collection['psirads'][scene, agent]], axis = 1)
                                if (data_collection['agenttypes'][scene, agent] == 1).all():
                                    agent_types['agent_'+str(agent_id)] = 'V'
                                else:
                                    agent_types['agent_'+str(agent_id)] = 'P'

                                agent_sizes['agent_'+str(agent_id)] = data_collection['shapes'][scene, agent][0]
                                # nan_count = np.isnan(path['v_'+str(agent_id)][:,0]).sum()
                                # if nan_count < min_nan_count and nan_count < 30:
                                #     min_nan_count = nan_count
                                #     best_id = agent_id

                                agent_id += 1
                                
                            else:
                                if np.isnan(data_collection['psirads'][scene, agent]).all():
                                    # Get psi_rad from velocity
                                    v = data_collection['vels'][scene, agent]
                                    psi_rad = np.arctan2(v[:,1], v[:,0])
                                    data_collection['psirads'][scene, agent] = psi_rad[:,None]

                                path['tar'] = np.concatenate([data_collection['trajs'][scene, agent], 
                                                            data_collection['vels'][scene, agent], 
                                                            data_collection['psirads'][scene, agent]], axis = 1)
                                if (data_collection['agenttypes'][scene, agent] == 1).all():
                                    agent_types['tar'] = 'V'
                                else:
                                    agent_types['tar'] = 'P'

                                agent_sizes['tar'] = data_collection['shapes'][scene, agent][0]
                        else:
                            continue   

                    
                    if not 'tar' in path:
                        # if best_id is None:
                        continue # remove instances without an agent that has a complete trajectory
                        # else:
                        #     path['tar'] = path['v_'+str(best_id)]
                        #     agent_types['tar'] = 'V'
                        #     path = path.drop('v_'+str(best_id))   
                        #     agent_types = agent_types.drop('v_'+str(best_id))

                    # path, agent_types, categories = self.sort_tracks(data_collection, path, agent_types, categories)

                    # assert 0 not in categories

                    # categories.insert(0,1)
                    # domain.category = pd.Series(categories, index = agent_types.index)

                    t = np.arange(0, 4, 0.1)

                    domain = pd.Series(np.zeros(3, object), index = ['graph_id', 'location', 'splitting'])#, 'category'])
                    
                    domain.graph_id = int(graph_id)
                    domain.location = data_collection['city']
                    domain.splitting = 'test'

                    self.Path.append(path)
                    self.Type_old.append(agent_types)
                    self.Size_old.append(agent_sizes)
                    self.T.append(t)
                    self.Domain_old.append(domain)

                    # Get the scene graph
                    lanegraph = get_lane_graph(map_path)
                    lanegraph_df = pd.DataFrame.from_dict(lanegraph, orient='index', dtype=object)
                    lanegraph_df.columns = [int(graph_id)]
                    self.SceneGraphs.loc[int(graph_id)] = lanegraph_df.iloc[:,0]

                    if self.num_samples % 5000 == 0:
                        self.check_created_paths_for_saving(force_save=True) 
                    else:
                        self.check_created_paths_for_saving(force_save=False)

                graph_id += 1


        
        self.check_created_paths_for_saving(last=True) 

        
        self.Path = pd.DataFrame(self.Path)
        self.Type_old = pd.DataFrame(self.Type_old)
        self.Size_old = pd.DataFrame(self.Size_old)
        self.T = np.array(self.T+[()], np.ndarray)[:-1]
        self.Domain_old = pd.DataFrame(self.Domain_old)
        self.SceneGraphs = pd.DataFrame(self.SceneGraphs)

    

    
    def fill_empty_path(self, path, t, domain, agent_types):
        for agent in path.index:
            if isinstance(path[agent], float):
                assert str(path[agent]) == 'nan'
            else:
                if agent_types[agent] == 'P':
                    path[agent] = self.extrapolate_path(path[agent], t, mode = 'vel') 
                else:
                    path[agent] = self.extrapolate_path(path[agent], t, mode = 'vel')
        
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