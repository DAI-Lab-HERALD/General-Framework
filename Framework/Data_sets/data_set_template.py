import pandas as pd
import numpy as np
import scipy as sp
import os
import torch
import psutil
import networkx as nx
from data_interface import data_interface
from inspect import signature
from utils.memory_utils import get_total_memory, get_used_memory

class data_set_template():
    # %% Implement the provision of data
    def __init__(self, 
                 Perturbation = None,
                 model_class_to_path = None, 
                 num_samples_path_pred = 20, 
                 enforce_num_timesteps_out = True, 
                 enforce_prediction_time = False, 
                 exclude_post_crit = True,
                 allow_extrapolation = True,
                 agents_to_predict = 'predefined',
                 overwrite_results = 'no',
                 allow_longer_predictions = True,
                 total_memory = psutil.virtual_memory().total):
        # Find path of framework
        self.path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
        
        # Save total memory
        self.total_memory = total_memory

        # Clarify that no data has been loaded yet
        self.data_loaded = False
        self.raw_data_loaded = False
        self.prediction_time_set = False

        # Set the model used in transformation functions
        if model_class_to_path == None:
            self.model_class_to_path = None
        elif model_class_to_path != None and model_class_to_path.get_output_type() == 'path_all_wi_pov':
            self.model_class_to_path = model_class_to_path
        else:
            raise TypeError("The chosen model does not predict trajectories.")

        # Set the number of paths that a trajectory prediction model has to predict
        if type(num_samples_path_pred) != type(0):
            raise TypeError("num_samples_path_pred should be an integer")
        self.num_samples_path_pred = max(1, int(num_samples_path_pred))

        self.set_scenario()
        if not hasattr(self, 'scenario'):
            raise AttributeError('Has to be defined in actual data-set class')

        self.Behaviors = np.array(list(self.scenario.give_classifications()[0].keys()))
        self.behavior_default = self.scenario.give_default_classification()

        if not self.behavior_default in self.Behaviors:
            raise AttributeError(
                'In the provided scenario, the default behavior is not part of the set of all behaviors.')
        self.classification_useful = len(self.Behaviors) > 1

        # Check if general input is possible
        self.general_input_available = (self.classification_useful and
                                        self.scenario.can_provide_general_input() is not None)
        
        if self.general_input_available:
            self.extra_input = self.scenario.can_provide_general_input()

        # Needed agents
        self.pov_agent = self.scenario.pov_agent()
        if self.pov_agent is not None:
            self.needed_agents = [self.scenario.pov_agent()] + self.scenario.classifying_agents()
        else:
            self.needed_agents = self.scenario.classifying_agents()
        
        # Determine if all predicted timesteps must be observable
        self.enforce_num_timesteps_out = enforce_num_timesteps_out
        self.allow_longer_predictions  = allow_longer_predictions
        self.enforce_prediction_time   = enforce_prediction_time
        self.exclude_post_crit         = exclude_post_crit
        self.overwrite_results         = overwrite_results
        self.allow_extrapolation       = allow_extrapolation
        self.agents_to_predict         = agents_to_predict

        self.p_quantile = np.linspace(0.1, 0.9, 9)
        self.path_models_trained = False

        # Handle perturbation
        if Perturbation is None:
            self.is_perturbed = False
            self.Perturbation = None
        else:
            self.is_perturbed = True
            self.Perturbation = Perturbation
            
        # Get file path for original paths
        self.file_path = (self.path + os.sep + 'Results' + os.sep +
                         self.get_name()['print'] + os.sep +
                         'Data' + os.sep +
                         self.get_name()['file'])
        
        
    def check_path_samples(self, Path, Type_old, T, Domain_old, num_samples, Size_old = None):
        # Check if the rigth path information if provided
        path_info = self.path_data_info()
        if not isinstance(path_info, list):
            raise TypeError("self.path_data_info() should return a list.")
        else:
            for i, info in enumerate(path_info):
                if not isinstance(info, str):
                    raise TypeError("Elements of self.path_data_info() should be strings. \n Element {} is not a string: ".format(i), info)
        
        # Check if x and y are included
        if path_info[0] != 'x':
            raise AttributeError("'x' should be included as the first element self.path_data_info().")
        if path_info[1] != 'y': 
            raise AttributeError("'y' should be included as the second element self.path_data_info().")

        # check some of the aspect to see if pre_process worked
        if not isinstance(Path, pd.core.frame.DataFrame):
            raise TypeError("Paths should be saved in a pandas data frame. \nInstead, type(Path) returns: ", type(Path))
        if len(Path) != num_samples:
            raise TypeError("Path does not have right number of sampels. \n len(Path) = {}, but the expected value is {}.".format(len(Path), num_samples))

        # check some of the aspect to see if pre_process worked
        if not isinstance(Type_old, pd.core.frame.DataFrame):
            raise TypeError("Agent Types (Type_old) should be saved in a pandas data frame. \nInstead, type(Type_old) returns: ", type(Type_old))
        if len(Type_old) != num_samples:
            raise TypeError("Type_old does not have right number of sampels. \n len(Type_old) = {}, but the expected value is {}.".format(len(Type_old), num_samples))
    
        if not isinstance(T, np.ndarray):
            raise TypeError("Time points (T) should be saved in a numpy array. \nInstead, type(T) returns: ", type(T))
        if len(T) != num_samples:
            raise TypeError("T does not have right number of sampels. \n len(T) = {}, but the expected value is {}.".format(len(T), num_samples))

        if not isinstance(Domain_old,  pd.core.frame.DataFrame):
            raise TypeError("Domain information (Domain_old) should be saved in a Pandas Dataframe. \nInstead, type(Domain_old) returns: ", type(Domain_old))
        if len(Domain_old) != num_samples:
            raise TypeError("Domain_old does not have right number of sampels. \n len(Domain_old) = {}, but the expected value is {}.".format(len(Domain_old), num_samples))
        
        if self.includes_images():
            if not 'image_id' in Domain_old.columns:
                raise AttributeError('For your dataset, you defined that self.includes_images() is True. \nTherefore, Domain_old should include the column image_id, which it does not.')
        
        # Check final paths
        path_names = Path.columns
        
        if (path_names != Type_old.columns).any():
            for path_name in path_names:
                if not path_name in Type_old.columns:
                    raise TypeError("Agent Paths (Path) and Types (Type_old) need to have the same columns. \n However, the column name {} from Path is missing in Type_old.".format(path_name))
            for path_name in Type_old.columns:
                if not path_name in path_names:
                    raise TypeError("Agent Paths (Path) and Types (Type_old) need to have the same columns. \n However, the column name {} from Type_old is missing in Path.".format(path_name))


        for needed_agent in self.needed_agents:
            if not needed_agent in path_names:
                raise AttributeError("The scenario you set as self.scenario in self.set_scenario() requires the following agent: {}. \n However, this agent name was not found in Path.columns.".format(needed_agent))
            
        check_size = Size_old is not None
        if check_size:
            if not isinstance(Size_old, pd.core.frame.DataFrame):
                raise TypeError("Size information (Size_old) should be saved in a pandas data frame. \nInstead, type(Size_old) returns: ", type(Size_old))
            if len(Size_old) != num_samples:
                raise TypeError("Size_old does not have right number of sampels. \n len(Size_old) = {}, but the expected value is {}.".format(len(Size_old), num_samples))
            if (path_names != Size_old.columns).any():
                for path_name in path_names:
                    if not path_name in Size_old.columns:
                        raise TypeError("Agent Paths (Path) and Sizes (Size_old) need to have the same columns. \n However, the column name {} from Path is missing in Size_old.".format(path_name))
                for path_name in Size_old.columns:
                    if not path_name in path_names:
                        raise TypeError("Agent Paths (Path) and Sizes (Size_old) need to have the same columns. \n However, the column name {} from Size_old is missing in Path.".format(path_name))
            
        

        for i in range(num_samples):
            # check if time input consists out of tuples
            if not isinstance(T[i], np.ndarray):
                raise TypeError("The entries in T are expected to be np.ndarrays. \n However, type(T[{}]) returns: ".format(i), type(T[i]))

            test_length = len(T[i])

            # ensure that test_length is at least 2
            if test_length < 2:
                raise ValueError("The entries in T are expected should have at least two entries. \n However, len(T[{}]) = {}.".format(i, test_length))
                
            for j, agent in enumerate(path_names):
                # check if time input consists out of tuples
                agent_path = Path.iloc[i, j]
                agent_type = Type_old.iloc[i, j]
                if check_size:
                    agent_size = Size_old.iloc[i, j]
                
                # if the agent exists in this sample, adjust this
                if isinstance(agent_path, np.ndarray):
                    if not len(agent_path.shape) == 2:
                        raise TypeError("The entries in Path are expected to be np.ndarrays. \n However, for agent {} (i.e., j = {}), type(Path[{}][{}]) returns: ".format(agent, j, i, j), type(agent_path))
                    if (not agent_path.shape[1] == len(path_info)) or (test_length != len(agent_path)):
                        raise TypeError("The entries in Path are expected to be np.ndarrays with the right shape. \n For agent {} (i.e., j = {}), we expect that Path.iloc[{}][{}].shape = (len(T[{}]), len(self.path_data_info())) = ({}, {}), but the actual shape is ({}, {}).".format(agent, j, i, j, i, len(T[i]), len(self.path_data_info()), agent_path.shape[0], agent_path.shape[1]))
                        
                    if str(agent_type) == 'nan':
                        raise ValueError("For entries in Path that are np.ndarrays, the agent type (Type_old) should not be 'nan'. \n However, for agent {} (i.e., j = {}), Type_old.iloc[{}][{}] = 'nan' was observed.".format(agent, j, i, j))

                    if not isinstance(agent_type, str):
                        raise TypeError("For entries in Path that are np.ndarrays, the agent type (Type_old) should be a string. \n However, for agent {} (i.e., j = {}), type(Type_old.iloc[{}][{}]) returns ".format(agent, j, i, j), type(agent_type)) 
                    
                    if check_size:
                        if not isinstance(agent_size, np.ndarray):
                            raise TypeError("For entries in Path that are np.ndarrays, the agent size (Size_old) should be a np.ndarray. \n However, for agent {} (i.e., j = {}), type(Size_old.iloc[{}][{}]) returns ".format(agent, j, i, j), type(agent_size))
                        if not len(agent_size) == 2:
                            raise TypeError("For entries in Path that are np.ndarrays, the agent size (Size_old) should be a np.ndarray of length 2. \n However, for agent {} (i.e., j = {}), len(Size_old.iloc[{}][{}]) = {}.".format(agent, j, i, j, len(agent_size)))

                
                else:
                    if agent in self.needed_agents:
                        raise TypeError("The scenario you set as self.scenario in self.set_scenario() requires the agent {} (i.e., j = {}) to have a trajectory defined as an np.ndarray. \n However, type(Path[{}][{}]) returns: ".format(agent, j, i, j), type(agent_path))
            
                    if str(agent_path) != 'nan':
                        raise TypeError("For entries in Path that are not np.ndarrays, the agent path should be 'nan'. \n However, for agent {} (i.e., j = {}), Path.iloc[{}][{}] returns ".format(agent, j, i, j), agent_path)
                    
                    if str(agent_type) != 'nan':
                        raise TypeError("For entries in Path that are not np.ndarrays, the agent type (Type_old) should be 'nan'. \n However, for agent {} (i.e., j = {}), Type_old.iloc[{}][{}] returns ".format(agent, j, i, j), agent_type)

                    if check_size:
                        if str(agent_size) != 'nan':
                            raise TypeError("For entries in Path that are not np.ndarrays, the agent size (Size_old) should be 'nan'. \n However, for agent {} (i.e., j = {}), Size_old.iloc[{}][{}] returns ".format(agent, j, i, j), agent_size)

        
    def check_image_samples(self, Images):
        assert isinstance(Images, pd.core.frame.DataFrame), "Images should be saved in a pandas data frame"
        assert 'Image' in Images.columns, "Images should have a column 'Image'"

        if not hasattr(Images, 'Target_MeterPerPx'):
            if not hasattr(self, 'Target_MeterPerPx'):
                raise AttributeError('Images without Px to Meter scaling are useless.')
            else:
                Images['Target_MeterPerPx'] = self.Target_MeterPerPx

    
    def check_sceneGraph_samples(self, Images):
        assert isinstance(Images, pd.core.frame.DataFrame), "SceneGraphs should be saved in a pandas data frame"
        # TODO: Implement this function
        
        
    def check_created_paths_for_saving(self, last = False, force_save = False):
        r'''
        This function checks if the current data should be saved to free up memory.
        It should be used during the extraction of datasets too large to be held in
        memory at once.
        
        It requires the following attributes to be set:
        **self.Path**:
            This is a list of pandas series. In each such series, each index includes the 
            trajectory of an agent (as a numpy array of shape :math:`\{\vert T_i \vert{\times} 2\}`),
            where the index name should be a unique identifier for the agent.
            It is possible that positional data for an agent is only available at parts of the 
            required time points, in which cases, the missing positions should be filled up with
            (np.nan, np.nan).
        
        **self.Type_old**:
            This is a list of pandas series. In each such series, the indices should be the same as
            in the corresponding series in **self.Path**, and the values should be the agent types, 
            with four types of agents currently implemented:
                - 'V': Vehicles like cars and trucks
                - 'M': Motorcycles
                - 'B': Bicycles
                - 'P': Pedestrians
            
        **self.T**:
            This is a list of numpy arrays. It should have the same length as **self.Path** and
            **self.Type_old**. Each array should have the length :math:`\vert T_i \vert` of the
            trajecories of the agents in the corresponding series in **self.Path**. The values should
            be the time points at which the positions of the agents were recorded.
            
        **self.Domain_old**:
            This is a list of pandas series. Each series corresponds to an entry in **self.Path**, 
            and contains the metadata of the corresponding scene. The metadata can include the 
            location of the scene, or the id of the corresponding image, or the identification marker
            of the scene in the raw data.
        
        
        Parameters
        ----------
        last : bool
            If true, the last of the data was added and should therefore be saved, no matter if 
            there still is available space on the RAM. During a run of *self.create_path_samples()*, 
            *self.check_created_paths_for_saving(last = True) should only be called at the end of
            *self.create_path_samples()*, once all possible samples have been iterated through.
            
        force_save : bool
            If true, the data is saved regardless of the memory usage. This might be useful if one
            runs the script on computers with much larger RAM, but still wants compatability with
            smaller systems.
        
        '''

        if not hasattr(self, 'saved_last_orig_paths'):
            self.saved_last_orig_paths = False
        
        # Check if the four required attributes exist
        if not all([hasattr(self, attr) for attr in ['Path', 'Type_old', 'T', 'Domain_old']]):
            raise AttributeError("The required arguments for saving have not been done.")
        
        # Test if final file allready exists
        final_test_name = self.file_path + '--all_orig_paths_LLL.npy'
        if os.path.isfile(final_test_name) and self.saved_last_orig_paths:
            raise AttributeError("*last = True* was passed more than once during self.create_path_samples().")
        
        num_samples = len(self.Path)
        if num_samples > 0:
            if (np.mod(num_samples, 100) == 0) or (not hasattr(self, 'num_overall_timesteps_per_sample')):
            
                # Check if the four attributes from self.create_path_samples are lists or dataframe/arrays
                if not isinstance(self.Path, pd.core.frame.DataFrame):
                    assert isinstance(self.Path, list), "Path should be a list."
                    # Transform to dataframe
                    Path_check = pd.DataFrame(self.Path)
                else:
                    Path_check = self.Path
                    
                # Get the memory needed to save data right now
                # Count the number of timesteps in each sample
                num_timesteps = np.array([len(t) for t in self.T])
                
                # Get the number of saved agents for each sample
                num_agents = (~Path_check.isnull()).sum(axis=1)
            
                num_overall_timesteps = (num_timesteps * num_agents).sum()
                self.num_overall_timesteps_per_sample = num_overall_timesteps / num_samples
            
            assert hasattr(self, 'num_overall_timesteps_per_sample'), "The number of overall timesteps per sample should be defined."
            
            # Get the needed memory per timestep
            memory_per_timestep = 1 + 8 * len(self.path_data_info())
            memory_used = num_samples * self.num_overall_timesteps_per_sample * memory_per_timestep
            
            # Get the currently available RAM space
            available_memory = self.total_memory - get_used_memory()

            # As data needs to be manipulated after loading, check if more than 25% of the memory is used
            if force_save or last or (memory_used > 0.25 * self.available_memory_creation) or (available_memory < 100 * 2**20):
                # Check if the four attributes from self.create_path_samples are lists or dataframe/arrays
                if not isinstance(self.Path, pd.core.frame.DataFrame):
                    assert isinstance(self.Path, list), "Path should be a list."
                    # Transform to dataframe
                    Path_check = pd.DataFrame(self.Path)
                else:
                    Path_check = self.Path
                    
                if not isinstance(self.Type_old, pd.core.frame.DataFrame):
                    assert isinstance(self.Type_old, list), "Type_old should be a list."
                    # Transform to dataframe
                    Type_old_check = pd.DataFrame(self.Type_old)
                else:
                    Type_old_check = self.Type_old
                    
                if not isinstance(self.T, np.ndarray):
                    assert isinstance(self.T, list), "T should be a list."
                    # Transform to array
                    T_check = np.array(self.T) 
                else:
                    T_check = self.T
                    
                if not isinstance(self.Domain_old, pd.core.frame.DataFrame):
                    assert isinstance(self.Domain_old, list), "Domain_old should be a list."
                    # Transform to dataframe
                    Domain_old_check = pd.DataFrame(self.Domain_old)
                else:
                    Domain_old_check = self.Domain_old

                # Check is self.Size_old exists
                if hasattr(self, 'Size_old'):
                    if not isinstance(self.Size_old, pd.core.frame.DataFrame):
                        assert isinstance(self.Size_old, list), "Size_old should be a list."
                        # Transform to dataframe
                        Size_old_check = pd.DataFrame(self.Size_old)
                    else:
                        Size_old_check = self.Size_old
                else:
                    Size_old_check = None
                    if last:
                        self.Size_old = None
                    
                # Check if some saved original data is allready available
                file_path_test = self.file_path + '--all_orig_paths'
                file_path_test_name = os.path.basename(file_path_test)
                file_path_test_directory = os.path.dirname(file_path_test)
                # Find files in same directory that start with file_path_test
                files = [f for f in os.listdir(file_path_test_directory) if f.startswith(file_path_test_name)]
                if len(files) > 0:
                    # Find the number that is attached to this file
                    file_number = np.array([int(f[len(file_path_test_name)+1:-4]) for f in files], int).max() + 1
                    if file_number > 999:
                        raise AttributeError("Too many files have been saved.")
                else:
                    file_number = 0
                    
                if last:
                    # During loading of files, check for existence of last file. If not there, rerun the whole extraction procedure
                    path_addition = '_LLL.npy'
                else:
                    path_addition = '_' + str(file_number).zfill(3) + '.npy'
                file_path_save = file_path_test + path_addition   

                num_samples_check = len(Path_check)
                
                # Check the samples
                self.check_path_samples(Path_check, Type_old_check, T_check, Domain_old_check, num_samples_check, Size_old_check)

                # Sparsify the data
                Path_check_sparse = self.get_sparse_path_data(Path_check, T_check)
                
                # Save the results
                os.makedirs(os.path.dirname(file_path_save), exist_ok=True)

                if Size_old_check is None:
                    test_data = np.array([Path_check_sparse, Type_old_check, T_check, Domain_old_check, num_samples_check], object)
                else:
                    test_data = np.array([Path_check_sparse, Type_old_check, Size_old_check, T_check, Domain_old_check, num_samples_check], object)
                np.save(file_path_save, test_data)
                
                # Reset the data to empty lists
                self.Path = []
                self.Type_old = []
                self.T = []
                self.Domain_old = []
                if hasattr(self, 'Size_old'):
                    self.Size_old = []
                
                # Delete num_timesteps_per_sample
                if hasattr(self, 'num_overall_timesteps_per_sample'):
                    del self.num_overall_timesteps_per_sample

                # Check if images need to be saved
                if hasattr(self, 'map_split_save'):
                    if self.map_split_save:
                        if self.includes_images():
                            self.check_image_samples(self.Images)

                            image_file = self.file_path + '--Images' + path_addition
                            image_data = np.array([self.Images, 0], object)
                            np.save(image_file, image_data)

                            # reset self Images
                            image_columns = ['Image', 'Target_MeterPerPx']
                            self.Images = pd.DataFrame(np.zeros((0, len(image_columns)), object), index = [], columns = image_columns)
                    
                        if self.includes_sceneGraphs(): 
                            self.check_sceneGraph_samples(self.SceneGraphs)
                            
                            sceneGraph_file = self.file_path + '--SceneGraphs' + path_addition
                            sceneGraph_data = np.array([self.SceneGraphs, 0], object)
                            np.save(sceneGraph_file, sceneGraph_data)
                            
                            # reset self SceneGraphs
                            sceneGraph_columns = self.SceneGraphs.columns  
                            self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGraph_columns)), object), index = [], columns = sceneGraph_columns)

        else:
            # This can only happen when calling last = True
            assert last, "Calling the function without any data should only be possible with last = True."

            # Find the number of files allready saved
            file_path_test = self.file_path + '--all_orig_paths'
            file_path_test_name = os.path.basename(file_path_test)
            file_path_test_directory = os.path.dirname(file_path_test)
            # Find files in same directory that start with file_path_test
            files = [f for f in os.listdir(file_path_test_directory) if f.startswith(file_path_test_name)]
            assert len(files) > 0, "slef.create_path_samples() does not produce any samples."
            
            # Find the number that is attached to this file
            file_number_overwrite = np.array([int(f[len(file_path_test_name)+1:-4]) for f in files], int).max()
            file_overwrite_old = file_path_test + '_' + str(file_number_overwrite).zfill(3) + '.npy'
            file_overwrite_new = file_path_test + '_LLL.npy'

            # Rename the file
            os.rename(file_overwrite_old, file_overwrite_new)

        if last:
            self.saved_last_orig_paths = True
                

    def get_sparse_path_data(self, Path, T):
        # Check identical length of inputs
        assert len(Path) == len(T), "Input lengths should be the same"
        # Prepare sparse pandas dataframe
        Path_helper = pd.DataFrame(np.zeros((0, 3 + len(self.path_data_info())), object), 
                                   columns = ['sample_index', 'agent_index', 'time_index'] + self.path_data_info())
        
        # Get num timesteps
        Num_timesteps = np.array([len(t) for t in T])
        unique_num_timesteps = np.unique(Num_timesteps)
        
        print("Transform {} paths to sparse format".format(len(Path)), flush=True)
        # Go through unique number of timesteps
        for i, num_timesteps in enumerate(unique_num_timesteps):
            used_samples = np.where(num_timesteps == Num_timesteps)[0]
            print("Step {} of {}: Transform the {} samples with {} timesteps".format(i+1, len(unique_num_timesteps), len(used_samples), num_timesteps), flush=True) 
            
            # Get the paths with this specific number of timesteps
            Path_samples = Path.iloc[used_samples]
            Path_samples_non_nan = ~Path_samples.isna()
            
            # Get the actually existing agents
            Sample_id, Agent_id = np.where(Path_samples_non_nan) # num_agents
            Sample_id_global = used_samples[Sample_id]
            
            # TODO: Check this line for potential correction
            Paths_useful = np.stack(list(Path_samples.values[Sample_id, Agent_id]), axis = 0) # num_agents x num_timesteps x num_data
            
            # Get existing timesteps of existing agents 
            useful_agents, useful_timesteps = np.where(np.isfinite(Paths_useful).any(-1))
            
            # Transform useful_agents_id onto oringinal sample/agent ids
            sample_id = Sample_id_global[useful_agents]
            agent_id  = Agent_id[useful_agents]
            
            # get num recordings
            num_recordings = len(useful_timesteps)
            locs = len(Path_helper) + np.arange(num_recordings)
            
            # Initialize new columns
            Path_helper = Path_helper.reindex(np.concatenate([Path_helper.index, locs]))
            
            # Write in the specific indices
            Path_helper.loc[locs, 'sample_index'] = sample_id
            Path_helper.loc[locs, 'agent_index'] = agent_id
            Path_helper.loc[locs, 'time_index'] = useful_timesteps
            
            # Write in the specific data
            Path_helper.loc[locs, self.path_data_info()] = Paths_useful[useful_agents, useful_timesteps]
        
        print("Transformed paths to sparse format", flush=True)
        return Path_helper
    
    def get_multiindex_path(self, Path):
        # Get max number of agents
        max_agents = Path['agent_index'].max() + 1
        max_timesteps = Path['time_index'].max() + 1
        id = Path['sample_index'] + Path['agent_index'] / max_agents + Path['time_index'] / max_agents / max_timesteps

        id_sort = np.argsort(id)
        Path_sparse = Path.iloc[id_sort].set_index(['sample_index', 'agent_index', 'time_index'])
        return Path_sparse

    def get_dense_path_sample(self, Path_sparse, sample_index, agent_name_array, num_timesteps):
        path_sparse = Path_sparse.loc[sample_index]

        # Get pandas dataframe
        path_data_dense = np.full((len(agent_name_array), num_timesteps, len(self.path_data_info())), np.nan, dtype = np.float32)

        # Transform sparse data to dense data
        path_data_sparse = path_sparse.to_numpy().astype(np.float32) # num_useful x n_data
        agent_ind, time_ind = path_sparse.index.get_level_values(0), path_sparse.index.get_level_values(1)
        path_data_dense[agent_ind, time_ind] = path_data_sparse

        # Map onto pandas series
        used_agents = np.unique(agent_ind)
        used_agents_name = agent_name_array[used_agents]
        path_data_dense_used = list(path_data_dense[used_agents])

        # Transform to pandads series
        path = pd.Series(path_data_dense_used, index = used_agents_name)
        
        # Add missing agents
        path = path.reindex(agent_name_array)

        return path


    def get_number_of_saved_samples(self):
        r'''
        This function returns the number of samples that have been saved so far
        by the framework. Assuming the order of the samples is not changed, this
        number can be used to skip the first *num_saved_samples* samples in the
        next run of *self.create_path_samples()*.
        
        Returns
        -------
        num_samples : int
            The number of samples that have been saved so far.
        '''
        
        test_file = self.file_path + '--all_orig_paths'
        
        test_file_directory = os.path.dirname(test_file)
        test_file_name = os.path.basename(test_file)

        os.makedirs(test_file_directory, exist_ok = True)
        
        # Find files in same directory that start with file_path_test
        files = [f for f in os.listdir(test_file_directory) if f.startswith(test_file_name)]
        num_samples = 0
        for file in files:
            file_path = test_file_directory + os.sep + file
            num_samples_file = np.load(file_path, allow_pickle=True)[-1]
            num_samples += num_samples_file
        
        return num_samples
                   
            
        
    
    def get_number_of_original_path_files(self):
        # Get name of final file
        test_file = self.file_path + '--all_orig_paths_LLL.npy'
        
        # Check if file exists
        if not os.path.isfile(test_file):
            raise AttributeError("The data has not been completely extracted yet.")
        
        # Get the corresponding directory
        test_file_tester = self.file_path + '--all_orig_paths'
        test_file_directory = os.path.dirname(test_file)
        
        # Find files in same directory that start with file_path_test
        num_files = len([f for f in os.listdir(test_file_directory) if f.startswith(os.path.basename(test_file_tester))])
        return num_files
    

    def extract_loaded_data(self, Loaded_data):
        if len(Loaded_data) == 5:
            [Path, Type_old, T, Domain_old, num_samples] = Loaded_data
            Size_old = None
        else:
            assert len(Loaded_data) == 6, "The loaded data should have 5 or 6 elements."
            [Path, Type_old, Size_old, T, Domain_old, num_samples] = Loaded_data

        # Make backwards compatible:
        # Check if Path is sparse
        sparse_columns = ['sample_index', 'agent_index', 'time_index'] + self.path_data_info()
        dense_columns = Type_old.columns

        sparse = all([col in Path.columns for col in sparse_columns])
        dense = all([col in Path.columns for col in dense_columns])

        # Ensure that always one of the two is true
        assert sparse != dense, "Path data should be either sparse or dense."

        if dense:
            Path = self.get_sparse_path_data(Path, T)
        
        return Path, Type_old, Size_old, T, Domain_old, num_samples



    def load_raw_data(self):
        if not self.raw_data_loaded:
            # If extraction was successful, this file should exist.
            test_file = self.file_path + '--all_orig_paths_LLL.npy'
            image_file1 = self.file_path + '--Images.npy'
            image_file2 = self.file_path + '--Images_LLL.npy'
            sceneGraph_file1 = self.file_path + '--SceneGraphs.npy'
            sceneGraph_file2 = self.file_path + '--SceneGraphs_LLL.npy'

            # Check if map files would be available
            image_available = os.path.isfile(image_file1) or os.path.isfile(image_file2)
            sceneGraph_available = os.path.isfile(sceneGraph_file1) or os.path.isfile(sceneGraph_file2)
            
            image_creation_unneeded = (not self.includes_images()) or image_available
            sceneGraph_creation_unneeded = (not self.includes_sceneGraphs()) or sceneGraph_available

            if os.path.isfile(test_file) and image_creation_unneeded and sceneGraph_creation_unneeded:
                # Get number of files used for saving
                self.number_original_path_files = self.get_number_of_original_path_files()
                
                if self.number_original_path_files == 1:
                    # Allready load samples for higher efficiency
                    Loaded_data = np.load(test_file, allow_pickle=True)
                    self.Path, self.Type_old, self.Size_old, self.T, self.Domain_old, self.num_samples = self.extract_loaded_data(Loaded_data)
            else:
                if not all([hasattr(self, attr) for attr in ['create_path_samples']]):
                    raise AttributeError("The raw data cannot be loaded.")
                
                # Get the currently available RAM space
                self.available_memory_creation = self.total_memory - get_used_memory()

                self.create_path_samples()

                # Check if the last file allready exists
                if os.path.isfile(test_file):
                    self.number_original_path_files = self.get_number_of_original_path_files()
                    # If there is only one file, load the data
                    if self.number_original_path_files == 1:
                        Loaded_data = np.load(test_file, allow_pickle=True)
                        self.Path, self.Type_old, self.Size_old, self.T, self.Domain_old, self.num_samples = self.extract_loaded_data(Loaded_data)
                    
                else:
                    # Check that no other save files exists
                    test_file_tester = self.file_path + '--all_orig_paths'
                    test_file_directory = os.path.dirname(test_file)
                    # Find files in same directory that start with file_path_test
                    os.makedirs(test_file_directory, exist_ok=True)
                    num_files = len([f for f in os.listdir(test_file_directory) if f.startswith(os.path.basename(test_file_tester))])
                    if num_files > 0:
                        raise AttributeError("Incomplete use of self.check_created_paths_for_saving.")
                    
                    self.number_original_path_files = 1

                    # Check if self.Size_old exists
                    if not hasattr(self, 'Size_old'):
                        self.Size_old = None
                    
                    # Validate the data                    
                    self.check_path_samples(self.Path, self.Type_old, self.T, self.Domain_old, self.num_samples, self.Size_old)
                
                    # save the results
                    os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

                    self.Path = self.get_sparse_path_data(self.Path, self.T)
                    
                    if self.Size_old is None:
                        test_data = np.array([self.Path, self.Type_old, self.T, self.Domain_old, self.num_samples], object)
                    else:
                        test_data = np.array([self.Path, self.Type_old, self.Size_old, self.T, self.Domain_old, self.num_samples], object)
                    
                    np.save(test_file, test_data)
                
                # Check if data needs to be saved:
                if (not hasattr(self, 'map_split_save')) or (not self.map_split_save):
                    # Save the image data
                    if self.includes_images():
                        if not hasattr(self, 'Images'):
                            raise AttributeError('Images are missing.')
                        
                        self.check_image_samples(self.Images)

                        image_data = np.array([self.Images, 0], object)
                        np.save(image_file1, image_data)

                    
                    # Save the sceneGraph data
                    if self.includes_sceneGraphs():
                        if not hasattr(self, 'SceneGraphs'):
                            raise AttributeError('SceneGraphs are missing.')
                        
                        self.check_sceneGraph_samples(self.SceneGraphs)
                        sceneGraph_data = np.array([self.SceneGraphs, 0], object)
                        np.save(sceneGraph_file1, sceneGraph_data)
                    
            self.raw_data_loaded = True
            
    def load_raw_images(self, path_addition = None):
        if self.includes_images():
            if hasattr(self, 'path_addition_image_old'):
                if self.path_addition_image_old == path_addition:
                    return

            image_file_test_1 = self.file_path + '--Images.npy'
            image_file_test_2 = self.file_path + '--Images_LLL.npy'

            # Check if they exist
            test_1_exists = os.path.isfile(image_file_test_1)
            test_2_exists = os.path.isfile(image_file_test_2)

            # if both not exist, load raw data
            if not (test_1_exists or test_2_exists):
                self.load_raw_data()

            # Recheck which file exists
            test_1_exists = os.path.isfile(image_file_test_1)
            test_2_exists = os.path.isfile(image_file_test_2)

            # Now, one them should exist, but not both
            assert test_1_exists != test_2_exists, "Only one of the two image files should exist."

            if test_1_exists:
                if not hasattr(self, 'Images'):
                    [self.Images, _] = np.load(image_file_test_1, allow_pickle=True)
            else:
                assert path_addition is not None, "The path addition is needed to load the correct file."
                image_file = self.file_path + '--Images' + path_addition + '.npy'
                [self.Images, _] = np.load(image_file, allow_pickle=True)

            self.path_addition_image_old = path_addition


    def load_raw_sceneGraphs(self, path_addition = None):
        if self.includes_sceneGraphs():
            if hasattr(self, 'path_addition_scenegraph_old'):
                if self.path_addition_scenegraph_old == path_addition:
                    return
            
            sceneGraph_file_test_1 = self.file_path + '--SceneGraphs.npy'
            sceneGraph_file_test_2 = self.file_path + '--SceneGraphs_LLL.npy'

            # Check if they exist
            test_1_exists = os.path.isfile(sceneGraph_file_test_1)
            test_2_exists = os.path.isfile(sceneGraph_file_test_2)

            # if both not exist, load raw data
            if not (test_1_exists or test_2_exists):
                self.load_raw_data()

            # Recheck which file exists
            test_1_exists = os.path.isfile(sceneGraph_file_test_1)
            test_2_exists = os.path.isfile(sceneGraph_file_test_2)

            # Now, one them should exist, but not both
            assert test_1_exists != test_2_exists, "Only one of the two sceneGraph files should exist."

            if test_1_exists:
                if not hasattr(self, 'SceneGraphs'):
                    [self.SceneGraphs, _] = np.load(sceneGraph_file_test_1, allow_pickle=True)
            else:
                assert path_addition is not None, "The path addition is needed to load the correct file."
                sceneGraph_file = self.file_path + '--SceneGraphs' + path_addition + '.npy'
                [self.SceneGraphs, _] = np.load(sceneGraph_file, allow_pickle=True)
            
            self.path_addition_scenegraph_old = path_addition

    
    def reset(self):
        self.data_loaded = False
        self.path_models_trained = False

    def getSceneGraphTrajdata(self, map_image):
        min_x, min_y = map_image.extent[:2]
        image_bounds = np.array([[min_x, min_y]])
        
        # prepare the sceneGraph
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

        # Get lane_ids
        map_lanes_tokens = {}
        for i, lane in enumerate(map_image.lanes):
            map_lanes_tokens[lane.id] = i

        # Assert tyhat all pre stuff us empty
        no_pre = True
        for lane_record in map_image.lanes:
            if len(lane_record.prev_lanes) > 0:
                no_pre = False
                break

        for lane_record in map_image.lanes:
            token = lane_record.id
            lane_id = map_lanes_tokens[token]
            center_pts = lane_record.center.points[:,:2]
            left_pts = lane_record.left_edge.points[:,:2]
            right_pts = lane_record.right_edge.points[:,:2]

            # Subtract image bounds from positions
            center_pts -= image_bounds
            left_pts -= image_bounds
            right_pts -= image_bounds

            # Mirror along y axis
            center_pts[:, 1] *= -1
            left_pts[:, 1] *= -1
            right_pts[:, 1] *= -1

            # Append lane markers
            centerlines.append(center_pts)
            left_boundaries.append(left_pts)
            right_boundaries.append(right_pts)

            # Append lane_idc
            lane_idcs += [lane_id] * (len(center_pts) - 1)
            num_nodes += len(center_pts) - 1

            # Get lane type
            # TODO: Determine intersections
            lane_type.append(('VEHICLE', False))

            # Get predecessor and successor connections
            if no_pre:
                for suc in lane_record.next_lanes:
                    suc_pairs = np.vstack((suc_pairs, [lane_id, map_lanes_tokens[suc]]))
                    pre_pairs = np.vstack((pre_pairs, [map_lanes_tokens[suc], lane_id]))

            else:
                for pre in lane_record.prev_lanes:
                    pre_pairs = np.vstack((pre_pairs, [lane_id, map_lanes_tokens[pre]]))

                for suc in lane_record.next_lanes:
                    suc_pairs = np.vstack((suc_pairs, [lane_id, map_lanes_tokens[suc]]))

            for left in lane_record.adj_lanes_left:
                left_pairs = np.vstack((left_pairs, [lane_id, map_lanes_tokens[left]]))

            for right in lane_record.adj_lanes_right:
                right_pairs = np.vstack((right_pairs, [lane_id, map_lanes_tokens[right]]))

        # Initialize graph
        graph = pd.Series([])
        graph['num_nodes'] = num_nodes
        graph['lane_idcs'] = np.array(lane_idcs)
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = right_pairs # The use of * -1 mirrors everything, switching sides
        graph['right_pairs'] = left_pairs # The use of * -1 mirrors everything, switching sides
        graph['left_boundaries'] = right_boundaries # The use of * -1 mirrors everything, switching sides
        graph['right_boundaries'] = left_boundaries # The use of * -1 mirrors everything, switching sides
        graph['centerlines'] = centerlines
        graph['lane_type'] = lane_type

        # Get available gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph = self.add_node_connections(graph, device = device)

        return graph



    def getSceneGraphNuScenes(self, map_image):
        from NuScenes.nusc_utils import extract_lane_and_edges, extract_lane_center, extract_area

        # Prepare transformation of lane types to argoverse style
        lane_user_dict = {'CAR': 'VEHICLE',
                          'TRUCK': 'VEHICLE',
                          'BUS': 'BUS'}

        # Setting the map bounds.
        min_x, min_y = map_image.explorer.canvas_min_x, map_image.explorer.canvas_min_y
        image_bounds = np.array([[min_x, min_y]])

        # prepare the sceneGraph
        num_nodes = 0
        lane_idcs = []
        pre_pairs = np.zeros((0,2), int)
        suc_pairs = np.zeros((0,2), int)

        left_boundaries = []
        right_boundaries = []
        centerlines = []

        lane_type = []

        # Get list of tokens
        map_lanes_tokens = {}
        i_counter = 0
        for i, lane in enumerate(map_image.lane):
            map_lanes_tokens[lane["token"]] = i
        i_counter += len(map_image.lane)

        map_connector_tokens = {}
        for i, lane in enumerate(map_image.lane_connector):
            map_connector_tokens[lane["token"]] = i + i_counter
        i_counter += len(map_image.lane_connector)

        map_ped_area_tokens = {}
        for i, lane in enumerate(map_image.ped_crossing):
            map_ped_area_tokens[lane["token"]] = i + i_counter
        i_counter += len(map_image.ped_crossing)

        for i, lane in enumerate(map_image.walkway):
            map_ped_area_tokens[lane["token"]] = i + i_counter
        
        # Go through lanes
        for lane_record in map_image.lane:
            # Get points describing the lane aspect
            center_pts, left_pts, right_pts = extract_lane_and_edges(map_image, lane_record)

            # Subtract image bounds from positions
            center_pts -= image_bounds
            left_pts -= image_bounds
            right_pts -= image_bounds

            # Mirror along y axis
            center_pts[:, 1] *= -1
            left_pts[:, 1] *= -1
            right_pts[:, 1] *= -1

            # Append lane markers
            centerlines.append(center_pts)
            left_boundaries.append(left_pts)
            right_boundaries.append(right_pts)

            # Get lane id
            lane_id = map_lanes_tokens[lane_record['token']]

            # Append lane_idc
            lane_idcs += [lane_id] * (len(center_pts) - 1)
            num_nodes += len(center_pts) - 1

            # Get lane type (lanes are no intersections, compared to lane connectors)
            lane_user = lane_record['lane_type']
            lane_type.append((lane_user_dict[lane_user], False))

            lane_record_token: str = lane_record["token"]

            pre_suc_connections = map_image.connectivity[lane_record_token]
            for pre in pre_suc_connections['incoming']:
                if pre in map_lanes_tokens:
                    pre_pairs = np.vstack((pre_pairs, [lane_id, map_lanes_tokens[pre]]))
                elif pre in map_connector_tokens:
                    pre_pairs = np.vstack((pre_pairs, [lane_id, map_connector_tokens[pre]]))
                    suc_pairs = np.vstack((suc_pairs, [map_connector_tokens[pre], lane_id]))
                else:
                    pass
            for suc in pre_suc_connections['outgoing']:
                if suc in map_lanes_tokens:
                    suc_pairs = np.vstack((suc_pairs, [lane_id, map_lanes_tokens[suc]]))
                elif suc in map_connector_tokens:
                    suc_pairs = np.vstack((suc_pairs, [lane_id, map_connector_tokens[suc]]))
                    pre_pairs = np.vstack((pre_pairs, [map_connector_tokens[suc], lane_id]))
                else:
                    pass
        
        # Get left and right pairs
        left_pairs, right_pairs = self.get_Left_Right_pairs(centerlines)

        for lane_record in map_image.lane_connector:
            # Unfortunately lane connectors in nuScenes have very simple exterior
            # polygons which make extracting their edges quite difficult, so we
            # only extract the centerline.
            center_pts = extract_lane_center(map_image, lane_record)

            # Subtract image bounds from positions
            center_pts -= image_bounds

            # Mirror along y axis
            center_pts[:, 1] *= -1

            # Append lane markers
            centerlines.append(center_pts)
            right_boundaries.append(np.zeros((0, 2)))
            left_boundaries.append(np.zeros((0, 2)))

            # Get lane id
            lane_id = map_connector_tokens[lane_record['token']]
            lane_idcs += [lane_id] * (len(center_pts) - 1)

            num_nodes += len(center_pts) - 1

            # Get lane type (lanes are no intersections, compared to lane connectors)
            lane_type.append(('VEHICLE', True))

        for ped_area_record in map_image.ped_crossing:
            polygon_pts = extract_area(map_image, ped_area_record)
            assert len(polygon_pts) >= 4, "Pedestrian crossing should have 4 points."

            center_pts, left_pts, right_pts = self.extract_polygon(polygon_pts, image_bounds)

            # Append lane markers
            centerlines.append(center_pts)
            left_boundaries.append(left_pts)
            right_boundaries.append(right_pts)

            # Get lane id
            lane_id = map_ped_area_tokens[ped_area_record['token']]
            lane_idcs += [lane_id] * (len(center_pts) - 1)

            num_nodes += len(center_pts) - 1

            # Get lane type (crosswalks are intersections, compared to walkways)
            lane_type.append(('PEDESTRIAN', True))

        for ped_area_record in map_image.walkway:
            polygon_pts = extract_area(map_image, ped_area_record)
            assert len(polygon_pts) >= 4, "Walkway should have at least 4 points."

            center_pts, left_pts, right_pts = self.extract_polygon(polygon_pts, image_bounds)

            # Append lane markers
            centerlines.append(center_pts)
            left_boundaries.append(left_pts)
            right_boundaries.append(right_pts)

            # Get lane id
            lane_id = map_ped_area_tokens[ped_area_record['token']]
            lane_idcs += [lane_id] * (len(center_pts) - 1)

            num_nodes += len(center_pts) - 1

            # Get lane type (walkways are no intersections, compared to crosswalks)
            lane_type.append(('PEDESTRIAN', False))


        # Check some things
        num_segments = len(centerlines)
        assert len(lane_idcs) == num_nodes, "Number of lane idcs should be equal to number of nodes."
        assert len(left_boundaries) == num_segments, "Number of left boundaries should be equal to number of segments."
        assert len(right_boundaries) == num_segments, "Number of right boundaries should be equal to number of segments."
        assert len(np.unique(lane_idcs)) == num_segments, "Multiple segements have identical lane idcs."


        graph = pd.Series([])
        graph['num_nodes'] = num_nodes
        graph['lane_idcs'] = np.array(lane_idcs)
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        graph['left_boundaries'] = right_boundaries # The use of * -1 mirrors everything, switching sides
        graph['right_boundaries'] = left_boundaries # The use of * -1 mirrors everything, switching sides
        graph['centerlines'] = centerlines
        graph['lane_type'] = lane_type


        # Get available gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph = self.add_node_connections(graph, device = device)

        return graph
    

    def get_Left_Right_pairs(self, centerlines):
        # get lane end and start points
        lane_start = np.zeros((0, 2))
        lane_end = np.zeros((0, 2))
        for center_pts in centerlines:
            # Append lane end and start points
            lane_start = np.vstack((lane_start, center_pts[0]))
            lane_end = np.vstack((lane_end, center_pts[-1]))

        D_start_start = np.linalg.norm(lane_start[:, np.newaxis] - lane_start[np.newaxis], axis = -1)
        D_end_end   = np.linalg.norm(lane_end[:, np.newaxis] - lane_end[np.newaxis], axis = -1)
        D_start_end = np.linalg.norm(lane_start[:, np.newaxis] - lane_end[np.newaxis], axis = -1)
        D_start_end = np.minimum(D_start_end, D_start_end.T)
        D_start_end = np.minimum(D_start_end, 15)

        # Get angle between norm vectors
        Norm_vec = lane_end - lane_start
        Angle = np.arctan2(Norm_vec[:, 1], Norm_vec[:, 0])
        D_angle = np.abs(Angle[:, np.newaxis] - Angle[np.newaxis])

        # Fit D_angle to be between -pi and pi
        D_angle = np.mod(D_angle + np.pi, 2 * np.pi) - np.pi

        potential_pair = (0<= D_angle) & (D_angle < np.pi / 18) & ((D_start_start < D_start_end) | (D_end_end < D_start_end))
        # Set main diagonal to False
        np.fill_diagonal(potential_pair, False)

        # Find subgraphs in potential pairs
        G = nx.Graph(potential_pair)
        unconnected_subgraphs = list(nx.connected_components(G))
        for subgraph in unconnected_subgraphs:
            if len(subgraph) <= 2:
                continue
            subgraph = np.array(list(subgraph))

            # Get the centerlines of the lanes
            C = [centerlines[i] for i in subgraph]

            # Get shortest distance between the lanes
            D = np.zeros((len(subgraph), len(subgraph)))
            for i in range(len(subgraph)):
                for j in range(i + 1, len(subgraph)):
                    D[i, j] = np.min(np.linalg.norm(C[i][:, np.newaxis] - C[j][np.newaxis], axis = -1))
                    D[j, i] = D[i, j]
            
            max_D = np.max(D) + 1
            np.fill_diagonal(D, max_D)

            # Only allow connections, where the distance is either the shortest or second shortest distance
            if len(subgraph) == 3:
                D_div = np.sort(D, axis = -1)[:, [0]]
                D_connect = D <= D_div
                D_connect = D_connect | D_connect.T
            else:
                D_div = np.sort(D, axis = -1)[:, [1]]
                D_connect = D <= D_div
                D_connect = D_connect & D_connect.T

            potential_pair[subgraph[:,np.newaxis], subgraph[np.newaxis]] = D_connect

        np.fill_diagonal(potential_pair, False)

        I1, I2 = np.where(potential_pair)

        assert (potential_pair == potential_pair.T).all(), "Potential pair matrix should be symmetric."

        # Get angle between lanes to see what is left pair and right pair
        Dref = lane_end[I2] - lane_start[I1]
        angle_ref = np.arctan2(Dref[:, 1], Dref[:, 0])
        D_angle_ref = angle_ref - Angle[I1]
        D_angle_ref = np.mod(D_angle_ref + np.pi, 2 * np.pi) - np.pi

        # get pairs where I2 is left of I1
        left_pair = D_angle_ref > 0

        potential_pair = potential_pair.astype(float)
        potential_pair[I1[left_pair], I2[left_pair]] = -1.0

        assert potential_pair.sum() == 0, "All potential pairs should be used."
        assert ((potential_pair + potential_pair.T) == 0).all(), "All potential pairs should be used."

        I1_left = I1[left_pair]
        I2_left = I2[left_pair]

        I1_right = I1[~left_pair]
        I2_right = I2[~left_pair]

        # Sort right by I2
        ind_sort = np.argsort(I2_right)
        I1_right = I1_right[ind_sort]
        I2_right = I2_right[ind_sort]

        left_pairs = np.stack((I1_left, I2_left), axis = -1)
        right_pairs = np.stack((I1_right, I2_right), axis = -1)

        return left_pairs, right_pairs


    def extract_polygon(self, polygon_pts, image_bounds):            
        # Find line furthes away from each other
        polygon_closed = np.concatenate((polygon_pts, polygon_pts[[0]]), 0)
        center = 0.5 * (polygon_closed[:-1] + polygon_closed[1:])

        # Get distances between the center points
        D_center = np.linalg.norm(center[np.newaxis] - center[:, np.newaxis], axis = -1)

        # Get the two points with the largest distance
        max_idx = np.unravel_index(np.argmax(D_center), D_center.shape)

        # Left is counting up along polygon_closed
        left_start = max_idx[0] + 1
        left_end = max_idx[1]
        if left_end == 0:
            left_end = len(polygon_closed) - 1

        # Right is counting down along polygon_closed
        right_start = max_idx[0]
        if right_start == 0:
            right_start = len(polygon_closed) - 1
        right_end = max_idx[1] + 1

        # Get the left and right points, interpolate between points two far away
        left_pts = []
        for i in range(left_start, left_end):
            dist = np.linalg.norm(polygon_closed[i] - polygon_closed[i + 1], -1)
            num_nodes_needed = max(2, np.ceil(dist).astype(int))

            interpolator = np.linspace(0, 1, num_nodes_needed)[:-1, np.newaxis]

            left_pts += list((1 - interpolator) * polygon_closed[[i]] + interpolator * polygon_closed[[i + 1]])
        # Add last point
        left_pts.append(polygon_closed[left_end])
        left_pts = np.array(left_pts)

        right_pts = []
        if right_end > right_start:
            right_end -= len(center)
            
        for i in range(right_start, right_end, -1):
            dist = np.linalg.norm(polygon_closed[i] - polygon_closed[i - 1], -1)
            num_nodes_needed = max(2, np.ceil(dist).astype(int))

            interpolator = np.linspace(0, 1, num_nodes_needed)[:-1, np.newaxis]

            right_pts += list((1 - interpolator) * polygon_closed[[i]] + interpolator * polygon_closed[[i - 1]])
        # Add last point
        right_pts.append(polygon_closed[right_end])
        right_pts = np.array(right_pts)

        # Get center points
        # Get distances between left and right points
        D_left_right = np.linalg.norm(left_pts[np.newaxis] - right_pts[:, np.newaxis], axis = -1)

        # Get the shorter side to be first dimension
        if D_left_right.shape[0] > D_left_right.shape[1]:
            D_left_right = D_left_right.T
            row_pts = left_pts
            col_pts = right_pts
        else:
            row_pts = right_pts
            col_pts = left_pts
        
        # For each row, get the index with the closest distance
        closest_idx = np.argmin(D_left_right, axis = 1)

        # Get center points
        center_pts = 0.5 * (row_pts + col_pts[closest_idx])

        # Subtract image bounds from positions
        center_pts -= image_bounds
        left_pts -= image_bounds
        right_pts -= image_bounds

        # Mirror along y axis
        center_pts[:, 1] *= -1
        left_pts[:, 1] *= -1
        right_pts[:, 1] *= -1
        
        assert len(center_pts) > 1, "Centerline should have at least 2 points."
        return center_pts, left_pts, right_pts


    def add_node_connections(self, graph, scales = [2, 4, 8, 16, 32], cross_dist = 6, cross_angle = 0.5 * np.pi, device = 'cpu'):
        '''
        This function adds node connections to the graph. 
        
        graph : pandas.Series 
            A pandas.Series representing the scene graph. It should contain the following entries:    
        
                num_nodes         - number of nodes in the scene graph.

                lane_idcs         - indices of the lane segments in the scene graph; array of length :math:`num_{nodes}`
                                    with *lane_idcs.max()* :math:`= num_{lanes} - 1`.

                pre_pairs         - array with shape :math:`\{num_{lane pre} {\times} 2\}` lane_idcs pairs where the
                                    first value of the pair is the source lane index and the second value is source's
                                    predecessor lane index.

                suc_pairs         - array with shape :math:`\{num_{lane suc} {\times} 2\}` lane_idcs pairs where the
                                    first value of the pair is the source lane index and the second value is source's
                                    successor lane index.

                left_pairs        - array with shape :math:`\{num_{lane left} {\times} 2\}` lane_idcs pairs where the
                                    first value of the pair is the source lane index and the second value is source's
                                    left neighbor lane index.

                right_pairs       - array with shape :math:`\{num_{lane right} {\times} 2\}` lane_idcs pairs where the
                                    first value of the pair is the source lane index and the second value is source's
                                    right neighbor lane index.

                left_boundaries   - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                                    :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                                    of points needed to describe the left boundary in travel direction of the current lane.
                                    Here, :math:`num_{nodes,l} = ` *(lane_idcs == l).sum()*. 
                                        
                right_boundaries  - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                                    :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                                    of points needed to describe the right boundary in travel direction of the current lane.

                centerlines       - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                                    :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                                    of points needed to describe the middle between the left and right boundary in travel
                                    direction of the current lane.
        
                lane_type         - an array with length :math:`num_{lanes}`, whose elements are tuples with the length :math:`2`,
                                    where the first element is a string that is either *'VEHILCE'*, '*BIKE*', or '*BUS*', and the second
                                    entry is a boolean, which is true if the lane segment is part of an intersection.

        scales : list
            A list of scales for neighbor dillation as per the implementation in LaneGCN. The scales should be strictly
            monotonically increasing. The first element should be larger than 1.

        cross_dist : float
            The distance at which two nodes are considered to be connected in the cross direction.

        cross_angle : float
            The angle at which two nodes are considered to be connected in the cross direction.

        device : str or torch.device
            The device on which the data should be stored. It can be either 'cpu' or a torch.device object.


        Returns
        -------
        graph : pandas.Series
            The updated scene graph. The following entries are added to the graph:

                ctrs    - array with shape :math:`\{num_{nodes} {\times} 2\}` where the entries represent locations between 
                          the centerline segments

                feats   - array with shape :math:`\{num_{nodes} {\times} 2\}` where the entries represent the offsets between
                          the centerline segments

                pre     - predecessor nodes of each node in the scene graph;
                          list of dictionaries where the length of the list is equal to *len(scales) + 1*, as per the 
                          implementation in LaneGCN. 
                          Each dictionary contains the keys 'u' and 'v', where 'u' is the *node index* of the source node and
                          'v' is the index of the target node giving edges pointing from a given source node 'u' to its
                          predecessor.

                suc     - successor nodes of each node in the scene graph;
                          list of dictionaries where the length of the list is equal to *len(scales) + 1*, as per the 
                          implementation in LaneGCN. 
                          Each dictionary contains the keys 'u' and 'v', where 'u' is the *node index* of the source node and
                          'v' is the index of the target node giving edges pointing from a given source node 'u' to its
                          successor.

                left    - left neighbor nodes of each node in the scene graph;
                          list with length 1 containing a dictionary with the keys 'u' and 'v', where 'u' is the *node index* of 
                          the source node and 'v' is the index of the target node giving edges pointing from a given source node 
                          'u' to its left neighbor.

                right   - right neighbor nodes of each node in the scene graph;
                          list with length 1 containing a dictionary with the keys 'u' and 'v', where 'u' is the *node index* of 
                          the source node and 'v' is the index of the target node giving edges pointing from a given source node 
                          'u' to its right neighbor.
                                

        
        '''
        graph_indices = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type']   

        assert isinstance(graph, pd.Series)
        assert np.in1d(graph_indices, graph.index).all()

        # Check scales (shoud be sorted, and the first element should be larger than 1)
        assert np.all(np.diff(scales) > 0)
        assert scales[0] > 1

        # Check if any nodes exist in the graph
        assert len(graph.lane_idcs) == graph.num_nodes

        if graph.num_nodes == 0:
            pre, suc, left, right = dict(), dict(), dict(), dict()
            for key in ['u', 'v']:
                pre[key], suc[key], left[key], right[key] = [], [], [], []

            graph['ctrs'] = np.zeros((0, 2), np.float32)
            graph['feats'] = np.zeros((0, 2), np.float32)

            graph['pre']   = [pre] * (len(scales) + 1)
            graph['suc']   = [suc] * (len(scales) + 1)
            graph['left']  = [left]
            graph['right'] = [right]

            return graph
        ##################################################################################
        #              Make checks on the graph data                                     #
        ##################################################################################

        unique_lane_segments = list(np.unique(graph.lane_idcs))
        num_segments = len(unique_lane_segments)

        if len(graph.pre_pairs) > 0:
            assert graph.pre_pairs.max() < num_segments
        if len(graph.suc_pairs) > 0:
            assert graph.suc_pairs.max() < num_segments
        if len(graph.left_pairs) > 0:
            assert graph.left_pairs.max() < num_segments
        if len(graph.right_pairs) > 0:
            assert graph.right_pairs.max() < num_segments

        assert len(graph.left_boundaries) == num_segments
        assert len(graph.right_boundaries) == num_segments
        assert len(graph.centerlines) == num_segments
        assert len(graph.lane_type) == num_segments
        assert graph.lane_idcs.max() < num_segments



        ##################################################################################
        # Add node connections                                                           #
        ##################################################################################

        ctrs  = np.zeros((graph.num_nodes, 2), np.float32)
        feats = np.zeros((graph.num_nodes, 2), np.float32)

        node_idcs = []

        for i, lane_segment in enumerate(unique_lane_segments):  
            lane_ind = np.where(graph.lane_idcs == lane_segment)[0]
            node_idcs.append(lane_ind)

            centerline = graph.centerlines[lane_segment]

            assert len(centerline) == len(lane_ind) + 1
            ctrs[lane_ind]  = np.asarray((centerline[:-1] + centerline[1:]) * 0.5, np.float32)
            feats[lane_ind] = np.asarray(centerline[1:] - centerline[:-1], np.float32)
        
        graph['ctrs'] = ctrs
        graph['feats'] = feats

        ##################################################################################
        # Add predecessors and successors                                                #
        ##################################################################################

        # predecessors and successors of a lane
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []

        for i, lane_segment in enumerate(unique_lane_segments):
            idcs = node_idcs[i]

            # points to the predecessor
            pre['u'] += list(idcs[1:])
            pre['v'] += list(idcs[:-1])

            # Get lane predecessores
            lane_pre = graph.pre_pairs[graph.pre_pairs[:, 0] == lane_segment, 1]
            for lane_segment_pre in lane_pre:
                if lane_segment_pre in unique_lane_segments:
                    idcs_pre = node_idcs[unique_lane_segments.index(lane_segment_pre)]
                    pre['u'].append(idcs[0])
                    pre['v'].append(idcs_pre[-1])

            # points to the successor
            suc['u'] += list(idcs[:-1])
            suc['v'] += list(idcs[1:])

            # Get lane successors
            lane_suc = graph.suc_pairs[graph.suc_pairs[:, 0] == lane_segment, 1]
            for lane_segment_suc in lane_suc:
                if lane_segment_suc in unique_lane_segments:
                    idcs_suc = node_idcs[unique_lane_segments.index(lane_segment_suc)]
                    suc['u'].append(idcs[-1])
                    suc['v'].append(idcs_suc[0])
        
        # we now compute lane-level features
        graph['pre'] = [pre]
        graph['suc'] = [suc]


        # longitudinal connections
        for key in ['pre', 'suc']:
            # Transform to numpy arrays
            for k2 in ['u', 'v']:
                graph[key][0][k2] = np.asarray(graph[key][0][k2], np.int64)
            
            assert len(graph[key]) == 1
            nbr = graph[key][0]

            # create a sparse matrix
            data = np.ones(len(nbr['u']), bool)
            csr = sp.sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(graph.num_nodes, graph.num_nodes))

            # prepare the output
            mat = csr.copy()

            current_scale = 1
            for i, scale in enumerate(scales):
                assert scale > current_scale, 'scales should be stricly monotonicly increasing.'
                continue_squaring = scale / current_scale >= 2
                while continue_squaring:
                    mat = mat * mat
                    current_scale *= 2
                    continue_squaring = scale / current_scale >= 2
                
                # multiple the original matrix to this 
                continue_multiplying = scale > current_scale
                if continue_multiplying:
                    mat = mat * csr
                    current_scale += 1
                    continue_multiplying = scale > current_scale

                # Save matrix
                nbr = dict()
                coo = mat.tocoo()
                nbr['u'] = coo.row.astype(np.int64)
                # print(len(coo.row))
                nbr['v'] = coo.col.astype(np.int64)
                graph[key].append(nbr)



        ##################################################################################
        # Add left and right node connections                                            #
        ##################################################################################
        # like pre and sec, but for left and right nodes
        left, right = dict(), dict()
        left['u'], left['v'] = [], []
        right['u'], right['v'] = [], []

        # indexing starts from 0, makes sense
        num_lanes = graph.lane_idcs.max() + 1

        ctrs  = torch.from_numpy(ctrs).to(device = device)
        feats = torch.from_numpy(feats).to(device = device)

        # get the needed bytes for each column
        # Get the number of boolean bytes
        memory_per_row = graph.num_nodes * 40 # This is a rough upper bound

        # Usable nodes_at once
        available_gpu_memory = torch.cuda.get_device_properties(device = device).total_memory - torch.cuda.memory_reserved(device = device)
        available_rows = available_gpu_memory // memory_per_row

        # Check the first multiple of power two
        available_rows = 2 ** np.floor(np.log2(available_rows)).astype(int)

        # get angle along lane
        t_nodes = torch.atan2(feats[:, 1], feats[:, 0])

        # Get lane indices
        lane_indices = torch.from_numpy(graph.lane_idcs).to(device = device)

        # Get the current matrices for pre and suc
        if len(graph['pre_pairs'].shape) == 2 and len(graph['suc_pairs'].shape) == 2:
            pre = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
            pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
            
            suc = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
            suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

        # get lane segments that are left
        mat_left = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
        mat_left[graph['left_pairs'][:, 0], graph['left_pairs'][:, 1]] = 1

        # get lane segments that are right
        mat_right = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
        mat_right[graph['right_pairs'][:, 0], graph['right_pairs'][:, 1]] = 1

        # Extend mat left and mat right to include the predecessors and successors of the left and right lanes
        mat_left = (torch.matmul(mat_left, pre) + torch.matmul(mat_left, suc) + mat_left) > 0.5
        mat_right = (torch.matmul(mat_right, pre) + torch.matmul(mat_right, suc) + mat_right) > 0.5

        # Prepare extraction to the available rows
        mat_left = mat_left[:, lane_indices]
        mat_right = mat_right[:, lane_indices]

        row_splits = np.arange(0, graph.num_nodes, available_rows)

        # Row: origin node (u), Columns: Candidate for left or right node (v)
        for i_start in row_splits:
            i_end = min(i_start + available_rows, graph.num_nodes)
            # allows us to index through all pairs of lane nodes
            row_idcs = torch.arange(i_start, i_end).to(device)

            # find possible left and right neighouring nodes
            if cross_angle is not None:
                # cross lane
                f2 = ctrs.unsqueeze(0) - ctrs[row_idcs].unsqueeze(1)

                # Get the angle between all node center connection
                f21 = f2[..., 1]
                f20 = f2[..., 0]
                del f2

                dt = torch.atan2(f21, f20) 

                del f21, f20

                # Get the difference in angle
                dt -= t_nodes[row_idcs].unsqueeze(1)

                
                # Roll around angles
                dt -= (dt > (2 * np.pi)).float() * (2 * torch.pi)
                dt += (dt < (-2 * np.pi)).float() * (2 * torch.pi)

                left_mask = torch.logical_and(dt > 0, dt < cross_angle)
                right_mask = torch.logical_and(dt < 0, dt > -cross_angle)

                del dt

            

            # distances between all node centres
            dist = ctrs[row_idcs].unsqueeze(1) - ctrs.unsqueeze(0)
            dist = dist ** 2
            dist = torch.sum(dist, dim=-1)
            dist = torch.sqrt(dist)

            # find left lane nodes
            # Get the nodes that do not belong to those lanes
            mask = mat_left[lane_indices[row_idcs]]
            if mask.any() and ((cross_angle is None) or left_mask.any()):

                # Ignore the nodes that are too far away or not in the correct angle
                left_dist = dist.clone()
                if cross_angle is not None:
                    mask &= left_mask
                    
                mask = mask.logical_not()
                left_dist[mask] = 1e6

                # Find the nodes whose nearest valid neighbor is close enough
                min_dist, min_idcs = left_dist.min(1)
                del left_dist
                mask = min_dist < cross_dist
                ui = row_idcs[mask]
                vi = min_idcs[mask]

                # Get the corresponding angles of the nodes
                t1 = t_nodes[ui]
                t2 = t_nodes[vi] 

                # Check if nodes are aligned enough
                dt = torch.abs(t1 - t2)
                m = dt > np.pi
                dt[m] = torch.abs(dt[m] - 2 * np.pi)
                m = dt < 0.25 * np.pi

                left['u'] += list(ui[m].cpu().numpy())
                left['v'] += list(vi[m].cpu().numpy())

            # find right lane nodes
            # Get the nodes that do not belong to those lanes
            mask = mat_right[lane_indices[row_idcs]]
            if mask.any() and ((cross_angle is None) or right_mask.any()):

                # Ignore the nodes that are too far away or not in the correct angle
                right_dist = dist.clone()
                if cross_angle is not None:
                    mask &= right_mask
                    
                mask = mask.logical_not()
                right_dist[mask] = 1e6

                # Find the nodes whose nearest valid neighbor is close enough
                min_dist, min_idcs = right_dist.min(1)
                del right_dist
                mask = min_dist < cross_dist
                ui = row_idcs[mask]
                vi = min_idcs[mask]

                # Get the corresponding angles of the nodes
                t1 = t_nodes[ui]
                t2 = t_nodes[vi]

                # Check if nodes are aligned enough
                dt = torch.abs(t1 - t2)
                m = dt > np.pi
                dt[m] = torch.abs(dt[m] - 2 * np.pi)
                m = dt < 0.25 * np.pi

                right['u'] += list(ui[m].cpu().numpy())
                right['v'] += list(vi[m].cpu().numpy())

        # Make to arrays
        left['u'] = np.array(left['u'], np.int64)
        left['v'] = np.array(left['v'], np.int64)
        right['u'] = np.array(right['u'], np.int64)
        right['v'] = np.array(right['v'], np.int64)

        graph['left'] = [left]
        graph['right'] = [right]

        return graph
    



    ######################################################################################################
    ######################################################################################################
    ###                                                                                                ###
    ###                                       Data Extraction                                          ###
    ###                                                                                                ###
    ######################################################################################################
    ######################################################################################################

    def classify_path(self, path, t, domain):
        r'''
        This function classifies a given set of trajectories.

        Parameters
        ----------
        path : pandas.Series
            A pandas series with :math:`(N_{agents})` entries, where each entry is itself a numpy array of 
            shape :math:`\{N_{preds} \times |t| \times N_{data} \}` or :math:`|t| \times N_{data}\}`. 
            Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*.
            The indices should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        t : numpy.ndarray
            A numpy array of lenght :math:`|t|`, recording the corresponding timesteps.

        Returns
        -------
        Class : integer
            Returns the class name for the current behavior

        T_Delta : pandas.Series
            This is a :math:`N_{classes}` dimensional Series.
            For each column, it returns an array of lenght :math:`|t|` with the predicted time 
            until the classification criteria will be met.

        T : pandas.Series
            This is a :math:`N_{classes}` dimensional Series.
            For each column, it returns the time at which the distance associated with each 
            behavior switches form positive to negative. If this does not happen in the possible
            time frame, the linear interpolation is used to predict this.
        '''

        if self.classification_useful:
            path_extra_dim = self.increase_path_dim(path)
            Dist = self.calculate_distance(path_extra_dim, t, domain)
            Dist = self.decrease_dist_dim(Dist)

            in_position = self.evaluate_scenario(path, Dist, domain)
            t_position = t[in_position]

            mean_dt = np.mean(t[1:] - t[:-1])
            n_dt = max(5, int(0.75 * self.dt / mean_dt))
            n_dt = min(n_dt, len(t)-1)

            T = np.zeros(len(self.Behaviors), float)
            T_D = np.empty(len(self.Behaviors), object)
            for i, beh in enumerate(self.Behaviors):
                Dist_dt = (Dist[beh][n_dt:] - Dist[beh][:-n_dt]) / (t[n_dt:] - t[:-n_dt])
                Dist_dt = np.concatenate((np.tile(Dist_dt[[0]], (n_dt)), Dist_dt), axis=0)

                T_D[i] = Dist[beh] / np.maximum(- Dist_dt, 1e-7)
                if not in_position.any():
                    T[i] = t[-1] + 1
                else:
                    Dt_in_pos = T_D[i][in_position]

                    time_change = np.where((Dt_in_pos[:-1] > 0) & (Dt_in_pos[1:] <= 0))[0]
                    if len(time_change) == 0:
                        if Dt_in_pos[0] <= 0:
                            T[i] = Dt_in_pos[0] + t_position[0]
                        else:
                            T[i] = Dt_in_pos[-1] + t_position[-1]
                    else:
                        ind = time_change[0]
                        T[i] = ((t_position[ind + 1] * Dt_in_pos[ind] - t_position[ind] * Dt_in_pos[ind + 1]) /
                                (Dt_in_pos[ind] - Dt_in_pos[ind + 1]))

            # Check if classification is possible
            if in_position.any() and T.min() > t_position.max():
                return [Dist, in_position,
                        self.behavior_default,
                        pd.Series(T_D, index=self.Behaviors),
                        pd.Series(T, index=self.Behaviors)]
            else:
                return [Dist, in_position,
                        self.Behaviors[T.argmin()],
                        pd.Series(T_D, index=self.Behaviors),
                        pd.Series(T, index=self.Behaviors)]
        else:
            in_position = self.evaluate_scenario(path, None, domain)
            if in_position is None:
                in_position = np.ones(len(t), bool)
            
            return None, in_position, self.behavior_default, None, None

    def increase_path_dim(self, path):
        path_out = path.copy(deep=True)
        for index in path_out.index:
            array = path_out[index]
            if not isinstance(array, float):
                if len(array.shape) == 1:
                    raise TypeError("A path has to have to dimesnions, for time and positions dimension")
                elif len(array.shape) == 2:
                    path_out[index] = array[np.newaxis, :]
                else:
                    path_out[index] = array
        return path_out

    def decrease_dist_dim(self, Dist):
        for index in Dist.index:
            if (len(Dist[index].shape) > 1) and (Dist[index].shape[0] == 1):
                Dist[index] = Dist[index][0]
        return Dist

    def projection_to_1D(self, path, t, domain):
        r'''
        This function gives a 1D projection of a certain scenario, which might represent a loss in information.

        Parameters
        ----------
        path : pandas.Series
            A pandas series with :math:`(N_{agents})` entries, where each entry is itself a numpy array of 
            shape :math:`\{N_{preds} \times |t| \times N_{data} \}` or :math:`|t| \times N_{data}\}`. 
            Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*.
            The indices should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        t : numpy.ndarray
            A numpy array of lenght :math:`|t|`, recording the corresponding timesteps.

        Returns
        -------
        Pred : pandas.Series
            This is a :math:`N_{classes} + N_{other dist}` dimensional Series.
            For each column, it returns an array of lenght :math:`|t|`.

        '''
        if self.general_input_available:
            path_extra_dim = self.increase_path_dim(path)

            Dist = self.calculate_distance(path_extra_dim, t, domain)
            Dist = self.decrease_dist_dim(Dist)
            
            if len(self.extra_input) > 0:
                Dist_oth = self.calculate_additional_distances(path, t, domain)
                for index in self.extra_input:
                    assert index in Dist_oth.index, "A required extracted input is missing."

                Pred = pd.concat([Dist, Dist_oth])
            else:
                Pred = Dist
            return Pred
        else:
            return None

    def extract_time_points(self, Path, Type, T, Domain_old, num_samples, path_file):
        # Replace in path file --all_orig_paths with --all_time_points
        time_file = path_file.replace('--all_orig_paths', '--all_time_points')

        if os.path.isfile(time_file):
            [
                local_id,
                local_t,
                local_D_class,
                local_behavior,
                local_T_D_class,
                local_T_class,
                local_t_start,
                local_t_decision,
                local_t_crit, _] = np.load(time_file, allow_pickle=True)

        else:
            local_id = []
            local_t = []
            local_D_class = []
            local_behavior = []
            local_T_D_class = []
            local_T_class = []
            local_t_start = []
            local_t_decision = []
            local_t_crit = []

            agent_name_array = np.array(Type.columns)
            Path_sparse = self.get_multiindex_path(Path)
            for i_sample in range(num_samples):
                if np.mod(i_sample, 100) == 0:
                    print('path ' + str(i_sample).rjust(len(str(num_samples))) + '/{} divided'.format(num_samples))

                domain = Domain_old.iloc[i_sample]
                t = np.array(T[i_sample])
                path = self.get_dense_path_sample(Path_sparse, i_sample, agent_name_array, len(t))

                # Get the corresponding class
                d_class, in_position, behavior, t_D_class, t_class = self.classify_path(path, t, domain)

                # Check if scenario is fulfilled
                if not in_position.any():
                    continue
                
                # Get decision time
                if self.classification_useful:
                    t_decision = t_class.to_numpy().min()
                else:
                    t_decision = t[in_position][-1]

                # check if decision can be observed
                if t_decision > t[in_position].max():
                    continue

                ts_allowed = in_position & (t <= t_decision)
                try:
                    ind_possible = np.where(ts_allowed)[0]
                    try:
                        ind_start = np.where(np.invert(ts_allowed[:ind_possible[-1]]))[0][-1] + 1
                    except:
                        ind_start = ind_possible[0]

                    t_start = t[ind_start]
                except:
                    # there never was as starting point here in the first place
                    continue

                # Determine tcrit
                if self.classification_useful and (self.pov_agent is not None):
                    t_D_default = np.minimum(1000, t_D_class[self.behavior_default])
                    
                    t_D_useful = self.scenario.calculate_safe_action(d_class, t_D_class, self, path, t, domain)
                    try:
                        Delta_tD = t_D_default - t_D_useful
                        critical = (t > t_start) & (t < t_decision) & (Delta_tD < 0)
                        ind_crit = np.where(critical)[0][0]
                        if Delta_tD[ind_crit - 1] < 0:
                            t_crit = t_start
                        else:
                            # Gap starts uncritical
                            fac_crit = Delta_tD[ind_crit] / (Delta_tD[ind_crit] - Delta_tD[ind_crit - 1])
                            t_crit = t[ind_crit - 1] * fac_crit + t[ind_crit] * (1 - fac_crit)
                    except:
                        if t_D_default[ind_start + 1] < 0:
                            t_crit = t_start
                        else:
                            t_crit = t_decision + 0.01
                else:
                    t_crit = t_decision + 0.01
                
                local_id.append(i_sample)
                local_t.append(t.astype(float))
                local_D_class.append(d_class)
                local_behavior.append(behavior)
                local_T_D_class.append(t_D_class)
                local_T_class.append(t_class)
                local_t_start.append(t_start)
                local_t_decision.append(t_decision)
                local_t_crit.append(t_crit)

            # Prevent problem if all t should have same length
            local_t.append([])
            
            local_id         = np.array(local_id)
            local_t          = np.array(local_t, object)[:-1]
            local_D_class    = pd.DataFrame(local_D_class)
            local_behavior   = np.array(local_behavior)
            local_T_D_class  = pd.DataFrame(local_T_D_class)
            local_T_class    = pd.DataFrame(local_T_class)
            local_t_start    = np.array(local_t_start)
            local_t_decision = np.array(local_t_decision)
            local_t_crit     = np.array(local_t_crit)

            save_data_time = np.array([local_id,
                                        local_t,
                                        local_D_class,
                                        local_behavior,
                                        local_T_D_class,
                                        local_T_class,
                                        local_t_start,
                                        local_t_decision,
                                        local_t_crit, 0], object)  # 0 is there to avoid some numpy load and save errros

            os.makedirs(os.path.dirname(time_file), exist_ok=True)
            np.save(time_file, save_data_time)
        
        return [local_id,
                local_t,
                local_D_class,
                local_behavior,
                local_T_D_class,
                local_T_class,
                local_t_start,
                local_t_decision,
                local_t_crit]

    def determine_dtc_boundary(self):
        # Get name of save file
        self.data_dtc_bound_file = self.file_path + '--all_fixed_size.npy'

        if os.path.isfile(self.data_dtc_bound_file):
            self.dtc_boundary = np.load(self.data_dtc_bound_file, allow_pickle=True)
        else:
            if self.classification_useful:
                # Get test boundaries
                num_bounderies = 20001
                dtc_boundaries = np.linspace(0, 20, num_bounderies)[:, np.newaxis]
                
                # initialize number of behaviors
                num_beh = np.zeros((num_bounderies, len(self.Behaviors)), int)
                
                str_helper = np.array(['ZZZ_', ''])
                
                # Cycle through existing paths
                for i_orig_path in range(self.number_original_path_files):
                
                    if self.number_original_path_files == 1:
                        path_file = self.file_path + '--all_orig_paths_LLL.npy'
                        
                        # Get the allready loaded data
                        Path_loaded = self.Path
                        T_loaded = self.T
                        Type_loaded = self.Type
                        Domain_old_loaded = self.Domain_old
                        num_samples_loaded = self.num_samples
                    else:
                        # Get data file
                        path_file = self.file_path + '--all_orig_paths'
                        
                        # Get path name adjustment
                        if i_orig_path < self.number_original_path_files - 1:
                            path_file_adjust = '_' + str(i_orig_path).zfill(3)
                        else:
                            path_file_adjust = '_LLL'
                        
                        path_file += path_file_adjust + '.npy'
                        
                        # Load the data
                        Loaded_data = np.load(path_file, allow_pickle=True)
                        Path_loaded, Type_loaded, _, T_loaded, Domain_old_loaded, num_samples_loaded = self.extract_loaded_data(Loaded_data)
                
                    # Load extracted time points
                    [
                        local_id,
                        local_t,
                        local_D_class,
                        local_behavior,
                        local_T_D_class,
                        local_T_class,
                        local_t_start,
                        local_t_decision,
                        local_t_crit] = self.extract_time_points(Path_loaded, Type_loaded, T_loaded, Domain_old_loaded, num_samples_loaded, path_file)
                

                
                
                    initial_size = np.zeros((1, len(local_id)), float)
                    final_size = np.zeros((1, len(local_id)), float)

                    for i, t_start in enumerate(local_t_start):
                        t_D_default = local_T_D_class.iloc[i][self.behavior_default]
                        [initial_size[0, i], final_size[0, i]] = np.interp([t_start, local_t_decision[i]], local_t[i], t_D_default)
                    
                    # Determine if a sample is included
                    included = (dtc_boundaries <= initial_size) & (dtc_boundaries > final_size)

                    included_behavior = np.core.defchararray.add(str_helper[included.astype(int)], local_behavior[np.newaxis, :])
                    for i, beh in enumerate(self.Behaviors):
                        num_beh[:, i] += np.sum(included_behavior == beh, axis=1)
                        
                # remove columns that are always zero from num_beh
                num_beh = num_beh[:, num_beh.sum(axis=0) > 0]
                
                self.dtc_boundary = dtc_boundaries[np.argmax(num_beh.min(axis=1))]
            else:
                self.dtc_boundary = np.array([0.0])

            os.makedirs(os.path.dirname(self.data_dtc_bound_file), exist_ok=True)
            np.save(self.data_dtc_bound_file, self.dtc_boundary)

        self.dtc_boundary = self.dtc_boundary[0]
        print('For predictions on dataset ' + self.get_name()['print'] + 
              ' at gaps with fixed sizes, a size of {:0.3f} s was chosen'.format(self.dtc_boundary))


    def extract_t0(self, t0_type, t, t_start, t_decision, t_crit, local_T_D_class, i_sample, behavior):
        if t0_type[:5] == 'start':
            T0 = [t_start]
        
        
        elif t0_type[:3] == 'all':
            # Get n
            if t0_type == 'all':
                n = 1
            else:
                assert t0_type[:4] == 'all_', 'Type is not admissable'
                n = max(1, min(99, int(t0_type[4:])))


            min_t0 = max(t_start, t.min() + self.dt * (self.num_timesteps_in_need - 1))
            if self.enforce_prediction_time or not self.classification_useful:
                max_t0 = min(t_decision, t.max() - self.dt * self.num_timesteps_out_need) + 1e-6
            else:
                max_t0 = t_decision + 1e-6
            T0 = np.arange(min_t0, max_t0, n * self.dt)

        elif t0_type[:3] == 'col':
            if self.classification_useful:
                t_D_default = local_T_D_class.iloc[i_sample][self.behavior_default]
                if t0_type[:9] == 'col_equal':
                    t_D_value = self.dtc_boundary
                elif t0_type[:7] == 'col_set':
                    t_D_value = self.dt * self.num_timesteps_out_real
                else:
                    raise TypeError("This type of starting point is not defined")

                try:
                    ind_0 = np.where((t_D_default <= t_D_value) & (t >= t_start))[0][0]
                    T0 = [t[ind_0] - (t_D_value - t_D_default[ind_0])]
                except:
                    assert behavior != self.behavior_default
                    T0 = [t_decision + self.dt]
            else:
                return ("the scenario " + self.scenario.get_name() + "cannot process the required method" +
                        " of extracting prediction times.")
        
        elif t0_type[:4] == 'crit':
            if self.classification_useful and (self.pov_agent is not None):
                T0 = [t_crit - 0.001]
            else:
                return [("the scenario " + self.scenario.get_name() + "cannot process the required method" +
                         " of extracting prediction times.")]
        else:
            raise TypeError("This type of starting point is not defined")
        return T0
    
    
    def check_t0_constraint(self, t0, t, t0_type, t_start, t_crit, t_decision):
        # Get the ealiest time where a prediction could be made 
        t0_min = max(t_start, np.min(t) + (self.num_timesteps_in_need - 1) * self.dt)
        
        # Get the time at which a prediction is no longer necessary/useful
        t0_max = t_decision - 1e-6
        
        # Predictions must be possible and useful for further path planning
        if self.exclude_post_crit:
            t0_max = min(t0_max, t_crit - 1e-6)
            
        # All path predictions must be comparable to true trajectories
        if self.enforce_num_timesteps_out:
            t0_max = min(t0_max, t.max() - self.num_timesteps_out_need * self.dt)
        else:
            t0_max = min(t0_max, t.max() - 1 * self.dt)
            
        
        # Update sample if necessary and permittable
        if (t0 >= t_start and 
            (((not self.classification_useful) and (t0 == t[0])) or
             (not self.enforce_prediction_time))):
            t0 = max(t0, t0_min)
        
        # exclude samples where t0 is not admissible during open gap
        if not (t0_min <= t0 and t0 <= t0_max):
            return None
        else:
            return t0
    
    def determine_required_timesteps(self, num_timesteps):
        # Determine the number of input timesteps (used and max required)
        if isinstance(num_timesteps, tuple):
            # Refers to sctual input data
            num_timesteps_real = int(min(99, num_timesteps[0]))
            num_timesteps_need = min(99, max(num_timesteps_real, num_timesteps[1]))

        # If only one value is given, assume that the required number of steps is identical
        elif isinstance(num_timesteps, int):
            num_timesteps_real = int(min(99, num_timesteps))  # Refers to sctual input data
            num_timesteps_need = min(99, num_timesteps)  # Restrictions on t0
            
        return num_timesteps_real, num_timesteps_need
    
    def set_extraction_parameters(self, t0_type, T0_type_compare, max_num_agents):
        assert isinstance(t0_type, str), "Prediction time method has to be a string."
        assert isinstance(T0_type_compare, list), "Prediction time constraints have to be in a list."
        for t in T0_type_compare:
            assert isinstance(t, str), "Prediction time constraints must come in the form of strings."
        
        self.t0_type = t0_type
        self.T0_type_compare = T0_type_compare
        if max_num_agents is not None:
            self.max_num_agents = max(1, max_num_agents)
        else:
            self.max_num_agents = None
        self.prediction_time_set = True
    

    def data_params_to_string(self, dt, num_timesteps_in, num_timesteps_out):
        self.dt = dt
        (self.num_timesteps_in_real, 
         self.num_timesteps_in_need)  = self.determine_required_timesteps(num_timesteps_in)
        (self.num_timesteps_out_real, 
         self.num_timesteps_out_need) = self.determine_required_timesteps(num_timesteps_out)
        
        # create possible file name
        t0_type_file_name = {'start':     'start',
                            'col_equal': 'fix_e',
                            'col_set':   'fix_s',
                            'crit':      'crit_'}
        
        if self.t0_type[:3] != 'all':
            t0_type_name = t0_type_file_name[self.t0_type]
        else:
            if self.t0_type == 'all':
                n = 1
            else:
                assert self.t0_type[:4] == 'all_', 'Type is not admissable'
                n = max(1, min(99, int(self.t0_type[4:])))
            t0_type_name = 'all' + str(n).zfill(2)
        t0_type_name += '_'
        
        if self.enforce_prediction_time:
            t0_type_name += 's' # s for severe
        else:
            t0_type_name += 'l' # l for lax
            
        if self.enforce_num_timesteps_out:
            t0_type_name += 's'
        else:
            t0_type_name += 'l'
            
        if self.max_num_agents is None:
            num = 0 
        else:
            num = self.max_num_agents


        # Check if extrapolation is not allowed
        if not self.allow_extrapolation:
            extra_string = '--No_Extrap'
        else:
            extra_string = ''


        folder = self.path + os.sep + 'Results' + os.sep + self.get_name()['print'] + os.sep + 'Data' + os.sep

        if self.is_perturbed:
            # Check if there is a .xlsx document for perturbations
            Pert_save_doc = (folder +
                          self.get_name()['file'] +
                          '--t0=' + t0_type_name +
                          '--dt=' + '{:0.2f}'.format(max(0, min(9.99, self.dt))).zfill(4) +
                          '_nI=' + str(self.num_timesteps_in_real).zfill(2) + 
                          'm' + str(self.num_timesteps_in_need).zfill(2) +
                          '_nO=' + str(self.num_timesteps_out_real).zfill(2) + 
                          'm' + str(self.num_timesteps_out_need).zfill(2) +
                          '_EC' * self.exclude_post_crit + '_IC' * (1 - self.exclude_post_crit) +
                          '--max_' + str(num).zfill(3) + '--Perturbations.xlsx')
            
            if os.path.isfile(Pert_save_doc):
                Pert_df = pd.read_excel(Pert_save_doc, index_col=0)
                # Check if a perturbation if the same attack and name allready exists
                previous_version = (Pert_df['attack'] == self.Perturbation.attack) & (Pert_df['name'] == self.Perturbation.name)

                if previous_version.any() > 0:
                    pert_index = np.min(Pert_df.index[previous_version])
                else:
                    pert_index = np.max(Pert_df.index) + 1
            else:
                # Create the dataframe
                Pert_df = pd.DataFrame(np.empty((0,2), str), columns=['attack', 'name'])
                pert_index = 0
            
            pert = pd.Series([self.Perturbation.attack, self.Perturbation.name], index=Pert_df.columns, name=pert_index)
            Pert_df.loc[pert_index] = pert

            os.makedirs(os.path.dirname(Pert_save_doc), exist_ok=True)
            Pert_df.to_excel(Pert_save_doc)

            pert_string = '--Pertubation_' + str(int(pert_index)).zfill(3)
        else:
            pert_string = ''

        # Assemble full data_file name    
        data_file = (folder + self.get_name()['file'] +
                     '--t0=' + t0_type_name +
                     '--dt=' + '{:0.2f}'.format(max(0, min(9.99, self.dt))).zfill(4) +
                     '_nI=' + str(self.num_timesteps_in_real).zfill(2) + 
                     'm' + str(self.num_timesteps_in_need).zfill(2) +
                     '_nO=' + str(self.num_timesteps_out_real).zfill(2) + 
                     'm' + str(self.num_timesteps_out_need).zfill(2) +
                     '_EC' * self.exclude_post_crit + '_IC' * (1 - self.exclude_post_crit) +
                     '--max_' + str(num).zfill(3) + extra_string + pert_string + '.npy')
        
        return data_file

    def get_number_of_data_files(self):
        # Check if file exists
        data_file_final = self.data_file[:-4] + '_LLL_LLL_data.npy'
        if not os.path.isfile(data_file_final):
            raise AttributeError("The data has not been completely compiled yet.")
        
        # Get the corresponding directory
        test_file_directory = os.path.dirname(self.data_file)
        
        # Find files in same directory that start with file_path_test
        domain_files = [test_file_directory + os.sep + f for f in os.listdir(test_file_directory) 
                        if (f.startswith(os.path.basename(self.data_file[:-4])) and f.endswith('_domain.npy'))]
        domain_files = [f for f in domain_files if self.data_file[:-4] == f[:-19]]
        
        # sort domain_files to ensure consistency
        domain_files = np.sort(domain_files).tolist()
        
        num_files = len(domain_files)
        return domain_files, num_files
    
    
    def get_data_from_orig_path(self, Path, Type_old, Size_old, T, Domain_old, num_samples, path_file, path_file_adjust):
        # Extract time points from raw data
        [
            local_id,
            local_t,
            local_D_class,
            local_behavior,
            local_T_D_class,
            local_T_class,
            local_t_start,
            local_t_decision,
            local_t_crit] = self.extract_time_points(Path, Type_old, T, Domain_old, num_samples, path_file)
        
        # Check if size is used
        size_given = Size_old is not None

        # Check if this is consistent over dataset
        if not hasattr(self, 'size_given'):
            self.size_given = size_given
        
        assert self.size_given == size_given, "The decision to use size has to be consisten over the dataset"

        # Get number of possible accepted/rejected samples in the whole dataset
        self.num_behaviors_local = np.zeros(len(self.Behaviors), int)
        
        # Delete all files that allready exist in this directory to avoid duplication
        data_file = self.data_file[:-4] + path_file_adjust + '_'
        data_file_directory = os.path.dirname(data_file)
        data_file_name = os.path.basename(data_file)
        
        # Find files in same directory that start with file_path_test
        delete_files = [data_file_directory + os.sep + f for f in os.listdir(data_file_directory)
                        if f.startswith(data_file_name)]
        # Delete all files that allready exist in this directory to avoid duplication
        for f in delete_files:
            os.remove(f) 
        
        # set number of maximum agents
        if self.max_num_agents is not None:
            min_num_agents = len(Type_old.columns)
            self.max_num_addable_agents = max(0, self.max_num_agents - min_num_agents)
            max_num_agent_local = self.max_num_addable_agents + min_num_agents
        else:
            self.max_num_addable_agents = None
        
        # prepare empty information
        # Input
        self.Input_prediction_local = []
        self.Input_path_local       = []
        self.Input_T_local          = []

        # Output
        self.Output_path_local   = []
        self.Output_T_local      = []
        self.Output_T_pred_local = []
        self.Output_A_local      = []
        self.Output_T_E_local    = []

        # Domain
        self.Type_local     = []
        self.Recorded_local = []
        self.Domain_local   = []

        if size_given:
            self.Size_local = []

        # Go through samples
        local_num_samples = len(local_id)

        predicted_saving_length = 0

        agent_name_array = np.array(Type_old.columns)

        Path_sparse = self.get_multiindex_path(Path)
        for i in range(local_num_samples):
            # print progress
            if np.mod(i, 1) == 0:
                print('path ' + str(i + 1).rjust(len(str(local_num_samples))) +
                    '/{}: divide'.format(local_num_samples))

            # load extracted data
            i_path = local_id[i]
            t = local_t[i]
            path = self.get_dense_path_sample(Path_sparse, i_path, agent_name_array, len(t))

            behavior = local_behavior[i]
            t_start = local_t_start[i]
            t_decision = local_t_decision[i]
            t_crit = local_t_crit[i]
            
            # Update self.num_behaviors_local
            self.num_behaviors_local[self.Behaviors == behavior] += 1

            # Get the time of prediction
            T0 = self.extract_t0(self.t0_type, t, t_start, t_decision, t_crit, local_T_D_class, i, behavior)
            
            # Extract comparable T0 types
            T0_compare = []
            if self.t0_type[:3] != 'all':
                for extra_t0_type in self.T0_type_compare:
                    if extra_t0_type[:3] == 'all':
                        raise TypeError("Comparing against the all method is not possible")
                    else:
                        T0_compare.append(self.extract_t0(extra_t0_type, t, t_start, t_decision, t_crit, local_T_D_class, i, behavior)) 
            
            for ind_t0, t0 in enumerate(T0):
                if len(T0) > 50:
                    if np.mod(ind_t0, 10) == 0:
                        print('path ' + str(i + 1).rjust(len(str(local_num_samples))) +
                            '/{} - prediction time {}/{}: divide'.format(local_num_samples, ind_t0 + 1, len(T0)))

                if isinstance(t0, str):
                    return t0
                # Prepare domain
                # load original path data
                domain = Domain_old.iloc[i_path].copy()

                # Needed for later recovery of path data
                domain['Path_ID'] = i_path
                if self.is_perturbed:
                    # Get perturbation index from file name
                    assert 'Pertubation_' in self.data_file, "Pertubation index is missing in file name."
                    pert_index = self.data_file.split('Pertubation_')[1][:3]
                    domain['Scenario'] = self.get_name()['print'] + ' (Pertubation_' + pert_index + ')'
                else:
                    domain['Scenario'] = self.get_name()['print']
                domain['Scenario_type'] = self.scenario.get_name()
                domain['t_0'] = t0
                
                agent_types = Type_old.iloc[i_path].copy()
                if size_given:
                    size = Size_old.iloc[i_path].copy()
                
                # Check if this t0 is applicable
                t0 = self.check_t0_constraint(t0, t, self.t0_type, t_start, t_crit, t_decision)
                if t0 == None:
                    continue
                
                # Ensure that the comparable t0 are applicable
                if self.t0_type[:3] != 'all':
                    T0_compare_ind = np.array([self.check_t0_constraint(t0_extra[ind_t0], t, self.t0_type, 
                                                                        t_start, t_crit, t_decision)
                                            for t0_extra in T0_compare])
                    
                    if (T0_compare_ind == None).any():
                        continue
                
                

                # calculate number of output time steps
                num_timesteps_out_pred = self.num_timesteps_out_real
                if self.classification_useful and self.allow_longer_predictions:
                    t_default = local_T_class.iloc[i][self.behavior_default]
                    
                    # set prediction horizon considered for classification
                    Pred_horizon_max = 2 * self.num_timesteps_in_real * self.dt
                    Pred_horizon = min(Pred_horizon_max, t_default - t0)

                    num_timesteps_out_pred = max(num_timesteps_out_pred, int(np.ceil(Pred_horizon / self.dt)) + 5)
                

                num_timesteps_out_data = min(num_timesteps_out_pred, int(np.floor((t.max() - t0) / self.dt)))

                # build new path data
                # create time points
                input_T       = t0 + np.arange(1 - self.num_timesteps_in_real, 1) * self.dt
                output_T      = t0 + np.arange(1, num_timesteps_out_data + 1) * self.dt
                output_T_pred = t0 + np.arange(1, num_timesteps_out_pred + 1) * self.dt

                # prepare empty pandas series for general values
                if self.general_input_available:
                    input_pred_old = self.projection_to_1D(path, t, domain)
                    input_pred_index = input_pred_old.index
                    input_prediction = pd.Series(np.empty(len(input_pred_index), np.ndarray), index=input_pred_index)
                    for index in input_pred_index:
                        input_prediction[index] = np.interp(input_T + 1e-5, t, input_pred_old[index].astype(np.float32),  # + 1e-5 is necessary to avoid nan for min(input_T) == min(t)
                                                            left=np.nan, right=np.nan)
                else:
                    input_prediction = pd.Series(np.nan * np.ones(1), index=['empty'])
                # prepare empty pandas series for path
                helper_path   = pd.Series(np.empty(len(path.index), np.ndarray), index=path.index)
                helper_T_appr = np.concatenate((input_T, output_T))
                helper_T_appr[0] += 1e-5
                helper_T_appr[-1] -= 1e-5
                
                assert isinstance(t, np.ndarray), "Time has to be a numpy array"
                
                correct_path = True
                for agent in path.index:
                    if not isinstance(path[agent], float):
                        # interpolate each dimension of the path along time axis
                        path_agent_new = []
                        for i_dim in range(path[agent].shape[-1]):
                            path_agent_new.append(np.interp(helper_T_appr, t, path[agent][:, i_dim], left=np.nan, right=np.nan))
                        helper_path[agent] = np.stack(path_agent_new, axis=-1).astype(np.float32)
                        
                        # Check if positional data is avialable after interpolatiion at at least two time steps
                        available_pos = np.isfinite(helper_path[agent][:self.num_timesteps_in_real, :2]).all(-1)
                        if np.sum(available_pos) <= 1:
                            helper_path[agent] = np.nan
                            agent_types[agent] = float('nan')
                            if size_given:
                                size[agent] = np.nan
                            
                    else:
                        helper_path[agent] = np.nan
                        agent_types[agent] = float('nan')
                        if size_given:
                            size[agent] = np.nan
                        
                    # check if needed agents have reuqired input and output
                    if agent in self.needed_agents:
                        if isinstance(helper_path[agent], float) or np.isnan(helper_path[agent][:,:2]).any():
                            correct_path = False
                            
                if not correct_path:
                    continue
                
                t_end = t0 + self.dt * num_timesteps_out_pred
                if t_end >= t_decision:
                    output_A = pd.Series(self.Behaviors == behavior, index=self.Behaviors)
                    output_T_E = t_decision
                else:
                    output_A = pd.Series(self.Behaviors == self.behavior_default, index=self.Behaviors)
                    output_T_E = t_end
                
                
                # reset time origin
                input_T = input_T - t0
                output_T = output_T - t0
                output_T_pred = output_T_pred - t0
                output_T_E = output_T_E - t0
                
                # Save information abut missing positions to overwrite them back later if needed
                recorded_positions = pd.Series(np.empty(len(helper_path.index), object), index=helper_path.index)
                
                
                for agent in helper_path.index:
                    if not isinstance(helper_path[agent], float):
                        available_pos = np.isfinite(helper_path[agent][:,:2]).all(-1)
                        assert available_pos.sum() > 1 
                        
                        # If an agent does not move at all, then mark the first timestep as unrecorded
                        if available_pos.all():
                            distances = np.linalg.norm(helper_path[agent][1:self.num_timesteps_in_real,:2] - helper_path[agent][:self.num_timesteps_in_real-1,:2], axis=-1)
                            if np.all(distances < 1e-2):
                                available_pos[self.num_timesteps_in_real - 1] = False
                            
                        recorded_positions[agent] = available_pos
                    else:
                        recorded_positions[agent] = np.nan
                
                # Combine input and output data
                helper_T = np.concatenate([input_T, output_T])
                
                # complete partially available paths
                fill_sig = signature(self.fill_empty_path)
                num_inputs = len(fill_sig.parameters) # self does not count
                if num_inputs == 4:
                    filled_data = self.fill_empty_path(helper_path, helper_T, domain, agent_types)
                elif num_inputs == 5:
                    if size_given:
                        filled_data = self.fill_empty_path(helper_path, helper_T, domain, agent_types, size)
                    else:
                        filled_data = self.fill_empty_path(helper_path, helper_T, domain, agent_types, None)
                else:
                    raise ValueError("The self.fill_empty_path() function does not have the correct number of inputs")

                if len(filled_data) == 2:
                    helper_path, agent_types = filled_data
                elif len(filled_data) == 3:
                    helper_path, agent_types, size = filled_data
                else:
                    raise ValueError("The filled data does not have the correct number of outputs (either 2 or 3)")

                # Update size with default values if new agents were added
                if size_given:
                    for agent in agent_types.index:
                        if (str(agent_types[agent]) != 'nan') and np.isnan(size[agent]).any():
                            if agent_types[agent] == 'V':
                                size[agent] + np.array([5.0, 2.0])
                            elif agent_types[agent] == 'M':
                                size[agent] + np.array([2.0, 0.5])
                            elif agent_types[agent] == 'B':
                                size[agent] + np.array([2.0, 0.5])
                            elif agent_types[agent] == 'P':
                                size[agent] + np.array([0.5, 0.5])
                            else:
                                raise KeyError("The agent type is not known")
                
                if self.max_num_agents is not None:
                    helper_path = helper_path.iloc[:max_num_agent_local]
                    agent_types = agent_types.iloc[:max_num_agent_local]
                    if size_given:
                        size = size.iloc[:max_num_agent_local]
                    
                
                # Split completed paths back into input and output
                input_path  = pd.Series(np.empty(len(helper_path.index), object), index=helper_path.index)
                output_path = pd.Series(np.empty(len(helper_path.index), object), index=helper_path.index)
                for agent in helper_path.index:
                    if not isinstance(helper_path[agent], float):
                        # Reset extrapolated data if necessary
                        if agent not in recorded_positions.index:
                            # This is the case if the agent was added in self.fill_empty_path
                            recorded_positions[agent] = np.zeros(len(helper_path[agent]), dtype = bool)
                            ind_start = 0
                            ind_last = len(helper_T)
                        else:
                            if not self.allow_extrapolation:
                                available_pos = recorded_positions[agent].copy()

                                # Do not delete interpolated data
                                ind_start = np.where(available_pos)[0][0]
                                ind_last = np.where(available_pos)[0][-1] + 1
                                
                                available_pos[ind_start:ind_last] = True

                                helper_path[agent][~available_pos] = np.nan
                            else:
                                ind_start = 0
                                ind_last = len(helper_T)
                        
                        # Check if the dimension (i.e., self.path_data_info() has correct length)
                        assert len(self.path_data_info()) == helper_path[agent].shape[-1], "The path data info does not match the path data"
                        
                        # Split by input and output, however, only include positions in the output
                        input_path[agent]  = helper_path[agent][:self.num_timesteps_in_real].astype(np.float32)
                        output_path[agent] = helper_path[agent][self.num_timesteps_in_real:].astype(np.float32)

                        # Guarantee that the input path does contain only nan value
                        if not (ind_start < self.num_timesteps_in_real - 1 and self.num_timesteps_in_real <= ind_last):
                            input_path[agent]         = np.nan
                            output_path[agent]        = np.nan
                            recorded_positions[agent] = np.nan
                            agent_types[agent] = float('nan')
                            if size_given:
                                size[agent] = np.nan
                        
                    else:
                        input_path[agent]         = np.nan
                        output_path[agent]        = np.nan
                        recorded_positions[agent] = np.nan
                        agent_types[agent]        = float('nan')
                        if size_given:
                            size[agent] = np.nan
                        
                # save results
                self.Input_prediction_local.append(input_prediction)
                self.Input_path_local.append(input_path)
                self.Input_T_local.append(input_T)

                self.Output_path_local.append(output_path)
                self.Output_T_local.append(output_T)
                self.Output_T_pred_local.append(output_T_pred)
                self.Output_A_local.append(output_A)
                self.Output_T_E_local.append(output_T_E)

                self.Type_local.append(agent_types)
                self.Recorded_local.append(recorded_positions)
                self.Domain_local.append(domain)

                if size_given:
                    self.Size_local.append(size)

                
                current_length = len(self.Input_T_local)
                if (current_length > predicted_saving_length) or (np.mod(current_length - 1, 5000) == 0):
                    memory_perc = self.check_extracted_data_for_saving(path_file_adjust)
                    predicted_saving_length = int(1 + current_length / memory_perc)
                    print(' ')
                    print('Current samples / Max num samples: ' + str(current_length) + ' / ' + str(predicted_saving_length))
                    print(' ')

        if (len(self.Input_path_local) > 0): 
            # last = true
            self.check_extracted_data_for_saving(path_file_adjust, True)   
        else:
            # check npy files in data directory
            directory = os.path.dirname(self.data_file)
            files = os.listdir(directory)

            # Filter files for this specific dataset
            data_file = self.data_file[:-4] + path_file_adjust

            files = [f for f in files if (f.startswith(data_file) and f.endswith('_data.npy'))]

            # Check if there are no other files available
            if len(files) == 0:
                raise RuntimeError("No samples have been extracted at all, as no other .npy files were found.")
            
            numbers = [int(f[-12:-9]) for f in files]

            highest_number = str(max(numbers)).zfill(3)

            overwrite_old = data_file + '_' + highest_number + '_'
            overwrite_new = data_file + '_LLL_'
            
            # Rename one selection (data, AM, Domain) to _LLL_LLL
            for ending in ['data.npy', '_AM.npy', '_Domain.npy']:
                overwrite_old_ending = overwrite_old + ending
                overwrite_new_ending = overwrite_new + ending
                os.rename(overwrite_old_ending, overwrite_new_ending)
        


    # get extrapolation functions
    def extrapolate_path(self, path, t, mode = 'pos'):
        r'''
        This function inter- and extrapolates the path data to the desired time points, 
        with the desired mode. The mode can be either 'pos' for assuming constant positions,
        or 'vel' for assuming constant velocities, or 'vel_turn' for assuming constant 
        longitudinal velocities and constant turning rates.

        Parameters
        ----------
        path : np.ndarray
            The path data to be inter- and extrapolated as an array with shape 
            :math:`\{\vert T_i \vert{\times} N_{data}\}`. Some time points may be missing.
        t : np.ndarray
            The time points of the path data with length :math:`\vert T_i \vert`.
        mode : str, optional
            The mode of the extrapolation, i.e. 'pos', 'vel', or 'vel_turn'. The default is 'pos'.
            Depending on available information, the model might overwrite this.

        Returns
        -------
        path_new : np.ndarray
            The inter- and extrapolated path data with shape :math:`\{\vert T \vert{\times} N_{data}\}`.
            It should no longer contain any missing values.
        '''
        
        # Initialize new path
        path_new = path.copy()

        # Find missing values
        missing_values = np.isnan(path).all(axis = -1)
        if not missing_values.any():
            return path
        
        # assert that either all variables are nan or none
        assert (missing_values == np.isnan(path).any(-1)).all(), "There are missing values in the path data"

        # Find existing values indices
        useful_index = np.where(~missing_values)[0]

        # Check if there is something to extrapolate from
        if len(useful_index) == 0:
            raise ValueError("There are no useful time points in the path data")
        elif len(useful_index) == 1:
            # If we only have one datastep, some modes required specific information to work

            # Check if we can extract velocities from one data point, if not, 
            # assume zero velocity and switch mode accordingly
            if mode in ['vel', 'vel_turn']:
                if not (np.in1d(['v', 'theta'], self.path_data_info()).all() or
                        np.in1d(['v_x', 'v_y'], self.path_data_info()).all()):
                    mode = 'pos'

            # Check if we can extract a turning rate from one data point, if not,
            # assume zero turning rate and switch mode accordingly
            if mode == 'vel_turn':
                if not 'd_theta' in self.path_data_info():
                    mode = 'vel'

        # Find useful values
        path_useful = path[~missing_values]
        t_useful    = t[~missing_values]

        # Prepare angle values, which are somewhat sketchy, by mapping onto 0 to 2 * pi
        if 'theta' in self.path_data_info():
            i_theta = self.path_data_info().index('theta')
            path_useful[:, i_theta] = np.mod(path_useful[:, i_theta], 2 * np.pi)

            # Check if d_theta is available
            if 'd_theta' in self.path_data_info():
                i_d_theta = self.path_data_info().index('d_theta')
            else:
                i_d_theta = None

            # Go through each gap, and if the gap is larger than np.pi, subtract 2 * np.pi from consecutive values
            # If the gap is smaller than - np.pi, add 2 * np.pi to consecutive values
            for i in range(1, len(useful_index) - 1):
                # find gapse
                if useful_index[i+1] - useful_index[i] > 1:
                    theta_gap = path_useful[i+1, i_theta] - path_useful[i, i_theta]
                    # check gap size
                    if i_d_theta is not None:
                        d_theta_mean = np.mean(path_useful[i:i+2, i_d_theta])
                        # if d_theta_mean > 0 and gap < -np.pi/4, add 2 * np.pi to consecutive values
                        # if d_theta_mean < 0 and gap > np.pi/4, subtract 2 * np.pi from consecutive values
                        if (d_theta_mean > 0) and (theta_gap < -np.pi/4):
                            path_useful[i+1:, i_theta] += 2 * np.pi
                        elif (d_theta_mean < 0) and (theta_gap > np.pi/4):
                            path_useful[i+1:, i_theta] -= 2 * np.pi
                    else:
                        # if the gap is larger than np.pi, subtract 2 * np.pi from consecutive values
                        # If the gap is smaller than - np.pi, add 2 * np.pi to consecutive values
                        if theta_gap > np.pi:
                            path_useful[i+1:, i_theta] -= 2 * np.pi
                        elif theta_gap < -np.pi:
                            path_useful[i+1:, i_theta] += 2 * np.pi
        
        if mode == 'pos':
            # extrapolate with constant position
            # extrapolate as constant values: x, y, theta
            # extrapolate as zeros: v_x, v_y, a_x, a_y, v, a, d_theta
            constant_values = np.array(['x', 'y', 'theta'])
            zero_values     = np.array(['v_x', 'v_y', 'a_x', 'a_y', 'v', 'a', 'd_theta'])

            constant_index = np.where(np.in1d(self.path_data_info(), constant_values))[0]
            zero_index     = np.where(np.in1d(self.path_data_info(), zero_values))[0]

            for i in constant_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = path_useful[0,i], right = path_useful[-1,i])
            for i in zero_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = 0.0, right = 0.0)
            
        elif mode == 'vel':
            # Extrapolate with constant velocity
            # Extrapolate based on last velocity: x, y
            # extrapolate as constant values: theta, v_x, v_y, v
            # extrapolate as zeros: a_x, a_y, a, d_theta
            
            constant_values = np.array(['theta', 'v_x', 'v_y', 'v'])
            zero_values     = np.array(['a_x', 'a_y', 'a', 'd_theta'])

            constant_index = np.where(np.in1d(self.path_data_info(), constant_values))[0]
            zero_index     = np.where(np.in1d(self.path_data_info(), zero_values))[0]

            for i in constant_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = path_useful[0,i], right = path_useful[-1,i])
            for i in zero_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = 0.0, right = 0.0)

            # Find the first and last velocities 
            # Get v_x
            if 'v_x' in self.path_data_info():
                i_v_x = self.path_data_info().index('v_x')
                v_x_start = path_useful[0, i_v_x]
                v_x_end   = path_useful[-1, i_v_x]
            elif np.in1d(['v', 'theta'], self.path_data_info()).all():
                i_v = self.path_data_info().index('v')
                i_theta = self.path_data_info().index('theta')
                v_x_start = path_useful[0, i_v] * np.cos(path_useful[0, i_theta])
                v_x_end   = path_useful[-1, i_v] * np.cos(path_useful[-1, i_theta])
            else:
                v_x_start = (path_useful[1,0] - path_useful[0,0]) / (t_useful[1] - t_useful[0])
                v_x_end   = (path_useful[-1,0] - path_useful[-2,0]) / (t_useful[-1] - t_useful[-2])
            
            # get v_y
            if 'v_y' in self.path_data_info():
                i_v_y = self.path_data_info().index('v_y')
                v_y_start = path_useful[0, i_v_y]
                v_y_end   = path_useful[-1, i_v_y]
            elif np.in1d(['v', 'theta'], self.path_data_info()).all():
                i_v = self.path_data_info().index('v')
                i_theta = self.path_data_info().index('theta')
                v_y_start = path_useful[0, i_v] * np.sin(path_useful[0, i_theta])
                v_y_end   = path_useful[-1, i_v] * np.sin(path_useful[-1, i_theta])
            else:
                v_y_start = (path_useful[1,1] - path_useful[0,1]) / (t_useful[1] - t_useful[0])
                v_y_end   = (path_useful[-1,1] - path_useful[-2,1]) / (t_useful[-1] - t_useful[-2])
            
            v_start = np.array([[v_x_start, v_y_start]])
            v_end   = np.array([[v_x_end, v_y_end]])
            
            # Interpolate first
            path_new[:,0] = np.interp(t, t_useful, path_useful[:,0], left = np.nan, right = np.nan)
            path_new[:,1] = np.interp(t, t_useful, path_useful[:,1], left = np.nan, right = np.nan)

            # Extraploate with the velocities
            path_new[:useful_index[0], :2]  = path_new[useful_index[[0]], :2] + v_start * (t[:useful_index[0]][:, np.newaxis] - t[useful_index[0]])
            path_new[useful_index[-1]:, :2] = path_new[useful_index[[-1]],:2] + v_end * (t[useful_index[-1]:][:, np.newaxis] - t[useful_index[-1]])


        elif mode == 'vel_turn':
            assert 'theta' in self.path_data_info(), "The angle theta is needed for the turning maneuver"
            i_theta = self.path_data_info().index('theta')
            # Extrapolate as turinging maneuver with constant velocity
            # Extrapolate linearly: theta
            # extrapolate as constant values: v, d_theta
            # extrapolate as zeros: a
            # Fit accordingly: x, y, v_x, v_y, a_x, a_y
            constant_values = np.array(['v', 'd_theta'])
            zero_values     = np.array(['a'])
            fit_values      = np.array(['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y'])
            
            constant_index = np.where(np.in1d(self.path_data_info(), constant_values))[0]
            zero_index     = np.where(np.in1d(self.path_data_info(), zero_values))[0]
            fit_index      = np.where(np.in1d(self.path_data_info(), fit_values))[0]

            for i in constant_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = path_useful[0,i], right = path_useful[-1,i])
            for i in zero_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = 0.0, right = 0.0)

            # Find the first and last angle change
            if 'd_theta' in self.path_data_info():
                i_d_theta = self.path_data_info().index('d_theta')
                d_theta_start = path_useful[0, i_d_theta]
                d_theta_end   = path_useful[-1, i_d_theta]
            else:
                d_theta_start = (path_useful[1, i_theta] - path_useful[0, i_theta]) / (t_useful[1] - t_useful[0])
                d_theta_end   = (path_useful[-1, i_theta] - path_useful[-2, i_theta]) / (t_useful[-1] - t_useful[-2])

            # Get the first and last absolute velocities
            if 'v' in self.path_data_info():
                i_v = self.path_data_info().index('v')
                v_start = path_useful[0, i_v]
                v_end   = path_useful[-1, i_v]
            elif np.in1d(['v_x', 'v_y'], self.path_data_info()).all():
                i_v_x = self.path_data_info().index('v_x')
                i_v_y = self.path_data_info().index('v_y')
                v_start = np.sqrt(path_useful[0, i_v_x]**2 + path_useful[0, i_v_y]**2)
                v_end   = np.sqrt(path_useful[-1, i_v_x]**2 + path_useful[-1, i_v_y]**2)
            else:
                # Approximate v_x and v_y based on pos
                v_x_start = (path_useful[1,0] - path_useful[0,0]) / (t_useful[1] - t_useful[0])
                v_y_start = (path_useful[1,1] - path_useful[0,1]) / (t_useful[1] - t_useful[0])
                v_x_end   = (path_useful[-1,0] - path_useful[-2,0]) / (t_useful[-1] - t_useful[-2])
                v_y_end   = (path_useful[-1,1] - path_useful[-2,1]) / (t_useful[-1] - t_useful[-2])
                v_start = np.sqrt(v_x_start**2 + v_y_start**2)
                v_end   = np.sqrt(v_x_end**2 + v_y_end**2)
            
            # Interpolate first
            path_new[:,i_theta] = np.interp(t, t_useful, path_useful[:,i_theta], left = np.nan, right = np.nan)
            for i in fit_index:
                path_new[:, i] = np.interp(t, t_useful, path_useful[:,i], left = np.nan, right = np.nan)

            # Extrapolate the past
            past_index = np.arange(useful_index[0]) 
            if len(past_index) > 0:
                theta_past = path_new[useful_index[0], i_theta] + d_theta_start * (t[past_index] - t[useful_index[0]])
                path_new[past_index, i_theta] = theta_past

                # get velocities
                v_x_past = v_start * np.cos(theta_past)
                if 'v_x' in self.path_data_info():
                    i_v_x = self.path_data_info().index('v_x')
                    path_new[past_index, i_v_x] = v_x_past
                v_y_past = v_start * np.sin(theta_past)
                if 'v_y' in self.path_data_info():
                    i_v_y = self.path_data_info().index('v_y')
                    path_new[past_index, i_v_y] = v_y_past

                # get accelerations
                if 'a_x' in self.path_data_info():
                    i_a_x = self.path_data_info().index('a_x')
                    path_new[past_index, i_a_x] = - v_start * d_theta_start * np.sin(theta_past)
                if 'a_y' in self.path_data_info():
                    i_a_y = self.path_data_info().index('a_y')
                    path_new[past_index, i_a_y] = v_start * d_theta_start * np.cos(theta_past)

                # Get positions
                if np.abs(d_theta_start) < 1e-4:
                    # Use constant velocity
                    v_x_start = v_start * np.cos(path_useful[0, i_theta])
                    v_y_start = v_start * np.sin(path_useful[0, i_theta])
                    path_new[past_index, 0] = path_new[useful_index[0], 0] + v_x_start * (t[past_index] - t[useful_index[0]])
                    path_new[past_index, 1] = path_new[useful_index[0], 1] + v_y_start * (t[past_index] - t[useful_index[0]])
                else:
                    # Find curvature
                    curvature = d_theta_start / v_start

                    # use circle equation to get the center
                    center_x = path_useful[0, 0] - np.sin(path_useful[0, i_theta]) / curvature
                    center_y = path_useful[0, 1] + np.cos(path_useful[0, i_theta]) / curvature
                    
                    # Get the positions by rearranging the circle equation
                    path_new[past_index, 0] = center_x + np.cos(theta_past) / curvature
                    path_new[past_index, 1] = center_y + np.sin(theta_past) / curvature

            # Extrapolate the future
            future_index = np.arange(useful_index[-1] + 1, len(t))
            if len(future_index) > 0:
                path_new[future_index, i_theta] = path_new[useful_index[-1], i_theta] + d_theta_end * (t[future_index] - t[useful_index[-1]])

                # Get velocities
                v_x_future = v_end * np.cos(path_new[future_index, i_theta])
                if 'v_x' in self.path_data_info():
                    i_v_x = self.path_data_info().index('v_x')
                    path_new[future_index, i_v_x] = v_x_future
                v_y_future = v_end * np.sin(path_new[future_index, i_theta])
                if 'v_y' in self.path_data_info():
                    i_v_y = self.path_data_info().index('v_y')
                    path_new[future_index, i_v_y] = v_y_future
                
                # Get accelerations
                if 'a_x' in self.path_data_info():
                    i_a_x = self.path_data_info().index('a_x')
                    path_new[future_index, i_a_x] = - v_end * d_theta_end * np.sin(path_new[future_index, i_theta])
                if 'a_y' in self.path_data_info():
                    i_a_y = self.path_data_info().index('a_y')
                    path_new[future_index, i_a_y] = v_end * d_theta_end * np.cos(path_new[future_index, i_theta])

                # Get positions
                if np.abs(d_theta_end) < 1e-4:
                    # Use constant velocity
                    v_x_end = v_end * np.cos(path_useful[-1, i_theta])
                    v_y_end = v_end * np.sin(path_useful[-1, i_theta])
                    path_new[future_index, 0] = path_new[useful_index[-1], 0] + v_x_end * (t[future_index] - t[useful_index[-1]])
                    path_new[future_index, 1] = path_new[useful_index[-1], 1] + v_y_end * (t[future_index] - t[useful_index[-1]])
                else:
                    # Find curvature
                    curvature = d_theta_end / v_end
                    
                    # use circle equation to get the center
                    center_x = path_useful[-1, 0] - np.sin(path_useful[-1, i_theta]) / curvature
                    center_y = path_useful[-1, 1] + np.cos(path_useful[-1, i_theta]) / curvature
                    
                    # Get the positions by rearranging the circle equation
                    path_new[future_index, 0] = center_x + np.cos(path_new[future_index, i_theta]) / curvature
                    path_new[future_index, 1] = center_y + np.sin(path_new[future_index, i_theta]) / curvature   

        else:
            raise ValueError("The mode is not known")

        # Move theta values back to -pi to pi
        if 'theta' in self.path_data_info():
            i_theta = self.path_data_info().index('theta')
            path_new[:, i_theta] = np.mod(path_new[:, i_theta] + np.pi, 2 * np.pi) - np.pi

        assert np.isfinite(path_new).all(), "There are non-finite values in the path data"
    
        return path_new


    def check_extracted_data_for_saving(self, path_file_adjust, last = False):
        # Get the dataset file
        data_file = self.data_file[:-4] + path_file_adjust
        
        # Get the memory needed to save data right now
        # Count the number of timesteps in each sample
        num_timesteps_in  = np.array([len(t) for t in self.Input_T_local])
        num_timesteps_out = np.array([len(t) for t in self.Output_T_local])
        
        # Get the number of saved agents for each sample
        self.Input_path = pd.DataFrame(self.Input_path_local)
        self.Input_prediction = pd.DataFrame(self.Input_prediction_local)
        num_agents = (~self.Input_path.isnull()).sum(axis=1)
        
        # Get the needed memory per timestep
        memory_per_timestep_in  = 1 + 8 * len(self.Input_prediction.columns) # one extra for timestep
        memory_per_timestep_out = 2 + 8 * 2 # one extra for timestep and recorded
        
        memory_used_path_in  = (num_timesteps_in * num_agents).sum() * memory_per_timestep_in
        memory_used_path_out = (num_timesteps_out * num_agents).sum() * (memory_per_timestep_out)
        
        # Get memory for prediction data
        memory_used_pred = num_timesteps_in.sum() * len(self.Input_prediction.columns) * 8
        
        memory_used = memory_used_path_in + memory_used_path_out + memory_used_pred
        
        # Get the currently available RAM space
        available_memory = self.total_memory - get_used_memory()
        
        # As data needs to be manipulated after loading, check if more than 40% of the memory is used
        # Alternatively, if less than 100 MB are available, save the data
        if last or (memory_used > 0.4 * self.available_memory_data_extraction) or (available_memory < 100 * 2**20):
            # Transform data to dataframes
            self.Input_T       = np.array(self.Input_T_local + [np.random.rand(0)], np.ndarray)[:-1]
            
            self.Output_path   = pd.DataFrame(self.Output_path_local)
            self.Output_T      = np.array(self.Output_T_local + [np.random.rand(0)], np.ndarray)[:-1]
            self.Output_T_pred = np.array(self.Output_T_pred_local + [np.random.rand(0)], np.ndarray)[:-1]
            self.Output_A      = pd.DataFrame(self.Output_A_local)
            self.Output_T_E    = np.array(self.Output_T_E_local, float)
            
            self.num_behaviors = self.num_behaviors_local.copy()

            self.Type     = pd.DataFrame(self.Type_local)
            self.Recorded = pd.DataFrame(self.Recorded_local)
            self.Domain   = pd.DataFrame(self.Domain_local)

            if self.size_given:
                self.Size = pd.DataFrame(self.Size_local)

            # Clear up memory by emptying local data files
            self.Input_prediction_local = []
            self.Input_path_local       = []
            self.Input_T_local          = []
            
            self.Output_path_local   = []
            self.Output_T_local      = []
            self.Output_T_pred_local = []
            self.Output_A_local      = []
            self.Output_T_E_local    = []
            
            self.Type_local     = []
            self.Recorded_local = []
            self.Domain_local   = []

            if self.size_given:
                self.Size_local = []

            self.num_behaviors_local = np.zeros(len(self.Behaviors), int)
            
            # Ensure that dataframes with agent columns have the same order
            Agents = self.Input_path.columns.to_list()
            self.Input_path  = self.Input_path[Agents]
            self.Output_path = self.Output_path[Agents]
            self.Type        = self.Type[Agents]
            self.Recorded    = self.Recorded[Agents]
            if self.size_given:
                self.Size = self.Size[Agents]

            # Ensure that indices of dataframes are the same
            self.Input_path = self.Input_path.reset_index(drop = True)
            self.Input_prediction.index = self.Input_path.index
            self.Output_path.index      = self.Input_path.index
            self.Output_A.index         = self.Input_path.index
            self.Type.index             = self.Input_path.index
            self.Recorded.index         = self.Input_path.index
            self.Domain.index           = self.Input_path.index
            if self.size_given:
                self.Size.index = self.Input_path.index
            
            ## Get the corresponding save file
            data_file_name = os.path.basename(data_file)
            data_file_directory = os.path.dirname(data_file)
            
            # Get data that has to be saved in Domain
            num_behaviors_out = self.Output_A[self.Behaviors].sum(axis=0).to_numpy()
            
            # Make directory
            os.makedirs(data_file_directory, exist_ok=True)
            
            # Find files in same directory that start with file_path_test
            files = [f for f in os.listdir(data_file_directory) if f.startswith(data_file_name)]
            if len(files) > 0:
                # Find the number that is attached to this file
                file_number = np.array([int(f[len(data_file_name)+1:-4]) for f in files], int).max() + 1
                if file_number > 999:
                    raise AttributeError("Too many files have been saved in this directory.")
            else:
                file_number = 0
                
            if last:
                data_file_addition = '_LLL'
            else:
                data_file_addition = '_' + str(file_number).zfill(3)
                
            # Check if there was any previous additions
            self.Domain['path_addition'] = path_file_adjust
            self.Domain['data_addition'] = data_file_addition
            self.Domain['Index_saved']   = self.Input_path.index
                
            
            
            # Apply perturbation if necessary:
            self.Domain['perturbation'] = False
            
            if self.is_perturbed:
                # Check the current data type
                if len(self.path_data_info()) > 2:
                    raise AttributeError("Perturbation is currently not possible for data that includes more information than positions.")


                # Get unperturbed save file
                data_file_perturbde_parts = data_file.split('--Pertubation_')
                data_file_unperturbed = data_file_perturbde_parts[0] + data_file_perturbde_parts[1][3:]
                data_file_unperturbed_save = data_file_unperturbed + data_file_addition + '_data.npy'
                domain_file_unperturbed_save = data_file_unperturbed + data_file_addition + '_domain.npy'
                agent_file_save = data_file_unperturbed + data_file_addition + '_AM.npy'
                
                # Check if the unperturbed dataset allready exists
                if not os.path.isfile(data_file_unperturbed_save):          
                    # Get the unperturbed save data
                    save_data_unperturbed = np.array([
                                                self.Input_prediction,
                                                self.Input_path,
                                                self.Input_T,
                                                
                                                self.Output_path,
                                                self.Output_T,
                                                self.Output_T_pred,
                                                self.Output_A,
                                                self.Output_T_E, 0], object)
                    
                    # Overwrite the Scenario column in self.Domain
                    perturbed_scenario = self.Domain.Scenario.copy()
                    self.Domain.Scenario = self.get_name()['print']

                    save_domain_unperturbed = np.array([self.Domain, self.num_behaviors, num_behaviors_out, Agents, 0], object)

                    if self.size_given:
                        save_agent_unperturbed = np.array([self.Type, self.Size, self.Recorded, 0], object)
                    else:
                        save_agent_unperturbed  = np.array([self.Type, self.Recorded, 0], object)
                    
                    
                    # Save the unperturbed data
                    np.save(data_file_unperturbed_save, save_data_unperturbed)
                    np.save(domain_file_unperturbed_save, save_domain_unperturbed)
                    np.save(agent_file_save, save_agent_unperturbed)

                    # Reset the Scenario column in self.Domain
                    self.Domain.Scenario = perturbed_scenario
                
                # Apply the perturbation
                self = self.Perturbation.perturb(self)
                
                # Set perturbation to True in the Domain to be saved in the perturbed data
                self.Domain['perturbation'] = True
                
                if self.classification_useful:
                    # save old num_samples_path_pred
                    num_samples_path_pred = self.num_samples_path_pred + 0
                    self.num_samples_path_pred = 1
                    
                    # Exctract the update behavior for perturbed output
                    Output_A, Output_T_E = self._path_to_class_and_time(self.Output_path, np.arange(len(self.Output_path)), self.Output_T_pred, self.Domain)
                    
                    # Reset num_samples_path_pred
                    self.num_samples_path_pred = num_samples_path_pred
                    
                    # Set Output_A to bool
                    self.Output_A = Output_A.astype(int).astype(bool)
                    assert (self.Output_A.sum(1) == 1).all(), "Behavior extraction of perturbed data failed."
                    
                    # Set Output_T_E from dataframe to corresponding value
                    self.Output_T_E = np.stack(Output_T_E.to_numpy()[Output_A.to_numpy().astype(bool)], 0).mean(1)
                    assert np.isfinite(self.Output_T_E).all(), "Behavior time extraction of perturbed data failed."
                
                # Remove unperturbed columns from domain
                self.Domain = self.Domain.drop(columns = ['Unperturbed_input', 'Unperturbed_output'], errors = 'ignore')
            
            save_data = np.array([
                                    self.Input_prediction,
                                    self.Input_path,
                                    self.Input_T,
                                    
                                    self.Output_path,
                                    self.Output_T,
                                    self.Output_T_pred,
                                    self.Output_A,
                                    self.Output_T_E, 0], object)
            
            save_domain = np.array([self.Domain, self.num_behaviors, num_behaviors_out, Agents, 0], object)

            if self.size_given:
                save_agent = np.array([self.Type, self.Size, self.Recorded, 0], object)
            else:
                save_agent = np.array([self.Type, self.Recorded, 0], object)
            
            
            data_file_save = data_file + data_file_addition + '_data.npy'
            domain_file_save = data_file + data_file_addition + '_domain.npy'
            agent_file_save = data_file + data_file_addition + '_AM.npy'
            
            # Save the data
            np.save(data_file_save, save_data)
            np.save(domain_file_save, save_domain)
            np.save(agent_file_save, save_agent)

            # Clear up memory by deleting the self attributes just saved
            del self.Input_prediction
            del self.Input_path
            del self.Input_T

            del self.Output_path
            del self.Output_T
            del self.Output_T_pred
            del self.Output_A
            del self.Output_T_E

            del self.Type
            del self.Recorded
            del self.Domain
            


            
        if not last:
            # return the currently used memory percentage
            return memory_used / (0.4 * self.available_memory_data_extraction)
        
    
    def get_data(self, dt, num_timesteps_in, num_timesteps_out):
        '''
        Parameters
        ----------


        Returns
        -------
        input and output data

        '''
        # Get the current time step size
        assert self.prediction_time_set, "No prediction time was set."

        self.data_file = self.data_params_to_string(dt, num_timesteps_in, num_timesteps_out)
        data_file_final = self.data_file[:-4] + '_LLL_LLL_data.npy'

        # check if same data set has already been done in the same way
        if not os.path.isfile(data_file_final):
            # load initial dataset, if not yet done
            self.load_raw_data()
            
            

            # If necessary, load constant gap size
            if ((self.t0_type[:9] == 'col_equal') or 
                ('col_equal' in [t0_type_extra[:9] for t0_type_extra in self.T0_type_compare])):
                self.determine_dtc_boundary()
            
            # Go through original data
            for i_orig_path in range(self.number_original_path_files):
                # Get path name adjustment
                path_file = self.file_path + '--all_orig_paths'
                
                # Get path file adjustment
                if self.number_original_path_files == 1:
                    # Get path name adjustment
                    path_file_adjust = '_LLL'
                else:
                    # Get path name adjustment
                    if i_orig_path < self.number_original_path_files - 1:
                        path_file_adjust = '_' + str(i_orig_path).zfill(3)
                    else:
                        path_file_adjust = '_LLL'
                        
                path_file += path_file_adjust + '.npy'
                
                # Check if data is allready completely extracted, making renew extraction unnecessary
                data_file_test = self.data_file[:-4] + path_file_adjust + '_LLL_data.npy'
                
                if os.path.isfile(data_file_test):
                    continue
                
            
                    
                # Get path data
                if self.number_original_path_files == 1:
                    # Get the allready loaded data
                    Path_loaded = self.Path
                    Type_old_loaded = self.Type_old
                    Size_old_loaded = self.Size_old
                    T_loaded = self.T
                    Domain_old_loaded = self.Domain_old
                    num_samples_loaded = self.num_samples
            
                else:
                    # Load the data
                    Loaded_data = np.load(path_file, allow_pickle=True)
                    Path_loaded, Type_old_loaded, Size_old_loaded, T_loaded, Domain_old_loaded, num_samples_loaded = self.extract_loaded_data(Loaded_data)

                # Get the currently available RAM space
                self.available_memory_data_extraction = self.total_memory - get_used_memory()

                # Adjust base data file name accordingly
                self.get_data_from_orig_path(Path_loaded, Type_old_loaded, Size_old_loaded, T_loaded, Domain_old_loaded, num_samples_loaded, path_file, path_file_adjust)
                
        
        # Get the number of files
        domain_files, self.number_data_files = self.get_number_of_data_files()
        
        # If only one file is available, load the data
        if self.number_data_files == 1:
            domain_file = self.data_file[:-4] + '_LLL_LLL_domain.npy'
            agent_file = self.data_file[:-4] + '_LLL_LLL_AM.npy'
            data_file = self.data_file[:-4] + '_LLL_LLL_data.npy'
            
            # Load the data
            [self.Input_prediction,
            self.Input_path,
            self.Input_T,

            self.Output_path,
            self.Output_T,
            self.Output_T_pred,
            self.Output_A,
            self.Output_T_E, _] = np.load(data_file, allow_pickle=True)
            
            [self.Domain, self.num_behaviors, self.num_behaviors_out, self.Agents, _] = np.load(domain_file, allow_pickle=True)

            Agent_data = np.load(agent_file, allow_pickle=True)
            if len(Agent_data) == 3:
                [self.Type, self.Recorded, _] = Agent_data
                self.Size = None
            else:
                assert len(Agent_data) == 4, "The loaded data has the wrong length."
                [self.Type, self.Size, self.Recorded, _] = Agent_data
        
        else:
            # Free up memory
            self.Input_prediction = None
            self.Input_path = None
            self.Input_T = None
            
            self.Output_path = None
            self.Output_T = None
            self.Output_T_pred = None
            self.Output_A = None
            
            self.Type = None
            self.Recorded = None
            
            # Combine the Domains, Output_T_pred and number of behaviors
            self.Domain = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Output_T_pred = np.zeros(0, object)
            
            self.num_behaviors     = np.zeros(len(self.Behaviors), int)
            self.num_behaviors_out = np.zeros(len(self.Behaviors), int)
            
            self.Agents = []
            
            # Get needed data files 
            for domain_file in domain_files:
                Domain, num_behaviors, num_behaviors_out, Agents, _ = np.load(domain_file, allow_pickle=True) 
                
                self.Domain = pd.concat([self.Domain, Domain], axis = 0)
                self.num_behaviors     += num_behaviors
                self.num_behaviors_out += num_behaviors_out

                # load Output_T_pred
                Output_T_pred = np.load(domain_file[:-11] + '_data.npy', allow_pickle=True)[5]

                # Make sure to use the right index
                Output_T_pred = Output_T_pred[Domain.Index_saved.to_numpy()]
                self.Output_T_pred = np.concatenate([self.Output_T_pred, Output_T_pred], 0)

                self.Agents += Agents
                
        self.Agents, index = np.unique(self.Agents, return_index = True)
        # Sort the agents, so that the entry appearing first is the one that is kept first
        self.Agents = list(self.Agents[np.argsort(index)])

        self.data_loaded = True

        if 'graph_id' in self.Domain.columns:
            graph_ids = self.Domain.graph_id.to_numpy()
            Path_ids = self.Domain[['Path_ID', 'path_addition']].to_numpy().astype(str)

            # Get unique Path_ids, with index
            index_unique_path = np.unique(Path_ids, axis = 0, return_index = True)[1]
            graph_ids_old = graph_ids[index_unique_path]

            # For each unique graph_id, check how often they are repeated
            unqiue_graph_id, counts = np.unique(graph_ids_old, return_counts = True)

            # Transfer to dictionary
            self.graph_count = dict(zip(unqiue_graph_id, counts))

            if np.max(counts) == 1:
                self.graph_count_always_one = True
            else:
                self.graph_count_always_one = False

        # check if dataset is useful
        if len(self.Domain) < 100:
            return "there are not enough samples for a reasonable training process."

        if self.classification_useful and np.sort(self.num_behaviors_out)[-2] < 10:
            return "the dataset is too unbalanced for a reasonable training process."

        return None

        
    
    def change_agent_name(self, old_name, new_name, domain_file):
        assert isinstance(old_name, str), "The old name has to be a string."
        assert isinstance(new_name, str), "The new name has to be a string."
        
        agent_file = domain_file[:-10] + 'AM.npy'
        data_file = domain_file[:-10] + 'data.npy'
    
        # Load the data
        data_data = np.load(data_file, allow_pickle=True)
        [_, Input_path, _, Output_path, _, Output_T_pred, _, _, _] = data_data
        
        if old_name in Input_path.columns:
            print(old_name + " column is in the input data. Replace it with " + new_name + ".")
            assert old_name in Output_path.columns, "The AV column is missing in the output data."
            
            # rename old_name columns to new_name column
            Input_path = Input_path.rename(columns = {old_name: new_name})
            Output_path = Output_path.rename(columns = {old_name: new_name})
            
            data_data[1] = Input_path
            data_data[3] = Output_path
            
            np.save(data_file, data_data)
        
        domain_data = np.load(domain_file, allow_pickle=True) 
        [_, _, _, Agents, _] = domain_data # Agents is a list
        if old_name in Agents:
            print(old_name + " is in the Agents list. Replace it with " + new_name + ".")
            # Replace old_name with new_name
            Agents[Agents.index(old_name)] = new_name
            domain_data[3] = Agents
            np.save(domain_file, domain_data)
        
        

        Agent_data = np.load(agent_file, allow_pickle=True)
        if len(Agent_data) == 3:
            [Type, Recorded, _] = Agent_data
            Size = None
        else:
            assert len(Agent_data) == 4, "The loaded data has the wrong length."
            [Type, Size, Recorded, _] = Agent_data
        
        if old_name in Type.columns:
            print(old_name + " column is in the Type data. Replace it with " + new_name + ".")
            assert old_name in Recorded.columns, "The AV column is missing in the Recorded data."
            
            Type = Type.rename(columns = {old_name: new_name})
            Recorded = Recorded.rename(columns = {old_name: new_name})
            
            Agent_data[0] = Type
            Agent_data[-2] = Recorded
            
            if Size is not None:
                assert old_name in Size.columns, "The AV column is missing in the Size data."
                Size = Size.rename(columns = {old_name: new_name})
                Agent_data[1] = Size
            
            np.save(agent_file, Agent_data)






    ##############################################################################################################
    ##############################################################################################################
    ###                                                                                                        ###
    ###                                        Framework interactions                                          ###
    ###                                                                                                        ###
    ##############################################################################################################
    ##############################################################################################################


    # %% Implement information sharing with other modules
    def change_result_directory(self, filepath, new_path_addon, new_file_addon, file_type = '.npy'):
        path_old = os.path.dirname(filepath)
        path_list = path_old.split(os.sep)
        path_list[-1] = new_path_addon
        path = os.sep.join(path_list)

        file_old = os.path.basename(filepath)
        if len(new_file_addon) > 0:
            file = file_old[:-4]  + '--' + new_file_addon + file_type
        else:
            file = file_old
        return path + os.sep + file
    
    
    def _interpolate_image(self, imgs_rot, pos_old, image):
        
        useful = ((0 <= pos_old[...,0]) & (pos_old[...,0] <= image.shape[1] - 1) &
                  (0 <= pos_old[...,1]) & (pos_old[...,1] <= image.shape[0] - 1))
        
        useful_ind, useful_row, useful_col = torch.where(useful)
        pos_old = pos_old[useful_ind, useful_row, useful_col,:]
        
        pos_up  = torch.ceil(pos_old).to(dtype = torch.int64)
        pos_low = torch.floor(pos_old).to(dtype = torch.int64)
        
        imgs_rot_uu = image[pos_up[:,1],  pos_up[:,0]]
        imgs_rot_ul = image[pos_up[:,1],  pos_low[:,0]]
        imgs_rot_lu = image[pos_low[:,1], pos_up[:,0]]
        imgs_rot_ll = image[pos_low[:,1], pos_low[:,0]]
        
        del pos_up, pos_low
        
        pos_fac = torch.remainder(pos_old, 1)
        
        imgs_rot_u = imgs_rot_uu * (pos_fac[:,[0]]) + imgs_rot_ul * (1 - pos_fac[:,[0]])
        imgs_rot_l = imgs_rot_lu * (pos_fac[:,[0]]) + imgs_rot_ll * (1 - pos_fac[:,[0]])
        
        del imgs_rot_uu, imgs_rot_ul, imgs_rot_lu, imgs_rot_ll
        
        imgs_rot_v = imgs_rot_u * (pos_fac[:,[1]]) + imgs_rot_l * (1 - pos_fac[:,[1]])
        
        if imgs_rot.shape[-1] == 1:
            imgs_rot[useful] = imgs_rot_v.mean(-1, keepdims = True).to(dtype = image.dtype)
        else:
            imgs_rot[useful] = imgs_rot_v.to(dtype = image.dtype)
        
        return imgs_rot
    
    
    def return_batch_images(self, domain, center, rot_angle, target_width, target_height, grayscale,
                            Imgs_rot, Imgs_index, print_progress = False):
        if self.includes_images():
            if print_progress:
                print('')
                print('Load needed images:', flush = True)
            
            
            # Find the gpu
            if not torch.cuda.is_available():
                device = torch.device('cpu')
                raise TypeError("The GPU has gone and fucked itself")
            else:
                if torch.cuda.device_count() == 1:
                    # If you have CUDA_VISIBLE_DEVICES set, which you should,
                    # then this will prevent leftover flag arguments from
                    # messing with the device allocation.
                    device = 'cuda:0'
            
                device = torch.device(device)
            
            # Get domain dividers
            Locations = domain.image_id.to_numpy()
            Path_additions = domain.path_addition.to_numpy()

            # Preemptively load first data
            self.load_raw_images('_000')
            if print_progress:
                print('')
                print('Extract rotation matrix', flush = True)
            
            # check if images are float
            if self.Images.Image.iloc[0].max() > 1:
                rgb = True
                assert self.Images.Image.iloc[0].max() < 256
            else:
                rgb = False
            
            if target_width is None:
                target_width = 500
                
            if target_height is None:
                target_height = 500
                
            if rot_angle is None:
                first_stage = False
            else:
                first_stage = True
            
            if hasattr(domain, 'rot_angle'):
                second_stage = True
                if print_progress:
                    print('Second rotation state is available.', flush = True)
            else: 
                second_stage = False
            
            if first_stage:
                # Get rotation matrix (R * x is from orignal to current)
                Rot_matrix = np.array([[np.cos(rot_angle), np.sin(rot_angle)],
                                       [-np.sin(rot_angle), np.cos(rot_angle)]]).transpose(2,0,1)
            
                Rot_matrix = torch.from_numpy(Rot_matrix).float().to(device = device)
                center     = torch.from_numpy(center).float().to(device = device)
                
            if second_stage:
                Rot_matrix_old = np.array([[np.cos(domain.rot_angle), np.sin(domain.rot_angle)],
                                           [-np.sin(domain.rot_angle), np.cos(domain.rot_angle)]]).transpose(2,0,1)
                center_old = np.stack([domain.x_center, domain.y_center], -1)
        
                Rot_matrix_old = torch.from_numpy(Rot_matrix_old).float().to(device = device)
                center_old     = torch.from_numpy(center_old).float().to(device = device)
            
            # setup meshgrid
            height_fac = (target_height - 1) / 2 
            width_fac = (target_width - 1) / 2           
            Ypx, Xpx = torch.meshgrid(torch.arange(-height_fac, height_fac + 0.5, device = device), 
                                      torch.arange(-width_fac, width_fac + 0.5, device = device),
                                      indexing = 'ij')
            
            Pos_old = torch.stack([Xpx, Ypx], -1).unsqueeze(0)
            
            Pos_old[...,1] *= -1
            # Pos_old: Position in goal coordinate system in Px.

            n = 250
            image_num = 0

            # Go through unique path_additions
            for path_addition in np.unique(Path_additions):
                # Load the images
                self.load_raw_images(path_addition)

                # Get the corresponding index
                path_indices = np.where(Path_additions == path_addition)[0]
                Locations_unique_path = Locations[path_indices]

                for location in np.unique(Locations_unique_path):
                    loc_indices = np.where(Locations_unique_path == location)[0]
                    
                    loc_Image = torch.from_numpy(self.Images.Image.loc[location]).to(device = device)
                    loc_M2px  = float(self.Images.Target_MeterPerPx.loc[location])
                    for i in range(0, len(loc_indices), n):
                        torch.cuda.empty_cache()
                        Index_local = np.arange(i, min(i + n, len(loc_indices)))
                        Index = path_indices[loc_indices[Index_local]]
                        
                        if print_progress:
                            print('rotating images ' + str(image_num + 1) + ' to ' + str(image_num + len(Index)) + 
                                  ' of ' + str(len(domain)) + ' total', flush = True)
                        image_num = image_num + len(Index)
                        Index_torch = torch.from_numpy(Index).to(device = device, dtype = torch.int64)
                        
                        # Position in goal coordinate system in Px.
                        pos_old = Pos_old * loc_M2px
                        
                        # Get position in the coordinate system in self.paths
                        if first_stage:
                            pos_old = torch.matmul(pos_old, Rot_matrix[Index_torch].unsqueeze(1))
                            pos_old = pos_old + center[Index_torch,:].unsqueeze(1).unsqueeze(1)
                        
                        # Get position im Image aligned coordinate system
                        if second_stage:
                            pos_old = torch.matmul(pos_old, Rot_matrix_old[Index_torch].unsqueeze(1))
                            pos_old = pos_old + center_old[Index_torch,:].unsqueeze(1).unsqueeze(1)
                        
                        # Get pixel position in Image
                        pos_old = pos_old / loc_M2px
                        pos_old[...,1] *= -1
                        
                        torch.cuda.empty_cache()
                        
                        # Enforce grayscale here using the gpu
                        if grayscale:
                            imgs_rot = torch.zeros((len(Index), target_height, target_width, 1), dtype = loc_Image.dtype, device = device)
                        else:
                            imgs_rot = torch.zeros((len(Index), target_height, target_width, 3), dtype = loc_Image.dtype, device = device)
                        
                        
                        imgs_rot = self._interpolate_image(imgs_rot, pos_old, loc_Image)
                        
                        if not rgb:
                            imgs_rot = 255 * imgs_rot
                            
                        torch.cuda.empty_cache()
                        Imgs_rot[Imgs_index[Index]] = imgs_rot.detach().cpu().numpy().astype('uint8')
        
            return Imgs_rot
        else:
            return None
            


    def cut_sceneGraph(self, loc_Graph, X, radius, wave_length = 1.0):
        # loc_Graph: SceneGraph of the location, as a pandas dataframe
        # X: Position of the agents in the location, with shape num_agents x 2
        # radius: Radius of the scene graph, in meters

        # Only keep non nan agents
        X_a = X[np.isfinite(X).all(-1)]
        assert len(X_a) > 0, "There are no agents in the scene."
        X_a = X_a[np.newaxis, :] # shape = (1, num_agents, 2)
        

        # Get contents of loc_Graph
        num_nodes = loc_Graph.num_nodes + 0 # Number of nodes in the scene graph
        lane_idcs = loc_Graph.lane_idcs.copy() # Lane indices of the nodes
        pre_pairs = loc_Graph.pre_pairs.copy() # Predecessor pairs of the nodes, array of shape (num_pre_pairs, 2)
        suc_pairs = loc_Graph.suc_pairs.copy() # Successor pairs of the nodes, array of shape (num_suc_pairs, 2)
        left_pairs = loc_Graph.left_pairs.copy() # Left pairs of the nodes, array of shape (num_left_pairs, 2)
        right_pairs = loc_Graph.right_pairs.copy() # Right pairs of the nodes, array of shape (num_right_pairs, 2)
        left_boundaries = np.array(list(loc_Graph.left_boundaries) + [0], object)[:-1] # Left boundaries of the nodes, array of shape (num_segments), with each element being an array of shape (num_points, 2)
        right_boundaries = np.array(list(loc_Graph.right_boundaries) + [0], object)[:-1] # Right boundaries of the nodes, array of shape (num_segments), with each element being an array of shape (num_points, 2)
        centerlines = np.array(list(loc_Graph.centerlines) + [0], object)[:-1] # Centerlines of the nodes, array of shape (num_segments), with each element being an array of shape (num_points, 2)

        # Go through segments
        Keep_segments = np.zeros(len(left_boundaries), bool)
        Keep_nodes = np.zeros(num_nodes, bool)
        
        # Get original lane id range, using
        new_lane = np.where(lane_idcs[1:] != lane_idcs[:-1])[0] + 1
        lane_ids = np.concatenate((lane_idcs[[0]], lane_idcs[new_lane]), 0)
        assert len(lane_ids) == len(centerlines), "Lane ids are not correct."
        
        # Check that pair ids are within lane_ids
        assert np.all(np.isin(pre_pairs.flatten(), lane_ids)), "Predecessor pair ids are not within lane_ids."
        assert np.all(np.isin(suc_pairs.flatten(), lane_ids)), "Successor pair ids are not within lane_ids."
        assert np.all(np.isin(left_pairs.flatten(), lane_ids)), "Left pair ids are not within lane_ids."
        assert np.all(np.isin(right_pairs.flatten(), lane_ids)), "Right pair ids are not within lane_ids."

        for i_lane, lane_id in enumerate(lane_ids):
            left_pts = left_boundaries[i_lane] # shape = (num_points, 2)
            right_pts = right_boundaries[i_lane] # shape = (num_points, 2)
            centerline_pts = centerlines[i_lane] # shape = (num_points, 2)

            # Get distance to agents (nanmin over the agents)
            dist_left   = np.nanmin(np.linalg.norm(left_pts[:,np.newaxis] - X_a, axis = -1), axis = 1)
            dist_right  = np.nanmin(np.linalg.norm(right_pts[:,np.newaxis] - X_a, axis = -1), axis = 1)
            dist_center = np.nanmin(np.linalg.norm(centerline_pts[:,np.newaxis] - X_a, axis = -1), axis = 1)

            # Check if the agent is within the radius
            keep_left = dist_left < radius
            keep_right = dist_right < radius
            keep_center = dist_center < radius

            # Check if the agent is to be kept at all
            if keep_center.sum() < 2:
                continue
            else:
                # If the last center node is not kept, remove all successor connections
                if not keep_center[-1]:
                    # Get all successors
                    suc_idcs = suc_pairs[:,0] == lane_id
                    suc_pairs = suc_pairs[~suc_idcs]

                # If the first center node is not kept, remove all predecessor connections
                if not keep_center[0]:
                    # Get all predecessors
                    pre_idcs = pre_pairs[:,0] == lane_id
                    pre_pairs = pre_pairs[~pre_idcs]
                
                # Remove the nodes that are not kept
                left_pts = left_pts[keep_left]
                right_pts = right_pts[keep_right]
                centerline_pts = centerline_pts[keep_center]

                # Filter out intemediat steps so general distance are kept within roughly 1 meters
                dist_cons_left = np.linalg.norm(left_pts[1:] - left_pts[:-1], axis = 1)
                dist_cons_right = np.linalg.norm(right_pts[1:] - right_pts[:-1], axis = 1)
                dist_cons_center = np.linalg.norm(centerline_pts[1:] - centerline_pts[:-1], axis = 1)

                # Cumulative distance
                cum_dist_left = np.concatenate([[0], np.cumsum(dist_cons_left / wave_length)]).astype(int)
                cum_dist_right = np.concatenate([[0], np.cumsum(dist_cons_right / wave_length)]).astype(int)  
                cum_dist_center = np.concatenate([[0], np.cumsum(dist_cons_center / wave_length)]).astype(int)

                # get points that are unnecessary
                remove_left = np.zeros(len(left_pts), bool)
                remove_left[1:] = cum_dist_left[1:] == cum_dist_left[:-1]

                remove_right = np.zeros(len(right_pts), bool)
                remove_right[1:] = cum_dist_right[1:] == cum_dist_right[:-1]

                remove_center = np.zeros(len(centerline_pts), bool)
                remove_center[1:-1] = cum_dist_center[1:-1] == cum_dist_center[:-2]

                # Update the points
                left_pts = left_pts[~remove_left]
                right_pts = right_pts[~remove_right]
                centerline_pts = centerline_pts[~remove_center]

                # Set lanes
                Keep_segments[i_lane] = True
                left_boundaries[i_lane] = left_pts
                right_boundaries[i_lane] = right_pts
                centerlines[i_lane] = centerline_pts

                # Update keep center
                keep_center[keep_center] = ~remove_center

                assert keep_center.sum() >= 2, "Not enough center points kept."

            # Number nodes to keep
            keep_num_nodes = keep_center.sum() - 1

            # Get the current nodes for this lane
            lane_nodes_id = np.where(lane_idcs == lane_id)[0]

            # Get the nodes to keep 
            Keep_nodes[lane_nodes_id[:keep_num_nodes]] = True
        
        # Keep lane segements
        left_boundaries = left_boundaries[Keep_segments]
        right_boundaries = right_boundaries[Keep_segments]
        centerlines = centerlines[Keep_segments]
        
        # Update lany types
        lane_type = [loc_Graph.lane_type[i] for i in np.where(Keep_segments)[0]]

        # Prepare lane_id_map
        max_id = lane_ids.max()
        if len(suc_pairs) > 0:
            max_id = max(max_id, suc_pairs.max())
        if len(pre_pairs) > 0:
            max_id = max(max_id, pre_pairs.max())
        if len(left_pairs) > 0:
            max_id = max(max_id, left_pairs.max())
        if len(right_pairs) > 0:
            max_id = max(max_id, right_pairs.max())
        lane_id_map = np.full((max_id + 1,), -1, dtype = int)
        
        # Update lane ids
        lane_ids = lane_ids[Keep_segments] 
        
        # Update lane ids
        if Keep_segments.any():
            assert Keep_nodes.any(), "No nodes are kept, but segments are kept."
            # Fill in mapping
            lane_id_map[lane_ids] = np.arange(len(lane_ids))
        else:
            assert not Keep_nodes.any(), "Nodes are kept, but no segments are kept."
            # Print warning
            print('Warning: With radius ' + str(radius) + ' no lane segments are foudn around the agents at positions ')
            print(X_a[0])       
            print('')  
          
        lane_idcs = lane_idcs[Keep_nodes]
        num_nodes = len(lane_idcs)
        
        if num_nodes > 0:
            assert np.array_equal(np.unique(lane_idcs), np.sort(lane_ids)), "Lane idcs are wrongly kept." 
            
        # Rename the pairs
        suc_pairs   = lane_id_map[suc_pairs]
        pre_pairs   = lane_id_map[pre_pairs]
        left_pairs  = lane_id_map[left_pairs]
        right_pairs = lane_id_map[right_pairs]

        # Rename the lane idcs
        lane_idcs   = lane_id_map[lane_idcs]
        
        # Remove pairs that containt removed nodes (now = -1)
        suc_pairs   = suc_pairs[(suc_pairs >= 0).all(1)]
        pre_pairs   = pre_pairs[(pre_pairs >= 0).all(1)]
        left_pairs  = left_pairs[(left_pairs >= 0).all(1)]
        right_pairs = right_pairs[(right_pairs >= 0).all(1)]

        # Assemble new graph
        loc_Graph_cut = pd.Series([num_nodes, lane_idcs, pre_pairs, suc_pairs, left_pairs, right_pairs, 
                                   left_boundaries, right_boundaries, centerlines, lane_type],
                                   index = ['num_nodes', 'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs',
                                             'left_boundaries', 'right_boundaries', 'centerlines', 'lane_type'])
        
        # Get the missing segments
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loc_Graph_cut = self.add_node_connections(loc_Graph_cut, device = device)

        return loc_Graph_cut

    def return_batch_sceneGraphs(self, domain, X, radius, wave_length, SceneGraphs, Graphs_Index, print_progress = False):
        if self.includes_sceneGraphs():
            if print_progress:    
                print('')
                print('Load needed scene graphs:', flush = True)
            
            # Get domain dividers
            Locations = domain.graph_id.to_numpy()
            Path_additions = domain.path_addition.to_numpy()
            
            graph_num = 0

            # Go through unique path_additions
            for path_addition in np.unique(Path_additions):
                # Load the scene graphs
                self.load_raw_sceneGraphs(path_addition)

                # Get the corresponding index
                path_indices = np.where(Path_additions == path_addition)[0]
                Locations_unique_path = Locations[path_indices]

                for location in np.unique(Locations_unique_path):
                    loc_indices = np.where(Locations_unique_path == location)[0]
                    
                    loc_Graph = self.SceneGraphs.loc[location]

                    if self.graph_count_always_one:
                        num = 1
                    else:
                        num = self.graph_count[location]

                    if (radius is None) or (num == 1):
                        Index = path_indices[loc_indices]
                        SceneGraphs[Graphs_Index[Index]] = [loc_Graph] * len(Index)
                    else:
                        for i in range(len(loc_indices)):
                            index = path_indices[loc_indices[i]]
                            
                            if print_progress and np.mod(graph_num, 100) == 0:
                                print('retrieving graphs ' + str(graph_num + 1) + 
                                    ' of ' + str(len(domain)) + ' total', flush = True)
                                
                            loc_Graph_cut = self.cut_sceneGraph(loc_Graph, X[index], radius, wave_length)
                            SceneGraphs[Graphs_Index[index]] = loc_Graph_cut

                            graph_num += 1
        
            return SceneGraphs
        else:
            return None
        
        

    # %% Implement transformation functions
    def train_path_models(self):
        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified")

        if not self.path_models_trained:
            self.path_models = pd.Series(np.empty(len(self.Behaviors), object), index=self.Behaviors)

            # Create data interface out of self
            # Get parameters
            parameters = [None,  self.num_samples_path_pred,
                          self.enforce_num_timesteps_out, self.enforce_prediction_time,
                          self.exclude_post_crit, self.allow_extrapolation,
                          self.agents_to_predict, 'no', False]
            
            # Get data set dict
            data_set_dict = {'scenario': self.__class__.__name__,
                             'max_num_agents': self.max_num_agents,
                             't0_type': self.t0_type,
                             'conforming_t0_types': self.T0_type_compare}
            
            # get the data_interface setup
            data = data_interface(data_set_dict, parameters)
            data.get_data(self.dt,
                          (self.num_timesteps_in_real, self.num_timesteps_in_need), 
                          (self.num_timesteps_out_real, self.num_timesteps_out_need),
                          keep_useless_samples = True)

            # Go through behaviors
            for beh in self.Behaviors:
                self.path_models[beh] = self.model_class_to_path({}, data, None, True, beh)
                self.path_models[beh].train()

            self.path_models_trained = True

    def transform_outputs(self, output, model_pred_type, metric_pred_type):
        # Check if tranformation is necessary
        if model_pred_type == metric_pred_type:
            return output

        # If the metric requires trajectory predictions
        if metric_pred_type == 'path_all_wo_pov':
            if model_pred_type == 'class':
                [Pred_index, Output_A_pred] = output
                Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain)

                Output_path_pred, Output_path_pred_probs = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'class_and_time':
                [Pred_index, Output_A_pred, Output_T_E_pred] = output
                Output_path_pred, Output_path_pred_probs = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'path_all_wi_pov':
                [Pred_index, Output_path_pred, Output_path_pred_probs] = output
            else:
                raise AttributeError(
                    "This type of output produced by the model is not implemented")

            Output_path_pred, Output_path_pred_probs = self.path_remove_pov_agent(Output_path_pred, Output_path_pred_probs, Pred_index, self.Domain)
            output_trans = [Pred_index, Output_path_pred, Output_path_pred_probs]

        elif metric_pred_type == 'path_all_wi_pov':
            if model_pred_type == 'class':
                [Pred_index, Output_A_pred] = output
                Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain)

                Output_path_pred, Output_path_pred_probs = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'class_and_time':
                [Pred_index, Output_A_pred, Output_T_E_pred] = output
                Output_path_pred, Output_path_pred_probs = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'path_all_wo_pov':
                [Pred_index, Output_path_pred, Output_path_pred_probs] = output
                Output_path_pred, Output_path_pred_probs = self.path_add_pov_agent(Output_path_pred, Output_path_pred_probs, Pred_index, self.Domain)
            else:
                raise AttributeError(
                    "This type of output produced by the model is not implemented")
            output_trans = [Pred_index, Output_path_pred, Output_path_pred_probs]

        # If the metric requires class predictions
        elif metric_pred_type == 'class':
            if model_pred_type == 'path_all_wo_pov':
                [Pred_index, Output_path_pred, Output_path_pred_probs] = output
                Output_path_pred, Output_path_pred_probs = self.path_add_pov_agent(Output_path_pred, Output_path_pred_probs, Pred_index, self.Domain)

                [Output_A_pred, _] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Domain)

            elif model_pred_type == 'path_all_wi_pov':
                [Pred_index, Output_path_pred, Output_path_pred_probs] = output
                [Output_A_pred, _] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Domain)

            elif model_pred_type == 'class_and_time':
                [Pred_index, Output_A_pred, _] = output
            else:
                raise AttributeError(
                    "This type of output produced by the model is not implemented")

            output_trans = [Pred_index, Output_A_pred]

        # If the metric requires class andf tiem predictions
        elif metric_pred_type == 'class_and_time':
            if model_pred_type == 'path_all_wo_pov':
                [Pred_index, Output_path_pred, Output_path_pred_probs] = output
                Output_path_pred, Output_path_pred_probs = self.path_add_pov_agent(Output_path_pred, Output_path_pred_probs, Pred_index, self.Domain)
                [Output_A_pred, Output_T_E_pred] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Domain)

            elif model_pred_type == 'path_all_wi_pov':
                [Pred_index, Output_path_pred, Output_path_pred_probs] = output
                [Output_A_pred, Output_T_E_pred] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Domain)

            elif model_pred_type == 'class':
                [Pred_index, Output_A_pred] = output
                Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain)
            else:
                raise AttributeError(
                    "This type of output produced by the model is not implemented")
            output_trans = [Pred_index, Output_A_pred, Output_T_E_pred]

        else:
            raise AttributeError(
                "This type of output required by the metric is not implemented")
        return output_trans

    def path_to_class_and_time_sample(self, paths, t, domain):
        if self.classification_useful:
            # Compared to before, the paths here are of a self.num_path_pred \times len(t) shape
            Dist = self.calculate_distance(paths, t, domain)
    
            T = np.zeros((self.num_samples_path_pred, len(self.Behaviors)), float)
            for i_beh, beh in enumerate(self.Behaviors):
                for j in range(self.num_samples_path_pred):
                    class_change = np.where((Dist[beh][j, 1:] <= 0) & (Dist[beh][j, :-1] > 0))[0]
                    if len(class_change) == 0:
                        if Dist[beh][j, 0] <= 0:
                            T[j, i_beh] = 0.5 * self.dt
                        else:
                            T[j, i_beh] = t[-1] + self.dt
                    else:
                        ind = class_change[0]
                        T[j, i_beh] = ((t[ind + 1] * Dist[beh][j, ind] - t[ind] * Dist[beh][j, ind + 1]) /
                                       (Dist[beh][j, ind] - Dist[beh][j, ind + 1]))
            default = T.min(axis=-1) > t[-1]
            T[default, self.Behaviors == self.behavior_default] = t[-1]
        else:
            T = np.ones((self.num_samples_path_pred, len(self.Behaviors)), float) * t[-1]
        return T
    
    
    def path_add_pov_agent(self, Output_path_pred, Output_path_pred_probs, Pred_index, Domain, use_model=False):
        if self.pov_agent is None:
            return Output_path_pred

        # Add extrapolated pov agent to data
        Index_old = list(Output_path_pred.columns)
        if self.pov_agent in Index_old:
            Index_add = Index_old.copy()
            # Remove the pov_agent from Index old
            Index_old.remove(self.pov_agent)
        else:
            Index_add = [self.pov_agent] + list(Index_old)

        assert (Pred_index == Output_path_pred.index).all(), "The index of the Output_path_pred does not match the Pred_index."
        Output_path_pred_add = pd.DataFrame(np.empty((len(Output_path_pred), len(Index_add)), object), columns=Index_add, index = Pred_index)
        Output_path_pred_probs_add = pd.DataFrame(np.empty((len(Output_path_pred), len(Index_add)), object), columns=Index_add, index = Pred_index)

        # Get the diffent files used
        additions = self.Domain[['path_addition', 'data_addition']].iloc[Pred_index].to_numpy().sum(-1)
        unique_additions = np.unique(additions)
        Output_path = pd.DataFrame(np.empty((len(Pred_index), len(self.Agents)), object),
                                   index = Pred_index, columns = self.Agents)
        for addition in unique_additions:
            use_addition = additions == addition
            
            ind_needed = self.Domain.Index_saved.iloc[Pred_index[use_addition]]
            # Get the corresponding Output_path
            data_file = self.data_file[:-4] + addition + '_data.npy'
            Output_path_local = np.load(data_file, allow_pickle=True)[3]
            Output_path[use_addition] = Output_path_local.loc[ind_needed]

        for i_full in Pred_index:
            t = self.Output_T_pred[i_full]
            t_true = self.Output_T[i_full]
            # interpolate the values at the new set of points using numpy.interp()
            path_old = Output_path.loc[i_full,self.pov_agent]
            path_old = path_old[:len(t_true)]
            if np.array_equal(t, t_true):
                # Ensure to use only x and y
                path_new = path_old[...,:2]
            else:
                path_new = np.stack([np.interp(t, t_true, path_old[:,i]) for i in range(2)], axis = -1)

                # use the gradient to estimate values outside the bounds of xp
                dx = np.stack([np.gradient(path_old[:,i], t_true) for i in range(2)], axis = -1)

                # Extraplate the values
                later_time = t > t_true[-1]
                path_new[later_time] = path_old[[-1]] + (t[later_time] - t_true[-1])[:,np.newaxis] * dx[[-1]]

            # Add new results
            Output_path_pred_add.loc[i_full, Index_old] = Output_path_pred.loc[i_full, Index_old]
            Output_path_pred_add.loc[i_full, self.pov_agent] = np.repeat(path_new[np.newaxis], self.num_samples_path_pred, axis = 0)
            
            Output_path_pred_probs_add.loc[i_full, Index_old] = Output_path_pred_probs.loc[i_full, Index_old]
            Output_path_pred_probs_add.loc[i_full, self.pov_agent] = np.full(self.num_samples_path_pred, np.nan, np.float32)
            
        return Output_path_pred_add, Output_path_pred_probs_add

    def path_remove_pov_agent(self, Output_path_pred, Output_path_pred_probs, Pred_index, Domain):
        Index_retain = np.array(self.pov_agent != Output_path_pred.columns)
        Output_path_pred_remove = Output_path_pred.iloc[:, Index_retain]
        Output_path_pred_probs_remove = Output_path_pred_probs.iloc[:, Index_retain]
        return Output_path_pred_remove, Output_path_pred_probs_remove

    
    def _path_to_class_and_time(self, Output_path_pred, Pred_index, Output_T_pred, Domain):
        Output_A_pred = pd.DataFrame(np.zeros((len(Output_path_pred), len(self.Behaviors)), float),
                                             columns = self.Behaviors)
        Output_T_E_pred = pd.DataFrame(np.empty((len(Output_path_pred), len(self.Behaviors)), object),
                                        columns = self.Behaviors)

        for i_sample, i_full in enumerate(Pred_index):
            paths = Output_path_pred.iloc[i_sample]
            
            # Check if we need to increase the paths dim
            need_dim_increase = False
            for agent in paths.index:
                if not isinstance(paths[agent], float):
                    need_dim_increase = len(paths[agent].shape) == 2
                    break
            
            # Increase dimensions if needed
            if need_dim_increase:
                paths = self.increase_path_dim(paths)
            
            t = Output_T_pred[i_full]
            domain = Domain.iloc[i_full]
            T_class = self.path_to_class_and_time_sample(paths, t, domain)

            output_A = np.arange(len(self.Behaviors))[np.newaxis] == T_class.argmin(axis=-1)[:,np.newaxis]

            Output_A_pred.iloc[i_sample] = pd.Series(output_A.mean(axis=0), index=self.Behaviors)

            for i_beh, beh in enumerate(self.Behaviors):
                T_beh = T_class[output_A[:, i_beh], i_beh]
                if len(T_beh) > 1:
                    Output_T_E_pred.iloc[i_sample, i_beh] = np.quantile(T_beh, self.p_quantile)
                if len(T_beh) == 1:
                    Output_T_E_pred.iloc[i_sample, i_beh] = np.full(len(self.p_quantile), T_beh[0])
                else:
                    Output_T_E_pred.iloc[i_sample, i_beh] = np.full(len(self.p_quantile), np.nan)
        return Output_A_pred, Output_T_E_pred            
        
        
    
    
    def path_to_class_and_time(self, Output_path_pred, Pred_index, Domain):
        if self.classification_useful:
            Output_A_pred, Output_T_E_pred = self._path_to_class_and_time(Output_path_pred, Pred_index, self.Output_T_pred, Domain)
        else:
            Output_A_pred = pd.DataFrame(np.ones((len(Output_path_pred), len(self.Behaviors)), float),
                                         columns = self.Behaviors)
            Output_T_E_pred = pd.DataFrame(np.empty((len(Output_path_pred), len(self.Behaviors)), object),
                                           columns = self.Behaviors)
            
        return Output_A_pred, Output_T_E_pred

    def class_to_time(self, Output_A_pred, Pred_index, Domain):
        # Remove other prediction type
        if self.classification_useful:
            self.train_path_models()

            Output_T_E_pred = pd.DataFrame(np.empty((len(Output_A_pred), len(self.Behaviors)), object),
                                            columns = self.Behaviors)
            
            # Predict paths for all samples
            Paths_beh = {}
            for beh in self.Behaviors:
                # Get file indices
                file_indices = self.path_models[beh].data_set.Domain.file_index.iloc[Pred_index]
                
                # Prepare the output dataframe
                Agents = np.array(self.path_models[beh].data_set.Agents)
                Paths_beh[beh] =  pd.DataFrame(np.empty((len(Pred_index), len(Agents)), np.ndarray), columns = Agents, index = Pred_index)
                for file_index in np.unique(file_indices):
                    self.path_models[beh].data_set.reset_prediction_analysis()
                    self.path_models[beh].data_set._extract_original_trajectories(file_index = file_index)
                    
                    used_index = np.where(file_indices == file_index)[0]
                    Pred_index_used = Pred_index[used_index]
                    
                    Paths_beh[beh].loc[Pred_index_used] = self.path_models[beh].predict_actual(Pred_index_used)[1]

            for i_sample, i_full in enumerate(Pred_index):
                t = self.Output_T_pred[i_full]
                domain = Domain.iloc[i_full]
                for i_beh, beh in enumerate(self.Behaviors):
                    paths_beh = Paths_beh[beh].loc[i_full]
                    T_class_beh = self.path_to_class_and_time_sample(paths_beh, t, domain)
                    T_beh = T_class_beh[((T_class_beh[:, i_beh] == T_class_beh.min(axis=-1)) &
                                            (T_class_beh[:, i_beh] <= t[-1])), i_beh]

                    if len(T_beh) > 0:
                        Output_T_E_pred.iloc[i_sample, i_beh] = np.quantile(T_beh, self.p_quantile)
                    else:
                        Output_T_E_pred.iloc[i_sample, i_beh] = np.full(len(self.p_quantile), np.nan)
        else:
            Output_T_E_pred = pd.DataFrame(np.empty(Output_A_pred.shape, object), columns = self.Behaviors)
        return Output_T_E_pred

    def class_and_time_to_path(self, Output_A_pred, Output_T_E_pred, Pred_index, Domain):
        assert self.classification_useful, "For not useful datasets training classification models should be impossible."
        # check if this has already been performed

        self.train_path_models()
        Index = self.Agents
        Index_needed = np.array([name in self.needed_agents for name in Index])

        Output_path_pred = pd.DataFrame(np.empty((len(Output_A_pred), len(Index)), object), columns=Index)
        Output_path_pred_probs = pd.DataFrame(np.empty((len(Output_A_pred), len(Index)), object), columns=Index)

        # Transform probabilities into integer numbers that sum up to self.num_samples_path_pred
        Path_num = np.floor(self.num_samples_path_pred * Output_A_pred.to_numpy()).astype(int)
        Remaining_sum = self.num_samples_path_pred - Path_num.sum(axis=1)
        Index_sort = np.argsort((Path_num / self.num_samples_path_pred - Output_A_pred).to_numpy(), axis=1)
        Add_n, Add_beh = np.where(Remaining_sum[:, np.newaxis] > np.arange(len(self.Behaviors))[np.newaxis])

        Path_num[Add_n, Index_sort[Add_n, Add_beh]] += 1
        assert (Path_num.sum(1) == self.num_samples_path_pred).all()
        
        # Get the predicted behavior of transformation model
        Paths_beh = {}
        Paths_beh_probs = {}
        for beh in self.Behaviors:
            # Get file indices
            file_indices = self.path_models[beh].data_set.Domain.file_index.iloc[Pred_index]
            
            # Prepare the output dataframe
            Agents = np.array(self.path_models[beh].data_set.Agents)
            Paths_beh[beh] = pd.DataFrame(np.empty((len(Pred_index), len(Agents)), np.ndarray), columns = Agents, index = Pred_index)
            Paths_beh_probs[beh] = pd.DataFrame(np.empty((len(Pred_index), len(Agents)), np.ndarray), columns = Agents, index = Pred_index)
            for file_index in np.unique(file_indices):
                self.path_models[beh].data_set.reset_prediction_analysis()
                self.path_models[beh].data_set._extract_original_trajectories(file_index = file_index)
                
                used_index = np.where(file_indices == file_index)[0]
                Pred_index_used = Pred_index[used_index]
                
                path_model_output = self.path_models[beh].predict_actual(Pred_index_used)
                Paths_beh[beh].loc[Pred_index_used] = path_model_output[1]
                Paths_beh_probs[beh].loc[Pred_index_used] = path_model_output[2]


        # Go over all samples individually
        for i_sample, i_full in enumerate(Pred_index):
            path_num = Path_num[i_sample]
            t = self.Output_T_pred[i_full]
            domain = Domain.iloc[i_full]
            for j, index in enumerate(Output_path_pred.columns):
                if Index_needed[j]:
                    Output_path_pred.iloc[i_sample, j] = np.zeros((self.num_samples_path_pred, len(t), 2), float)
                    Output_path_pred_probs.iloc[i_sample, j] = np.zeros(self.num_samples_path_pred, float)

            output_T_E_pred = Output_T_E_pred.iloc[i_sample]
            ind_n_start = 0
            for i_beh, beh in enumerate(self.Behaviors):
                num_beh_paths = path_num[i_beh]
                if num_beh_paths >= 1:
                    ind_n_end = ind_n_start + num_beh_paths
                    paths_beh = Paths_beh[beh].loc[i_full]
                    paths_beh_probs = Paths_beh_probs[beh].loc[i_full]
                    T_class_beh = self.path_to_class_and_time_sample(paths_beh, t, domain)

                    Index_beh = np.where(((T_class_beh[:, i_beh] == T_class_beh.min(axis=-1)) &
                                          (T_class_beh[:, i_beh] <= t[-1])))[0]

                    if len(Index_beh) > 0:
                        T_beh = T_class_beh[Index_beh, i_beh]

                        # Sample according to provided distribution
                        p_quantile_expanded = np.concatenate(([0.0], self.p_quantile, [1.0]), axis=0)

                        if np.all(output_T_E_pred[beh] <= t[-1]):
                            T_pred_beh_expanded = np.concatenate(([0.0], output_T_E_pred[beh], 
                                                                  [max(t[-1], output_T_E_pred[beh].max()) + 1e-4]), axis=0)
                            T_sampled = np.interp(np.random.rand(num_beh_paths), p_quantile_expanded, T_pred_beh_expanded)

                            # Get closest value
                            Distance = np.abs(np.subtract.outer(T_sampled, T_beh))
                            Rand_ind = np.random.rand(*Distance.shape).argsort(-1)
                            Dist_rand = Distance[np.tile(np.arange(len(T_sampled))[:, np.newaxis], 
                                                         (1, len(T_beh))), Rand_ind]

                            Index_sampled = Rand_ind[np.arange(len(T_sampled)), np.argmin(Dist_rand, axis=1)]
                            assert (
                                Distance.min(-1) == Distance[np.arange(len(T_sampled)), Index_sampled]).all()
                        else:
                            # This is the case where no time prediction is made, so simply select latest cases
                            if len(T_beh) >= num_beh_paths:
                                Index_sampled = np.argpartition(T_beh, -num_beh_paths)[-num_beh_paths:]
                            else:
                                Index_sampled = (np.random.rand(num_beh_paths) * len(T_beh) * (1 - 1e-5)).astype(int)
                        Index_used = Index_beh[Index_sampled]
                    else:
                        Index_used = np.argpartition(T_class_beh.min(axis=-1)-T_class_beh[:, i_beh], -num_beh_paths)[-num_beh_paths:]

                    for j, index in enumerate(Output_path_pred.columns):
                        if Index_needed[j]:
                            Output_path_pred.iloc[i_sample, j][ind_n_start:ind_n_end] = paths_beh[index][Index_used]
                            Output_path_pred_probs.iloc[i_sample, j][ind_n_start:ind_n_end] = paths_beh_probs[index][Index_used]

                    # Reset ind_start for next possible behavior
                    ind_n_start = ind_n_end
        
        return Output_path_pred, Output_path_pred_probs

    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                         Data-set dependend functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################

    # %% Define data set dependent functions

    def get_name(self = None):
        r'''
        Provides a dictionary with the different names of the dataset:
            
        names = {'print': 'printable_name', 'file': 'files_name', 'latex': r'latex_name'}.
        
        Returns
        -------
        names : dict
            The first key of names ('print')  will be primarily used to refer to the dataset in console outputs. 
            
            The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
            (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. 
            
            The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
            latex commands - such as using '$$' for math notation.
        
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

    def future_input(self = None):
        r'''
        return True: The future data of the pov agent can be used as input.
        This is especially feasible if the ego agent was controlled by an algorithm in a simulation,
        making the recorded future data similar to the ego agent's planned path at each point in time.
        
        return False: This usage of future ego agent's trajectories as model input is prevented. This is especially advisable
        if the behavior of the vehicle might include too many clues for a prediction model to use.
        
        Returns
        -------
        future_input_decision : bool
        
        '''
        raise AttributeError("Has to be overridden in actual data-set class.")
        
    def includes_images(self = None):
        r'''
        If True, then image data can be returned (if true, .image_id has to be a column of 
        **self.Domain_old** to indicate which of the saved images is linked to which sample).
        If False, then no image data is provided, and models have to content without them.
        
        Returns
        -------
        image_decision : bool
        
        '''
        raise AttributeError("Has to be overridden in actual data-set class.")

    def includes_sceneGraphs(self = None):
        r'''
        If True, then scene graph data can be returned (if true, .graph_id has to be a column of 
        **self.Domain_old** to indicate which of the saved graphs is linked to which sample).
        If False, then no scene graph data is provided, and models have to content without them.
        
        Returns
        -------
        sceneGraph_decision : bool
        
        '''
        raise AttributeError("Has to be overridden in actual data-set class.")
        
    def set_scenario(self):
        r'''
        Sets the scenario <scenario_class> to which this dataset belongs, using an imported class.
        
        It should contain the command:
            self.scenario = <scenario_class>()
            
        Furthermore, if general information about the dataset is needed for later steps - 
        and not only the extraction of the data from its original recorded form - those 
        can be defined here. For example, certain distance measurements such as the radius 
        of a roundabout might be needed here.
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

    def create_path_samples(self):
        r'''
        Loads the original trajectory data from wherever it is saved.
        Then, this function has to extract for each potential test case in the data set 
        some required information. This information has to be collected in the following attributes, 
        which do not have to be returned, but only defined in this function:
    
        **self.Path** : pandas.DataFrame          
              A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. 
              Here, each row :math:`i` represents one recorded sample, while each column includes the 
              trajectory of an agent as a numpy array of shape :math:`\{\vert T_i \vert{\times} N_{data}\}`. 
              Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*. 
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
            that gives us the scaling of the images in the unit :math:`m /` Px. 


        It might also be possible that the selected dataset can provide scene graphs. In this case, it is
        paramount that **self.Domain_old** contains a column named 'graph_id', so that scene graphs can
        be assigned to each sample with only having to save one scene graph for each location instead for
        each sample:
    
        **self.SceneGraphs** : pandas.DataFrame 
            A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} 15\}`.
            The columns (order is not relevant) correspond to the following features:
    
            num_nodes         - number of nodes in the scene graph.
    
            lane_idcs         - indices of the lane segments in the scene graph; array of length :math:`num_{nodes}`
                                with *lane_idcs.max()* :math:`= num_{lanes} - 1`.
    
            pre_pairs         - array with shape :math:`\{num_{lane pre} {\times} 2\}` lane_idcs pairs where the
                                first value of the pair is the source lane index and the second value is source's
                                predecessor lane index.
    
            suc_pairs         - array with shape :math:`\{num_{lane suc} {\times} 2\}` lane_idcs pairs where the
                                first value of the pair is the source lane index and the second value is source's
                                successor lane index.
    
            left_pairs        - array with shape :math:`\{num_{lane left} {\times} 2\}` lane_idcs pairs where the
                                first value of the pair is the source lane index and the second value is source's
                                left neighbor lane index.
    
            right_pairs       - array with shape :math:`\{num_{lane right} {\times} 2\}` lane_idcs pairs where the
                                first value of the pair is the source lane index and the second value is source's
                                right neighbor lane index.
    
            left_boundaries   - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                                :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                                of points needed to describe the left boundary in travel direction of the current lane.
                                Here, :math:`num_{nodes,l} = ` *(lane_idcs == l).sum()*. 
                                     
            right_boundaries  - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                                :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                                of points needed to describe the right boundary in travel direction of the current lane.
    
            centerlines       - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                                :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                                of points needed to describe the middle between the left and right boundary in travel
                                direction of the current lane.
        
            lane_type         - an array with length :math:`num_{lanes}`, whose elements are tuples with the length :math:`2`,
                                where the first element is a string that is either *'VEHILCE'*, *'BIKE'*, '*PEDESTRIAN*', 
                                or '*BUS*', and the second entry is a boolean, which is true if the lane segment is part of 
                                an intersection.

            pre               - predecessor nodes of each node in the scene graph;
                                list of dictionaries where the length of the list is equal to the number of scales for the neighbor
                                dilation as per the implementation in LaneGCN. 
                                Each dictionary contains the keys 'u' and 'v', where 'u' is the *node index* of the source node and
                                'v' is the index of the target node giving edges pointing from a given source node 'u' to its
                                predecessor.
    
            suc               - successor nodes of each node in the scene graph;
                                list of dictionaries where the length of the list is equal to the number of scales for the neighbor
                                dilation as per the implementation in LaneGCN. 
                                Each dictionary contains the keys 'u' and 'v', where 'u' is the *node index* of the source node and
                                'v' is the index of the target node giving edges pointing from a given source node 'u' to its
                                successor.
    
            left              - left neighbor nodes of each node in the scene graph;
                                list containing a dictionary with the keys 'u' and 'v', where 'u' is the *node index* of the source 
                                node and 'v' is the index of the target node giving edges pointing from a given source node 'u' to 
                                its left neighbor.
    
            right             - right neighbor nodes of each node in the scene graph;
                                list containing a dictionary with the keys 'u' and 'v', where 'u' is the *node index* of the source 
                                node and 'v' is the index of the target node giving edges pointing from a given source node 'u' to 
                                its right neighbor.

        It is paramount that the indices of this DataFrame are equivalent to the unique values found in 
        **self.Domain_old**['graph_id']. All of the positions in *left_boundaries*, *right_boundaries*, and *centerlines* are given
        in the same original coordinate system used for the trajectory data in **self.Path**. Any transformations such as alignment
        with a desired axis should be performed within the model itself.
    
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')
    
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
        raise AttributeError('Has to be overridden in actual data-set class.')

    def calculate_distance(self, path, t, domain):
        r'''
        If the chosen scenario contains a number of possible behaviors, as which recorded or
        predicted trajectories might be classified, this function calculates the abridged distance of the 
        relevant agents in a scenario toward fulfilling each of the possible classification criteria. 
        If the classification criterium is not yet fulfilled, those distances are positive, while them being negative 
        means that a certain behavior has occurred.
        
        This function extracts these distances for one specific sample.

        Parameters
        ----------
        path : pandas.Series
            A pandas series with :math:`(N_{agents})` entries,
            where each entry is itself a numpy array of shape :math:`\{N_{preds} \times |t| \times N_{data}\}`.
            Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*, or 2, if only
            predicted trajectories are evaluated.
            The indices should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        t : numpy.ndarray
            A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
            at which the positions in **path** were recorded.
        domain : pandas.Series
            A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
            sample. Its entries contain at least all the columns of **self.Domain_old**. 

        Returns
        -------
        Dist : pandas.Series
            This is a series with :math:`N_{classes}` entries.
            For each column, it returns an array of shape :math:`\{N_{preds} \times |t|\}` with the distance to the classification marker.
            The column names should correspond to the attribute self.Behaviors = list(self.scenario.give_classifications().keys()). 
            How those distances are defined depends on the scenario and behavior.
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

    def evaluate_scenario(self, path, D_class, domain):
        r'''
        It might be that the given scenario requires all agents to be in certain positions so that
        it can be considered that the scenario is indeed there. This function makes that evaluation.

        This function tests this for one specific sample.

        Parameters
        ----------
        path : pandas.Series
            A pandas series with :math:`(N_{agents})` entries,
            where each entry is itself a numpy array of lenght :math:`\{|t| \times N_{data}\}`.
            Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*.
            The indices should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        D_class : pandas.Series
            This is a series with :math:`N_{classes}` entries.
            For each column, it returns an array of lenght :math:`|t|` with the distance to the classification marker.
            The column names should correspond to the attribute self.Behaviors = list(self.scenario.give_classifications().keys()). 
            How those distances are defined depends on the scenario and behavior.
        domain : pandas.Series
            A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
            sample. Its entries contain at least all the columns of **self.Domain_old**. 

        Returns
        -------
        in_position : numpy.ndarray
            This is a :math:`|t|` dimensional boolean array, which is true if all agents are
            in a position where the scenario is valid.
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

    def calculate_additional_distances(self, path, t, domain):
        r'''
        Some models cannot deal with trajectory data and instead are constrained to quasi-one-dimensional
        data. While here the main data are the distances to the classification created in self.calculate_distance(),
        this might be incomplete to fully describe the current situation. Consequently, it might be necessary
        to extract further characteristic distances.

        This function extracts these distances for one specific sample.

        Parameters
        ----------
        path : pandas.Series
            A pandas series with :math:`(N_{agents})` entries,
            where each entry is itself a numpy array of shape :math:`\{|t| \times N_{data}\}`.
            Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*.
            The indices should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        t : numpy.ndarray
            A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
            at which the positions in **path** were recorded.
        domain : pandas.Series
            A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
            sample. Its entries contain at least all the columns of **self.Domain_old**. 

        Returns
        -------
        Dist_other : pandas.Series
            This is a :math:`N_{other dist}` dimensional Series.
            For each column, it returns an array of lenght :math:`|t|` with the distance to the classification marker.

            These columns should contain the minimum required distances set in self.scenario.can_provide_general_input().
            If *self.scenario.can_provide_general_input() in [None, []]*, one should return None instead.
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

    def fill_empty_path(self, path, t, domain, agent_types, size = None):
        r'''
        After extracting the trajectories of a sample at the given input and output timesteps, it might be possible
        that an agent's trajectory is only partially recorded over this timespan, resulting in the position values being np.nan
        at those missing time points. The main cause here is likely that the agent is outside the area over which its position 
        could be recorded. 

        However, some models might be unable to deal with such missing data. Consequently, it is required to fill those missing 
        positions with extrapolated data. 

        Additionally, it might be possible that **path** does not contain all the agents which were present during 
        the *input* timesteps. As those might still be influencing the future behavior of the agents already included in 
        **path**, they can be added here. Consequntly, math:`N_{agents, full} \geq N_{agents}` will be the case.
        
        Parameters
        ----------
        path : pandas.Series
            A pandas series with :math:`(N_{agents})` entries,
            where each entry is itself a numpy array of shape :math:`\{|t| \times N_{data}\}`.
            Here, :math:`N_{data}` is the length of the list returned by *self.path_data_info()*.
            The indices should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        t : numpy.ndarray
            A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
            at which the positions in **path** were recorded.
        domain : pandas.Series
            A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
            sample. Its entries contain at least all the columns of **self.Domain_old**. 
        agent_types : pandas.Series 
            A pandas series with :math:`(N_{agents})` entries, that records the type of the agents for the considered
            sample by using a single letter string. The indices should correspond to the columns in **self.Type_old** 
            created in self.create_path_samples() and should include at least the relevant agents described in 
            self.create_sample_paths. Consequently, the column names are identical to those of **path**.
        size : pandas.Series
            A pandas series with :math:`(N_{agents})` entries, that records the size of the agents for the considered
            sample in a numpy array (np.array([length, width])). The indices should correspond to the columns in **self.Type_old** 
            created in self.create_path_samples() and should include at least the relevant agents described in 
            self.create_sample_paths. Consequently, the column names are identical to those of **path**.
            If the dataset does not record sizes, this will be *None*.


        Returns
        -------
        path_full : pandas.Series
            A pandas series with :math:`(N_{agents, full})` entries,
            where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
            All columns of **path** should be included here. For those agents where trajectories are recorded, those trajectories 
            can also no longer contain np.nan as a position value.
        agent_types_full : pandas.Series 
            A pandas series with :math:`(N_{agents, full})` entries, that records the type of the agents for the considered
            sample. The indices should correspond to the columns in **path_full** and include all columns of **agent_types**.
        size_full : pandas.Series (optional)
            A pandas series with :math:`(N_{agents, full})` entries, that records the size of the agents for the considered
            sample in a numpy array (np.array([length, width])). The indices should correspond to the columns in **path_full** 
            and include all columns of **size**. Returning this is optional. if it is not returned, defualt agent sizes are assumed
            for added agents.

        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

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
        raise AttributeError("Has to be overridden in actual data-set class.")

        
