import pandas as pd
import numpy as np
import os
import torch
import psutil
from data_interface import data_interface
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
        
        
    def check_path_samples(self, Path, Type_old, T, Domain_old, num_samples):
        # Check if the rigth path information if provided
        path_info = self.path_data_info()
        if not isinstance(path_info, list):
            raise TypeError("self.path_data_info() should return a list.")
        else:
            for info in path_info:
                if not isinstance(info, str):
                    raise TypeError("Elements of self.path_data_info() should be strings.")
        
        # Check if x and y are included
        if path_info[0] != 'x':
            raise AttributeError("'x' should be included as the first element self.path_data_info().")
        if path_info[1] != 'y': 
            raise AttributeError("'y' should be included as the second element self.path_data_info().")

        # check some of the aspect to see if pre_process worked
        if not isinstance(Path, pd.core.frame.DataFrame):
            raise TypeError("Paths should be saved in a pandas data frame")
        if len(Path) != num_samples:
            raise TypeError("Path does not have right number of sampels")

        # check some of the aspect to see if pre_process worked
        if not isinstance(Type_old, pd.core.frame.DataFrame):
            raise TypeError("Agent Types should be saved in a pandas data frame")
        if len(Type_old) != num_samples:
            raise TypeError("Type dataframe does not have right number of sampels")
    
        if not isinstance(T, np.ndarray):
            raise TypeError("Time points should be saved in a numpy array")
        if len(T) != num_samples:
            raise TypeError("Time points des not have right number of sampels")

        if not isinstance(Domain_old,  pd.core.frame.DataFrame):
            raise TypeError("Domain information should be saved in a Pandas Dataframe.")
        
        if len(Domain_old) != num_samples:
            raise TypeError("Domain information should have correct number of sampels")
        
        if self.includes_images():
            if not 'image_id' in Domain_old.columns:
                raise AttributeError('Image identification is missing')
        
        # Check final paths
        path_names = Path.columns
        
        if (path_names != Type_old.columns).any():
            raise TypeError("Agent Paths and Types need to have the same columns.")
        
        for needed_agent in self.needed_agents:
            if not needed_agent in path_names:
                raise AttributeError("Agent " + needed_agent + " must be included in the paths")
        
        for i in range(num_samples):
            # check if time input consists out of tuples
            if not isinstance(T[i], np.ndarray):
                raise TypeError("A time point samples is expected to be a np.ndarray.")

            test_length = len(T[i])
            for j, agent in enumerate(path_names):
                # check if time input consists out of tuples
                agent_path = Path.iloc[i, j]
                agent_type = Type_old.iloc[i, j]
                # For needed agents, nan is not admissible
                if agent in self.needed_agents:
                    if not isinstance(agent_path, np.ndarray):
                        raise TypeError("Path is expected to be consisting of np.ndarrays.")
                else:
                    if not isinstance(agent_path, np.ndarray):
                        if str(agent_path) != 'nan':
                            raise TypeError("Path is expected to be consisting of np.ndarrays.")
                        
                        if str(agent_type) != 'nan':
                            raise TypeError("If no path is given, there should be no agent type.")
                            
                
                # if the agent exists in this sample, adjust this
                if isinstance(agent_path, np.ndarray):
                    if not len(agent_path.shape) == 2:
                        raise TypeError("Path is expected to be consisting of np.ndarrays with two dimension.")
                    if not agent_path.shape[1] == len(path_info):
                        raise TypeError("Path is expected to be consisting of np.ndarrays of shape (n x len(self.path_data_info())).")
                        
                    # test if input tuples have right length
                    if test_length != len(agent_path):
                        raise TypeError("Path sample does not have a matching number of timesteps.")
                        
                    if str(agent_type) == 'nan':
                        raise ValueError("For a given path, the agent type must not be nan.")
        
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
            self.check_path_samples(Path_check, Type_old_check, T_check, Domain_old_check, num_samples_check)
            
            # Save the results
            os.makedirs(os.path.dirname(file_path_save), exist_ok=True)
            test_data = np.array([Path_check, Type_old_check, T_check, Domain_old_check, num_samples_check], object)
            np.save(file_path_save, test_data)
            
            # Reset the data to empty lists
            self.Path = []
            self.Type_old = []
            self.T = []
            self.Domain_old = []
            
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
                        sceneGrap_columns = ['ctrs', 'num_nodes', 'feats', 'centerlines', 'left_boundaries', 'right_boundaries', 'pre', 'suc', 
                                             'lane_idcs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'left', 'right']   
                        self.SceneGraphs = pd.DataFrame(np.zeros((0, len(sceneGrap_columns)), object), index = [], columns = sceneGrap_columns)

        if last:
            self.saved_last_orig_paths = True
                
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
            [_, _, _, _, num_samples_file] = np.load(file_path, allow_pickle=True)
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
                    [self.Path,
                    self.Type_old,
                    self.T,
                    self.Domain_old,
                    self.num_samples] = np.load(test_file, allow_pickle=True)
            else:
                if not all([hasattr(self, attr) for attr in ['create_path_samples']]):
                    raise AttributeError("The raw data cannot be loaded.")
                
                # Get the currently available RAM space
                self.available_memory_creation = self.total_memory - get_used_memory()

                self.create_path_samples()
                # Check if the las file allready exists
                if os.path.isfile(test_file):
                    self.number_original_path_files = self.get_number_of_original_path_files()
                    # If there is only one file, load the data
                    if self.number_original_path_files == 1:
                        [self.Path,
                        self.Type_old,
                        self.T,
                        self.Domain_old,
                        self.num_samples] = np.load(test_file, allow_pickle=True)
                    
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
                    
                    # Validate the data                    
                    self.check_path_samples(self.Path, self.Type_old, self.T, self.Domain_old, self.num_samples)
                
                    # save the results
                    os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                    
                    test_data = np.array([self.Path,
                                        self.Type_old,
                                        self.T,
                                        self.Domain_old,
                                        self.num_samples], object)
                    
                    np.save(test_file, test_data)
                
                # Check if data needs to be saved:
                if not hasattr(self, 'map_split_save'):
                    self.map_split_save = False
                
                if not self.map_split_save:
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

            T = np.zeros(len(self.Behaviors), float)
            T_D = np.empty(len(self.Behaviors), object)
            for i, beh in enumerate(self.Behaviors):
                Dist_dt = (Dist[beh][n_dt:] - Dist[beh]
                           [:-n_dt]) / (t[n_dt:] - t[:-n_dt])
                Dist_dt = np.concatenate(
                    (np.tile(Dist_dt[[0]], (n_dt)), Dist_dt), axis=0)

                T_D[i] = Dist[beh] / np.maximum(- Dist_dt, 1e-7)
                if not in_position.any():
                    T[i] = t[-1] + 1
                else:
                    Dt_in_pos = T_D[i][in_position]

                    time_change = np.where(
                        (Dt_in_pos[1:] <= 0) & (Dt_in_pos[:-1] > 0))[0]
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
            
            Dist_oth = self.calculate_additional_distances(path, t, domain)
            for index in self.extra_input:
                assert index in Dist_oth.index, "A required extracted input is missing."

            Pred = pd.concat([Dist, Dist_oth])
            return Pred
        else:
            return None

    def extract_time_points(self, Path, T, Domain_old, num_samples, path_file):
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

            for i_sample in range(num_samples):
                if np.mod(i_sample, 100) == 0:
                    print('path ' + str(i_sample).rjust(len(str(num_samples))) + '/{} divided'.format(num_samples))

                path = Path.iloc[i_sample]
                domain = Domain_old.iloc[i_sample]
                t = np.array(T[i_sample])

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
                        [Path_loaded, _,
                        T_loaded,
                        Domain_old_loaded,
                        num_samples_loaded] = np.load(path_file, allow_pickle=True)
                
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
                        local_t_crit] = self.extract_time_points(Path_loaded, T_loaded, Domain_old_loaded, num_samples_loaded, path_file)
                

                
                
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
            num_timesteps_real = min(99, num_timesteps[0])
            num_timesteps_need = max(num_timesteps_real, min(99, num_timesteps[1]))

        # If only one value is given, assume that the required number of steps is identical
        elif isinstance(num_timesteps, int):
            num_timesteps_real = min(99, num_timesteps)  # Refers to sctual input data
            num_timesteps_need = min(99, num_timesteps)  # Restrictions on t0
            
        return int(num_timesteps_real), int(num_timesteps_need)
    
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
        num_files = len(domain_files)
        return domain_files, num_files
    
    
    def get_data_from_orig_path(self, Path, Type_old, T, Domain_old, num_samples, path_file, path_file_adjust):
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
            local_t_crit] = self.extract_time_points(Path, T, Domain_old, num_samples, path_file)

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
            min_num_agents = len(Path.columns)
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

        # Go through samples
        local_num_samples = len(local_id)

        predicted_saving_length = 0

        for i in range(local_num_samples):
            # print progress
            if np.mod(i, 1) == 0:
                print('path ' + str(i + 1).rjust(len(str(local_num_samples))) +
                    '/{}: divide'.format(local_num_samples))

            # load extracted data
            i_path = local_id[i]
            path = Path.iloc[i_path]
            t = local_t[i]

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
                if self.classification_useful:
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
                helper_T_appr = np.concatenate((input_T + 1e-5, output_T - 1e-5))
                
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
                            
                    else:
                        helper_path[agent] = np.nan
                        agent_types[agent] = float('nan')
                        
                    # check if needed agents have reuqired input and output
                    if agent in self.needed_agents:
                        if np.isnan(helper_path[agent][:,:2]).any():
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
                            distances = np.linalg.norm(helper_path[agent][1:,:2] - helper_path[agent][:-1,:2], axis=-1)
                            if np.all(distances < 1e-2):
                                available_pos[self.num_timesteps_in_real - 1] = False
                            
                        recorded_positions[agent] = available_pos
                    else:
                        recorded_positions[agent] = np.nan
                
                # Combine input and output data
                helper_T = np.concatenate([input_T, output_T])
                
                # complete partially available paths
                helper_path, agent_types = self.fill_empty_path(helper_path, helper_T, domain, agent_types)
                
                if self.max_num_agents is not None:
                    helper_path = helper_path.iloc[:max_num_agent_local]
                    agent_types = agent_types.iloc[:max_num_agent_local]
                    
                
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
                        
                        # Split by input and output, however, only include positions in the output
                        input_path[agent]  = helper_path[agent][:self.num_timesteps_in_real].astype(np.float32)
                        output_path[agent] = helper_path[agent][self.num_timesteps_in_real:].astype(np.float32)

                        # Guarantee that the input path does contain only nan value
                        if not (ind_start < self.num_timesteps_in_real - 1 and self.num_timesteps_in_real <= ind_last):
                            input_path[agent]         = np.nan
                            output_path[agent]        = np.nan
                            recorded_positions[agent] = np.nan
                            agent_types[agent] = float('nan')
                        
                    else:
                        input_path[agent]         = np.nan
                        output_path[agent]        = np.nan
                        recorded_positions[agent] = np.nan
                        agent_types[agent]        = float('nan')
                        
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
            files = os.path.listdir(directory)

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

            self.num_behaviors_local = np.zeros(len(self.Behaviors), int)
            
            # Ensure that dataframes with agent columns have the same order
            Agents = self.Input_path.columns.to_list()
            self.Input_path  = self.Input_path[Agents]
            self.Output_path = self.Output_path[Agents]
            self.Type        = self.Type[Agents]
            self.Recorded    = self.Recorded[Agents]

            # Ensure that indices of dataframes are the same
            self.Input_path = self.Input_path.reset_index(drop = True)
            self.Input_prediction.index = self.Input_path.index
            self.Output_path.index      = self.Input_path.index
            self.Output_A.index         = self.Input_path.index
            self.Type.index             = self.Input_path.index
            self.Recorded.index         = self.Input_path.index
            self.Domain.index           = self.Input_path.index
            
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
            save_agent  = np.array([self.Type, self.Recorded, 0], object)
            
            
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
                    T_loaded = self.T
                    Domain_old_loaded = self.Domain_old
                    num_samples_loaded = self.num_samples
            
                else:
                    # Load the data
                    [Path_loaded,
                    Type_old_loaded,
                    T_loaded,
                    Domain_old_loaded,
                    num_samples_loaded] = np.load(path_file, allow_pickle=True)

                # Get the currently available RAM space
                self.available_memory_data_extraction = self.total_memory - get_used_memory()

                # Adjust base data file name accordingly
                self.get_data_from_orig_path(Path_loaded, Type_old_loaded, T_loaded, Domain_old_loaded, num_samples_loaded, path_file, path_file_adjust)
                
        
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
            [self.Type, self.Recorded, _] = np.load(agent_file, allow_pickle=True)
        
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
        # check if dataset is useful
        if len(self.Domain) < 100:
            return "there are not enough samples for a reasonable training process."

        if self.classification_useful and np.sort(self.num_behaviors_out)[-2] < 10:
            return "the dataset is too unbalanced for a reasonable training process."

        return None

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
            
        
        
        
      
    def return_batch_sceneGraphs(self, domain, SceneGraphs, Graphs_Index, print_progress=False): # TODO
        if self.includes_sceneGraphs():
            if print_progress:    
                print('')
                print('Load needed scene graphs:', flush = True)
            
            # Find the gpu
            if not torch.cuda.is_available():
                device = torch.device('cpu')
                raise TypeError("GPU cannot be detected")
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
            
            n = 250
            
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
                    for i in range(0, len(loc_indices), n):
                        torch.cuda.empty_cache()
                        Index_local = np.arange(i, min(i + n, len(loc_indices)))
                        Index = path_indices[loc_indices[Index_local]]
                        
                        if print_progress:
                            print('retrieving graphs ' + str(graph_num + 1) + ' to ' + str(graph_num + len(Index)) + 
                                ' of ' + str(len(domain)) + ' total', flush = True)
                        graph_num = graph_num + len(Index)
                            
                        torch.cuda.empty_cache()
                        SceneGraphs[Graphs_Index[Index]] = [loc_Graph]*len(Index)
        
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

                Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'class_and_time':
                [Pred_index, Output_A_pred, Output_T_E_pred] = output
                Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'path_all_wi_pov':
                [Pred_index, Output_path_pred] = output
            else:
                raise AttributeError(
                    "This type of output produced by the model is not implemented")

            Output_path_pred = self.path_remove_pov_agent(Output_path_pred, Pred_index, self.Domain)
            output_trans = [Pred_index, Output_path_pred]

        elif metric_pred_type == 'path_all_wi_pov':
            if model_pred_type == 'class':
                [Pred_index, Output_A_pred] = output
                Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain)

                Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'class_and_time':
                [Pred_index, Output_A_pred, Output_T_E_pred] = output
                Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index, self.Domain)
            elif model_pred_type == 'path_all_wo_pov':
                [Pred_index, Output_path_pred] = output
                Output_path_pred = self.path_add_pov_agent(Output_path_pred, Pred_index, self.Domain)
            else:
                raise AttributeError(
                    "This type of output produced by the model is not implemented")
            output_trans = [Pred_index, Output_path_pred]

        # If the metric requires class predictions
        elif metric_pred_type == 'class':
            if model_pred_type == 'path_all_wo_pov':
                [Pred_index, Output_path_pred] = output
                Output_path_pred = self.path_add_pov_agent(Output_path_pred, Pred_index, self.Domain)

                [Output_A_pred, _] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Domain)

            elif model_pred_type == 'path_all_wi_pov':
                [Pred_index, Output_path_pred] = output
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
                [Pred_index, Output_path_pred] = output
                Output_path_pred = self.path_add_pov_agent(Output_path_pred, Pred_index, self.Domain)
                [Output_A_pred, Output_T_E_pred] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Domain)

            elif model_pred_type == 'path_all_wi_pov':
                [Pred_index, Output_path_pred] = output
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
    
    
    def path_add_pov_agent(self, Output_path_pred, Pred_index, Domain, use_model=False):
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
        Output_path_pred_add = pd.DataFrame(np.empty((len(Output_path_pred), len(Index_add)), object),
                                            columns=Index_add, index = Pred_index)

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
        
        return Output_path_pred_add

    def path_remove_pov_agent(self, Output_path_pred, Pred_index, Domain):
        Index_retain = np.array(self.pov_agent != Output_path_pred.columns)
        Output_path_pred_remove = Output_path_pred.iloc[:, Index_retain]
        return Output_path_pred_remove

    
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

        Output_path_pred = pd.DataFrame(np.empty((len(Output_A_pred), len(Index)), object),
                                        columns=Index)

        # Transform probabilities into integer numbers that sum up to self.num_samples_path_pred
        Path_num = np.floor(self.num_samples_path_pred * Output_A_pred.to_numpy()).astype(int)
        Remaining_sum = self.num_samples_path_pred - Path_num.sum(axis=1)
        Index_sort = np.argsort((Path_num / self.num_samples_path_pred - Output_A_pred).to_numpy(), axis=1)
        Add_n, Add_beh = np.where(Remaining_sum[:, np.newaxis] > np.arange(len(self.Behaviors))[np.newaxis])

        Path_num[Add_n, Index_sort[Add_n, Add_beh]] += 1
        assert (Path_num.sum(1) == self.num_samples_path_pred).all()
        
        # Get the predicted behavior of transformation model
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


        # Go over all samples individually
        for i_sample, i_full in enumerate(Pred_index):
            path_num = Path_num[i_sample]
            t = self.Output_T_pred[i_full]
            domain = Domain.iloc[i_full]
            for j, index in enumerate(Output_path_pred.columns):
                if Index_needed[j]:
                    Output_path_pred.iloc[i_sample, j] = np.zeros((self.num_samples_path_pred, len(t), 2), float)

            output_T_E_pred = Output_T_E_pred.iloc[i_sample]
            ind_n_start = 0
            for i_beh, beh in enumerate(self.Behaviors):
                num_beh_paths = path_num[i_beh]
                if num_beh_paths >= 1:
                    ind_n_end = ind_n_start + num_beh_paths
                    paths_beh = Paths_beh[beh].loc[i_full]
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

                    # Reset ind_start for next possible behavior
                    ind_n_start = ind_n_end
        
        return Output_path_pred

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
            The columns correspond to the following features:
            1.  ctrs              - locations between the centerline segments in the global coordinate system
                                    :math:`\{num_{nodes} {\times} 2\}`
            2.  num_nodes         - number of nodes in the scene graph
            3.  feats             - centerline segment offsets :math:`\{num_{nodes} {\times} 2\}`
            4.  centerlines       - midpoint between the left and right boundary of the section of a lane segment in
                                    the global coordinate system; 
                                    array of arrays that represent the individual lane segments 
                                    :math:`\{num_{nodes} {\times} 2\}`
            5.  left_boundaries   - left boundary of the section of a lane segment in the global coordinate system; 
                                    array of arrays that represent the individual lane segments
                                    :math:`\{num_{nodes} {\times} 2\}`
            6.  right_boundaries  - right boundary of the section of a lane segment in the global coordinate system; 
                                    array of arrays that represent the individual lane segments
                                    :math:`\{num_{nodes} {\times} 2\}`
            7.  pre               - predecessor nodes of each node in the scene graph;
                                    list of dictionaries where the length of the list is equal to the number of scales
                                    for the neighbor dilation as per the implementation in LaneGCN;
                                    each dictionary contains the keys 'u' and 'v' where 'u' is the *node index* of the 
                                    source node and 'v' is the index of the target node giving edges pointing from a 
                                    given source node 'u' to its predessesor
            8.  suc               - successor nodes of each node in the scene graph;
                                    list of dictionaries where the length of the list is equal to the number of scales
                                    for the neighbor dilation as per the implementation in LaneGCN;
                                    each dictionary contains the keys 'u' and 'v' where 'u' is the *node index* of the
                                    source node and 'v' is the index of the target node giving edges pointing from a
                                    given source node 'u' to its successor
            9.  lane_idcs         - indices of the lane segments in the scene graph; array of length :math:`num_{nodes}`
            10. pre_pairs         - array of lane_idcs pairs where the first value of the pair is the source *lane index*
                                    and the second value is source's predecessor lane index
            11. suc_pairs         - array of lane_idcs pairs where the first value of the pair is the source *lane index*
                                    and the second value is source's successor lane index
            12. left_pairs        - array of lane_idcs pairs where the first value of the pair is the source *lane index*
                                    and the second value is source's left neighbour lane index
            13. right_pairs       - array of lane_idcs pairs where the first value of the pair is the source *lane index*
                                    and the second value is source's right neighbour lane index
            14. left              - left neighbor nodes of each node in the scene graph;
                                    array containing a dictionary with the keys 'u' and 'v' where 'u' is the *node index* 
                                    of the source node and 'v' is the index of the target node giving edges pointing from a 
                                    given source node 'u' to its left neighbor
            15. right             - right neighbor nodes of each node in the scene graph;
                                    array containing a dictionary with the keys 'u' and 'v' where 'u' is the *node index* 
                                    of the source node and 'v' is the index of the target node giving edges pointing from a 
                                    given source node 'u' to its right neighbor

            It is paramount that the indices of this DataFrame are equivalent to the unique values found in 
            **self.Domain_old**['graph_id']. All of the information is represented within the original coordinate system.
            Any transformations such as alignment with a desired axis should be performed within the model itself.
    
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
            If self.can_provide_general_input() == False, one should return None instead.
        '''
        raise AttributeError('Has to be overridden in actual data-set class.')

    def fill_empty_path(self, path, t, domain, agent_types):
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
            sample. The indices should correspond to the columns in **self.Type_old** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths. Consequently, the 
            column names are identical to those of **path**.

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

        
