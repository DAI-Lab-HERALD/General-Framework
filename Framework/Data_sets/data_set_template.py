import pandas as pd
import numpy as np
import os
import torch
import psutil
from data_interface import data_interface


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
                 overwrite_results = 'no'):
        # Find path of framework
        self.path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])

        # Clarify that no data has been loaded yet
        self.data_loaded = False
        self.raw_data_loaded = False
        self.raw_images_loaded = False
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
        
        self.prediction_overwrite = self.overwrite_results in ['model', 'prediction']

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
                    if not agent_path.shape[1] == 2:
                        raise TypeError("Path is expected to be consisting of np.ndarrays of shape (n x 2).")
                        
                    # test if input tuples have right length
                    if test_length != len(agent_path):
                        raise TypeError("Path sample does not have a matching number of timesteps.")
                        
                    if str(agent_type) == 'nan':
                        raise ValueError("For a given path, the agent type must not be nan.")
        
    def check_image_samples(self, Images):
        if not hasattr(Images, 'Target_MeterPerPx'):
            if not hasattr(self, 'Target_MeterPerPx'):
                raise AttributeError('Images without Px to Meter scaling are useless.')
            else:
                Images['Target_MeterPerPx'] = self.Target_MeterPerPx
        
        
    def check_created_paths_for_saving(self, last = False):
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
            If true, the last of the data was added and should therefore be saved. During a run
            of self.create_path_samples(), this should be set to *True* exactly once.
        
        '''
        
        # Check if the four required attributes exist
        if not all([hasattr(self, attr) for attr in ['Path', 'Type_old', 'T', 'Domain_old']]):
            raise AttributeError("The required arguments for saving have not been done.")
        
        # Check if the four attributes from self.create_path_samples are lists or dataframe/arrays
        if not isinstance(self.Path, pd.core.frame.DataFrame):
            assert isinstance(self.Path, list), "Path should be a list."
            # Transform to dataframe
            Path_check = pd.DataFrame(self.Path)
            
        if not isinstance(self.Type_old, pd.core.frame.DataFrame):
            assert isinstance(self.Type_old, list), "Type_old should be a list."
            # Transform to dataframe
            Type_old_check = pd.DataFrame(self.Type_old)
            
        if not isinstance(self.T, np.ndarray):
            assert isinstance(self.T, list), "T should be a list."
            # Transform to array
            T_check = np.array(self.T) 
            
        if not isinstance(self.Domain_old, pd.core.frame.DataFrame):
            assert isinstance(self.Domain_old, list), "Domain_old should be a list."
            # Transform to dataframe
            Domain_old_check = pd.DataFrame(self.Domain_old)
            
        # Test if final file allready exists
        final_test_name = self.file_path + '--all_orig_paths.npy'
        if os.path.isfile(final_test_name):
            raise AttributeError("*last = True* was passed more than once during self.create_path_samples().")
            
        # Get the currently available RAM space
        available_memory = psutil.virtual_memory().total - psutil.virtual_memory().used
        
        # Get the memory needed to save data right now
        # Count the number of timesteps in each sample
        num_timesteps = np.array([len(t) for t in self.T])
        
        # Get the number of saved agents for each sample
        num_agents = (~Path_check.isnull()).sum(axis=1)
        
        # Get the needed memory per timestep
        memory_per_timestep = 2 * 8 + 1
        memory_used = (num_timesteps * num_agents).sum() * memory_per_timestep
        
        # As data needs to be manipulated after loading, check if more than 25% of the memory is used
        if memory_used > 0.25 * available_memory or last:
            # Check if some saved original data is allready available
            file_path_test = self.file_path + '--all_orig_paths'
            file_path_test_name = os.path.basename(file_path_test)
            file_path_test_directory = os.path.dirname(file_path_test)
            # Find files in same directory that start with file_path_test
            files = [f for f in os.listdir(file_path_test_directory) if f.startswith(file_path_test_name)]
            if len(files) > 0:
                # Find the number that is attached to this file
                file_number = np.array([int(f[len(file_path_test_name)+1:-4]) for f in files], int).max() + 1
            else:
                file_number = 0
                
            if last:
                # During loading of files, check for existence of last file. If not there, rerun the whole extraction procedure
                file_path_save = file_path_test + '.npy'
            else:
                file_path_save = file_path_test + '_' + str(file_number).zfill(3) + '.npy'
                
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
                
             
            
        
    
    def get_number_of_original_path_files(self):
        # Get name of final file
        test_file = self.file_path + '--all_orig_paths.npy'
        
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
            test_file = self.file_path + '--all_orig_paths.npy'
            image_file = self.file_path + '--Images.npy'
            
            image_creation_unneeded = (not self.includes_images()) or os.path.isfile(image_file)

            if os.path.isfile(test_file) and image_creation_unneeded:
                # Get number of files used for saving
                self.number_original_path_files = self.get_number_of_original_path_files()
                
                if self.number_original_path_files == 1:
                    # Allready load samples for higher efficiency
                    [self.Path,
                    self.Type_old,
                    self.T,
                    self.Domain_old,
                    self.num_samples] = np.load(test_file, allow_pickle=True)
                
                # Load Images
                if self.includes_images():
                    [self.Images, _] = np.load(image_file, allow_pickle=True)
                else:
                    self.Images = None
            else:
                if not all([hasattr(self, attr) for attr in ['create_path_samples']]):
                    raise AttributeError("The raw data cannot be loaded.")
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
                
                # Save the image data
                if self.includes_images():
                    if not hasattr(self, 'Images'):
                        raise AttributeError('Images are missing.')
                    
                    self.check_image_samples(self.Images)

                    image_data = np.array([self.Images, 0], object)
                    np.save(image_file, image_data)
                else:
                    self.Images = None
                    
                np.save(test_file, test_data)
                
            self.raw_data_loaded = True
            self.raw_images_loaded = True
            
    def load_raw_images(self):
        if not self.raw_images_loaded and self.includes_images():
            image_file = self.file_path + '--Images.npy'
            if os.path.isfile(image_file):
                [self.Images, _] = np.load(image_file, allow_pickle=True)
            else:
                self.load_raw_data()
                # save the results
                image_data = np.array([self.Images, 0], object)

                os.makedirs(os.path.dirname(image_file), exist_ok=True)
                np.save(image_file, image_data)

            self.raw_images_loaded = True

    def reset(self):
        self.data_loaded = False
        self.path_models_trained = False


    def classify_path(self, path, t, domain):
        r'''
        This function classifies a given set of trajectories.

        Parameters
        ----------
        path : pandas.Series
            A pandas series of :math:`(2 N_{agents})` dimensions,
            where each entry is itself a numpy array of lenght :math:`|T|`, the number of recorded timesteps.
        t : numpy.ndarray
            A numpy array of lenght :math:`|T|`, recording the corresponding timesteps.

        Returns
        -------
        Class : integer
            Returns the class name for the current behavior

        T_Delta : pandas.Series
            This is a :math:`N_{classes}` dimensional Series.
            For each column, it returns an array of lenght :math:`|T|` with the predicted time 
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
            A pandas series of :math:`(2 N_{agents})` dimensions,
            where each entry is itself a numpy array of lenght :math:`|T|`, the number of recorded timesteps.
        t : numpy.ndarray
            A numpy array of lenght :math:`|T|`, recording the corresponding timesteps.

        Returns
        -------
        Pred : pandas.Series
            This is a :math:`N_{classes} + N_{other dist}` dimensional Series.
            For each column, it returns an array of lenght :math:`|T|`.

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
            self.dtc_boundary = np.load( self.data_dtc_bound_file, allow_pickle=True)
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
                        path_file = self.file_path + '--all_orig_paths.npy'
                        
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
                            path_file_adjust = ''
                        
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


    def extract_t0(self, t0_type, t, t_start, t_decision, t_crit, i_sample, behavior):
        if t0_type[:5] == 'start':
            T0 = [t_start]
        
        
        elif t0_type[:3] == 'all':
            min_t0 = max(t_start, t.min() + self.dt * (self.num_timesteps_in_need - 1))
            if self.enforce_prediction_time or not self.classification_useful:
                max_t0 = min(t_decision, t.max() - self.dt * self.num_timesteps_out_need) + 1e-6
            else:
                max_t0 = t_decision + 1e-6
            T0 = np.arange(min_t0, max_t0, self.dt)

        elif t0_type[:3] == 'col':
            if self.classification_useful:
                t_D_default = self.T_D_class.iloc[i_sample][self.behavior_default]
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
        if type((1, 1)) == type(num_timesteps):
            # Refers to sctual input data
            num_timesteps_real = min(99, num_timesteps[0])
            num_timesteps_need = max(num_timesteps_real, min(99, num_timesteps[1]))

        # If only one value is given, assume that the required number of steps is identical
        elif type(1) == type(num_timesteps):
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
        t0_type_file_name = {'start':            'start',
                             'all':              'all_p',
                             'col_equal':        'fix_e',
                             'col_set':          'fix_s',
                             'crit':             'crit_'}
        t0_type_name = t0_type_file_name[self.t0_type] + '_'
        
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
        
        if self.agents_to_predict == 'predefined':
            pat = '0'
        elif self.agents_to_predict == 'all':
            pat = 'A'
        else:
            pat = self.agents_to_predict[0]


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
                          '--max_' + str(num).zfill(3) + '_agents_' + pat + '--Perturbations.xlsx')
            
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
                     '--max_' + str(num).zfill(3) + '_agents_' + pat +
                     extra_string + pert_string + '.npy')
        
        return data_file

    def get_number_of_data_files(self):
        # Check if file exists
        if not os.path.isfile(self.data_file):
            raise AttributeError("The data has not been completely compiled yet.")
        
        # Get the corresponding directory
        test_file_directory = os.path.dirname(self.data_file)
        
        # Find files in same directory that start with file_path_test
        num_files = len([f for f in os.listdir(test_file_directory) if f.startswith(os.path.basename(self.data_file[:-4]))])
        return num_files
    
    
    
    def get_data_from_orig_path(self, Path, Type_old, T, Domain_old, num_samples, path_file, data_file):
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
        num_behaviors = np.array([(beh == local_behavior).sum() for beh in self.Behaviors])
        
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

            # Get the time of prediction
            T0 = self.extract_t0(self.t0_type, t, t_start, t_decision, t_crit, i, behavior)
            
            # Extract comparable T0 types
            T0_compare = []
            if self.t0_type[:3] != 'all':
                for extra_t0_type in self.T0_type_compare:
                    if extra_t0_type[:3] == 'all':
                        raise TypeError("Comparing against the all method is not possible")
                    else:
                        T0_compare.append(self.extract_t0(extra_t0_type, t, t_start, t_decision, t_crit, i, behavior)) 
            
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
                    pert_index = int(self.data_file.split('_')[-1].split('.')[0])
                    domain['Scenario'] = self.get_name()['print'] + ' (Pertubation_' + str(pert_index).zfill(3) + ')'
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
                
                assert t.dtype == input_T.dtype
                
                correct_path = True
                for agent in path.index:
                    if not isinstance(path[agent], float):
                        pos_x = path[agent][:,0]
                        pos_y = path[agent][:,1]
                        helper_path[agent] = np.stack([np.interp(helper_T_appr, t, pos_x, left=np.nan, right=np.nan),
                                                    np.interp(helper_T_appr, t, pos_y, left=np.nan, right=np.nan)], 
                                                    axis = -1).astype(np.float32)
                        
                        if np.sum(np.isfinite(helper_path[agent][:self.num_timesteps_in_real]).all(-1)) <= 1:
                            helper_path[agent]  = np.nan
                            agent_types[agent] = float('nan')
                            
                    else:
                        helper_path[agent]  = np.nan
                        agent_types[agent] = float('nan')
                        
                    # check if needed agents have reuqired input and output
                    if agent in self.needed_agents:
                        if np.isnan(helper_path[agent]).any():
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
                        available_pos = np.isfinite(helper_path[agent]).all(-1)
                        assert available_pos.sum() > 1 
                        
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
                                available_pos = recorded_positions[agent]

                                # Do not delete interpolated data
                                ind_start = np.where(available_pos)[0][0]
                                ind_last = np.where(available_pos)[0][-1] + 1
                                
                                available_pos[ind_start:ind_last] = True

                                helper_path[agent][~available_pos] = np.nan
                            else:
                                ind_start = 0
                                ind_last = len(helper_T)
                        
                        input_path[agent]  = helper_path[agent][:self.num_timesteps_in_real, :].astype(np.float32)
                        output_path[agent] = helper_path[agent][self.num_timesteps_in_real:len(helper_T), :].astype(np.float32)

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
                
                # Check if saving is needed
                self.check_extracted_data_for_saving(num_behaviors, data_file)
                
        # Save the data with last = True
        self.check_extracted_data_for_saving(num_behaviors, data_file, True)
    
    
    def check_extracted_data_for_saving(self, num_behaviors, data_file, last = False):
        
        # Get the currently available RAM space
        available_memory = psutil.virtual_memory().total - psutil.virtual_memory().used
        
        # Get the memory needed to save data right now
        # Count the number of timesteps in each sample
        num_timesteps_in  = np.array([len(t) for t in self.Input_T_local])
        num_timesteps_out = np.array([len(t) for t in self.Output_T_local])
        
        # TODO: Check if this also works for manually assinged np.nan values
        # Get the number of saved agents for each sample
        num_agents = (~self.Input_path_local.isnull()).sum(axis=1)
        
        # Get the needed memory per timestep
        memory_per_timestep = 2 * 8 + 2 # one extra for timestep and recorded
        
        memory_used_path_in  = (num_timesteps_in * num_agents).sum() * memory_per_timestep
        memory_used_path_out = (num_timesteps_out * num_agents).sum() * (memory_per_timestep + 1)
        
        # Get memory for prediction data
        memory_used_pred = num_timesteps_in.sum() * len(self.Input_prediction_local.columns) * 8
        
        memory_used = memory_used_path_in + memory_used_path_out + memory_used_pred
        
        # As data needs to be manipulated after loading, check if more than 25% of the memory is used
        if memory_used > 0.4 * available_memory or last:
            # Transform data to dataframes
            self.Input_prediction = pd.DataFrame(self.Input_prediction_local)
            self.Input_path       = pd.DataFrame(self.Input_path_local)
            self.Input_T          = np.array(self.Input_T_local + [np.random.rand(0)], np.ndarray)[:-1]
            
            self.Output_path   = pd.DataFrame(self.Output_path_local)
            self.Output_T      = np.array(self.Output_T_local + [np.random.rand(0)], np.ndarray)[:-1]
            self.Output_T_pred = np.array(self.Output_T_pred_local + [np.random.rand(0)], np.ndarray)[:-1]
            self.Output_A      = pd.DataFrame(self.Output_A_local)
            self.Output_T_E    = np.array(self.Output_T_E_local, float)
            
            self.Type     = pd.DataFrame(self.Type_local).reset_index(drop = True)
            self.Recorded = pd.DataFrame(self.Recorded_local).reset_index(drop = True)
            self.Domain   = pd.DataFrame(self.Domain_local).reset_index(drop = True)
            
            # Ensure that dataframes with agent columns have the same order
            Agents = self.Input_path.columns
            self.Input_path  = self.Input_path[Agents]
            self.Output_path = self.Output_path[Agents]
            self.Type        = self.Type[Agents]
            self.Recorded    = self.Recorded[Agents]

            # Ensure that indices of dataframes are the same
            self.Input_prediction.index = self.Input_path.index
            self.Output_path.index      = self.Input_path.index
            self.Output_A.index         = self.Input_path.index
            self.Type.index             = self.Input_path.index
            self.Recorded.index         = self.Input_path.index
            self.Domain.index           = self.Input_path.index
            
            ## Get the corresponding save file
            data_file_name = os.path.basename(data_file)
            data_file_directory = os.path.dirname(data_file)
            
            # Make directory
            os.makedirs(data_file_directory, exist_ok=True)
            
            # Find files in same directory that start with file_path_test
            files = [f for f in os.listdir(data_file_directory) if f.startswith(data_file_name)]
            if len(files) > 0:
                # Find the number that is attached to this file
                file_number = np.array([int(f[len(data_file_name)+1:-4]) for f in files], int).max() + 1
            else:
                file_number = 0
                
                
            # Apply perturbation if necessary:
            self.Domain['perturbation'] = False
            
            if self.is_perturbed:
                self = self.Perturbation.perturb(self)
                
                # Get unperturbed data
                Input_path_unperturbed  = self.Domain['Unperturbed_input']
                Output_path_unperturbed = self.Domain['Unperturbed_output']
                
                # Transform series to list
                Input_path_unperturbed  = [path[0] for path in Input_path_unperturbed.to_list()]
                Output_path_unperturbed = [path[0] for path in Output_path_unperturbed.to_list()]
                
                # Transform list to dataframe
                Input_path_unperturbed  = pd.DataFrame(Input_path_unperturbed, index = self.Domain.index)
                Output_path_unperturbed = pd.DataFrame(Output_path_unperturbed, index = self.Domain.index)
                
                # Remove unperturbed columns from domain
                self.Domain = self.Domain.drop(columns = ['Unperturbed_input', 'Unperturbed_output'])
                self.Domain['perturbation'] = False
                
                # Get the unperturbed save data
                save_data_unperturbed = [
                                            self.Input_prediction,
                                            Input_path_unperturbed,
                                            self.Input_T,
                                            
                                            Output_path_unperturbed,
                                            self.Output_T,
                                            self.Output_T_pred,
                                            self.Output_A,
                                            self.Output_T_E,
                                            
                                            self.Type,
                                            self.Recorded,
                                            self.Domain,
                                            num_behaviors, 0]
                
                # Get unperturbed save file
                
                data_file_unperturbed = '--'.join(data_file.split('--').pop(-1))
                if last:
                    data_file_unperturbed_save = data_file_unperturbed + '.npy'
                else:
                    data_file_unperturbed_save = data_file_unperturbed + '_' + str(file_number).zfill(3) + '.npy'
                
                # Save the unperturbed data
                np.save(data_file_unperturbed_save, save_data_unperturbed)
                
                # Set perturbation to True
                self.Domain['perturbation'] = True
            
            save_data = np.array([
                                    self.Input_prediction,
                                    self.Input_path,
                                    self.Input_T,
                                    
                                    self.Output_path,
                                    self.Output_T,
                                    self.Output_T_pred,
                                    self.Output_A,
                                    self.Output_T_E,
                                    
                                    self.Type,
                                    self.Recorded,
                                    self.Domain,
                                    num_behaviors, 0], object)
            
                
            if last:
                # During loading of files, check for existence of last file. If not there, rerun the whole extraction procedure
                data_file_save = data_file + '.npy'
            else:
                data_file_save = data_file + '_' + str(file_number).zfill(3) + '.npy'
                
            # Save the data
            np.save(data_file_save, save_data)
            
            # Empty local data files
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
        

        # check if same data set has already been done in the same way
        if not os.path.isfile(self.data_file):
            # load initial dataset, if not yet done
            self.load_raw_data()
            
            

            # If necessary, load constant gap size
            if ((self.t0_type[:9] == 'col_equal') or 
                ('col_equal' in [t0_type_extra[:9] for t0_type_extra in self.T0_type_compare])):
                self.determine_dtc_boundary()
            
            # Go through original data
            for i_orig_path in range(self.number_original_path_files):
            
                if self.number_original_path_files == 1:
                    path_file = self.file_path + '--all_orig_paths.npy'
                    
                    # Get the allready loaded data
                    Path_loaded = self.Path
                    Type_old_loaded = self.Type_old
                    T_loaded = self.T
                    Domain_old_loaded = self.Domain_old
                    num_samples_loaded = self.num_samples
                    
                    data_file_local = self.data_file[:-4]
                    
            
                else:
                    # Get data file
                    path_file = self.file_path + '--all_orig_paths'
                    
                    # Get path name adjustment
                    if i_orig_path < self.number_original_path_files - 1:
                        path_file_adjust = '_' + str(i_orig_path).zfill(3)
                    else:
                        path_file_adjust = ''
                    
                    path_file += path_file_adjust + '.npy'
                    
                    # Load the data
                    [Path_loaded,
                    Type_old_loaded,
                    T_loaded,
                    Domain_old_loaded,
                    num_samples_loaded] = np.load(path_file, allow_pickle=True)
                    
                    data_file_local = self.data_file[:-4] + path_file_adjust
                    

                self.get_data_from_orig_path(Path_loaded, Type_old_loaded, T_loaded, Domain_old_loaded, num_samples_loaded, path_file, data_file_local)
                
                
        
        # Get the number of files
        self.number_data_files = self.get_number_of_data_files(self.data_file)
        
        # If only one file is available, load the data
        if self.number_data_files == 1:
            # Load the data
            [self.Input_prediction,
            self.Input_path,
            self.Input_T,

            self.Output_path,
            self.Output_T,
            self.Output_T_pred,
            self.Output_A,
            self.Output_T_E,

            self.Type,
            self.Recorded,
            self.Domain,
            self.num_behaviors, _] = np.load(self.data_file, allow_pickle=True)
                
                
                
                

        self.data_loaded = True
        # check if dataset is useful
        if len(self.Output_A) < 100:
            return "there are not enough samples for a reasonable training process."

        if self.classification_useful:
            beh_counts = np.unique(self.Output_A.to_numpy(), axis=0, return_counts=True)[1]
            if len(beh_counts) == 1:
                return "the dataset is too unbalanced for a reasonable training process."
            if 10 > np.sort(beh_counts)[-2]:
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
                            Imgs_rot, Imgs_index):
        if self.includes_images():
            print('')
            print('Load needed images:', flush = True)
            
            self.load_raw_images()
            
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
            
            print('')
            print('Get locations:', flush = True)
            Locations = domain.image_id.to_numpy()
            
            print('Extract rotation matrix', flush = True)
            max_size = 2 * max(self.Images.Image.iloc[0].shape[:2]) + 1
            
            # check if images are float
            if self.Images.Image.iloc[0].max() > 1:
                rgb = True
                assert self.Images.Image.iloc[0].max() < 256
            else:
                rgb = False
            
            if target_width is None:
                target_width = max_size
                
            if target_height is None:
                target_height = max_size
                
            if rot_angle is None:
                first_stage = False
            else:
                first_stage = True
            
            if hasattr(domain, 'rot_angle'):
                second_stage = True
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
                
            # CPU
            print('Reserved memory for rotated images.', flush = True)
            CPU_mem = psutil.virtual_memory()
            cpu_total = CPU_mem.total / 2 ** 30
            cpu_used  = CPU_mem.used / 2 ** 30
            print('CPU: {:5.2f}/{:5.2f} GB are available'.format(cpu_total - cpu_used, cpu_total), flush = True)

            torch.cuda.empty_cache()
            gpu_total         = torch.cuda.get_device_properties(device = device).total_memory  / 2 ** 30
            gpu_reserved      = torch.cuda.memory_reserved(device = device) / 2 ** 30
            gpu_max_reserved  = torch.cuda.max_memory_reserved(device = device) / 2 ** 30
            torch.cuda.reset_peak_memory_stats()
            print('GPU: {:5.2f}/{:5.2f} GB are available'.format(gpu_total - gpu_reserved, gpu_total), flush = True)
            print('GPU previous min available: {:0.2f}'.format(gpu_total - gpu_max_reserved), flush = True)
            print('', flush = True)
            print('start rotating', flush = True)
            n = 250
            
            image_num = 0
            for location in np.unique(Locations):
                loc_indices = np.where(Locations == location)[0]
                
                loc_Image = torch.from_numpy(self.Images.Image.loc[location]).to(device = device)
                loc_M2px  = float(self.Images.Target_MeterPerPx.loc[location])
                for i in range(0, len(loc_indices), n):
                    torch.cuda.empty_cache()
                    Index_local = np.arange(i, min(i + n, len(loc_indices)))
                    Index = loc_indices[Index_local]
                    
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
            
        
        
        
        
        
        
        
        
        

    # %% Implement transformation functions
    def train_path_models(self):
        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified")

        if not self.path_models_trained:
            self.path_models = pd.Series(np.empty(len(self.Behaviors), object), index=self.Behaviors)
            self.path_models_pred = pd.Series(np.empty(len(self.Behaviors), object), index=self.Behaviors)

            # Create data interface out of self
            # Get parameters
            parameters = [None, 
                            self.num_samples_path_pred,
                            self.enforce_num_timesteps_out,
                            self.enforce_prediction_time,
                            self.exclude_post_crit,
                            self.allow_extrapolation,
                            self.agents_to_predict,
                            'no']
            # Get data set dict
            data_set_dict = {'scenario': self.__class__.__name__,
                             'max_num_agents': self.max_num_agents,
                             't0_type': self.t0_type,
                             'conforming_t0_types': self.T0_type_compare}
            data = data_interface(data_set_dict, parameters)
            data.get_data(self.dt,
                          (self.num_timesteps_in_real, self.num_timesteps_in_need), 
                          (self.num_timesteps_out_real, self.num_timesteps_out_need))

            for beh in self.Behaviors:
                
                self.path_models[beh] = self.model_class_to_path({}, data, None, True, beh)
                self.path_models[beh].train()
                beh_pred = self.path_models[beh].predict()
                beh_pred[1].index = beh_pred[0]
                self.path_models_pred[beh] = beh_pred[1]

            self.path_models_trained = True

    def transform_outputs(self, output, model_pred_type, metric_pred_type, pred_save_file):
        # Check if tranformation is necessary
        if model_pred_type == metric_pred_type:
            return output

        # If transformation is actually necessary
        else:
            # If the metric requires trajectory predictions
            if metric_pred_type == 'path_all_wo_pov':
                if model_pred_type == 'class':
                    [Pred_index, Output_A_pred] = output
                    Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain, pred_save_file)

                    Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index,
                                                                   self.Domain, pred_save_file)
                elif model_pred_type == 'class_and_time':
                    [Pred_index, Output_A_pred, Output_T_E_pred] = output
                    Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index,
                                                                   self.Domain, pred_save_file)
                elif model_pred_type == 'path_all_wi_pov':
                    [Pred_index, Output_path_pred] = output
                else:
                    raise AttributeError(
                        "This type of output produced by the model is not implemented")

                Output_path_pred = self.path_remove_pov_agent(Output_path_pred, Pred_index, 
                                                              self.Domain, pred_save_file)
                output_trans = [Pred_index, Output_path_pred]

            elif metric_pred_type == 'path_all_wi_pov':
                if model_pred_type == 'class':
                    [Pred_index, Output_A_pred] = output
                    Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain, pred_save_file)

                    Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index,
                                                                   self.Domain, pred_save_file)
                elif model_pred_type == 'class_and_time':
                    [Pred_index, Output_A_pred, Output_T_E_pred] = output
                    Output_path_pred = self.class_and_time_to_path(Output_A_pred, Output_T_E_pred, Pred_index,
                                                                   self.Domain, pred_save_file)
                elif model_pred_type == 'path_all_wo_pov':
                    [Pred_index, Output_path_pred] = output
                    Output_path_pred = self.path_add_pov_agent(Output_path_pred, Pred_index,
                                                               self.Domain, pred_save_file)
                else:
                    raise AttributeError(
                        "This type of output produced by the model is not implemented")
                output_trans = [Pred_index, Output_path_pred]

            # If the metric requires class predictions
            elif metric_pred_type == 'class':
                if model_pred_type == 'path_all_wo_pov':
                    [Pred_index, Output_path_pred] = output
                    Output_path_pred = self.path_add_pov_agent(Output_path_pred, Pred_index, 
                                                               self.Domain, pred_save_file)

                    [Output_A_pred, _] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Output_T_pred,
                                                                     self.Domain, pred_save_file)

                elif model_pred_type == 'path_all_wi_pov':
                    [Pred_index, Output_path_pred] = output
                    [Output_A_pred, _] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Output_T_pred,
                                                                     self.Domain, pred_save_file)

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
                    Output_path_pred = self.path_add_pov_agent(Output_path_pred, Pred_index, 
                                                               self.Domain, pred_save_file)
                    [Output_A_pred,
                     Output_T_E_pred] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Output_T_pred,
                                                                    self.Domain, pred_save_file)

                elif model_pred_type == 'path_all_wi_pov':
                    [Pred_index, Output_path_pred] = output
                    [Output_A_pred,
                     Output_T_E_pred] = self.path_to_class_and_time(Output_path_pred, Pred_index, self.Output_T_pred,
                                                                    self.Domain, pred_save_file)

                elif model_pred_type == 'class':
                    [Pred_index, Output_A_pred] = output
                    Output_T_E_pred = self.class_to_time(Output_A_pred, Pred_index, self.Domain, pred_save_file)
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
                    class_change = np.where(
                        (Dist[beh][j, 1:] <= 0) & (Dist[beh][j, :-1] > 0))[0]
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
    
    
    def path_add_pov_agent(self, Output_path_pred, Pred_index, Domain, pred_save_file, use_model=False):
        test_file = '--'.join(pred_save_file.split('--')[:-1]) + '--pred_tra_wip_00.npy'
        if os.path.isfile(test_file) and not self.prediction_overwrite:
            output = np.load(test_file, allow_pickle = True)
            
            Output_path_pred_add = [output[1]]
            
            save = 1 
            new_test_file = test_file[:-6] + str(save).zfill(2)+ '.npy'
            while os.path.isfile(new_test_file):
                output = np.load(new_test_file, allow_pickle = True)
                Output_path_pred_add.append(output[1])
                
                save += 1 
                new_test_file = test_file[:-6] + str(save).zfill(2)+ '.npy'
            
            Output_path_pred_add = pd.concat(Output_path_pred_add)
            
            return Output_path_pred_add
        
        # Add extrapolated pov agent to data
        Index_old = Output_path_pred.columns
        
        if self.pov_agent is None:
            Index_new = [self.pov_agent]
        else:
            Index_new = []
            
        Index_add = Index_new + list(Index_old)
        Output_path_pred_add = pd.DataFrame(np.empty((len(Output_path_pred), len(Index_add)), object),
                                            columns=Index_add)

        to_save_timesteps = np.zeros(len(Output_path_pred))
        for i_sample, i_full in enumerate(Pred_index):
            if self.pov_agent is None:
                path_true = self.Ouput_path.iloc[i_full]
                t = self.Output_T_pred[i_full]
                t_true = self.Output_T[i_full]
                index = Index_new[0]
                # interpolate the values at the new set of points using numpy.interp()
                path_index_old = path_true[index]
                path_index_new = np.stack([np.interp(t, t_true, path_index_old[:,0]),
                                           np.interp(t, t_true, path_index_old[:,1])], axis = -1)
    
                # use the gradient to estimate values outside the bounds of xp
                dx = np.stack([np.gradient(path_index_old[:,0], t_true),
                               np.gradient(path_index_old[:,1], t_true)], axis = -1)
                later_time = t > t_true[-1]
                path_index_new[later_time] = path_index_old[[-1]] + (t[later_time] - t_true[-1])[:,np.newaxis] * dx[[-1]]
    
                # Add new results
                Output_path_pred_add.iloc[i_sample, 0] = np.tile(path_index_new[np.newaxis],
                                                                 (self.num_samples_path_pred, 1, 1))
            Output_path_pred_add.iloc[i_sample, -len(Index_old):] = Output_path_pred.iloc[i_sample]
            
            num_pred_agents = np.sum(np.array([isinstance(Output_path_pred_add.iloc[i_sample,j], np.ndarray) 
                                               for j in range(Output_path_pred_add.shape[1])]))
            to_save_timesteps[i_sample] = len(self.Output_T_pred[i_full]) * self.num_samples_path_pred * num_pred_agents
        
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        savable_timesteps_total = 2 ** 27 # 2 ** 27 should be 1 GB
        
        Unsaved_indices = np.arange(len(Pred_index))
        save = 0
        while len(Unsaved_indices) > 0:
            to_save_timesteps_cum = np.cumsum(to_save_timesteps[Unsaved_indices])
            if to_save_timesteps_cum[-1] <= savable_timesteps_total:
                num_saved = len(Unsaved_indices)
            else:
                num_saved = np.where(to_save_timesteps_cum > savable_timesteps_total)[0][0]

            save_indices = Unsaved_indices[:num_saved]
            save_data = np.array([Pred_index[save_indices], 
                                  Output_path_pred_add.iloc[save_indices], 
                                  0], object)
            
            save_file = test_file[:-6] + str(save).zfill(2) + '.npy'
            np.save(save_file, save_data)
            
            save += 1
            Unsaved_indices = Unsaved_indices[num_saved:]
            
            print('Saved part {} of predicted trajectories'.format(save))

        return Output_path_pred_add

    def path_remove_pov_agent(self, Output_path_pred, Pred_index, Domain, pred_save_file):
        test_file = '--'.join(pred_save_file.split('--')[:-1]) + '--pred_tra_wop_00.npy'
        if os.path.isfile(test_file) and not self.prediction_overwrite:
            output = np.load(test_file, allow_pickle = True)
            
            Output_path_pred_remove = [output[1]]
            
            save = 1 
            new_test_file = test_file[:-6] + str(save).zfill(2)+ '.npy'
            while os.path.isfile(new_test_file):
                output = np.load(new_test_file, allow_pickle = True)
                Output_path_pred_remove.append(output[1])
                
                save += 1 
                new_test_file = test_file[:-6] + str(save).zfill(2)+ '.npy'
            
            Output_path_pred_remove = pd.concat(Output_path_pred_remove)
            
            return Output_path_pred_remove
        
        Index_retain = np.array([(name != self.pov_agent) for name in Output_path_pred.columns])
        Output_path_pred_remove = Output_path_pred.iloc[:, Index_retain]
        
        # Get number of predicted timesteps per trajectory
        to_save_timesteps = np.zeros(len(Output_path_pred))
        for i_sample, i_full in enumerate(Pred_index):
            num_pred_agents = np.sum(np.array([isinstance(Output_path_pred_remove.iloc[i_sample,j], np.ndarray) 
                                               for j in range(Output_path_pred_remove.shape[1])]))
            to_save_timesteps[i_sample] = len(self.Output_T_pred[i_full]) * self.num_samples_path_pred * num_pred_agents
        
        # Save predicted trajectories
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        savable_timesteps_total = 2 ** 27 # 2 ** 27 should be 1 GB
        
        Unsaved_indices = np.arange(len(Pred_index))
        save = 0
        while len(Unsaved_indices) > 0:
            to_save_timesteps_cum = np.cumsum(to_save_timesteps[Unsaved_indices])
            if to_save_timesteps_cum[-1] <= savable_timesteps_total:
                num_saved = len(Unsaved_indices)
            else:
                num_saved = np.where(to_save_timesteps_cum > savable_timesteps_total)[0][0]

            save_indices = Unsaved_indices[:num_saved]
            save_data = np.array([Pred_index[save_indices], 
                                  Output_path_pred_remove.iloc[save_indices], 
                                  0], object)
            
            save_file = test_file[:-6] + str(save).zfill(2) + '.npy'
            np.save(save_file, save_data)
            
            save += 1
            Unsaved_indices = Unsaved_indices[num_saved:]
            
        return Output_path_pred_remove

    def path_to_class_and_time(self, Output_path_pred, Pred_index, 
                               Output_T_pred, Domain, pred_save_file, save_results=True):
        if self.classification_useful:
            # Remove other prediction method
            test_file = '--'.join(pred_save_file.split('--')[:-1]) + '--pred_class_time.npy'
    
            if os.path.isfile(test_file) and not self.prediction_overwrite:
                [Output_A_pred,
                 Output_T_E_pred, _] = np.load(test_file, allow_pickle=True)
            else:
                Output_A_pred = pd.DataFrame(np.zeros((len(Output_path_pred), len(self.Behaviors)), float),
                                             columns = self.Behaviors)
                Output_T_E_pred = pd.DataFrame(np.empty((len(Output_path_pred), len(self.Behaviors)), object),
                                               columns = self.Behaviors)
    
                for i_sample, i_full in enumerate(Pred_index):
                    paths = Output_path_pred.iloc[i_sample]
                    t = Output_T_pred[i_full]
                    domain = Domain.iloc[i_full]
                    T_class = self.path_to_class_and_time_sample(paths, t, domain)
    
                    output_A = np.arange(len(self.Behaviors))[np.newaxis] == T_class.argmin(axis=-1)[:,np.newaxis]
    
                    Output_A_pred.iloc[i_sample] = pd.Series(output_A.mean(axis=0), index=self.Behaviors)
    
                    for i_beh, beh in enumerate(self.Behaviors):
                        T_beh = T_class[output_A[:, i_beh], i_beh]
                        if len(T_beh) > 0:
                            Output_T_E_pred.iloc[i_sample, i_beh] = np.quantile(T_beh, self.p_quantile)
                        else:
                            Output_T_E_pred.iloc[i_sample, i_beh] = np.full(len(self.p_quantile), np.nan)
                if save_results:
                    save_data = np.array(
                        [Output_A_pred, Output_T_E_pred, 0], object)
                    os.makedirs(os.path.dirname(test_file), exist_ok=True)
                    np.save(test_file, save_data)
        else:
            Output_A_pred = pd.DataFrame(np.ones((len(Output_path_pred), len(self.Behaviors)), float),
                                         columns = self.Behaviors)
            Output_T_E_pred = pd.DataFrame(np.empty((len(Output_path_pred), len(self.Behaviors)), object),
                                           columns = self.Behaviors)
            
        return Output_A_pred, Output_T_E_pred

    def class_to_time(self, Output_A_pred, Pred_index, Domain, pred_save_file):
        # Remove other prediction type
        if self.classification_useful:
            test_file = '--'.join(pred_save_file.split('--')[:-1]) + '--pred_class_time.npy'
            if os.path.isfile(test_file) and not self.prediction_overwrite:
                [_, Output_T_E_pred, _] = np.load(test_file, allow_pickle=True)
            else:
                self.train_path_models()
    
                Output_T_E_pred = pd.DataFrame(np.empty((len(Output_A_pred), len(self.Behaviors)), object),
                                               columns = self.Behaviors)
                for i_sample, i_full in enumerate(Pred_index):
                    t = self.Output_T_pred[i_full]
                    domain = Domain.iloc[i_full]
                    for i_beh, beh in enumerate(self.Behaviors):
                        paths_beh = self.path_models_pred[beh].loc[i_full]
                        T_class_beh = self.path_to_class_and_time_sample(paths_beh, t, domain)
                        T_beh = T_class_beh[((T_class_beh[:, i_beh] == T_class_beh.min(axis=-1)) &
                                             (T_class_beh[:, i_beh] <= t[-1])), i_beh]
    
                        if len(T_beh) > 0:
                            Output_T_E_pred.iloc[i_sample, i_beh] = np.quantile(T_beh, self.p_quantile)
                        else:
                            Output_T_E_pred.iloc[i_sample, i_beh] = np.full(len(self.p_quantile), np.nan)
    
                save_data = np.array([Output_A_pred, Output_T_E_pred, 0], object)
                os.makedirs(os.path.dirname(test_file), exist_ok=True)
                np.save(test_file, save_data)
        else:
            Output_T_E_pred = pd.DataFrame(np.empty(Output_A_pred.shape, object), columns = self.Behaviors)
        return Output_T_E_pred

    def class_and_time_to_path(self, Output_A_pred, Output_T_E_pred, Pred_index, Domain, pred_save_file):
        assert self.classification_useful, "For not useful datasets training classification models should be impossible."
        # check if this has already been performed
        test_file = '--'.join(pred_save_file.split('--')[:-1]) + '--pred_tra_wip_00.npy'
        if os.path.isfile(test_file) and not self.prediction_overwrite:
            output = np.load(test_file, allow_pickle = True)
            
            Output_path_pred = [output[1]]
            
            save = 1 
            new_test_file = test_file[:-6] + str(save).zfill(2) + '.npy'
            while os.path.isfile(new_test_file):
                output = np.load(new_test_file, allow_pickle = True)
                Output_path_pred.append(output[1])
                
                save += 1 
                new_test_file = test_file[:-6] + str(save).zfill(2)+ '.npy'
            
            Output_path_pred = pd.concat(Output_path_pred)
            
            return Output_path_pred
        
        self.train_path_models()
        Index = self.Output_path.columns
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
        
        # Go over all samples individually
        to_save_timesteps = np.zeros(len(Output_path_pred))
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
                    paths_beh = self.path_models_pred[beh].loc[i_full]
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
            
            num_pred_agents = np.sum(np.array([isinstance(Output_path_pred.iloc[i_sample,j], np.ndarray) 
                                               for j in range(Output_path_pred.shape[1])]))
            to_save_timesteps[i_sample] = len(self.Output_T_pred[i_full]) * self.num_samples_path_pred * num_pred_agents
        
        # Save predicted trajectories
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        savable_timesteps_total = 2 ** 27 # 2 ** 27 should be 1 GB
        
        Unsaved_indices = np.arange(len(Pred_index))
        save = 0
        while len(Unsaved_indices) > 0:
            to_save_timesteps_cum = np.cumsum(to_save_timesteps[Unsaved_indices])
            if to_save_timesteps_cum[-1] <= savable_timesteps_total:
                num_saved = len(Unsaved_indices)
            else:
                num_saved = np.where(to_save_timesteps_cum > savable_timesteps_total)[0][0]

            save_indices = Unsaved_indices[:num_saved]
            save_data = np.array([Pred_index[save_indices], 
                                  Output_path_pred.iloc[save_indices], 
                                  0], object)
            
            save_file = test_file[:-6] + str(save).zfill(2) + '.npy'
            np.save(save_file, save_data)
            
            save += 1
            Unsaved_indices = Unsaved_indices[num_saved:]
            
            print('Saved part {} of predicted trajectories'.format(save))
            
        Unsaved_indices = np.arange(len(Pred_index))
        split_factor = 1
        save = 0
        while len(Unsaved_indices) > 0:
            try:
                num_saved = int(np.ceil(len(Pred_index) / split_factor))
                save_indices = Unsaved_indices[:num_saved]
                save_data = np.array([Pred_index[save_indices], 
                                      Output_path_pred.iloc[save_indices], 
                                      0], object)
                
                save_file = test_file[:-6] + str(save).zfill(2) + '.npy'
                
                np.save(save_file, save_data)
                
                save += 1
                Unsaved_indices = Unsaved_indices[num_saved:]
                
            except:
                split_factor *= 1.99
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
            that gives us the scaling of the images in the unit :math:`m /` Px. 
    
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
            where each entry is itself a numpy array of shape :math:`\{N_{preds} \times |t| \times 2 \}`.
            The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
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
            where each entry is itself a numpy array of lenght :math:`\{|t| \times 2 \}`.
            The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
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
            where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
            The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
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
            where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
            The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
            and should include at least the relevant agents described in self.create_sample_paths.
        t : numpy.ndarray
            A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
            at which the positions in **path** were recorded.
        domain : pandas.Series
            A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
            sample. Its entries contain at least all the columns of **self.Domain_old**. 
        agent_types : pandas.Series 
            A pandas series with :math:`(N_{agents})` entries, that records the type of the agents for the considered
            sample. The columns should correspond to the columns in **self.Type_old** created in self.create_path_samples()
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
            sample. The columns should correspond to the columns in **path_full** and include all columns of **agent_types**.
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

        
