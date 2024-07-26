import pandas as pd
import numpy as np
import os


class splitting_template():
    def __init__(self, data_set, test_part = 0.2, repetition = (0), train_pert = False, test_pert = False, train_on_test = False):
        # Save data_set location
        self.data_set = data_set
        if self.data_set is not None:
            self.Domain = self.data_set.Domain
        
        # Set the part of datasets sorted into test sets
        self.test_part = min(1.0, max(0.0, test_part))
        
        # Filter out usable repetitions
        assert len(repetition) > 0
        self.repetition = []
        for rep in repetition:
            # Avoid duplications
            if rep in self.repetition:
                continue
            
            # Check type 
            if isinstance(rep, int):
                self.repetition.append(max(0,rep))
            elif isinstance(rep, str):
                if self.can_process_str_repetition():
                    rep_numbers = self.tranform_str_to_number(rep)
                    error_str = 'self.transform_str_to_number should return a list of integers.'
                    assert all((isinstance(x, np.int64) or isinstance(x, int)) for x in rep_numbers), error_str
                    self.repetition += rep_numbers
            else:
                raise TypeError('Repetitions must be integers or strings.')
        
        # Get max number of repetition 
        self.max_max_rep = 10
        max_rep = self.repetition_number()
        if max_rep is None:
            max_rep = self.max_max_rep
        else:
            max_rep = min(self.max_max_rep, max_rep)
            
        self.repetition = np.unique(self.repetition).astype(int)
        self.repetition = list(self.repetition[self.repetition < max_rep])

        # Check for pertubation if they exist
        if train_pert or test_pert:
            if not self.Domain.perturbation.any():
                raise AttributeError('The domain does not contain any perturbed data.')
            
        self.train_pert    = train_pert
        self.test_pert     = test_pert
        self.train_on_test = train_on_test
       
    def split_data(self):
        self.split_file = self.data_set.change_result_directory(self.data_set.data_file, 'Splitting', 
                                                                self.get_name()['file'] + self.get_rep_str())
        
        # Reset and reassmble the dataset
        self.data_set.reset(deep = False)
        self.data_set.assemble_data(self.split_file)
        
        # Load the splitting if it exists
        if os.path.isfile(self.split_file):
            [self.Train_index, 
             self.Test_index, _] = np.load(self.split_file, allow_pickle = True)
        else:
            self.Train_index, self.Test_index = self.split_data_method()
             
            # check if split was successful
            if not all([hasattr(self, attr) for attr in ['Train_index', 'Test_index']]):
                 raise AttributeError("The splitting into train and test set was unsuccesful")
            
            # Use unique indices
            self.Train_index = np.unique(self.Train_index)
            self.Test_index  = np.unique(self.Test_index)
            
            # Random shuffle
            np.random.shuffle(self.Train_index)
            np.random.shuffle(self.Test_index)
            
            # Save the split
            save_data = np.array([self.Train_index, 
                                    self.Test_index,
                                    0], object) #0 is there to avoid some numpy load and save errors
            
            
            os.makedirs(os.path.dirname(self.split_file), exist_ok=True)
            np.save(self.split_file, save_data)
        
        # Check if the dataset needs to be adjusted because of perturbations
        self.check_perturbations()
        
        # If we train on test set, add test set to training set
        if self.train_on_test:
            combined_index = np.concatenate([self.Train_index, self.Test_index])
            
            _, return_index = np.unique(combined_index, return_index = True)
            return_index = np.sort(return_index)
            self.Train_index = combined_index[return_index]
    
    def check_perturbations(self):
        # Overwrite the respective File location for the perturbed data
        perturbed = self.Domain.perturbation
        train_perturbed = perturbed & np.in1d(np.arange(len(perturbed)), self.Train_index)
        test_perturbed  = perturbed & np.in1d(np.arange(len(perturbed)), self.Test_index)
            
        # Check if there is data that needs to be both perturbed and unperturbed
        if self.train_pert != self.test_pert:
            needed_both = train_perturbed & test_perturbed
            
            if np.any(needed_both):
                # Duplicate the needed samples and then adjust the corresponding indices
                needed_indices_old = np.where(needed_both)[0]
                needed_indices_new = np.arange(len(needed_indices_old)) + len(self.Domain)
                
                # Remove needed_indices_old from Test index and replace them with needed_indices_new
                overwrite_test = np.in1d(self.Test_index, needed_indices_old)
                self.Test_index[overwrite_test] = needed_indices_new
                    
                ## Duplicate the data
                # Domain (pandas)
                Domain_new = self.Domain.iloc[needed_indices_old].copy()
                Domain_new.index = needed_indices_new
                self.data_set.Domain = pd.concat([self.Domain, Domain_new])
                self.Domain = self.data_set.Domain
                
                # Pred_agents_eval_all (numpy)
                Pred_agents_eval_all_new = self.data_set.Pred_agents_eval_all[needed_indices_old]
                self.data_set.Pred_agents_eval_all = np.concatenate([self.data_set.Pred_agents_eval_all, Pred_agents_eval_all_new])
                
                # Pred_agents_pred_all (numpy)
                Pred_agents_pred_all_new = self.data_set.Pred_agents_pred_all[needed_indices_old]
                self.data_set.Pred_agents_pred_all = np.concatenate([self.data_set.Pred_agents_pred_all, Pred_agents_pred_all_new])
                
                # Not_pov_agent (numpy)
                Not_pov_agent_new = self.data_set.Not_pov_agent[needed_indices_old]
                self.data_set.Not_pov_agent = np.concatenate([self.data_set.Not_pov_agent, Not_pov_agent_new])
                
                if self.data_set.data_in_one_piece:
                    # Input_prediction (pandas)
                    Input_prediction_new = self.data_set.Input_prediction.loc[needed_indices_old].copy()
                    Input_prediction_new.index = needed_indices_new
                    self.data_set.Input_prediction = pd.concat([self.data_set.Input_prediction, Input_prediction_new])
                    
                    # Input_path (pandas)
                    Input_path_new = self.data_set.Input_path.loc[needed_indices_old].copy()
                    Input_path_new.index = needed_indices_new
                    self.data_set.Input_path = pd.concat([self.data_set.Input_path, Input_path_new])
                    
                    # Input_T (numpy)
                    Input_T_new = self.data_set.Input_T[needed_indices_old]
                    self.data_set.Input_T = np.concatenate([self.data_set.Input_T, Input_T_new])
                    
                    # Output_path (pandas)
                    Output_path_new = self.data_set.Output_path.loc[needed_indices_old].copy()
                    Output_path_new.index = needed_indices_new
                    self.data_set.Output_path = pd.concat([self.data_set.Output_path, Output_path_new])
                    
                    # Output_T (numpy)
                    Output_T_new = self.data_set.Output_T[needed_indices_old]
                    self.data_set.Output_T = np.concatenate([self.data_set.Output_T, Output_T_new])
                    
                    # Output_T_pred (numpy)
                    Output_T_pred_new = self.data_set.Output_T_pred[needed_indices_old]
                    self.data_set.Output_T_pred = np.concatenate([self.data_set.Output_T_pred, Output_T_pred_new])
                    
                    # Output_A (pandas)
                    Output_A_new = self.data_set.Output_A.loc[needed_indices_old].copy()
                    Output_A_new.index = needed_indices_new
                    self.data_set.Output_A = pd.concat([self.data_set.Output_A, Output_A_new])
                    
                    # Output_T_E (numpy)
                    Output_T_E_new = self.data_set.Output_T_E[needed_indices_old]
                    self.data_set.Output_T_E = np.concatenate([self.data_set.Output_T_E, Output_T_E_new])
                    
                    # Type (pandas)
                    Type_new = self.data_set.Type.loc[needed_indices_old].copy()
                    Type_new.index = needed_indices_new
                    self.data_set.Type = pd.concat([self.data_set.Type, Type_new])
                    
                    # Recorded (pandas)
                    Recorded_new = self.data_set.Recorded.loc[needed_indices_old].copy()
                    Recorded_new.index = needed_indices_new
                    self.data_set.Recorded = pd.concat([self.data_set.Recorded, Recorded_new])
                
        
        # Overwrite the respective File location for the perturbed data
        perturbed = self.Domain.perturbation
        train_perturbed = perturbed & np.in1d(np.arange(len(perturbed)), self.Train_index)
        test_perturbed  = perturbed & np.in1d(np.arange(len(perturbed)), self.Test_index)
        
        # Get data that needs be overridden
        overwrite_with_unperturbed = np.zeros(len(perturbed), dtype = bool)
        if not self.train_pert:
            overwrite_with_unperturbed |= train_perturbed
        if not self.test_pert:
            overwrite_with_unperturbed |= test_perturbed
        
        if np.any(overwrite_with_unperturbed):
            index_overwrite = np.where(overwrite_with_unperturbed)[0]
            
            # Overwrite the respective file location
            Files_used = self.Domain.file_index.iloc[index_overwrite]
            
            for file_index in np.unique(Files_used):
                # Get indices
                index_overwrite_local = np.where(Files_used == file_index)[0]
                index_local = index_overwrite[index_overwrite_local]
                
                # Get the current file name
                file = self.data_set.Files[file_index]
                
                # Check that this is indeed a perturbed file
                assert '--Pertubation' in file, 'This file is not a perturbed file.'
                
                pre_perturbation, post_perturbation = file.split('--Pertubation')
                file_unperturbed = pre_perturbation + post_perturbation[4:]
                

                # Check if unperturbed file is in file list
                if file_unperturbed in self.data_set.Files:
                    file_index_unperturbed = self.Domain.Files.index(file_unperturbed)

                    # Overwrite the perturbed file
                    self.Domain.file_index.iloc[index_local] = file_index_unperturbed
                    
                else:
                    # Get new file index
                    self.data_set.Files.append(file_unperturbed)
                    self.Domain.file_index.iloc[index_local] = len(self.data_set.Files) - 1

                
                # Get perturbed scenario name
                scenario = self.Domain.iloc[index_local].Scenario
                assert len(np.unique(scenario)) == 1, 'All scenarios should be the same.'
                scenario = scenario.iloc[0]

                # Get the unperturbed scenario name
                assert ' (Pertubation_' in scenario, 'This scenario is not a perturbed scenario.'
                scenario_unperturbed = scenario.split(' (Pertubation_')[0]

                # Check if unperturbed scenario allready exists
                if scenario_unperturbed not in self.Domain.Scenario:
                    assert scenario_unperturbed not in self.data_set.Datasets.keys(), 'This scenario is allready in the dataset.'
                    
                    # Load the corresponding dataset
                    perturbed_dataset = self.data_set.Datasets[scenario]
                    data_set_unperturbed = self.get_new_dataset(perturbed_dataset)
                    self.data_set.Datasets[scenario_unperturbed] = data_set_unperturbed

                # Overwrite the scenario name
                self.Domain.Scenario.iloc[index_local] = scenario_unperturbed

                # If we can load complete dataset, we will also overwrite this
                if self.data_set.data_in_one_piece:
                    # Load the respective input and output data
                    data_file_unperturbed = file_unperturbed + '_data.npy'
                    [_, Input_path, _, Output_path, _, _, Output_A, Output_T_E, _] = np.load(data_file_unperturbed, allow_pickle=True)
                    
                    # Apply the local index
                    used_index = self.Domain.iloc[index_local].Index_saved
                    Input_path  = Input_path.loc[used_index]
                    Output_path = Output_path.loc[used_index]
                    Output_A    = Output_A.loc[used_index]
                    Output_T_E  = Output_T_E[used_index]
                    
                    # Overwrite the data
                    index_local_loc = self.Domain.iloc[index_local].index
                    self.data_set.Input_path.loc[index_local_loc, Input_path.columns]   = Input_path
                    self.data_set.Output_path.loc[index_local_loc, Output_path.columns] = Output_path
                    self.data_set.Output_A.loc[index_local_loc, Output_A.columns]       = Output_A 
                    self.data_set.Output_T_E[index_local]                               = Output_T_E
            
            # Set overwritten files to status unperturbed          
            self.Domain.perturbation.iloc[overwrite_with_unperturbed] = False

    def get_new_dataset(self, perturbed_dataset):
        # Get the dataset class
        data_set_class = perturbed_dataset.__class__

        # Initialize unperturbed dataset
        Perturbation = None
        parameters = [self.data_set.model_class_to_path,
                      self.data_set.num_samples_path_pred,
                      self.data_set.enforce_num_timesteps_out,
                      self.data_set.enforce_prediction_time,
                      self.data_set.exclude_post_crit,
                      self.data_set.allow_extrapolation,
                      self.data_set.agents_to_predict,
                      self.data_set.overwrite_results]
        
        data_set = data_set_class(Perturbation, *parameters)

        # Get data for unperturbed dataset
        data_set.get_data(perturbed_dataset.dt,
                          (perturbed_dataset.num_timesteps_in_real, perturbed_dataset.num_timesteps_in_need),
                          (perturbed_dataset.num_timesteps_out_real, perturbed_dataset.num_timesteps_out_need))

        return data_set

    
    def get_rep_str(self):
        # Save repetition as string
        dig = len(str(self.max_max_rep - 1))
        rep_str = '_'.join([str(rep).zfill(dig) for rep in self.repetition])

        # Save the train_on_test usage as string
        if self.train_on_test:
            rep_str += '_tot'

        # Save pertubation usage as string
        rep_str += '_pert=' + str(int(self.train_pert)) + str(int(self.test_pert))
        return '_' + rep_str

    def check_splitability(self):            
        if len(self.repetition) == 0:
            return 'this splitting method has no viable repetitions.'
        
        return self.check_splitability_method()  
    
    
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                      Splitting method dependend functions                         ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    def get_name(self):
        r'''
        Provides a dictionary with the different names of the splitting method
            
        Returns
        -------
        names : dict
          The first key of names ('print')  will be primarily used to refer to the splitting method in console outputs. 
                
          The 'file' key has to be a string with exactly **12 characters**, that does not include any folder separators 
          (for any operating system), as it is mostly used to indicate that certain result files belong to this splitting
          method. 
                
          The 'latex' key string is used in automatically generated tables and figures for latex and can include 
          latex commands - such as using '$$' for math notation.
            
        '''
        raise AttributeError('Has to be overridden in actual method.')
        
    def split_data_method(self):
        # this function takes the given input and then creates a 
        # split according to a desied method
        # creates:
            # self.Train_index -    A 1D numpy including the samples IDs of the training set
            # self.Test_index -     A 1D numpy including the samples IDs of the test set
        raise AttributeError('Has to be overridden in actual method.')
    
    def check_splitability_method(self):
        # Provides feedback on if a splitting method can be used, as it might be
        # related to only specifc datasets/scenarios/etc.
        raise AttributeError('Has to be overridden in actual method.') 
        
    def repetition_number(self):
        # Provides the number of repetitions that the model requires
        raise AttributeError('Has to be overridden in actual method.') 
        
    def can_process_str_repetition(self = None):
        # Decides whether the splitting method can take string repetitions
        raise AttributeError('Has to be overridden in actual method.') 
        
    def tranform_str_to_number(self, rep_str):
        # Given a string, returns the actual number that would have been
        # used if the string was not given
        raise AttributeError('Has to be overridden in actual method.') 
