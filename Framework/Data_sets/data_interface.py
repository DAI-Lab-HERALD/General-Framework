import numpy as np
import pandas as pd
import importlib
import psutil
import os
import warnings
import networkx as nx
import scipy as sp
from utils.memory_utils import get_total_memory, get_used_memory

from rome.ROME import ROME

class data_interface(object):
    def __init__(self, data_set_dict, parameters):
        # Initialize path
        self.path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
        
        # Borrow dataset paprameters
        self.model_class_to_path       = parameters[0]
        self.num_samples_path_pred     = parameters[1]
        self.enforce_num_timesteps_out = parameters[2]
        self.enforce_prediction_time   = parameters[3]
        self.exclude_post_crit         = parameters[4]
        self.allow_extrapolation       = parameters[5]
        self.agents_to_predict         = parameters[6]
        self.overwrite_results         = parameters[7]
        self.allow_longer_predictions  = parameters[8]
        self.save_predictions          = parameters[9]
        self.total_memory              = parameters[10]
        
        # Remove total memory from parameters
        parameters = [parameters[i] for i in range(len(parameters) - 1)]

        if isinstance(data_set_dict, dict):
            data_set_dict = [data_set_dict]
        else:
            assert isinstance(data_set_dict, list), "To combine datasets, put the dataset dictionaries into lists."
        
        # Initialize datasets
        self.Datasets = {}
        self.Latex_names = []
        for i_data, data_dict in enumerate(data_set_dict):
            assert isinstance(data_dict, dict), "Dataset is not provided as a dictionary."
            assert 'scenario' in data_dict.keys(), "Dataset name is missing  (required key: 'scenario')."
            assert 't0_type' in data_dict.keys(), "Prediction time is missing  (required key: 't0_type')."

            # Check if theis is supposed to be a perturbed dataset
            if 'perturbation' in data_dict.keys():
                perturbation = data_dict['perturbation']
                # Check if perturbation is in required format
                assert isinstance(perturbation, dict), "Desired perturbation is not provided as a dictionary."
                
                # Check for valid keys
                assert 'attack' in perturbation.keys(), "Perturbation attack type is missing (required key: 'attack')."

                # Pass the experiment parameters to perturbation method
                perturbation['exp_parameters'] = parameters + [self.total_memory]
                
                # Get perturbation type
                pert_name = perturbation['attack']

                pert_module = importlib.import_module(pert_name)
                pert_class = getattr(pert_module, pert_name)

                Perturbation = pert_class(perturbation)

            else:
                Perturbation = None
            
            data_set_name = data_dict['scenario']
            t0_type       = data_dict['t0_type']
            if 'conforming_t0_types' in data_dict.keys():
                Comp_t0_types = data_dict['conforming_t0_types']
            else:
                Comp_t0_types = []
            
            if 'max_num_agents' in data_dict.keys():
                max_num_agents = data_dict['max_num_agents']
                if max_num_agents is not None:
                    assert isinstance(max_num_agents, int), "the maximum number of agents must be either None or an integer."
            else:
                max_num_agents = None
            
            if t0_type in Comp_t0_types:
                T0_type_compare = list(Comp_t0_types).remove(t0_type)
            else:
                T0_type_compare = []
                
            data_set_module = importlib.import_module(data_set_name)
            data_set_class = getattr(data_set_module, data_set_name)
            
            parameters_pass = [parameters[i] for i in range(len(parameters) - 1)] + [self.total_memory]
            data_set = data_set_class(Perturbation, *parameters_pass)

            data_set.set_extraction_parameters(t0_type, T0_type_compare, max_num_agents)
            
            latex_name = data_set.get_name()['latex']
            if latex_name[:6] == r'\emph{' and latex_name[-1] == r'}':
                latex_name = latex_name[6:-1]
                
            self.Latex_names.append(latex_name) 

            data_set_name = 'data_set_' + str(i_data)

            self.Datasets[data_set_name] = data_set
            if i_data == 0:
                self.data_set_under = data_set
        
        self.num_datasets   = len(self.Datasets.values())
        self.single_dataset = self.num_datasets <= 1    
        
        # Get relevant scenario information
        scenario_names = []
        scenario_general_input = []
        
        scenario_pov_agents = np.empty(self.num_datasets, object)
        scenario_needed_agents = np.empty(self.num_datasets, object)
        
        self.classification_possible = False
        self.scenario_behaviors = []
        self.Behaviors = []
        for i, data_set in enumerate(self.Datasets.values()):
            # Get classifiable behavior
            self.scenario_behaviors.append(data_set.Behaviors)
            self.Behaviors = self.Behaviors + list(data_set.Behaviors)
            
            # Get pov anf needed agents 
            scenario_names.append(data_set.scenario.get_name())
            scenario_general_input.append(data_set.scenario.can_provide_general_input() is not None)
            pov_agent = data_set.scenario.pov_agent()
            cl_agents = data_set.scenario.classifying_agents()
            
            
            if pov_agent is not None:
                needed_agents = [pov_agent] + cl_agents
            else:
                needed_agents = cl_agents
            
            scenario_pov_agents[i] = pov_agent
            scenario_needed_agents[i] = needed_agents
            
            self.classification_possible = self.classification_possible or data_set.classification_useful 
            
        self.Behaviors = np.unique(self.Behaviors)
        
        self.unique_scenarios, scenario_index = np.unique(scenario_names, return_index = True) 
        self.scenario_pov_agents = scenario_pov_agents[scenario_index]
        self.scenario_needed_agents = scenario_needed_agents[scenario_index]
        
        self.classification_useful = len(self.unique_scenarios) == 1 and self.classification_possible
    
        # Check if general input is possible
        self.general_input_available = all(scenario_general_input) and self.classification_useful
        
        # get scenario name
        if len(self.unique_scenarios) == 1:
            self.scenario_name = self.unique_scenarios[0]
        else:
            self.scenario_name = 'Combined scenarios'
        
        # Get a needed name:
        if len(np.unique([data_set.t0_type for data_set in self.Datasets.values()])) == 1:
            self.t0_type = self.data_set_under.t0_type
        else:
            self.t0_type = 'mixed'
            
        self.p_quantile = self.data_set_under.p_quantile
        
        
        max_num_agents = np.array([data_set.max_num_agents for data_set in self.Datasets.values()])
        if np.all(max_num_agents == None):
            self.max_num_agents = None
        else:        
            max_num_agents = max_num_agents[max_num_agents != None]
            self.max_num_agents = max_num_agents.min()
        
        self.data_loaded = False

    # Get the path_data_info function in position
    def unique_data_paths(self):
        path_data_type = ['x', 'y']

        # Look through other datasets for other information
        Other_type = []
        Other_type_combined = []
        for data_set in self.Datasets.values():
            data_set_type = np.array(data_set.path_data_info())
            assert list(data_set_type[:2]) == path_data_type, 'Path data information is not consistent.'
            other_type = list(data_set_type[2:])
            Other_type.append(other_type)
            Other_type_combined.append(''.join(other_type))

        _, index, indices = np.unique(Other_type_combined, return_index = True, return_inverse = True)

        # Get unique types
        Unique_types = [path_data_type + Other_type[i] for i in index]
        return Unique_types, indices

    
    def reset(self, deep = True):
        if deep:
            for data_set in self.Datasets.values():
                data_set.reset()
        
        self.data_loaded = False
        
        # Delete extracted data
        if hasattr(self, 'X_orig') and hasattr(self, 'Y_orig'):
            del self.X_orig
            del self.Y_orig
        if hasattr(self, 'orig_file_index'):
            del self.orig_file_index
        
        if hasattr(self, 'Pred_agents_eval_all') and hasattr(self, 'Pred_agents_pred_all'):
            del self.Pred_agents_eval_all
            del self.Pred_agents_pred_all
            
        if hasattr(self, '_checked_pred_agents'):
            del self._checked_pred_agents
            
        if hasattr(self, 'Not_pov_agent'):
            del self.Not_pov_agent
        
        if hasattr(self, 'Subgroups') and hasattr(self, 'Path_true_all'):
            del self.Subgroups
            del self.Path_true_all
            
        if hasattr(self, 'Log_prob_true_joint'):
            del self.Log_prob_true_joint
            del self.KDE_joint
            
        if hasattr(self, 'Log_prob_true_indep'):
            del self.Log_prob_true_indep
            del self.KDE_indep
    

    def change_result_directory(self, filepath, new_path_addon, new_file_addon = '', file_type = '.npy'):
        return self.data_set_under.change_result_directory(filepath, new_path_addon, new_file_addon, file_type)
    
    
    def determine_required_timesteps(self, num_timesteps):
        return self.data_set_under.determine_required_timesteps(num_timesteps)
    
    
    def set_data_file(self, dt, num_timesteps_in, num_timesteps_out):
        (self.num_timesteps_in_real, 
        self.num_timesteps_in_need)  = self.determine_required_timesteps(num_timesteps_in)
        (self.num_timesteps_out_real, 
        self.num_timesteps_out_need) = self.determine_required_timesteps(num_timesteps_out)
        if self.single_dataset:
            data_file = self.data_set_under.data_params_to_string(dt, num_timesteps_in, num_timesteps_out)

            self.data_file = data_file[:-4]

        else:
            # Get data_file from every constituent dataset
            self.data_file = (self.data_set_under.path + os.sep + 
                              'Results' + os.sep +
                              self.get_name()['print'] + os.sep +
                              'Data' + os.sep + self.get_name()['file'])

            Data_files = []
            max_len = 0
            for data_set in self.Datasets.values():
                data_file = data_set.data_params_to_string(dt, num_timesteps_in, num_timesteps_out)
                Data_files.append(data_file.split(os.sep)[-1])
                max_len = max(max_len, len(data_file.split('--')))
            
            Data_files_array = np.zeros((len(Data_files), max_len - 1), dtype = object)
            for i, data_file in enumerate(Data_files):
                strs = data_file[:-4].split('--')[1:]
                Data_files_array[i, :len(strs)] = strs
            Data_files_array = Data_files_array.astype(str)
            unique_parts = np.unique(Data_files_array)
            # Check if there is '0' in the unique parts
            if '0' in unique_parts:
                unique_parts = unique_parts[unique_parts != '0']

            # Find the parts with 't0'
            t0_parts = unique_parts[np.array(['t0' == s[:2] for s in unique_parts])]

            if len(t0_parts) == 1:
                self.data_file += '--' + t0_parts[0]
            else:
                # Assert that last two letters are the same
                assert np.all([s[-2:] == t0_parts[0][-2:] for s in t0_parts[1:]])

                # Check if all t0_parts start with 'all
                if np.all([s[:3] == 'all' for s in t0_parts]):
                    self.data_file += '--all_m_' + t0_parts[0][-2:]
                else:
                    self.data_file += '--mixed_' + t0_parts[0][-2:]

            # Find the parts with 'dt'
            dt_parts = unique_parts[np.array(['dt' == s[:2] for s in unique_parts])]
            assert len(dt_parts) == 1
            self.data_file += '--' + dt_parts[0]

            # Add the maximum number of agents
            if self.max_num_agents is None:
                num = 0 
            else:
                num = self.max_num_agents
            self.data_file += '--max_' + str(num).zfill(3)


            if 'No_extrap' in unique_parts:
                self.data_file += '--No_Extrap'

            # Look for perturbation parts
            pert_parts = []
            for i in range(Data_files_array.shape[0]):
                includes_pert = False
                for j in range(Data_files_array.shape[1]):
                    s = Data_files_array[i][j]
                    if 'Pertubation' == s[:11]:
                        pert_parts.append(s)
                        includes_pert = True
                if not includes_pert:
                    pert_parts.append('')

            pert_parts = unique_parts[np.array(['Pertubation' == s[:11] for s in unique_parts])]
            if not np.all(pert_parts == ''):
                useful_pert_parts = pert_parts[pert_parts != '']
                if len(useful_pert_parts) > 1:
                    self.data_file += '--Perturbations_(' 
                    for i, pert_part in enumerate(useful_pert_parts):
                        self.data_file += pert_part[12:] 
                        if i < len(useful_pert_parts) - 1:
                            self.data_file += '_'
                    self.data_file += ')'
                elif len(useful_pert_parts) == 1:
                    self.data_file += '--' + useful_pert_parts[0]
            

        # Add the prediction type            
        if self.agents_to_predict == 'predefined':
            pat = '0'
        elif self.agents_to_predict == 'all':
            pat = 'A'
        else:
            pat = self.agents_to_predict[0]

        self.data_file = self.data_file + '--agents_' + pat

        self.data_file += '.npy'
    

    def _extract_save_files_from_data_set(self, data_set):
        additions = data_set.Domain[['path_addition', 'data_addition']].to_numpy().sum(-1)
        unique_additions, file_number = np.unique(additions, return_inverse = True)
        
        Files = []
        data_file = data_set.data_file[:-4]
        for unique_addition in unique_additions:
            Files.append(data_file + unique_addition)
            
        return Files, file_number
    

    def set_default_size(self, Type):
        Size = pd.DataFrame(columns=Type.columns, index=Type.index, dtype = object)

        default_dict = {
            'V': np.array([5.0, 2.0]),
            'M': np.array([2.0, 0.5]),
            'B': np.array([2.0, 0.5]),
            'P': np.array([0.5, 0.5]),
        }
        T_array = Type.to_numpy()

        for typ, size in default_dict.items():
            useful = T_array == typ
            size_array = np.array([size] * useful.sum() + ['helper'], dtype = object)[:-1]
            Size.values[useful] = size_array

        return Size

    
    def assemble_data(self, file_identifier, keep_useless_samples = False):
        self.Files = []
        self.Agents = []
        
        self.assembled_data_file = file_identifier
        
        # Prepare the information needed here
        self.Domain = pd.DataFrame(np.zeros((0,0), np.ndarray))
        self.num_behaviors = pd.Series(np.zeros(len(self.Behaviors), int), index = self.Behaviors)
        self.num_behaviors_out = pd.Series(np.zeros(len(self.Behaviors), int), index = self.Behaviors)
        
        # If possible, also load other data into one piece
        if self.data_in_one_piece:
            # Set up 12 required attributes resulting from this function
            self.Input_prediction = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Input_path       = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Input_T          = np.zeros(0, np.ndarray)

            self.Output_path      = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Output_T         = np.zeros(0, np.ndarray)
            self.Output_T_pred    = np.zeros(0, np.ndarray)
            self.Output_A         = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Output_T_E       = np.zeros(0, float)

            self.Type             = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Size             = pd.DataFrame(np.zeros((0,0), np.ndarray))
            self.Recorded         = pd.DataFrame(np.zeros((0,0), np.ndarray))

        
        
        for i_data_set, data_set in enumerate(self.Datasets.values()):
            # Save file indices
            data_set_files, file_index_local = self._extract_save_files_from_data_set(data_set)
            
            # Remove the addition from the domai
            Domain_local = data_set.Domain.drop(['data_addition'], axis = 1)
            Domain_local['file_index'] = file_index_local + len(self.Files)

            # Save the data type index
            Domain_local['data_type_index'] = self.Data_set_inverse[i_data_set]
            
            # Add the new files 
            self.Files += data_set_files
            assert len(self.Files) == (max(Domain_local['file_index']) + 1), 'File index is not correctly assigned.'

            # Add the old index of the local domain
            Domain_local['Index_intern'] = np.arange(len(Domain_local))

            # Adjust the file Index
            self.Domain = pd.concat((self.Domain, Domain_local))
            
            # Combine number of behaviors
            self.num_behaviors[data_set.Behaviors]     += data_set.num_behaviors 
            self.num_behaviors_out[data_set.Behaviors] += data_set.num_behaviors_out
            
            # Get new agents
            self.Agents += data_set.Agents
            
            if self.data_in_one_piece:
                # Always free up memory if possible
                # Consider generalized input
                if self.general_input_available:
                    self.Input_prediction = pd.concat((self.Input_prediction, data_set.Input_prediction))
                else:
                    IP = pd.DataFrame(np.ones((len(data_set.Output_T), 1), float) * np.nan, columns = ['empty'])
                    self.Input_prediction = pd.concat((self.Input_prediction, IP))

                self.Input_path       = pd.concat((self.Input_path, data_set.Input_path))
                self.Input_T          = np.concatenate((self.Input_T, data_set.Input_T), axis = 0)
                
                self.Output_path      = pd.concat((self.Output_path, data_set.Output_path))
                self.Output_T_pred    = np.concatenate((self.Output_T_pred, data_set.Output_T_pred), axis = 0)
                self.Output_T         = np.concatenate((self.Output_T, data_set.Output_T), axis = 0)
                self.Output_T_E       = np.concatenate((self.Output_T_E, data_set.Output_T_E), axis = 0)
                self.Output_A         = pd.concat((self.Output_A, data_set.Output_A))
                
                self.Type             = pd.concat((self.Type, data_set.Type))
                self.Recorded         = pd.concat((self.Recorded, data_set.Recorded))

                # Get data_set size
                if data_set.Size is not None:
                    data_set_size = data_set.Size
                else:
                    data_set_size = self.set_default_size(data_set.Type)
                self.Size = pd.concat((self.Size, data_set_size))
        
        # combine the agents
        self.Agents, index = np.unique(self.Agents, return_index = True)
        # Sort the agents, so that the entry appearing first is the one that is kept first
        self.Agents = list(self.Agents[np.argsort(index)])    
        
        if self.data_in_one_piece:
            # Ensure agent order is the same everywhere
            self.Input_path  = self.Input_path[self.Agents]
            self.Output_path = self.Output_path[self.Agents]
            self.Type        = self.Type[self.Agents]
            self.Size        = self.Size[self.Agents]
            self.Recorded    = self.Recorded[self.Agents]
        
            # Ensure behavior stuff is aligned and fill missing behaviors
            self.Output_A = self.Output_A.reindex(columns = self.Behaviors).fillna(False)
        
        # Set final values
        self.data_loaded = True
        
        # Remove useless samples
        self._determine_pred_agents_unchecked()
        Num_eval_agents = (self.Pred_agents_eval_all & self.Not_pov_agent).sum(1)
        Num_pred_agents = (self.Pred_agents_pred_all & self.Not_pov_agent).sum(1)
        
        Useful_agents = (Num_eval_agents + Num_pred_agents) > 0 
        
        if Useful_agents.sum() < 5:
            complete_failure = "not enough prediction problems are avialable."
        else:
            complete_failure = None
        
        # Overwrite
        if not keep_useless_samples:
            self.Domain           = self.Domain.iloc[Useful_agents].reset_index(drop = True)
            self.Pred_agents_eval_all = self.Pred_agents_eval_all[Useful_agents]
            self.Pred_agents_pred_all = self.Pred_agents_pred_all[Useful_agents]
            self.Not_pov_agent = self.Not_pov_agent[Useful_agents]
            
            if self.data_in_one_piece:
                self.Input_prediction = self.Input_prediction.iloc[Useful_agents].reset_index(drop = True)
                self.Input_path       = self.Input_path.iloc[Useful_agents].reset_index(drop = True)
                self.Input_T          = self.Input_T[Useful_agents]

                self.Output_path      = self.Output_path.iloc[Useful_agents].reset_index(drop = True)
                self.Output_T         = self.Output_T[Useful_agents]
                self.Output_T_pred    = self.Output_T_pred[Useful_agents]
                self.Output_A         = self.Output_A.iloc[Useful_agents].reset_index(drop = True)
                self.Output_T_E       = self.Output_T_E[Useful_agents]

                self.Type             = self.Type.iloc[Useful_agents].reset_index(drop = True)
                self.Size             = self.Size.iloc[Useful_agents].reset_index(drop = True)
                self.Recorded         = self.Recorded.iloc[Useful_agents].reset_index(drop = True)
        
        return complete_failure
    
    
    def get_data(self, dt, num_timesteps_in, num_timesteps_out, keep_useless_samples = False):
        # Save dt
        self.dt = dt
        
        self.set_data_file(dt, num_timesteps_in, num_timesteps_out)
        complete_failure = None
        self.data_in_one_piece = True
        new_keys = []
        for data_set in self.Datasets.values():
            data_failure = data_set.get_data(dt, num_timesteps_in, num_timesteps_out)
            if data_failure is not None:
                complete_failure = data_failure[:-1] + ' (in ' + data_set.get_name()['print'] + ').'
            
            # Check the number of save parts in the dataset
            if data_set.number_data_files > 1:
                self.data_in_one_piece = False

            # get scenario name from the Domain there
            scenario_name = data_set.Domain.Scenario.iloc[0]
            new_keys.append(scenario_name)

        # Ocerwrite the keys in self.Datasets with new keys
        self.Datasets = dict(zip(new_keys, self.Datasets.values()))

        # Get the unique data types included
        self.Input_data_type, self.Data_set_inverse = self.unique_data_paths()

        # Data_set can only be in one piece if all the path data is of the same type
        if len(self.Input_data_type) > 1:
            self.data_in_one_piece = False
                
        # Check for memory
        if self.data_in_one_piece:
            available_memory = self.total_memory - get_used_memory()
            needed_memory = 0
            for data_set in self.Datasets.values():
                data_file_final = data_set.data_file[:-4] + '_LLL_LLL_data.npy'
                needed_memory += os.path.getsize(data_file_final)

            if needed_memory > available_memory * 0.6:
                self.data_in_one_piece = False
        
        complete_failure = self.assemble_data(self.data_file, keep_useless_samples)
        
        return complete_failure
        
        
    
    def get_Target_MeterPerPx(self, domain): 
        assert self.data_loaded, 'Data has not been loaded yet.'
        return self.Datasets[domain.Scenario].Images.Target_MeterPerPx.loc[domain.image_id]
    
    
    def return_batch_images(self, domain, center, rot_angle, target_width, target_height, 
                            grayscale = False, return_resolution = False, print_progress = False):
        
        if target_height is None:
            target_height = 1250
        if target_width is None:
            target_width = 1250
            
        if grayscale:
            Imgs = np.zeros((len(domain), target_height, target_width, 1), dtype = 'uint8')
        else:
            Imgs = np.zeros((len(domain), target_height, target_width, 3), dtype = 'uint8')
        
        Imgs.fill(0)   
        
        if return_resolution:
            Imgs_m_per_px = np.zeros(len(domain), dtype = 'float64')
        
        for data_set in self.Datasets.values():
            if data_set.includes_images():
                # Find the indices of the samples coming from this dataset
                data_set_file = os.path.basename(data_set.data_file[:-4])
                data_set_name = data_set.get_name()['print']
                if 'Pertubation' in data_set_file:
                    pert_index = data_set_file.split('Pertubation_')[-1]
                    data_set_name += ' (Pertubation_' + pert_index + ')'
                Use = (domain['Scenario'] == data_set_name).to_numpy()
                
                # Ignore if no images from dataset are used
                if not Use.any():
                    continue
                
                if print_progress:
                    print('')
                    print('Rotate images from dataset ' + data_set.get_name()['print'])

                if center is None:
                    center_use = None
                else:
                    center_use = center[Use]
                    
                if rot_angle is None:
                    rot_angle_use = None
                else:
                    rot_angle_use = rot_angle[Use]
                
                Index_use = np.where(Use)[0]
                Imgs = data_set.return_batch_images(domain.iloc[Use], center_use, rot_angle_use, 
                                                    target_width, target_height, grayscale, 
                                                    Imgs, Index_use, print_progress)
                if return_resolution:
                    Imgs_m_per_px[Use] = data_set.Images.Target_MeterPerPx.loc[domain.image_id.iloc[Use]]
        if return_resolution:
            return Imgs, Imgs_m_per_px
        else:
            return Imgs
        
    
    
    def return_batch_sceneGraphs(self, domain, X, radius = None, wave_length = 1.0, print_progress = False):
        
        SceneGraphs = np.full(len(domain), np.nan, dtype = object)
        
        for data_set in self.Datasets.values():
            if data_set.includes_sceneGraphs():
                if print_progress:
                    print('')
                    print('Get scene graphs from dataset ' + data_set.get_name()['print'])
                
                # Find the indices of the samples coming from this dataset
                data_set_file = os.path.basename(data_set.data_file[:-4])
                data_set_name = data_set.get_name()['print']
                if 'Pertubation' in data_set_file:
                    pert_index = data_set_file.split('Pertubation_')[-1]
                    data_set_name += ' (Pertubation_' + pert_index + ')'
                Use = (domain['Scenario'] == data_set_name).to_numpy()
                
                # Ignore if no scene graphs from dataset are used
                if not Use.any():
                    continue
                                
                Index_use = np.where(Use)[0]
                SceneGraphs = data_set.return_batch_sceneGraphs(domain.iloc[Use], X[Use], radius, wave_length, SceneGraphs, Index_use, print_progress)

        return SceneGraphs
    
    def classify_data(self, Paths, Domain, num_paths, num_timesteps):
        r'''
        Classify the data into the different scenarios.

        Parameters
        ----------
        Paths : pandas.DataFrame
            Paths of the agents. Has shape num_samples x num_agents, which can
            contain if available a trajectory in form of a numpy array of shape
            num_paths x num_timesteps x 2.
        Domain : pandas.DataFrame
            Domain of the data. Contains anxillieary information. length is num_samples.

        Returns
        -------
        P : numpy.ndarray
            Probability of the different scenarios. Shape is num_samples x num_paths x num_classes.
            Uses a one-hot encoding.
        class_names : list
            Names of the classes.
        
        '''

        # get class names
        class_names = self.Behaviors

        # get the number of predicted paths
        P = np.full((len(Paths), num_paths, len(class_names)), np.nan, float)
        

        # get unique datasets in the domain
        for data_set in self.Datasets.values():
            data_set_name = data_set.Domain.Scenario
            assert len(np.unique(data_set_name)) == 1, 'Data set should only have one name.'
            data_set_name = data_set_name.iloc[0]

            # Get predictions from this dataset
            use = Domain.Scenario == data_set_name

            if not use.any():
                continue

            # get old num path predicts
            num_samples_path_pred_old = data_set.num_samples_path_pred
            data_set.num_samples_path_pred = num_paths
            
            use_index = np.where(use.to_numpy())[0]
            beh_index = np.arange(len(data_set.Behaviors))[np.newaxis] # [1 x num_classes_use]
            t = np.arange(1, num_timesteps + 1) * self.dt

            # Get the index of the dataset behviors in class names
            class_index = self.get_indices_1D(data_set.Behaviors, class_names)

            for i in use_index:
                path = Paths.iloc[i]
                domain = Domain.iloc[i]
                try:
                    # Get the class of the path
                    T_class = data_set.path_to_class_and_time_sample(path, t, domain) # [num_paths x num_classes_use]
                    Used_class = T_class.argmin(axis=-1, keepdims=True) # [num_paths x 1]
                    output_A = beh_index == Used_class # [num_paths x num_classes_use]
                    output_A = output_A.astype(float)

                    # save the output
                    p = np.zeros((num_paths, len(class_names)))
                    p[:, class_index] = output_A
                    P[i] = p
                except:
                    print('Error in classifying path in dataset ' + data_set_name + '.')
                

            # reset num_samples_path_pred
            data_set.num_samples_path_pred = num_samples_path_pred_old
        
        return P, class_names
    
    
    def transform_outputs(self, output, model_pred_type, metric_pred_type):
        if model_pred_type == metric_pred_type:
            return output

        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified.")
        
        # Check if this can be handled for a single dataset
        output_index = output[0]
        # Get parts of output
        output_data  = output[1:]


        # Set up empty splitting parts
        output_trans = {}
        output_index_rel = {}
        output_data_set_name = self.Domain.Scenario.iloc[output_index]
        
        # Go through part datasets
        for data_set in self.Datasets.values():
            data_set_name = data_set.Domain.Scenario
            assert len(np.unique(data_set_name)) == 1, 'Data set should only have one name.'
            data_set_name = data_set_name.iloc[0]

            # Get predictions from this dataset
            use = output_data_set_name == data_set_name

            if not use.any():
                continue
            
            # Get the relative position of the index
            use = np.where(use.to_numpy())[0]
            output_index_rel[data_set] = use

            # Get the index of the current samples in the data_set internal indexing
            output_index_dataset = np.array(self.Domain.Index_intern.iloc[output_index[use]])

            # Get output_data for the specific dataset
            output_dataset = [output_index_dataset]
            for out in output_data:
                assert isinstance(out, pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
                out_data = out.iloc[use]

                # check if this is a class thing
                behavior_data = np.in1d(np.array(out_data.columns), self.Behaviors)
                if behavior_data.all():
                    out_data = out_data.reindex(columns = data_set.Behaviors).fillna(0.0)

                agent_data = np.in1d(np.array(out_data.columns), self.Agents)
                if agent_data.all():
                    out_data = out_data.reindex(columns = data_set.Agents).fillna(0.0)

                # Apply output_index_dataset as index to out_data
                out_data.index = pd.Index(output_index_dataset)
            
                output_dataset.append(out_data)

            # Get transfromed outputs for the datasset
            output_trans_dataset = data_set.transform_outputs(output_dataset, model_pred_type, metric_pred_type)
            output_trans[data_set] = output_trans_dataset
        
        ## Reassemble the output
        example_output = list(output_trans.values())[0]
        output_trans_all = [output_index]

        # Go through all parts of the transformed prediction
        for j in range(1, len(example_output)):
            output_trans_j = []
            for data_set in output_trans.keys():
                output_trans_dataset_j = output_trans[data_set][j]
                use = output_index_rel[data_set]

                # set index of output_trans_dataset_j to be use
                assert isinstance(output_trans_dataset_j, pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
                output_trans_dataset_j.index = pd.Index(use)

                behavior_data = np.in1d(np.array(output_trans_dataset_j.columns), self.Behaviors)
                if behavior_data.all():
                    output_trans_dataset_j = output_trans_dataset_j.reindex(columns = self.Behaviors).fillna(0.0)

                agent_data = np.in1d(np.array(output_trans_dataset_j.columns), self.Agents)
                if agent_data.all():
                    output_trans_dataset_j = output_trans_dataset_j.reindex(columns = self.Agents)

                output_trans_j.append(output_trans_dataset_j)
            
            # Combine the output
            output_trans_j = pd.concat(output_trans_j)

            # Sort the dataframe by the index
            output_trans_j = output_trans_j.sort_index()

            # Overwrite the index with original index
            output_trans_j.index = output_data[0].index

            output_trans_all.append(output_trans_j)
        
        return output_trans_all
        
    
    def provide_map_drawing(self, domain):
        assert self.data_loaded, 'Data has not been loaded yet.'
        return self.Datasets[domain.Scenario].provide_map_drawing(domain)
    
    
    def future_input(self):
        return all([data_set.future_input() for data_set in self.Datasets.values()])
        
    
    def includes_images(self):
        return any([data_set.includes_images() for data_set in self.Datasets.values()])
    
    def includes_sceneGraphs(self):
        return any([data_set.includes_sceneGraphs() for data_set in self.Datasets.values()])
    
    def get_name(self):
        if self.single_dataset:
            return self.data_set_under.get_name()
        else:
            file_name = ('Comb_' + '_'.join([s[0] for s in self.Latex_names]) + '_' * 5)[:10]
            print_name = 'Combined dataset (' + r' & '.join(self.Latex_names) + ')'
            print_name = print_name.replace('/', ' & ')
            print_name = print_name.replace(os.sep, ' & ')
            names = {'print': print_name,
                     'file': file_name,
                     'latex': r'/'.join(self.Latex_names)}
            return names

    #############################################################################################################################
    #############################################################################################################################
    #############################################################################################################################

    def get_indices_1D(self, A, B):
        # Gives the indices of A in B
        assert len(A.shape) == 1, 'A should be a 1D array.'
        assert len(B.shape) == 1, 'B should be a 1D array.'

        index = (A[:, np.newaxis] == B[np.newaxis])
        assert np.all(index.sum(1) == 1), 'Agents are not unique.'
        
        return index.argmax(1)

    
    #%% Useful function for later modules
    def _extract_original_trajectories(self, file_index = 0):
        ## NOTE: Method has been adjusted for large datasets
        if hasattr(self, 'X_orig') and hasattr(self, 'Y_orig'):
            # Check if the currently extracted values correspond to the file index
            if hasattr(self, 'orig_file_index'):
                if file_index == self.orig_file_index:
                    return
        else:
            assert not hasattr(self, 'orig_file_index'), 'Original trajectories have been extracted for unknown file index.'
        
        self.orig_file_index = file_index
            
        # Load the specific file
        if self.data_in_one_piece:
            Input_path    = self.Input_path
            Output_path   = self.Output_path
            Output_T      = self.Output_T
            Output_T_pred = self.Output_T_pred
            Output_A      = self.Output_A
            assert file_index == 0, 'Only one file index is available.'

            input_path_type = self.Input_data_type[0]
        else:
            file = self.Files[file_index] + '_data.npy'
            [_, Input_path, _, Output_path, Output_T, Output_T_pred, Output_A, _, _] = np.load(file, allow_pickle=True)
            
            # Get the required inidices
            ind = self.Domain[self.Domain.file_index == file_index].Index_saved
            Input_path    = Input_path.loc[ind]
            Output_path   = Output_path.loc[ind]
            
            Output_T      = Output_T[ind]
            Output_T_pred = Output_T_pred[ind]
            Output_A      = Output_A.loc[ind].reindex(columns = self.Behaviors).fillna(False)

            input_path_type_index = self.Domain[self.Domain.file_index == file_index].data_type_index
            assert len(np.unique(input_path_type_index)) == 1, 'Different data types should not occur in a single file.'
            input_path_type = self.Input_data_type[input_path_type_index.iloc[0]]
            
            
        # Get the number of prediction time steps
        self.N_O_data_orig = np.array([len(Output_T[i_sample]) for i_sample in range(len(Output_T))], int)
        self.N_O_pred_orig = np.array([len(Output_T_pred[i_sample]) for i_sample in range(len(Output_T_pred))], int)

        self.Output_A_file = Output_A

        # Useful agents
        Used_agents = Input_path.notna()

        self.Used_samples, self.Used_agents = np.where(Used_agents)

        # Transform the agent indices to correspond with self.Agents
        Agent_index = self.get_indices_1D(Input_path.columns.to_numpy(), np.array(self.Agents))
        self.Used_agents = Agent_index[self.Used_agents]

        # Get corresponding sparse matrix
        sparse_matrix_shape = (len(Input_path), len(self.Agents))
        sparse_matrix_data = np.arange(len(self.Used_samples), dtype=int) + 1
                
        sparse_matrix = sp.sparse.coo_matrix((sparse_matrix_data, (self.Used_samples, self.Used_agents)),
                                             shape = sparse_matrix_shape)
        
        # Convert to csr for more efficient lookup
        self.sparse_matrix_orig = sparse_matrix.tocsr()

        # Transform paths into numpy
        self.X_orig = np.ones([len(self.Used_samples), self.num_timesteps_in_real, len(input_path_type)], dtype = np.float32) * np.nan
        self.Y_orig = np.ones([len(self.Used_samples), self.N_O_data_orig.max(), len(input_path_type)], dtype = np.float32) * np.nan

        # Extract data from original number a samples
        for i in range(len(self.Used_samples)):
            # get specific indices
            i_sample = self.Used_samples[i]
            i_agent  = self.Used_agents[i]

            # Get corresponding agent name
            agent = self.Agents[i_agent]

            # Check for accurate performance
            assert agent in Input_path.columns, 'Transform of agent indices failed.'
            assert isinstance(Input_path[agent].iloc[i_sample], np.ndarray), 'Input path is not a numpy array, nonna() failed' 

            # Transfare the data to numpy array
            n_time = self.N_O_data_orig[i_sample]
            self.X_orig[i] = Input_path[agent].iloc[i_sample].astype(np.float32)
            self.Y_orig[i, :n_time] = Output_path[agent].iloc[i_sample][:n_time].astype(np.float32)
            
            
    
    def _determine_pred_agents_unchecked(self):
        ## NOTE: Method has been adjusted for large datasets
        if not (hasattr(self, 'Pred_agents_eval_all') and hasattr(self, 'Pred_agents_pred_all')):
            # Get unique boolean needed agents
            needed_agents_bool = []
            for needed_agents in self.scenario_needed_agents:
                needed_agents_bool.append(np.in1d(self.Agents, needed_agents))
                    
            needed_agents_bool = np.stack(needed_agents_bool, axis = 0)
            
            # Get the unique scenario id for every sample
            Scenario = self.Domain.Scenario_type.to_numpy()
            if len(self.unique_scenarios) > 1:
                Scenario_id = self.get_indices_1D(Scenario, self.unique_scenarios)
            else:
                Scenario_id = np.zeros(len(Scenario), int)
            
            # Get needed agents for all cases
            Needed_agents = needed_agents_bool[Scenario_id]
            
            
                
            if self.agents_to_predict == 'predefined':
                self.Pred_agents_eval_all = Needed_agents
                self.Pred_agents_pred_all = Needed_agents
            else:
                if self.data_in_one_piece:
                    Recorded_agents = np.zeros(Needed_agents.shape, bool)
                    # Get the number of timesteps in each sample
                    N_O_data = np.array([len(output_T) for output_T in self.Output_T])
                    
                    # Go through unique number of output timesteps
                    for n_o in np.unique(N_O_data):
                        use_samples = np.where(N_O_data == n_o)[0]
                        use_recorded = self.Recorded.iloc[use_samples]

                        # Get the non nan cells
                        use_rec_index, use_rec_agent = np.where(use_recorded.notna())

                        # Get agent that are fully observed
                        Allowable = np.stack(use_recorded.to_numpy()[use_rec_index, use_rec_agent], 0).all(-1)
                        Recorded_agents[use_samples[use_rec_index], use_rec_agent] = Allowable
                    
                    # Get correct type 
                    if self.agents_to_predict != 'all':
                        Correct_type_agents = self.Type.to_numpy() == self.agents_to_predict
                    else:
                        Correct_type_agents = np.ones(Needed_agents.shape, bool)
                else:
                    Recorded_agents     = np.zeros((len(self.Domain), len(self.Agents)), bool)
                    Correct_type_agents = np.zeros((len(self.Domain), len(self.Agents)), bool)
                    
                    for file_index in range(len(self.Files)):
                        used = self.Domain.file_index == file_index
                        used_index = np.where(used)[0]
                        
                        # Get corresponding agent files
                        agent_file = self.Files[file_index] + '_AM.npy'
                        data_file = self.Files[file_index] + '_data.npy'
                        
                        # Load the agent files
                        Agent_data = np.load(agent_file, allow_pickle=True)
                        if len(Agent_data) == 3:
                            [Type_local, Recorded_local, _] = Agent_data
                        else:
                            assert len(Agent_data) == 4, 'Agent data should have 3 or 4 entries.'
                            [Type_local, _, Recorded_local, _] = Agent_data

                        # Load Output_T
                        [_, _, _, _, Output_T, _, _, _, _] = np.load(data_file, allow_pickle=True)

                        
                        # Get the corresponding indices
                        ind_saved = self.Domain[used].Index_saved
                        Type_local     = Type_local.loc[ind_saved]
                        Recorded_local = Recorded_local.loc[ind_saved]
                        Output_T       = Output_T[ind_saved]

                        # Get the agent indices
                        agent_index = self.get_indices_1D(Type_local.columns.to_numpy(), np.array(self.Agents))

                        # Get the number of timesteps in each sample
                        N_O_data = np.array([len(output_T) for output_T in Output_T])
                        
                        # Go through unique number of output timesteps
                        for n_o in np.unique(N_O_data):
                            use_samples = np.where(N_O_data == n_o)[0]
                            use_recorded = Recorded_local.iloc[use_samples]

                            # Get the non nan cells
                            use_rec_index, use_rec_agent = np.where(use_recorded.notna())

                            # Get agent that are fully observed
                            Allowable = np.stack(use_recorded.to_numpy()[use_rec_index, use_rec_agent], 0).all(-1)
                            Recorded_agents[used_index[use_samples[use_rec_index]], agent_index[use_rec_agent]] = Allowable
                     
                        used_2D = np.tile(used_index[:, np.newaxis], (1, len(agent_index)))
                        agent_index_2D = np.tile(agent_index[np.newaxis], (len(used_index), 1))

                        # Get correct type 
                        if self.agents_to_predict != 'all':
                            Correct_type_agents[used_2D, agent_index_2D] = Type_local.to_numpy() == self.agents_to_predict
                        else:
                            Correct_type_agents[used_2D, agent_index_2D] = True
                    
                
                self.Pred_agents_eval_all = Correct_type_agents & (Needed_agents | Recorded_agents)
                self.Pred_agents_pred_all = Needed_agents | (Correct_type_agents & Recorded_agents)
            
            
            # NuScenes exemption:
            if ((self.get_name()['print'] == 'NuScenes') and
                (self.num_timesteps_in_real == 4) and 
                (self.num_timesteps_out_real == 12) and
                (self.dt == 0.5) and 
                (self.t0_type in ['all', 'all_1'])):

                # Get recorded agents
                if self.data_in_one_piece:
                    Recorded_agents = np.zeros(Needed_agents.shape, bool)
                    # Get the number of timesteps in each sample
                    N_O_data = np.array([len(output_T) for output_T in self.Output_T])
                    
                    # Go through unique number of output timesteps
                    for n_o in np.unique(N_O_data):
                        use_samples = np.where(N_O_data == n_o)[0]
                        use_recorded = self.Recorded.iloc[use_samples]

                        # Get the non nan cells
                        use_rec_index, use_rec_agent = np.where(use_recorded.notna())

                        # Get agent that are fully observed
                        Allowable = np.stack(use_recorded.to_numpy()[use_rec_index, use_rec_agent], 0).all(-1)
                        Recorded_agents[use_samples[use_rec_index], use_rec_agent] = Allowable
                    
                    # Get correct type 
                    if self.agents_to_predict != 'all':
                        Correct_type_agents = self.Type.to_numpy() == self.agents_to_predict
                    else:
                        Correct_type_agents = np.ones(Needed_agents.shape, bool)
                else:
                    Recorded_agents     = np.zeros((len(self.Domain), len(self.Agents)), bool)
                    Correct_type_agents = np.zeros((len(self.Domain), len(self.Agents)), bool)
                    
                    for file_index in range(len(self.Files)):
                        used = self.Domain.file_index == file_index
                        used_index = np.where(used)[0]
                        
                        # Get corresponding agent files
                        agent_file = self.Files[file_index] + '_AM.npy'
                        data_file = self.Files[file_index] + '_data.npy'
                        
                        # Load the agent files
                        Agent_data = np.load(agent_file, allow_pickle=True)
                        if len(Agent_data) == 3:
                            [Type_local, Recorded_local, _] = Agent_data
                        else:
                            assert len(Agent_data) == 4, 'Agent data should have 3 or 4 entries.'
                            [Type_local, _, Recorded_local, _] = Agent_data

                        # Load Output_T
                        [_, _, _, _, Output_T, _, _, _, _] = np.load(data_file, allow_pickle=True)

                        
                        # Get the corresponding indices
                        ind_saved = self.Domain[used].Index_saved
                        Type_local     = Type_local.loc[ind_saved]
                        Recorded_local = Recorded_local.loc[ind_saved]
                        Output_T       = Output_T[ind_saved]

                        # Get the agent indices
                        agent_index = self.get_indices_1D(Type_local.columns.to_numpy(), np.array(self.Agents))

                        # Get the number of timesteps in each sample
                        N_O_data = np.array([len(output_T) for output_T in Output_T])
                        
                        # Go through unique number of output timesteps
                        for n_o in np.unique(N_O_data):
                            use_samples = np.where(N_O_data == n_o)[0]
                            use_recorded = Recorded_local.iloc[use_samples]

                            # Get the non nan cells
                            use_rec_index, use_rec_agent = np.where(use_recorded.notna())

                            # Get agent that are fully observed
                            Allowable = np.stack(use_recorded.to_numpy()[use_rec_index, use_rec_agent], 0).all(-1)
                            Recorded_agents[used_index[use_samples[use_rec_index]], agent_index[use_rec_agent]] = Allowable
                
                # Get predefined predicted agents for NuScenes
                Pred_agents_N = np.zeros(Needed_agents.shape, bool)
                PA = self.Domain.pred_agents
                PT = self.Domain.pred_timepoints
                T0 = self.Domain.t_0
                for i_sample in range(len(self.Domain)):
                    pt = PT.iloc[i_sample]
                    t0 = T0.iloc[i_sample]
                    i_time = np.argmin(np.abs(t0 - pt))
                    
                    pas = PA.iloc[i_sample]
                    pa = np.stack(pas.to_numpy().tolist(), 1)[i_time]

                    i_agents = self.get_indices_1D(pas.index.to_numpy(), np.array(self.Agents))

                    Pred_agents_N[i_sample, i_agents] = pa
                
                self.Pred_agents_eval_all = (Pred_agents_N & Recorded_agents)
                self.Pred_agents_pred_all = (Pred_agents_N & Recorded_agents) | Needed_agents
            
            
        if not hasattr(self, 'Not_pov_agent'):
            # Get unique boolean pov agents
            pov_agents_bool = []
            Agents_array = np.array(self.Agents)
            for pov_agent in self.scenario_pov_agents:
                if pov_agent is None:
                    pov_agents_bool.append(np.zeros(len(Agents_array), bool))
                else:
                    pov_agents_bool.append(Agents_array == pov_agent)
                    
            pov_agents_bool = np.stack(pov_agents_bool, axis = 0)
            
            # Get the unique scenario id for every sample
            Scenario = self.Domain.Scenario_type.to_numpy()
            if len(self.unique_scenarios) > 1:
                Scenario_id = self.get_indices_1D(Scenario, self.unique_scenarios)
            else:
                Scenario_id = np.zeros(len(Scenario), int)
            
            # Get pov boolean for each agent
            Pov_agent = pov_agents_bool[Scenario_id]
            self.Not_pov_agent = np.invert(Pov_agent)
        
                    
    def _determine_pred_agents(self, pred_pov = True, eval_pov = True):
        assert self.data_loaded, 'Data has not been loaded.'
        ## NOTE: Method has been adjusted for large datasets
        self._determine_pred_agents_unchecked()
        
        # Check the all trajectories
        if not hasattr(self, '_checked_pred_agents'):
            self._checked_pred_agents = False
            
        if not self._checked_pred_agents:
            if self.data_in_one_piece:
                self._extract_original_trajectories()
                
                # Check if everything needed is there
                Pred_agents_sparse = self.Pred_agents_pred_all[self.Used_samples, self.Used_agents]
                assert not np.isnan(self.X_orig[Pred_agents_sparse][...,:2]).all((1,2)).any(), 'A needed agent is not given.'
                assert not np.isnan(self.Y_orig[Pred_agents_sparse][...,:2]).all((1,2)).any(), 'A needed agent is not given.'
            
            else:
                # Go through all file indicies
                for file_index in range(len(self.Files)):            
                    # Get path data
                    self._extract_original_trajectories(file_index)
                    
                    used_parts = self.Domain.file_index == file_index
                     
                    # Check if everything needed is there
                    Pred_agents_sparse = self.Pred_agents_pred_all[used_parts][self.Used_samples, self.Used_agents]
                    assert not np.isnan(self.X_orig[Pred_agents_sparse][...,:2]).all((1,2)).any(), 'A needed agent is not given.'
                    assert not np.isnan(self.Y_orig[Pred_agents_sparse][...,:2]).all((1,2)).any(), 'A needed agent is not given.'
            
            self._checked_pred_agents = True
            
        
        if pred_pov:
            self.Pred_agents_pred = self.Pred_agents_pred_all.copy()
        else:
            self.Pred_agents_pred = self.Pred_agents_pred_all & self.Not_pov_agent
            
        if eval_pov:
            self.Pred_agents_eval = self.Pred_agents_eval_all.copy()
        else:
            self.Pred_agents_eval = self.Pred_agents_eval_all & self.Not_pov_agent
        
        
    
    def _group_indentical_inputs(self, eval_pov = True):
        ## NOTE: Method has been adjusted for large datasets
        if hasattr(self, 'Subgroups'):
            if hasattr(self, 'eval_pov_old'):
                if self.eval_pov_old == eval_pov:
                    return
        
        # Save the old settings for extraction
        self.eval_pov_old = eval_pov
        
        # Get save file
        test_file = self.assembled_data_file[:-4] + '_subgroups.npy'
        if os.path.isfile(test_file):
            self.Subgroups = np.load(test_file)
        else:
            self._determine_pred_agents(eval_pov = eval_pov)
            if self.data_in_one_piece:
                # Check if the dataset actually supports this
                data_set = list(self.Datasets.values())[0]
                if hasattr(data_set, 'has_repeated_inputs') and data_set.has_repeated_inputs():
                    self.Subgroups = np.arange(len(self.Domain)) + 1
                else:  
                    # Get the same entries in full dataset
                    T = self.Type.to_numpy().astype(str)
                    PA_str = self.Pred_agents_eval.astype(str) 
                    if hasattr(self.Domain, 'location'):
                        Loc = np.tile(self.Domain.location.to_numpy().astype(str)[:,np.newaxis], (1, T.shape[1]))
                        Div = np.stack((T, PA_str, Loc), axis = -1)
                    else:
                        Div = np.stack((T, PA_str), axis = -1)
                        
                    Div_unique, Div_inverse, _ = np.unique(Div, axis = 0,
                                                        return_inverse = True, 
                                                        return_counts = True)
                    
                    # Prepare subgroup saving
                    self.Subgroups = np.zeros(len(T), int)
                    subgroup_index = 1
                    
                    self._extract_original_trajectories()
                    # go through all potentiall similar entries
                    for div_inverse in range(len(Div_unique)):
                        # Get potentially similar samples
                        index = np.where(Div_inverse == div_inverse)[0]
                        
                        # Get agents that are there
                        T_div = Div_unique[div_inverse,:,0]
                        useful_agents = np.where(T_div != 'nan')[0]

                        # X.shape: len(index) x len(useful_agents) x nI x 2
                        X = np.zeros((len(index), len(useful_agents), self.X_orig.shape[-2], 2), np.float32)

                        use_X_orig = np.in1d(self.Used_samples, index) & np.in1d(self.Used_agents, useful_agents)
                        used_orig_samples = self.Used_samples[use_X_orig]
                        used_orig_agents  = self.Used_agents[use_X_orig]

                        # Get inverse of index 
                        index_inverse = np.zeros(index.max() + 1, int)
                        index_inverse[index] = np.arange(len(index), dtype = int)

                        # Get inverse of useful_agents
                        useful_agents_inverse = np.zeros(useful_agents.max() + 1, int)
                        useful_agents_inverse[useful_agents] = np.arange(len(useful_agents), dtype = int)

                        X[index_inverse[used_orig_samples], useful_agents_inverse[used_orig_agents]] = self.X_orig[use_X_orig,...,:2]
                        
                        # Get maximum number of samples comparable to all samples (assume 2GB RAM useage)
                        max_num = np.floor(2 ** 29 / np.prod(X.shape))
                        
                        # Prepare maximum differences
                        D_max = np.zeros((len(index), len(index)), np.float32)
                        
                        # Calculate differences
                        for i in range(int(np.ceil(len(index) / max_num))):
                            d_index = np.arange(max_num * i, min(max_num * (i + 1), len(index)), dtype = int) 
                            D = np.abs(X[d_index, np.newaxis] - X[np.newaxis])
                            D_max[d_index] = np.nanmax(D, (2,3,4))

                        # Find identical trajectories
                        Identical = D_max < 1e-3
                        
                        # Remove self references
                        Identical[np.arange(len(index)), np.arange(len(index))] = False

                        # Get graph
                        G = nx.Graph(Identical)
                        unconnected_subgraphs = list(nx.connected_components(G))
                        
                        for subgraph in unconnected_subgraphs:
                            # Set subgraph
                            self.Subgroups[index[list(subgraph)]] = subgroup_index
                            
                            # Update parameters
                            subgroup_index += 1
                        
                        
            else:
                # Assume that identical inputs will not be found across multiple files
                self.Subgroups = np.zeros(len(self.Domain), int)
                subgroup_index = 1
                
                # Go through all datasets
                Has_repeated_inputs = {}
                for data_set in self.Datasets.values():
                    data_set_file_name = os.path.basename(data_set.data_file)
                    has_repeated_inputs = hasattr(data_set, 'has_repeated_inputs') and data_set.has_repeated_inputs()
                    Has_repeated_inputs[data_set_file_name[:-4]] = has_repeated_inputs
                
                for file_index in range(len(self.Files)):
                    used = self.Domain.file_index == file_index
                    used_index = np.where(used)[0]
                    
                    Domain = self.Domain[used]
                    
                    # Check if calculation is needed
                    file_name_check = os.path.basename(self.Files[file_index][:-8])
                    
                    if not Has_repeated_inputs[file_name_check]:
                        self.Subgroups[used_index] = np.arange(subgroup_index, len(Domain) + subgroup_index)
                        subgroup_index += len(Domain)
                        continue
                    
                    # Load agent types
                    agent_file = self.Files[file_index] + '_AM.npy'
                    Type = np.load(agent_file, allow_pickle = True)[0]
                    Type = Type.loc[Domain.Index_saved]

                    # Transform to numpy
                    T = Type.to_numpy().astype(str)

                    # Get the corresponding agent indices
                    agent_index = self.get_indices_1D(Type.columns.to_numpy(), np.array(self.Agents))
                    
                    # Get data corresponding to file
                    Pred_agents_eval = self.Pred_agents_eval[used]
                    Pred_agents_eval = Pred_agents_eval[:, agent_index]

                    # Get Pred agent as string
                    PA_str = Pred_agents_eval.astype(str)
                    
                    # Get the same entries in full dataset
                    if hasattr(Domain, 'location'):
                        Loc = np.tile(Domain.location.to_numpy().astype(str)[:,np.newaxis], (1, T.shape[1]))
                        Div = np.stack((T, PA_str, Loc), axis = -1)
                    else:
                        Div = np.stack((T, PA_str), axis = -1) # shape: num_samples x num_agents x 2
                        
                    Div_unique, Div_inverse, _ = np.unique(Div, axis = 0, return_inverse = True,  return_counts = True)
                    
                    # Get the underlying trajectories
                    self._extract_original_trajectories(file_index)

                    # Initialize file subgroups
                    Subgroups = np.zeros(len(Domain), int)

                    # go through all potentiall similar entries
                    for div_inverse in range(len(Div_unique)):
                        # Get potentially similar samples
                        index = np.where(Div_inverse == div_inverse)[0]
                        
                        # Get agents that are there
                        T_div = Div_unique[div_inverse,:,0]
                        useful_agents = np.where(T_div != 'nan')[0]
                        
                        # Get corresponding input path                        
                        # X.shape: len(index) x len(useful_agents) x nI x 2
                        X = np.zeros((len(index), len(useful_agents), self.X_orig.shape[-2], 2), np.float32)

                        use_X_orig = np.in1d(self.Used_samples, index) & np.in1d(self.Used_agents, useful_agents)
                        used_orig_samples = self.Used_samples[use_X_orig]
                        used_orig_agents  = self.Used_agents[use_X_orig]

                        # Get inverse of index 
                        index_inverse = np.zeros(index.max() + 1, int)
                        index_inverse[index] = np.arange(len(index), dtype = int)

                        # Get inverse of useful_agents
                        useful_agents_inverse = np.zeros(useful_agents.max() + 1, int)
                        useful_agents_inverse[useful_agents] = np.arange(len(useful_agents), dtype = int)

                        X[index_inverse[used_orig_samples], useful_agents_inverse[used_orig_agents]] = self.X_orig[use_X_orig,...,:2]

                        
                        # Get maximum number of samples comparable to all samples (assume 2GB RAM useage)
                        max_num = np.floor(2 ** 29 / np.prod(X.shape))
                        
                        # Prepare maximum differences
                        D_max = np.zeros((len(index), len(index)), np.float32)
                        
                        # Calculate differences
                        for i in range(int(np.ceil(len(index) / max_num))):
                            d_index = np.arange(max_num * i, min(max_num * (i + 1), len(index)), dtype = int) 
                            D = np.abs(X[d_index, np.newaxis] - X[np.newaxis])
                            D_max[d_index] = np.nanmax(D, (2,3,4))

                        # Find identical trajectories
                        Identical = D_max < 1e-3
                        
                        # Remove self references
                        Identical[np.arange(len(index)), np.arange(len(index))] = False

                        # Get graph
                        G = nx.Graph(Identical)
                        unconnected_subgraphs = list(nx.connected_components(G))
                        
                        for subgraph in unconnected_subgraphs:
                            # Set subgraph
                            Subgroups[index[list(subgraph)]] = subgroup_index
                            
                            # Update parameters
                            subgroup_index += 1   
                    
                    # Assemble complete subgroups
                    self.Subgroups[used_index] = Subgroups
                    
            # Save subgroups
            os.makedirs(os.path.dirname(test_file), exist_ok = True)
            np.save(test_file, self.Subgroups)
        
        # check if all samples are accounted for
        assert self.Subgroups.min() > 0
        
    def _extract_identical_inputs(self, eval_pov = True, file_index = 0):
        ## NOTE: Method has not been adjusted for large datasets
        # Get the original trajectories
        self._extract_original_trajectories(file_index)

        # Extract the predicted agents
        self._determine_pred_agents(eval_pov = eval_pov)

        # get input trajectories
        self._group_indentical_inputs(eval_pov = eval_pov)

        # get the corresponfing use_indices
        if self.data_in_one_piece:
            used = np.ones(len(self.Domain), bool)
        else:
            used = self.Domain.file_index == file_index
        used_index = np.where(used)[0]
        
        # Get pred agents
        nto = self.num_timesteps_out_real

        # Get the local Pred_agents_eval
        Pred_agents_eval    = self.Pred_agents_eval[used_index]
        self.Subgroups_file = self.Subgroups[used_index]

        # Get the maximum number of pred agents
        max_num_pred_agents = Pred_agents_eval.sum(1).max()

        # Get indices of agents that are pred indices
        Use_indices = Pred_agents_eval[self.Used_samples, self.Used_agents]
        Used_pred_samples   = self.Used_samples[Use_indices]
        Used_pred_agents    = self.Used_agents[Use_indices]

        # Get unique subgroups
        self.unique_subgroups, subgroups_inverse, size_unique_subgroups = np.unique(self.Subgroups_file, return_inverse = True, return_counts = True)

        # Get the corresponding subgroups
        Used_pred_subgroups = subgroups_inverse[Used_pred_samples]

        # Get the count of the subgroups, i.e., the number correponding to the n-th occurence of the subgroup
        Used_pred_subgroups_df = pd.DataFrame(subgroups_inverse[:,np.newaxis], columns = ['Subgroups'])
        Used_pred_subgroups_count = Used_pred_subgroups_df.groupby('Subgroups').cumcount().to_numpy()[Used_pred_samples]

        # Find agents to be predicted first
        i_agent_sort = np.argsort(-Pred_agents_eval.astype(float), axis = 1)
        i_agent_sort_inverse = np.argsort(i_agent_sort, axis = 1)

        # Get the sorted Pred_agents
        self.Pred_agents_eval_sorted = np.take_along_axis(Pred_agents_eval, i_agent_sort[:,:max_num_pred_agents], axis = 1)
        self.Agents_eval_sorted      = np.array(self.Agents)[i_agent_sort[:,:max_num_pred_agents]]
        
        # Adjust the agent_id_for the pred_agents
        Used_pred_agents_sort = i_agent_sort_inverse[Used_pred_samples, Used_pred_agents]
        assert Used_pred_agents_sort.max() < max_num_pred_agents, 'Sorting of agents failed.'
        
        # Prepare saving of observerd futures
        self.Path_true_all = np.full((len(self.unique_subgroups), size_unique_subgroups.max(), max_num_pred_agents, nto, 2), np.nan, dtype = np.float32)

        self.Path_true_all[Used_pred_subgroups, Used_pred_subgroups_count, Used_pred_agents_sort] = self.Y_orig[Use_indices, :nto, :2]

        # Get current pov agent
        Scenario = self.Domain.iloc[used_index].Scenario_type.to_numpy()
        assert len(np.unique(Scenario)) == 1, 'Scenario should be unique.'

        Scenario_id = self.get_indices_1D(Scenario[[0]], self.unique_scenarios)[0]
        self.pov_agent = self.scenario_pov_agents[Scenario_id]
            
        
    def _get_joint_KDE_probabilities(self, exclude_ego = False, file_index = 0):
        if hasattr(self, 'Log_prob_true_joint'):
            assert hasattr(self, 'excluded_ego_joint'), 'Excluded ego has not been defined.'
            assert hasattr(self, 'file_index_joint'), 'File index has not been defined.'
            if (self.excluded_ego_joint == exclude_ego) and (self.file_index_joint == file_index):
                return
        
        # Save last setting 
        self.excluded_ego_joint = exclude_ego
        self.file_index_joint   = file_index

        # Check if dataset has all valuable stuff
        self._extract_identical_inputs(eval_pov = not exclude_ego, file_index = file_index)

        # Shape: Num_samples
        self.Log_prob_true_joint = np.zeros(self.Pred_agents_eval_sorted.shape[0], dtype = np.float32)

        # Get the current file name and replace it to the Predictions folder
        file_addon = 'joint_gt_KDE_'
        if exclude_ego:
            file_addon += 'wo_pov'
        else:
            file_addon += 'wi_pov'
        file_addon += '_FI_' + str(file_index)

        safe_file = self.change_result_directory(self.assembled_data_file, 'Predictions', file_addon)

        # Check if KDE models can be loaded
        if os.path.isfile(safe_file):
            self.KDE_joint_data = np.load(safe_file, allow_pickle = True)[0]
            clustering_loaded = True
        else:
            self.KDE_joint_data = {}
            clustering_loaded = False
        
        Num_steps = np.minimum(self.num_timesteps_out_real, self.N_O_data_orig)
        
        self.KDE_joint = {}
        print('Calculate joint PDF on ground truth probabilities.', flush = True)
        for i_subgroup, subgroup in enumerate(self.unique_subgroups):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(i_subgroup + 1, len(self.unique_subgroups)), flush = True)
            s_ind = np.where(self.Subgroups_file == subgroup)[0]
            
            assert len(np.unique(self.Pred_agents_eval_sorted[s_ind], axis = 0)) == 1
            pred_agents = self.Pred_agents_eval_sorted[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            nto_subgroup = Num_steps[s_ind]
            Paths_subgroup = self.Path_true_all[i_subgroup,:len(s_ind)]
            
            self.KDE_joint[subgroup] = {}
            if clustering_loaded:
                assert subgroup in self.KDE_joint_data, 'Subgroup not found in loaded data.'
            else:
                self.KDE_joint_data[subgroup] = {}

            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                n_ind = np.where(nto == nto_subgroup)[0]
                nto_index = s_ind[n_ind]
                
                # Should be shape: num_subgroup_samples x num_agents x num_T_O x 2
                paths_true = Paths_subgroup[n_ind][:,pred_agents,:nto]
                        
                # Collapse agents
                num_features = pred_agents.sum() * nto * 2
                paths_true_comp = paths_true.reshape(len(n_ind), num_features)
                
                # Train model
                if clustering_loaded:
                    assert nto in self.KDE_joint_data[subgroup], 'Number of output timesteps not found in loaded data.'
                    kde_data = self.KDE_joint_data[subgroup][nto]

                    cluster_labels = kde_data['cluster_labels']
                    assert len(cluster_labels) == len(paths_true_comp), 'Cluster labels do not match the number of samples.'
                    kde = ROME().fit(paths_true_comp, clusters = cluster_labels)
                else:
                    kde = ROME().fit(paths_true_comp)
                    kde_data = {'cluster_labels': kde.labels_}
                    self.KDE_joint_data[subgroup][nto] = kde_data

                log_prob_true = kde.score_samples(paths_true_comp)
                
                self.KDE_joint[subgroup][nto] = kde
                self.Log_prob_true_joint[nto_index] = log_prob_true

        # Save the KDE models
        if (not clustering_loaded) and self.save_predictions:
            os.makedirs(os.path.dirname(safe_file), exist_ok = True)
            np.save(safe_file, np.array([self.KDE_joint_data, 0], dtype = object))
            
            
    def _get_indep_KDE_probabilities(self, exclude_ego = False, file_index = 0):
        if hasattr(self, 'Log_prob_true_indep'):
            assert hasattr(self, 'excluded_ego_indep'), 'Excluded ego has not been defined.'
            assert hasattr(self, 'file_index_indep'), 'File index has not been defined.'
            if (self.excluded_ego_indep == exclude_ego) and (self.file_index_indep == file_index):
                return
        
        # Save last setting 
        self.excluded_ego_indep = exclude_ego
        self.file_index_indep   = file_index
        
        # Check if dataset has all valuable stuff
        self._extract_identical_inputs(eval_pov = not exclude_ego, file_index = file_index)
        
        # Shape: Num_samples x num agents
        self.Log_prob_true_indep = np.zeros(self.Pred_agents_eval_sorted.shape, dtype = np.float32)

        # Get the current file name and replace it to the Predictions folder
        # Independent KDEs do not need to be saved for different pov settings,
        # as the agents are saved individually
        file_addon = 'indep_gt_KDE' 
        file_addon += '_FI_' + str(file_index)

        safe_file = self.change_result_directory(self.assembled_data_file, 'Predictions', file_addon)

        # Check if KDE models can be loaded
        if os.path.isfile(safe_file):
            self.KDE_indep_data = np.load(safe_file, allow_pickle = True)[0]
            clustering_loaded = True
        else:
            self.KDE_indep_data = {}
            clustering_loaded = False
        
        Num_steps = np.minimum(self.num_timesteps_out_real, self.N_O_data_orig)
        
        self.KDE_indep = {}
        print('Calculate indep PDF on ground truth probabilities.', flush = True)
        for i_subgroup, subgroup in enumerate(self.unique_subgroups):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(i_subgroup + 1, len(self.unique_subgroups)), flush = True)
            s_ind = np.where(self.Subgroups_file == subgroup)[0]
            
            assert len(np.unique(self.Pred_agents_eval_sorted[s_ind], axis = 0)) == 1
            pred_agents = self.Pred_agents_eval_sorted[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            pred_agents_id = np.where(pred_agents)[0]
            
            nto_subgroup = Num_steps[s_ind]
            Paths_subgroup = self.Path_true_all[i_subgroup,:len(s_ind)]
            
            self.KDE_indep[subgroup] = {}

            if not subgroup in self.KDE_indep_data:
                self.KDE_indep_data[subgroup] = {}

            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                n_ind = np.where(nto == nto_subgroup)[0]
                nto_index = s_ind[n_ind]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = Paths_subgroup[n_ind][:,pred_agents,:nto]
                
                num_features = nto * 2
                
                self.KDE_indep[subgroup][nto] = {}
                if not nto in self.KDE_indep_data[subgroup]:
                    self.KDE_indep_data[subgroup][nto] = {}

                for i_agent, i_agent_orig in enumerate(pred_agents_id):
                    agent = self.Agents_eval_sorted[nto_index, i_agent_orig]
                    assert len(np.unique(agent)) == 1, 'Agent is not unique.'
                    agent = agent[0]
                    
                    # Get agent
                    paths_true_agent = paths_true[:,i_agent]
                
                    # Collapse agents
                    paths_true_agent_comp = paths_true_agent.reshape(len(n_ind), num_features)
                        
                    # Train model
                    # Check if current agent is pov agent

                    if clustering_loaded and (agent != self.pov_agent):
                        assert agent in self.KDE_indep_data[subgroup][nto], 'Agent not found in loaded data.'
                    
                    if agent in self.KDE_indep_data[subgroup][nto]:
                        kde_data = self.KDE_indep_data[subgroup][nto][agent]
                        cluster_labels = kde_data['cluster_labels']
                        assert len(cluster_labels) == len(paths_true_agent_comp), 'Cluster labels do not match the number of samples.'
                        kde = ROME().fit(paths_true_agent_comp, clusters = cluster_labels)
                    else:
                        kde = ROME().fit(paths_true_agent_comp)
                        kde_data = {'cluster_labels': kde.labels_}
                        self.KDE_indep_data[subgroup][nto][agent] = kde_data

                    log_prob_true_agent = kde.score_samples(paths_true_agent_comp)
                    
                    self.KDE_indep[subgroup][nto][agent] = kde
                    self.Log_prob_true_indep[nto_index,i_agent_orig] = log_prob_true_agent
        
        # Save the KDE models
        if self.save_predictions:
            os.makedirs(os.path.dirname(safe_file), exist_ok = True)
            np.save(safe_file, np.array([self.KDE_indep_data, 0], dtype = object))
