import numpy as np
import pandas as pd
import importlib
import os
import warnings
import networkx as nx

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

        if isinstance(data_set_dict, dict):
            data_set_dict = [data_set_dict]
        else:
            assert isinstance(data_set_dict, list), "To combine datasets, put the dataset dictionaries into lists."
        
        # Initialize datasets
        self.Datasets = {}
        self.Latex_names = []
        for data_dict in data_set_dict:
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

                # TODO: For further methods, check 
                if perturbation['attack'] == 'Adversarial':
                    perturbation['exp_parameters'] = parameters
                
                # Get perturbation type
                pert_name = perturbation['attack']

                # Remove attack from dictionary
                perturbation.pop('attack')


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
            
            data_set = data_set_class(Perturbation, *parameters)

            data_set.set_extraction_parameters(t0_type, T0_type_compare, max_num_agents)
            
            latex_name = data_set.get_name()['latex']
            if latex_name[:6] == r'\emph{' and latex_name[-1] == r'}':
                latex_name = latex_name[6:-1]
                
            self.Latex_names.append(latex_name) 

            data_set_name = data_set.get_name()['print']
            if data_set.is_perturbed:
                # Get perturbation index from filename
                file_name = data_set.data_params_to_string(0.1, 10, 10)
                pert_index = int(file_name.split('_')[-1].split('.')[0])
                data_set_name += ' (Perturbed ' + str(pert_index) + ')'

            self.Datasets[data_set_name] = data_set
            
        self.single_dataset = len(self.Datasets.keys()) <= 1    
        
        # Get relevant scenario information
        scenario_names = []
        scenario_general_input = []
        
        scenario_pov_agents = np.empty(len(self.Datasets), object)
        scenario_needed_agents = np.empty(len(self.Datasets), object)
        
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
            t0_type_file_name = {'start':     'start',
                                 'all':       'all_p',
                                 'col_equal': 'fix_e',
                                 'col_set':   'fix_s',
                                 'crit':      'crit_'}
            
            self.t0_type = list(self.Datasets.values())[0].t0_type
            self.t0_type_name = t0_type_file_name[self.t0_type]
        else:
            self.t0_type = 'mixed'
            self.t0_type_name = 'mixed'
            
        self.p_quantile = list(self.Datasets.values())[0].p_quantile
        
        
        max_num_agents = np.array([data_set.max_num_agents for data_set in list(self.Datasets.values())])
        if np.all(max_num_agents == None):
            self.max_num_agents = None
        else:        
            max_num_agents = max_num_agents[max_num_agents != None]
            self.max_num_agents = max_num_agents.min()
        
        self.data_loaded = False
        
    
    def reset(self):
        for data_set in self.Datasets.values():
            data_set.reset()
        
        self.data_loaded = False
        
        # Delete extracted data
        if hasattr(self, 'X_orig') and hasattr(self, 'Y_orig'):
            del self.X_orig
            del self.Y_orig
        
        if hasattr(self, 'Pred_agents_eval_all') and hasattr(self, 'Pred_agents_pred_all'):
            del self.Pred_agents_eval_all
            del self.Pred_agents_pred_all
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
    
    def change_result_directory(self, filepath, new_path_addon, new_file_addon, file_type = '.npy'):
        return list(self.Datasets.values())[0].change_result_directory(filepath, new_path_addon, new_file_addon, file_type)
    
    
    def determine_required_timesteps(self, num_timesteps):
        return list(self.Datasets.values())[0].determine_required_timesteps(num_timesteps)
    
    
    def set_data_file(self, dt, num_timesteps_in, num_timesteps_out):
        (self.num_timesteps_in_real, 
        self.num_timesteps_in_need)  = self.determine_required_timesteps(num_timesteps_in)
        (self.num_timesteps_out_real, 
        self.num_timesteps_out_need) = self.determine_required_timesteps(num_timesteps_out)
        if self.single_dataset:
            self.data_file = list(self.Datasets.values())[0].data_params_to_string(dt, num_timesteps_in, num_timesteps_out)
        else:
            # Get data_file from every constituent dataset
            self.data_file = (list(self.Datasets.values())[0].path + os.sep + 
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

                self.data_file += '--mixed_' + t0_parts[0][-2:]

            # Find the parts with 'dt'
            dt_parts = unique_parts[np.array(['dt' == s[:2] for s in unique_parts])]
            assert len(dt_parts) == 1
            self.data_file += '--' + dt_parts[0]

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
            
            self.data_file += '.npy'
    
    def get_data(self, dt, num_timesteps_in, num_timesteps_out):
        self.set_data_file(dt, num_timesteps_in, num_timesteps_out)
        complete_failure = None
        for data_set in self.Datasets.values():
            data_failure = data_set.get_data(dt, num_timesteps_in, num_timesteps_out)
            if data_failure is not None:
                complete_failure = data_failure[:-1] + ' (in ' + data_set.get_name()['print'] + ').'
            
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
        self.Recorded         = pd.DataFrame(np.zeros((0,0), np.ndarray))
        self.Domain           = pd.DataFrame(np.zeros((0,0), np.ndarray))
        
        self.num_behaviors = pd.Series(np.zeros(len(self.Behaviors), int), index = self.Behaviors)
        
        for data_set in self.Datasets.values():
            # Combine datasets
            self.Input_path       = pd.concat((self.Input_path, data_set.Input_path))
            self.Input_T          = np.concatenate((self.Input_T, data_set.Input_T), axis = 0)
            
            self.Output_path      = pd.concat((self.Output_path, data_set.Output_path))
            self.Output_T_E       = np.concatenate((self.Output_T_E, data_set.Output_T_E), axis = 0)
            
            self.Type             = pd.concat((self.Type, data_set.Type))
            self.Recorded         = pd.concat((self.Recorded, data_set.Recorded))
            self.Domain           = pd.concat((self.Domain, data_set.Domain))
            
            self.Output_T_pred    = np.concatenate((self.Output_T_pred, data_set.Output_T_pred), axis = 0)
            self.Output_T         = np.concatenate((self.Output_T, data_set.Output_T), axis = 0)
            
            self.Output_A         = pd.concat((self.Output_A, data_set.Output_A))
            
            # Get num behaviors
            self.num_behaviors[data_set.Behaviors] += data_set.num_behaviors 
            
            # Consider generalized input
            if self.general_input_available:
                self.Input_prediction = pd.concat((self.Input_prediction, data_set.Input_prediction))
                
            else:
                IP = pd.DataFrame(np.ones((len(data_set.Output_T), 1), float) * np.nan, columns = ['empty'])
                self.Input_prediction = pd.concat((self.Input_prediction, IP))
        
        
        # Ensure that the same order of agents is maintained for input and output paths
        self.Output_path = self.Output_path[self.Input_path.columns]
        
        # Ensure agent order is the same everywhere
        Agents = np.array(self.Input_path.columns)
        self.Output_path = self.Output_path[Agents]
        self.Type        = self.Type[Agents]
        self.Recorded    = self.Recorded[Agents]
        
        # Ensure behavior stuff is aligned and fill missing behaviors
        self.Output_A = self.Output_A[self.Behaviors].fillna(False)
        
        # Set final values
        self.data_loaded = True
        self.dt = dt
        
        # Remove useless samples
        self._determine_pred_agents()
        Num_eval_agents = self.Pred_agents_eval.sum(1)
        Num_pred_agents = self.Pred_agents_pred.sum(1)
        
        Useful_agents = (Num_eval_agents + Num_pred_agents) > 0 
        
        if Useful_agents.sum() < 5:
            complete_failure = "not enough prediction problems are avialable."
        
        # Overwrite
        self.Input_prediction = self.Input_prediction.iloc[Useful_agents].reset_index(drop = True)
        self.Input_path       = self.Input_path.iloc[Useful_agents].reset_index(drop = True)
        self.Input_T          = self.Input_T[Useful_agents]

        self.Output_path      = self.Output_path.iloc[Useful_agents].reset_index(drop = True)
        self.Output_T         = self.Output_T[Useful_agents]
        self.Output_T_pred    = self.Output_T_pred[Useful_agents]
        self.Output_A         = self.Output_A.iloc[Useful_agents].reset_index(drop = True)
        self.Output_T_E       = self.Output_T_E[Useful_agents]

        self.Type             = self.Type.iloc[Useful_agents].reset_index(drop = True)
        self.Recorded         = self.Recorded.iloc[Useful_agents].reset_index(drop = True)
        self.Domain           = self.Domain.iloc[Useful_agents].reset_index(drop = True)
        
        self.num_behaviors = pd.Series(np.zeros(len(self.Behaviors), int), index = self.Behaviors)
        
        # Overwrite old saved aspects
        self.X_orig = self.X_orig[Useful_agents]
        self.Y_orig = self.Y_orig[Useful_agents]
        
        self.N_O_data_orig = self.N_O_data_orig[Useful_agents]
        self.N_O_pred_orig = self.N_O_pred_orig[Useful_agents]
        
        self.Pred_agents_eval_all = self.Pred_agents_eval_all[Useful_agents]
        self.Pred_agents_pred_all = self.Pred_agents_pred_all[Useful_agents]
        self.Pred_agents_eval     = self.Pred_agents_eval[Useful_agents]
        self.Pred_agents_pred     = self.Pred_agents_pred[Useful_agents]
        
        if hasattr(self, 'Not_pov_agent'):
            self.Not_pov_agent = self.Not_pov_agent[Useful_agents]
        
        
        return complete_failure
        
        
    
    def get_Target_MeterPerPx(self, domain):
        return self.Datasets[domain.Scenario].Images.Target_MeterPerPx.loc[domain.image_id]
    
    
    def return_batch_images(self, domain, center, rot_angle, target_width, target_height, 
                            grayscale = False, return_resolution = False):
        
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
                print('')
                print('Rotate images from dataset ' + data_set.get_name()['print'])
                
                Use = (domain['Scenario'] == data_set.get_name()['print']).to_numpy()
                
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
                                                    Imgs, Index_use)
                if return_resolution:
                    Imgs_m_per_px[Use] = data_set.Images.Target_MeterPerPx.loc[domain.image_id.iloc[Use]]
        if return_resolution:
            return Imgs, Imgs_m_per_px
        else:
            return Imgs
    
    
    def transform_outputs(self, output, model_pred_type, metric_pred_type, pred_save_file):
        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified.")
            
        if self.single_dataset:
            data_set = list(self.Datasets.values())[0]
            output_trans_all = data_set.transform_outputs(output, model_pred_type, metric_pred_type, pred_save_file)
            
        else:
            # Set up empty splitting parts
            output_trans = []
            Uses = []
            
            # Get parts of output
            output_data  = output[1:]
            output_index = output[0]
            
            # Go through part datasets
            for i, data_set in enumerate(self.Datasets.values()):
                # Get predictions from this dataset
                use = self.Domain['Scenario'].iloc[output_index] == data_set.get_name()['print']
                use = np.where(use.to_numpy())[0]
                Uses.append(use)
                
                # Get output_data for the specific dataset
                output_d_data = []
                for out in output_data:
                    assert isinstance(out, pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
                    out_data = out.iloc[use]
                    # check if this is a class thing
                    if np.in1d(np.array(out_data.columns), self.Behaviors).all():
                        out_data = out_data[data_set.Behaviors]
                
                    output_d_data.append(out_data)
                # Get output_d_index
                dataset_index = np.where((self.Domain['Scenario'] == data_set.get_name()['print']).to_numpy())[0]
                if np.all(dataset_index == output_index[use]):
                    output_d_index = np.arange(len(dataset_index))
                else:
                    match = dataset_index[np.newaxis] == output_index[use, np.newaxis]
                    assert np.all(match.sum(axis = 1) == 1)
                    output_d_index = match.argmax(axis = 1)
                
                # assemble output for dataset
                output_d = [output_d_index] + output_d_data            
                
                ds_save_file = pred_save_file.replace(os.sep + self.get_name()['file'], os.sep + data_set.get_name()['file'])
                output_trans.append(data_set.transform_outputs(output_d, model_pred_type, metric_pred_type, ds_save_file))
            
            output_trans_all = [output_index]
            # Go through all parts of the transformed prediction
            for j in range(1, len(output_trans[0])):
                assert isinstance(output_trans[0][j], pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
                # Collect columns
                columns = []
                for out in output_trans:
                    columns.append(out[j].columns)
                columns = np.unique(np.concatenate(columns))
                
                array_type = output_trans[0][j].to_numpy().dtype
                
                output_trans_all.append(pd.DataFrame(np.zeros((len(output_index), len(columns)), array_type),
                                                     columns = list(columns), index = output_index))
                
                for i, out in enumerate(output_trans):
                    c_index = (out[j].columns.to_numpy()[:,np.newaxis] == columns[np.newaxis]).argmax(axis = - 1)
                    output_trans_all[j].iloc[Uses[i], c_index] = out[j]
                
        return output_trans_all
        
    
    def provide_map_drawing(self, domain):
        return self.Datasets[domain.Scenario].provide_map_drawing(domain)
    
    
    def future_input(self):
        return all([data_set.future_input() for data_set in self.Datasets.values()])
        
    
    def includes_images(self):
        return any([data_set.includes_images() for data_set in self.Datasets.values()])
    
    
    def get_name(self):
        if self.single_dataset:
            return list(self.Datasets.values())[0].get_name()
        else:
            file_name = ('Comb_' + '_'.join([s[0] for s in self.Latex_names]) + '_' * 5)[:10]
            print_name = 'Combined dataset (' + r' & '.join(self.Latex_names) + ')'
            print_name = print_name.replace('/', ' & ')
            print_name = print_name.replace(os.sep, ' & ')
            names = {'print': print_name,
                     'file': file_name,
                     'latex': r'/'.join(self.Latex_names)}
            return names
            
    #%% Useful function for later modules
    def _extract_original_trajectories(self):
        if hasattr(self, 'X_orig') and hasattr(self, 'Y_orig'):
            return
        
        self.N_O_data_orig = np.zeros(len(self.Output_T), int)
        self.N_O_pred_orig = np.zeros(len(self.Output_T), int)
        for i_sample in range(self.Output_T.shape[0]):
            self.N_O_data_orig[i_sample] = len(self.Output_T[i_sample])
            self.N_O_pred_orig[i_sample] = len(self.Output_T_pred[i_sample])
        
        Agents = np.array(self.Input_path.columns)
        
        X_help = self.Input_path.to_numpy()
        Y_help = self.Output_path.to_numpy()
            
        self.X_orig = np.ones(list(X_help.shape) + [self.num_timesteps_in_real, 2], dtype = np.float32) * np.nan
        self.Y_orig = np.ones(list(Y_help.shape) + [self.N_O_data_orig.max(), 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample in range(self.X_orig.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if not isinstance(X_help[i_sample, i_agent], float):    
                    n_time = self.N_O_data_orig[i_sample]
                    self.X_orig[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                    self.Y_orig[i_sample, i_agent, :n_time] = Y_help[i_sample, i_agent][:n_time].astype(np.float32)

                    
    def _determine_pred_agents(self, pred_pov = True, eval_pov = True):
        if not (hasattr(self, 'Pred_agents_eval_all') and hasattr(self, 'Pred_agents_pred_all')):
            Agents = np.array(self.Recorded.columns)
            
            # Get unique boolean needed agents
            needed_agents_bool = []
            for needed_agents in self.scenario_needed_agents:
                needed_agents_bool.append(np.in1d(Agents, needed_agents))
                    
            needed_agents_bool = np.stack(needed_agents_bool, axis = 0)
            
            # Get the unique scenario id for every sample
            Scenario = self.Domain.Scenario_type.to_numpy()
            Scenario_id = (Scenario[:,np.newaxis] == self.unique_scenarios[np.newaxis])
            assert np.all(Scenario_id.sum(1) == 1)
            Scenario_id = Scenario_id.argmax(1)
            
            # Get needed agents for all cases
            Needed_agents = needed_agents_bool[Scenario_id]
            
            # Get path data
            self._extract_original_trajectories()
            
            if self.agents_to_predict == 'predefined':
                self.Pred_agents_eval_all = Needed_agents
                self.Pred_agents_pred_all = Needed_agents
            else:
                Recorded_agents = np.zeros(Needed_agents.shape, bool)
                for i_sample in range(len(self.Recorded)):
                    R = self.Recorded.iloc[i_sample]
                    for i_agent, agent in enumerate(Agents):
                        if isinstance(R[agent], np.ndarray):
                            Recorded_agents[i_sample, i_agent] = np.all(R[agent])

                # remove non-moving agents
                Tr = np.concatenate((self.X_orig, self.Y_orig), axis = 2)
                Dr = np.abs(Tr[:,:,1:] - Tr[:,:,:-1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category = RuntimeWarning)
                    Dr = np.nanmax(Dr, (2, 3))
                
                # Get moving vehicles
                Moving_agents = Dr > 0.01
                
                # Get correct type 
                T = self.Type[Agents].to_numpy()
                if self.agents_to_predict != 'all':
                    Correct_type_agents = T == self.agents_to_predict
                else:
                    Correct_type_agents = np.ones(T.shape, bool)
                
                Extra_agents = (Correct_type_agents & Moving_agents & Recorded_agents)
                
                self.Pred_agents_eval_all = (Correct_type_agents & Needed_agents) | Extra_agents
                self.Pred_agents_pred_all = Needed_agents | Extra_agents
            
            
            # NuScenes exemption:
            if ((self.get_name()['print'] == 'NuScenes') and
                (self.num_timesteps_in_real == 4) and 
                (self.num_timesteps_out_real == 12) and
                (self.dt == 0.5) and 
                (self.t0_type == 'all')):
                
                # Get predefined predicted agents for NuScenes
                Pred_agents_N = np.zeros(Needed_agents.shape, bool)
                PA = self.Domain.pred_agents
                PT = self.Domain.pred_timepoints
                T0 = self.Domain.t_0
                for i_sample in range(len(self.Recorded)):
                    pt = PT.iloc[i_sample]
                    t0 = T0.iloc[i_sample]
                    i_time = np.argmin(np.abs(t0 - pt))
                    
                    pas = PA.iloc[i_sample]
                    i_agents = (Agents[np.newaxis] == np.array(pas.index)[:,np.newaxis]).argmax(1)
                    pa = np.stack(pas.to_numpy().tolist(), 1)[i_time]
                    
                    Pred_agents_N[i_sample, i_agents] = pa
                
                self.Pred_agents_eval_all = Pred_agents_N
                self.Pred_agents_pred_all = Pred_agents_N | Needed_agents
                
            # Check if everything needed is there
            assert not np.isnan(self.X_orig[self.Pred_agents_pred_all]).all((1,2)).any(), 'A needed agent is not given.'
            assert not np.isnan(self.Y_orig[self.Pred_agents_pred_all]).all((1,2)).any(), 'A needed agent is not given.'
        
        if not (pred_pov and eval_pov):
            if not hasattr(self, 'Not_pov_agent'):
                
                # Get unique boolean pov agents
                Agents = np.array(self.Recorded.columns)
                pov_agents_bool = []
                for pov_agent in self.scenario_pov_agents:
                    if pov_agent is None:
                        pov_agents_bool.append(np.zeros(len(Agents), bool))
                    else:
                        pov_agents_bool.append(Agents == pov_agent)
                        
                pov_agents_bool = np.stack(pov_agents_bool, axis = 0)
                
                # Get the unique scenario id for every sample
                Scenario = self.Domain.Scenario_type.to_numpy()
                Scenario_id = (Scenario[:,np.newaxis] == self.unique_scenarios[np.newaxis])
                assert np.all(Scenario_id.sum(1) == 1)
                Scenario_id = Scenario_id.argmax(1)
                
                # Get pov boolean for each agent
                Pov_agent = pov_agents_bool[Scenario_id]
                self.Not_pov_agent = np.invert(Pov_agent)
            
        if pred_pov:
            self.Pred_agents_pred = self.Pred_agents_pred_all.copy()
        else:
            self.Pred_agents_pred = self.Pred_agents_pred_all & self.Not_pov_agent
            
        if eval_pov:
            self.Pred_agents_eval = self.Pred_agents_eval_all.copy()
        else:
            self.Pred_agents_eval = self.Pred_agents_eval_all & self.Not_pov_agent
        
    
    def _group_indentical_inputs(self, eval_pov = True):
        self._extract_original_trajectories()
        self._determine_pred_agents(eval_pov = eval_pov)
        
        if hasattr(self, 'Subgroups'):
            return
        
        # Get save file
        test_file = self.data_file [:-4] + '_subgroups.npy'
        if os.path.isfile(test_file):
            self.Subgroups = np.load(test_file)
        else:
            # Get the same entrie in full dataset
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
            
            # go through all potentiall similar entries
            for div_inverse in range(len(Div_unique)):
                # Get potentially similar samples
                index = np.where(Div_inverse == div_inverse)[0]
                
                # Get agents that are there
                T_div = Div_unique[div_inverse,:,0]
                useful_agents = np.where(T_div != 'nan')[0]
                
                # Get corresponding input path
                X = self.X_orig[index[:,np.newaxis], useful_agents[np.newaxis]]
                # X.shape: len(index) x len(useful_agents) x nI x 2
                
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
            
            # Save subgroups
            os.makedirs(os.path.dirname(test_file), exist_ok = True)
            np.save(test_file, self.Subgroups)
        
        # check if all samples are accounted for
        assert self.Subgroups.min() > 0
    
        
    def _extract_identical_inputs(self, eval_pov = True):
        if hasattr(self, 'Path_true_all'):
            return
        
        # get input trajectories
        self._group_indentical_inputs(eval_pov = eval_pov)
        
        # Get pred agents
        nto = self.num_timesteps_out_real
        
        num_samples = len(self.Pred_agents_eval)
        max_num_pred_agents = self.Pred_agents_eval.sum(1).max()
        
        # Find agents to be predicted first
        i_agent_sort = np.argsort(-self.Pred_agents_eval.astype(float))
        i_agent_sort = i_agent_sort[:,:max_num_pred_agents]
        i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_pred_agents))
        
        # reset prediction agents
        self.Pred_agents_eval_sorted = self.Pred_agents_eval[i_sampl_sort, i_agent_sort]
        
        # Sort future trajectories
        Path_true = self.Y_orig[i_sampl_sort, i_agent_sort, :nto]
        
        # Prepare saving of observerd futures
        max_len = np.unique(self.Subgroups, return_counts = True)[1].max()
        self.Path_true_all = np.ones((self.Subgroups.max() + 1, max_len, max_num_pred_agents, nto, 2)) * np.nan 

        # Go throug subgroups for data
        for subgroup in np.unique(self.Subgroups):
            # get samples with similar inputs
            s_ind = np.where(subgroup == self.Subgroups)[0]
            
            # Get pred agents
            pred_agents = self.Pred_agents_eval_sorted[s_ind] # shape: len(s_ind) x num_agents
            
            # Get future pbservations for all agents
            path_true = Path_true[s_ind] # shape: len(s_ind) x max_num_pred_agents x nto x 2
            
            # Remove not useful agents
            path_true[~pred_agents] = np.nan
            
            # For some reason using pred_agents here moves the agent dimension to the front
            self.Path_true_all[subgroup,:len(s_ind)] = path_true
            
        
    def _get_joint_KDE_probabilities(self, exclude_ego = False):
        if hasattr(self, 'Log_prob_true_joint'):
            if self.excluded_ego_joint == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego_joint = exclude_ego
        
        # Check if dataset has all valuable stuff
        self._determine_pred_agents(eval_pov = ~exclude_ego)
        self._extract_identical_inputs(eval_pov = ~exclude_ego)
        Pred_agents = self.Pred_agents_eval_sorted
        
        # Shape: Num_samples
        self.Log_prob_true_joint = np.zeros(Pred_agents.shape[0], dtype = np.float32)
        
        Num_steps = np.minimum(self.num_timesteps_out_real, self.N_O_data_orig)
        
        self.KDE_joint = {}
        print('Calculate joint PDF on ground truth probabilities.', flush = True)
        for subgroup in np.unique(self.Subgroups):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(subgroup, len(np.unique(self.Subgroups))), flush = True)
            s_ind = np.where(self.Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[s_ind], axis = 0)) == 1
            pred_agents = Pred_agents[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            nto_subgroup = Num_steps[s_ind]
            Paths_subgroup = self.Path_true_all[subgroup,:len(s_ind)]
            
            self.KDE_joint[subgroup] = {}
            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                n_ind = np.where(nto == nto_subgroup)[0]
                nto_index = s_ind[n_ind]
                
                # Should be shape: num_subgroup_samples x num_agents x num_T_O x 2
                paths_true = Paths_subgroup[n_ind][:,pred_agents,:nto]
                        
                # Collapse agents
                num_features = pred_agents.sum() * nto * 2
                paths_true_comp = paths_true.reshape(-1, num_features)
                
                # Train model
                kde = ROME().fit(paths_true_comp)
                log_prob_true = kde.score_samples(paths_true_comp)
                
                self.KDE_joint[subgroup][nto] = kde
                self.Log_prob_true_joint[nto_index] = log_prob_true
            
            
    def _get_indep_KDE_probabilities(self, exclude_ego = False):
        if hasattr(self, 'Log_prob_true_indep'):
            if self.excluded_ego_indep == exclude_ego:
                return
        
        Agents = np.array(self.Input_path.columns)
        
        # Save last setting 
        self.excluded_ego_indep = exclude_ego
        
        # Check if dataset has all valuable stuff
        self._determine_pred_agents(eval_pov = ~exclude_ego)
        self._extract_identical_inputs(eval_pov = ~exclude_ego)
        Pred_agents = self.Pred_agents_eval_sorted
        
        # Shape: Num_samples x num agents
        self.Log_prob_true_indep = np.zeros(Pred_agents.shape, dtype = np.float32)
        
        Num_steps = np.minimum(self.num_timesteps_out_real, self.N_O_data_orig)
        
        self.KDE_indep = {}
        print('Calculate indep PDF on ground truth probabilities.', flush = True)
        for subgroup in np.unique(self.Subgroups):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(subgroup, len(np.unique(self.Subgroups))), flush = True)
            s_ind = np.where(self.Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[s_ind], axis = 0)) == 1
            pred_agents = Pred_agents[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            pred_agents_id = np.where(pred_agents)[0]
            
            nto_subgroup = Num_steps[s_ind]
            Paths_subgroup = self.Path_true_all[subgroup,:len(s_ind)]
            
            self.KDE_indep[subgroup] = {}
            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                n_ind = np.where(nto == nto_subgroup)[0]
                nto_index = s_ind[n_ind]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = Paths_subgroup[n_ind][:,pred_agents,:nto]
                
                num_features = nto * 2
                
                self.KDE_indep[subgroup][nto] = {}
                for i_agent, i_agent_orig in enumerate(pred_agents_id):
                    agent = Agents[i_agent_orig]
                    
                    # Get agent
                    paths_true_agent = paths_true[:,i_agent]
                
                    # Collapse agents
                    paths_true_agent_comp = paths_true_agent.reshape(-1, num_features)
                        
                    # Train model
                    kde = ROME().fit(paths_true_agent_comp)
                    log_prob_true_agent = kde.score_samples(paths_true_agent_comp)
                    
                    self.KDE_indep[subgroup][nto][agent] = kde
                    self.Log_prob_true_indep[nto_index,i_agent_orig] = log_prob_true_agent
                    