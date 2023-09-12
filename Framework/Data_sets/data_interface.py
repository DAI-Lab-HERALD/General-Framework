import numpy as np
import pandas as pd
import importlib
from scenario_none import scenario_none
import os
import warnings
import networkx as nx

class data_interface(object):
    def __init__(self, data_dicts, parameters):
        # Initialize path
        self.path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
        
        if isinstance(data_dicts, dict):
            data_dicts = [data_dicts]
        else:
            assert isinstance(data_dicts, list), "To combine datasets, put the dataset dictionaries into lists."
        
        # Initialize datasets
        self.Datasets = {}
        self.Latex_names = []
        for data_dict in data_dicts:
            assert isinstance(data_dict, dict), "Dataset is not provided as a dictionary."
            assert 'scenario' in data_dict.keys(), "Dataset name is missing."
            assert 't0_type' in data_dict.keys(), "Prediction time is missing."
            assert 'conforming_t0_types' in data_dict.keys(), "Prediction time constraints are missing."
            
            data_set_name = data_dict['scenario']
            t0_type       = data_dict['t0_type']
            Comp_t0_types = data_dict['conforming_t0_types']
            
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
            
            data_set = data_set_class(*parameters)
            data_set.set_extraction_parameters(t0_type, T0_type_compare, max_num_agents)
            
            latex_name = data_set.get_name()['latex']
            if latex_name[:6] == r'\emph{' and latex_name[-1] == r'}':
                latex_name = latex_name[6:-1]
                
            self.Latex_names.append(latex_name) 
            self.Datasets[data_set.get_name()['print']] = data_set
            
        self.single_dataset = len(self.Datasets.keys()) <= 1    
        
        # Borrow dataset paprameters
        self.model_class_to_path       = parameters[0]
        self.num_samples_path_pred     = parameters[1]
        self.enforce_num_timesteps_out = parameters[2]
        self.enforce_prediction_time   = parameters[3]
        self.exclude_post_crit         = parameters[4]
        self.allow_extrapolation       = parameters[5]
        self.agents_to_predict         = parameters[6]
        self.overwrite_results         = parameters[7]
        
        # Get scenario
        scenario_names = []
        for data_set in self.Datasets.values():
            scenario_names.append(data_set.scenario.get_name())
            
        if len(np.unique(scenario_names)) == 1:
            self.scenario = data_set.scenario
        else:
            self.scenario = scenario_none()
        
        # Get relevant scenario information
        self.Behaviors = np.array(list(self.scenario.give_classifications()[0].keys()))
        self.behavior_default = self.scenario.give_default_classification()
        self.classification_useful = len(self.Behaviors) > 1

        # Needed agents
        self.pov_agent = self.scenario.pov_agent()
        if self.pov_agent is not None:
            self.needed_agents = [self.scenario.pov_agent()] + self.scenario.classifying_agents()
        else:
            self.needed_agents = self.scenario.classifying_agents()
        
        assert len(self.needed_agents) > 0, "There must be predictable agents."
    
        # Check if general input is possible
        self.general_input_available = (self.scenario.can_provide_general_input() != False and
                                        self.classification_useful)
        
        # Get a needed name:
        if len(np.unique([data_set.t0_type for data_set in self.Datasets.values()])) == 1:
            t0_type_file_name = {'start':            'start',
                                 'all':              'all_p',
                                 'col_equal':        'fix_e',
                                 'col_set':          'fix_s',
                                 'crit':             'crit_'}
            
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
        
        if hasattr(self, 'Pred_agents_eval') and hasattr(self, 'Pred_agents_pred'):
            del self.Pred_agents_eval
            del self.Pred_agents_pred
        
        if hasattr(self, 'Subgroups') and hasattr(self, 'Path_true_all'):
            del self.Subgroups
            del self.Path_true_all
    
    def change_result_directory(self, filepath, new_path_addon, new_file_addon, file_type = '.npy'):
        return list(self.Datasets.values())[0].change_result_directory(filepath, new_path_addon, new_file_addon, file_type)
    
    
    def determine_required_timesteps(self, num_timesteps):
        return list(self.Datasets.values())[0].determine_required_timesteps(num_timesteps)
    
    
    def set_data_file(self, dt, num_timesteps_in, num_timesteps_out):
        (self.num_timesteps_in_real, 
         self.num_timesteps_in_need)  = self.determine_required_timesteps(num_timesteps_in)
        (self.num_timesteps_out_real, 
         self.num_timesteps_out_need) = self.determine_required_timesteps(num_timesteps_out)
        
        t0_type_name_addon = '_'
        if self.enforce_prediction_time:
            t0_type_name_addon += 's' # s for severe
        else:
            t0_type_name_addon += 'l' # l for lax
            
        if self.enforce_num_timesteps_out:
            t0_type_name_addon += 's'
        else:
            t0_type_name_addon += 'l'
        
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
        
        
        self.data_file = (list(self.Datasets.values())[0].path + os.sep + 'Results' + os.sep +
                          self.get_name()['print'] + os.sep +
                          'Data' + os.sep +
                          self.get_name()['file'] +
                          '--t0=' + self.t0_type_name + t0_type_name_addon +
                          '--dt=' + '{:0.2f}'.format(max(0, min(9.99, dt))).zfill(4) +
                          '_nI=' + str(self.num_timesteps_in_real).zfill(2) + 
                          'm' + str(self.num_timesteps_in_need).zfill(2) +
                          '_nO=' + str(self.num_timesteps_out_real).zfill(2) + 
                          'm' + str(self.num_timesteps_out_need).zfill(2) +
                          '_EC' * self.exclude_post_crit + '_IC' * (1 - self.exclude_post_crit) +
                          '--max_' + str(num).zfill(3) + '_agents_' + pat +
                          '.npy')
    
    def get_data(self, dt, num_timesteps_in, num_timesteps_out):
        self.set_data_file(dt, num_timesteps_in, num_timesteps_out)
        complete_failure = None
        for data_set in self.Datasets.values():
            data_failure = data_set.get_data(dt, num_timesteps_in, num_timesteps_out)
            if data_failure is not None:
                complete_failure = data_failure[:-1] + ' (in ' + data_set.get_name()['print'] + ').'
            
        # Set up ten required attributes resulting from this function
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
        
        self.num_behaviors = np.zeros(len(self.Behaviors), int)
        
        for data_set in self.Datasets.values():
            self.Input_path       = pd.concat((self.Input_path, data_set.Input_path))
            self.Input_T          = np.concatenate((self.Input_T, data_set.Input_T), axis = 0)
            
            self.Output_path      = pd.concat((self.Output_path, data_set.Output_path))
            self.Output_T_E       = np.concatenate((self.Output_T_E, data_set.Output_T_E), axis = 0)
            
            self.Type             = pd.concat((self.Type, data_set.Type))
            self.Recorded         = pd.concat((self.Recorded, data_set.Recorded))
            self.Domain           = pd.concat((self.Domain, data_set.Domain))
            
            if self.scenario.get_name() != data_set.scenario.get_name():
                assert self.scenario.get_name() == 'No specific scenario'
                
                num_samples = len(data_set.Output_T)
                OA = pd.DataFrame(np.ones((num_samples, 1), bool), columns = self.Behaviors)
                self.Output_A         = pd.concat((self.Output_A, OA))
                
                IP = pd.DataFrame(np.ones((num_samples, 1), float) * np.nan, columns = ['empty'])
                self.Input_prediction = pd.concat((self.Input_prediction, IP))
                
                OT  = data_set.Output_T
                OTP = data_set.Output_T_pred
                for i in range(num_samples):
                    OT[i]  = OT[i][:data_set.num_timesteps_out_real]
                    OTP[i] = OTP[i][:data_set.num_timesteps_out_real]
                    
                self.Output_T      = np.concatenate((self.Output_T, OT), axis = 0)
                self.Output_T_pred = np.concatenate((self.Output_T_pred, OTP), axis = 0)
                
                
                self.num_behaviors += data_set.num_behaviors.sum()
                
            else:
                self.Output_A         = pd.concat((self.Output_A, data_set.Output_A))
                self.Output_T_pred    = np.concatenate((self.Output_T_pred, data_set.Output_T_pred), axis = 0)
                self.Output_T         = np.concatenate((self.Output_T, data_set.Output_T), axis = 0)
                self.Input_prediction = pd.concat((self.Input_prediction, data_set.Input_prediction))
                
                self.num_behaviors += data_set.num_behaviors
        
        
        # Ensure that the same order of agents is maintained for input and output paths
        self.Output_path = self.Output_path[self.Input_path.columns]
        
        # Ensure matching indices
        self.Input_prediction = self.Input_prediction.reset_index(drop = True)
        self.Input_path       = self.Input_path.reset_index(drop = True)
        self.Output_path      = self.Output_path.reset_index(drop = True)
        self.Output_A         = self.Output_A.reset_index(drop = True)
        self.Type             = self.Type.reset_index(drop = True)
        self.Recorded         = self.Recorded.reset_index(drop = True)
        self.Domain           = self.Domain.reset_index(drop = True)
        
        # Ensure agent order is the same everywhere
        Agents = np.array(self.Input_path.columns)
        self.Output_path = self.Output_path[Agents]
        self.Type        = self.Type[Agents]
        self.Recorded    = self.Recorded[Agents]
        
        # Ensure behavior stuff is aligned
        self.Output_A = self.Output_A[self.Behaviors]
        
        
        # Set final values
        self.data_loaded = True
        self.dt = dt
        
        return complete_failure
        
        
    
    def get_Target_MeterPerPx(self, domain):
        return self.Datasets[domain['Scenario']].Images.Target_MeterPerPx.loc[domain.image_id]
    
    
    def return_batch_images(self, domain, center, rot_angle, target_width, target_height, 
                            grayscale = False, return_resolution = False):
        if target_height is None:
            target_height = 500
        if target_width is None:
            target_width = 500
            
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
            Uses.append(use)
            
            # Get output_data for the specific dataset
            output_d_data = []
            for out in output_data:
                assert isinstance(out, pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
                output_d_data.append(out[use])
            
            # Get output_d_index
            dataset_index = np.where((self.Domain['Scenario'] == data_set.get_name()['print']).to_numpy())[0]
            if np.all(dataset_index == output_index[use.to_numpy()]):
                output_d_index = np.arange(len(dataset_index))
            else:
                match = dataset_index[np.newaxis] == output_index[use.to_numpy(),np.newaxis]
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
            for i, out in enumerate(output_trans):
                columns.append(out[j].columns)
            columns = np.unique(np.concatenate(columns))
            
            array_type = output_trans[0][j].to_numpy().dtype
            
            output_trans_all.append(pd.DataFrame(np.zeros((len(output_index), len(columns)), array_type),
                                                 columns = list(columns), index = output_index))
            
            for i, out in enumerate(output_trans):
                c_index = (out[j].columns.to_numpy()[:,np.newaxis] == columns[np.newaxis]).argmax(axis = - 1)
                use_index = np.where(Uses[i])[0]
                try:
                    output_trans_all[j].iloc[use_index, c_index] = out[j]
                except:
                    assert False
                
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
                    
                    
    def _determine_pred_agents(self):
        if hasattr(self, 'Pred_agents_eval') and hasattr(self, 'Pred_agents_pred'):
            return
        
        Agents = np.array(self.Recorded.columns)
        Required_agents = np.array([agent in self.needed_agents for agent in Agents])
        Required_agents = np.tile(Required_agents[np.newaxis], (len(self.Recorded), 1))
        
        # Get path data
        self._extract_original_trajectories()
        
        if self.agents_to_predict == 'predefined':
            self.Pred_agents_eval = Required_agents
            self.Pred_agents_pred = Required_agents
        else:
            Recorded_agents = np.zeros(Required_agents.shape, bool)
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
            
            self.Pred_agents_eval = (Correct_type_agents & Required_agents) | Extra_agents
            self.Pred_agents_pred = Required_agents | Extra_agents
        
        
        # NuScenes exemption:
        if ((self.get_name()['print'] == 'NuScenes') and
            (self.num_timesteps_in_real == 4) and 
            (self.num_timesteps_out_real == 12) and
            (self.dt == 0.5) and 
            (self.t0_type == 'all')):
            
            # Get predefined predicted agents for NuScenes
            Pred_agents_N = np.zeros(Required_agents.shape, bool)
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
            
            self.Pred_agents_eval = Pred_agents_N
            self.Pred_agents_pred = Pred_agents_N
            
        # Check if everything needed is there
        assert not np.isnan(self.X_orig[self.Pred_agents_pred]).all((1,2)).any(), 'A needed agent is not given.'
        assert not np.isnan(self.Y_orig[self.Pred_agents_pred]).all((1,2)).any(), 'A needed agent is not given.'
        
        
    def _extract_identical_inputs(self):
        if hasattr(self, 'Subgroups') and hasattr(self, 'Path_true_all'):
            return
        
        # get input trajectories
        self._extract_original_trajectories()
        self._determine_pred_agents()
        
        # Get the same entrie in full dataset
        T = self.Type.to_numpy().astype(str)
        PA_str = self.Pred_agents_eval.astype(str) 
        if hasattr(self.Domain, 'location'):
            Loc = np.tile(self.Domain.location.to_numpy().astype(str)[:,np.newaxis], (1, T.shape[1]))
            Div = np.stack((T, PA_str, Loc), axis = -1)
        else:
            Div = np.stack((T, PA_str), axis = -1)
            
        Div_unique, Div_inverse, Div_counts = np.unique(Div, axis = 0, 
                                                        return_inverse = True, 
                                                        return_counts = True)
        
        # Prepare subgroup saving
        self.Subgroups = np.zeros(len(T), int)
        max_len = 0
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
                D = X[d_index, np.newaxis] - X[np.newaxis]
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
                max_len = max(max_len, len(subgraph))
        
        # check if all samples are accounted for
        assert self.Subgroups.min() > 0
        
        # Get pred agents
        nto = self.num_timesteps_out_real
        
        num_samples = len(self.Pred_agents_eval)
        max_num_pred_agents = self.Pred_agents_eval.sum(1).max()
        
        # Find agents to be predicted first
        i_agent_sort = np.argsort(-self.Pred_agents_eval.astype(float))
        i_agent_sort = i_agent_sort[:,:max_num_pred_agents]
        i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_pred_agents))
        
        # reset prediction agents
        Pred_agents = self.Pred_agents_eval[i_sampl_sort, i_agent_sort]
        
        # Sort future trajectories
        Path_true = self.Y_orig[i_sampl_sort, i_agent_sort, :nto]
        
        # Prepare saving of observerd futures
        self.Path_true_all = np.ones((self.Subgroups.max() + 1, max_len, max_num_pred_agents, nto, 2)) * np.nan 
        
        # Go throug subgroups for data
        for subgroup in np.unique(self.Subgroups):
            # get samples with similar inputs
            s_ind = np.where(subgroup == self.Subgroups)[0]
            
            # Get pred agents
            pred_agents = Pred_agents[s_ind] # shape: len(s_ind) x num_agents
            
            # Get future pbservations for all agents
            path_true = Path_true[s_ind] # shape: len(s_ind) x max_num_pred_agents x nto x 2
            
            # Remove not useful agents
            path_true[~pred_agents] = np.nan
            
            # For some reason using pred_agents here moves the agent dimension to the front
            self.Path_true_all[subgroup,:len(s_ind)] = path_true