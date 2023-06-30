import numpy as np
import pandas as pd
import importlib
from scenario_none import scenario_none
import os

class data_interface(object):
    def __init__(self, data_dicts, parameters):
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
            
            if t0_type in Comp_t0_types:
                T0_type_compare = list(Comp_t0_types).remove(t0_type)
            else:
                T0_type_compare = []
                
            data_set_module = importlib.import_module(data_set_name)
            data_set_class = getattr(data_set_module, data_set_name)
            
            data_set = data_set_class(*parameters)
            data_set.set_prediction_time(t0_type, T0_type_compare)
            
            latex_name = data_set.get_name()['latex']
            if latex_name[:6] == r'\emph{' and latex_name[-1] == r'}':
                latex_name = latex_name[6:-1]
                
            self.Latex_names.append(latex_name) 
            self.Datasets[data_set.get_name()['print']] = data_set
            
        self.single_dataset = len(self.Datasets.keys()) <= 1    
        
        # Borrow dataset paprameters
        self.model_class_to_path      = parameters[0]
        self.num_samples_path_pred    = parameters[1]
        self.enforce_prediction_times = parameters[3]
        self.exclude_post_crit        = parameters[4]
        self.overwrite_results        = parameters[5]
        
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
            
        self.data_loaded = False
        
    
    def reset(self):
        for data_set in self.Datasets.values():
            data_set.reset()
        
        self.data_loaded = False
    
    
    def change_result_directory(self, filepath, new_path_addon, new_file_addon, file_type = '.npy'):
        return list(self.Datasets.values())[0].change_result_directory(filepath, new_path_addon, new_file_addon, file_type)
    
    
    def determine_required_timesteps(self, num_timesteps):
        return list(self.Datasets.values())[0].determine_required_timesteps(num_timesteps)
    
    
    def set_data_file(self, dt, num_timesteps_in, num_timesteps_out):
        (self.num_timesteps_in_real, 
         self.num_timesteps_in_need)  = self.determine_required_timesteps(num_timesteps_in)
        (self.num_timesteps_out_real, 
         self.num_timesteps_out_need) = self.determine_required_timesteps(num_timesteps_out)
        
        self.exclude_post_crit      = list(self.Datasets.values())[0].exclude_post_crit
        
        if self.enforce_prediction_times:
            t0_type_name_addon = '_s'
        else:
            t0_type_name_addon = '_l'
        
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

        self.Domain           = pd.DataFrame(np.zeros((0,0), np.ndarray))
        
        self.num_behaviors = np.zeros(len(self.Behaviors), int)
        
        for data_set in self.Datasets.values():
            self.Input_path       = pd.concat((self.Input_path, data_set.Input_path))
            self.Input_T          = np.concatenate((self.Input_T, data_set.Input_T), axis = 0)
            
            self.Output_path      = pd.concat((self.Output_path, data_set.Output_path))
            self.Output_T_E       = np.concatenate((self.Output_T_E, data_set.Output_T_E), axis = 0)
            
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
        self.Domain           = self.Domain.reset_index(drop = True)
        
        self.data_loaded = True
        return complete_failure
        
        
    
    def get_Target_MeterPerPx(self, domain):
        return self.Datasets[domain['Scenario']].Images.Target_MeterPerPx.loc[domain.image_id]
    
    
    def return_batch_images(self, domain, center, rot_angle, target_width, target_height, grayscale = False):
        if grayscale:
            Imgs = np.zeros((len(domain), target_height, target_width, 1), dtype = 'uint8')
        else:
            Imgs = np.zeros((len(domain), target_height, target_width, 3), dtype = 'uint8')
            
        for data_set in self.Datasets.values():
            if data_set.includes_images():
                print('')
                print('Rotate images from dataset ' + data_set.get_name()['print'])
                
                Use = (domain['Scenario'] == data_set.get_name()['print']).to_numpy()

                Imgs[Use] = data_set.return_batch_images(domain.iloc[Use], center[Use], rot_angle[Use], 
                                                         target_width, target_height, grayscale)
        return Imgs
    
    
    def transform_outputs(self, output, model_pred_type, metric_pred_type, pred_save_file):
        if not self.data_loaded:
            raise AttributeError("Input and Output data has not yet been specified.")
            
        output_trans = []
        Uses = []
        for i, data_set in enumerate(self.Datasets.values()):
            use = self.Domain['Scenario'] == data_set.get_name()['print']
            output_d = []
            for out in output:
                assert isinstance(out, pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
                output_d.append(out[use])
            Uses.append(use)
            ds_save_file = pred_save_file.replace(os.sep + self.get_name()['file'], os.sep + data_set.get_name()['file'])
            output_trans.append(data_set.transform_outputs(output_d, model_pred_type, metric_pred_type, ds_save_file))
        
        output_trans_all = []
        for j in range(len(output_trans[0])):
            assert isinstance(output_trans[0][j], pd.DataFrame), 'Predicted outputs should be pandas.DataFrame'
            # Collect columns
            columns = []
            for i, out in enumerate(output_trans):
                columns.append(out[j].columns)
            columns = np.unique(np.concatenate(columns))
            
            array_type = output_trans[0][j].to_numpy().dtype
            
            output_trans_all.append(pd.DataFrame(np.zeros((len(self.Domain), len(columns)), array_type),
                                                 columns = list(columns)))
            
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
            
            names = {'print': 'Combined dataset (' + r' & '.join(self.Latex_names) + ')',
                     'file': file_name,
                     'latex': r'/'.join(self.Latex_names)}
            return names
            
            
            