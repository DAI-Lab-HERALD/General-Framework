import pandas as pd
import numpy as np
import os
import torch
import warnings
import scipy as sp
import importlib
import psutil
import shutil
from pathlib import Path
from utils.memory_utils import get_total_memory, get_used_memory

from rome.ROME import ROME

class model_template():
    def __init__(self, model_kwargs, data_set, splitter, evaluate_on_train_set, behavior = None):
        # Set model kwargs
        self.model_kwargs = model_kwargs
        
        # Check if model can calculate probabilities
        self.predict_path_probs = False
        if self.get_output_type()[:4] == 'path':
            # Needs to be path prediction model
            if hasattr(self, 'provides_likelihoods'):
                # The function giving this information needs to exists
                if self.provides_likelihoods():
                    self.predict_path_probs = True
                    assert hasattr(self, 'calculate_log_likelihoods'), "If the model provides likelihoods, the function self.calculate_log_likelihoods() has to be defined."
        
        if data_set is not None:
            # Load gpu
            if self.requires_torch_gpu():
                # Check special settings
                look_for_gpu = True
                if 'gpu' in model_kwargs.keys():
                    if model_kwargs['gpu'] == False:
                        look_for_gpu = False
                
                if torch.cuda.is_available() and look_for_gpu:
                    self.device = torch.device('cuda', index=0) 
                    torch.cuda.set_device(0)
                else:
                    if look_for_gpu:
                        warnings.warn('''No GPU could be found. Program is proccessed on the CPU.
                                  
                        This might lead either to a model failure or significantly decreased
                        training and prediction speed.''')
                        
                    self.device = torch.device('cpu')
            
            
            self.data_set = data_set
            self.splitter = splitter
            
            # Get overwrite decision
            self.model_overwrite      = self.data_set.overwrite_results in ['model']
            self.prediction_overwrite = self.data_set.overwrite_results in ['model', 'prediction']
            
            self.dt = data_set.dt
            self.has_map = self.data_set.includes_images()
            self.has_graph = self.data_set.includes_sceneGraphs() 
            
            self.num_timesteps_in = data_set.num_timesteps_in_real
            self.num_timesteps_out = data_set.num_timesteps_out_real
            
            self.agents_to_predict = data_set.agents_to_predict
            self.general_input_available = self.data_set.general_input_available
            
            self.t_e_quantile = self.data_set.p_quantile
                
            
            self.num_samples_path_pred = self.data_set.num_samples_path_pred
            self.evaluate_on_train_set = evaluate_on_train_set
    
            if behavior == None:
                self.is_data_transformer = False
                
                if hasattr(self.splitter, 'Train_index'):
                    self.Index_train = self.splitter.Train_index
                        
                    self.model_file = data_set.change_result_directory(splitter.split_file,
                                                                       'Models', self.get_name()['file'])
                    
                    # Check if this is a fintuned model
                    if 'pretrained' in self.model_kwargs.keys():
                        pretrained_path = self.model_kwargs['pretrained']
                        
                        assert isinstance(pretrained_path, str), 'The pretrained model should be a string.'

                        # Check that string is either a full path, or that it starts with results
                        if pretrained_path.startswith('Results'):
                            pretrained_path = self.model_file.split('Results')[0] + pretrained_path

                        # Check if the pretrained model exists
                        if not os.path.isfile(pretrained_path):
                            print('The desired pretrained model, defined to be found at the path\n"' + pretrained_path 
                                    + '"\nwas not found. Training will continue with random parameter initialization.')
                            pretrained_path = None
                            
                        self.model_kwargs['pretrained'] = pretrained_path

                        # get the last folder of the pretrained model indicating the dataset folder
                        if pretrained_path is not None:
                            pretrained_folder = Path(pretrained_path).parent.parent.name
                            self.pretrained_string = '--pretrain_' + pretrained_folder

                            self.model_file = self.model_file[:-4] + self.pretrained_string + '.npy'
                    
                    self.model_file_metric = self.model_file + ''
                    if '_pert=' in self.model_file:
                        pert_split = self.model_file.split('_pert=')
                        self.model_file = pert_split[0] + '_pert=' + pert_split[1][0] + pert_split[1][2:]
                        
                        # If model is trained on unperturbed data, remove perturbation name from model file
                        if pert_split[1][0] == '0':
                            if '--Pertubation_' in self.model_file:
                                pert_name_split = self.model_file.split('--Pertubation_')
                                self.model_file = pert_name_split[0] + pert_name_split[1][3:]

                    self.simply_load_results = False
                    # Get the prediction location file
                    self.pred_loc_file = self.data_set.change_result_directory(self.model_file_metric, 'Predictions')
                else:
                    self.simply_load_results = True
                
            else:
                self.is_data_transformer = True
                self.simply_load_results = False
                
                self.Index_train = np.where(data_set.Output_A[behavior])[0]
                
                if len(self.Index_train) == 0:
                    # There are no samples of the required behavior class
                    # Use those cases where the other behaviors are the furthers away
                    num_long_index = int(len(data_set.Output_A) / len(data_set.Behaviors))
                    self.Index_train = np.argsort(data_set.Output_T_E)[-num_long_index :]
                
                self.model_file = data_set.data_file[:-4] + '-transform_path_(' + behavior + ').npy'
                self.model_file_metric = self.model_file + ''
                
                self.pred_loc_file = self.model_file[:-4] + '--Predictions.npy'
                        
            # Set trained to flase, this prevents a prediction on an untrained model
            self.trained = False
            self.extracted_data = False
            self.depict_results = False

            # Initialize actual model
            if not self.simply_load_results:
                self.setup_method()
        else:
            self.depict_results = True
    

    def load_pretrained_model(self, pretrained_path):
        if pretrained_path is not None:
            assert 'pretrain' in self.model_file, 'Model file does not match a pretrained model.'
            ## Copy over file from old model
            # Find all files starting with model_file_option[:-4] and copy them to the self.model_file[:-4] equivalent
            option_folder = os.path.dirname(pretrained_path)
            option_file = os.path.basename(pretrained_path)[:-4]
            actual_folder = os.path.dirname(self.model_file)
            actual_file = os.path.basename(self.model_file)[:-4]

            for file in os.listdir(option_folder):
                if file.startswith(option_file):
                    old_path = os.path.join(option_folder, file)
                    new_path = os.path.join(actual_folder, file.replace(option_file, actual_file))
                    if os.path.isfile(old_path):
                        if not os.path.exists(new_path):
                            shutil.copyfile(old_path, new_path)
                    else:
                        shutil.copytree(old_path, new_path, dirs_exist_ok = True)
            

            # Load the model
            self.weights_saved = list(np.load(self.model_file, allow_pickle = True)[:-1])
            self.load_method()

            # Delete the new files, as the model is technically not trained yet
            for file in os.listdir(actual_folder):
                if file.startswith(actual_file):
                    old_path = os.path.join(actual_folder, file)
                    if os.path.isfile(old_path):
                        os.remove(old_path)
                    else:
                        shutil.rmtree(old_path, ignore_errors=True)
                    

    def train_actual(self):
        # Check if a pretrained model should be fitted
        if 'pretrained' in self.model_kwargs.keys():
            pretrained_path = self.model_kwargs['pretrained']
            self.load_pretrained_model(pretrained_path)

            
        # creates /overrides self.weights_saved
        self.train_method() 
        #0 is there to avoid some numpy load and save errros
        save_data = np.array(self.weights_saved + [0], object) 
        
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        np.save(self.model_file, save_data)
        
        if self.save_params_in_csv():
            for i, param in enumerate(self.weights_saved):
                param_name = [name for name in vars(self) if np.array_equal(getattr(self, name), param)][0]
                if i == 0:
                    np.savetxt(self.model_file[:-4] + '.txt', param,
                            fmt       = '%10.5f',
                            delimiter = ', ',
                            header    = param_name + ':',
                            footer    = '\n \n',
                            comments  = '')
                else:
                    with open(self.model_file[:-4] + '.txt', 'a') as f:
                        np.savetxt(f, param,
                                fmt       = '%10.5f',
                                delimiter = ', ',
                                header    = param_name + ':',
                                footer    = '\n \n',
                                comments  = '')
        
        if self.provides_epoch_loss() and hasattr(self, 'train_loss'):
            self.loss_file = self.model_file[:-4] + '--train_loss.npy'
            assert isinstance(self.train_loss, np.ndarray), "The train loss should be a numpy array."
            assert len(self.train_loss.shape) == 2, "The train loss should be a 2D numpy array."
            np.save(self.loss_file, self.train_loss.astype(np.float32))
    
    def train(self):
        assert not self.simply_load_results, 'This model instance is nonly for loading results.'
        self.model_mode = 'train'
        if os.path.isfile(self.model_file) and not self.model_overwrite:
            self.weights_saved = list(np.load(self.model_file, allow_pickle = True)[:-1])
            self.load_method()
        
        else:
            # Check if the model maybe is saved under different file name
            if (not self.is_data_transformer) and (self.splitter.split_file_option is not None):
                print('    Looking for model trained under previous experiment.', flush = True)
                # Get the corresponding model file
                model_file_option = self.data_set.change_result_directory(self.splitter.split_file_option,
                                                                          'Models', self.get_name()['file'])
                
                
                if '_pert=' in model_file_option:
                    pert_split = model_file_option.split('_pert=')
                    model_file_option = pert_split[0] + '_pert=' + pert_split[1][0] + pert_split[1][2:]
                    
                    # If model is trained on unperturbed data, remove perturbation name from model file
                    if pert_split[1][0] == '0':
                        if '--Pertubation_' in model_file_option:
                            pert_name_split = model_file_option.split('--Pertubation_')
                            model_file_option = pert_name_split[0] + pert_name_split[1][3:]
                
                print('    Model file option:', model_file_option, flush = True)
                # Check if that model exists
                if os.path.isfile(model_file_option) and not self.model_overwrite:

                    ## Copy over file from old model
                    # Find all files starting with model_file_option[:-4] and copy them to the self.model_file[:-4] equivalent
                    option_folder = os.path.dirname(model_file_option)
                    option_file = os.path.basename(model_file_option)[:-4]
                    actual_folder = os.path.dirname(self.model_file)
                    actual_file = os.path.basename(self.model_file)[:-4]

                    for file in os.listdir(option_folder):
                        if file.startswith(option_file):
                            old_path = os.path.join(option_folder, file)
                            new_path = os.path.join(actual_folder, file.replace(option_file, actual_file))
                            if os.path.isfile(old_path):
                                if not os.path.exists(new_path):
                                    shutil.copyfile(old_path, new_path)
                            else:
                                shutil.copytree(old_path, new_path, dirs_exist_ok = True)


                    # Load the model
                    self.weights_saved = list(np.load(self.model_file, allow_pickle = True)[:-1])
                    self.load_method()
                
                else:
                    self.train_actual()
            
            else:
                self.train_actual()
                
           
        self.trained = True
        
        print('')
        print('The model ' + self.get_name()['print'] + ' was successfully trained.')
        print('')
        return self.model_file
        
    def predict_actual(self, Index = None):
        # Reset prediction analysis
        self.reset_prediction_analysis()

        if Index is not None:
            self.Index_test = Index

        self.model_mode = 'pred'

        # apply model to test samples
        if self.get_output_type()[:4] == 'path':
            self.create_empty_output_path()
            if len(self.Index_test) > 0:
                self.predict_method()
            output = [self.Index_test, self.Output_path_pred, self.Output_path_pred_probs]
        elif self.get_output_type() == 'class':
            self.create_empty_output_A()
            if len(self.Index_test) > 0:
                self.predict_method()
            output = [self.Index_test, self.Output_A_pred]
        elif self.get_output_type() == 'class_and_time':
            self.create_empty_output_A()
            self.create_empty_output_T()
            if len(self.Index_test) > 0:
                self.predict_method()
            output = [self.Index_test, self.Output_A_pred, self.Output_T_E_pred]
        else:
            raise TypeError("This output type for models is not implemented.")
        return output
        
    def reset_prediction_analysis(self):
        assert not self.simply_load_results, 'This model instance is nonly for loading results.'
        # Reset potential batch extraction
        if hasattr(self, 'Ind_pred'):
            del self.Ind_pred

        # Delete potential old results
        if hasattr(self, 'Path_pred'):
            del self.Path_pred
            del self.Path_true
        
        if hasattr(self, 'Log_prob_joint_pred'):
            del self.Log_prob_joint_pred
            del self.Log_prob_joint_true
        
        if hasattr(self, 'Log_prob_indep_pred'):
            del self.Log_prob_indep_pred
            del self.Log_prob_indep_true
        
        if hasattr(self, 'Log_prob_true_indep_pred'):
            del self.Log_prob_true_indep_pred
        
        if hasattr(self, 'Log_prob_true_joint_pred'):
            del self.Log_prob_true_joint_pred
            
        if hasattr(self, 'Type'):
            del self.Type
            del self.Size



    def Sort_Metrics(self, Metric_dict_list, print_status_function):
        # Check if train and test set are identical
        identical_test_set = np.array_equal(np.unique(self.splitter.Train_index), np.unique(self.splitter.Test_index))

        # Go through Metric_list and check if any of them are allready evaluated, if so, remove them from the list
        Metric_train_list = []
        Metric_test_list  = []
        for metric_dict in Metric_dict_list:
            # Get metric name
            metric_name = metric_dict['metric']
            metric_kwargs = metric_dict['kwargs']

            # Get metric class
            metric_module = importlib.import_module(metric_name)
            metric_class = getattr(metric_module, metric_name)  
            
            # Initialize the metric
            metric = metric_class(metric_kwargs, self.data_set, self.splitter, self)
                
            # Test if metric is applicable
            metric_failure = metric.check_applicability()
        
            # print metric status to output
            print_status_function(metric, metric_failure)
            
            # Do not use metric if it cannot be applied
            if metric_failure is not None:
                continue

            # Get metric save file
            metric_file = metric.data_set.change_result_directory(self.model_file_metric, 'Metrics', metric.get_name()['file'])

            if os.path.isfile(metric_file) and not metric.metric_override:
                # Load results to check if the train set is allready evaluated
                Results = list(np.load(metric_file, allow_pickle = True)[:-1])
                if (Results[0] is None) and self.evaluate_on_train_set:
                    # Add the training part to required list
                    Metric_train_list.append(metric)
            else:
                # check if we need to evaluate on train set
                if self.evaluate_on_train_set and not identical_test_set:
                    # Add train set to metric list
                    Metric_train_list.append(metric)
                
                # Add test set to metric list
                Metric_test_list.append(metric)

        Metric_dict = {'Train': Metric_train_list, 'Test': Metric_test_list}
        Result_dict = {'Train': {}, 'Test': {}}

        # Go through needed metrics and sort by metric type
        for mode, Metric_list in Metric_dict.items():
            Metric_type_dict = {}
            Result_mode_dict = {}
            for metric in Metric_list:
                metric_type = metric.get_output_type()
                if metric_type not in Metric_type_dict.keys():
                    Metric_type_dict[metric_type] = []
                Metric_type_dict[metric_type].append(metric)
                Result_mode_dict[metric] = {'Value': [], 'Weight': []}
            
            Metric_dict[mode] = Metric_type_dict
            Result_dict[mode] = Result_mode_dict
        
        return Metric_dict, Result_dict, identical_test_set


    def add_to_index_df(self, Index_df, Index_file, num_samples_file, model_type, file_index = 0):
        # get subgroups of Index file
        eval_pov = model_type != 'path_all_wo_pov'
        self.data_set._group_indentical_inputs(eval_pov = eval_pov)
        Subgroups = self.data_set.Subgroups[Index_file]

        # Sort Index file by subgroups
        subgroup_sort_ind = np.argsort(Subgroups)

        Subgroups  = Subgroups[subgroup_sort_ind]
        Index_file = Index_file[subgroup_sort_ind]

        # Get locations at which the subgroup changes
        subgroup_change = np.array(list(np.where(Subgroups[:-1] != Subgroups[1:])[0] + 1) + [len(Subgroups)])
            
        # Predict the number of origdata size needed for output
        parts_needed_rel = (self.num_timesteps_out * self.num_samples_path_pred) / (self.num_timesteps_out + self.num_timesteps_in)
        
        parts_needed = int(np.ceil(2.5 * parts_needed_rel * len(Index_file) / num_samples_file)) # The 2.5 is a safety margin

        # Get number of samples that can actually be predicted
        index_length = int(np.ceil(len(Index_file) / parts_needed))

        i_min = 0

        while i_min < len(Index_file):
            # Get upper bound for index
            i_max_low = i_min + index_length
            i_max = subgroup_change[subgroup_change >= i_max_low]
            if len(i_max) == 0:
                i_max = len(Index_file)
            else:
                i_max = min(i_max[0], len(Index_file))

            # Get the current indices
            Index = Index_file[i_min:i_max]

            if len(Index) == 0:
                    continue
            
            # Add to dataframe
            Index_series = pd.Series({'Index': Index, 'file_index': file_index})
            Index_df.loc[len(Index_df)] = Index_series

            # reset i_min
            i_min = i_max + 0
    
        return Index_df



    def get_index_df(self, Index_all, model_type):
        Index_df = pd.DataFrame(np.zeros((0,2), object), columns = ['Index', 'file_index'])

        if self.data_set.data_in_one_piece:
            # get the curent train/test index
            num_samples_file = len(self.data_set.Domain)
            Index_file = Index_all
            
            Index_df = self.add_to_index_df(Index_df, Index_file, num_samples_file, model_type)

        else:
            # Go through all the files corresponding to Index_all
            potential_files = np.unique(self.data_set.Domain.iloc[Index_all].file_index)

            for file_index in potential_files:
                # Get actual Index
                useful = (self.data_set.Domain.file_index == file_index) 
                num_samples_file = np.sum(useful)

                # get the curent train/test index
                useful &= np.in1d(np.arange(len(self.data_set.Domain)), Index_all)
                Index_file = np.where(useful)[0]

                Index_df = self.add_to_index_df(Index_df, Index_file, num_samples_file, model_type, file_index)

        return Index_df
    

    def append_result_dict(self, mode, metric, Result_dict, result, Index, Index_df):
        # Get weights
        metric_combination = metric.partial_calculation()

        if metric_combination == 'Sample':
            weight = len(Index)

        elif metric_combination == 'Subgroups':
            Subgroups = self.data_set.Subgroups[Index]
            weight = len(np.unique(Subgroups))

        elif metric_combination == 'Pred_agents':
            weight = self.data_set.Pred_agents_eval[Index].sum()

        elif metric_combination == 'Subgroup_pred_agents':
            Subgroups = self.data_set.Subgroups[Index]
            Subgroup_unique_index = np.unique(Subgroups, return_index = True)[1]
            weight = self.data_set.Pred_agents_eval[Index[Subgroup_unique_index]].sum()

        elif metric_combination == 'Min':
            if len(Result_dict[mode][metric]['Weight']) > 0:
                old_value_min = min(Result_dict[mode][metric]['Value'])
                if result[0] < old_value_min:
                    weight = 1
                    Result_dict[mode][metric]['Weight'] = [0] * len(Result_dict[mode][metric]['Weight'])
                else:
                    weight = 0
            else:
                weight = 1

        elif metric_combination == 'Max':
            if len(Result_dict[mode][metric]['Weight']) > 0:
                old_value_max = max(Result_dict[mode][metric]['Value'])
                if result[0] > old_value_max:
                    weight = 1
                    Result_dict[mode][metric]['Weight'] = [0] * len(Result_dict[mode][metric]['Weight'])
                else:
                    weight = 0
            else:
                weight = 1

        else:
            weight = 1
            assert len(Index_df) == 1, 'Partial evaluation is not possible for metric ' + metric.get_name()['print'] + '.'


        Result_dict[mode][metric]['Value'].append(result)
        Result_dict[mode][metric]['Weight'].append(weight)

        return Result_dict
    
    def save_metric_results(self, Result_dict, identical_test_set = False):
        for mode, Result_mode_dict in Result_dict.items():
            for metric, Result in Result_mode_dict.items():
                metric_file = metric.data_set.change_result_directory(self.model_file_metric, 'Metrics', metric.get_name()['file'])
                
                if os.path.isfile(metric_file):
                    Results = list(np.load(metric_file, allow_pickle = True)[:-1])
                else:
                    Results = [None, None]


                values  = Result['Value']
                weights = np.array(Result['Weight'])    

                if len(values) == 0:
                    raise ValueError('No values were calculated for metric ' + metric.get_name()['print'] + '.')
                elif len(values) == 1:
                    result = values[0]
                else:
                    if len(values[0]) == 1:
                        values_array = np.array(values)[:,0]
                        # Get average value
                        result = [np.average(values_array, weights = weights)]
                    else:
                        result = metric.combine_results(values, weights)
                    

                # Overwrite Results
                if not identical_test_set:
                    if mode == 'Train':
                        result_index = 0
                    else:
                        result_index = 1
                    Results[result_index] = result
                else:
                    Results = [result, result]

                # Save results
                print('The metric ' + metric.get_name()['print'] + ' was successfully evaluated.')
                save_data = np.array(Results + [0], object) # 0 is there to avoid some numpy load and save errros
                os.makedirs(os.path.dirname(metric_file), exist_ok=True)
                np.save(metric_file, save_data)

    
    def predict_Index(self, Index, model_type, file_index = 0):
        # Get original data
        self.data_set._extract_original_trajectories(file_index = file_index)

        # Load potential output
        Index_loaded, Index_missing, output_loaded = self.load_predictions(Index, model_type)

        # Make predictions on the given testing set
        output_missing = self.predict_actual(Index_missing) 

        # Save predictions if allowed
        if self.data_set.save_predictions:
            self.save_made_predictions(Index_missing, output_missing, model_type)

        # Combine loaded and made predictions
        output = self.combine_loaded_and_made_predictions(Index, Index_loaded, Index_missing, output_loaded, output_missing, model_type)
        
        return output

        
        
    

    def predict_and_evaluate(self, Metric_dict_list, print_status_function):
        assert not self.depict_results, 'This model instance is only for loading results.'
        assert self.data_set is not None, 'This model instance is only for loading results.'

        # Preselct the metrics based on their existence and other requirements
        Metric_dict, Result_dict, identical_test_set = self.Sort_Metrics(Metric_dict_list, print_status_function)

        # Get the type of prediction in this output
        model_type = self.get_output_type()

        # Go through needed metrics and sort by metric type
        for mode, Metric_type_dict in Metric_dict.items():
            if mode == 'Train':
                Index_all = self.splitter.Train_index
            else:
                Index_all = self.splitter.Test_index
            
            if len(Metric_type_dict) == 0:
                continue
            # Get the index dataframe
            Index_df = self.get_index_df(Index_all, model_type)
            
            for i_index in range(len(Index_df)):
                Index = Index_df.iloc[i_index].Index
                file_index = Index_df.iloc[i_index].file_index
                
                # Get output
                output = self.predict_Index(Index, model_type, file_index)

                for metric_type, Metric_list in Metric_type_dict.items():
                    output_trans = self.transform_output(output, Index, model_type, metric_type)

                    for metric in Metric_list:
                        # Evaluate metric
                        result = metric._evaluate_on_subset(output_trans, Index)

                        # Append results to result dict
                        Result_dict = self.append_result_dict(mode, metric, Result_dict, result, Index, Index_df)

        self.save_metric_results(Result_dict, identical_test_set)


    def transform_output(self, output, Index, model_type, metric_type):
        if metric_type == model_type:
            output_trans = output
        else:
            # Load potential transformed output
            Index_loaded, Index_missing, output_trans_loaded = self.load_predictions(Index, metric_type)

            # Transform output to only use missing index
            used_index = self.data_set.get_indices_1D(Index_missing, Index)
            output_missing = []
            for out in output:
                if isinstance(out, np.ndarray):
                    output_missing.append(out[used_index])
                elif isinstance(out, pd.DataFrame):
                    output_missing.append(out.iloc[used_index])
                else: 
                    raise TypeError('Output type not implemented.')

            if len(Index_missing) > 0:
                output_trans_missing = self.data_set.transform_outputs(output_missing, model_type, metric_type)

                # Save transformed predictions if allowed
                if self.data_set.save_predictions:
                    self.save_made_predictions(Index_missing, output_trans_missing, metric_type)

                # Combine loaded and made predictions
                output_trans = self.combine_loaded_and_made_predictions(Index, Index_loaded, Index_missing, output_trans_loaded, output_trans_missing, metric_type) 
            else:
                output_trans = output_trans_loaded
        return output_trans

    def combine_loaded_and_made_predictions(self, Index, Index_loaded, Index_made, output_loaded, output_made, pred_type):
        # Check if index worked
        assert (output_loaded[0] == Index_loaded).all(), 'Loaded index is not correct.'
        assert (output_made[0] == Index_made).all(), 'Made index is not correct.'

        assert np.array_equal(np.unique(Index), np.unique(np.concatenate((Index_loaded, Index_made)))), 'Indices do not overlap.' 
        
        assert not np.in1d(Index_loaded, Index_made).any(), 'Made and loaded samples overlap.'
        
        # Start assembling output1
        output = [Index]

        # Go through different pred types and check outputs
        if pred_type[:4] == 'path':
            assert len(output_loaded) == 3, 'Loaded output is not correct.'
            assert len(output_made) == 3, 'Made output is not correct.'
        elif pred_type == 'class':
            assert len(output_loaded) == 2, 'Loaded output is not correct.'
            assert len(output_made) == 2, 'Made output is not correct.'
        elif pred_type == 'class_and_time':
            assert len(output_loaded) == 3, 'Loaded output is not correct.'
            assert len(output_made) == 3, 'Made output is not correct.'
        else:
            raise TypeError('This type of prediction is not implemented.')
        
        for i in range(1, len(output_loaded)):
            out_loaded = output_loaded[i]
            out_made = output_made[i]

            # Check if this are pandas dataframes
            assert isinstance(out_loaded, pd.DataFrame), 'Loaded output is not correct.'
            assert isinstance(out_made, pd.DataFrame), 'Made output is not correct.'

            # Check if indices work
            assert np.array_equal(Index_loaded, out_loaded.index.to_numpy()), 'Loaded path is not correct.'
            assert np.array_equal(Index_made, out_made.index.to_numpy()), 'Made path is not correct.'

            # Check that columns are the same
            if not np.array_equal(out_loaded.columns, out_made.columns):
                print('Columns are not the same for i = {}.'.format(i))
                print('Loaded columns:', out_loaded.columns)
                print('Made columns:', out_made.columns)
                raise ValueError('Columns are not the same.')

            # Combine the paths in correct order
            out = pd.concat([out_loaded, out_made], axis = 0)
            out = out.loc[Index]

            # Last sanity check
            assert np.array_equal(Index, out.index.to_numpy()), 'Loaded path is not correct.'

            # Add to output
            output.append(out)

        return output


    #################################################################################################
    #                                Loading and saving predictions                                 #
    #################################################################################################
    def load_predictions(self, Index, pred_type):
        
        if not os.path.isfile(self.pred_loc_file) or self.prediction_overwrite:
            Index_missing = Index
            Index_loaded = np.array([], int)

            output_loaded = [Index_loaded]
            # Check requirements for output
            if pred_type[:4] == 'path':
                columns = self.data_set.Agents 
                num_outputs = 2
            elif pred_type == 'class':
                columns = self.data_set.Behaviors
                num_outputs = 1
            elif pred_type == 'class_and_time':
                columns = self.data_set.Behaviors
                num_outputs = 2

            # Prepare empty output
            for _ in range(num_outputs):
                out_loaded = pd.DataFrame(np.zeros((len(Index_loaded), len(columns)), object), columns = columns)
                output_loaded.append(out_loaded)

        else:
            # Check if the prediction location dataframe is allread loaded or not
            if not hasattr(self, 'Pred_locator'):
                if os.path.isfile(self.pred_loc_file):
                    # Load the prediction location dataframe
                    self.Pred_locator = np.load(self.pred_loc_file, allow_pickle = True)[0]
                else:
                    self.Pred_locator = pd.DataFrame(np.empty((len(self.data_set.Domain), 2), object), 
                                                     columns = ['class', 'paths'], index = self.data_set.Domain.index)
            # Pred locator is a dataframe of shape = (len(self.data_set.Domain), 2), with columns 'class' and 'paths'
            if pred_type[:4] == 'path':
                pred_name = 'paths'
                columns = self.data_set.Agents 
                num_outputs = 2
            elif pred_type[:5] == 'class':
                pred_name = 'class'
                columns = self.data_set.Behaviors
                if pred_type == 'class':
                    num_outputs = 1
                else:
                    num_outputs = 2
            else:
                raise TypeError('This type of prediction is not implemented.')
            
            # Get the potential file numbers needed
            pred_file_numbers = self.Pred_locator[pred_name].iloc[Index]

            # Get the missing indices
            missing = pred_file_numbers.isna().to_numpy()
            Index_missing = Index[missing]
            
            # Get the loaded files
            Index_loaded  = Index[~missing]
            pred_file_numbers = pred_file_numbers[~missing].to_numpy()
            
            # Prepare the output
            output_loaded = [Index_loaded]
            for _ in range(num_outputs):
                out_loaded = pd.DataFrame(np.zeros((len(Index_loaded), len(columns)), object), columns = columns, index = Index_loaded)
                output_loaded.append(out_loaded)
            
            # Go through unique pred file numbers
            for pred_file_number in np.unique(pred_file_numbers):
                # Get indices saved in current file
                use_ind = pred_file_numbers == pred_file_number
                Index_loaded_file = Index_loaded[use_ind]
                
                # Get current file
                pred_file = self.pred_loc_file[:-4] + '--' + pred_name + '_' + str(pred_file_number) + '.npy'
                
                # Load the results
                pred_results = np.load(pred_file, allow_pickle = True)
                
                for i in range(num_outputs):
                    if len(pred_results) > i:
                        output_loaded[i + 1].loc[Index_loaded_file] = pred_results[i].loc[Index_loaded_file]

            if len(Index_loaded) > 0:
                # Do some controls for certain data types
                if pred_type[:4] == 'path':
                    output_log_probs = output_loaded[2]
                    missing_new = output_log_probs.isna().to_numpy().all(-1)
                    
                    if pred_type == 'path_all_wi_pov':
                        out_path = output_loaded[1]

                        # Get scenario type
                        scenario_types = self.data_set.Domain.Scenario_type.loc[Index_loaded].to_numpy()
                        assert len(np.unique(scenario_types)) == 1, 'There are multiple scenario types in the same file.'
                        scenario_types = scenario_types[0]

                        i_scenario = np.where(self.data_set.unique_scenarios == scenario_types)[0][0]
                        pov_agent = self.data_set.scenario_pov_agents[i_scenario]

                        if pov_agent is None:
                            missing_new = np.zeros(len(Index_loaded), bool)
                        else:
                            missing_new = out_path[pov_agent].isna()
                
                elif pred_type == 'class_and_time':
                    out_time = output_loaded[2]
                    missing_new = out_time.isna().any(axis = 1)

                else:
                    missing_new = np.zeros(len(Index_loaded), bool)

                if missing_new.any():
                    # Overwrite indices
                    Index_missing = np.concatenate((Index_missing, Index_loaded[missing_new]))
                    Index_loaded  = Index_loaded[~missing_new]

                    # Adjust outptus
                    output_loaded[0] = Index_loaded
                    for i in range(1, len(output_loaded)):
                        output_loaded[i] = output_loaded[i].loc[Index_loaded]   

        return Index_loaded, Index_missing, output_loaded

    def save_made_predictions(self, Index, output, pred_type):
        # Check if the Pred loc file exists
        if not hasattr(self, 'Pred_locator'):
            if os.path.isfile(self.pred_loc_file):
                # Load the prediction location dataframe
                self.Pred_locator = np.load(self.pred_loc_file, allow_pickle = True)[0]
            else:
                self.Pred_locator = pd.DataFrame(np.empty((len(self.data_set.Domain), 2), object), 
                                                 columns = ['class', 'paths'], index = self.data_set.Domain.index)
                
        # Do stuff depending on the pred type
        if pred_type[:4] == 'path':
            pred_name = 'paths'
            columns = self.data_set.Agents 
            num_outputs = 2
            num_outputs_req = 2
            
            # get approximate number of agents
            num_agents_per_sample = self.data_set.Pred_agents_pred_all.sum(-1)
            num_agents = 0.8 * num_agents_per_sample.mean() + 0.2 * num_agents_per_sample.max()
            n_bytes_per_sample = num_agents * (self.num_timesteps_out * 1.25) * 8
            
        elif pred_type[:5] == 'class':
            pred_name = 'class'
            columns = self.data_set.Behaviors
            
            num_outputs_req = 2
            if pred_type == 'class':
                num_outputs = 1
            else:
                num_outputs = 2
                
            # Get available memory 
            n_bytes_per_sample = len(columns) * 4 * (1 + len(self.t_e_quantile))
            
        else:
            raise TypeError('This type of prediction is not implemented.')
        
        # Check if some of the samples are overwrites
        index_exists = self.Pred_locator.loc[Index, pred_name].notna().to_numpy()
        Index_overwrite = Index[index_exists]
        Index_new = Index[~index_exists]
        
        pred_file_numbers_exists = self.Pred_locator.loc[Index_overwrite, pred_name].to_numpy()
        for pred_file_number in np.unique(pred_file_numbers_exists):
            # Get indices saved in current file
            use_ind = pred_file_numbers_exists == pred_file_number
            Index_overwrite_file = Index_overwrite[use_ind]
            
            # Get current file
            pred_file = self.pred_loc_file[:-4] + '--' + pred_name + '_' + str(pred_file_number) + '.npy'
            
            # Load the results
            pred_results = np.load(pred_file, allow_pickle = True)
            
            for i in range(num_outputs):
                pred_results[i].loc[Index_overwrite_file] = output[i + 1].loc[Index_overwrite_file]
            
            # Save the results
            np.save(pred_file, np.array(pred_results, object))
        
        
        # Get the samples that can be saved to one file
        available_memory = self.data_set.total_memory - get_used_memory()
        samples_per_file = int(0.5 * available_memory / n_bytes_per_sample)
        
        # Start saving
        completed_saving = False
        while not completed_saving:
            # Get current unique file numbers and their count
            pred_file_numbers, pred_file_counts = np.unique(self.Pred_locator[pred_name][self.Pred_locator[pred_name].notna()], return_counts = True)
            
            # Check if we can append to an allready existing file
            if (len(pred_file_numbers) > 0) and (0.8 * samples_per_file > pred_file_counts.min()):
                pred_file_number = pred_file_numbers[pred_file_counts.argmin()]
                
                # Get the numbers we need right now
                num_saved = samples_per_file - pred_file_counts.min()
            
            else:
                if len(pred_file_numbers) == 0:
                    pred_file_number = 0
                else:
                    pred_file_number = pred_file_numbers.max() + 1
                
                # Get the number of savable stuff
                num_saved = min(samples_per_file, len(Index_new))
                
            # Get the current file
            pred_file = self.pred_loc_file[:-4] + '--' + pred_name + '_' + str(pred_file_number) + '.npy'
                
            # Get the output saved in this iteration  
            Index_new_saved = Index_new[:num_saved]   
            
            # Load the results
            if os.path.isfile(pred_file):
                pred_results = np.load(pred_file, allow_pickle = True)
            else:
                pred_results = [pd.DataFrame(np.zeros((0, len(columns)), object), columns = columns) for _ in range(num_outputs_req)]
                pred_results = pred_results + [0]
                
                
            # Go through the outputs
            for i in range(num_outputs):
                pred_results[i] = pd.concat([pred_results[i], output[i + 1].loc[Index_new_saved]], axis = 0)
            for i in range(num_outputs, num_outputs_req):
                out_empty = pd.DataFrame(np.empty((len(Index_new_saved), len(columns)), object), columns = columns, index = Index_new_saved)
                pred_results[i] = pd.concat([pred_results[i], out_empty], axis = 0)

            
            # Save the results
            os.makedirs(os.path.dirname(pred_file), exist_ok=True)
            np.save(pred_file, np.array(pred_results, object))    
            
            # Overwrite Pred_locator
            self.Pred_locator.loc[Index_new_saved, pred_name] = pred_file_number
            
            # Overwrite Index_new
            Index_new = Index_new[num_saved:]
            
            # Check if we are done
            if len(Index_new) == 0:
                completed_saving = True
                
                
        # Save pred locator
        os.makedirs(os.path.dirname(self.pred_loc_file), exist_ok=True)
        np.save(self.pred_loc_file, np.array([self.Pred_locator, 0], object))

    

    #################################################################################################
    #################################################################################################
    ###                                                                                           ###
    ###                                     Less important methods not                            ###
    ###                                     called in experiment.py                               ###
    ###                                                                                           ###
    #################################################################################################
    #################################################################################################



    def extract_images(self, X, Img_needed, domain_needed):
        '''
        Returns image data 
        img # num_overall_agents, height, width, channels
        img_m_per_px # num_overall_agents
        '''
        # try/except is unreliable, so we have to check preemtively if enough memory is available
        available_memory = self.data_set.total_memory - get_used_memory()

        # Calculate required memory (img_needed is main culprit, is a u_int8 datatype, X is float32)
        img_size = Img_needed.sum() * self.target_width * self.target_height
        if  not self.grayscale:
            img_size *= 3
        required_memory = np.prod(X.shape) * 4 + img_size

        # Check if enough memory is available
        if required_memory < 0.4 * available_memory:
            # Only use the input positions
            X = X[..., :2].copy()
            centre = X[Img_needed, -1,:].copy()
            x_rel = centre - X[Img_needed, -2,:]
            rot = np.angle(x_rel[:,0] + 1j * x_rel[:,1]) 

            if hasattr(self, 'use_batch_extraction') and self.use_batch_extraction:
                print_progress = False
            else:
                print_progress = True
                

            img_needed, img_m_per_px_needed = self.data_set.return_batch_images(domain_needed, centre, rot,
                                                                                target_height = self.target_height, 
                                                                                target_width = self.target_width,
                                                                                grayscale = self.grayscale,
                                                                                return_resolution = True,
                                                                                print_progress = print_progress)
            use_batch_extraction = False
        else:
            img_needed = None
            img_m_per_px_needed = None
            use_batch_extraction = True

        
        return img_needed, img_m_per_px_needed, use_batch_extraction


    def extract_sceneGraphs(self, domain_needed, X, radius = 100, wave_length = 1.0):
        '''
        Returns scene graph data 
        '''
        # X.shape = (num_samples, num_agents, 2)
        # try/except is unreliable, so we have to check preemtively if enough memory is available
        available_memory = self.data_set.total_memory - get_used_memory()

        # For a given scene graph with radius 100 m, assume 0.2 MB per sample, with this being quadratic to the radius
        required_memory_per_sample = 200 * 2 ** 10 * (radius / 100) ** 2
        required_memory = required_memory_per_sample * len(domain_needed)

        if required_memory < 0.4 * available_memory:
            # Get scene graphs
            if hasattr(self, 'use_graph_batch_extraction') and self.use_graph_batch_extraction:
                print_progress = False
            else:
                print_progress = True

            graph_needed = self.data_set.return_batch_sceneGraphs(domain_needed, X, radius, wave_length, print_progress)
            use_batch_extraction = False
        else:
            graph_needed = None
            use_batch_extraction = True
        
        return graph_needed, use_batch_extraction


    def get_agent_data_to_array(self, Type, Size):
        # Get agent types to array
        T = Type.to_numpy().astype(str)
        T[T == 'nan'] = '0'

        # Get agent size to array
        overwrite = Size.isna().to_numpy()
        S = Size.to_numpy()
        
        # overwrtite nan stuff
        overwrite_array = np.array([np.full(2, np.nan)] * overwrite.sum() + ['test'], dtype = object)[:-1]
        S[overwrite] = overwrite_array
        S = np.stack(S.tolist())

        return T, S


    def _extract_types(self):
        ## NOTE: Method has been adjusted for large datasets
        # Get pred agents
        
        if not hasattr(self, 'Type'):
            assert not hasattr(self, 'Size'), 'Size should be done separately to Types.'
            if self.data_set.data_in_one_piece:
                # Get agent types to array
                T, S = self.get_agent_data_to_array(self.data_set.Type, self.data_set.Size)
            else:
                T = np.full(self.data_set.Pred_agents_pred_all.shape, '0', str)
                S = np.full((*self.data_set.Pred_agents_pred_all.shape, 2), np.nan)
                for file_index in range(len(self.data_set.Files)):
                    used = self.data_set.Domain.file_index == file_index
                    used_index = np.where(used)[0]
                    
                    agent_file = self.data_set.Files[file_index] + '_AM.npy'
                    agent_data = np.load(agent_file, allow_pickle = True)
                    if len(agent_data) == 3:
                        T_local, _, _ = agent_data
                        S_local = self.data_set.set_default_size(T_local)
                    else:
                        assert len(agent_data) == 4, 'Agent data should be of length 3 or 4.'
                        T_local, S_local, _, _ = agent_data
                    
                    # Get agent inices
                    agent_index = self.data_set.get_indices_1D(np.array(T_local.columns), np.array(self.data_set.Agents))
                    
                    # get agent types top array
                    T_local, S_local = self.get_agent_data_to_array(T_local, S_local)

                    # Get used indices
                    ind = self.data_set.Domain[used].Index_saved
                    used_2d = np.tile(used_index[:,np.newaxis], (1, len(agent_index)))
                    agent_2d = np.tile(agent_index[np.newaxis,:], (len(used_index), 1))

                    # Fill in the data
                    T[used_2d, agent_2d] = T_local[ind]
                    S[used_2d, agent_2d] = S_local[ind]
    
            self.Type = T.astype(str)
            self.Size = S.astype(np.float32)
    

    def get_orig_data_index(self, Sample_ind, Agent_ind = None):
        # assert that sample_ind includes only integers
        assert isinstance(Sample_ind, np.ndarray), 'Sample index should be integers.'

        Out_shape = Sample_ind.shape
        if Agent_ind is None:
            assert len(Out_shape) == 1, 'If no agent index is given, the sample index should be 1D.'

            # Get unique sample indices
            Sample_ind_unique, Sample_ind_inverse = np.unique(Sample_ind, return_inverse = True)

            # Find original data in these indices
            samples_included = np.in1d(self.data_set.Used_samples, Sample_ind_unique)

            result = np.where(samples_included)[0]
            used_samples = self.data_set.Used_samples[result]
            used_agents  = self.data_set.Used_agents[result]

            # Get inverse of used samples
            index_inverse = np.zeros(Sample_ind_unique.max() + 1, int)
            index_inverse[Sample_ind_unique] = np.arange(len(Sample_ind_unique), dtype = int)

            mask = np.zeros((len(Sample_ind_unique), len(self.data_set.Agents)), bool)
            mask[index_inverse[used_samples], used_agents] = True

            # Get the original data index
            # Transform result to array
            result_array = np.full(mask.shape, -1, int)
            result_array[mask] = result

            result_array = result_array[Sample_ind_inverse]
            mask         = mask[Sample_ind_inverse]
            result = result_array[mask]

        else:
            assert Sample_ind.shape == Agent_ind.shape, 'Sample and Agent index should have the same shape.'

            # Flatten the data
            Sample_ind = Sample_ind.flatten()
            Agent_ind  = Agent_ind.flatten()
            
            # Use sparse matrix for lookups
            result = self.data_set.sparse_matrix_orig[Sample_ind, Agent_ind].A1

            # Get 1D mask
            mask = result != 0

            # Reset the original increase in index needed for mask identification
            result -= 1

            # Get only relevant data
            result = result[mask]

            # Transform mask to original input shape
            mask = mask.reshape(Out_shape)
        return result, mask




    def prepare_batch_generation_single(self, Pred_agents_pred):
        # Get required timesteps
        N_O_pred = self.data_set.N_O_pred_orig.copy()
        N_O_data = self.data_set.N_O_data_orig.copy() 
        
        # Reorder agents to save data
        if self.predict_single_agent:
            # set agent to be predicted into first location
            sample_id, pred_agent_id = np.where(Pred_agents_pred)
            
            # Get sample id
            num_agents = Pred_agents_pred.shape[1]

            Sample_id = np.tile(sample_id[:,np.newaxis], (1, num_agents))
            
            # Roll agents so that pred agent is first
            Agent_id = np.tile(np.arange(num_agents)[np.newaxis,:], (len(sample_id), 1))
            Agent_id = Agent_id + pred_agent_id[:,np.newaxis]
            Agent_id = np.mod(Agent_id, num_agents) 
            
            # Create Multiindex for the original data index
            Data_index, Data_index_mask = self.get_orig_data_index(Sample_id, Agent_id)
            assert Data_index_mask[:,0].all(), 'Pred agents should be available.'

            # Trasnform data index
            Data_index_array = np.full(Sample_id.shape, -1, int)
            Data_index_array[Data_index_mask] = Data_index
            Data_index_comp = np.repeat(Data_index_array[:,[0]], num_agents, 1)[Data_index_mask]

            # get differnce between agents
            Diff = self.data_set.X_orig[Data_index,..., :2] - self.data_set.X_orig[Data_index_comp,..., :2]

            # Get squared distance
            Diff = np.linalg.norm(Diff, axis = -1)

            # Get minimum distance over time (can include nan values)
            Diff = np.nanmin(Diff, -1)

            # Reshape back to original shape
            D = np.full(Sample_id.shape, np.nan, np.float32)
            D[Data_index_mask] = Diff

            # Find closest distance between agents during past observation (only positions)
            Agent_sorted_id = np.argsort(D, axis = 1)
            Agent_id = np.take_along_axis(Agent_id, Agent_sorted_id, axis = 1)
            
            # Sort agents according to distance from pred agent
            # Project out the sample ID

            _, Data_available = self.get_orig_data_index(Sample_id, Agent_id)
            useful_agents = np.where(Data_available.any(0))[0]

            # Set agents to nan that are to far away from the predicted agent
            num_agent = self.data_set.max_num_agents
            if num_agent is not None:
                useful_agents = useful_agents[:num_agent]
            
        else:
            # Sort agents to move the absent ones to the behind, and pred agents first
            Value = - Pred_agents_pred.astype(int)

            finite_timesteps = np.isfinite(self.data_set.X_orig).sum((1,2)) + np.isfinite(self.data_set.Y_orig)[:,:N_O_data.max()].sum((1,2))
            Value[self.data_set.Used_samples, self.data_set.Used_agents] -= finite_timesteps 
            
            # Sort by value
            Agent_id = np.argsort(Value, axis = 1)
            Sample_id = np.repeat(np.arange(Agent_id.shape[0])[:,np.newaxis], Agent_id.shape[1], 1)

            # Apply the sorting to the index
            # Agent_id_inverse = sp.stats.rankdata(Value, axis = 1, method='ordinal').astype(int) - 1
            # The upper method is not needed is Agent_id needs to be calculated anyway
            Agent_id_inverse = np.argsort(Agent_id, axis = 1)
            Used_agents_sort = Agent_id_inverse[self.data_set.Used_samples, self.data_set.Used_agents]
            
            # remove nan columns
            useful_agents = np.unique(Used_agents_sort)
        
        return N_O_data, N_O_pred, useful_agents, Sample_id, Agent_id

        

    #%%     
    def prepare_batch_generation(self):
        ## NOTE: Method has been adjusted for large datasets
        # Required attributes of the model
        # self.min_t_O_train: How many timesteps do we need for training
        # self.max_t_O_train: How many timesteps do we allow training for
        # self.predict_single_agent: Are joint predictions not possible
        # self.can_use_map: Can use map or not
        # self.can_use_graph: Can use scene graph or not 
        # If self.can_use_map, the following is also required
        # self.target_width:
        # self.target_height:
        # self.grayscale: Are image required in grayscale
        
        if self.extracted_data:
            return
        
        # Determine needed agents
        self.data_set._determine_pred_agents(pred_pov = self.get_output_type() == 'path_all_wi_pov')

        # Load data type
        self._extract_types()
            
        # Determine map use
        use_map = self.has_map and self.can_use_map

        # Determine graph use
        use_graph = self.has_graph and self.can_use_graph 
        
        if self.data_set.data_in_one_piece:
            # Extract old trajectories
            self.data_set._extract_original_trajectories()
            
            [N_O_data, N_O_pred, useful_agents, 
                Sample_id, Agent_id] = self.prepare_batch_generation_single(self.data_set.Pred_agents_pred)
        
        else:
            num_data_samples, max_num_agents = self.data_set.Pred_agents_pred.shape

            # Initialize the sample and agent id
            Sample_id = np.zeros((0,max_num_agents), int)
            Agent_id = np.zeros((0,max_num_agents), int)

            # Initialize the number of future timesteps
            N_O_data = np.zeros(num_data_samples, int)
            N_O_pred = np.zeros(num_data_samples, int)
            
            useful_agents = np.array([0], int)

            for file_index in range(len(self.data_set.Files)):
                used = self.data_set.Domain.file_index == file_index
                used_index = np.where(used)[0]

                Pred_agents_pred_local = self.data_set.Pred_agents_pred[used_index]
                
                # Get the original data
                self.data_set._extract_original_trajectories(file_index)
                
                # Extract the required information
                [N_O_data_local, N_O_pred_local, useful_agents_local, 
                    Sample_id_local, Agent_id_local] = self.prepare_batch_generation_single(Pred_agents_pred_local)

                # Adjust Sample_id_local to actual position in dataset
                Sample_id_local = used_index[Sample_id_local]

                # Fill in the global arrays
                N_O_data[used_index] = N_O_data_local 
                N_O_pred[used_index] = N_O_pred_local

                # Add to global arrays
                Sample_id = np.concatenate((Sample_id, Sample_id_local), 0)
                Agent_id  = np.concatenate((Agent_id, Agent_id_local), 0)

                # Track useful agents (this are index values)
                useful_agents = np.unique(np.concatenate((useful_agents, useful_agents_local), 0))
        
        # remove nan columns
        Sample_id = Sample_id[:,useful_agents]
        Agent_id  = Agent_id[:,useful_agents]
        
        # Save ID
        self.ID = np.stack((Sample_id, Agent_id), -1) 
        
        # Get pred agents
        self.Pred_agents = self.data_set.Pred_agents_pred[Sample_id, Agent_id] # num_samples, num_agents
        if self.predict_single_agent:
            # Only first agent gets predicted
            self.Pred_agents[:,1:] = False
            
        # Get agent types
        self.T = self.Type[Sample_id, Agent_id] # num_samples, num_agents

        # Get agent sizes
        self.S = self.Size[Sample_id, Agent_id] # num_samples, num_agents, 2
        
        # Get the number of future timesteps
        self.N_O_pred = N_O_pred[Sample_id[:,0]]
        self.N_O_data = N_O_data[Sample_id[:,0]]
        
        if self.data_set.data_in_one_piece:
            # Get trajectory data
            self.Data_index, self.Data_index_mask = self.get_orig_data_index(Sample_id, Agent_id) 
            # self.Data_index.shape = self.Data_index_mask.sum()
            # self.Data_index_mask.shape = num_samples, num_agents
            
            if use_map or use_graph:
                # Transform self.Data_index to an array
                Data_index_array = np.full(self.Data_index_mask.shape, -1, int)
                Data_index_array[self.Data_index_mask] = self.Data_index

            # Get images
            if use_map:
                # Get metadata
                domain_old = self.data_set.Domain
                if self.predict_single_agent:
                    Img_needed = np.zeros(self.Data_index_mask.shape[:2], bool)
                    Img_needed[:,0] = True
                else:
                    Img_needed = self.T != '0'

                domain_needed = domain_old.iloc[self.ID[Img_needed][:,0]]

                # Precalculate an X that is allready reduced to the needed agents
                # assert that agents with needed images are in Data_index_mask
                assert self.Data_index_mask[Img_needed].all(), 'All agents with images should be available.'

                # Get the needed agent data
                Data_index_needed = Data_index_array[Img_needed]
                Img_needed_inside = np.ones(len(Data_index_needed), bool)

                self.img, self.img_m_per_px, self.use_batch_extraction = self.extract_images(self.data_set.X_orig[Data_index_needed, ..., :2].astype(np.float32), 
                                                                                                Img_needed_inside, domain_needed)
                self.img_needed_sample, self.img_needed_agent = np.where(Img_needed)

            else:
                self.img = None
                self.img_m_per_px = None


            # Get graphs
            if use_graph:
                # Get metadata
                domain_old = self.data_set.Domain
                domain_needed = domain_old.iloc[self.ID[:,0,0]]

                # Get the mean positions of all pred agents at prediction time
                X_last = self.data_set.X_orig[..., -1, :2] # num_samples_agents, 2

                X_last_all = np.full((self.Pred_agents.shape[0], self.Pred_agents.shape[1], 2), np.nan, np.float32)

                # check if all agents are available
                assert self.Data_index_mask[self.Pred_agents].all(), 'All agents should be available.'

                # Get the needed agent data
                Data_index_needed = Data_index_array[self.Pred_agents]
                X_last_all[self.Pred_agents] = X_last[Data_index_needed] 

                if hasattr(self, 'sceneGraph_radius'):
                    radius = self.sceneGraph_radius
                else:
                    radius = 100
                
                if hasattr(self, 'sceneGraph_wave_length'):
                    wave_length = self.sceneGraph_wave_length
                else:
                    wave_length = 1.0
                    
                self.graph, self.use_graph_batch_extraction = self.extract_sceneGraphs(domain_needed, X_last_all, radius, wave_length)
                self.graph_needed_sample = np.arange(len(self.ID))
            else:
                self.graph = None

        else:
            self.use_batch_extraction = True
            self.use_graph_batch_extraction = True
            self.img = None
            self.img_m_per_px = None
            self.graph = None
        
        self.extracted_data = True
    
    
    def provide_all_included_agent_types(self):
        ## NOTE: Method has been adjusted for large datasets
        '''
        This function allows a quick generation of all the available agent types. Right now, the following are implemented:
        - 'P':    Pedestrian
        - 'B':    Bicycle
        - 'M':    Motorcycle
        - 'V':    All other vehicles (cars, trucks, etc.)     
    
        Returns
        -------
        T_all : np.ndarray
          This is a one-dimensional numpy array that includes all agent types that can be found in the given dataset.
    
        '''
        # get all agent types
        self._extract_types()
        T_all = np.unique(self.Type)
        T_all = T_all[T_all != '0']
        return T_all
    
    def _extract_useful_training_samples(self):
        ## NOTE: Method has been adjusted for large datasets
        # Get training samples
        I_train = np.where(np.in1d(self.ID[:,0,0], self.Index_train))[0]
        
        # Get samples with enough timesteps
        remain_samples = self.N_O_data[I_train] >= self.min_t_O_train

        # Only use samples with enough timesteps for training
        I_train = I_train[remain_samples]
        
        return I_train
    
    def provide_all_training_trajectories(self, return_categories = False):
        ## NOTE: Method has been adjusted for large datasets
        r'''
        This function provides trajectroy data an associated metadata for the training of model
        during prediction and training. It returns the whole training set (including validation set)
        in one go

        Parameters
        ----------
        return_categories : bool, optional
            This indicates if the categories (**C**, see below) of the samples should be returned. 
            The default is *False*.

        Returns
        -------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
            values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
            *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
            the position data (*self.input_data_type = ['x', 'y', ...]*). 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values.
        Y : np.ndarray, optional
            This is the future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times N_{data}\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        S_train : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
            where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
            (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
            instead be np.nan.
        C : np.ndarray
            Optional return provided when return_categories = True. 
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
            category of agent observed, where the categories are dataset specific.
        img : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
            It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
            of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
            at its current position, and are rotated so that the agent is right now driving to the right. 
            If an agent is not observed at prediction time, 0 values are returned.
        img_m_per_px : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
            the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
        graph : np.ndarray
            This is a numpy array with length :math:`N_{samples}`, where the entries are pandas.Series with the following entries:
            
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

        Pred_agents : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
            if it expected by the framework that a prediction will be made for the specific agent.
            
            If only one agent has to be predicted per sample, for **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
            be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.
        Sample_id : np.ndarray, optional
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted. This value is only returned for **mode** = *'pred'*.
        Agent_id : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted.. This value is only returned for **mode** = *'pred'*.

        '''
        assert self.data_set.data_in_one_piece, 'This method is only useful for datasets that are in one piece.'
        
        assert self.get_output_type()[:4] == 'path'
        
        self.prepare_batch_generation()
        
        I_train = self._extract_useful_training_samples()
        
        # Get the data index information for train set
        Data_index_array = np.full(self.Data_index_mask.shape, -1, int)
        Data_index_array[self.Data_index_mask] = self.Data_index

        data_index_array = Data_index_array[I_train]
        data_index_mask  = self.Data_index_mask[I_train]

        data_index = data_index_array[data_index_mask]

        # Preset the output arrays
        X_train = np.full((*data_index_mask.shape, *self.data_set.X_orig.shape[-2:]), np.nan, np.float32)
        Y_train = np.full((*data_index_mask.shape, *self.data_set.Y_orig.shape[-2:]), np.nan, np.float32)

        X_train[data_index_mask] = self.data_set.X_orig[data_index]
        Y_train[data_index_mask] = self.data_set.Y_orig[data_index]

        # Get globally saved data
        T_train = self.T[I_train]
        S_train = self.S[I_train]
        Pred_agents_train = self.Pred_agents[I_train]
        
        if self.img is not None:
            if self.predict_single_agent:
                Img_needed = np.zeros(X_train.shape[:2], bool)
                Img_needed[:,0] = True
            else:
                Img_needed = T_train != '0'
                
            sample_needed, agent_needed = np.where(Img_needed)
            sample_needed = I_train[sample_needed]

            ind_needed = sample_needed + 1j * agent_needed
            ind_given = self.img_needed_sample + 1j * self.img_needed_agent

            # Find at which places in img_needed_sample one can find ind_advance
            image_ind = self.data_set.get_indices_1D(ind_needed, ind_given)

            img_needed          = self.img[image_ind]
            img_m_per_px_needed = self.img_m_per_px[image_ind]

            img_train          = np.zeros((*Img_needed.shape, self.target_height, self.target_width, img_needed.shape[-1]), np.uint8)
            img_m_per_px_train = np.ones(Img_needed.shape, np.float32) * np.nan

            img_train[Img_needed]          = img_needed
            img_m_per_px_train[Img_needed] = img_m_per_px_needed
        else:
            if hasattr(self, 'use_batch_extraction'):
                assert self.use_batch_extraction, 'Image data could still be extracted'
                print('Image data could not be extracted due to memory issues for the whole dataset.')
            img_train = None
            img_m_per_px_train = None

        
        if self.graph is not None:
            graph_ind = self.data_set.get_indices_1D(I_train, self.graph_needed_sample)
            graph_needed = self.graph[graph_ind]
            if self.predict_single_agent:
                Graph_needed = np.zeros(X_train.shape[:2], bool)
                Graph_needed[:,0] = True
            else:
                Graph_needed = T_train != '0'
            
            graph_train = np.zeros((*Graph_needed.shape, graph_needed.shape[-1]), np.float32)
            graph_train[Graph_needed] = graph_needed
        
        Sample_id_train = self.ID[I_train,0,0]
        Agents = np.array(self.data_set.Agents)
        Agent_id_train = Agents[self.ID[I_train,:,1]]
        
        # get input data type
        self.input_data_type = self.data_set.Input_data_type[0]
        
        if return_categories:

            # Get categories
            Sample_id = self.ID[I_train,0,0]

            # Get agent predictions
            if 'category' in self.data_set.Domain.columns:
                # Sample_id = self.ID[ind_advance,0,0]
                C = self.data_set.Domain.category.iloc[Sample_id]
                C = pd.DataFrame(C.to_list())

                # Add columns missing from self.data_set.Agents
                C = C.reindex(columns = self.data_set.Agents, fill_value = np.nan)

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C_train = C.to_numpy().astype(int)
                C_train = np.take_along_axis(C_train, self.ID[I_train,:,1], 1)
            else:
                C_train = None

            return [X_train, Y_train, T_train, S_train, C_train, img_train, img_m_per_px_train, graph_train,
                    Pred_agents_train, Sample_id_train, Agent_id_train]
        else:
            return [X_train, Y_train, T_train, S_train, img_train, img_m_per_px_train, graph_train,
                    Pred_agents_train, Sample_id_train, Agent_id_train]
        
        
    def _update_available_samples(self, Ind_advance, ind_advance):
        # Get the indices that will remain
        ind_remain = np.setdiff1d(Ind_advance[0], ind_advance)

        # Check if epoch is completed
        if len(ind_remain) == 0:
            epoch_done = True
            ind_all = np.concatenate((Ind_advance[1], ind_advance))
            np.random.shuffle(ind_all)
            
            Ind_advance[1] = np.array([], int)
            Ind_advance[0] = ind_all
            
        else:
            epoch_done = False
            Ind_advance[1] = np.concatenate((Ind_advance[1], ind_advance))
            Ind_advance[0] = ind_remain  
        
        return epoch_done, Ind_advance
    

    def get_batch_number(self, mode, batch_size, val_split_size = 0.0, ignore_map = False):

        r'''
        This function provides trajectroy data an associated metadata for the training of model
        during prediction and training.

        Parameters
        ----------
        mode : str
            This discribes the type of data needed. *'pred'* will indicate that this is for predictions,
            while during training, *'train'* and *'val'* indicate training and validation set respectively.
        batch_size : int
            The number of samples to be selected.
        val_split_size : float, optional
            The part of the overall training set that is set aside for model validation during the
            training process. The default is *0.0*.
        ignore_map : bool, optional
            This indicates if image data is not needed, even if available in the dataset 
            and processable by the model. The default is *False*.


        Returns
        -------
        num_batches : int
            The number of batches that would be generated under the given settings
        '''

        # Prepare data
        self.prepare_batch_generation()
        
        # Get specific dataset
        if mode == 'pred':
            I_pred = np.where(np.in1d(self.ID[:,0,0], self.Index_test))[0]
            Ind_advance = I_pred
            N_O = self.N_O_pred
            
        elif mode == 'val':
            I_train = self._extract_useful_training_samples()
            num_train = int(len(I_train) * (1 - val_split_size))
            Ind_advance = I_train[num_train:]
            N_O = np.minimum(self.N_O_data, self.max_t_O_train)
            
        elif mode == 'train':
            I_train = self._extract_useful_training_samples()
            num_train = int(len(I_train) * (1 - val_split_size))
            Ind_advance = I_train[:num_train]
            N_O = np.minimum(self.N_O_data, self.max_t_O_train)
        
        else:
            raise TypeError("Unknown mode.")
        
        # Get domain data
        Sample_id_advance = self.ID[Ind_advance,0,0]
        Domain_advance = self.data_set.Domain.iloc[Sample_id_advance]
        N_O_advance = N_O[Ind_advance]   # Number of timesteps
        
        # Combine batch deciding data
        Data = []
        
        if not ignore_map and self.has_map:
            Data.append(Domain_advance.image_id.to_numpy().astype(int))
        
        if not self.data_set.data_in_one_piece:
            Data.append(Domain_advance.file_index.to_numpy().astype(int))
        else:
            Scenarios = Domain_advance.Scenario.to_numpy()
            Data.append(np.unique(Scenarios, return_inverse = True)[1])
        
        if self.predict_single_agent:
            T_advance = self.T[Ind_advance,0]
            Data.append(np.unique(T_advance, return_inverse = True)[1])
        
        Data = np.stack(Data, 1) # num_samples, num_data
        Data_type_ind = np.unique(Data, return_inverse = True)[1]     

        # Geerate batches
        num_batches = 0
        for i in range(Data_type_ind.max() + 1):
            Ind_advance_i = Ind_advance[Data_type_ind == i]
            N_O_advance_i = N_O[Ind_advance_i]

            # Check for number of timesteps
            if mode == 'train':
                N_O_advance_i[:] = N_O_advance_i.min()
            
            num_unique_counts = np.unique(N_O_advance_i, return_counts = True)[1]
            
            num_batches_unique = np.ceil(num_unique_counts / batch_size)
            
            num_batches += num_batches_unique.sum()

        return num_batches




    
    def provide_batch_data(self, mode, batch_size, val_split_size = 0.0, ignore_map = False, ignore_graph = False, 
                           return_categories = False, return_classifications = False):
        ## NOTE: Method has been adjusted for large datasets
        r'''
        This function provides trajectroy data an associated metadata for the training of model
        during prediction and training.

        Parameters
        ----------
        mode : str
            This discribes the type of data needed. *'pred'* will indicate that this is for predictions,
            while during training, *'train'* and *'val'* indicate training and validation set respectively.
        batch_size : int
            The number of samples to be selected.
        val_split_size : float, optional
            The part of the overall training set that is set aside for model validation during the
            training process. The default is *0.0*.
        ignore_map : bool, optional
            This indicates if image data is not needed, even if available in the dataset 
            and processable by the model. The default is *False*.
        ignore_graph : bool, optional
            This indicates if scene graph data is not needed, even if available in the dataset
            and processable by the model. The default is *False*.
        return_categories : bool, optional
            This indicates if the categories (**C**, see below) of the samples should be returned. 
            The default is *False*.
        return_classifications : bool, optional
            This indicates if the behavior probabilities (**P**, see below) of the samples should be returned.
            If the underlying datasets do not include behavior classifications, None is returned instead. 
            Given that this encodes future behavior, if **mode** = *'pred'*, the framework will ignore this value.
            The default is *False*. 


        Returns
        -------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
            values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
            *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
            the position data (*self.input_data_type = ['x', 'y', ...]*). It must be noted that *self.input_data_type*
            will always correspond to the output of the *path_data_info()* of the data_set from which this batch data
            was loaded. If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
        Y : np.ndarray, optional
            This is the future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times N_{data}\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        S : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
            where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
            (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
            instead be np.nan.
        C : np.ndarray, optional
            Optional return provided when return_categories = True. 
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
            category of agent observed, where the categories are dataset specific.
        P : np.ndarray, optional
            Optional return provided when return_classifications = True.
            This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array. It includes float values that indicate
            the probability of the agent to belong to a specific class. The classes are dataset specific. Given that this is the 
            ground truth, each row should be a one-hot encoded vector. If the dataset does not include behavior classifications,
            None is returned instead.
        class_names : list, optional
            Optional return provided when return_classifications = True.
            This is a list of length :math:`N_{classes}` of strings that indicate the names of the classes. If the dataset does 
            not include behavior classifications, None is returned instead.
        img : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
            It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
            of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
            at its current position, and are rotated so that the agent is right now driving to the right. 
            If an agent is not observed at prediction time, 0 values are returned.
        img_m_per_px : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
            the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
        graph : np.ndarray
            This is a numpy array with length :math:`N_{samples}`, where the entries are pandas.Series with the following entries:
            
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

        Pred_agents : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
            if it expected by the framework that a prediction will be made for the specific agent.
            
            If only one agent has to be predicted per sample, for **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
            be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.
        num_steps : int
            This is the number of future timesteps provided in the case of traning in expected in the case of prediction. In the 
            former case, it has the value :math:`N_{O}`.
        Sample_id : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted.
        Agent_id : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted (for corresponding string names see self.data_set.Agents).
        epoch_done : bool
            This indicates wether one has just sampled all batches from an epoch and has to go to the next one.

        '''
        reset_train_indices = False
        if hasattr(self, 'val_split_size') and mode == 'train':
            if hasattr(self, 'Ind_train') and val_split_size != self.val_split_size:
                # Check if a new epoch is starting
                if len(self.Ind_train[1]) == 0:
                    reset_train_indices = True

        self.val_split_size = val_split_size
        
        self.prepare_batch_generation()

        if mode == 'pred':
            return_classifications = False
        
        if mode == 'pred':
            assert self.model_mode == 'pred', 'During prediction, testing set should be called.'
            if not hasattr(self, 'Ind_pred'):
                I_pred = np.where(np.in1d(self.ID[:,0,0], self.Index_test))[0]
                self.Ind_pred = [I_pred, np.array([], int)]
                
            N_O = self.N_O_pred
            Ind_advance = self.Ind_pred
            
        elif mode == 'val':
            assert self.model_mode == 'train', 'During validation, the validation part of the training set should be called.'
            if not hasattr(self, 'Ind_val'):
                I_train = self._extract_useful_training_samples()
                num_train = int(len(I_train) * (1 - val_split_size))

                # Set train/val indices
                self.Ind_train = [I_train[:num_train], np.array([], int)]
                self.Ind_val   = [I_train[num_train:], np.array([], int)]
                    
                
            N_O = np.minimum(self.N_O_data, self.max_t_O_train)
            Ind_advance = self.Ind_val
            
        elif mode == 'train':
            assert self.model_mode == 'train', 'During training, the non-validation part of the training set should be called.'
            if not hasattr(self, 'Ind_train') or reset_train_indices:
                I_train = self._extract_useful_training_samples()
                num_train = int(len(I_train) * (1 - val_split_size))

                # Set train/val indices
                self.Ind_train = [I_train[:num_train], np.array([], int)]
                self.Ind_val   = [I_train[num_train:], np.array([], int)]

                if reset_train_indices:
                    np.random.shuffle(self.Ind_train[0])
                    np.random.shuffle(self.Ind_val[0])
            
            N_O = np.minimum(self.N_O_data, self.max_t_O_train)
            Ind_advance = self.Ind_train
        
        else:
            raise TypeError("Unknown mode.")
        
        # Get data needed for selecting batch from available
        Sample_id_advance = self.ID[Ind_advance[0],0,0]
        Domain_advance = self.data_set.Domain.iloc[Sample_id_advance]
        N_O_advance = N_O[Ind_advance[0]]   # Number of timesteps


        Use_candidate = np.ones(len(Ind_advance[0]), bool)
        if not ignore_map and self.has_map:
            # Check for file index and image id
            Image_id_advance = Domain_advance.image_id.to_numpy() # Image id
            Use_candidate &= Image_id_advance == Image_id_advance[0]

        # For large dataset, check for file index as well
        if not self.data_set.data_in_one_piece:
            File_index_advance = Domain_advance.file_index.to_numpy() # File index
            Use_candidate &= File_index_advance == File_index_advance[0]
        else:
            # sort by dataset
            Scenario_advance = Domain_advance.Scenario.to_numpy() # Scenario
            Use_candidate &= Scenario_advance == Scenario_advance[0]

        
        # Check for predicted agent type
        if self.predict_single_agent:
            T_advance = self.T[Ind_advance[0],0]
            Use_candidate &= (T_advance == T_advance[0])

        # Check for number of timesteps
        if mode == 'train':
            Use_candidate_N_O = (N_O_advance == N_O_advance[0]) & Use_candidate
            if Use_candidate_N_O.sum() < batch_size:
                N_O_advance_possible = N_O_advance[Use_candidate & (N_O_advance >= N_O_advance[0])]
                N_O_possible, N_O_counts = np.unique(N_O_advance_possible, return_counts = True)
                N_O_cum = np.cumsum(N_O_counts)
                
                if N_O_cum[-1] > batch_size:
                    needed_length = np.where(N_O_cum > batch_size)[0][0]
                    Use_candidate_N_O = (N_O_advance >= N_O_advance[0]) & (N_O_advance <= N_O_possible[needed_length])
                else:
                    Use_candidate_N_O = (N_O_advance >= N_O_advance[0])
        else:
            Use_candidate_N_O = N_O_advance == N_O_advance[0]

        Use_candidate &= Use_candidate_N_O
            
             
        # Find in remaining samples those whose type corresponds to that of the first
        Ind_candidates = np.where(Use_candidate)[0]
        
        # Get the final indices to be returned
        ind_advance = Ind_advance[0][Ind_candidates[:batch_size]]

        # Sort ind_advance
        ind_advance = np.sort(ind_advance)
        
        # check if epoch is completed, if so, shuffle and reset index
        epoch_done, Ind_advance = self._update_available_samples(Ind_advance, ind_advance) 

        ## Prepare the data to be returned
        # get num_steps
        num_steps = N_O[ind_advance].min()
        if mode != 'pred':
            assert num_steps <= self.max_t_O_train, 'Number of timesteps is too large for training, got {}, while max is {}'.format(num_steps, self.max_t_O_train)
        
        # Get data as available for whole dataset
        T = self.T[ind_advance]
        S = self.S[ind_advance]
        Pred_agents = self.Pred_agents[ind_advance]

        # Get the corresponding sample_ids 
        Sample_id = self.ID[ind_advance,:,0]
        Agent_id  = self.ID[ind_advance,:,1]

        # Get the corresponding input_data_type
        input_data_type_indices = self.data_set.Domain.iloc[Sample_id[:,0]].data_type_index
        assert len(np.unique(input_data_type_indices)) == 1, 'Only one data type should be used in each batch'
        self.input_data_type = self.data_set.Input_data_type[input_data_type_indices.iloc[0]]

        num_dim = len(self.input_data_type)

        # Prepare the output arrays
        X = np.full((len(ind_advance), self.ID.shape[1], self.num_timesteps_in, num_dim), np.nan, np.float32)
        Y = np.full((len(ind_advance), self.ID.shape[1], num_steps, num_dim), np.nan, np.float32)

        # Get data that is potentially not available yet
        if self.data_set.data_in_one_piece:
            Data_index_array = np.full(self.Data_index_mask.shape, -1, int)
            Data_index_array[self.Data_index_mask] = self.Data_index

            data_index_array = Data_index_array[ind_advance]
            data_index_mask  = self.Data_index_mask[ind_advance]

            data_index = data_index_array[data_index_mask]

            X[data_index_mask] = self.data_set.X_orig[data_index]
            Y[data_index_mask] = self.data_set.Y_orig[data_index, :num_steps]

            if return_classifications:
                Output_A = self.data_set.Output_A.iloc[Sample_id[:,0]] 
                P = Output_A.to_numpy().astype(np.float32)
                class_names = self.data_set.Behaviors

        else:

            # Get the corresponding domain_files
            Domain_files = self.data_set.Domain.iloc[Sample_id[:,0]].file_index

            # Assert that only one file is used
            assert len(np.unique(Domain_files)) == 1
            domain_file = Domain_files.iloc[0]

            # Load corresponding data
            self.data_set._extract_original_trajectories(file_index = int(domain_file))

            # Transform sample_id to the id in the extracted data
            used_index = np.where(self.data_set.Domain.file_index == domain_file)[0]

            # Det the position of Sample_id in used_index
            Sample_id_used = self.data_set.get_indices_1D(Sample_id[:,0], used_index)
            Sample_id_used = np.tile(Sample_id_used[:,np.newaxis], (1, Sample_id.shape[1]))

            # Get original indices
            data_index, data_index_mask = self.get_orig_data_index(Sample_id_used, Agent_id)

            X[data_index_mask] = self.data_set.X_orig[data_index].astype(np.float32)
            Y[data_index_mask] = self.data_set.Y_orig[data_index, :num_steps].astype(np.float32)

            if return_classifications:
                Output_A = self.data_set.Output_A_file.iloc[Sample_id_used[:,0]]
                P = Output_A.to_numpy().astype(np.float32)
                class_names = self.data_set.Behaviors

        # Check if images need to be extracted
        if hasattr(self, 'use_batch_extraction') and (not ignore_map) and self.has_map and self.can_use_map:
            if self.predict_single_agent:
                Img_needed = np.zeros(X.shape[:2], bool)
                Img_needed[:,0] = True
            else:
                Img_needed = T != '0'

            if self.use_batch_extraction:
                domain_needed = self.data_set.Domain.iloc[Sample_id[Img_needed]]
                img_needed, img_m_per_px_needed, unsuccesful = self.extract_images(X, Img_needed, domain_needed)
                if unsuccesful:
                    raise MemoryError('Not enough memory to extract images even with batches. Consider using grey scale images or a smaller resolution.' +
                                      '\nNote, however, that other errors might have caused this as well.')
            else:
                sample_needed, agent_needed = np.where(Img_needed)
                sample_needed = ind_advance[sample_needed]

                ind_needed = sample_needed + 1j * agent_needed
                ind_given = self.img_needed_sample + 1j * self.img_needed_agent

                # Find at which places in img_needed_sample one can find ind_advance
                image_ind = self.data_set.get_indices_1D(ind_needed, ind_given)

                img_needed          = self.img[image_ind]
                img_m_per_px_needed = self.img_m_per_px[image_ind]

            if self.predict_single_agent:
                Img_needed = Img_needed[:,:1]
            
            # Transfrom img needed back according to Img_needed into required format
            img          = np.zeros((*Img_needed.shape, self.target_height, self.target_width, img_needed.shape[-1]), np.uint8)
            img_m_per_px = np.ones(Img_needed.shape, np.float32) * np.nan

            img[Img_needed]          = img_needed
            img_m_per_px[Img_needed] = img_m_per_px_needed

        else:
            img          = None
            img_m_per_px = None

        # Ignore identical columns
        Sample_id = Sample_id[:,0]

        # Check if graphs need to be extracted
        if hasattr(self, 'use_graph_batch_extraction') and (not ignore_graph) and self.has_graph and self.can_use_graph:

            if self.use_graph_batch_extraction:
                domain = self.data_set.Domain.iloc[Sample_id]
                X_last_all = X[...,-1,:2].copy() # num_samples x num_agents x 2
                X_last_all[~Pred_agents] = np.nan # Only consider pred agents
                if hasattr(self, 'sceneGraph_radius'):
                    radius = self.sceneGraph_radius
                else:
                    radius = 100
                
                if hasattr(self, 'sceneGraph_wave_length'):
                    wave_length = self.sceneGraph_wave_length
                else:
                    wave_length = 1.0
                
                graph, unsuccesful = self.extract_sceneGraphs(domain, X_last_all, radius, wave_length)

                if unsuccesful:
                    MemoryError('Not enough memory to extract graphs even with batches.' )
            else:
                # Find at which places in self.graph_needed_sample one can find ind_advance
                graph_ind = self.data_set.get_indices_1D(ind_advance, self.graph_needed_sample)

                graph = self.graph[graph_ind]
        else:
            graph = None

        if return_categories:
            if 'category' in self.data_set.Domain.columns:
                C = self.data_set.Domain.category.iloc[Sample_id]
                C = pd.DataFrame(C.to_list())

                # Adjust columns to match self.data_set.Agents
                C = C.reindex(columns = self.data_set.Agents, fill_value = np.nan)

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C = C.to_numpy().astype(int)
                C = np.take_along_axis(C, Agent_id, 1)

                if (C == 4).all():
                    C = None

            else:
                C = None
            if mode == 'pred':
                self.batch_data = [X, Y, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id]
                return     X,    T, S, C,                 img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
            else:
                if return_classifications:
                    return X, Y, T, S, C, P, class_names, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
                else:
                    return X, Y, T, S, C,                 img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
        else:
            if mode == 'pred':
                self.batch_data = [X, Y, T, S, None, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id]
                return     X,    T, S,                    img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
            else:
                if return_classifications:
                    return X, Y, T, S,    P, class_names, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
                else:
                    return X, Y, T, S,                    img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
    
    def classify_data(self, Pred, Sample_id, Agent_id):
        r'''
        This function classifies the predicted data into the categories of the dataset. It is only useful if the dataset
        includes categories. The function will return the categories of the predicted data.

        Parameters
        ----------
        Pred : np.ndarray
            This is the predicted future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or on some timesteps partially not observed, then this can include np.nan values. 
            The required value of :math:`N_{preds}` is given in **self.num_samples_path_pred**.
        Sample_id : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted.
        Agent_id : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted.

        Returns
        -------
        P_hot : np.ndarray
            This is a :math:`\{N_{samples} \times N_{preds} \times N_{classes}\}` dimensional numpy array, which for each 
            class contains the probability that it was observed in the sample. As this are observed values, 
            per row, the sum of the values will be 1 (i. e., one-hot encoded). 

            If classification is not possible (not all required agents available, or no categories in the dataset),
            then all values in the row will be np.nan.
        
        class_names : list
            This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
            class.
        '''

        # Get domain
        Domain = self.data_set.Domain.iloc[Sample_id] # (N_samples)

        # Transform Preds to pandas data Frame
        # Set up the pandas data frame
        Pred_df = pd.DataFrame(np.empty((len(Pred), len(self.data_set.Agents)), object), 
                               columns = self.data_set.Agents)
        Domain.index = Pred_df.index

        # Fill the data frame
        useful = np.isfinite(Pred).all(-1).any(-1).any(-1) # (N_samples x N_agents)
        sample_id = np.arange(Pred.shape[0])[:,np.newaxis].repeat(Pred.shape[1], axis = 1)
        sample_id = sample_id[useful]
        agents_id = Agent_id[useful]

        # Fill in the data
        pred_list = list(Pred[useful])
        Pred_df.values[sample_id, agents_id] = pred_list

        P_hot, class_names = self.data_set.classify_data(Pred_df, Domain, Pred.shape[-3], Pred.shape[-2]) # 

        P_hot = P_hot.reshape(len(Sample_id), -1, P_hot.shape[-1])

        return P_hot, class_names

    
    def save_predicted_batch_data(self, Pred, Sample_id, Agent_id, Pred_agents = None, Log_probs = None):
        r'''

        Parameters
        ----------
        Pred : np.ndarray
            This is the predicted future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or on some timesteps partially not observed, then this can include np.nan values.
            The required value of :math:`N_{preds}` is given in **self.num_samples_path_pred**.
        Sample_id : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted.
        Agent_id : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted.
        Pred_agents : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
            if it expected by the framework that a prediction will be made for the specific agent.
            
            This input does not have to be provided if the model can only predict one single agent at the same time and is therefore
            incapable of joint predictions, i. e., it is only expected if **self.predict_single_agent** =  *True*.
        Log_probs : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents} \times N_{preds}\}` dimensional numpy array. it includes float values, with 
            the model assigned log likelihoods for each corresponding prediction. For agents that doe not need to be predicted 
            (i.e. ~Pred_agents), this can be np.nan values.
            
            This input does not have to be provided if the model does not predict likelihoods.

        Returns
        -------
        None.

        '''
        assert self.get_output_type()[:4] == 'path'
        if self.predict_single_agent:
            if len(Pred.shape) == 4:
                Pred = Pred[:,np.newaxis]

            if Pred.shape[1] == 1:
                Pred = np.concatenate([Pred, np.full((Pred.shape[0], Agent_id.shape[1] - 1, *Pred.shape[2:]), np.nan)], axis = 1)

            if Pred_agents is None:
                Pred_agents = np.zeros(Agent_id.shape, bool)
                Pred_agents[:,0] = True
        else:
            assert Pred_agents is not None

        assert Sample_id.shape == Agent_id.shape[:1]
        assert Pred.shape[2] == self.num_samples_path_pred
        assert Pred.shape[4] >= 2
        assert Pred.shape[:2] == Agent_id.shape
        assert Pred_agents.shape == Agent_id.shape

        assert hasattr(self, 'batch_data'), 'This should not have happened here.'
        num_steps_required = self.batch_data[-3]
        assert Pred.shape[3] >= num_steps_required, 'Number of timesteps in prediction {} is not equal to the number of timesteps required {}.'.format(Pred.shape[-2], num_steps_required)
        Pred = Pred[..., :num_steps_required, :] # Only predict required length

        # Get ready to adjust for potential changes of Agent id in the model pertrub method function
        Sample_id_given = self.batch_data[-2].copy()
        assert np.array_equal(Sample_id_given, Sample_id), "Batch samples are missing."
        Agent_id_given = self.batch_data[-1].copy()
        max_agent_id = max(Agent_id_given.max(), Agent_id.max())
        Pred_agents_given = self.batch_data[-4].copy()

        # Prepare corresponding sample id:
        sample_id = np.arange(Agent_id.shape[0])[:,np.newaxis].repeat(Agent_id.shape[1], axis = 1)
        sample_id_given = np.arange(Agent_id_given.shape[0])[:,np.newaxis].repeat(Agent_id_given.shape[1], axis = 1)

        # Check if all agents are provided
        pred_samples, pred_agents = np.where(Pred_agents)
        pred_samples_given, pred_agents_given = np.where(Pred_agents_given)

        Pred_agents_id = np.stack((Sample_id[pred_samples], Agent_id[pred_samples, pred_agents]), axis = -1) # Num_pred_agents x 2
        Pred_agents_id_given = np.stack((Sample_id_given[pred_samples_given], Agent_id_given[pred_samples_given, pred_agents_given]), axis = -1) # Num_pred_agents x 2

        # Check if all agents are provided
        same_agent = (Pred_agents_id_given[:,np.newaxis] == Pred_agents_id[np.newaxis,:]).all(-1) # Num_pred_agents x Num_pred_agents
        assert same_agent.any(-1).all(), 'Predicted agents are not the same as the given agents.'

        if self.predict_path_probs:
            if Log_probs is not None:
                assert Pred.shape[[0,2]] == Log_probs.shape[[0,2]]
                assert Log_probs.shape[1] == Pred.shape[1]
                if Log_probs.shape[1] == 1:
                    Log_probs = np.concatenate([Log_probs, np.full((Log_probs.shape[0], Agent_id.shape[1] - 1, Log_probs.shape[-1]), np.nan)], axis = 1)
            else: 
                # Use the calculate_log_likelihoods function to calculate the log likelihoods
                Y_gt = self.batch_data[1].copy()


                # Pred corresponds to Agent_id, while we need something that corresponds to Agent_id_given
                # Map Y_pred_k onto Y_pred_k_used accordingly
                Pred_actual_help = np.full((Pred.shape[0], max_agent_id, *Pred.shape[2:]), np.nan)
                Pred_actual_help[sample_id, Agent_id] = Pred
                Pred_actual = Pred_actual_help[sample_id_given, Agent_id_given]

                # Go through predictions
                # Get actual Log_probs
                Log_probs_actual = np.full(Pred_actual.shape[:3], np.nan, np.float32)
                for k in Log_probs.shape[-1]:
                    Y_pred_k = Pred_actual[:,:,k] # num_samples x num_agents x N_O x 2
                    self.batch_data[1] = Y_pred_k
                    log_probs_k = self.calculate_log_likelihoods(*self.batch_data) # num_samples x M_agents
                    assert len(log_probs_k.shape) == 2
                    assert Pred.shape[:2] == log_probs_k.shape

                    Log_probs_actual[:,:,k] = log_probs_k
                
                # Transform
                Log_probs_help = np.full((Pred.shape[0], max_agent_id, *Log_probs.shape[2:]), np.nan, np.float32)
                Log_probs_help[sample_id_given, Agent_id_given] = Log_probs_actual
                Log_probs = Log_probs_help[sample_id, Agent_id]
                    
                # Set ground truth back to batch data
                self.batch_data[1] = Y_gt

            ##Calculate the corresponding ground truth likelihoods
            # Check for batch data
            gt_probs = self.calculate_log_likelihoods(*self.batch_data) # Num_samples x M_agents
            
            assert len(gt_probs.shape) == 2
            assert self.batch_data[1].shape[:2] == gt_probs.shape

            # Map onto actual Agent_id
            gt_probs_help = np.full((Pred.shape[0], max_agent_id), np.nan, np.float32)
            gt_probs_help[sample_id_given, Agent_id_given] = gt_probs
            gt_probs = gt_probs_help[sample_id, Agent_id]

            # Append log likelihoods with ground truth
            Log_probs = np.concatenate([Log_probs, gt_probs[...,np.newaxis]], axis = -1)
        
        
        # Get agent string names
        Agents = np.array(self.data_set.Agents)

        for i, i_sample in enumerate(Sample_id):
            for j, agent_id in enumerate(Agent_id[i]):
                if not Pred_agents[i,j]:
                    continue
                agent = Agents[agent_id]
                pred_traj = Pred[i, j][...,:2].astype('float32') # Only keep x and y coordinates
                assert np.isfinite(pred_traj).all(), 'Predicted trajectory contains non-finite values.'
                self.Output_path_pred.loc[i_sample, agent] = None
                self.Output_path_pred.loc[i_sample, agent] = pred_traj 
                
                if self.predict_path_probs:
                    pred_probs = Log_probs[i,j].astype('float32')
                    assert np.isfinite(pred_probs).all(), 'Predicted log likelihoods contains non-finite values.'
                    self.Output_path_pred_probs.loc[i_sample, agent] = None
                    self.Output_path_pred_probs.loc[i_sample, agent] = pred_probs   
    
    
    def get_classification_distibution(self, train = True):
        r'''
        This function returns the distribution of the classes in the dataset. It can be used to determine the class weights for
        classification models.

        Parameters
        ----------
        train : bool, optional
            This discribes whether one wants to generate training or testing data. The default is True.
        
        Returns
        -------
        class_names : list
            This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
            class.
        class_counts : np.ndarray
            This is a :math:`N_{classes}` dimensional numpy array with int values. 
            It contains the number of class instances in the dataset.
        '''
        
        # Select current samples
        if train:
            assert self.model_mode == 'train', 'During training, training set should be called.'
            Index = self.Index_train
        else:
            assert self.model_mode == 'pred', 'During training, training set should be called.'
            Index = self.Index_test

        # Assemble output_A
        class_names = self.data_set.Behaviors
        if self.data_set.data_in_one_piece:
            Output_A = self.data_set.Output_A.iloc[Index]
            Output_A = Output_A[class_names]
        else:
            Output_A = pd.DataFrame(np.zeros((len(Index), len(class_names)), bool), columns = class_names)

            Domain = self.data_set.Domain.iloc[Index]
            file_indices = Domain.file_index.to_numpy()
            for file_index in np.unique(file_indices):
                use = file_indices == file_index
                
                # Get the specific index in the extracted data
                ind_data = Domain[use].Index_saved

                # Load the data
                data_file = self.data_set.Files[file_index] + '_data.npy'

                # Load the data and extract the data
                [_, _, _, _, _, _, Output_A_file, _, _] = np.load(data_file, allow_pickle = True)

                Output_A.iloc[use] = Output_A_file.iloc[ind_data][class_names].fillna(False)
            
        
        class_counts = Output_A.sum().to_numpy()

        return class_names, class_counts




    def get_classification_data(self, train = True, return_categories = False):
        r'''
        This function retuns inputs and outputs for classification models.

        Parameters
        ----------
        train : bool, optional
            This discribes whether one wants to generate training or testing data. The default is True.
        return_categories : bool, optional
            This indicates if the categories (**C**, see below) of the samples should be returned. 
            The default is *False*.

        Returns
        -------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
            values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
            *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
            the position data (*self.input_data_type = ['x', 'y', ...]*). If an agent is fully or or some timesteps 
            partially not observed, then this can include np.nan values.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings 
            that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
            for available types). If an agent is not observed at all, the value will instead be '0'.
        S : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
            where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
            (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
            instead be np.nan.
        C : np.ndarray, optional
            Optional return provided when return_categories = True. 
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
            category of agent observed, where the categories are dataset specific.
        agent_names : list
            This is a list of length :math:`N_{agents}`, where each string contains the name of a possible 
            agent.
        D : np.ndarray
            This is the generalized past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{dist} \times N_{I}\}` dimensional numpy array with float values. 
            It is dependent on the scenario and represenst characteristic attributes of a scene such as 
            distances between vehicles.
        dist_names : list
            This is a list of length :math:`N_{dist}`, where each string contains the name of a possible 
            characteristic distance.
        class_names : list
            This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
            class.
        P : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array, which for each 
            class contains the probability that it was observed in the sample. As this are observed values, 
            per row, there should be exactly one value 1 and the rest should be zeroes.
            It is only retuned if **train** = *True*.
        DT : np.ndarray, optional
            This is a :math:`N_{samples}` dimensional numpy array, which for each 
            class contains the time period after the prediction time at which the fullfilment of the 
            classification crieria could be observed. It is only retuned if **train** = *True*.
        

        '''
        ## NOTE: Method has been adjusted for large datasets
        self._extract_types()

        # Assert that this is on a single file
        assert self.data_set.data_in_one_piece, 'This method is only useful for datasets that are in one piece.'
        
        # Determine map use
        self.data_set._extract_original_trajectories()
        
        # Extract data from original number a samples
        if self.general_input_available:
            D_help = self.data_set.Input_prediction.to_numpy()
            D = np.ones(list(D_help.shape) + [self.num_timesteps_in], dtype = np.float32) * np.nan
            for i_sample in range(D.shape[0]):
                for i_dist in range(D.shape[1]):
                    D[i_sample, i_dist] = D_help[i_sample, i_dist].astype(np.float32)
            dist_names = self.data_set.Input_prediction.columns
        else:
            D = np.zeros((len(self.data_set.Domain),0, self.num_timesteps_in), dtype = np.float32)
            dist_names = []
        
        # Select current samples
        if train:
            assert self.model_mode == 'train', 'During training, training set should be called.'
            Index = self.Index_train
        else:
            assert self.model_mode == 'pred', 'During training, training set should be called.'
            Index = self.Index_test

        # Get other inputs
        T = self.Type[Index]
        S = self.Size[Index]
        D = D[Index]
        P = self.data_set.Output_A.to_numpy().astype(np.float32)[Index]
        DT = self.data_set.Output_T_E.astype(np.float32)[Index]
        
        # Get the input paths
        # Get the specific data given the index
        data_index, data_mask = self.get_orig_data_index(Index)

        # Initialize and then fill in X
        X = np.full((*T.shape, *self.data_set.X_orig.shape[-2:]), np.nan, np.float32)
        X[data_mask] = self.data_set.X_orig[data_index].astype(np.float32)

        self.input_data_type = self.data_set.Input_data_type[0]
        
        class_names = self.data_set.Behaviors
        agent_names = self.data_set.Agents
        

        if return_categories:
            if 'category' in self.data_set.Domain.columns:
                C = self.data_set.Domain.category.iloc[Index]
                C = pd.DataFrame(C.to_list())

                # Add columns missing from self.data_set.Agents
                C = C.reindex(columns = self.data_set.Agents, fill_value = np.nan)

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C = C.to_numpy()
            else:
                C = None
            
            if train:
                return X, T, S, C, agent_names, D, dist_names, class_names, P, DT
            else:
                return X, T, S, C, agent_names, D, dist_names, class_names
        else:
            if train:
                return X, T, S, agent_names, D, dist_names, class_names, P, DT
            else:
                return X, T, S, agent_names, D, dist_names, class_names
            


    def get_batch_classification_data(self, Sample_id):
        r'''
        This function retuns inputs and outputs for classification models. It has to be called after
        calling self.provide_batch_data().

        Parameters
        ----------
        Sample_id : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which 
            original samples come from. This should be the Sample_id returned by self.provide_batch_data().

        Returns
        -------
        D : np.ndarray
            This is the generalized past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{dist} \times N_{I}\}` dimensional numpy array with float values. 
            It is dependent on the scenario and represenst characteristic attributes of a scene such as 
            distances between vehicles.
        dist_names : list
            This is a list of length :math:`N_{dist}`, where each string contains the name of a possible 
            characteristic distance.
        class_names : list
            This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
            class.
        P : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array, which for each 
            class contains the probability that it was observed in the sample. As this are observed values, 
            per row, there should be exactly one value 1 and the rest should be zeroes.
            It is not returned if Sample_id was generated in *self.provide_batch_data(mode = 'pred')*.
        DT : np.ndarray, optional
            This is a :math:`N_{samples}` dimensional numpy array, which for each 
            class contains the time period after the prediction time at which the fullfilment of the 
            classification crieria could be observed. 
            It is not returned if Sample_id was generated in *self.provide_batch_data(mode = 'pred')*.
        
        '''
        train = self.model_mode != 'pred'

        class_names = self.data_set.Behaviors

        if self.data_set.data_in_one_piece:
            # Extract data from original number a samples
            if self.general_input_available:
                D_help = self.data_set.Input_prediction.to_numpy()[Sample_id]
                D = np.ones(list(D_help.shape) + [self.num_timesteps_in], dtype = np.float32) * np.nan
                for i_sample in range(D.shape[0]):
                    for i_dist in range(D.shape[1]):
                        D[i_sample, i_dist] = D_help[i_sample, i_dist].astype(np.float32)
                dist_names = self.data_set.Input_prediction.columns
            else:
                D = np.zeros((len(Sample_id),0,self.num_timesteps_in), dtype = np.float32)
                dist_names = []

            # Get other inputs
            P = self.data_set.Output_A.iloc[Sample_id].to_numpy().astype(np.float32)
            DT = self.data_set.Output_T_E[Sample_id].astype(np.float32)
        
        
        else:
            domain = self.data_set.Domain.iloc[Sample_id]
            # Get the current file index
            file_indices = domain.file_index

            # assert that only one file is used
            assert len(np.unique(file_indices)) == 1
            file_index = file_indices.iloc[0]

            # Get the specific index in the extracted data
            ind_data = domain.Index_saved

            # Load the data
            data_file = self.data_set.Files[file_index] + '_data.npy'

            # Load the data and extract the data
            [Input_prediction, _, _, _, _, _, Output_A, Output_T_E, _] = np.load(data_file, allow_pickle = True)

            Input_prediction = Input_prediction.loc[ind_data] 
            Output_A = Output_A.loc[ind_data]
            Output_T_E = Output_T_E[ind_data]


            # Get D
            if self.general_input_available:
                D_help = Input_prediction.to_numpy()
                D = np.ones(list(Input_prediction.shape) + [self.num_timesteps_in], dtype = np.float32) * np.nan
                for i_sample in range(D.shape[0]):
                    for i_dist in range(D.shape[1]):
                        D[i_sample, i_dist] = Input_prediction.iloc[i_sample, i_dist].astype(np.float32)
                dist_names = Input_prediction.columns
            else:
                D = np.zeros((len(Sample_id),0,self.num_timesteps_in), dtype = np.float32)
                dist_names = []

            P = Output_A[class_names].fillna(0.0).to_numpy().astype(np.float32)
            DT = Output_T_E.astype(np.float32)

        if train:
            return D, dist_names, class_names, P, DT
        else:
            return D, dist_names, class_names
            

    
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


        ##################################################################################
        # Add node connections                                                           #
        ##################################################################################

        ctrs  = np.zeros((graph.num_nodes, 2), np.float32)
        feats = np.zeros((graph.num_nodes, 2), np.float32)

        node_idcs = []

        unique_lane_segments = list(np.unique(graph.lane_idcs))
        for lane_segment in unique_lane_segments:        
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

        # indexing starts from 0, makes sense
        num_lanes = len(unique_lane_segments)

        ctrs  = torch.from_numpy(ctrs).to(device = device)
        feats = torch.from_numpy(feats).to(device = device)

        # distances between all node centres
        dist = torch.sqrt(((ctrs.unsqueeze(1) - ctrs.unsqueeze(0)) ** 2).sum(2))
        
        
        # allows us to index through all pairs of lane nodes
        # if num_nodes == 3: [0, 1, 2]
        row_idcs = torch.arange(graph.num_nodes).to(dist.device)
        # if num_nodes == 3: [0, 0, 0, 1, 1, 1, 2, 2, 2]
        hi = row_idcs.unsqueeze(1).repeat(1, graph.num_nodes).flatten()
        # if num_nodes == 3: [0, 1, 2, 0, 1, 2, 0, 1, 2]
        wi = row_idcs.unsqueeze(0).repeat(graph.num_nodes, 1).flatten()

        # find possible left and right neighouring nodes
        if cross_angle is not None:
            # along lane
            t_nodes = torch.atan2(feats[:, 1], feats[:, 0])

            # cross lane
            f2 = ctrs[wi] - ctrs[hi]

            # Get the angle of the current node 
            t1 = t_nodes[hi]
            # Get the angle between all node center connection
            t2 = torch.atan2(f2[:, 1], f2[:, 0])

            # Get the difference in angle
            dt = t2 - t1
            
            # Roll around angles
            m = dt > 2 * np.pi
            dt[m] = dt[m] - 2 * np.pi
            m = dt < -2 * np.pi
            dt[m] = dt[m] + 2 * np.pi
            left_mask = torch.logical_and(dt > 0, dt < cross_angle).logical_not()
            right_mask = torch.logical_and(dt < 0, dt > -cross_angle).logical_not()

        # Get the current matrices for pre and suc
        pre_suc_valid = False 
        if len(graph['pre_pairs'].shape) == 2 and len(graph['suc_pairs'].shape) == 2:
            pre = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
            pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
            
            suc = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
            suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1
            
            pre_suc_valid = True

        # find left lane nodes
        if len(graph['left_pairs']) > 0 and pre_suc_valid:
            # get lane segments that are either left, or the predecessor/successor of a left lane
            mat = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
            mat[graph['left_pairs'][:, 0], graph['left_pairs'][:, 1]] = 1
            mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

            # Get the nodes that do not belong to those lanes
            mask = ~ mat[graph.lane_idcs[hi.cpu().numpy()], graph.lane_idcs[wi.cpu().numpy()]]

            # Ignore the nodes that are too far away or not in the correct angle
            left_dist = dist.clone()
            left_dist[hi[mask], wi[mask]] = 1e6
            if cross_angle is not None:
                left_dist[hi[left_mask], wi[left_mask]] = 1e6

            # Find the nodes whose nearest valid neighbor is close enough
            min_dist, min_idcs = left_dist.min(1)
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

            left['u'] = ui[m].cpu().numpy().astype(np.int16)
            left['v'] = vi[m].cpu().numpy().astype(np.int16)
        else:
            left['u'] = np.zeros(0, np.int16)
            left['v'] = np.zeros(0, np.int16)

        # find right lane nodes
        if len(graph['right_pairs']) > 0 and pre_suc_valid:
            # get lane segments that are either right, or the predecessor/successor of a right lane
            mat = torch.zeros((num_lanes, num_lanes), device = device, dtype = torch.float)
            mat[graph['right_pairs'][:, 0], graph['right_pairs'][:, 1]] = 1
            mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

            # Get the nodes that do not belong to those lanes
            mask = ~ mat[graph.lane_idcs[hi.cpu().numpy()], graph.lane_idcs[wi.cpu().numpy()]]

            # Ignore the nodes that are too far away or not in the correct angle
            right_dist = dist.clone()
            right_dist[hi[mask], wi[mask]] = 1e6
            if cross_angle is not None:
                right_dist[hi[right_mask], wi[right_mask]] = 1e6

            # Find the nodes whose nearest valid neighbor is close enough
            min_dist, min_idcs = right_dist.min(1)
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

            right['u'] = ui[m].cpu().numpy().astype(np.int16)
            right['v'] = vi[m].cpu().numpy().astype(np.int16)
        else:
            right['u'] = np.zeros(0, np.int16)
            right['v'] = np.zeros(0, np.int16)

        graph['left'] = [left]
        graph['right'] = [right]

        return graph
    

    def create_empty_output_path(self):
        Agents = np.array(self.data_set.Agents)
        num_rows = len(self.Index_test)
        
        self.Output_path_pred = pd.DataFrame(np.empty((num_rows, len(Agents)), np.ndarray), 
                                             columns = Agents, index = self.Index_test)
        
        self.Output_path_pred_probs = pd.DataFrame(np.empty((num_rows, len(Agents)), np.ndarray), 
                                                    columns = Agents, index = self.Index_test)
        
    
    
    def create_empty_output_A(self):
        Behaviors = np.array(self.data_set.Behaviors)
        num_rows = len(self.Index_test)
            
        self.Output_A_pred = pd.DataFrame(np.zeros((num_rows, len(Behaviors)), float), 
                                          columns = Behaviors, index = self.Index_test)
        
    
    def create_empty_output_T(self):
        Behaviors = np.array(self.data_set.Behaviors)
        num_rows = len(self.Index_test)
            
        self.Output_T_E_pred = pd.DataFrame(np.empty((num_rows, len(Behaviors)), np.ndarray), 
                                            columns = Behaviors, index = self.Index_test)
        for i in range(num_rows):
            for j in range(self.Output_T_E_pred.shape[1]):
                self.Output_T_E_pred.iloc[i,j] = np.ones(len(self.t_e_quantile), float) * np.nan
        
        
    def check_trainability(self):
        # Get predicted agents
        if self.get_output_type()[:4] == 'path':
            self.data_set._determine_pred_agents(eval_pov = self.get_output_type() == 'path_all_wi_pov')
            if self.data_set.Pred_agents_eval.sum() == 0:
                return 'there is no agent of which a trajectory can be predicted.'
            
        if self.get_output_type() in ['class', 'class_and_time']:
            if not self.data_set.classification_useful:
                return 'a classification model cannot be trained on a classless dataset.'
              
        return self.check_trainability_method()
    
    def save_predicted_classifications(self, class_names, P, DT = None):
        r'''
        This function saves the predictions made by the classification model.

        Parameters
        ----------
        class_names : list
            This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
            class.
        P : np.ndarray
            This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array, which for each 
            class contains the predicted probability that it was observed in the sample. As this are 
            probability values, each row should sum up to 1. 
        DT : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{classes} \times N_{q-values}\}` dimensional numpy array, 
            which for each class contains the predicted time after the prediction time at which the 
            fullfilment of the classification crieria for each value could be observed. Each such prediction 
            consists out of the qunatile values (**self.t_e_quantile**) of the predicted distribution.
            The default values is None. An entry is only expected for models which are designed to make 
            these predictions.

        Returns
        -------
        None.

        '''
        
        
        assert self.get_output_type()[:5] == 'class'
        assert P.shape == self.Output_A_pred[class_names].to_numpy().shape
        for i in range(len(P)):
            for j, name in enumerate(class_names):
                self.Output_A_pred.iloc[i][name] = P[i,j]
        
        
        if self.get_output_type() == 'class_and_time':
            assert DT.shape[:2] == self.Output_T_E_pred[class_names].to_numpy().shape
            assert DT.shape[2] == len(self.t_e_quantile)
            for i in range(len(DT)):
                for j, name in enumerate(class_names):
                    self.Output_T_E_pred.iloc[i][name] = DT[i,j]
    
    # %% Method needed for evaluation_template
    def _transform_predictions_to_numpy(self, Pred_index, Output_path_pred, 
                                        exclude_ego = False, exclude_late_timesteps = True, get_for_pred_agents = False):
        if hasattr(self, 'Path_pred') and hasattr(self, 'Path_true') and hasattr(self, 'Pred_step'):
            if self.excluded_ego == exclude_ego:
                if np.array_equal(self.extracted_pred_index, Pred_index):
                    return
        
        # Save last setting 
        self.excluded_ego = exclude_ego
        self.extracted_pred_index = Pred_index

        # Get the file index
        if not self.data_set.data_in_one_piece:
            file_indices = self.data_set.Domain.file_index.iloc[Pred_index].to_numpy()
            assert len(np.unique(file_indices)) == 1, 'This method is only useful for datasets that are in one piece.'
            file_index = file_indices[0]

            used_index = np.where(self.data_set.Domain.file_index == file_index)[0]

            Pred_index_data = self.data_set.get_indices_1D(Pred_index, used_index)

        else:
            file_index = 0

            Pred_index_data = Pred_index

        # Check if data is there
        self.data_set._extract_original_trajectories(file_index)
        self.data_set._determine_pred_agents(eval_pov = not exclude_ego)
        self._extract_types()
        
        # Initialize output
        num_samples = len(Output_path_pred)
        
        # Get predicted timesteps
        Nto_i = self.data_set.N_O_data_orig[Pred_index_data]
        if exclude_late_timesteps:
            Nto_i = np.minimum(self.num_timesteps_out, Nto_i)

        nto_max = Nto_i.max()
        
        # Get pred agents
        if get_for_pred_agents:
            Pred_agents = self.data_set.Pred_agents_pred[Pred_index]
        else:
            Pred_agents = self.data_set.Pred_agents_eval[Pred_index]
        max_num_pred_agents = Pred_agents.sum(1).max()
        
        
        i_agent_sort = np.argsort(-Pred_agents.astype(float))
        i_agent_sort = i_agent_sort[:,:max_num_pred_agents]
        i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_pred_agents))
        
        # Get true predictions
        self.Path_true = np.full((*i_sampl_sort.shape, nto_max, 2), np.nan, np.float32)
        data_index, data_mask = self.get_orig_data_index(Pred_index_data[i_sampl_sort], i_agent_sort)
        self.Path_true[data_mask] = self.data_set.Y_orig[data_index, :nto_max, :2]

        # Get predicted timesteps
        self.Pred_step = Nto_i[:,np.newaxis] > np.arange(nto_max)[np.newaxis]
        self.Pred_step = self.Pred_step[:,np.newaxis] & Pred_agents[i_sampl_sort, i_agent_sort, np.newaxis]
        self.Pred_step = self.Pred_step & np.isfinite(self.Path_true).all(-1)
        
        # Remove nan values
        self.Path_true[~self.Pred_step] = 0.0
        self.Path_true = self.Path_true[:,np.newaxis]
        
        # Get predicted trajectories
        self.Path_pred = np.zeros((self.num_samples_path_pred, num_samples,
                                   max_num_pred_agents, nto_max, 2), dtype = np.float32)
        
        Agents = np.array(self.data_set.Agents)
        for i in range(num_samples):
            pred_agents = np.where(self.Pred_step[i].any(-1))[0]
            
            # Avoid useless samples 
            if len(pred_agents) == 0:
                continue

            pred_agents_id = Agents[i_agent_sort[i, pred_agents]]
            path_pred_orig = Output_path_pred.loc[Pred_index[i], pred_agents_id]
            path_pred = np.stack(path_pred_orig.to_numpy(), axis = 1)
        
            nto_i = min(Nto_i[i], path_pred.shape[2])
            # Assign to full length label
            self.Path_pred[:,i, pred_agents, :nto_i] = path_pred[:,:,:nto_i]
        
        # Set missing values to zero
        self.Path_pred[:,~self.Pred_step,:] = 0.0 
        #Transpose inot right order
        self.Path_pred = self.Path_pred.transpose(1,0,2,3,4)
        
        # Get agent predictions
        self.T_pred = self.Type[Pred_index[i_sampl_sort], i_agent_sort]

        # Get size of agents
        self.S_pred = self.Size[Pred_index[i_sampl_sort], i_agent_sort]

        # Set agent types of agents not included in Pred step to '0'
        self.T_pred[~self.Pred_step.any(-1)] = '0'
        self.S_pred[~self.Pred_step.any(-1)] = np.nan

        # Save the id of the agents used here
        self.Pred_agent_id = i_agent_sort

        # Get agent predictions
        if 'category' in self.data_set.Domain.columns:
            # Sample_id = self.ID[ind_advance,0,0]
            C = self.data_set.Domain.category.iloc[Pred_index]
            C = pd.DataFrame(C.to_list())

            # Add columns missing from self.data_set.Agents
            C = C.reindex(columns = self.data_set.Agents, fill_value = 4)

            # Replace missing agents
            C = C.fillna(4)

            # Get to numpy and apply indices
            C = C.to_numpy().astype(int)
            self.C_pred = C[i_sampl_sort, i_agent_sort]
                
    
    #####################################################################################################
    #                                   Get KDE_true(x_pred)                                            #
    #####################################################################################################


    def _get_joint_KDE_true_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_true_joint_pred'):
            if self.excluded_ego_true_joint == exclude_ego:
                if np.array_equal(self.extracted_pred_index_true_joint, Pred_index):
                    return
                
        # Get the current file_index
        if self.data_set.data_in_one_piece:
            file_index = 0
        else:
            file_indices = self.data_set.Domain.file_index.iloc[Pred_index].to_numpy()
            assert len(np.unique(file_indices)) == 1, 'This method is only useful for datasets that are in one piece.'
            file_index = file_indices[0]
        
        # Have the dataset load
        self.data_set._get_joint_KDE_probabilities(exclude_ego, file_index)
        
        # Save last setting 
        self.excluded_ego_true_joint = exclude_ego
        self.extracted_pred_index_true_joint = Pred_index
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds
        self.Log_prob_true_joint_pred = np.zeros(self.Path_pred.shape[:2], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
        
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = not exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        
        for subgroup in np.unique(Subgroups):
            s_ind = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[s_ind], axis = 0)) == 1
            pred_agents = Pred_agents[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            nto_subgroup = Num_steps[s_ind]
            
            for nto in np.unique(nto_subgroup):
                nto_index = s_ind[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                        
                # Collapse agents
                num_features = pred_agents.sum() * nto * 2
                paths_pred_comp = paths_pred.reshape(*paths_pred.shape[:2], num_features)
                
                # Collapse agents further
                paths_pred_comp = paths_pred_comp.reshape(-1, num_features)
                
                # Evaluate trejatories
                log_prob_pred = self.data_set.KDE_joint[subgroup][nto].score_samples(paths_pred_comp)
                self.Log_prob_true_joint_pred[nto_index] = log_prob_pred.reshape(*paths_pred.shape[:2])
            
            
    def _get_indep_KDE_true_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_true_indep_pred'):
            if self.excluded_ego_true_indep == exclude_ego:
                if np.array_equal(self.extracted_pred_index_true_indep, Pred_index):
                    return
                
        # Get the current file_index
        if self.data_set.data_in_one_piece:
            file_index = 0
        else:
            file_indices = self.data_set.Domain.file_index.iloc[Pred_index].to_numpy()
            assert len(np.unique(file_indices)) == 1, 'This method is only useful for datasets that are in one piece.'
            file_index = file_indices[0]
        
        # Have the dataset load
        self.data_set._get_indep_KDE_probabilities(exclude_ego, file_index)
        
        # Save last setting 
        self.excluded_ego_true_indep = exclude_ego
        self.extracted_pred_index_true_indep = Pred_index
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds x num agents
        self.Log_prob_true_indep_pred = np.zeros(self.Path_pred.shape[:3], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
       
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = not exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        Agents = np.array(self.data_set.Agents)[self.Pred_agent_id]
        
        for subgroup in np.unique(Subgroups):
            s_ind = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[s_ind], axis = 0)) == 1
            pred_agents = Pred_agents[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue
            
            pred_agents_id = np.where(pred_agents)[0]
            
            nto_subgroup = Num_steps[s_ind]
            
            for nto in np.unique(nto_subgroup):
                nto_index = s_ind[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                
                num_features = nto * 2
                
                for i_agent, i_agent_orig in enumerate(pred_agents_id):
                    agent = Agents[nto_index, i_agent_orig]
                    assert len(np.unique(agent)) == 1
                    agent = agent[0]
                    
                    # Get agent
                    paths_pred_agent = paths_pred[:,:,i_agent]
                
                    # Collapse agents
                    paths_pred_agent_comp = paths_pred_agent.reshape(*paths_pred_agent.shape[:2], num_features)
                    
                    # Collapse agents further
                    paths_pred_agent_comp = paths_pred_agent_comp.reshape(-1, num_features)
                    
                    log_prob_pred_agent = self.data_set.KDE_indep[subgroup][nto][agent].score_samples(paths_pred_agent_comp)

                    self.Log_prob_true_indep_pred[nto_index,:,i_agent_orig] = log_prob_pred_agent.reshape(*paths_pred.shape[:2])


    
    #####################################################################################################
    #                           Get KDE_pred(x) and KDE_pred(x_pred)                                    #
    #####################################################################################################

    def _get_joint_KDE_pred_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False, get_for_pred_agents = False):
        if hasattr(self, 'Log_prob_joint_pred') and hasattr(self, 'Log_prob_joint_true'):
            if self.excluded_ego_joint == exclude_ego:
                if np.array_equal(self.extracted_pred_index_joint, Pred_index):
                    return
        
        # Get save file for KDE saving
        file_addon = 'joint_KDE'
        if exclude_ego:
            file_addon += 'wo_pov'
        else:
            file_addon += 'wi_pov'
        kde_file = self.data_set.change_result_directory(self.model_file_metric, 'Predictions', file_addon)

        if not hasattr(self, 'exclude_ego_joint'):
            self.excluded_ego_joint = None
            
        if not ((self.excluded_ego_joint == exclude_ego) and hasattr(self, 'KDE_joint_data')):
            # Load kde data if it exists
            if os.path.exists(kde_file):
                self.KDE_joint_data = np.load(kde_file, allow_pickle = True)[0]
            else:
                self.KDE_joint_data = {}
            
        # Save last setting 
        self.excluded_ego_joint = exclude_ego
        self.extracted_pred_index_joint = Pred_index
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego, get_for_pred_agents = get_for_pred_agents)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds
        self.Log_prob_joint_true = np.zeros(self.Path_true.shape[:2], dtype = np.float32)
        self.Log_prob_joint_pred = np.zeros(self.Path_pred.shape[:2], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
        
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = not exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        
        print('Calculate joint PDF on predicted probabilities.', flush = True)
        for i, subgroup in enumerate(np.unique(Subgroups)):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(i + 1, len(np.unique(Subgroups))), flush = True)
            s_ind = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[s_ind], axis = 0)) == 1
            pred_agents = Pred_agents[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            nto_subgroup = Num_steps[s_ind]

            # Expand upon subgroup
            if not subgroup in self.KDE_joint_data:
                self.KDE_joint_data[subgroup] = {}
            
            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                nto_index = s_ind[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = self.Path_true[nto_index][:,:,pred_agents,:nto]
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                        
                # Collapse agents
                num_features = pred_agents.sum() * nto * 2
                paths_true_comp = paths_true.reshape(*paths_true.shape[:2], num_features)
                paths_pred_comp = paths_pred.reshape(*paths_pred.shape[:2], num_features)
                
                # Collapse agents further
                paths_true_comp = paths_true_comp.reshape(-1, num_features)
                paths_pred_comp = paths_pred_comp.reshape(-1, num_features)

                assert np.isfinite(paths_true_comp).all(), 'There are nan values in the true data.'
                assert np.isfinite(paths_pred_comp).all(), 'There are nan values in the predicted data.'
                
                if not nto in self.KDE_joint_data[subgroup] or self.prediction_overwrite:
                    # Only use select number of samples for training kde
                    # use_preds = np.arange(len(paths_pred_comp))
                    use_preds = np.unique(paths_pred_comp, axis = 0, return_index = True)[1]
                    print('        Number of unique samples: {}/{}'.format(len(use_preds), len(paths_pred_comp)), flush = True)
                    np.random.seed(0)
                    np.random.shuffle(use_preds)
                    max_preds = min(3000, len(use_preds))
                    
                    # Get approximated probability distribution
                    log_pred_satisfied = False
                    i = 0

                    while not log_pred_satisfied and i * max_preds < len(use_preds):
                        test_ind = use_preds[i * max_preds : (i + 1) * max_preds]

                        kde = ROME().fit(paths_pred_comp[test_ind])
                        if i == 0:
                            # Get the combined test indices
                            test_ind_all = test_ind

                            # Get combined kde
                            kde_all = kde
                            labels_all = kde_all.labels_
                        else:
                            # Get the combined test indices
                            test_ind_all = use_preds[:max_preds * (i + 1)]

                            # Get the old combined labels
                            labels_all = kde_all.labels_
                            max_label = labels_all.max() + 1

                            # Get new labels and adjust them
                            labels_new = kde.labels_
                            labels_new[labels_new != -1] += max_label

                            # Get new combined labels
                            labels_all = np.concatenate([labels_all, labels_new])

                            # Refit kde_all
                            kde_all = ROME().fit(paths_pred_comp[test_ind_all], clusters = labels_all)

                        # Score samples
                        log_prob_true = kde_all.score_samples(paths_true_comp)
                        log_prob_pred = kde_all.score_samples(paths_pred_comp)
                        
                        # Check if we sufficiently represent predicted distribution
                        print('            ' + str(i))

                        # Check if further training is needed
                        not_test_ind_all = use_preds[max_preds * (i + 1):]
                        if len(not_test_ind_all) < 30:
                            log_pred_satisfied = True
                        else:
                            included_quant = np.quantile(log_prob_pred[test_ind_all], [0.1, 0.3, 0.5, 0.7, 0.9])
                            unincluded_quant = np.quantile(log_prob_pred[not_test_ind_all], [0.1, 0.3, 0.5, 0.7, 0.9])

                            diff = np.abs(included_quant - unincluded_quant)

                            log_pred_satisfied = np.max(diff) < 0.5
                            print('            Diff train/val: ' + str(np.max(diff)))
                        

                        i += 1

                    # Get the Pred indices of the used train data
                    pred_index_all   = np.repeat(Pred_index[nto_index], self.num_samples_path_pred, axis = 0)
                    pred_index_train = pred_index_all[test_ind_all]

                    # Get the path sample indices
                    sample_index_all   = np.tile(np.arange(self.num_samples_path_pred), len(nto_index))
                    sample_index_train = sample_index_all[test_ind_all]

                    kde_data = {'pred_index': pred_index_train, 'sample_index': sample_index_train, 'cluster_labels': labels_all}
                    self.KDE_joint_data[subgroup][nto] = kde_data
                
                else:
                    kde_data = self.KDE_joint_data[subgroup][nto]
                    pred_index_train   = kde_data['pred_index']
                    sample_index_train = kde_data['sample_index']
                    cluster_labels     = kde_data['cluster_labels']

                    # Get the corresponding training samples
                    pred_index_int_train = self.data_set.get_indices_1D(pred_index_train, Pred_index)
                    path_pred_train = self.Path_pred[pred_index_int_train, sample_index_train][:,pred_agents,:nto] # Shape: num_train_samples x num_agents x num_T_O x 2

                    # Collapse features
                    path_pred_comp_train = path_pred_train.reshape(-1, num_features)

                    # Get the kde
                    kde_all = ROME().fit(path_pred_comp_train, clusters = cluster_labels)

                    # Score samples
                    log_prob_true = kde_all.score_samples(paths_true_comp)
                    log_prob_pred = kde_all.score_samples(paths_pred_comp)



                self.Log_prob_joint_true[nto_index] = log_prob_true.reshape(*paths_true.shape[:2])
                self.Log_prob_joint_pred[nto_index] = log_prob_pred.reshape(*paths_pred.shape[:2])
        
        # Save the KDE data
        if self.data_set.save_predictions:
            os.makedirs(os.path.dirname(kde_file), exist_ok = True)
            np.save(kde_file, np.array([self.KDE_joint_data, 0], dtype = object))
            
            
    def _get_indep_KDE_pred_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False, get_for_pred_agents = False):
        if hasattr(self, 'Log_prob_indep_pred') and hasattr(self, 'Log_prob_indep_true'):
            if self.excluded_ego_indep == exclude_ego:
                if np.array_equal(self.extracted_pred_index_indep, Pred_index):
                    return
                
        # Get save file for KDE saving
        file_addon = 'indep_KDE'
        kde_file = self.data_set.change_result_directory(self.model_file_metric, 'Predictions', file_addon)
        if not hasattr(self, 'KDE_indep_data'):
            # Load kde data if it exists
            if os.path.exists(kde_file):
                self.KDE_indep_data = np.load(kde_file, allow_pickle = True)[0]
            else:
                self.KDE_indep_data = {}
        
        # Save last setting 
        self.excluded_ego_indep = exclude_ego
        self.extracted_pred_index_indep = Pred_index
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego, get_for_pred_agents = get_for_pred_agents)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds x num agents
        self.Log_prob_indep_true = np.zeros(self.Path_true.shape[:-2], dtype = np.float32)
        self.Log_prob_indep_pred = np.zeros(self.Path_pred.shape[:-2], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
       
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = not exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        Agents = np.array(self.data_set.Agents)[self.Pred_agent_id]
        
        print('Calculate indep PDF on predicted probabilities.', flush = True)
        for i, subgroup in enumerate(np.unique(Subgroups)):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(i + 1, len(np.unique(Subgroups))), flush = True)
            s_ind = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[s_ind], axis = 0)) == 1
            pred_agents = Pred_agents[s_ind[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            pred_agents_id = np.where(pred_agents)[0]
            
            nto_subgroup = Num_steps[s_ind]

            if not subgroup in self.KDE_indep_data:
                self.KDE_indep_data[subgroup] = {}
            
            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                nto_index = s_ind[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = self.Path_true[nto_index][:,:,pred_agents,:nto]
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                
                num_features = nto * 2
                
                if not nto in self.KDE_indep_data[subgroup]:
                    self.KDE_indep_data[subgroup][nto] = {}

                for i_agent, i_agent_orig in enumerate(pred_agents_id):
                    # Get the agent name
                    agent = Agents[nto_index, i_agent_orig]
                    assert len(np.unique(agent)) == 1
                    agent = agent[0]
                    print('            Agent ' + agent + ' ({:3.0f}/{:3.0f})'.format(i_agent + 1, len(pred_agents_id)), flush = True)

                    # Get agent
                    paths_true_agent = paths_true[:,:,i_agent]
                    paths_pred_agent = paths_pred[:,:,i_agent]
                
                    # Collapse agents
                    paths_true_agent_comp = paths_true_agent.reshape(*paths_true_agent.shape[:2], num_features)
                    paths_pred_agent_comp = paths_pred_agent.reshape(*paths_pred_agent.shape[:2], num_features)
                    
                    # Collapse agents further
                    paths_true_agent_comp = paths_true_agent_comp.reshape(-1, num_features)
                    paths_pred_agent_comp = paths_pred_agent_comp.reshape(-1, num_features)

                    assert np.isfinite(paths_true_agent_comp).all(), 'There are nan values in the true data.'
                    assert np.isfinite(paths_pred_agent_comp).all(), 'There are nan values in the predicted data.'
                    
                    if not agent in self.KDE_indep_data[subgroup][nto] or self.prediction_overwrite:
                        # Only use select number of samples for training kde
                        # use_preds = np.arange(len(paths_pred_agent_comp))
                        use_preds = np.unique(paths_pred_agent_comp, axis = 0, return_index = True)[1]
                        print('            Number of unique samples: {}/{}'.format(len(use_preds), len(paths_pred_agent_comp)), flush = True)
                        np.random.seed(0)
                        np.random.shuffle(use_preds)
                        max_preds = min(3000, len(use_preds))
                    
                        # Get approximated probability distribution
                        log_pred_satisfied = False
                        i = 0

                        while not log_pred_satisfied and i * max_preds < len(use_preds):
                            test_ind = use_preds[i * max_preds : (i + 1) * max_preds]

                            kde = ROME().fit(paths_pred_agent_comp[test_ind])
                            if i == 0:
                                # Get the combined test indices
                                test_ind_all = test_ind

                                # Get combined kde
                                kde_all = kde
                                labels_all = kde_all.labels_
                            else:
                                # Get the combined test indices
                                test_ind_all = use_preds[:max_preds * (i + 1)]

                                # Get the old combined labels
                                labels_all = kde_all.labels_
                                max_label = labels_all.max() + 1

                                # Get new labels and adjust them
                                labels_new = kde.labels_
                                labels_new[labels_new != -1] += max_label

                                # Get new combined labels
                                labels_all = np.concatenate([labels_all, labels_new])

                                # Refit kde_all
                                kde_all = ROME().fit(paths_pred_agent_comp[test_ind_all], clusters = labels_all)

                            # Score samples
                            log_prob_true_agent = kde_all.score_samples(paths_true_agent_comp)
                            log_prob_pred_agent = kde_all.score_samples(paths_pred_agent_comp)
                            
                            # Check if we sufficiently represent predicted distribution
                            print('            ' + str(i))

                            # Check if further training is needed
                            not_test_ind_all = use_preds[max_preds * (i + 1):]
                            if len(not_test_ind_all) < 30:
                                log_pred_satisfied = True
                            else:
                                included_quant = np.quantile(log_prob_pred_agent[test_ind_all], [0.1, 0.3, 0.5, 0.7, 0.9])
                                unincluded_quant = np.quantile(log_prob_pred_agent[not_test_ind_all], [0.1, 0.3, 0.5, 0.7, 0.9])

                                diff = np.abs(included_quant - unincluded_quant)

                                log_pred_satisfied = np.max(diff) < 0.5
                                print('            Diff train/val: ' + str(np.max(diff)))
                            

                            i += 1

                        # Save KDE data
                        # save the pred indices
                        pred_index_all   = np.repeat(Pred_index[nto_index], self.num_samples_path_pred, axis = 0)
                        pred_index_train = pred_index_all[test_ind_all]

                        # Get the path sample indices
                        sample_index_all   = np.tile(np.arange(self.num_samples_path_pred), len(nto_index))
                        sample_index_train = sample_index_all[test_ind_all]

                        kde_data = {'pred_index': pred_index_train, 'sample_index': sample_index_train, 'cluster_labels': labels_all}
                        self.KDE_indep_data[subgroup][nto][agent] = kde_data
                    
                    else:
                        # Get the saved data
                        kde_data = self.KDE_indep_data[subgroup][nto][agent]
                        pred_index_train   = kde_data['pred_index']
                        sample_index_train = kde_data['sample_index']
                        cluster_labels     = kde_data['cluster_labels']

                        # Get the corresponding training samples
                        pred_index_int_train = self.data_set.get_indices_1D(pred_index_train, Pred_index)
                        path_pred_agent_train = self.Path_pred[pred_index_int_train, sample_index_train][:,i_agent_orig,:nto] # Shape: num_train_samples x num_T_O x 2

                        # Collapse features
                        path_pred_agent_comp_train = path_pred_agent_train.reshape(-1, num_features)

                        # Get the kde
                        kde_all = ROME().fit(path_pred_agent_comp_train, clusters = cluster_labels)

                        # Score samples
                        log_prob_true_agent = kde_all.score_samples(paths_true_agent_comp)
                        log_prob_pred_agent = kde_all.score_samples(paths_pred_agent_comp)
                            

                    self.Log_prob_indep_true[nto_index,:,i_agent_orig] = log_prob_true_agent.reshape(*paths_true.shape[:2])
                    self.Log_prob_indep_pred[nto_index,:,i_agent_orig] = log_prob_pred_agent.reshape(*paths_pred.shape[:2])
        
        # Save the KDE data
        if self.data_set.save_predictions:
            os.makedirs(os.path.dirname(kde_file), exist_ok = True)
            np.save(kde_file, np.array([self.KDE_indep_data, 0], dtype = object))
                
    #%% 
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                            Model dependend functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    

    def get_name(self = None):
        # Provides a dictionary with the different names of the dataset:
        # Name = {'print': 'printable_name', 'file': 'name_used_in_files', 'latex': r'latex_name'}
        # If the latex name includes mathmode, the $$ has to be included
        # Here, it has to be noted that name_used_in_files will be restricted in its length.
        # For models, this length is 10 characters, without a '-' inside
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def requires_torch_gpu(self = None):
        # Returns true or false, depending if the model does calculations on the gpu
        raise AttributeError('Has to be overridden in actual model.')
        
            
    def get_output_type(self = None):
        # Should return 'class', 'class_and_time', 'path_all_wo_pov', 'path_all_wi_pov'
        # the same as above, only this time callable from class
        raise AttributeError('Has to be overridden in actual model.')
        
    def save_params_in_csv(self = None):
        # Returns true or false, depending on if the params of the trained model should be saved as a .csv file or not
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def provides_epoch_loss(self = None):
        # Returns true or false, depending on if the model provides losses or not
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def check_trainability_method(self):
        # checks if current environment (i.e, number of agents, number of input timesteps) make this trainable.
        # If not, it also retuns the reason in form of a string.
        # If it is trainable, return None
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def setup_method(self):
        # setsup the model, but only use the less expensive calculations.
        # Anything more expensive, especially if only needed for training,
        # should be done in train_method or load_method instead.
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def train_method(self):
        # Uses input data to train the model, saving the weights needed to fully
        # describe the trained model in self.weight_saved
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def load_method(self):
        # Builds the models using loaded weights in self.weights_saved
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def predict_method(self):
        # takes test input and uses that to predict the output
        raise AttributeError('Has to be overridden in actual model.')
    
    def provides_likelihoods(self):
        r'''
        This function returns the information of wheter the model can provide likelihoods associated
        with predicted trajectories. 

        WARNING: If the underlying probability density is not normalized, the metrics based on these
        likelihoods will become meaningless! Please keep this in mind.
        
        Returns
        -------
        can_make_prob_prediction : bool
            The boolean value depicting the ability of the model to calculate log likelihoods.
      
        '''
        return False #TODO: Remove
        raise AttributeError('Has to be overridden in actual model.')
        
    def calculate_log_likelihoods(self, X, Y, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id): 
        r'''
        Given an batch of input, the model calculates the predicted probability density function. This is 
        then applied to the provided ground truth trajectories.

        WARNING: If the underlying probability density is not normalized, the metrics based on these
        likelihoods will become meaningless! Please keep this in mind.
        
        Parameters
        -------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
            values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
            *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
            the position data (*self.input_data_type = ['x', 'y', ...]*). It must be noted that *self.input_data_type*
            will always correspond to the output of the *path_data_info()* of the data_set from which this batch data
            was loaded. If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
        Y : np.ndarray
            This is the future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times N_{data}\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        S : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
            where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
            (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
            instead be np.nan.
        C : np.ndarray 
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
            category of agent observed, where the categories are dataset specific. If the dataset does not include such
            information, it will be set to None.
        img : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
            It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
            of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
            at its current position, and are rotated so that the agent is right now driving to the right. 
            If an agent is not observed at prediction time, 0 values are returned.
        img_m_per_px : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
            the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
        graph : np.ndarray
            This is a numpy array with length :math:`N_{samples}`, where the entries are pandas.Series with the following entries:
            
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

        Pred_agents : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
            if it expected by the framework that a prediction will be made for the specific agent.
            
            If only one agent has to be predicted per sample, for **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
            be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.
        num_steps : int
            This is the number of future timesteps provided in the case of traning in expected in the case of prediction. In the 
            former case, it has the value :math:`N_{O}`.
        Sample_id : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted.
        Agent_id : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted (for corresponding string names see self.data_set.Agents).
            
            
        Returns
        -------
        GT_log_probs : np.ndarray
            This is a :math:`\{N_{samples} \times M_{agents} \times N_{preds}\}` dimensional numpy array. it includes float values, with 
            the model assigned log likelihoods. Here, :math:`M_{agents} = N_{agents}` if **self.predict_single_agent** = *False* (i. e., 
            the model expects marginal likelihoods), while joint likelihoods are expected for the case of **self.predict_single_agent** = 
            *True*, (resulting in :math:`M_{agents} = 1`). In the former cases, this can include np.nan values for non predicted agents.
        
        '''
        
        raise AttributeError('Has to be overridden in actual model.')           

