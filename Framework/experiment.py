import numpy as np
import torch
import pandas as pd
import os
import importlib
import sys
import warnings
import psutil
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import time
import seaborn as sns
from utils.memory_utils import get_total_memory, get_used_memory

# allow for latex code
# from matplotlib import rc
# rc('text', usetex=True)

# import data_interface
path = os.path.dirname(os.path.realpath(__file__))

# Add path towards scenarios
scenario_path = path + os.sep + 'Scenarios' + os.sep
if not scenario_path in sys.path:
    sys.path.insert(0, scenario_path)
    
# Add path towards datasets
data_set_path = path + os.sep + 'Data_sets' + os.sep
if not data_set_path in sys.path:
    sys.path.insert(0, data_set_path)
       
# Add path towards splitting methods
split_path = path + os.sep + 'Splitting_methods' + os.sep
if not split_path in sys.path:
    sys.path.insert(0, split_path)

# Add path towards models
model_path = path + os.sep + 'Models' + os.sep
if not model_path in sys.path:
    sys.path.insert(0, model_path)

# Add path towards metrics
metrics_path = path + os.sep + 'Evaluation_metrics' + os.sep
if not metrics_path in sys.path:
    sys.path.insert(0, metrics_path)

# Add path towards perturbation methods
perturbation_path = path + os.sep + 'Perturbation_methods' + os.sep
if not perturbation_path in sys.path:
    sys.path.insert(0, perturbation_path)

from data_interface import data_interface

# Filter out the DeprecationWarning messages
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# plt.rcParams['text.usetex'] = False


class Experiment():
    def __init__(self, Experiment_name = ''):
        self.path = os.path.dirname(os.path.realpath(__file__))
            
        self.provided_modules = False
        self.provided_setting = False
        self.results_loaded = False
        
        assert type(Experiment_name) == type('Test'), "Experiment_name must be a string."
        self.Experiment_name = Experiment_name
        
        print('This are the current computer specs:')
        print('')
        print('NumPy version:   ', np.__version__)
        print('PyTorch version: ', torch.__version__)
        print('Pandas version:  ', pd.__version__)
        print('')

        print('Processor memories:')
        # CPU
        CPU_mem = psutil.virtual_memory()
        self.total_memory = get_total_memory()
        cpu_total = self.total_memory / 2 ** 30
        cpu_used  = get_used_memory() / 2 ** 30
        print('CPU: {:5.2f}/{:5.2f} GB are available'.format(cpu_total - cpu_used, cpu_total))

        # GPU
        if not torch.cuda.is_available():
            print("GPU: not available")
        else:
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            gpu_total         = torch.cuda.get_device_properties(device = device).total_memory  / 2 ** 30
            gpu_reserved      = torch.cuda.memory_reserved(device = device) / 2 ** 30
            torch.cuda.reset_peak_memory_stats()
            print('GPU: {:5.2f}/{:5.2f} GB are available'.format(gpu_total - gpu_reserved, gpu_total))

        print('')
    
    #%% Experiment setup        
    def set_modules(self, Data_sets, Data_params, Splitters, Models, Metrics):
        assert isinstance(Data_sets, list), "Data_sets must be a list."
        assert len(Data_sets) > 0, "Data_sets must not be empty."
        
        assert isinstance(Data_params, list), "Data_params must be a list."
        assert len(Data_params) > 0, "Data_params must not be empty."
        
        assert isinstance(Splitters, list), "Splitters must be a list."
        assert len(Splitters) > 0, "Splitters must not be empty."
        
        assert isinstance(Models, list), "Models must be a list."
        assert len(Models) > 0, "Models must not be empty."
        
        assert isinstance(Metrics, list), "Metrics must be a list."

        if not 'NWA' in self.Experiment_name: 
            assert len(Metrics) > 0, "Metrics must not be empty."
        
        self.num_data_sets   = len(Data_sets)
        self.num_data_params = len(Data_params)
        self.num_models      = len(Models)
        self.num_metrics     = len(Metrics)
        
        self.Data_sets   = Data_sets
        self.Data_params = Data_params
        
        # Check if multiple splitter repetitions have been provided
        self.Splitters = []
        for split_dict in Splitters:
            assert isinstance(split_dict, dict), "Split is not provided as a dictionary."
            assert 'Type' in split_dict.keys(), "Type name is missing from split."
            
            splitter_name = split_dict['Type']
            
            if 'test_part' in split_dict.keys():
                splitter_tp = split_dict['test_part']
                assert isinstance(splitter_tp, float), "Test split size has to be a float"
                assert ((0 < splitter_tp) and (splitter_tp < 1)), "Test split size has to be in (0, 1)"
            else:
                splitter_tp = 0.2
            
            if 'repetition' in split_dict.keys():
                reps = split_dict['repetition']
                if not isinstance(reps, list):
                    reps = [reps]

                for i, rep in enumerate(reps):
                    assert (isinstance(rep, int) or
                            isinstance(rep, str) or
                            isinstance(rep, tuple)), "Split repetition has a wrong format."
                    if isinstance(rep, tuple):
                        assert len(rep) > 0, "Some repetition information must be given."
                        for rep_part in rep:
                            assert (isinstance(rep_part, int) or
                                    isinstance(rep_part, str)), "Split repetition has a wrong format."
                    else:
                        reps[i] = (rep,)
            else:
                reps = [(0,)]
            
            if 'train_pert' in split_dict.keys():
                train_pert = split_dict['train_pert']
                assert isinstance(train_pert, bool), "train_pert must be a boolean."
            else:
                train_pert = False

            if 'test_pert' in split_dict.keys():
                test_pert = split_dict['test_pert']
                assert isinstance(test_pert, bool), "test_pert must be a boolean."
            else:
                test_pert = False

            if 'train_on_test' in split_dict.keys():
                train_on_test = split_dict['train_on_test']
                assert isinstance(train_on_test, bool), "train_on_test must be a boolean."
            else:
                train_on_test = False
                
                    
            for rep in reps:
                new_split_dict = {'Type': splitter_name, 'repetition': rep, 'test_part': splitter_tp, 
                                  'train_pert': train_pert, 'test_pert': test_pert, 'train_on_test': train_on_test}
                self.Splitters.append(new_split_dict)
        
        self.num_splitters = len(self.Splitters)
        
        # Check if Models are all depicted correctly
        self.Models = []
        for model in Models:
            if isinstance(model, str):
                model_dict = {'model': model, 'kwargs': {}}
            elif isinstance(model, dict):
                assert 'model' in model.keys(), "No model name is provided."
                assert isinstance(model['model'], str), "A model is set as a string."
                model_dict = model
                if not 'kwargs' in model.keys():
                    model_dict['kwargs'] = {}
                else:
                    assert isinstance(model_dict['kwargs'], dict), "The kwargs value must be a dictionary."
            else:
                raise TypeError("The provided model must be string or dictionary")
            
            self.Models.append(model_dict)

        # Check if Metrics are all depicted correctly
        self.Metrics = []
        for metric in Metrics:
            if isinstance(metric, str):
                metric_dict = {'metric': metric, 'kwargs': {}}
            elif isinstance(metric, dict):
                assert 'metric' in metric.keys(), "No metric name is provided."
                assert isinstance(metric['metric'], str), "A metric is set as a string."
                metric_dict = metric
                if not 'kwargs' in metric.keys():
                    metric_dict['kwargs'] = {}
                else:
                    assert isinstance(metric_dict['kwargs'], dict), "The kwargs value must be a dictionary."
            else:
                raise TypeError("The provided metric must be string or dictionary")
            
            self.Metrics.append(metric_dict)
        
        self.provided_modules = True
        
        
    def set_parameters(self, model_for_path_transform,
                       num_samples_path_pred = 100, 
                       enforce_num_timesteps_out = False, 
                       enforce_prediction_times = True, 
                       exclude_post_crit = True,
                       allow_extrapolation = True,
                       agents_to_predict = 'predefined',
                       overwrite_results = False,
                       save_predictions = True,
                       evaluate_on_train_set = True,
                       allow_longer_predictions = True):
        
        model_to_path_module = importlib.import_module(model_for_path_transform)
        model_class_to_path = getattr(model_to_path_module, model_for_path_transform)
        
        # Test the parameters        
        if model_class_to_path == None or model_class_to_path.get_output_type() != 'path_all_wi_pov':
            raise TypeError("The chosen model does not predict trajectories.")

        # Set the number of paths that a trajectory prediction model has to predict
        assert isinstance(num_samples_path_pred, int), "num_samples_path_pred should be an integer."
        
        assert isinstance(enforce_num_timesteps_out, bool), "enforce_num_timesteps_out should be a boolean."
        
        assert isinstance(enforce_prediction_times, bool), "enforce_prediction_time should be a boolean."
        
        assert isinstance(exclude_post_crit, bool), "exclude_post_crit should be a boolean."
        
        assert isinstance(allow_extrapolation, bool), "allow_extrapolation should be a boolean."
        
        assert isinstance(agents_to_predict, str), "dynamic_prediction_agents should be a string."
        
        # make it backward compatiable
        if isinstance(overwrite_results, bool):
            if overwrite_results == False:
                overwrite_results = 'no'
            else:
                overwrite_results = 'model'
        
        assert isinstance(overwrite_results, str), "overwrite_results should be a string."
        assert overwrite_results in ['model', 'prediction', 'metric', 'no']
        
        
        assert isinstance(evaluate_on_train_set, bool), "evaluate_on_train_set should be a boolean."
        

        # Save parameters needed in actual data_set_template
        self.parameters = [model_class_to_path, num_samples_path_pred, 
                           enforce_num_timesteps_out, enforce_prediction_times, 
                           exclude_post_crit, allow_extrapolation, 
                           agents_to_predict, overwrite_results, 
                           allow_longer_predictions, save_predictions,  
                           self.total_memory]
        
        # Save the remaining parameters
        self.evaluate_on_train_set = evaluate_on_train_set
        
        self.provided_setting = True
    
      
    #%% Running
    def print_data_set_status(self, i, j, data_set, data_param, data_failure):
        
        print('')
        print('------------------------------------------------------------------------------------')
        print('On dataset ' + data_set.get_name()['print'] + 
              ' at prediction time setting ' + data_set.t0_type + 
              ' ({}/{})'.format(i + 1, self.num_data_sets), flush = True)
        n_I = data_param['num_timesteps_in']
        
        if not isinstance(n_I, tuple):
            n_I = (n_I, n_I)
        
        print('with dt = ' + '{:0.2f}'.format(max(0, min(9.99, data_param['dt']))).zfill(4) + 
              ' and n_I = {}->{}'.format(*n_I) + ' ({}/{})'.format(j + 1, self.num_data_params), flush = True)
        
        if data_set.classification_useful:
            sample_string = ''
            for beh in data_set.Behaviors:
                if beh != data_set.Behaviors[-1]:
                    if len(data_set.Behaviors) > 2:
                        sample_string += '{}/{} '.format(data_set.num_behaviors_out[beh], 
                                                         data_set.num_behaviors[beh]) + beh + ', '
                    else:
                        sample_string += '{}/{} '.format(data_set.num_behaviors_out[beh], 
                                                         data_set.num_behaviors[beh]) + beh + ' '
                else:                            
                    sample_string += 'and {}/{} '.format(data_set.num_behaviors_out[beh], 
                                                         data_set.num_behaviors[beh]) + beh        
            sample_string += ' samples are admissible.'
        else:
            sample_string = ''
        print(sample_string)
        print('')
        
        if data_failure is not None:
            print('However, this dataset is not usable, because ' + data_failure)
            print('')
            
            
    def print_split_status(self, k, splitter, split_failure):      
        if split_failure is not None:
            print('However, ' + splitter.get_name()['print'] + 
                  ' ({}/{})'.format(k + 1, self.num_splitters) + 
                  ' is not applicable, because ' + split_failure, flush = True)
            print('')
        else:
            print('Under ' + splitter.get_name()['print'] + 
                  ' ({}/{})'.format(k + 1, self.num_splitters), flush = True)
            
    def print_model_status(self, l, model, model_failure):                
        print('train the model ' + model.get_name()['print'] + 
              ' ({}/{}).'.format(l + 1, self.num_models), flush = True)
        print('')
        
        if model_failure is not None:
            print('However, the model ' + model.get_name()['print'] + ' cannot be trained, because ' + model_failure, flush = True)
            print('', flush = True)
        else:
            print('The model ' + model.get_name()['print'] + ' will be trained.', flush = True)
            print('', flush = True)

    def print_metric_status(self, metric, metric_failure = None):
        if not metric.data_set.classification_possible:
            if metric.get_output_type()[:5] == 'class':
                metric_failure = 'the dataset does not allow for classifications.'
        
        if metric_failure is not None:
            print('The metric ' + metric.get_name()['print'] + ' cannot be used, because ' + metric_failure, flush = True)
            print('', flush = True)
        else:
            print('The metric ' + metric.get_name()['print'] + ' is used for evaluation. ' + 
                  'This might require the training of a transformation model.', flush = True)
            

    def run(self):
        assert self.provided_modules, "No modules have been provided. Run self.set_modules() first."
        assert self.provided_setting, "No parameters have been provided. Run self.set_parameters() first."

        print('Starting the running of the benchmark', flush = True)
        for i, data_set_dict in enumerate(self.Data_sets):
            # Get data set class
            data_set = data_interface(data_set_dict, self.parameters)
            
            # Go through each type of data params
            for j, data_param in enumerate(self.Data_params):
                # Reset data set
                data_set.reset()
                # Select or load repective datasets
                data_failure = data_set.get_data(**data_param)
                
                # Do not proceed if dataset cannot be created
                if data_failure is not None:
                    # Provide outputs
                    self.print_data_set_status(i, j, data_set, data_param, data_failure)
                    continue
                
                # Go through each splitting method
                for k, splitter_param in enumerate(self.Splitters):
                    # Get splitting method class
                    splitter_name       = splitter_param['Type']
                    splitter_rep        = splitter_param['repetition']
                    splitter_tp         = splitter_param['test_part']
                    splitter_train_pert = splitter_param['train_pert']
                    splitter_test_pert  = splitter_param['test_pert']
                    splitter_tot        = splitter_param['train_on_test']

                    splitter_module = importlib.import_module(splitter_name)
                    splitter_class = getattr(splitter_module, splitter_name)

                    # Initialize Splitting method
                    splitter = splitter_class(data_set, splitter_tp, splitter_rep, splitter_train_pert, splitter_test_pert, splitter_tot)
                    
                    # Check if splitting method can be used
                    split_failure = splitter.check_splitability()
                    
                    # Do not use splitter if it cannot be used
                    if split_failure is not None:
                        # Provide outputs
                        self.print_data_set_status(i, j, data_set, data_param, data_failure)
                        self.print_split_status(k, splitter, split_failure)
                        continue
                    
                    # Use splitting method to get train and test samples
                    splitter.split_data()
                                                               
                    # Go through each model to be trained
                    for l, model_dict in enumerate(self.Models):
                        # get model subjects
                        model_name   = model_dict['model']
                        model_kwargs = model_dict['kwargs']
                        
                        # Get model class
                        model_module = importlib.import_module(model_name)
                        model_class = getattr(model_module, model_name)
                        
                        # Initialize the model
                        model = model_class(model_kwargs, data_set, splitter, self.evaluate_on_train_set)
                        
                        # Check if the model can be trained on this dataset
                        model_failure = model.check_trainability()
                        
                        # Provide outputs
                        self.print_data_set_status(i, j, data_set, data_param, data_failure)
                        self.print_split_status(k, splitter, split_failure)
                        self.print_model_status(l, model, model_failure) 
                        
                        # Do not train model if not possible
                        if model_failure is not None:
                            continue
                        
                        # Train the model on the given training set
                        model.train()
                        
                        # For large dataset, the separate calculation of the predictions and metrics is not
                        # possible due to memory constraints. Therefore, the predictions and evaluations are calculated
                        # at the same time. For this, we have to create a new function called predict_and_evaluate()
                        model.predict_and_evaluate(self.Metrics, self.print_metric_status)
    
    #%% Loading results
    def load_results(self, plot_if_possible = True, return_train_results = False, return_train_loss = False):
        assert self.provided_modules, "No modules have been provided. Run self.set_modules() first."
        assert self.provided_setting, "No parameters have been provided. Run self.set_parameters() first."
        
        # PReprocess splitters
        Split_type = []
        for i, split in enumerate(self.Splitters):
            Split_type.append(split['Type'])

        self.Split_types, split_index = np.unique(Split_type, return_inverse = True)
        self.Split_indices = []
        for i, split_type in enumerate(self.Split_types):
            self.Split_indices.append(np.where(split_index == i)[0])
            
        # Metrics
        Metrics_minimize = [] 
        Metrics_log_scale = []

        for metric_dict in self.Metrics:
            metric_name = metric_dict['metric']
            metric_module = importlib.import_module(metric_name)
            metric_class = getattr(metric_module, metric_name)
            Metrics_minimize.append(metric_class.get_opt_goal() == 'minimize')
            Metrics_log_scale.append(metric_class.is_log_scale())
            
        self.Metrics_minimize  = np.array(Metrics_minimize)
        self.Metrics_log_scale = np.array(Metrics_log_scale)

        self.Results = np.ones((self.num_data_sets,
                                self.num_data_params,
                                self.num_splitters,
                                self.num_models,
                                self.num_metrics),
                               float) * np.nan
        
        if return_train_results:
            self.Train_results = np.ones((self.num_data_sets,
                                          self.num_data_params,
                                          self.num_splitters,
                                          self.num_models,
                                          self.num_metrics),
                                         float) * np.nan
            
        if return_train_loss:
            self.Train_loss = np.ones((self.num_data_sets,
                                       self.num_data_params,
                                       self.num_splitters,
                                       self.num_models),
                                      np.ndarray) * np.nan

        for i, data_set_dict in enumerate(self.Data_sets):
                # Get data set class
            data_set = data_interface(data_set_dict, self.parameters)

            for j, data_param in enumerate(self.Data_params):
                data_set.reset()
                data_set.get_data(**data_param)
                for k, splitter_param in enumerate(self.Splitters):
                    splitter_name       = splitter_param['Type']
                    splitter_rep        = splitter_param['repetition']
                    splitter_tp         = splitter_param['test_part']
                    splitter_train_pert = splitter_param['train_pert']
                    splitter_test_pert  = splitter_param['test_pert']
                    splitter_tot        = splitter_param['train_on_test']

                    splitter_module = importlib.import_module(splitter_name)
                    splitter_class = getattr(splitter_module, splitter_name)

                    splitter = splitter_class(data_set, splitter_tp, splitter_rep, splitter_train_pert, splitter_test_pert, splitter_tot)

                    # Get the name of the splitmethod used.
                    splitter_str = splitter.get_name()['file'] + splitter.get_rep_str()

                    for m, metric_dict in enumerate(self.Metrics):
                        metric_name = metric_dict['metric']
                        metric_kwargs = metric_dict['kwargs']
                        metric_module = importlib.import_module(metric_name)
                        metric_class = getattr(metric_module, metric_name)
                        metric = metric_class(metric_kwargs, data_set, splitter, None)
                            
                        create_plot = plot_if_possible and metric.allows_plot()
                        if create_plot:
                            fig, ax = plt.subplots(figsize = (4,4))
                        
                        for l, model_dict in enumerate(self.Models):
                            # get model subjects
                            model_name   = model_dict['model']
                            model_kwargs = model_dict['kwargs']
                            
                            # Get model instance
                            model_module = importlib.import_module(model_name)
                            model_class = getattr(model_module, model_name)
                            
                            # Initialize the model
                            model = model_class(model_kwargs, data_set, splitter, self.evaluate_on_train_set)
                            model_str = model.get_name()['file']
                            if '--pretrain' in model.model_file:
                                model_str += '--pretrain_' + model.model_file[:-4].split('--pretrain_')[-1]
                            
                            results_file_name = (data_set.data_file[:-4] + '--' + 
                                                 # Add splitting method
                                                 splitter_str + '--' + 
                                                 # Add model name
                                                 model_str + '--' + 
                                                 # Add metric name
                                                 metric.get_name()['file']  + '.npy')
                            
                            results_file_name = results_file_name.replace(os.sep + 'Data' + os.sep,
                                                                          os.sep + 'Metrics' + os.sep)
                            
                            
                            
                            # print('--'.join(os.path.basename(results_file_name).split('--')[-2:]))
                            if os.path.isfile(results_file_name):
                                try:
                                    metric_result = np.load(results_file_name, allow_pickle = True)[:-1]
                                    
                                    if return_train_results:
                                        train_results = metric_result[0]
                                        if train_results is not None:
                                            self.Train_results[i,j,k,l,m] = train_results[0]
                                    
                                    test_results = metric_result[1]    
                                    self.Results[i,j,k,l,m] = test_results[0]

                                    
                                    if create_plot:
                                        figure_file = data_set.change_result_directory(results_file_name, 'Metric_figures', '')
                                        
                                        # remove model name from figure file
                                        num = 6 + len(model.get_name()['file']) + len(metric.get_name()['file'])
                                        figure_file = figure_file[:-num] + metric.get_name()['file'] + '.pdf'
                                        
                                        os.makedirs(os.path.dirname(figure_file), exist_ok = True)
                                        saving_figure = l == (self.num_models - 1)
                                        metric.create_plot(test_results, figure_file, fig, ax, saving_figure, model)
                                except:
                                    print('Desired result cannot be opened: ' + '--'.join(results_file_name.split('--')[-2:]))
                            else:
                                print('Desired result not findable.')
                                
                            if m == 0 and return_train_loss:
                                if model.provides_epoch_loss():
                                    # Adjust splitter_str
                                    splitter_str_new = splitter_str + ''
                                    if '_pert=' in splitter_str_new:
                                        pert_split = splitter_str_new.split('_pert=')
                                        splitter_str_new = pert_split[0] + '_pert=' + pert_split[1][0] + pert_split[1][2:]

                                    train_loss_file_name = (data_set.data_file[:-4] + '--' + 
                                                            # Add splitting method
                                                            splitter_str_new + '--' +  
                                                            # Add model name
                                                            model.get_name()['file']  + '--train_loss.npy')
                                    
                                    train_loss_file_name = train_loss_file_name.replace(os.sep + 'Data' + os.sep,
                                                                                        os.sep + 'Models' + os.sep)
                                    
                                    if os.path.isfile(train_loss_file_name):
                                        train_loss = np.load(train_loss_file_name, allow_pickle = True)
                                        
                                        self.Train_loss[i,j,k,l] = train_loss
                                        
                                    else:
                                        print('Desired train loss is not available not findable')
                                else:
                                    print('The model ' + model.get_name()['print'] + ' does not provide training losses.')
                                  
        self.results_loaded = True
        
        if return_train_results and return_train_loss:
            return self.Results, self.Train_results, self.Train_loss
        elif return_train_results and not return_train_loss:
            return self.Results, self.Train_results
        elif not return_train_results and return_train_loss:
            return self.Results, self.Train_loss
        else:
            return self.Results
        
    #%% Draw figures
    def write_single_data_point(self, x, y, dx, dy, color, split_type, log):
        point_string = ''
        if split_type == 'Random_split':
            if log:
                ddy = dy ** 0.5
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y * ddy, x - dx, y / ddy))
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x + dx, y * ddy, x + dx, y / ddy))
            else:
                ddy = 0.5 * dy
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y + ddy, x - dx, y - ddy))
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x + dx, y + ddy, x + dx, y - ddy))
            point_string += (r'        \draw[' + color + r'] ' + 
                            '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y, x + dx, y))
        
        elif split_type in ['Location_split', 'Cross_split', 'no_split'] :
            if log:
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x, y * dy, x, y / dy))
            else:
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x, y + dy, x, y - dy))
            point_string += (r'        \draw[' + color + r'] ' + 
                            '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y, x + dx, y))
            
        elif split_type == 'Critical_split':
            if log:
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y * dy, x + dx, y / dy))
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y / dy, x + dx, y * dy))
                
            else:
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y + dy, x + dx, y - dy))
                point_string += (r'        \draw[' + color + r'] ' + 
                                '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x - dx, y - dy, x + dx, y + dy))
        
        elif split_type[:14] == 'Location_split':
            step, steps = np.array(split_type[15:].split('_')).astype(int)
            angle = 2 * np.pi *  step / steps
            c, s = np.cos(angle), np.sin(angle)
            Rot_matrix = np.array([[c,s],
                                   [-s,c]])
            values = np.array([[0,  1, - 1],
                               [1.67, 0, 0]])
            
            new_values = np.dot(Rot_matrix, values)
            
            if log:
                for value in new_values.T:
                    Dx, Dy = value
                    ddx = dx * Dx
                    ddy = dy ** Dy
                    point_string += (r'        \draw[' + color + r'] ' + 
                                    '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x, y, x + ddx, y * ddy))
                
            else:
                for value in new_values.T:
                    Dx, Dy = value
                    ddx = dx * Dx
                    ddy = dy * Dy
                    point_string += (r'        \draw[' + color + r'] ' + 
                                    '({:0.3f}, {:0.5f}) -- ({:0.3f}, {:0.5f}); \n'.format(x, y, x + ddx, y + ddy))
                
        else:
            raise TypeError("This type of splitting type has no visualization marker")
        return point_string


    def write_data_point_into_plot(self, x, dx, values, color, split_type, include_only_mean, dy = None, log = False):
        data_string = ' \n'
        
        colort  = color + r', thin'
        colorvt = color + r'!50!white, very thin'
        
        if len(values) > 1 and not include_only_mean:
            for i, y in enumerate(values):
                vdx = 0.5 * dx
                if log:
                    vdy = dy ** 0.5 
                else:
                    vdy = 0.5 * dy
                xv = x + 2 * dx * i / (len(values) - 1) - dx
                if split_type[:14] == 'Location_split':
                    split_type_i = split_type[:14] + '_{}_{}'.format(i, len(values))
                else:
                    split_type_i = split_type
                if dy is None:
                    data_string += self.write_single_data_point(xv, y, vdx, vdx, colorvt, split_type_i, log)
                else:
                    data_string += self.write_single_data_point(xv, y, vdx, vdy, colorvt, split_type_i, log)
                    
        if dy is None:        
            data_string += self.write_single_data_point(x, np.nanmean(values), dx, dx, colort, split_type, log)
        else:
            data_string += self.write_single_data_point(x, np.nanmean(values), dx, dy, colort, split_type, log)
        
        return data_string

    
    def draw_figure(self, include_only_mean = False, produce_single = False, plot_height = 2, plot_width = None, plot_x_labels = True):
        assert self.results_loaded, "No results are loaded yet. Use self.load_results()."
        
        T0_names = {'start':     r'Earliest',
                    'col_equal': r'Fixed-time (equal)',
                    'col_set':   r'Fixed-time',
                    'crit':      r'Last useful',
                    'mixed':     r'Mixed'}
        
        if include_only_mean:
            addon = '_(mean).tex'
        else:
            addon = '.tex'
            
        # Get maximum figure width in cm
        num_para_values = self.num_models * self.num_data_sets * self.num_data_params
        
        # Define empty spaces
        outer_space = 1.5
            
        inter_plot_space = 0.3 
        
        plot_height = plot_height
        if plot_width is not None:
            allowed_width = plot_width * self.num_data_sets + outer_space + inter_plot_space * (self.num_data_sets - 1)
        
        else:
            ## Based on IEEE transactions template
            if 60 < num_para_values:
                allowed_width = 22 # use textheight
            elif 30 < num_para_values <= 60:
                allowed_width = 18.13275 # use textwidth
            else:
                allowed_width = 8.85553  # use 
                allowed_width = 3 
            plot_width  = (allowed_width - outer_space - inter_plot_space * (self.num_data_sets - 1)) / self.num_data_sets
        
        overall_height = (outer_space + self.num_metrics * plot_height +
                          inter_plot_space * ((self.num_metrics - 1)))
        
        
        Figure_string  = r'\documentclass[journal]{standalone}' + ' \n' + ' \n'
        
        Figure_string += r'\input{header}' + ' \n'
        
        Figure_string += r'\begin{document}' + ' \n'
        
        if 60 < num_para_values:
            Figure_string += r'\begin{tikzpicture}[rotate = -90,transform shape]' + ' \n'
        else:
            Figure_string += r'\begin{tikzpicture}' + ' \n'
        
           
        # Define colors
        rgb_low  = np.array([94,60,153], int)
        rgb_high = np.array([253,184,99], int)
        
        Colors = ['lowcolor'] 
        Colors_string = ''
        Colors_string += r'    \definecolor{lowcolor}{RGB}{' + '{},{},{}'.format(*rgb_low) + r'}' + ' \n'
        
        for i in range(self.num_data_params - 2):
            new_name = 'midcolor_{}'.format(i + 1)
            Colors.append(new_name)
            
            fac = (i + 1) / (self.num_data_params - 1)
            rgb = (fac * rgb_high + (1 - fac) * rgb_low).astype(int)
            
            Colors_string += r'    \definecolor{' + new_name + r'}{RGB}{' + '{},{},{}'.format(*rgb) + r'}' + ' \n'
            
        Colors.append('highcolor')
        Colors_string += r'    \definecolor{highcolor}{RGB}{' + '{},{},{}'.format(*rgb_high) + r'}' + ' \n'
        
        Figure_string += Colors_string
        # Start drawing the outer grid
        # Draw the left line 
        if self.num_data_sets > 1:
            Figure_string += ' \n' + r'    % Draw the outer x-axis' + ' \n'
            Figure_string += (r'    \draw[black] (1.5, 0.25) -- (1.5, 0.5) -- ' + 
                              '({:0.3f}, 0.5) -- ({:0.3f}, 0.25); \n'.format(allowed_width, allowed_width))
            Figure_string += r'    \draw[black] (1.5, 0.75) -- (1.5, 0.5);' + ' \n'
            Figure_string += (r'    \draw[black] ' + 
                              '({:0.3f}, 0.5) -- ({:0.3f}, 0.75); \n'.format(allowed_width, allowed_width))
            
            for i, data_set_dict in enumerate(self.Data_sets):
                # Get data set class
                data_set = data_interface(data_set_dict, self.parameters)
                
                label_value = outer_space + (i + 0.5) * plot_width + i * inter_plot_space
                
                if 60 < num_para_values:
                    Figure_string += (r'    \node[black, rotate = 180, above, align = center, font = \footnotesize] at ' + 
                                      '({:0.3f}, 0.5) '.format(label_value) + 
                                      r'{' + data_set.get_name()['latex'] + r'};' + ' \n') 
                else:
                    Figure_string += (r'    \node[black, below, align = center, font = \footnotesize] at ' + 
                                      '({:0.3f}, 0.5) '.format(label_value) + 
                                      r'{' + data_set.get_name()['latex'] + r'};' + ' \n') 
                
                if i < self.num_data_sets - 1:
                    x_value = outer_space + (i + 1) * (plot_width + inter_plot_space) - 0.5 * inter_plot_space
                    Figure_string += (r'    \draw[black] ' + 
                                      '({:0.3f}, 0.5) -- ({:0.3f}, 0.25); \n'.format(x_value, x_value))
                    Figure_string += (r'    \draw[black, dashed] ' + 
                                      '({:0.3f}, 0.5) -- ({:0.3f}, {:0.3f}); \n'.format(x_value, x_value, overall_height))
        
        
        # start drawing the individual plots
        for j, metric_dict in enumerate(self.Metrics):
            metric_name = metric_dict['metric']
            metric_kwargs = metric_dict['kwargs']
            metric_module = importlib.import_module(metric_name)
            metric_class = getattr(metric_module, metric_name)
            metric = metric_class(metric_kwargs, None, None, None)
            
            Figure_string += ' \n' + r'    % Draw the metric ' + metric.get_name()['print'] + ' \n'
            
            Metric_results = self.Results[..., j]
            metric_is_log  = self.Metrics_log_scale[j]
            if metric_is_log:
                min_value = 10 ** np.floor(np.log10(np.nanmin(Metric_results)) + 1e-5)
                max_value = 10 ** np.ceil(np.log10(np.nanmax(Metric_results)) - 1e-5)
            else:
                min_value = np.floor(np.nanmin(Metric_results) * 10 + 1e-5) / 10 
                max_value = np.ceil(np.nanmax(Metric_results) * 10 - 1e-5) / 10
            
            y_value  = overall_height - plot_height * (j + 1) - inter_plot_space * (j)
            
            y_max = self.num_models * plot_height / plot_width
            n_y_tick = 4 # Redo
            
            for k, data_set_dict in enumerate(self.Data_sets):
                # Get data set class
                data_set = data_interface(data_set_dict, self.parameters)
                
                x_value = outer_space + k * (plot_width + inter_plot_space)
                
                Plot_string = ''
                
                if data_set.t0_type[:3] == 'all':
                    t0_name = 'A' + data_set.t0_type[1:] 
                else:
                    t0_name = T0_names[data_set.t0_type]

                Plot_string += (' \n' + r'    % Draw the metric ' + metric.get_name()['print'] + 
                                ' for ' + t0_name + ' on dataset ' + data_set.get_name()['print'] +  ' \n')
                
                
                Results_jk = Metric_results[k]
                
                if metric_is_log:
                    Plot_string += r'    \begin{semilogyaxis}[' + ' \n'
                else:
                    Plot_string += r'    \begin{axis}[' + ' \n'
                Plot_string += r'        at = {(' + '{:0.3f}cm, {:0.3f}cm'.format(x_value, y_value) + r')},' + ' \n'
                Plot_string += r'        width = ' + '{:0.3f}'.format(plot_width) + r'cm,' + ' \n'
                Plot_string += r'        height = ' + '{:0.3f}'.format(plot_height) + r'cm,' + ' \n'
                Plot_string += r'        scale only axis = true,' + ' \n'
                Plot_string += r'        axis lines = left,' + ' \n'
                Plot_string += r'        xmin = 0,' + ' \n'
                Plot_string += r'        xmax = ' + str(self.num_models) + r',' + ' \n'
                Plot_string += r'        xtick = {' + str([*range(1, self.num_models + 1)])[1:-1] + r'},' + ' \n'
                Plot_string += r'        xticklabels = {' 
                if (1 + j) == self.num_metrics: 
                    for m, model_dict in enumerate(self.Models):
                        # get model subjects
                        model_name   = model_dict['model']
                        model_kwargs = model_dict['kwargs']
                        
                        # Get model instance
                        model_module = importlib.import_module(model_name)
                        model_class = getattr(model_module, model_name) 
                        model = model_class(model_kwargs, None, None, self.evaluate_on_train_set)
                        
                        Plot_string += model.get_name()['latex']
                        if m < self.num_models - 1:
                            Plot_string += ', '
                Plot_string += r'},' + ' \n' 
                Plot_string += (r'        x tick label style = {rotate=90, yshift = ' +
                                '{:0.3f}'.format(0.5 * plot_width / self.num_models) + 
                                r'cm, xshift = 0.1cm, align = right, font=\tiny},' + ' \n')
                Plot_string += r'        xmajorgrids = true,' + ' \n'
                Plot_string += r'        ymin = ' + '{:0.3f}'.format(min_value) + r',' + ' \n'
                Plot_string += r'        ymax = ' + '{:0.3f}'.format(max_value) + r',' + ' \n'
                if not metric_is_log:
                    Plot_string += (r'        ytick = {' + 
                                    '{:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}'.format(*np.linspace(min_value, max_value, n_y_tick + 1)) +
                                    r'},' + ' \n')
                    Plot_string += r'        scaled y ticks = base 10:0,' + ' \n'
                Plot_string += r'        yticklabels = {\empty},' + ' \n'
                Plot_string += r'        ymajorgrids = true,' + ' \n'
                Plot_string += r'    ]' + ' \n'
                
                # Add values
                dx = min(y_max / 2.5, 1 / (1.75 * self.num_data_params)) * 0.5
                if metric_is_log:
                    dy = 10 ** (dx * np.log10(max_value / min_value) / y_max)
                else:
                    dy = dx * (max_value - min_value) / y_max
                for m, model_dict in enumerate(self.Models):
                    # get model subjects
                    model_name   = model_dict['model']
                    model_kwargs = model_dict['kwargs']
                    
                    for n, data_params in enumerate(self.Data_params):
                        x_pos = m + (n + 1) / (self.num_data_params + 1)
                        results = Results_jk[n,:,m]
                            
                        for s, split_type in enumerate(self.Split_types):
                            split_index = self.Split_indices[s]
                            results_split = results[split_index]
                            good_results = np.isfinite(results_split)
                            if good_results.any():
                                Plot_string += self.write_data_point_into_plot(x_pos, dx, results_split[good_results], 
                                                                               Colors[n], split_type, include_only_mean, 
                                                                               dy, metric_is_log)
                        
                
                if metric_is_log:
                    Plot_string += r'    \end{semilogyaxis}' + ' \n'
                else:
                    Plot_string += r'    \end{axis}' + ' \n'
                
                Figure_string += Plot_string
                # Add y-axis labels if needed
                if k == 0:
                    Plot_string += ' \n' + r'    % Draw the inner y-axis' + ' \n' 
                    
                    if metric_is_log:
                        # Add lower bound
                        Figure_string += (r'    \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                          '(1.25, {:0.3f}) '.format(y_value) + 
                                          r'{$10^{' + '{}'.format(int(np.log10(min_value))) + r'}$};' + ' \n') 
                        # Add upper bound
                        Figure_string += (r'    \node[black, rotate = 90, inner sep = 0, left, font = \footnotesize] at ' + 
                                          '(1.25, {:0.3f}) '.format(y_value + plot_height) + 
                                          r'{$10^{' + '{}'.format(int(np.log10(max_value))) + r'}$};' + ' \n') 
                        
                    else:
                        # Add lower bound
                        Figure_string += (r'    \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                          '(1.25, {:0.3f}) '.format(y_value) + 
                                          r'{$' + '{}'.format(min_value) + r'$};' + ' \n') 
                        # Add upper bound
                        Figure_string += (r'    \node[black, rotate = 90, inner sep = 0, left, font = \footnotesize] at ' + 
                                          '(1.25, {:0.3f}) '.format(y_value + plot_height) + 
                                          r'{$' + '{}'.format(max_value) + r'$};' + ' \n') 
                    # Add metric name
                    metric_name_latex = metric.get_name()['latex']
                    
                    if metric_class.get_opt_goal() == 'minimize':
                        metric_name_latex += r' $\downarrow'
                    else:
                        metric_name_latex += r' $\uparrow'
                    
                    metric_bounds = metric_class.metric_boundaries()
                    if metric_bounds[0] is not None:
                        metric_name_latex += r'_{' + str(metric_bounds[0]) + r'}'
                    else:
                        metric_name_latex += r'_{\hphantom{0}}'
                    
                    if metric_bounds[1] is not None:
                        metric_name_latex += r'^{' + str(metric_bounds[1]) + r'}$'
                    else:
                        metric_name_latex += r'^{\hphantom{0}}$'
                        
                    
                    
                    Figure_string += (r'    \node[black, rotate = 90, font = \footnotesize] at ' + 
                                      '(0.9, {:0.3f}) '.format(y_value + 0.5 * plot_height) + 
                                      r'{' + metric_name_latex + r'};' + ' \n')
                    
                
                
                if produce_single:
                    Plot_string += ' \n' + r'    % Draw the inner y-axis' + ' \n' 
                    
                    if metric_is_log:
                        # Add lower bound
                        Plot_string += (r'    \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                        '(-0.25, 0.0) ' + 
                                        r'{$10^{' + '{}'.format(int(np.log10(min_value))) + r'}$};' + ' \n') 
                        # Add upper bound
                        Plot_string += (r'    \node[black, rotate = 90, inner sep = 0, left, font = \footnotesize] at ' + 
                                        '(-0.25, {:0.3f}) '.format(plot_height) + 
                                        r'{$10^{' + '{}'.format(int(np.log10(max_value))) + r'}$};' + ' \n') 
                        
                    else:
                        # Add lower bound
                        Plot_string += (r'    \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                        '(-0.25, 0.0) ' + 
                                        r'{$' + '{}'.format(min_value) + r'$};' + ' \n') 
                        # Add upper bound
                        Plot_string += (r'    \node[black, rotate = 90, inner sep = 0, left, font = \footnotesize] at ' + 
                                        '(-0.25, {:0.3f}) '.format(plot_height) + 
                                        r'{$' + '{}'.format(max_value) + r'$};' + ' \n') 
                    
                    Plot_string = Colors_string + Plot_string
                    
                    Plot_string = Plot_string.replace('at = {(' + '{:0.3f}cm, {:0.3f}cm'.format(x_value, y_value) + r')}',
                                                      'at = {(0.0, 0.0)}')
                    
                    if plot_x_labels and (1 + j) != self.num_metrics:
                        Label_string = r'xticklabels = {'  
                        for m, model_dict in enumerate(self.Models):
                            # get model subjects
                            model_name   = model_dict['model']
                            model_kwargs = model_dict['kwargs']
                            
                            # Get model instance
                            model_module = importlib.import_module(model_name)
                            model_class = getattr(model_module, model_name) 
                            model = model_class(model_kwargs, None, None, self.evaluate_on_train_set)
                            Label_string += model.get_name()['latex']
                            if m < self.num_models - 1:
                                Label_string += ', '
                        Label_string += r'}' 
                        
                        
                        Plot_string = Plot_string.replace(r'xticklabels = {}', Label_string)
                        
                    if not plot_x_labels and  (1 + j) == self.num_metrics:
                        Label_string = r'xticklabels = {' 
                        for m, model_dict in enumerate(self.Models):
                            # get model subjects
                            model_name   = model_dict['model']
                            model_kwargs = model_dict['kwargs']
                            
                            # Get model instance
                            model_module = importlib.import_module(model_name)
                            model_class = getattr(model_module, model_name) 
                            model = model_class(model_kwargs, None, None, self.evaluate_on_train_set)
                            Label_string += model.get_name()['latex']
                            if m < self.num_models - 1:
                                Label_string += ', '
                        Label_string += r'}' 
                        
                        Plot_string = Plot_string.replace(Label_string, r'xticklabels = {}')
                        
                    
                    plot_file_name = (self.path + os.sep + 'Latex_files' + os.sep + 'Figure_' + 
                                      metric_name + '_' + data_set.t0_type + '_' + data_set.get_name()['file'] + '_' + 
                                      self.Experiment_name + addon)
                    # Print latex file
                    Plot_lines = Plot_string.split('\n')
                    os.makedirs(os.path.dirname(plot_file_name), exist_ok = True)
                    pf = open(plot_file_name, 'w+')
                    for line in Plot_lines:
                        pf.write(line + '\n')
                    pf.close()
                    
                    
            
        # Add the legend for n_I
        Figure_string += ' \n' + r'    % Draw the legend for data params' + ' \n'
        
        legend_x_offset = 1.5
        
        if self.num_data_sets > 1:
            legend_y_offset = 0.0
        else:
            legend_y_offset = 0.5
        
        
        if 60 < num_para_values:
            legend_width = overall_height - legend_x_offset
        else:
            legend_width = allowed_width - legend_x_offset
        
        if self.num_data_params > 1 and len(self.Split_types) > 1:
            legend_height = -1.1
            y0_nI = - 0.3 
            y0_st = - 0.8
        elif self.num_data_params > 1 and len(self.Split_types) == 1:
            legend_height = -0.6
            y0_nI = - 0.3 
        elif self.num_data_params == 1 and len(self.Split_types) > 1:
            legend_height = -0.6
            y0_st = - 0.3
        else:
            legend_height = 0.0
            
        if legend_height < 0:
        
            Figure_string += r'    \filldraw[draw=black,fill = black!25!white, fill opacity = 0.2]'
            if 60 < num_para_values:
                Figure_string += '({:0.3f}, {:0.3f}) rectangle ({:0.3f}, {:0.3f});'.format(allowed_width + inter_plot_space - legend_y_offset,
                                                                                           legend_x_offset,
                                                                                           allowed_width + inter_plot_space + 
                                                                                           legend_height - legend_y_offset, 
                                                                                           overall_height) + ' \n'
            else:
                Figure_string += '({:0.3f}, {:0.3f}) rectangle ({:0.3f}, {:0.3f});'.format(legend_x_offset, legend_y_offset, 
                                                                                           allowed_width, legend_height + legend_y_offset) + ' \n'
            
        if self.num_data_params > 1:   
            for i, data_params in enumerate(self.Data_params):
                dx = 0.1
                legend_entry_width = legend_width / (self.num_data_params)
                x0_nI = legend_x_offset + i * legend_entry_width
                if 60 < num_para_values:
                    Figure_string += self.write_single_data_point(allowed_width + inter_plot_space - y0_nI - legend_y_offset, 
                                                                  x0_nI + 0.3, dx, dx, Colors[i], 'Critical_split', False)
                    Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                      '({:0.3f}, {:0.3f}) '.format(allowed_width + inter_plot_space - y0_nI - legend_y_offset, 
                                                                   x0_nI + 0.6) + r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$};' + ' \n')
                    
                else:
                    # Random split
                    Figure_string += self.write_single_data_point(x0_nI + 0.3, y0_nI + legend_y_offset, dx, dx, Colors[i], 'Critical_split', False)
                    Figure_string += (r'        \node[black, inner sep = 0, right, font = \footnotesize] at ' + 
                                      '({:0.3f}, {:0.3f}) '.format(x0_nI + 0.6, y0_nI + legend_y_offset) + 
                                      r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$};' + ' \n')
        
        if len(self.Split_types) > 1:   
            for i, split_type in enumerate(self.Split_types):
                split_name = ' '.join(split_type.split('_'))
                dx = 0.1
                legend_entry_width = legend_width / (len(self.Split_types))
                x0_st = legend_x_offset + i * legend_entry_width
                
                if 60 < num_para_values:
                    Figure_string += self.write_single_data_point(allowed_width + inter_plot_space - y0_st - legend_y_offset, 
                                                                  x0_st + 0.3, dx, dx, 'black', split_type, False)
                    Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                      '({:0.3f}, {:0.3f}) '.format(allowed_width + inter_plot_space - y0_st - legend_y_offset, 
                                                                   x0_st + 0.6) + r'{' + split_name + r'};' + ' \n')
                    
                else:
                    # Random split
                    Figure_string += self.write_single_data_point(x0_st + 0.3, y0_st + legend_y_offset, dx, dx, 'black', split_type, False)
                    Figure_string += (r'        \node[black, inner sep = 0, right, font = \footnotesize] at ' + 
                                      '({:0.3f}, {:0.3f}) '.format(x0_st + 0.6, y0_st + legend_y_offset) + 
                                      r'{' + split_name + r'};' + ' \n')
        
        
        Figure_string += r'\end{tikzpicture}' + ' \n'
        Figure_string += r'\end{document}'
        
        figure_file_name = (self.path + os.sep + 'Latex_files' + os.sep + 'Figure_' + self.Experiment_name + addon)
        # Print latex file
        Figure_lines = Figure_string.split('\n')
        os.makedirs(os.path.dirname(figure_file_name), exist_ok = True)
        f = open(figure_file_name, 'w+')
        for line in Figure_lines:
            f.write(line + '\n')
        f.close() 
        
    #%% Write tables
    def write_tables(self, dataset_row = True, use_scriptsize = False, depict_std = True):
        assert self.results_loaded, "No results are loaded yet. Use self.load_results()."
        
        if dataset_row:
            Table_iterator = self.Metrics
            Row_iterator   = self.Data_sets
            row_name       = 'Datasets'
            
        else:
            Table_iterator = self.Data_sets
            Row_iterator   = self.Metrics
            row_name       = 'Metrics'
            
        for k, table_name in enumerate(Table_iterator):
            if dataset_row:
                table_name_name = table_name['metric']
                table_name_kwargs = table_name['kwargs']
                table_module = importlib.import_module(table_name_name)
                table_class = getattr(table_module, table_name_name)
                table_object = table_class(table_name_kwargs, None, None, None)

                table_filename = table_object.get_name()['file']
            else:
                table_item = data_interface(table_name, self.parameters)
                table_filename = table_item.get_name()['file']
                 
            
            nP = self.num_data_params
            num_data_columns = self.num_data_params * self.num_models
            if 9 < num_data_columns:
                width = r'\textheight'
                num_tables = int(np.ceil(self.num_models / np.floor(12 / self.num_data_params)))
            elif 4 < num_data_columns <= 9:
                width = r'\textwidth'
                num_tables = 1
            else:
                width = r'\linewidth'
                num_tables = 1
                
            models_per_table = int(np.ceil(self.num_models / num_tables))
            # Allow for split table
            Output_strings = [r'\begin{tabularx}{' + width + r'}'] * num_tables
            for n in range(num_tables):
                models_n = np.arange(n * models_per_table, min(self.num_models, (n + 1) * models_per_table))
                Output_strings[n] += r'{X' + models_per_table * (r' | ' + r'Z' * nP) + r'} '
                Output_strings[n] += '\n'
                if n > 0:
                    Output_strings[n] += r'\multicolumn{' + str(models_per_table * nP + 1) + r'}{c}{} \\' + ' \n'
                Output_strings[n] += r'\toprule[1pt] '
                Output_strings[n] += '\n'
                if n == 0:
                    Output_strings[n] += r'\multirow{{2}}{{*}}{{\textbf{{Dataset}}}} '.format()
                    Output_strings[n] += r'& \multicolumn{{{}}}{{c}}{{\textbf{{Models}}}} \\'.format(models_per_table * nP) 
                    Output_strings[n] += '\n'
                for model_idx in models_n:
                    model_dict = self.Models[model_idx]
                    
                    # get model subjects
                    model_name   = model_dict['model']
                    model_kwargs = model_dict['kwargs']
                        
                    # Get model instance
                    model_module = importlib.import_module(model_name)
                    model_class = getattr(model_module, model_name) 
                    model = model_class(model_kwargs, None, None, self.evaluate_on_train_set)
                    Output_strings[n] += r'& \multicolumn{{{}}}'.format(nP) + r'{c}{' + model.get_name()['latex'] + r'}'
                Output_strings[n] += r' \\'
                Output_strings[n] += '\n'
                Output_strings[n] += r'\midrule[1pt] '
                Output_strings[n] += '\n'
                
                
                # Get maximum number of pre decimal points
                if dataset_row:
                    Table_results = self.Results[...,k]
                else:
                    Table_results = self.Results[k].transpose(3,0,1,2)
                
                table_shape = list(Table_results.shape)
                table_shape[2] = len(self.Split_types)
                Table_results_mean = np.ones(table_shape)
                
                for i_split, indices in enumerate(self.Split_indices):
                    Table_results_mean[:,:,i_split] = np.nanmean(Table_results[:,:,indices], axis = 2)
                min_value = '{:0.3f}'.format(np.nanmin(Table_results_mean))
                max_value = '{:0.3f}'.format(np.nanmax(Table_results_mean))
                extra_str_length = max(len(min_value), len(max_value)) - 4
                
                
                for i, row_name in enumerate(Row_iterator): 
                    if dataset_row:
                        row_item = data_interface(row_name, self.parameters)
                        row_latexname = row_item.get_name()['latex']
                    else:
                        row_name_name = row_name['metric']
                        row_name_kwargs = row_name['kwargs']
                        row_module = importlib.import_module(row_name_name)
                        row_class  = getattr(row_module, row_name_name)
                        row_object = row_class(row_name_kwargs, None, None, None)
                        
                        row_latexname = row_object.get_name()['latex']
                        if row_object.get_opt_goal() == 'minimize':
                            row_latexname += r' $\downarrow'
                        else:
                            row_latexname += r' $\uparrow'
                            
                        metric_bounds = row_object.metric_boundaries()
                        if metric_bounds[0] is not None:
                            row_latexname += r'_{' + str(metric_bounds[0]) + r'}'
                        else:
                            row_latexname += r'_{\hphantom{0}}'
                        
                        if metric_bounds[1] is not None:
                            row_latexname += r'^{' + str(metric_bounds[1]) + r'}$'
                        else:
                            row_latexname += r'^{\hphantom{0}}$'
                    
                    if dataset_row:
                        metric_class  = table_class
                        metric_index  = k
                        dataset_index = i
                    else:
                        metric_class  = row_class
                        metric_index  = i
                        dataset_index = k
                    
                    if use_scriptsize:
                        t_single = (r'& {{\scriptsize ${:0.3f}$}} ' * len(models_n) * nP + 
                                    '& ' * (models_per_table - len(models_n)) * nP + r'\\')
                        t_multi  = (r'& {{\scriptsize ${:0.3f}^{{\pm {:0.3f}}}$}} ' * len(models_n) * nP + 
                                    '& ' * (models_per_table - len(models_n)) * nP + r'\\')
                    else:
                        t_single = (r'& {{${:0.3f}$}} ' * len(models_n) * nP + 
                                    '& ' * (models_per_table - len(models_n)) * nP + r'\\')
                        t_multi  = (r'& {{${:0.3f}^{{\pm {:0.3f}}}$}} ' * len(models_n) * nP + 
                                    '& ' * (models_per_table - len(models_n)) * nP + r'\\')
                    
                    multi = any([len(s) > 1 for s in self.Split_indices]) and depict_std
                    
                    Strs = []
                    Results_split = []
                    for l, split_type in enumerate(self.Split_types):
                        split_index = self.Split_indices[l]
                        
                        results_split = self.Results[dataset_index][:, split_index][:, : , :, metric_index]
                    
                        results_split_mean = np.nanmean(results_split, axis = 1)
                        results_split_std  = np.nanstd(results_split, axis = 1)
                        
                        mean_in = results_split_mean[:,models_n].T.reshape(-1)
                        std_in  = results_split_std[:,models_n].T.reshape(-1)
                        std_in[np.isnan(std_in)] = 0.0
                        
                        if multi:
                            Str = t_multi.format(*np.array([mean_in, std_in]).T.reshape(-1))
                            Str = Str.replace("\\pm 0.000", r"\hphantom{\pm 0.000}")
                        else:
                            Str = t_single.format(*mean_in)
                        
                        # Adapt length to align decimal points
                        Str_parts = Str.split('$} ')
                        for idx, string in enumerate(Str_parts):
                            if len(string) == 0:
                                continue
                            previous_string = string.split('.')[0].split('$')[-1]
                            overwrite_string = False
                            if previous_string[0] == '-':
                                overwrite_string = previous_string[1:].isnumeric()
                            else:
                                overwrite_string = previous_string.isnumeric()
                            if overwrite_string:
                                needed_buffer = extra_str_length - len(previous_string)  
                                if needed_buffer > 0:
                                    if use_scriptsize:
                                        start_index = 16
                                    else:
                                        start_index = 4
                                        
                                    
                                    Str_parts[idx] = string[:start_index] + r'\hphantom{' + '0' * needed_buffer + r'}' + string[start_index:]
                            
                            # Check for too long stds
                            if multi and not Str_parts[idx] == r'\\':
                                string_parts = Str_parts[idx].split('^')
                                if 'hphantom' not in string_parts[1]:
                                    std_number = string_parts[1][5:10]
                                    if std_number[-1] == '.':
                                        std_number = std_number[:-1] + r'\hphantom{0}'
                                    string_parts[1] = r'{\pm ' + std_number + r'}' 
                            
                                Str_parts[idx] = '^'.join(string_parts)
                                
                        Str = '$} '.join(Str_parts)
                        
                        # Underline best value
                        if metric_class.get_opt_goal() == 'minimize':
                            best_value_random = np.nanmin(results_split_mean, axis = -1)
                            adapt_string = True
                        elif metric_class.get_opt_goal() == 'maximize':
                            best_value_random = np.nanmax(results_split_mean, axis = -1)
                            adapt_string = True
                        else:
                            adapt_string = False
                    
                    
                        if adapt_string:
                            Str_parts = Str.split('$} ')
                            for idx, bv in enumerate(best_value_random):
                                useful = np.where(bv == results_split_mean[idx, models_n])[0] * nP + idx
                                for useful_idx in useful:
                                    start_string, rest_string = Str_parts[useful_idx].split('$')
                                    string_parts = rest_string.split('^')
                                    underlinable_parts = string_parts[0].split('}')
                                    underlinable_parts[-1] = r'\underline{' + underlinable_parts[-1] + r'}'
                                    string_parts[0] = '}'.join(underlinable_parts)
                                    rest_string = '^'.join(string_parts)
                                    Str_parts[useful_idx] = start_string + r'$' + rest_string
                                    
                            Str = '$} '.join(Str_parts)
                        
                        Strs.append(Str)
                        Results_split.append(results_split)
                    
                    if not all([np.isnan(results_split).all() for results_split in Results_split]):
                        Output_strings[n] += row_latexname + ' '
                        for strs in Strs:    
                            Output_strings[n] += strs + ' \n'
                            
                        Output_strings[n] += '\midrule \n'
                
                # replace last midrule with bottom rule  
                Output_strings[n]  = Output_strings[n][:-10] + r'\bottomrule[1pt]'
                Output_strings[n] += '\n'
                Output_strings[n] += r'\end{tabularx}' + ' \n' 
                
            Output_string = '\n'.join(Output_strings)
            # split string into lines
            Output_lines = Output_string.split('\n')
            
            if dataset_row:
                table_file_name = (self.path + '/Latex_files/Table_' + self.Experiment_name  + '_' +
                                   table_filename + '.tex')
            else:
                table_file_name = (self.path + '/Latex_files/Table_' + self.Experiment_name  + '_' +
                                   table_filename + '_at_' + table_item.t0_type + '.tex')
            
            os.makedirs(os.path.dirname(table_file_name), exist_ok = True)
            t = open(table_file_name, 'w+')
            for line in Output_lines:
                t.write(line + '\n')
            t.close()
                
    #%% Draw singel example
    
    def _get_data(self):
        # select dataset if needed:
        if self.num_data_sets > 1:
            print('------------------------------------------------------------------', flush = True)
            sample_string = 'In the current experiment, the following datasets are available:'
            for i, data_set_dict in enumerate(self.Data_sets):
                # Get data set class
                data_set = data_interface(data_set_dict, self.parameters)
                sample_string += '\n{}: '.format(i + 1) + data_set.get_name()['print'] + ' - with predictions at ' + data_set.t0_type + 'times.' 
            print(sample_string, flush = True)  
            print('Select the desired dataset by typing a number between 1 and {} for the specific dataset): '.format(self.num_data_sets), flush = True)
            print('', flush = True)
            try:
                i_d = int(input('Enter a number: ')) - 1
            except:
                i_d = -1
            while i_d not in range(self.num_data_sets):
                print('This answer was not accepted. Please repeat: ', flush = True)
                print('', flush = True)
                try:
                    i_d = int(input('Enter a number: ')) - 1
                except:
                    i_d = -1 
            
            data_set_dict = self.Data_sets[i_d]
        else:
            data_set_dict = self.Data_sets[0]
        
        # Prevent model retraining
        parameters = [param for param in self.parameters]
        parameters[7] = 'no'
        
        data_set = data_interface(data_set_dict, parameters)
        
        if self.num_data_params > 1:
            print('------------------------------------------------------------------', flush = True)
            sample_string = 'In the current experiment, the following data extraction parameters are available:'
            for i, d_para in enumerate(self.Data_params):
                sample_string += '\n{}: '.format(i + 1) + str(d_para).replace(', ', '\n    ') + '\n'
            print(sample_string, flush = True)  
            print('Select the desired dataset by typing a number between 1 and {} for the specific parameters): '.format(self.num_data_params), flush = True)
            print('', flush = True)
            try:
                i_d = int(input('Enter a number: ')) - 1
            except:
                i_d = -1
            while i_d not in range(self.num_data_params):
                print('This answer was not accepted. Please repeat: ', flush = True)
                print('', flush = True)
                try:
                    i_d = int(input('Enter a number: ')) - 1
                except:
                    i_d = -1 
            
            data_param = self.Data_params[i_d]
        else:
            data_param = self.Data_params[0]
        
        # Load predicted data 
        data_set.get_data(**data_param)
        
        if self.num_splitters > 1:
            print('------------------------------------------------------------------', flush = True)
            sample_string = 'In the current experiment, the following splitters are available:'
            for i, s_param in enumerate(self.Splitters):
                s_name   = s_param['Type']
                s_rep    = s_param['repetition']
                s_tp     = s_param['test_part']                
                s_trainp = s_param['train_pert']
                s_testp  = s_param['test_pert']
                s_tot    = s_param['train_on_test']

                s_class = getattr(importlib.import_module(s_name), s_name)
                s_inst  = s_class(None, s_tp, s_rep, s_trainp, s_testp, s_tot)
                
                sample_string += '\n{}: '.format(i + 1) + s_inst.get_name()['print']  
            print(sample_string, flush = True)  
            print('Select the desired dataset by typing a number between 1 and {} for the specific splitter): '.format(self.num_splitters), flush = True)
            print('', flush = True)
            try:
                i_d = int(input('Enter a number: ')) - 1
            except:
                i_d = -1
            while i_d not in range(self.num_splitters):
                print('This answer was not accepted. Please repeat: ', flush = True)
                print('', flush = True)
                try:
                    i_d = int(input('Enter a number: ')) - 1
                except:
                    i_d = -1 
            
            split_param = self.Splitters[i_d]
        else:
            split_param = self.Splitters[0]
        
        split_name = split_param['Type']
        split_class = getattr(importlib.import_module(split_name), split_name)
        
        # Use splitting method to get train and test samples
        split_rep    = split_param['repetition']
        split_tp     = split_param['test_part']
        split_testp  = split_param['test_pert']
        split_trainp = split_param['train_pert']
        split_tot    = split_param['train_on_test'] 

        splitter = split_class(data_set, split_tp, split_rep, split_trainp, split_testp, split_tot)
        splitter.split_data() 
        
        # Get test index 
        if self.plot_train:
            Index = splitter.Train_index
        else:
            Index = splitter.Test_index
        
        return [data_set, data_param, splitter, Index]
    
    
    def _get_model_selection(self, data_set, splitter):
        if self.num_models > 1:
            print('------------------------------------------------------------------', flush = True)
            sample_string = 'In the current experiment, the following models are available:'
            for i, m_dict in enumerate(self.Models):
                # get model subjects
                m_name   = m_dict['model']
                m_kwargs = m_dict['kwargs']
                
                # Get model instance
                m_class = getattr(importlib.import_module(m_name), m_name)
                try:
                    sample_string += '\n{}: '.format(i + 1) + m_class.get_name()['print']  
                except:
                    mm = m_class(m_kwargs, data_set, None, self.evaluate_on_train_set)
                    sample_string += '\n{}: '.format(i + 1) + mm.get_name()['print']  
            print(sample_string, flush = True)  
            print('Select the desired dataset by typing a number between 1 and {} for the specific model): '.format(self.num_models), flush = True)
            print('', flush = True)
            try:
                i_d = int(input('Enter a number: ')) - 1
            except:
                i_d = -1
            while i_d not in range(self.num_models):
                print('This answer was not accepted. Please repeat: ', flush = True)
                print('', flush = True)
                try:
                    i_d = int(input('Enter a number: ')) - 1
                except:
                    i_d = -1 
            
            model_dict = self.Models[i_d]
        else:
            model_dict = self.Models[0]
        

        # get the model
        model_name   = model_dict['model']
        model_kwargs = model_dict['kwargs']
        
        # Get model class
        model_class = getattr(importlib.import_module(model_name), model_name)
       
        # Load specific model
        model = model_class(model_kwargs, data_set, splitter, self.evaluate_on_train_set)
        model.train()

        return model
        
    
    def _get_data_sample(self, sample_ind, data_set, model, Output_A, Domain):
        
        domain           = Domain.loc[sample_ind]
        output_A         = Output_A.loc[sample_ind]


        # get empty inputs
        input_path  = pd.DataFrame(np.empty((len(sample_ind), len(data_set.Agents)), dtype = object), columns = data_set.Agents, index = sample_ind)
        output_path = pd.DataFrame(np.empty((len(sample_ind), len(data_set.Agents)), dtype = object), columns = data_set.Agents, index = sample_ind)
        output_T_E  = np.empty(len(sample_ind), dtype = object)

        # Go through needed data
        file_indices = domain.file_index
        for file_index in np.unique(file_indices):
            use_index = file_indices == file_index

            # Load raw darta
            file = data_set.Files[file_index] + '_data.npy'
            [_, Input_path, _, Output_path, _, _, _, Output_T_E, _] = np.load(file, allow_pickle = True)

            ind = domain[use_index].Index_saved

            ind_sample = np.where(use_index)[0]
            ind_agent  = data_set.get_indices_1D(np.array(Input_path.columns), np.array(data_set.Agents))
            input_path.iloc[ind_sample, ind_agent]  = Input_path.iloc[ind]
            output_path.iloc[ind_sample, ind_agent] = Output_path.iloc[ind]
            output_T_E[use_index]  = Output_T_E[ind]
        
        # Load raw darta
        ind_p = np.array([name for name in input_path.columns 
                          if isinstance(input_path.iloc[0][name], np.ndarray)])
        op = np.stack(output_path[ind_p].to_numpy().tolist(), 0) # n_samples x n_a x n_O x 2
        # Only one example of input paths is needed
        ip = np.stack(input_path[ind_p].to_numpy().tolist(), 0)[0] # n_a x n_I x 2
        
        if data_set.includes_images():
            img = data_set.return_batch_images(domain.iloc[[0]], None, None, 
                                               None, None, grayscale = False) 
            img = img[0] / 255
        else:
            img = None

        if data_set.includes_sceneGraphs():
            if hasattr(model, 'sceneGraph_radius'):
                radius = model.sceneGraph_radius
            else:
                radius = None
            graph = data_set.return_batch_sceneGraphs(domain.iloc[[0]], ip[np.newaxis, :, -1], radius) 
            graph = graph[0]
        else:
            graph = None

        # Ensure that op is not longer than 3000 samples
        if len(op) > 3000:
            np.random.seed(0)
            np.random.shuffle(op)
            op = op[:3000]
            
        return [op, ip, ind_p, output_A, output_T_E, img, graph, domain]
    
    
    def _get_data_sample_pred(self, model, Index, ip, ind_p):

        
        output = model.predict_actual(Index)
        
        output_pred = model.transform_output(output, Index, model.get_output_type(), 'path_all_wi_pov')
        output_path_pred = output_pred[1]
        output_path_pred_probs = output_pred[2]
        
        ind_pp = np.array([name for name in output_path_pred.columns 
                           if isinstance(output_path_pred.iloc[0][name], np.ndarray)])
        
        use_input = np.in1d(ind_p, ind_pp)
        
        opp = np.stack(output_path_pred[ind_pp].to_numpy().tolist(), 0) # n_samples x n_a x n_p x n_O x 2
        opp_probs = np.stack(output_path_pred_probs[ind_pp].to_numpy().tolist(), 0).astype(float) # n_samples x n_a x (n_p + 1)
        if len(opp_probs.shape) == 2:
            assert opp_probs.shape[:2] == opp.shape[:2]
            opp_probs = opp_probs[...,np.newaxis].repeat(opp.shape[2] + 1, axis = -1)
        # Remove ground truth predictions
        opp_probs = opp_probs[:, :, :-1]

        # Combine identical samples and number path predicted into one dimension
        opp = opp.transpose(0,2,1,3,4).reshape(opp.shape[0] * opp.shape[2], opp.shape[1], opp.shape[3], opp.shape[4])
        opp_probs = opp_probs.transpose(0,2,1).reshape(opp.shape[0], opp.shape[1])

        max_v = np.nanmax(np.stack([np.nanmax(opp, axis = (0,1,2)),
                                    np.max(ip[use_input,...,:2], axis = (0,1))], axis = 0), axis = 0)
        
        min_v = np.nanmin(np.stack([np.nanmin(opp, axis = (0,1,2)),
                                    np.min(ip[use_input,...,:2], axis = (0,1))], axis = 0), axis = 0)

        max_v = np.ceil(max_v)
        min_v = np.floor(min_v)
        
        # Ensure that opp is not longer than 3000 samples
        if len(opp) > 3000:
            np.random.seed(0)
            used = np.arange(len(opp))
            np.random.shuffle(used)
            used = used[:3000]
            opp = opp[used]
            opp_probs = opp_probs[used]
        return [opp, opp_probs, ind_pp, min_v, max_v]
            
    
    def _draw_background(self, ax, data_set, img, graph, domain):
        # Load line segments of data_set 
        map_lines_solid, map_lines_dashed = data_set.provide_map_drawing(domain = domain.iloc[0])
        
        bcdict = {'red': ((0, 0.0, 0.0),
                          (1, 0.0, 0.0)),
               'green': ((0, 0.0, 0.0),
                         (1, 0.0, 0.0)),
               'blue': ((0, 0.0, 0.0),
                        (1, 0.0, 0.0))}

        cmap = LinearSegmentedColormap('custom_cmap_l', bcdict)
        
        map_solid = LineCollection(map_lines_solid, cmap=cmap, linewidths=2, linestyle = 'solid')
        map_solid_colors = list(np.ones(len(map_lines_solid)))
        map_solid.set_array(np.asarray(map_solid_colors))
        
        map_dashed = LineCollection(map_lines_dashed, cmap=cmap, linewidths=1, linestyle = 'dashed')
        map_dashed_colors = list(np.ones(len(map_lines_dashed)))
        map_dashed.set_array(np.asarray(map_dashed_colors))
        
        # Add background picture
        if data_set.includes_images():
            height, width, _ = list(np.array(img.shape) * data_set.get_Target_MeterPerPx(domain.iloc[0]))
            ax.imshow(img, extent=[-width/2, width/2, -height/2, height/2], interpolation='nearest')

        if data_set.includes_sceneGraphs():
            # Draw only centerlines 
            centerlines = graph.centerlines

            for centerline in centerlines:
                ax.plot(centerline[:,0], centerline[:,1], 'k', linewidth = 2)
        
        # Draw boundaries
        ax.add_collection(map_solid)
        ax.add_collection(map_dashed)
    
    def _select_testing_samples(self, data_set, load_all, Output_A, plot_similar_futures, Index):
        print('------------------------------------------------------------------', flush = True)
        # For plot similar futures, transform Chosen_index into list of arrays
        if plot_similar_futures:
            # Get subgroups
            data_set._group_indentical_inputs()
            
            subgroup = data_set.Subgroups[Index]

            # Assemble sets of similar futures
            Chosen_sets = []
            for s in np.unique(subgroup):
                Chosen_sets.append(Index[subgroup == s])
        else:
            if (not load_all) and (len(Output_A.columns) > 1) and (not plot_similar_futures):
                sample_string = 'In this case '
                for n_beh, beh in enumerate(Output_A.columns):
                    if beh != Output_A.columns[-1]:
                        sample_string += '{} '.format(Output_A[beh].to_numpy().sum()) + beh + ' ({})'.format(n_beh + 1)
                        if len(Output_A.columns) > 2:
                            sample_string += ', '
                        else:
                            sample_string += ' '
                    else:                            
                        sample_string += 'and {} '.format(Output_A[beh].to_numpy().sum()) + beh + ' ({})'.format(n_beh + 1)       
                sample_string += ' samples are available.'
                print(sample_string, flush = True)  
                print('Select behavior, by typing a number between 1 and {} for the specific behavior): '.format(len(Output_A.columns)), flush = True)
                print('', flush = True)
                try:
                    n_beh = int(input('Enter a number: ')) - 1
                except:
                    n_beh = -1
                while n_beh not in range(len(Output_A.columns)):
                    print('This answer was not accepted. Please repeat: ', flush = True)
                    print('', flush = True)
                    try:
                        n_beh = int(input('Enter a number: ')) - 1
                    except:
                        n_beh = -1 
                
                Chosen_sets = Index[np.where(Output_A.iloc[:, n_beh].to_numpy())[0]]
            else:
                Chosen_sets = Index[np.arange(len(Output_A))]
            
            
            Chosen_sets = list(Chosen_sets[:,np.newaxis])


        if not load_all:
            print('', flush = True)

            print('Type a number between in between 1 and {} (or a list [...] including those samples) to select the example: '.format(len(Chosen_sets)), flush = True)
            print('', flush = True)
            input_string = input('Enter a number: ')
            try:
                ind = np.array([int(input_string) - 1])
            except:
                if input_string[0] == '[' and input_string[-1] == ']':
                    try:
                        ind = np.array(input_string.strip('[]').split(','), int) - 1
                    except:
                        ind = np.array([-1])
                else:
                    ind = np.array([-1])
            ind = np.unique(ind)
            
            while ind.min() < 0 or ind.max() >= len(Chosen_sets):
                print('This answer was not accepted. Please repeat: ', flush = True)
                print('', flush = True)
                input_string = input('Enter a number: ')
                try:
                    ind = np.array([int(input_string) - 1])
                except:
                    if input_string[0] == '[' and input_string[-1] == ']':
                        try:
                            ind = np.array(input_string.strip('[]').split(','), int)
                        except:
                            ind = np.array([-1])
                    else:
                        ind = np.array([-1])
                ind = np.unique(ind)
            print('')
            
            sample_inds = []
            for ind_sample in ind:
                sample_inds.append((ind_sample, Chosen_sets[ind_sample]))
        
        else:
            sample_inds = []
            for i, set in enumerate(Chosen_sets):
                sample_inds.append((i, set))
            
        return sample_inds
    
        
    def _get_path_likelihoods(self, data_set, model, sample_ind, opp, ind_pp, joint = False):
        # presafe model parameters
        save_prediction = bool(int(model.data_set.save_predictions)) # Be sure to make copy
        num_samples_path_pred = int(float(model.num_samples_path_pred))

        # Set save predictions to false to not touch saved data
        model.data_set.save_predictions = False

        # Overwrite number of predictions
        model.num_samples_path_pred = max(1, 3000 - len(opp))
        # Run the actual prediction
        Pred_index, Output_path_pred, _ = model.predict_actual(sample_ind[[0]])

        # Concatenate old predictions with new ones
        for i, agent in enumerate(ind_pp):
            assert isinstance(Output_path_pred[agent].iloc[0], np.ndarray)
            if len(opp) < 3000:
                Output_path_pred[agent].iloc[0] = np.concatenate((opp[:,i], Output_path_pred[agent].iloc[0]), axis = 0)
            else:
                Output_path_pred[agent].iloc[0] = opp[:,i]
        model.num_samples_path_pred = 3000

        # Get indpendent likelihoods
        # Prevent damage to save files
        data_set.save_predictions  = False
        model.prediction_overwrite = True

        if joint:
            model._get_joint_KDE_pred_probabilities(Pred_index, Output_path_pred, get_for_pred_agents = True)
            Lp = model.Log_prob_joint_pred[0, :opp.shape[0], np.newaxis] # num_preds x 1
            Lp = np.repeat(Lp, opp.shape[1], axis = 1) # num_preds x num_agents
        else:
            model._get_indep_KDE_pred_probabilities(Pred_index, Output_path_pred, get_for_pred_agents = True)
            # Only consider the likelihoods of trajectories in opp
            Lp = model.Log_prob_indep_pred[0, :opp.shape[0], :opp.shape[1]] # num_preds x num_agents

        # Reset model parameters
        model.data_set.save_predictions = save_prediction
        model.num_samples_path_pred     = num_samples_path_pred

        return Lp

    
    def plot_paths(self, load_all = False, 
                   plot_similar_futures = False, 
                   plot_train = False,
                   only_show_pred_agents = False,
                   likelihood_visualization = False,
                   joint_likelihoods = False,
                   plot_only_lines = False):
        assert self.provided_modules, "No modules have been provided. Run self.set_modules() first."
        assert self.provided_setting, "No parameters have been provided. Run self.set_parameters() first."
        
        self.plot_train = plot_train
        if self.plot_train:
            assert self.evaluate_on_train_set, "Training samples need to be predicted if they are to be plotted."
        
        plt.close('all')
        time.sleep(0.05)
        
        # Get dataset and splitter
        [data_set, data_param, splitter, Index] = self._get_data()

        # Get model
        model = self._get_model_selection(data_set, splitter)

        # Get possible domains
        Domain = data_set.Domain.loc[Index]

        # Get Output_A
        file_indices = np.unique(Domain.file_index)
        Output_A = pd.DataFrame(np.zeros((len(Index), len(data_set.Behaviors)), int), columns = data_set.Behaviors, index = Index)

        for file_index in file_indices:
            Index_file = np.where(Domain.file_index == file_index)[0]
            file = data_set.Files[file_index] + '_data.npy'
            Output_A_file = np.load(file, allow_pickle = True)[6]
            ind = Domain.Index_saved.iloc[Index_file]

            Output_A.iloc[Index_file] = Output_A_file.reindex(columns = data_set.Behaviors).iloc[ind].to_numpy()

        sample_inds = self._select_testing_samples(data_set, load_all, Output_A, plot_similar_futures, Index)

        
        ## Get specific case
        for sample_name, sample_ind in sample_inds: 

            [op, ip, ind_p,  
             output_A, output_T_E, img, graph, domain] = self._get_data_sample(sample_ind, data_set, model, Output_A, Domain)
            
                                                                        
            [opp, Lp, ind_pp, min_v, max_v] = self._get_data_sample_pred(model, sample_ind, ip, ind_p)
            


            if only_show_pred_agents:
                useful_p = np.in1d(ind_p, ind_pp)

                ind_p = ind_p[useful_p]
                assert np.in1d(ind_pp, ind_p).all()

                ip = ip[useful_p]
                op = op[:,useful_p]

            # Get likelihoods of the given samples, based on 3000 predictions
            if likelihood_visualization:
                # Only available for path prediction models
                if not np.isfinite(Lp).all():
                    if model.get_output_type() == 'path_all_wi_pov':
                        Lp = self._get_path_likelihoods(data_set, model, sample_ind, opp, ind_pp, joint = joint_likelihoods)
            


            # plot figure
            fig, ax = plt.subplots(figsize = (10,8))            
            
            # plot map
            self._draw_background(ax, data_set, img, graph, domain)
            
            # plot inputs
            colors = sns.color_palette("bright", len(ind_p))
            for i, agent in enumerate(ind_p):
                if plot_only_lines:
                    color = np.array(colors[i]) * 0.75
                    if len(color) == 4:
                        color[-1] = 1
                else:
                    color = np.array(colors[i])

                if plot_only_lines:
                    ax.plot(ip[i,:,0], ip[i,:,1], color = color, label = r'$A_{' + agent + r'}$', linewidth = 3)
                    ax.scatter(ip[i,-1:,0], ip[i,-1:,1], color = color, marker = 'o', s = 5)
                else:
                    ax.plot(ip[i,:,0], ip[i,:,1], color = color, 
                            marker = 'o', ms = 2.5, label = r'$A_{' + agent + r'}$', linewidth = 0.75)
                
                # For multiple GT (and pred agent), plot GT first
                if plot_similar_futures and agent in ind_pp:
                    linewidth_factor = (100 / len(op)) ** (1 / 2.5)
                    for j in range(len(op)):
                        if plot_only_lines:
                            ax.plot(np.concatenate((ip[i,-1:,0], op[j,i,:,0])), 
                                    np.concatenate((ip[i,-1:,1], op[j,i,:,1])),
                                    color = color, linewidth = 2 * linewidth_factor)
                        else:
                            ax.plot(np.concatenate((ip[i,-1:,0], op[j,i,:,0])), 
                                    np.concatenate((ip[i,-1:,1], op[j,i,:,1])),
                                    color = color, marker = 'x', ms = 2, 
                                    markeredgewidth = 0.5, linewidth = 0.25 * linewidth_factor)


                # plot predicted future
                if agent in ind_pp:
                    # Get prediction colors
                    color_preds = np.ones((len(opp), 4), float)
                    color_preds[:, :3] = 1 / 3 + 2 / 3 * color[np.newaxis]


                    i_agent = np.where(agent == ind_pp)[0][0]

                    # Add alpha value if needed
                    linewidth_factor = (100 / len(opp)) ** (1 / 2.5)

                    if likelihood_visualization:
                        lp = Lp[:,i_agent]
                        l = ((lp - np.min(lp)) / (np.max(lp) - np.min(lp)))
                        color_preds[:, 3] = 0.9 / (1 + np.exp(5 * (np.median(l) - l)))
                    else:
                        color_preds[:, 3] = 0.8

                    color_preds[:, 3] = color_preds[:, 3] ** (1 / linewidth_factor)

                    J_sort = np.argsort(color_preds[:, 3])


                    for j in J_sort:
                        if plot_only_lines:
                            ax.plot(np.concatenate((ip[i,-1:,0], opp[j,i_agent,:,0])), 
                                    np.concatenate((ip[i,-1:,1], opp[j,i_agent,:,1])), 
                                    color = color_preds[j], linewidth = 2 * linewidth_factor)
                        else:
                            ax.plot(np.concatenate((ip[i,-1:,0], opp[j,i_agent,:,0])), 
                                    np.concatenate((ip[i,-1:,1], opp[j,i_agent,:,1])), 
                                    color = color_preds[j], marker = 'x', ms = 2, markeredgewidth = 0.5, linewidth = 0.25 * linewidth_factor)

                # For single GT, plot GT last
                if not plot_similar_futures:
                    assert len(op) == 1, "Only one prediction is allowed for single GT."

                    if plot_only_lines:
                        ax.plot(np.concatenate((ip[i,-1:,0], op[0,i,:,0])), 
                                np.concatenate((ip[i,-1:,1], op[0,i,:,1])),
                                color = color, linestyle = 'dashed', linewidth = 3)
                    else:
                        ax.plot(np.concatenate((ip[i,-1:,0], op[0,i,:,0])), 
                                np.concatenate((ip[i,-1:,1], op[0,i,:,1])),
                                color = color, marker = 'x', ms = 2.5, 
                                markeredgewidth = 0.5, linestyle = 'dashed', linewidth = 0.75)
            
            
            # Format plot
            ax.set_aspect('equal', adjustable='box') 
            ax.set_xlim([min_v[0],max_v[0]])
            ax.set_ylim([min_v[1],max_v[1]])
            title = (r'Data set: ' + data_set.get_name()['print'] +  
                     r' with $\delta t = ' + str(data_param['dt']) + 
                     r'$, Model: ' + model.get_name()['print'])
            behs = np.array(output_A.columns)
            if len(behs) > 1 and not plot_similar_futures:
                title += r': \nTrue behavior: ' + behs[output_A.iloc[0].to_numpy().astype(bool)][0] + r' at $t = ' + str(output_T_E[0])[:5] + '$' 

            ax.set_title(title)
            plt.axis('off')
            plt.legend()
            plt.tight_layout()
            if not load_all:
                plt.show()
            
            if self.plot_train:
                fig_str = 'traj_plot_test__'
            else:
                fig_str = 'traj_plot_train_'
            
            figure_file = data_set.change_result_directory(model.model_file, 'Metric_figures', 
                                                           fig_str + '{}'.format(sample_name + 1), '.pdf')
            
            os.makedirs(os.path.dirname(figure_file), exist_ok = True)
            fig.savefig(figure_file)
            
            plt.close(fig)
                                                            
