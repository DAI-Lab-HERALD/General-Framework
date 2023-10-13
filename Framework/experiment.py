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
    
    #%% Experiment setup        
    def set_modules(self, Data_sets, Data_params, Splitters, Models, Metrics):
        assert type(Data_sets) == type([0]), "Data_sets must be a list."
        assert len(Data_sets) > 0, "Data_sets must ot be empty."
        
        assert type(Data_params) == type([0]), "Data_params must be a list."
        assert len(Data_params) > 0, "Data_params must ot be empty."
        
        assert type(Splitters) == type([0]), "Splitters must be a list."
        assert len(Splitters) > 0, "Splitters must ot be empty."
        
        assert type(Models) == type([0]), "Models must be a list."
        assert len(Models) > 0, "Models must ot be empty."
        
        assert type(Metrics) == type([0]), "Metrics must be a list."
        assert len(Metrics) > 0, "Metrics must ot be empty."
        
        self.num_data_sets   = len(Data_sets)
        self.num_data_params = len(Data_params)
        self.num_models      = len(Models)
        self.num_metrics     = len(Metrics)
        
        self.Data_sets   = Data_sets
        self.Data_params = Data_params
        self.Models      = Models
        self.Metrics     = Metrics
        
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
                if isinstance(reps, list):
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
                    assert (isinstance(reps, int) or
                            isinstance(reps, str) or
                            isinstance(reps, tuple)), "Split repetition has a wrong format."
                    if isinstance(reps, tuple):
                        assert len(reps) > 0, "Some repetition information must be given."
                        for rep_part in reps:
                            assert (isinstance(rep_part, int) or
                                    isinstance(rep_part, str)), "Split repetition has a wrong format."
                    else:
                        reps = (reps,)
                        
                    reps = [reps]
            else:
                reps = [(0,)]
                
            for rep in reps:
                new_split_dict = {'Type': splitter_name, 'repetition': rep, 'test_part': splitter_tp}
                self.Splitters.append(new_split_dict)
        
        self.num_splitters = len(self.Splitters)
        
        self.provided_modules = True
        
    def set_parameters(self, model_for_path_transform,
                       num_samples_path_pred = 100, 
                       enforce_num_timesteps_out = False, 
                       enforce_prediction_time = True, 
                       exclude_post_crit = True,
                       allow_extrapolation = True,
                       dynamic_prediction_agents = False,
                       overwrite_results = False,
                       evaluate_on_train_set = True):
        
        model_to_path_module = importlib.import_module(model_for_path_transform)
        model_class_to_path = getattr(model_to_path_module, model_for_path_transform)
        
        # Test the parameters        
        if model_class_to_path == None or model_class_to_path.get_output_type() != 'path_all_wi_pov':
            raise TypeError("The chosen model does not predict trajectories.")

        # Set the number of paths that a trajectory prediction model has to predict
        assert isinstance(num_samples_path_pred, int), "num_samples_path_pred should be an integer."
        
        assert isinstance(enforce_num_timesteps_out, bool), "enforce_num_timesteps_out should be a boolean."
        
        assert isinstance(enforce_prediction_time, bool), "enforce_prediction_time should be a boolean."
        
        assert isinstance(exclude_post_crit, bool), "exclude_post_crit should be a boolean."
        
        assert isinstance(allow_extrapolation, bool), "allow_extrapolation should be a boolean."
        
        assert isinstance(dynamic_prediction_agents, str), "dynamic_prediction_agents should be a string."
        
        # make it backward compatiable
        if isinstance(overwrite_results, bool):
            if overwrite_results == False:
                overwrite_results = 'no'
            else:
                overwrite_results = 'model'
        
        assert isinstance(overwrite_results, str), "overwrite_results should be a string."
        assert overwrite_results in ['model', 'prediction', 'metric', 'no']
        
        
        assert isinstance(evaluate_on_train_set, bool), "evaluate_on_train_set should be a boolean."
        
        self.parameters = [model_class_to_path, num_samples_path_pred, 
                           enforce_num_timesteps_out, enforce_prediction_time, 
                           exclude_post_crit, allow_extrapolation, 
                           dynamic_prediction_agents, overwrite_results]
        
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
                        sample_string += '{}/{} '.format(data_set.Output_A[beh].to_numpy().sum(), 
                                                         data_set.num_behaviors[beh]) + beh + ', '
                    else:
                        sample_string += '{}/{} '.format(data_set.Output_A[beh].to_numpy().sum(), 
                                                         data_set.num_behaviors[beh]) + beh + ' '
                else:                            
                    sample_string += 'and {}/{} '.format(data_set.Output_A[beh].to_numpy().sum(), 
                                                         data_set.num_behaviors[beh]) + beh        
            sample_string += ' samples are admissible.'
        else:
            if hasattr(data_set, 'Output_T'):
                sample_string = '{}/{} samples are admissible.'.format(len(data_set.Output_T), 
                                                                       data_set.num_behaviors[0])
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
        
        print('This are the current computer specs:')
        print('')
        print('NumPy version:   ', np.__version__)
        print('PyTorch version: ', torch.__version__)
        print('Pandas version:  ', pd.__version__)
        print('')

        print('Processor memories:')
        # CPU
        CPU_mem = psutil.virtual_memory()
        cpu_total = CPU_mem.total / 2 ** 30
        cpu_used  = CPU_mem.used / 2 ** 30
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
                    splitter_name = splitter_param['Type']
                    splitter_rep = splitter_param['repetition']
                    splitter_tp = splitter_param['test_part']
                        
                    splitter_module = importlib.import_module(splitter_name)
                    splitter_class = getattr(splitter_module, splitter_name)
                    
                    # Initialize Splitting method
                    splitter = splitter_class(data_set, splitter_tp, splitter_rep)
                    
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
                    for l, model_name in enumerate(self.Models):
                        # Get model class
                        model_module = importlib.import_module(model_name)
                        model_class = getattr(model_module, model_name)
                        
                        # Initialize the model
                        model = model_class(data_set, splitter, self.evaluate_on_train_set)
                        
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
                        
                        # Make predictions on the given testing set
                        output = model.predict()
                        
                        # Get the type of prediction in this output
                        model_type = model.get_output_type()
                        
                        # Go through each metric used on the current prediction time
                        for m, metric_name in enumerate(self.Metrics):
                            # Get metric class
                            mwtric_module = importlib.import_module(metric_name)
                            metric_class = getattr(mwtric_module, metric_name)  
                            
                            # Initialize the metric
                            metric = metric_class(data_set, splitter, model)
                            
                            # Test if metric is applicable
                            metric_failure = metric.check_applicability()
                            
                            # print metric status to output
                            self.print_metric_status(metric, metric_failure)
                            
                            # Do not use metric if it cannot be applied
                            if metric_failure is not None:
                                continue
                            
                            # Get the output type the metric works on:
                            metric_type = metric.get_output_type()
                            
                            # Allow for possible transformation of prediction
                            output_trans = data_set.transform_outputs(output, model_type, 
                                                                      metric_type, model.pred_file)
                            
                            # Evaluate transformed output
                            metric.evaluate_prediction(output_trans)
    
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

        for metric_name in self.Metrics:
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

        for m, metric_name in enumerate(self.Metrics):
            metric_module = importlib.import_module(metric_name)
            metric_class = getattr(metric_module, metric_name)
            metric = metric_class(None, None, None)
            
            # Get index of result array at which comparable result is saved
            for i, data_set_dict in enumerate(self.Data_sets):
                # Get data set class
                data_set = data_interface(data_set_dict, self.parameters)

                for j, data_param in enumerate(self.Data_params):
                    data_set.get_data(**data_param)
                    for k, splitter_param in enumerate(self.Splitters):
                        splitter_name = splitter_param['Type']
                        splitter_rep = splitter_param['repetition']
                        splitter_tp = splitter_param['test_part']
                            
                        splitter_module = importlib.import_module(splitter_name)
                        splitter_class = getattr(splitter_module, splitter_name)
                        
                        splitter = splitter_class(data_set, splitter_tp, splitter_rep)
                        
                        # Get the name of the splitmethod used.
                        splitter_str = splitter.get_name()['file'] + splitter.get_rep_str()
                            
                        create_plot = plot_if_possible and metric.allows_plot()
                        if create_plot:
                            fig, ax = plt.subplots(figsize = (5,5))
                        
                        for l, model_name in enumerate(self.Models):
                            model_module = importlib.import_module(model_name)
                            model_class = getattr(model_module, model_name)
                            
                            # Initialize the model
                            model = model_class(data_set, splitter, self.evaluate_on_train_set)
                            
                            results_file_name = (data_set.data_file[:-4] + '--' + 
                                                 # Add splitting method
                                                 splitter_str + '--' + 
                                                 # Add model name
                                                 model.get_name()['file']  + '--' + 
                                                 # Add metric name
                                                 metric_class.get_name()['file']  + '.npy')
                            
                            results_file_name = results_file_name.replace(os.sep + 'Data' + os.sep,
                                                                          os.sep + 'Metrics' + os.sep)
                            
                            
                            
                            
                            if os.path.isfile(results_file_name):
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
                                    num = 16 + len(metric_class.get_name()['file'])
                                    figure_file = figure_file[:-num] + metric_class.get_name()['file'] + '.pdf'
                                    
                                    os.makedirs(os.path.dirname(figure_file), exist_ok = True)
                                    saving_figure = l == (self.num_models - 1)
                                    metric.create_plot(test_results, figure_file, fig, ax, saving_figure, model)
                            else:
                                print('Desired result not findable')
                                
                            if m == 0 and return_train_loss:
                                if model.provides_epoch_loss():
                                    train_loss_file_name = (data_set.data_file[:-4] + '--' + 
                                                            # Add splitting method
                                                            splitter_str + '--' +  
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
                    'all':       r'All',
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
        for j, metric_name in enumerate(self.Metrics):
            metric_module = importlib.import_module(metric_name)
            metric_class = getattr(metric_module, metric_name)
            
            Figure_string += ' \n' + r'    % Draw the metric ' + metric_class.get_name()['print'] + ' \n'
            
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
                
                Plot_string += (' \n' + r'    % Draw the metric ' + metric_class.get_name()['print'] + 
                                ' for ' + T0_names[data_set.t0_type] + 
                                ' on dataset ' + data_set.get_name()['print'] +  ' \n')
                
                
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
                    for m, model_name in enumerate(self.Models):
                        model_module = importlib.import_module(model_name)
                        model_class = getattr(model_module, model_name) 
                        Plot_string += model_class.get_name()['latex']
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
                for m, model_name in enumerate(self.Models):
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
                    Figure_string += (r'    \node[black, rotate = 90, font = \footnotesize] at ' + 
                                      '(0.9, {:0.3f}) '.format(y_value + 0.5 * plot_height) + 
                                      r'{' + metric_class.get_name()['latex'] + r'};' + ' \n')
                    
                
                
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
                        for m, model_name in enumerate(self.Models):
                            model_module = importlib.import_module(model_name)
                            model_class = getattr(model_module, model_name) 
                            Label_string += model_class.get_name()['latex']
                            if m < self.num_models - 1:
                                Label_string += ', '
                        Label_string += r'}' 
                        
                        
                        Plot_string = Plot_string.replace(r'xticklabels = {}', Label_string)
                        
                    if not plot_x_labels and  (1 + j) == self.num_metrics:
                        Label_string = r'xticklabels = {' 
                        for m, model_name in enumerate(self.Models):
                            model_module = importlib.import_module(model_name)
                            model_class = getattr(model_module, model_name) 
                            Label_string += model_class.get_name()['latex']
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
                                                                  x0_nI + 0.3, dx, Colors[i], 'Critical_split')
                    Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                      '({:0.3f}, {:0.3f}) '.format(allowed_width + inter_plot_space - y0_nI - legend_y_offset, 
                                                                   x0_nI + 0.6) + r'{$n_I =' + str(data_params['num_timesteps_in'][0]) + r'$};' + ' \n')
                    
                else:
                    # Random split
                    Figure_string += self.write_single_data_point(x0_nI + 0.3, y0_nI + legend_y_offset, dx, Colors[i], 'Critical_split')
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
                                                                  x0_st + 0.3, dx, 'black', split_type)
                    Figure_string += (r'        \node[black, rotate = 90, inner sep = 0, right, font = \footnotesize] at ' + 
                                      '({:0.3f}, {:0.3f}) '.format(allowed_width + inter_plot_space - y0_st - legend_y_offset, 
                                                                   x0_st + 0.6) + r'{' + split_name + r'};' + ' \n')
                    
                else:
                    # Random split
                    Figure_string += self.write_single_data_point(x0_st + 0.3, y0_st + legend_y_offset, dx, 'black', split_type)
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
                table_module = importlib.import_module(table_name)
                table_class = getattr(table_module, table_name)
                table_filename = table_class.get_name()['file']
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
                    model_name = self.Models[model_idx]
                    model_module = importlib.import_module(model_name)
                    model_class = getattr(model_module, model_name) 
                    Output_strings[n] += r'& \multicolumn{{{}}}'.format(nP) + r'{c}{' + model_class.get_name()['latex'] + r'}'
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
                min_value = '{:0.3f}'.format(Table_results_mean.min())
                max_value = '{:0.3f}'.format(Table_results_mean.max())
                extra_str_length = max(len(min_value), len(max_value)) - 4
                
                
                for i, row_name in enumerate(Row_iterator): 
                    if dataset_row:
                        row_item = data_interface(row_name, self.parameters)
                        row_latexname = row_item.get_name()['latex']
                    else:
                        row_module = importlib.import_module(row_name)
                        row_class  = getattr(row_module, row_name)
                        row_latexname = row_class.get_name()['latex']
                    
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
                        
                        results_split = self.Results[dataset_index, :, split_index, :, metric_index]
                    
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
                            previous_string = string.split('.')[0].split('$')[-1]
                            if previous_string.isnumeric():
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
        parameters[-1] = False
        
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
                s_name  = s_param['Type']
                s_rep = s_param['repetition']
                s_tp = s_param['test_part']
                
                s_class = getattr(importlib.import_module(s_name), s_name)
                s_inst  = s_class(None, s_tp, s_rep)
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
        split_rep = split_param['repetition']
        split_tp = split_param['test_part']
        
        
        splitter = split_class(data_set, split_tp, split_rep)
        splitter.split_data() 
        
        # Get test index 
        if self.plot_train:
            Index = splitter.Train_index
        else:
            Index = splitter.Test_index
        
        # Load needed files
        Input_path       = data_set.Input_path.iloc[Index]
        Output_path      = data_set.Output_path.iloc[Index]
        Output_A         = data_set.Output_A.iloc[Index]
        Output_T_E       = data_set.Output_T_E[Index]
        Domain           = data_set.Domain.iloc[Index]
        
        return [data_set, data_param, splitter, 
                Input_path, Output_path, 
                Output_A, Output_T_E, Domain]
    
    
    def _get_model_selection(self, data_set):
        if self.num_models > 1:
            print('------------------------------------------------------------------', flush = True)
            sample_string = 'In the current experiment, the following splitters are available:'
            for i, m_name in enumerate(self.Models):
                m_class = getattr(importlib.import_module(m_name), m_name)
                try:
                    sample_string += '\n{}: '.format(i + 1) + m_class.get_name()['print']  
                except:
                    mm = m_class(data_set, None, self.evaluate_on_train_set)
                    sample_string += '\n{}: '.format(i + 1) + mm.get_name()['print']  
            print(sample_string, flush = True)  
            print('Select the desired dataset by typing a number between 1 and {} for the specific splitter): '.format(self.num_models), flush = True)
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
            
            model_name = self.Models[i_d]
        else:
            model_name = self.Models[0]
            
        return model_name
        
    
    
    def _get_data_pred(self, data_set, splitter, model_name):
        model_class = getattr(importlib.import_module(model_name), model_name)
       
        # Load specific model
        model = model_class(data_set, splitter, self.evaluate_on_train_set)
        model.train()
        output = model.predict()
        
        output_trans_path = data_set.transform_outputs(output, model.get_output_type(), 
                                                       'path_all_wi_pov', model.pred_file)
        
        # Get predictions        
        Output_path_pred = output_trans_path[1]
        Pred_index = output_trans_path[0]
        
        # Transform index
        if self.plot_train:
            Index = splitter.Train_index
        else:
            Index = splitter.Test_index
            
        TP_index = Index[:,np.newaxis] == Pred_index[np.newaxis]
        assert (TP_index.sum(1) == 1).all()
        TP_index = TP_index.argmax(1)
        
        # Load needed files
        Output_path_pred = Output_path_pred.iloc[TP_index]
        
        return model, Output_path_pred
    
    def _get_data_sample(self, sample_ind, data_set, 
                         Input_path, Output_path, 
                         Output_A, Output_T_E, Domain):
        
        input_path       = Input_path.iloc[sample_ind]
        output_path      = Output_path.iloc[sample_ind]
        output_A         = Output_A.iloc[sample_ind]
        output_T_E       = Output_T_E[sample_ind]
        domain           = Domain.iloc[sample_ind]
        
        # Load raw darta
        ind_p = np.array([name for name in input_path.index 
                          if isinstance(input_path[name], np.ndarray)])
        op = np.stack(output_path[ind_p].to_numpy(), 0) # n_a x n_O x 2
        ip = np.stack(input_path[ind_p].to_numpy(), 0) # n_a x n_I x 2
        
        max_v = np.nanmax(np.stack([np.max(op, axis = (0,1)),
                                    np.max(ip, axis = (0,1))], axis = 0), axis = 0)
        
        min_v = np.nanmin(np.stack([np.min(op, axis = (0,1)),
                                    np.min(ip, axis = (0,1))], axis = 0), axis = 0)

        max_v = np.ceil((max_v + 5) / 5) * 5
        min_v = np.floor((min_v - 5) / 5) * 5
        
        if data_set.includes_images():
            img = data_set.return_batch_images(Domain.iloc[[sample_ind]], None, None, 
                                               None, None, grayscale = False) 
            img = img[0] / 255
        else:
            img = None
            
        return [op, ip, ind_p, min_v, max_v,
                output_A, output_T_E, img, domain]
    
    
    def _get_data_sample_pred(self, sample_ind, Output_path_pred):
        
        output_path_pred = Output_path_pred.iloc[sample_ind]
        
        
        ind_pp = np.array([name for name in output_path_pred.index 
                           if isinstance(output_path_pred[name], np.ndarray)])
        
        opp = np.stack(output_path_pred[ind_pp].to_numpy(), 0) # n_a x n_p x n_O x 2
        
        return [opp, ind_pp]
    
    def _get_data_sample_image(self, data_set, domain):
        if data_set.includes_images():
            img = data_set.return_batch_images(domain, None, None, 
                                                None, None, grayscale = False) 
            img = img / 255
        else:
            img = None
            
    
    def _draw_background(self, ax, data_set, img, domain):
        # Load line segments of data_set 
        map_lines_solid, map_lines_dashed = data_set.provide_map_drawing(domain = domain)
        
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
            height, width, _ = list(np.array(img.shape) * data_set.get_Target_MeterPerPx(domain))
            ax.imshow(img, extent=[-width/2, width/2, -height/2, height/2], interpolation='nearest')
        
        # Draw boundaries
        ax.add_collection(map_solid)
        ax.add_collection(map_dashed)
    
    def _select_testing_samples(self, load_all, Output_A):
        if not load_all:
            print('------------------------------------------------------------------', flush = True)
            if len(Output_A.columns) > 1:
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
                
                Chosen_index = np.where(Output_A.iloc[:, n_beh].to_numpy())[0]
            else:
                Chosen_index = np.where(Output_A.iloc[:, 0].to_numpy())[0]
                  
            print('', flush = True)
            
            print('Type a number between in between 1 and {} to select the example: '.format(len(Chosen_index)), flush = True)
            print('', flush = True)
            try:
                ind = int(input('Enter a number: ')) - 1
            except:
                ind = -1
            while ind < 0 or ind >= len(Chosen_index):
                print('This answer was not accepted. Please repeat: ', flush = True)
                print('', flush = True)
                try:
                    ind = int(input('Enter a number: ')) - 1
                except:
                    ind = -1
            print('')
            
            sample_inds = [Chosen_index[ind]]
        
        else:
            sample_inds = np.arange(len(Output_A))
            
        return sample_inds
    
    
    def _get_similar_inputs(self, data_set, splitter, sample_ind):
        data_set._extract_identical_inputs()
        
        if self.plot_train:
            Index = splitter.Train_index
        else:
            Index = splitter.Test_index
        
        subgroup = data_set.Subgroups[Index[sample_ind]]
        path_true_all = data_set.Path_true_all[subgroup]
        
        path_true_all = path_true_all[np.isfinite(path_true_all).any((1,2,3))]
        return path_true_all
        
    
    
    def plot_paths(self, load_all = False, 
                   plot_similar_futures = False, 
                   plot_train = False):
        assert self.provided_modules, "No modules have been provided. Run self.set_modules() first."
        assert self.provided_setting, "No parameters have been provided. Run self.set_parameters() first."
        
        self.plot_train = plot_train
        if self.plot_train:
            assert self.evaluate_on_train_set, "Training samples need to be predicted if they are to be plotted."
        
        plt.close('all')
        time.sleep(0.05)
        
        [data_set, data_param, splitter, 
         Input_path, Output_path, 
         Output_A, Output_T_E, Domain] = self._get_data()
        
        model_name = self._get_model_selection(data_set)
        model, Output_path_pred = self._get_data_pred(data_set, splitter, model_name)
        
        sample_inds = self._select_testing_samples(load_all, Output_A)
        
        ## Get specific case
        for sample_ind in sample_inds:   
            [op, ip, ind_p, min_v, max_v, 
             output_A, output_T_E, img, domain] = self._get_data_sample(sample_ind, data_set,
                                                                        Input_path, Output_path, 
                                                                        Output_A, Output_T_E, Domain)
                                                                        
            [opp, ind_pp] = self._get_data_sample_pred(sample_ind, Output_path_pred)
            
            
            # Check if multiple true input should be drawn
            if plot_similar_futures:
                paths_true_all = self._get_similar_inputs(data_set, splitter, sample_ind)

            
            
            # plot figure
            fig, ax = plt.subplots(figsize = (10,8))            
            
            # plot map
            self._draw_background(ax, data_set, img, domain)
            
            # plot inputs
            colors = matplotlib.cm.tab20(range(len(ind_p)))
            for i, agent in enumerate(ind_p):
                color = colors[i]
                ax.plot(ip[i,:,0], ip[i,:,1], color = color, 
                        marker = 'o', ms = 2.5, label = r'$A_{' + agent + r'}$', linewidth = 0.75)
                
                # plot predicted future
                if agent in ind_pp:
                    color_pred = np.ones(4, float)
                    color_pred[:3] = 1 - 0.5 * (1 - color[:3])
                    i_agent = np.where(agent == ind_pp)[0][0]
                    
                    num_samples_path_pred = self.parameters[1]
                    for j in range(num_samples_path_pred):
                        ax.plot(np.concatenate((ip[i,-1:,0], opp[i_agent,j,:,0])), 
                                np.concatenate((ip[i,-1:,1], opp[i_agent,j,:,1])), 
                                color = color_pred, marker = 'x', ms = 1, markeredgewidth = 0.25, 
                                linestyle = 'dashed', linewidth = 0.25)
                        
                    # plot true future
                    if plot_similar_futures:
                        for j_true in range(len(paths_true_all)):
                            ax.plot(np.concatenate((ip[i,-1:,0], paths_true_all[j_true, i_agent,:,0])), 
                                    np.concatenate((ip[i,-1:,1], paths_true_all[j_true, i_agent,:,1])),
                                    color = color, marker = 'x', ms = 1, markeredgewidth = 0.25, linewidth = 0.25)

                ax.plot(np.concatenate((ip[i,-1:,0], op[i,:,0])), 
                        np.concatenate((ip[i,-1:,1], op[i,:,1])),
                        color = color, marker = 'x', ms = 2.5, 
                        markeredgewidth = 0.25, linewidth = 0.75)
            
            # allow for latex code
            from matplotlib import rc
            # rc('text', usetex=True)
            
            # Format plot
            ax.set_aspect('equal', adjustable='box') 
            ax.set_xlim([min_v[0],max_v[0]])
            ax.set_ylim([min_v[1],max_v[1]])
            title = (r'Data set: ' + data_set.get_name()['print'] +  
                     r' with $\delta t = ' + str(data_param['dt']) + 
                     r'$, Model: ' + model.get_name()['print'])
            behs = np.array(output_A.index)
            if len(behs) > 1:
                title += r': \\True behavior: ' + behs[output_A][0] + r' at $t = ' + str(output_T_E)[:5] + '$' 
            ax.set_title(title)
            plt.axis('off')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            if self.plot_train:
                fig_str = 'traj_plot_test__'
            else:
                fig_str = 'traj_plot_train_'
            
            figure_file = data_set.change_result_directory(model.model_file, 'Metric_figures', 
                                                           fig_str + '{}'.format(sample_ind), '.pdf')
            
            os.makedirs(os.path.dirname(figure_file), exist_ok = True)
            fig.savefig(figure_file)
            
            plt.close(fig)
