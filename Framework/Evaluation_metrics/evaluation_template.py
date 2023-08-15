import pandas as pd
import numpy as np
import os


class evaluation_template():
    def __init__(self, data_set, splitter, model):
        if data_set is not None:
            self.depict_results = False
            self.Input_prediction_full = data_set.Input_prediction
            self.Input_path_full       = data_set.Input_path
            self.Input_T_full          = data_set.Input_T
            
            self.Output_T_full         = data_set.Output_T
            self.Output_A_full         = data_set.Output_A
            self.Output_T_E_full       = data_set.Output_T_E
            
            self.Domain_full           = data_set.Domain
            
            self.num_samples_full      = len(self.Output_T_full)
            
            # Ensure the right numebr of agents
            if self.get_output_type()[:4] == 'path':
                Agents = np.array(self.Input_path_full.columns)
                Agent_index = np.ones(len(Agents), bool)
                
                if self.get_output_type() == 'path_all_wo_pov':
                    pov_bool = Agents == data_set.pov_agent
                    Agent_index[pov_bool] = False
            else:
                Agent_index = np.zeros(len(self.Input_path_full.columns), bool)
            
            self.Output_path_full = data_set.Output_path.iloc[:, Agent_index]
            self.Type_full = data_set.Type.iloc[:, Agent_index]
            self.Pred_agents_full = model._determine_pred_agents(data_set, 
                                                                 data_set.Recorded,
                                                                 data_set.dynamic_prediction_agents)
            self.Pred_agents_full = self.Pred_agents_full[:, Agent_index]
            
            self.data_set = data_set
            self.splitter = splitter
            self.model = model
            
    
            if self.requires_preprocessing():
                test_file = data_set.change_result_directory(splitter.split_filse,
                                                             'Metrics', self.get_name()['file'] + '_weights')
                if os.path.isfile(test_file):
                    self.weights_saved = list(np.load(test_file, allow_pickle = True)[:-1])  
                else:
                    self.setup_method() # output needs to be a list of components
                    
                    save_data = np.array(self.weights_saved + [0], object) # 0 is there to avoid some numpy load and save errros
                    
                    os.makedirs(os.path.dirname(test_file), exist_ok=True)
                    np.save(test_file, save_data)
        else:
            self.depict_results = True
    
    
    def _set_current_data(self, Index, Predictions):
        self.Input_path        = self.Input_path_full.iloc[Index]
        self.Input_T           = self.Input_T_full[Index]
        
        self.Output_path       = self.Output_path_full.iloc[Index]
        self.Output_T          = self.Output_T_full[Index]
        self.Output_A          = self.Output_A_full.iloc[Index]
        self.Output_T_E        = self.Output_T_E_full[Index]
        
        self.Type              = self.Type_full.iloc[Index]
        self.Pred_agents       = self.Pred_agents_full[Index]
        self.Domain            = self.Domain_full.iloc[Index]
        
        self.num_samples = len(self.Output_path)
        
                    
        if self.get_output_type()[:4] == 'path':
            self.Output_path_pred = Predictions[0].iloc[Index]
            
        elif self.get_output_type() == 'class_and_time':
            self.Output_A_pred   = Predictions[0].iloc[Index]
            self.Output_T_E_pred = Predictions[1].iloc[Index]
            
        elif self.get_output_type() == 'class':
            self.Output_A_pred = Predictions[0].iloc[Index]
    
    
    def get_true_and_predicted_class_probabilities(self):
        pass
    
    
    def get_true_and_predicted_paths(self, num_preds = None, return_types = False):
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        nto = self.data_set.num_timesteps_out_real
        
        if num_preds == None:
            num_preds = self.data_set.num_samples_path_pred
            idx = np.arange(self.data_set.num_samples_path_pred)
        elif num_preds <= self.data_set.num_samples_path_pred:
            idx = np.random.permutation(self.data_set.num_samples_path_pred)[:num_preds]#
        else:
            idx = np.random.randint(0, self.data_set.num_samples_path_pred, num_preds)
        
        num_samples, num_agents = self.Output_path_pred.shape
        
        Path_pred = np.zeros((num_samples, num_preds, num_agents, nto, 2))
        Path_true = np.zeros((num_samples, 1, num_agents, nto, 2))
        Pred_step = np.zeros((num_samples, num_agents, nto), bool)
        
        for i in range(num_samples):
            nto_i = min(nto, len(self.Output_T[i]))
            
            pred_agents = self.Pred_agents[i] 
            
            path_pred = self.Output_path_pred.iloc[i, pred_agents]
            path_pred = np.stack(path_pred.to_numpy(), axis = 1)
            
            path_true = self.Output_path.iloc[i, pred_agents]
            path_true = np.stack(path_true, axis = 0)[np.newaxis]
            
            # For some reason using pred_agents here moves the agent dimension to the front
            Path_pred[i,:,pred_agents,:nto_i] = path_pred[idx,:,:nto_i].transpose(1,0,2,3)
            Path_true[i,:,pred_agents,:nto_i] = path_true[:,:,:nto_i].transpose(1,0,2,3)
            
            Pred_step[i,pred_agents,:nto_i] = True
        
        if return_types:
            Types = self.Type.to_numpy()
            return Path_true, Path_pred, Pred_step, Types     
        else:
            return Path_true, Path_pred, Pred_step
    
        
    def evaluate_prediction(self, Output_pred, create_plot_if_possible = False):
        if self.depict_results:
            raise AttributeError("This loaded version only allows for plotting results.")
        
        self.metric_file = self.data_set.change_result_directory(self.model.model_file,
                                                                 'Metrics', self.get_name()['file'])
        
        if os.path.isfile(self.metric_file) and not self.data_set.overwrite_results:
            Results = list(np.load(self.metric_file, allow_pickle = True)[:-1])
        else:
            # Get output_data_train           
            if self.get_output_type()[:4] == 'path':
                [Output_path_pred] = Output_pred
                # reorder columns if needed
                Output_path_pred = Output_path_pred[self.Output_path_full.columns]
                for i in range(self.num_samples_full):                
                    test_length = len(self.Output_T_full[i])
                    for j in range(Output_path_pred.shape[1]):
                        # Ensure that prediction has corresponding ground truth
                        if not isinstance(Output_path_pred.iloc[i,j], np.ndarray):
                            assert not self.Pred_agents_full[i,j], "Desired agent is missing"
                            continue
                        
                        Output_path_pred.iloc[i,j] = Output_path_pred.iloc[i,j][:,:test_length]
                        if self.Pred_agents_full[i,j]:
                            assert np.isfinite(Output_path_pred.iloc[i,j]).all(), "NaN positions are predicted."
                            
                Predictions = [Output_path_pred]
            elif self.get_output_type() == 'class_and_time':
                [Output_A_pred, Output_T_E_pred] = Output_pred
                
                # reorder columns if needed
                Output_A_pred   = Output_A_pred[self.Output_A_full.columns]
                Output_T_E_pred = Output_T_E_pred[self.Output_A_full.columns]
                
                Predictions = [Output_A_pred, Output_T_E_pred]
                
            elif self.get_output_type() == 'class':
                [Output_A_pred] = Output_pred
                
                # reorder columns if needed
                Output_A_pred = Output_A_pred[self.Output_A_full.columns]
                
                Predictions = [Output_A_pred]
            else:
                raise AttributeError("This type of prediction is not implemented")
            
            
            Results = []
            # Evaluate the model both on the training set and the testing set
            if not np.array_equal(np.unique(self.splitter.Train_index), 
                                  np.unique(self.splitter.Test_index)):
            
                Indeces = [self.splitter.Train_index, self.splitter.Test_index]
                
                for Index in Indeces:
                    self._set_current_data(Index, Predictions)
                    
                    Results.append(self.evaluate_prediction_method()) # output needs to be a list of components
            else:
                Index = self.splitter.Test_index
                self._set_current_data(Index)
                
                results = self.evaluate_prediction_method()
                
                Results.append(results)
                Results.append(results)
            
            save_data = np.array(Results + [[0] * len(Results[0])], object) #0 is there to avoid some numpy load and save errros
            
            os.makedirs(os.path.dirname(self.metric_file), exist_ok=True)
            np.save(self.metric_file, save_data)
        
        if create_plot_if_possible:
            self.create_plot(Results[1], self.metric_file)
 
        return Results
        
    
    def create_plot(self, results, test_file):
        # Function that visualizes result if possible
        if self.allows_plot():
            raise AttributeError('Has to be overridden in actual metric class.')
        else:
            pass
        
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                    Evaluation metric dependend functions                          ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    def setup_method(self):
        # Will do any preparation the method might require, like calculating
        # weights.
        # creates:
            # self.weights_saved -  The weights that were created for this metric,
            #                       will be in the form of a list
        raise AttributeError('Has to be overridden in actual metric class.')
    
    
    def evaluate_prediction_method(self):
        # Takes true outputs and corresponding predictions to calculate some
        # metric to evaluate a model
        raise AttributeError('Has to be overridden in actual metric class.')
        # return results # results is a list
        
    
    def get_output_type(self = None):
        # Should return 'class', 'class_and_time', 'path_tar', 'path_all'
        raise AttributeError('Has to be overridden in actual metric class')
        
    
    def is_log_scale(self = None):
        # Should return 'False' or 'True'
        raise AttributeError('Has to be overridden in actual metric class')
        
    
    def get_opt_goal(self = None):
        # Should return 'minimize' or 'maximize'
        raise AttributeError('Has to be overridden in actual metric class')
        

    def get_name(self = None):
        r'''
        Provides a dictionary with the different names of the evaluation metric.
            
        Returns
        -------
        names : dict
          The first key of names ('print')  will be primarily used to refer to the evaluation metric in console outputs. 
                
          The 'file' key has to be a string that does not include any folder separators 
          (for any operating system), as it is mostly used to indicate that certain result files belong to this evaluation metric. 
                
          The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
          latex commands - such as using '$$' for math notation.
            
        '''
        raise AttributeError('Has to be overridden in actual metric class')
        
    def requires_preprocessing(self):
        # Returns a boolean output, True if preprocesing of true output
        # data for the calculation of weights is required, which might be 
        # avoided in repeated cases
        raise AttributeError('Has to be overridden in actual metric class.')
        
    def allows_plot(self):
        # Returns a boolean output, True if a plot can be created, False if not.
        raise AttributeError('Has to be overridden in actual metric class.')
        
    def check_applicability(self):
        # Provides feedback on if a metric can be used, as it might be
        # related to only specifc datasets/scenarios/models/etc.
        # Returns None if metric is unrestricedly applicable.
        raise AttributeError('Has to be overridden in actual metric class.')
        
        
