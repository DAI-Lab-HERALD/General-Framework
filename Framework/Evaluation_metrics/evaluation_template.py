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
            
            self.t_e_quantile = self.data_set.p_quantile
            
    
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
        '''
        This returns the true and predicted classification probabilities.

        Returns
        -------
        P_true : np.ndarray
            This is the true probabilities with which one will observe a class, in the form of a
            :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array with float values. 
            One value per row will be one, whil ethe others will be zero.
        P_pred : np.ndarray
            This is the predicted probabilities with which one will observe a class, in the form of 
            a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array with float values. 
            The sum in each row will be 1.
        Class_names : list
            The list with :math:`N_{classes}` entries contains the names of the aforementioned
            behaviors.

        '''
        assert self.get_output_type()[:5] == 'class', 'This is not a classification metric.'
        P_true = self.Output_A.to_numpy().astype(float)
        P_pred = self.Output_A_pred.to_numpy().astype(float)
        Class_names = self.Output_A.columns
        
        return P_true, P_pred, Class_names
    
    
    def get_true_and_predicted_class_times(self):
        '''
        This returns the true and predicted classification timepoints, at which a certain
        behavior can be first classified.

        Returns
        -------
        T_true : np.ndarray
            This is the true time points at which one will observe a class, in the form of a
            :math:`\{N_{samples} \times N_{classes} \times 1\}` dimensional numpy array with float 
            values. One value per row will be given (actual observed class), while the others 
            will be np.nan.
        T_pred : np.ndarray
            This is the predicted time points at which one will observe a class, in the form of a
            :math:`\{N_{samples} \times N_{classes} \times N_{quantiles}\}` dimensional numpy array 
            with float values. Along the last dimesnion, the time values corresponf to the quantile 
            values of the predicted distribution of the time points. The quantile values can be found
            in **self.t_e_quantile**.
        Class_names : list
            The list with :math:`N_{classes}` entries contains the names of the aforementioned
            behaviors.

        '''
        assert self.get_output_type() == 'class_and_time', 'This is not a classification metric.'
        Class_names = self.Output_A.columns
        T_true = np.ones((*self.Output_A.shape, 1)) * np.nan
        T_pred = np.ones((*self.Output_A.shape, self.t_e_quantile)) * np.nan
        
        T_true[np.arange(len(T_true)), np.argmax(self.Output_A.to_numpy(), 1)] = self.Output_T_E.to_numpy()
        
        for i in range(T_pred.shape[0]):
            for j in range(T_pred.shape[1]):
                T_pred[i,j] = self.Output_T_E_pred.iloc[i,j]
        
        return T_true, T_pred, Class_names
    
    
    def get_true_and_predicted_paths(self, num_preds = None, return_types = False):
        '''
        This returns the true and predicted trajectories.

        Parameters
        ----------
        num_preds : int, optional
            The number :math:`N_{preds}` of different predictions used. The default is None,
            in which case all available predictions are used.
        return_types : bool, optional
            Decides if agent types are returned as well. The default is False.

        Returns
        -------
        Path_true : np.ndarray
            This is the true observed trajectory of the agents, in the form of a
            :math:`\{N_{samples} \times 1 \times N_{agents} \times N_{O} \times 2\}` dimensional numpy 
            array with float values. If an agent is fully or or some timesteps partially not observed, 
            then this can include np.nan values.
        Path_pred : np.ndarray
            This is the predicted furure trajectories of the agents, in the form of a
            :math:`\{N_{samples} \times N_{preds} \times N_{agents} \times N_{O} \times 2\}` dimensional 
            numpy array with float values. If an agent is fully or or some timesteps partially not observed, 
            then this can include np.nan values.
        Pred_steps : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times N_{O}\}` dimensional numpy array with 
            boolean values. It indicates for each agent and timestep if the prediction should influence
            the final metric result.
        T : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings 
            that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
            for available types). If an agent is not observed at all, the value will instead be '0'.
            It is only returned if **return_types** is *True*.

        '''
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
            Types = Types.astype(str)
            Types[Types == 'nan'] = '0'
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
        

        
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                    Evaluation metric dependend functions                          ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    

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
        
    def setup_method(self):
        # Will do any preparation the method might require, like calculating
        # weights.
        # creates:
            # self.weights_saved -  The weights that were created for this metric,
            #                       will be in the form of a list
        raise AttributeError('Has to be overridden in actual metric class.')
        
    def requires_preprocessing(self):
        # Returns a boolean output, True if preprocesing of true output
        # data for the calculation of weights is required, which might be 
        # avoided in repeated cases
        raise AttributeError('Has to be overridden in actual metric class.')
        
    
    def get_output_type(self = None):
        # Should return 'class', 'class_and_time', 'path_tar', 'path_all'
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def check_applicability(self):
        # Provides feedback on if a metric can be used, as it might be
        # related to only specifc datasets/scenarios/models/etc.
        # Returns None if metric is unrestricedly applicable.
        raise AttributeError('Has to be overridden in actual metric class.')
        
    
    def get_opt_goal(self = None):
        # Should return 'minimize' or 'maximize'
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def evaluate_prediction_method(self):
        # Takes true outputs and corresponding predictions to calculate some
        # metric to evaluate a model
        raise AttributeError('Has to be overridden in actual metric class.')
        # return results # results is a list
    
    
    def is_log_scale(self = None):
        # Should return 'False' or 'True'
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def allows_plot(self):
        # Returns a boolean output, True if a plot can be created, False if not.
        raise AttributeError('Has to be overridden in actual metric class.')
        
        
    def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
        '''
        This function creates the final plot.
        
        This function is cycled over all included models, so they can be combined
        in one figure. However, it is also possible to save a figure for each model,
        if so desired. In that case, a new instanc of fig and ax should be created and
        filled instead of the ones passed as parameters of this functions, as they are
        shared between all models.
        
        If only one figure is created over all models, this function should end with:
            
        if save:
            ax.legend() # Depending on if this is desired or not
            fig.show()
            fig.savefig(test_file, bbox_inches='tight')  

        Parameters
        ----------
        results : list
            This is the list produced by self.evaluate_prediction_method().
        test_file : str
            This is the location at which the combined figure of all models can be
            saved (it ends with '.pdf'). If one saves a result for each separate model, 
            one should adjust the filename to indicate the actual model.
        fig : matplotlib.pyplot.Figure
            This is the overall figure that is shared between all models.
        ax : matplotlib.pyplot.Axes
            This is the overall axes that is shared between all models.
        save : bool, optional
            This is the trigger that indicates if one currently is plotting the last
            model, which should indicate that the figure should now be saved. The default is False.
        model_class : Framework_Model, optional
            The model for which the current results were calculated. The default is None.

        Returns
        -------
        None.

        '''
        # Function that visualizes result if possible
        if self.allows_plot():
            raise AttributeError('Has to be overridden in actual metric class.')
        else:
            pass
