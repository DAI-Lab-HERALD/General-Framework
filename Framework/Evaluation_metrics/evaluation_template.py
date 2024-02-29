import pandas as pd
import numpy as np
import os


class evaluation_template():
    def __init__(self, data_set, splitter, model):
        if data_set is not None:
            self.data_set = data_set
            self.splitter = splitter
            self.model = model
            
            self.depict_results = False
            
            self.Output_A_full   = self.data_set.Output_A
            self.Output_T_E_full = self.data_set.Output_T_E
            self.Scenario_full   = self.data_set.Domain.Scenario_type
            
            self.t_e_quantile = self.data_set.p_quantile
            
            self.metric_override = self.data_set.overwrite_results in ['model', 'prediction', 'metric']
            
            if self.requires_preprocessing():
                test_file = self.data_set.change_result_directory(splitter.split_filse,
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
    

    
    #%% Helper functions
            
    
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
        
        # Get true and predicted probabilities
        P_true = self.Output_A.to_numpy().astype(float)
        P_pred = self.Output_A_pred.fillna(0.0).to_numpy().astype(float)
        Class_names = self.Output_A.columns
        
        # Remove cases where there was only on possible case to predict
        use_column = np.zeros(P_true.shape[1], bool)
        for i, behs in enumerate(self.data_set.scenario_behaviors):
            if len(behs) > 1:
                if self.data_set.unique_scenarios[i] in self.Scenario:
                    use_column |= np.in1d(Class_names, behs)
        
        # Remove useless columns
        P_true = P_true[:,use_column]
        P_pred = P_pred[:,use_column]
        
        Class_names = Class_names[use_column]
        
        # Remove useles rows
        use_row = P_true.sum(1) == 1
        
        P_true = P_true[use_row]
        P_pred = P_pred[use_row]
        
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
        
        T_true[np.arange(len(T_true)), np.argmax(self.Output_A.to_numpy(), 1), 0] = self.Output_T_E.to_numpy()
        
        for i in range(T_pred.shape[0]):
            for j in range(T_pred.shape[1]):
                t_pred = self.Output_T_E_pred.iloc[i,j]
                if isinstance(t_pred, np.ndarray):
                    T_pred[i,j] = t_pred
        
        # Get use column and use row
        P_true = self.Output_A.to_numpy().astype(float)
        use_column = np.zeros(P_true.shape[1], bool)
        for i, behs in enumerate(self.data_set.scenario_behaviors):
            if len(behs) > 1:
                if self.data_set.unique_scenarios[i] in self.Scenario:
                    use_column |= np.in1d(Class_names, behs)
        
        P_true = P_true[:,use_column]
        use_row = P_true.sum(1) == 1
        
        # remove useles rows and columns
        Class_names = Class_names[use_column]
        T_true = T_true[use_row][:,use_column]
        T_pred = T_pred[use_row][:,use_column]
        
        return T_true, T_pred, Class_names
    
    
    def get_true_and_predicted_paths(self, num_preds = None, return_types = False, exclude_late_timesteps = True):
        '''
        This returns the true and predicted trajectories.

        Parameters
        ----------
        num_preds : int, optional
            The number :math:`N_{preds}` of different predictions used. The default is None,
            in which case all available predictions are used.
        return_types : bool, optional
            Decides if agent types are returned as well. The default is False.
        exclude_late_timesteps : bool, optional
            Decides if predicted timesteps after the set prediction horizon should be excluded. 
            The default is True.

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
        Types : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings 
            that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
            for available types). If an agent is not observed at all, the value will instead be '0'.
            It is only returned if **return_types** is *True*.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        
        # Get the stochastic prediction indices
        if num_preds is None:
            idx = np.arange(self.data_set.num_samples_path_pred)
        else:
            if num_preds <= self.data_set.num_samples_path_pred:
                idx = np.random.permutation(self.data_set.num_samples_path_pred)[:num_preds] #
            else:
                idx = np.random.randint(0, self.data_set.num_samples_path_pred, num_preds)
        
        self.model._transform_predictions_to_numpy(self.Pred_index, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov',
                                                   exclude_late_timesteps)
        
        Path_true = self.model.Path_true[self.Index_curr_pred]
        Path_pred = self.model.Path_pred[self.Index_curr_pred][:, idx]
        Pred_step = self.model.Pred_step[self.Index_curr_pred]

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        Path_true = Path_true[Use_samples]
        Path_pred = Path_pred[Use_samples]
        Pred_step = Pred_step[Use_samples]

        if return_types:
            Types = self.model.T_pred[self.Index_curr_pred]
            Types = Types[Use_samples]
            return Path_true, Path_pred, Pred_step, Types     
        else:
            return Path_true, Path_pred, Pred_step
        
    
    def get_other_agents_paths(self, return_types = False):
        '''
        This returns the true observed trajectories of all agents that are not the
        predicted agents.

        Parameters
        ----------
        return_types : bool, optional
            Decides if agent types are returned as well. The default is False.

        Returns
        -------
        Path_other : np.ndarray
            This is the true observed trajectory of the agents, in the form of a
            :math:`\{N_{samples} \times 1 \times N_{agents_other} \times N_{O} \times 2\}` dimensional numpy 
            array with float values. If an agent is fully or or some timesteps partially not observed, 
            then this can include np.nan values.
        Types : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents_other}\}` dimensional numpy array. It includes strings 
            that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
            for available types). If an agent is not observed at all, the value will instead be '0'.
            It is only returned if **return_types** is *True*.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        
        self.data_set._extract_original_trajectories()
        self.data_set._determine_pred_agents(eval_pov = self.get_output_type() != 'path_all_wo_pov')

        Y_orig = self.data_set.Y_orig[self.Index_curr]
        Pred_agents = self.data_set.Pred_agents_eval[self.Index_curr]
        Other_agents = (~Pred_agents) & np.isfinite(Y_orig).all(-1).any(-1) # at least one position must be fully known

        num_samples = len(self.Index_curr)
        max_num_other_agents = Other_agents.sum(1).max()

        i_agent_sort = np.argsort(-Other_agents.astype(float))
        i_agent_sort = i_agent_sort[:,:max_num_other_agents]
        i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_other_agents))

        Path_other = Y_orig[i_sampl_sort, i_agent_sort][:,np.newaxis]

        if return_types:
            Types = self.data_set.Type.iloc[self.Index_curr].to_numpy()
            Types = Types.astype(str)
            Types[self.T_pred == 'nan'] = '0'

            Types = Types[i_sampl_sort, i_agent_sort]
            
            return Path_other, Types
        
        else:
            return Path_other
        
        
    def get_KDE_probabilities(self, joint_agents = True):
        '''
        This return the probabilities asigned to trajectories according to 
        a Gaussian KDE method.

        Parameters
        ----------
        joint_agents : bool, optional
            This says if the probabilities for the predicted trajectories
            are to be calcualted for all agents jointly. If this is the case,
            then, :math:`N_{agents}` in the output is 1. The default is True.

        Returns
        -------
        KDE_pred_log_prob_true : np.ndarray
            This is a :math:`\{N_{samples} \times 1 \times N_{agents}\}`
            array that includes the probabilites for the true observations according to
            the KDE model trained on the predicted trajectories.
        KDE_pred_log_prob_pred : np.ndarray
            This is a :math:`\{N_{samples} \times N_{preds} \times N_{agents}\}`
            array that includes the probabilites for the predicted trajectories 
            according to the KDE model trained on the predicted trajectories.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        
        if joint_agents:
            self.model._get_joint_KDE_pred_probabilities(self.Pred_index, self.Output_path_pred, 
                                                         self.get_output_type() == 'path_all_wo_pov')
            
            KDE_pred_log_prob_true = self.model.Log_prob_joint_true[:,:,np.newaxis]
            KDE_pred_log_prob_pred = self.model.Log_prob_joint_pred[:,:,np.newaxis]
            
        else:
            self.model._get_indep_KDE_pred_probabilities(self.Pred_index, self.Output_path_pred, 
                                                         self.get_output_type() == 'path_all_wo_pov')
            
            KDE_pred_log_prob_true = self.model.Log_prob_indep_true
            KDE_pred_log_prob_pred = self.model.Log_prob_indep_pred
        
        KDE_pred_log_prob_true = KDE_pred_log_prob_true[self.Index_curr_pred]
        KDE_pred_log_prob_pred = KDE_pred_log_prob_pred[self.Index_curr_pred]
        
        # Get useful samples
        self.model._transform_predictions_to_numpy(self.Pred_index, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov')
        Pred_step = self.model.Pred_step[self.Index_curr_pred]

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        KDE_pred_log_prob_true = KDE_pred_log_prob_true[Use_samples]
        KDE_pred_log_prob_pred = KDE_pred_log_prob_pred[Use_samples]

        return KDE_pred_log_prob_true, KDE_pred_log_prob_pred
    
    
    def get_true_prediction_with_same_input(self):
        '''
        This returns the true trajectories from the current sample as well as all
        other samples which had the same past trajectories. It should be used only
        in conjunction with *get_true_and_predicted_paths()*.

        Returns
        -------
        Path_true_all : np.ndarray
            This is the true observed trajectory of the agents, in the form of a
            :math:`\{N_{subgroups} \times N_{same} \times N_{agents} \times N_{O} \times 2\}` 
            dimensional numpy array with float values. If an agent is fully or on some 
            timesteps partially not observed, then this can include np.nan values. It
            must be noted that :math:`N_{same}` is the maximum number of similar samples,
            so for a smaller number, there will also be np.nan values.
        Subgroup_ind : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with int values. 
            All samples with the same value belong to a group with the same corresponding
            input. This can be used to avoid having to evaluate the same metric values
            for identical samples. It must however be noted, that due to randomness in 
            the model, the predictions made for these samples might differ.
            
            The value in this array will indicate which of the entries of **Path_true_all**
            should be chosen.

        '''
        self.data_set._extract_identical_inputs(eval_pov = self.get_output_type() == 'path_all_wi_pov')

        # Get useful samples
        self.model._transform_predictions_to_numpy(self.Pred_index, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov')
        Pred_step = self.model.Pred_step[self.Index_curr_pred]

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        Use_subgroups = self.data_set.Subgroups[self.Index_curr_pred][Use_samples]

        Subgroup_unique, Subgroup = np.unique(Use_subgroups, return_inverse = True)
        Path_true_all = self.data_set.Path_true_all[Subgroup_unique]
        
        return Path_true_all, Subgroup
    
    
    def get_true_likelihood(self, joint_agents = True):
        '''
        This return the probabilities asigned to ground truth trajectories 
        according to a Gaussian KDE method fitted to the ground truth samples
        with an identical inputs.
        

        Parameters
        ----------
        joint_agents : bool, optional
            This says if the probabilities for the predicted trajectories
            are to be calcualted for all agents jointly. If this is the case,
            then, :math:`N_{agents}` in the output is 1. The default is True.

        Returns
        -------
        KDE_true_log_prob_true : np.ndarray
            This is a :math:`\{N_{samples} \times 1 \times N_{agents}\}`
            array that includes the probabilites for the true observations according 
            to the KDE model trained on the grouped true trajectories.
            
        KDE_true_log_prob_pred : np.ndarray
            This is a :math:`\{N_{samples} \times N_{preds} \times N_{agents}\}`
            array that includes the probabilities for the predicted trajectories
            according to the KDE model trained on the grouped true trajectories.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        
        if joint_agents:
            self.data_set._get_joint_KDE_probabilities(self.get_output_type() == 'path_all_wo_pov')
            self.model._get_joint_KDE_true_probabilities(self.Pred_index, self.Output_path_pred, 
                                                         self.get_output_type() == 'path_all_wo_pov')
            
            KDE_true_log_prob_true = self.data_set.Log_prob_true_joint[:,np.newaxis,np.newaxis]
            KDE_true_log_prob_pred = self.model.Log_prob_true_joint_pred[:,:,np.newaxis]
            
        else:
            self.data_set._get_indep_KDE_probabilities(self.get_output_type() == 'path_all_wo_pov')
            self.model._get_indep_KDE_true_probabilities(self.Pred_index, self.Output_path_pred, 
                                                         self.get_output_type() == 'path_all_wo_pov')
            
            KDE_true_log_prob_true = self.data_set.Log_prob_true_indep[:,np.newaxis,:]
            KDE_true_log_prob_pred = self.model.Log_prob_true_indep_pred
        
        KDE_true_log_prob_true = KDE_true_log_prob_true[self.Index_curr]
        KDE_true_log_prob_pred = KDE_true_log_prob_pred[self.Index_curr_pred]
        
        # Get useful samples
        self.model._transform_predictions_to_numpy(self.Pred_index, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov')
        Pred_step = self.model.Pred_step[self.Index_curr_pred]

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        KDE_true_log_prob_true = KDE_true_log_prob_true[Use_samples]
        KDE_true_log_prob_pred = KDE_true_log_prob_pred[Use_samples]

        return KDE_true_log_prob_true, KDE_true_log_prob_pred
    
    #%% Actual evaluation functions
    def _set_current_data(self, Output_pred):
        self.Pred_index = Output_pred[0]
        
        match = self.Pred_index[np.newaxis,:] == self.Index_curr[:,np.newaxis]
        
        if not (match.sum(1) == 1).all():
            return False
        
        self.Index_curr_pred = match.argmax(1)
        
        if self.get_output_type()[:5] == 'class':
            # Get label predictions
            Output_A_pred = Output_pred[1]
            columns = Output_A_pred.columns
            
            self.Output_A_pred = Output_A_pred[columns].iloc[self.Index_curr_pred]
            self.Output_A      = self.Output_A_full[columns].iloc[self.Index_curr]
            self.Scenario      = np.unique(self.Scenario_full.iloc[self.Index_curr])
            
            if self.get_output_type() == 'class_and_time':
                Output_T_E_pred = Output_pred[2]
                
                # reorder columns if needed
                self.Output_T_E      = self.Output_T_E_full[self.Index_curr]
                self.Output_T_E_pred = Output_T_E_pred[columns].iloc[self.Index_curr_pred]
        
        else:
            Agents = np.array(self.data_set.Output_path.columns)
            self.Output_path_pred = Output_pred[1][Agents]
        
        return True
        
        
    def _evaluate_on_subset(self, Output_pred, create_plot):
        if len(self.Index_curr) == 0:
            return None
        
        available = self._set_current_data(Output_pred)
        if available:
            results = self.evaluate_prediction_method()
        
            if create_plot:
                self.create_plot(results, self.metric_file)
        else:
            results = None
        
        return results
    
        
    def evaluate_prediction(self, Output_pred, create_plot_if_possible = False):
        if self.depict_results:
            raise AttributeError("This loaded version only allows for plotting results.")
        
        self.metric_file = self.data_set.change_result_directory(self.model.model_file,
                                                                 'Metrics', self.get_name()['file'])
        
        if os.path.isfile(self.metric_file) and not self.metric_override:
            Results = list(np.load(self.metric_file, allow_pickle = True)[:-1])
            
            if (Results[0] is None) and self.model.evaluate_on_train_set:
                # Get train results
                self.Index_curr = (self.splitter.Train_index)
                Results[0] = self._evaluate_on_subset(Output_pred, create_plot_if_possible)
     
            return Results
        else:
            
            Results = []
            # Evaluate the model both on the training set and the testing set
            if not np.array_equal(np.unique(self.splitter.Train_index), 
                                  np.unique(self.splitter.Test_index)):
                
                # check what predictions are available
                if self.model.evaluate_on_train_set:
                    Indices = [self.splitter.Train_index, self.splitter.Test_index]
                else:
                    Indices = [np.array([]), self.splitter.Test_index]
                    
                    
                for self.Index_curr in Indices:
                    Results.append(self._evaluate_on_subset(Output_pred, create_plot_if_possible)) # output needs to be a list of components
            else:
                self.Index_curr = self.splitter.Test_index
                
                results = self._evaluate_on_subset(Output_pred, create_plot_if_possible)
                Results.append(results)
                Results.append(results)
            
            save_data = np.array(Results + [0], object) #0 is there to avoid some numpy load and save errros
            
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
        
        
    def metric_boundaries(self = None):
        # Should return a list with two entries. These are the minimum and 
        # maximum possible values. If no such boundary on potential metric values 
        # exists, then those values should be none instead
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
