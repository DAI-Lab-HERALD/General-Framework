import pandas as pd
import numpy as np
import os
import torch
import warnings
import scipy as sp
import importlib
import psutil

from rome.ROME import ROME

class model_template():
    def __init__(self, model_kwargs, data_set, splitter, evaluate_on_train_set, behavior = None):
        # Set model kwargs
        self.model_kwargs = model_kwargs
        
        if data_set is not None:
            # Load gpu
            if self.requires_torch_gpu():
                if torch.cuda.is_available():
                    self.device = torch.device('cuda', index=0) 
                    torch.cuda.set_device(0)
                else:
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
            
            self.input_names_train = np.array(data_set.Agents)
            
            self.t_e_quantile = self.data_set.p_quantile
                
            
            self.num_samples_path_pred = self.data_set.num_samples_path_pred
            self.evaluate_on_train_set = evaluate_on_train_set
    
            if behavior == None:
                self.is_data_transformer = False
                
                if hasattr(self.splitter, 'Train_index'):
                    self.Index_train = self.splitter.Train_index
                
                    if self.get_output_type() == 'path_all_wi_pov':
                        pred_string = 'pred_tra_wip_'
                    elif self.get_output_type() == 'path_all_wo_pov':
                        pred_string = 'pred_tra_wop_'
                    elif self.get_output_type() == 'class':
                        pred_string = 'pred_classified'
                    elif self.get_output_type() == 'class_and_time':
                        pred_string = 'pred_class_time'
                    else:
                        raise AttributeError("This type of prediction is not implemented")
                        
                    self.model_file = data_set.change_result_directory(splitter.split_file,
                                                                       'Models', self.get_name()['file'])
                    self.pred_file  = data_set.change_result_directory(self.model_file,
                                                                       'Predictions', pred_string)
                    
                    if '_pert=' in self.model_file:
                        pert_split = self.model_file.split('_pert=')
                        self.model_file = pert_split[0] + '_pert=' + pert_split[1][0] + pert_split[1][2:]
                        
                    self.simply_load_results = False
                else:
                    self.simply_load_results = True
                
            else:
                self.is_data_transformer = True
                self.simply_load_results = False
                
                self.Index_train = np.where(data_set.Output_A[behavior])[0]
                self.Index_test  = np.arange(len(data_set.Output_A))
                
                if len(self.Index_train) == 0:
                    # There are no samples of the required behavior class
                    # Use those cases where the other behaviors are the furthers away
                    num_long_index = int(len(data_set.Output_A) / len(data_set.Behaviors))
                    self.Index_train = np.argsort(data_set.Output_T_E)[-num_long_index :]
                
                self.model_file = data_set.data_file[:-4] + '-transform_path_(' + behavior + ').npy'
                self.pred_file = self.model_file[:-4] + '-pred_tra_wip_.npy'
            
            self.setup_method()
            
            # Set trained to flase, this prevents a prediction on an untrained model
            self.trained = False
            self.extracted_data = False
            self.depict_results = False
        else:
            self.depict_results = True
        
        
    
    def train(self):
        assert not self.simply_load_results, 'This model instance is nonly for loading results.'
        self.model_mode = 'train'
        if os.path.isfile(self.model_file) and not self.model_overwrite:
            self.weights_saved = list(np.load(self.model_file, allow_pickle = True)[:-1])
            self.load_method()
        
        else:
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
           
        self.trained = True
        
        print('')
        print('The model ' + self.get_name()['print'] + ' was successfully trained.')
        print('')
        return self.model_file
        
    def predict_actual(self, Index = None):
        if Index is not None:
            self.Index_test = Index

        # apply model to test samples
        if self.get_output_type()[:4] == 'path':
            self.create_empty_output_path()
            self.predict_method()
            output = [self.Index_test, self.Output_path_pred]
        elif self.get_output_type() == 'class':
            self.create_empty_output_A()
            self.predict_method()
            output = [self.Index_test, self.Output_A_pred]
        elif self.get_output_type() == 'class_and_time':
            self.create_empty_output_A()
            self.create_empty_output_T()
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



    def Sort_Metrics(self, Metric_name_list, print_status_function):
        # Check if train and test set are identical
        identical_test_set = np.array_equal(np.unique(self.splitter.Train_index), np.unique(self.splitter.Test_index))

        # Go through Metric_list and check if any of them are allready evaluated, if so, remove them from the list
        Metric_train_list = []
        Metric_test_list  = []
        for metric_name in Metric_name_list:
            # Get metric class
            metric_module = importlib.import_module(metric_name)
            metric_class = getattr(metric_module, metric_name)  
            
            # Initialize the metric
            metric = metric_class(self.data_set, self.splitter, self)
                
            # Test if metric is applicable
            metric_failure = metric.check_applicability()
        
            # print metric status to output
            print_status_function(metric, metric_failure)
            
            # Do not use metric if it cannot be applied
            if metric_failure is not None:
                continue

            # Get metric save file
            metric_file = metric.data_set.change_result_directory(self.model_file, 'Metrics', metric.get_name()['file'])

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

        # Go through needed metrics and sort by metric type
        for mode, Metric_list in Metric_dict.item():
            Metric_type_dict = {}
            for metric in Metric_list:
                metric_type = metric.get_output_type()
                if metric_type not in Metric_type_dict.keys():
                    Metric_type_dict[metric_type] = []
                Metric_type_dict[metric_type].append(metric)
            
            Metric_dict[mode] = Metric_type_dict
        
        return Metric_dict, identical_test_set


    def get_index_df(self, Index_all):
        Index_df = pd.DataFrame(np.zeros((0,2), object), columns = ['Index', 'file_index'])

        if self.data_set.data_in_one_piece:
            # get the curent train/test index
            num_samples_file = len(self.data_set.Domain)
            useful = np.in1d(np.arange(num_samples_file), Index_all)
            Index_file = np.where(useful)[0]

            # Get the N_data from the current file
            data_type_index = np.unique(self.data_set.Domain.iloc[Index_file].data_type_index)
            assert len(data_type_index) == 1, 'There should only be one data type index per file.'
            
            # Predict the number of origdata size needed for output
            parts_needed_rel = ((2 * self.num_timesteps_out * self.num_samples_path_pred) / 
                                (2 * self.num_timesteps_out + self.num_timesteps_in * len(data_type_index[0])))
            
            parts_needed = int(np.ceil(1.2 * parts_needed_rel * len(Index_file) / num_samples_file)) # The 1.2 is a safety margin

            # Get number of samples that can actually be predicted
            Index_length = int(np.ceil(len(Index_file) / parts_needed))

            for i_part in range(parts_needed):
                Index = Index_file[i_part * Index_length : min((i_part + 1) * Index_length, len(Index_file))]

                Index_series = pd.Series({'Index': Index, 'file_index': 0})
                Index_df = Index_df.append(Index_series, ignore_index = True)

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

                # Get the N_data from the current file
                data_type_index = np.unique(self.data_set.Domain.iloc[Index_file].data_type_index)
                assert len(data_type_index) == 1, 'There should only be one data type index per file.'
                
                # Predict the number of origdata size needed for output
                parts_needed_rel = ((2 * self.num_timesteps_out * self.num_samples_path_pred) / 
                                    (2 * self.num_timesteps_out + self.num_timesteps_in * len(data_type_index[0])))
                
                parts_needed = int(np.ceil(1.2 * parts_needed_rel * len(Index_file) / num_samples_file)) # The 1.2 is a safety margin

                # Get number of samples that can actually be predicted
                Index_length = int(np.ceil(len(Index_file) / parts_needed))

                for i_part in range(parts_needed):
                    Index = Index_file[i_part * Index_length : min((i_part + 1) * Index_length, len(Index_file))]

                    Index_series = pd.Series({'Index': Index, 'file_index': file_index})
                    Index_df = Index_df.append(Index_series, ignore_index = True)

        return Index_df


    def predict_and_evaluate(self, Metric_name_list, print_status_function):
        assert not self.depict_results, 'This model instance is only for loading results.'
        assert self.data_set is not None, 'This model instance is only for loading results.'

        # Preselct the metrics based on their existence and other requirements
        Metric_dict, identical_test_set = self.Sort_Metrics(Metric_name_list, print_status_function)

        # Get the type of prediction in this output
        model_type = self.get_output_type()

        # Go through needed metrics and sort by metric type
        for mode, Metric_type_dict in Metric_dict.item():
            if mode == 'Train':
                Index_all = metric.splitter.Train_index
                result_index = 0
            else:
                Index_all = metric.splitter.Test_index
                result_index = 1

            # Get the index dataframe
            Index_df = self.get_index_df(Index_all)

            # Get the memory saving mode of the metrics

            # TODO: Metrics that cannot be split require that len(Index_df) == 1
            
            for i_index in range(len(Index_df)):
                Index = Index_df.iloc[i_index].Index
                file_index = Index_df.iloc[i_index].file_index

                # Reset prediction analysis
                self.reset_prediction_analysis()

                # Get original data
                self.data_set._extract_original_trajectories(file_index = file_index)

                # Make predictions on the given testing set
                output = self.predict_actual(Index) 

                for metric_type, Metric_list in Metric_type_dict.items():
                        
                    # Allow for possible transformation of prediction
                    Output_pred_trans = self.data_set.transform_outputs(output, model_type, metric_type, self.pred_file)
                

                    # TODO:
                    # Allow for disableing of the saving of the predictions

                    # TODO:
                    # Potentially, allow for the saving of the KDE models 

                    # TODO:
                    # Adjust the collection of the output metric accordingly

                    for metric in Metric_list:
                        # Get the metric file name
                        metric_file = metric.data_set.change_result_directory(self.model_file, 'Metrics', metric.get_name()['file'])

                        if os.path.isfile(metric_file) and not metric.metric_override:
                            Results = list(np.load(metric_file, allow_pickle = True)[:-1])
                        else:
                            Results = [None, None]

                        # Evaluate metric
                        results = metric._evaluate_on_subset(Output_pred_trans, Index)

                        # Overwrite Results
                        if not identical_test_set:
                            Results[result_index] = results
                        else:
                            Results = [results, results]

                        # Save results
                        save_data = np.array(Results + [0], object) # 0 is there to avoid some numpy load and save errros
                        os.makedirs(os.path.dirname(metric_file), exist_ok=True)
                        np.save(metric_file, save_data)



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
        available_memory = psutil.virtual_memory().total - psutil.virtual_memory().used

        # Calculate required memory (img_needed is main culprit, is a u_int8 datatype, X is float32)
        img_size = Img_needed.sum() * self.target_width * self.target_height
        if  not self.grayscale:
            img_size *= 3
        required_memory = np.prod(X.shape) * 4 + img_size

        # Check if enough memory is available
        if required_memory < 0.4 * available_memory:
            # Only use the input positions
            X = X[..., :2]
            centre = X[Img_needed, -1,:]
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


    def extract_sceneGraphs(self, domain_needed):
        '''
        Returns scene graph data 
        '''
        # TODO: Implement pre-estimation of needed memory
        graph_needed = self.data_set.return_batch_sceneGraphs(domain_needed)
        use_batch_extraction = False
        
        return graph_needed, use_batch_extraction

    def _extract_types(self):
        ## NOTE: Method has been adjusted for large datasets
        if not hasattr(self, 'Type'):
            if self.data_set.data_in_one_piece:
                # Get agent types
                T = self.data_set.Type.to_numpy()
                T = T.astype(str)
                T[T == 'nan'] = '0'
            else:
                T = np.zeros(self.data_set.Pred_agents_pred.shape, str)
                for file_index in range(len(self.data_set.Files)):
                    used = self.data_set.Domain.file_index == file_index
                    used_index = np.where(used)[0]
                    
                    agent_file = self.data_set.Files[file_index] + '_AM.npy'
                    T_local, _, _ = np.load(agent_file, allow_pickle = True)
                    T_local = T_local.astype(str)
                    T_local[T_local == 'nan'] = '0'
                    
                    T[used] = T_local
                
            self.Type = T.astype(str)
    

    def get_orig_data_index(self, Sample_ind, Agent_ind = None, mode = 'scipy'):
        # assert that sample_ind includes only integers
        assert Sample_ind.dtype == np.array([1]).dtype, 'Sample index should be integers.'

        Out_shape = Sample_ind.shape
        if Agent_ind is None:
            assert len(Out_shape) == 1, 'If no agent index is given, the sample index should be 1D.'

            # Get unique sample indices
            Sample_ind_unique, Sample_ind_inverse = np.unique(Sample_ind, return_inverse = True)

            # Find original data in these indices
            samples_included = np.in1d(self.data_set.Used_samples, Sample_ind_unique)

            result = np.where(samples_included)[0]
            used_samples = self.data_set.Used_samples[samples_included]
            used_agents  = self.data_set.Used_agents[samples_included]

            # Get inverse of used samples
            index_inverse = np.zeros(Sample_ind_unique.max() + 1, int)
            index_inverse[Sample_ind_unique] = np.arange(len(Sample_ind_unique), dtype = int)

            mask = np.zeros((len(Sample_ind_unique), len(self.data_set.Agents)), bool)
            mask[index_inverse[used_samples], used_agents] = True

            # Get the original data index
            result = result[Sample_ind_inverse]
            mask   = mask[Sample_ind_inverse]

        else:
            assert Sample_ind.shape == Agent_ind.shape, 'Sample and Agent index should have the same shape.'

            # Flatten the data
            Sample_ind = Sample_ind.flatten()
            Agent_ind  = Agent_ind.flatten()

            if mode == 'pandas':
                # Create orginal multiindex pandas dataframe
                Used_df = pd.DataFrame({'row': self.data_set.Used_samples,
                                        'col': self.data_set.Used_agents,
                                        'data': np.arange(len(self.data_set.Used_samples), dtype=int)})['data']
                
                # Create multiindex for the desired data
                lookup_df = pd.DataFrame({'row': Sample_ind, 'col': Agent_ind}).set_index(['row', 'col']).index

                # Perform the lookup
                result = lookup_df.map(Used_df).values.to_numpy().astype(float)
                mask = np.isfinite(result)
                result = result.astype(int)
            elif mode == 'scipy':
                sparse_matrix = sp.sparse.coo_matrix((np.arange(len(self.data_set.Used_samples), dtype=int) + 1, 
                                                    (self.data_set.Used_samples, self.data_set.Used_agents)),
                                                    shape = (max(self.data_set.Used_samples.max(), Sample_ind.max()) + 1,
                                                            max(self.data_set.Used_agents.max(), Agent_ind.max()) + 1))
                
                # Convert to csr for more efficient lookup
                sparse_matrix = sparse_matrix.tocsr()

                result = sparse_matrix[Sample_ind, Agent_ind].A1
                mask = result != 0
                result -= 1
            else:
                raise AttributeError('Mode ' + mode + ' not implemented.')

            # Transfrom result to original shape
            result = result[mask]
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
        
        if not self.extracted_data:
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
                
                useful_agents = np.zeros(max_num_agents, bool)

                for file_index in range(len(self.Files)):
                    used = self.Domain.file_index == file_index
                    used_index = np.where(used)[0]

                    Pred_agents_pred_local = self.Pred_agents_pred[used_index]
                    
                    # Get the original data
                    self.data_set._extract_original_trajectories(self, file_index = 0)
                    
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
                    useful_agents = np.unqiue(np.concatenate((useful_agents, useful_agents_local), 0))
            
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
            
            # Get the number of future timesteps
            self.N_O_pred = N_O_pred[Sample_id[:,0]]
            self.N_O_data = N_O_data[Sample_id[:,0]]
            
            if self.data_set.data_in_one_piece:
                # Get trajectory data
                self.Data_index, self.Data_index_mask = self.get_orig_data_index(Sample_id, Agent_id) 
                # self.Data_index.shape = self.Data_index_mask.sum()
                # self.Data_index_mask.shape = num_samples, num_agents

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

                    # Transform self.Data_index to an array
                    Data_index_array = np.full(self.Data_index_mask.shape, -1, int)
                    Data_index_array[self.Data_index_mask] = self.Data_index

                    Data_index_needed = Data_index_array[Img_needed]
                    Img_needed_inside = np.ones(len(Data_index_needed), bool)

                    self.img, self.img_m_per_px, self.use_batch_extraction = self.extract_images(self.data_set.X_orig[Data_index_needed, ..., :2].astype(np.float32), 
                                                                                                 Img_needed_inside, domain_needed)
                    self.img_needed_sample = np.where(Img_needed)[0]

                else:
                    self.img = None
                    self.img_m_per_px = None


                # Get graphs
                if use_graph:
                    # Get metadata
                    domain_old = self.data_set.Domain
                    Graph_needed = np.zeros(self.X.shape[:2], bool)
                    Graph_needed[:,0] = True

                    domain_needed = domain_old.iloc[self.ID[Graph_needed][:,0]]
                    self.graph, self.use_graph_batch_extraction = self.extract_sceneGraphs(domain_needed)
                    self.graph_needed_sample = np.where(Graph_needed)[0]
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
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        img : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
            It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
            of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
            at its current position, and are rotated so that the agent is right now driving to the right. 
            If an agent is not observed at prediction time, 0 values are returned.
        img_m_per_px : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
            the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
            Both for **Y**, **img** and 
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
        Pred_agents_train = self.Pred_agents[I_train]
        
        if self.img is not None:
            use_images = np.in1d(self.img_needed_sample, I_train)

            img_needed          = self.img[use_images]
            img_m_per_px_needed = self.img_m_per_px[use_images]

            if self.predict_single_agent:
                Img_needed = np.zeros(X_train.shape[:2], bool)
                Img_needed[:,0] = True
            else:
                Img_needed = T_train != '0'

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
            use_graphs = np.in1d(self.graph_needed_sample, I_train)
            graph_needed = self.graph[use_graphs]
            if self.predict_single_agent:
                Graph_needed = np.zeros(X_train.shape[:2], bool)
                Graph_needed[:,0] = True
            else:
                Graph_needed = T_train != '0'
            
            graph_train = np.zeros((*Graph_needed.shape, graph_needed.shape[-1]), np.float32)
            graph_train[Graph_needed] = graph_needed
        
        Sample_id_train = self.ID[I_train,0,0]
        Agents = np.array(self.input_names_train)
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

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C_train = C.to_numpy().astype(int)
            else:
                C_train = None

            return [X_train, Y_train, T_train, C_train, img_train, img_m_per_px_train, graph_train,
                    Pred_agents_train, Sample_id_train, Agent_id_train]
        else:
            return [X_train, Y_train, T_train, img_train, img_m_per_px_train, graph_train,
                    Pred_agents_train, Sample_id_train, Agent_id_train]
        
        
    def _update_available_samples(self, Ind_advance, ind_advance):
        ## NOTE: Method has been adjusted for large datasets
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
    
    
    def provide_batch_data(self, mode, batch_size, val_split_size = 0.0, ignore_map = False, ignore_graph = False, return_categories = False):
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
        ignore_map : ignore_map, optional
            This indicates if image data is not needed, even if available in the dataset 
            and processable by the model. The default is *False*.

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
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        C : np.ndarray
            TODO
        img : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
            It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
            of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
            at its current position, and are rotated so that the agent is right now driving to the right. 
            If an agent is not observed at prediction time, 0 values are returned.
        img_m_per_px : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
            the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
            Both for **Y**, **img** and 
        graph : TODO
        Pred_agents : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
            if it expected by the framework that a prediction will be made for the specific agent.
            
            If only one agent has to be predicted per sample, for **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
            be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.
        num_steps : int
            This is the number of future timesteps provided in the case of traning in expected in the case of prediction. In the 
            former case, it has the value :math:`N_{O}`.
        Sample_id : np.ndarray, optional
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted. This value is only returned for **mode** = *'pred'*.
        Agent_id : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted.. This value is only returned for **mode** = *'pred'*.
        epoch_done : bool
            This indicates wether one has just sampled all batches from an epoch and has to go to the next one.

        '''
        
        self.prepare_batch_generation()
        
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
                self.Ind_val = [I_train[num_train:], np.array([], int)]
                
            N_O = np.minimum(self.N_O_data, self.max_t_O_train)
            Ind_advance = self.Ind_val
            
        elif mode == 'train':
            assert self.model_mode == 'train', 'During training, the non-validation part of the training set should be called.'
            if not hasattr(self, 'Ind_train'):
                I_train = self._extract_useful_training_samples()
                num_train = int(len(I_train) * (1 - val_split_size))
                self.Ind_train = [I_train[:num_train], np.array([], int)]
            
            N_O = np.minimum(self.N_O_data, self.max_t_O_train)
            Ind_advance = self.Ind_train
        
        else:
            raise TypeError("Unknown mode.")
        
        # Get data needed for selecting batch from available
        Sample_id_advance  = self.ID[Ind_advance[0],0,0]
        File_index_advance = self.data_set.Domain.file_index.iloc[Sample_id_advance].to_numpy() # File index
        Image_id_advance   = self.data_set.Domain.image_id.iloc[Sample_id_advance].to_numpy() # Image id
        N_O_advance = N_O[Ind_advance[0]]   # Number of timesteps

        # Check for file index and image id
        Use_candidate = Image_id_advance == Image_id_advance[0]

        # For large dataset, check for file index as well
        if self.data_set.data_in_one_piece:
            Use_candidate &= File_index_advance == File_index_advance[0]
        
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
        
        # Update the used samples in this epoch
        epoch_done, Ind_advance = self._update_available_samples(Ind_advance, ind_advance) 

        ## Prepare the data to be returned
        # get num_steps
        num_steps = N_O[ind_advance].min()
        
        # Get data as available for whole dataset
        T = self.T[ind_advance]
        Pred_agents = self.Pred_agents[ind_advance]

        # Get the corresponding sample_ids 
        Sample_id = self.ID[ind_advance,:,0]
        Agent_id  = self.ID[ind_advance,:,1]

        # Prepare the output arrays
        X = np.full((len(ind_advance), self.ID.shape[1], self.data_set.X_orig.shape[-2], self.data_set.X_orig.shape[-1]), np.nan, np.float32)
        Y = np.full((len(ind_advance), self.ID.shape[1], num_steps, 2), np.nan, np.float32)

        # Get data that is potentially not available yet
        if self.data_set.data_in_one_piece:
            Data_index_array = np.full(self.Data_index_mask.shape, -1, int)
            Data_index_array[self.Data_index_mask] = self.Data_index

            data_index_array = Data_index_array[ind_advance]
            data_index_mask  = self.Data_index_mask[ind_advance]

            data_index = data_index_array[data_index_mask]

            X[data_index_mask] = self.data_set.X_orig[data_index]
            Y[data_index_mask] = self.data_set.Y_orig[data_index, :num_steps]
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
            Sample_id_used = Sample_id[:,[0]] == used_index[np.newaxis, :]

            assert (Sample_id_used.sum(1) == 1).all(), "Something went wrong with finding samples"
            Sample_id_used = np.tile(Sample_id_used.argmax(1)[:,np.newaxis], (1, Sample_id.shape[1]))

            # Get original indices
            Data_index, Data_index_mask = self.get_orig_data_index(Sample_id_used, Agent_id)

            X[Data_index_mask] = self.data_set.X_orig[Data_index].astype(np.float32)
            Y[Data_index_mask] = self.data_set.Y_orig[Data_index, :num_steps].astype(np.float32)

        # Get the corresponding input_data_type
        input_data_type_indices = self.data_set.Domain.iloc[Sample_id[:,0]].data_type_index
        assert len(np.unique(input_data_type_indices)) == 1, 'Only one data type should be used in each batch'
        self.input_data_type = self.data_set.Input_data_type[input_data_type_indices.iloc[0]]

        # Check if images need to be extracted
        if hasattr(self, 'use_batch_extraction') and (not ignore_map):
            if self.predict_single_agent:
                Img_needed = np.zeros(X.shape[:2], bool)
                Img_needed[:,0] = True
            else:
                Img_needed = T != '0'

            if self.use_batch_extraction:
                domain_needed = self.data_set.Domain.iloc[self.ID[ind_advance,:,0][Img_needed]]
                img_needed, img_m_per_px_needed, unsuccesful = self.extract_images(X, Img_needed, domain_needed)
                if unsuccesful:
                    raise MemoryError('Not enough memory to extract images even with batches. Consider using grey scale images or a smaller resolution.' +
                                      '\nNote, however, that other errors might have caused this as well.')
            else:
                # Find at which places in self.img_needed_sample one can find ind_advance
                use_images = np.in1d(self.img_needed_sample, ind_advance)

                img_needed          = self.img[use_images]
                img_m_per_px_needed = self.img_m_per_px[use_images]

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


        # Check if graphs need to be extracted
        if hasattr(self, 'use_batch_extraction') and (not ignore_graph) and self.has_graph:

            Graph_needed = np.zeros(X.shape[:2], bool)
            Graph_needed[:,0] = True

            if self.use_batch_extraction:
                domain_needed = self.data_set.Domain.iloc[self.ID[ind_advance,:,0][Graph_needed]]
                graph_needed, unsuccesful = self.extract_sceneGraphs(domain_needed)
                if unsuccesful:
                    MemoryError('Not enough memory to extract graphs even with batches.' )
            else:
                # Find at which places in self.graph_needed_sample one can find ind_advance
                use_graphs = np.in1d(self.graph_needed_sample, ind_advance)

                graph_needed          = self.graph[use_graphs]

            Graph_needed = Graph_needed[:,:1]
            
            # Transfrom graph needed back according to Graph_needed into required format
            graph          = np.full((Graph_needed.shape), np.nan, dtype=object)

            graph[Graph_needed]          = graph_needed

        else:
            graph          = None

        # check if epoch is completed, if so, shuffle and reset index
        Sample_id = Sample_id[:,0]

        if return_categories:
            if 'category' in self.data_set.Domain.columns:
                Agent_id = self.ID[ind_advance,:,1]
                
                C = self.data_set.Domain.category.iloc[Sample_id]
                C = pd.DataFrame(C.to_list())

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C = C.to_numpy().astype(int)
                C = C[:, Agent_id]

            else:
                C = None
            if mode == 'pred':
                Agents = np.array(self.input_names_train)
                Agent_id = Agents[self.ID[ind_advance,:,1]]
                return X,    T, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
            else:
                return X, Y, T, C, img, img_m_per_px, graph, Pred_agents, num_steps,                      epoch_done
        else:
            if mode == 'pred':
                Agents = np.array(self.input_names_train)
                Agent_id = Agents[self.ID[ind_advance,:,1]]
                return X,    T, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
            else:
                return X, Y, T, img, img_m_per_px, graph, Pred_agents, num_steps,                      epoch_done
    
    
    def save_predicted_batch_data(self, Pred, Sample_id, Agent_id, Pred_agents = None):
        r'''

        Parameters
        ----------
        Pred : np.ndarray
            This is the predicted future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or on some timesteps partially not observed, then this can include np.nan values. 
            The required value of :math:`N_{preds}` is given in **self.num_samples_path_pred**.
        Sample_id : np.ndarray, optional
            This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
            in the dataset this sample was extracted.
        Agent_id : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
            original agent in the dataset this agent was extracted.
        Pred_agents : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
            if it expected by the framework that a prediction will be made for the specific agent.
            
            This input does not have to be provided if the model can only predict one single agent at the same time and is therefore
            incaable of joint predictions.

        Returns
        -------
        None.

        '''
        assert self.get_output_type()[:4] == 'path'
        if self.predict_single_agent:
            if len(Pred.shape) == 4:
                Pred = Pred[:,np.newaxis]
                
            assert Pred.shape[:1] == Sample_id.shape
            if Pred_agents is None:
                Pred_agents = np.zeros(Agent_id.shape, bool)
                Pred_agents[:,0] = True
        else:
            assert Pred_agents is not None
            assert Pred.shape[:2] == Agent_id.shape
            assert Pred_agents.shape == Agent_id.shape
        
        assert Sample_id.shape == Agent_id.shape[:1]
        assert Pred.shape[2] == self.num_samples_path_pred
        assert Pred.shape[4] == 2
        
        for i, i_sample in enumerate(Sample_id):
            for j, agent in enumerate(Agent_id[i]):
                if not Pred_agents[i,j]:
                    continue
                self.Output_path_pred.loc[i_sample][agent] = None
                self.Output_path_pred.loc[i_sample][agent] = Pred[i, j,:, :, :].astype('float32')
    
    
    def get_classification_data(self, train = True, return_categories = False):
        r'''
        This function retuns inputs and outputs for classification models.

        Parameters
        ----------
        train : bool, optional
            This discribes whether one wants to generate training or testing data. The default is True.

        return_categories : bool, optional
            This discribes whether one wants to return the categories of the samples. The default is False.

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
        else:
            D = np.zeros((len(self.Domain),0), dtype = np.float32)
        
        # Select current samples
        if train:
            assert self.model_mode == 'train', 'During training, training set should be called.'
            Index = self.Index_train
        else:
            assert self.model_mode == 'pred', 'During training, training set should be called.'
            Index = self.Index_test

        # Get other inputs
        T = self.Type[Index]
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
        
        class_names = self.data_set.Output_A.columns
        agent_names = self.data_set.Input_path.columns
        dist_names = self.data_set.Input_prediction.columns
        

        if return_categories:
            if 'category' in self.data_set.Domain.columns:
                C = self.data_set.Domain.category.iloc[Index]
                C = pd.DataFrame(C.to_list())

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C = C.to_numpy()
            else:
                C = None
            
            if train:
                return X, T, C, agent_names, D, dist_names, class_names, P, DT
            else:
                return X, T, C, agent_names, D, dist_names, class_names
        else:
            if train:
                return X, T, agent_names, D, dist_names, class_names, P, DT
            else:
                return X, T, agent_names, D, dist_names, class_names
    
    
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
        
    
    
    def create_empty_output_path(self):
        Agents = np.array(self.data_set.Output_path.columns)
        
        Index_test = self.Index_test
        num_rows = len(Index_test)
        
        self.Output_path_pred = pd.DataFrame(np.empty((num_rows, len(Agents)), np.ndarray), 
                                             columns = Agents, index = Index_test)
        
    
    
    def create_empty_output_A(self):
        Behaviors = np.array(self.data_set.Behaviors)
        
        Index_test = self.Index_test
        num_rows = len(Index_test)
            
        self.Output_A_pred = pd.DataFrame(np.zeros((num_rows, len(Behaviors)), float), 
                                          columns = Behaviors, index = Index_test)
        
    
    def create_empty_output_T(self):
        Behaviors = np.array(self.data_set.Behaviors)
        
        Index_test = self.Index_test
        num_rows = len(Index_test)
            
        self.Output_T_E_pred = pd.DataFrame(np.empty((num_rows, len(Behaviors)), np.ndarray), 
                                            columns = Behaviors, index = Index_test)
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
    
    # %% Method needed for evaluation_template
    def _transform_predictions_to_numpy(self, Pred_index, Output_path_pred, 
                                        exclude_ego = False, exclude_late_timesteps = True):
        if hasattr(self, 'Path_pred') and hasattr(self, 'Path_true') and hasattr(self, 'Pred_step'):
            if self.excluded_ego == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego = exclude_ego
        
        # Check if data is there
        self.data_set._extract_original_trajectories()
        self.data_set._determine_pred_agents(eval_pov = ~exclude_ego)
        
        # Initialize output
        num_samples = len(Output_path_pred)
        
        # Get predicted timesteps
        nto = self.num_timesteps_out
        if exclude_late_timesteps:
            nto_max = nto
        else:
            nto_max = self.data_set.N_O_data_orig[Pred_index].max()
        
        Nto_i = np.minimum(nto_max, self.data_set.N_O_data_orig[Pred_index])

        # Get pred agents
        max_num_pred_agents = self.data_set.Pred_agents_eval.sum(1).max()
        Pred_agents = self.data_set.Pred_agents_eval[Pred_index]
        
        
        i_agent_sort = np.argsort(-Pred_agents.astype(float))
        i_agent_sort = i_agent_sort[:,:max_num_pred_agents]
        i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_pred_agents))
        
        # Get true predictions
        data_index, data_mask = self.get_orig_data_index(Pred_index)
        self.Path_true = np.full((*Pred_agents.shape, *self.data_set.Y_orig.shape[-2:]), np.nan, np.float32)
        self.Path_true[data_mask] = self.data_set.Y_orig[data_index]

        # Apply sorting index
        self.Path_true = self.Path_true[i_sampl_sort, i_agent_sort, :nto_max]

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
        
        for i in range(num_samples):
            pred_agents = np.where(self.Pred_step[i].any(-1))[0]
            
            # Avoid useless samples 
            if len(pred_agents) == 0:
                continue

            pred_agents_id = i_agent_sort[i, pred_agents]
            path_pred_orig = Output_path_pred.iloc[i, pred_agents_id]
            path_pred = np.stack(path_pred_orig.to_numpy(), axis = 1)
        
            nto_i = min(Nto_i[i], path_pred.shape[2])
            # Assign to full length label
            self.Path_pred[:,i, pred_agents, :nto_i] = path_pred[:,:,:nto_i]
        
        # Set missing values to zero
        self.Path_pred[:,~self.Pred_step,:] = 0.0 
        #Transpose inot right order
        self.Path_pred = self.Path_pred.transpose(1,0,2,3,4)
        
        # Get agent predictions
        self.T_pred = self.data_set.Type.iloc[Pred_index].to_numpy()
        self.T_pred = self.T_pred.astype(str)
        self.T_pred[self.T_pred == 'nan'] = '0'
        self.T_pred = self.T_pred[i_sampl_sort, i_agent_sort]

        # Set agent types of agents not included in Pred step to '0'
        self.T_pred[~self.Pred_step.any(-1)] = '0'

        # Get agent predictions
        if 'category' in self.data_set.Domain.columns:
            # Sample_id = self.ID[ind_advance,0,0]
            C = self.data_set.Domain.category
            C = pd.DataFrame(C.to_list())

            # Replace missing agents
            C = C.fillna(4)

            # Get to numpy and apply indices
            C = C.to_numpy().astype(int)
            self.C_pred = C[i_sampl_sort, i_agent_sort]
    
    
    def _get_joint_KDE_pred_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_joint_pred') and hasattr(self, 'Log_prob_joint_true'):
            if self.excluded_ego_joint == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego_joint = exclude_ego
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds
        self.Log_prob_joint_true = np.zeros(self.Path_true.shape[:-3], dtype = np.float32)
        self.Log_prob_joint_pred = np.zeros(self.Path_pred.shape[:-3], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
        
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = ~exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        
        print('Calculate joint PDF on predicted probabilities.', flush = True)
        for i, subgroup in enumerate(np.unique(Subgroups)):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(i + 1, len(np.unique(Subgroups))), flush = True)
            subgroup_index = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[subgroup_index], axis = 0)) == 1
            pred_agents = Pred_agents[subgroup_index[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            nto_subgroup = Num_steps[subgroup_index]
            
            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                nto_index = subgroup_index[np.where(nto == nto_subgroup)[0]]
                
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
                
                # Only use select number of samples for training kde
                use_preds = np.arange(len(paths_pred_comp))
                np.random.seed(0)
                np.random.shuffle(use_preds)
                max_preds = min(max(1 * len(paths_true_comp), 5 * self.num_samples_path_pred, 2500), len(use_preds))
                
                # Get approximated probability distribution
                log_probs_true = []
                log_probs_pred = []
                log_pred_satisfied = False
                i = 0
                while not log_pred_satisfied and (i + 1) * max_preds <= len(use_preds):
                    test_ind = use_preds[i * max_preds : (i + 1) * max_preds]
                    path_pred_train = paths_pred_comp[test_ind]

                    kde = ROME().fit(path_pred_train)

                    # Score samples
                    log_probs_true.append(kde.score_samples(paths_true_comp))
                    log_probs_pred.append(kde.score_samples(paths_pred_comp))
                    
                    weights = np.ones((i + 1, 1)) / (i + 1)
                    # Check if we sufficiently represent predicted distribution
                    print('            ' + str(i))
                    i += 1


                log_prob_pred = sp.special.logsumexp(np.stack(log_probs_pred, axis = 0), 
                                                                b = weights, axis = 0)
                    # unincluded = use_preds[max_preds * (i + 1):]
                    # if len(unincluded) == 0:
                    #     log_pred_satisfied = True
                    # else:
                    #     included = use_preds[:max_preds * (i + 1)]

                    #     included_quant = np.quantile(log_prob_pred[included], [0.1, 0.3, 0.5, 0.7, 0.9])
                    #     unincluded_quant = np.quantile(log_prob_pred[unincluded], [0.1, 0.3, 0.5, 0.7, 0.9])

                    #     diff = np.abs(included_quant - unincluded_quant)

                    #     log_pred_satisfied = np.max(diff) < 0.5
                    #     print(np.max(diff))

                
                log_prob_true = sp.special.logsumexp(np.stack(log_probs_true, axis = 0), 
                                                            b = weights, axis = 0)
            
                self.Log_prob_joint_true[nto_index] = log_prob_true.reshape(*paths_true.shape[:2])
                self.Log_prob_joint_pred[nto_index] = log_prob_pred.reshape(*paths_pred.shape[:2])
            
            
    def _get_indep_KDE_pred_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_indep_pred') and hasattr(self, 'Log_prob_indep_true'):
            if self.excluded_ego_indep == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego_indep = exclude_ego
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds x num agents
        self.Log_prob_indep_true = np.zeros(self.Path_true.shape[:-2], dtype = np.float32)
        self.Log_prob_indep_pred = np.zeros(self.Path_pred.shape[:-2], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
       
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = ~exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        
        print('Calculate indep PDF on predicted probabilities.', flush = True)
        for i, subgroup in enumerate(np.unique(Subgroups)):
            print('    Subgroup {:5.0f}/{:5.0f}'.format(i + 1, len(np.unique(Subgroups))), flush = True)
            subgroup_index = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[subgroup_index], axis = 0)) == 1
            pred_agents = Pred_agents[subgroup_index[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            pred_agents_id = np.where(pred_agents)[0]
            
            nto_subgroup = Num_steps[subgroup_index]
            
            for i_nto, nto in enumerate(np.unique(nto_subgroup)):
                print('        Number output timesteps: {:3.0f} ({:3.0f}/{:3.0f})'.format(nto, i_nto + 1, len(np.unique(nto_subgroup))), flush = True)
                nto_index = subgroup_index[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = self.Path_true[nto_index][:,:,pred_agents,:nto]
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                
                num_features = nto * 2
                
                for i_agent, i_agent_orig in enumerate(pred_agents_id):
                    # Get agent
                    paths_true_agent = paths_true[:,:,i_agent]
                    paths_pred_agent = paths_pred[:,:,i_agent]
                
                    # Collapse agents
                    paths_true_agent_comp = paths_true_agent.reshape(*paths_true_agent.shape[:2], num_features)
                    paths_pred_agent_comp = paths_pred_agent.reshape(*paths_pred_agent.shape[:2], num_features)
                    
                    # Collapse agents further
                    paths_true_agent_comp = paths_true_agent_comp.reshape(-1, num_features)
                    paths_pred_agent_comp = paths_pred_agent_comp.reshape(-1, num_features)
                    
                    # Only use select number of samples for training kde
                    use_preds = np.arange(len(paths_pred_agent_comp))
                    np.random.seed(0)
                    np.random.shuffle(use_preds)
                    max_preds = min(max(1 * len(paths_true_agent_comp), 5 * self.num_samples_path_pred, 2500), len(use_preds))
                    
                    # Get approximated probability distribution
                    log_probs_true_agent = []
                    log_probs_pred_agent = []

                    log_pred_satisfied = False
                    i = 0
                    while not log_pred_satisfied and (i + 1) * max_preds <= len(use_preds):
                        test_ind = use_preds[i * max_preds : (i + 1) * max_preds]
                        path_pred_train = paths_pred_agent_comp[test_ind]
                        kde = ROME().fit(path_pred_train)
                        
                        # Score samples
                        log_probs_true_agent.append(kde.score_samples(paths_true_agent_comp))
                        log_probs_pred_agent.append(kde.score_samples(paths_pred_agent_comp))
                        
                        weights = np.ones((i + 1, 1)) / (i + 1)
                        # Check if we sufficiently represent predicted distribution
                        print('            ' + str(i))
                        i += 1


                    log_prob_pred_agent = sp.special.logsumexp(np.stack(log_probs_pred_agent, axis = 0), 
                                                                   b = weights, axis = 0)
                        
                        
                        # unincluded = use_preds[max_preds * (i + 1):]
                        # if len(unincluded) == 0:
                        #     log_pred_satisfied = True
                        # else:
                        #     included = use_preds[:max_preds * (i + 1)]

                        #     included_quant = np.quantile(log_prob_pred_agent[included], [0.1, 0.3, 0.5, 0.7, 0.9])
                        #     unincluded_quant = np.quantile(log_prob_pred_agent[unincluded], [0.1, 0.3, 0.5, 0.7, 0.9])

                        #     diff = np.abs(included_quant - unincluded_quant)

                        #     # We reject distribution as unequal with 99.9% confidence
                        #     t_value = 3.09 * log_prob_pred_agent.std() / np.sqrt(len(use_preds))

                        #     log_pred_satisfied = np.max(diff) < t_value
                        #     print(i, np.max(diff), t_value)
                        

                    
                    log_prob_true_agent = sp.special.logsumexp(np.stack(log_probs_true_agent, axis = 0), 
                                                               b = weights, axis = 0)

                    
                    self.Log_prob_indep_true[nto_index,:,i_agent_orig] = log_prob_true_agent.reshape(*paths_true.shape[:2])
                    self.Log_prob_indep_pred[nto_index,:,i_agent_orig] = log_prob_pred_agent.reshape(*paths_pred.shape[:2])
                
                
    def _get_joint_KDE_true_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_true_joint_pred'):
            if self.excluded_ego_true_joint == exclude_ego:
                return
        
        # Have the dataset load
        self.data_set._get_joint_KDE_probabilities(exclude_ego)
        
        # Save last setting 
        self.excluded_ego_true_joint = exclude_ego
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds
        self.Log_prob_true_joint_pred = np.zeros(self.Path_pred.shape[:-3], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
        
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = ~exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        
        for subgroup in np.unique(Subgroups):
            subgroup_index = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[subgroup_index], axis = 0)) == 1
            pred_agents = Pred_agents[subgroup_index[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue

            nto_subgroup = Num_steps[subgroup_index]
            
            for nto in np.unique(nto_subgroup):
                nto_index = subgroup_index[np.where(nto == nto_subgroup)[0]]
                
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
                return
        
        # Have the dataset load
        self.data_set._get_indep_KDE_probabilities(exclude_ego)
        
        # Save last setting 
        self.excluded_ego_true_indep = exclude_ego
        
        # Check if dataset has all valuable stuff
        self._transform_predictions_to_numpy(Pred_index, Output_path_pred, exclude_ego)
        
        # get predicted agents
        Pred_agents = self.Pred_step.any(-1)
        
        # Shape: Num_samples x num_preds x num agents
        self.Log_prob_true_indep_pred = np.zeros(self.Path_pred.shape[:-2], dtype = np.float32)
        
        Num_steps = self.Pred_step.sum(-1).max(-1)
       
        # Get identical input samples
        self.data_set._group_indentical_inputs(eval_pov = ~exclude_ego)
        Subgroups = self.data_set.Subgroups[Pred_index]
        
        for subgroup in np.unique(Subgroups):
            subgroup_index = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[subgroup_index], axis = 0)) == 1
            pred_agents = Pred_agents[subgroup_index[0]]
            
            # Avoid useless samples
            if not pred_agents.any():
                continue
            
            pred_agents_id = np.where(pred_agents)[0]
            
            nto_subgroup = Num_steps[subgroup_index]
            
            for nto in np.unique(nto_subgroup):
                nto_index = subgroup_index[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                
                num_features = nto * 2
                
                for i_agent, i_agent_orig in enumerate(pred_agents_id):
                    agent = self.input_names_train[i_agent_orig]
                    
                    # Get agent
                    paths_pred_agent = paths_pred[:,:,i_agent]
                
                    # Collapse agents
                    paths_pred_agent_comp = paths_pred_agent.reshape(*paths_pred_agent.shape[:2], num_features)
                    
                    # Collapse agents further
                    paths_pred_agent_comp = paths_pred_agent_comp.reshape(-1, num_features)
                    
                    log_prob_pred_agent = self.data_set.KDE_indep[subgroup][nto][agent].score_samples(paths_pred_agent_comp)

                    self.Log_prob_true_indep_pred[nto_index,:,i_agent_orig] = log_prob_pred_agent.reshape(*paths_pred.shape[:2])
                
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
        
