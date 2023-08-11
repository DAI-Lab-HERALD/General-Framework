import pandas as pd
import numpy as np
import os
import torch
import warnings

class model_template():
    def __init__(self, data_set, splitter, behavior = None): 
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
        
        
        
        if behavior == None:
            self.is_data_transformer = False
            
            Index = splitter.Train_index
            
            self.model_file = data_set.change_result_directory(splitter.split_file,
                                                               'Models', self.get_name()['file'])
            
            
            if self.get_output_type() == 'path_all_wi_pov':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_tra_wi_pov')
            
            elif self.get_output_type() == 'path_all_wo_pov':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_tra_wo_pov')
                
            elif self.get_output_type() == 'class':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_classified')
                
            elif self.get_output_type() == 'class_and_time':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_class_time')
            
            else:
                raise AttributeError("This type of prediction is not implemented")
            
            
        else:
            self.is_data_transformer = True
            Index = np.where(data_set.Output_A[behavior])[0]
            if len(Index) == 0:
                # There are no samples of the required behavior class
                # Use those cases where the other behaviors are the furthers away
                num_long_index = int(len(data_set.Output_A) / len(data_set.Behaviors))
                Index = np.argsort(data_set.Output_T_E)[-num_long_index :]
            
            save_file = data_set.data_file[:-4] + '-transform_path_(' + behavior + ').npy'
            self.model_file = save_file
            self.pred_file = self.model_file[:-4] + '-pred_tra_wi_pov.npy'
        
        self.data_set = data_set
        
        self.dt = self.data_set.dt
        
        self.input_names_train = data_set.Input_path.columns
        # Only provide one kind of input (if model cheats and tries to use both)
        if self.get_input_type()['past'] == 'general':
            self.Input_prediction_train = data_set.Input_prediction.iloc[Index]
            self.Input_path_train       = None
            
        elif self.get_input_type()['past'] == 'path':
            self.Input_prediction_train = None
            self.Input_path_train       = data_set.Input_path.iloc[Index]
            
        elif self.get_input_type()['past'] == 'both':
            self.Input_prediction_train = data_set.Input_prediction.iloc[Index]
            self.Input_path_train       = data_set.Input_path.iloc[Index]
        else:
            raise AttributeError("This kind of past input information is not implemented.")
            
        
        if self.get_input_type()['future']:
            Pov_ind = self.data_set.pov_agent == np.array(self.input_names_train)
            self.Input_future_train = data_set.Output_path.iloc[Index, Pov_ind]
        else:
            self.Input_future_train = data_set.Output_path.iloc[Index, np.zeros(len(self.input_names_train), bool)]
            
        # check if model is allowed
        
        self.Input_T_train           = data_set.Input_T[Index]
        
        self.Output_path_train       = data_set.Output_path.iloc[Index]
        self.Output_T_train          = data_set.Output_T[Index]
        self.Output_T_pred_train     = data_set.Output_T_pred[Index]
        self.Output_A_train          = data_set.Output_A.iloc[Index]
        self.Output_T_E_train        = data_set.Output_T_E[Index]
        
        self.Type_train              = data_set.Type.iloc[Index]
        self.Domain_train            = data_set.Domain.iloc[Index]
        
        self.num_samples_path_pred = self.data_set.num_samples_path_pred
        
        self.num_samples_train = len(Index)
        
        # Find Pred_agents
        
        Agents = np.array(self.input_names_train)
        self.Pred_agents = np.array([agent in self.data_set.needed_agents for agent in Agents])
        assert self.Pred_agents.sum() > 0, "nothing to predict"
        
        self.setup_method()
        
        # Set trained to flase, this prevents a prediction on an untrained model
        self.trained = False
        self.extracted_data = False
        
        
    def train(self):
        self.extracted_data = False
        if os.path.isfile(self.model_file) and not self.data_set.overwrite_results:
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
            
            if self.provides_epoch_loss():
                self.loss_file = self.model_file[:-4] + '--train_loss.npy'
                assert hasattr(self, 'train_loss'), "No train loss is provided."
                assert isinstance(self.train_loss, np.ndarray), "The train loss should be a numpy array."
                assert len(self.train_loss.shape) == 2, "The train loss should be a 2D numpy array."
                np.save(self.loss_file, self.train_loss.astype(np.float32))
           
        self.trained = True
        
        print('')
        print('The model ' + self.get_name()['print'] + ' was successfully trained.')
        print('')
        return self.model_file
        
        
    def predict(self):
        self.extracted_data = False
        # perform prediction
        if os.path.isfile(self.pred_file) and not self.data_set.overwrite_results:
            output = list(np.load(self.pred_file, allow_pickle = True)[:-1])
        else:
            if self.get_input_type()['past'] == 'general':
                self.Input_prediction_test = self.data_set.Input_prediction
                self.Input_path_test       = None
                
            elif self.get_input_type()['past'] == 'path':
                self.Input_prediction_test = None
                self.Input_path_test       = self.data_set.Input_path
                
            elif self.get_input_type()['past'] == 'both':
                self.Input_prediction_test = self.data_set.Input_prediction
                self.Input_path_test       = self.data_set.Input_path
                
            if self.get_input_type()['future']:
                Pov_ind = self.data_set.pov_agent == np.array(self.input_names_train)
                self.Input_future_test = self.data_set.Output_path.iloc[:, Pov_ind]
            
            # Make predictions on all samples, test and training samples
            self.Input_T_test             = self.data_set.Input_T
            self.Output_T_pred_test       = self.data_set.Output_T_pred

            self.Type_test                = self.data_set.Type
            self.Domain_test              = self.data_set.Domain            
            # save number of training samples
            self.num_samples_test = len(self.Input_T_test)
            
            # apply model to test samples
            output = self.predict_method() # output needs to be a list of components
            
            save_data = np.array(output + [0], object) #0 is there to avoid some numpy load and save errros
            
            os.makedirs(os.path.dirname(self.pred_file), exist_ok=True)
            np.save(self.pred_file, save_data)
                        
        print('')
        print('The model ' + self.get_name()['print'] + ' successfully made predictions.')
        print('')
        return output
    
    
    def prepare_batch_generation(self, train = True, val_split_size = 0.1):
        # Required attributes of the model
        # self.min_t_O_train: How many timesteps do we need for training
        # self.max_t_O_train: How many timesteps do we allow training for
        # self.predict_single_agent: Are joint predictions not possible
        # self.can_use_map: Can use map or not
        # If self.can_use_map, the following is also required
        # self.target_width:
        # self.target_height:
        # self.grayscale: Are image required in grayscale
        
        
        
        # Get general outputs
        if train:
            N_O = np.zeros(len(self.Output_T_train), int)
            for i_sample in range(self.Output_T_train.shape[0]):
                N_O[i_sample] = len(self.Output_T_train[i_sample])
            
            remain_samples = N_O >= self.min_t_O_train
            N_O = np.minimum(N_O[remain_samples], self.max_t_O_train)
            N_I = len(self.Input_path_train.to_numpy()[0,0])
            
            X_help = self.Input_path_train.to_numpy()[remain_samples]
            Y_help = self.Output_path_train.to_numpy()[remain_samples]
            Types  = self.Type_train.to_numpy()[remain_samples]
            domain_old = self.Domain_train.iloc[remain_samples]
            
        else:
            N_O = np.zeros(len(self.Output_T_pred_test), int)
            for i_sample in range(self.Output_T_pred_test.shape[0]):
                N_O[i_sample] = len(self.Output_T_pred_test[i_sample])

            N_I = len(self.Input_path_test.to_numpy()[0,0])
            
            X_help     = self.Input_path_test.to_numpy()
            Types      = self.Type_test.to_numpy()
            domain_old = self.Domain_test
            
        # Determine needed agents
        Agents = np.array(self.input_names_train)
        
        use_map = self.data_set.includes_images() and self.can_use_map
            
        X = np.ones(list(X_help.shape) + [N_I, 2], dtype = np.float32) * np.nan
        if train:
            Y = np.ones(list(Y_help.shape) + [self.max_t_O_train, 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample in range(X.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if isinstance(X_help[i_sample, i_agent], float):
                    assert not self.Pred_agents[i_agent], 'A needed agent is not given.'
                else:    
                    X[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                    if train:
                        n_time = min(self.max_t_O_train, N_O[i_sample])
                        Y[i_sample, i_agent, :n_time] = Y_help[i_sample, i_agent][:n_time].astype(np.float32)
        
        
        if self.predict_single_agent:
            N_O = N_O.repeat(self.Pred_agents.sum())
            # set agent to be predicted into first location
            Xi = []
            T = []
            ID = []
            for i_agent in np.where(self.Pred_agents)[0]:
                reorder_index = np.array([i_agent] + list(np.arange(i_agent)) + 
                                         list(np.arange(i_agent + 1, Xi.shape[1])))
                Xi.append(X[:,reorder_index])
                T.append(Types[:, reorder_index])
                sample_ID = np.tile(np.arange(len(X))[:,np.newaxis], (1, len(Agents)))
                Agent_ID  = np.tile(reorder_index[np.newaxis], (len(X), 1))
                ID.append(np.stack((sample_ID, Agent_ID), -1))
                
            X  = np.stack(Xi, axis = 1).reshape(-1, X.shape[1], N_I, 2)
            T  = np.stack(T, axis = 1).reshape(-1, X.shape[1])
            ID = np.stack(ID, axis = 1).reshape(-1, X.shape[1], 2)
            if train: 
                Y = Y[:, self.Pred_agents].reshape(-1, 1, N_O.max(), 2)
            
            if use_map:
                centre = X[:,0,-1,:] #x_t.squeeze(-2)
                x_rel = centre - X[:,0,-2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 

                domain_repeat = domain_old.loc[domain_old.index.repeat(self.Pred_agents.sum())]
            
                img, img_m_per_px = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                                      target_height = self.target_height, 
                                                                      target_width = self.target_width, 
                                                                      grayscale = self.grayscale, 
                                                                      return_resolution = True)
                
                img          = img[:,np.newaxis]
                img_m_per_px = img_m_per_px[:,np.newaxis]
                
        else:
            T = Types
            
            sample_ID = np.tile(np.arange(len(X))[:,np.newaxis], (1, len(Agents)))
            Agent_ID  = np.tile(np.arange(len(Agents)[np.newaxis,:], (len(X), 1)))
            ID = np.stack((sample_ID, Agent_ID), -1)
            
            if use_map:
                Img_needed = T != 48
                
                
                centre = X[Img_needed, -1,:]
                x_rel = centre - X[Img_needed, -2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 
            
                domain_index = domain_old.index.to_numpy()
                domain_index = domain_index.repeat(Img_needed.sum(1))
                domain_repeat = domain_old.loc[domain_index]
                
                if self.grayscale:
                    channels = 1
                else:
                    channels = 3
            
                img = np.zeros((X.shape[0], X.shape[1], self.target_height, self.target_width, channels), dtype = 'uint8')
                img_m_per_px = np.ones(T.shape) * np.nan
                
                img[Img_needed], img_m_per_px[Img_needed] = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                                                              target_height = self.target_height, 
                                                                                              target_width = self.target_width, 
                                                                                              grayscale = self.grayscale, 
                                                                                              return_resolution = True)
                
        self.X = X.astype(torch.float32) # num_samples, num_agents, num_timesteps, 2
        self.T = T.astype(torch.float32) # num_samples, num_agents
        self.N_O = N_O
        self.ID = ID
        
        if train:
            self.Y = Y.astype(torch.float32) # num_samples, num_agents, num_timesteps, 2
        else:
            self.Y = None
        
        if use_map:
            self.img = img  # num_samples, num_agents, height, width, channels
            self.img_m_per_px = img_m_per_px # num_samples, num_agents
        else:
            self.img = None
            self.img_m_per_px = None
        
        self.extracted_data = True
        
        # Prepare split into main and val set
        Index = np.arange(len(self.X))
        np.random.shuffle(Index)
        
        num_norm = int(len(self.X) * (1 - val_split_size))
        
        self.Index_norm = [Index[:num_norm], np.array([])]
        self.Index_val  = [Index[num_norm:], np.array([])]
    
    
    def provide_batch_data(self, mode, batch_size, val_split_size = 0.0, ignore_map = False):
        
        assert mode in ['train', 'val', 'pred'], "Unknown Mode"
        
        if mode == 'pred':
            val_split_size = 0.0
            advance_norm = True
        elif mode == 'train':
            advance_norm = True
        else:
            advance_norm = False
            
        if not self.extracted_data:
            if mode == 'pred':
                train_mode = False
            else:
                train_mode = True
        
            self.prepare_batch_generation(train = train_mode, val_split_size = val_split_size)
        
        
        
            
        # Ensure that for single_pred_agents, the predictions are alwways the same type
        if advance_norm:
            Index_advance = self.Index_norm
        else:
            Index_advance = self.Index_val
            
        
        # Find identical agents
        N_O_advance = self.N_O[Index_advance[0]]
        Use_candidate = N_O_advance == N_O_advance[0]
        
        # If not enough agents are available during training, use the next ones
        if mode == 'train' and Use_candidate.sum() < batch_size:
            N_O_possible, N_O_counts = np.unique(N_O_advance[N_O_advance >= N_O_advance[0]], return_counts = True)
            N_O_cum = np.cumsum(N_O_counts)
            
            if N_O_cum[-1] > batch_size:
                needed_length = np.where(N_O_cum > batch_size)[0][0]
                Use_candidate = (N_O_advance >= N_O_advance[0]) & (N_O_advance <= N_O_possible[needed_length])
            else:
                Use_candidate = (N_O_advance >= N_O_advance[0])
            
        if self.predict_single_agent:
            T_advance = self.T[Index_advance[0],0]
            Use_candidate = Use_candidate & (T_advance == T_advance[0])
             
        # Find in remaining samples those whose type corresponds to that of the first
        Ind_candidates = np.where(Use_candidate)[0]
        ind_advance = Index_advance[0][Ind_candidates[:batch_size]]
        
        
        if len(Ind_candidates) > batch_size:
            pos_max = Ind_candidates[batch_size - 1] + 1
            
            index_begin = Index_advance[0][:pos_max][~Use_candidate[:pos_max]]
            index_end   = Index_advance[0][pos_max:]
            
            ind_remain = np.concat((index_begin, index_end))
        else:
            ind_remain = Index_advance[0][~Use_candidate]
            
        X = self.X[ind_advance]
        T = self.T[ind_advance]
        
        if self.predict_single_agent:
            assert len(np.unique(T[:,0])) == 0
        
        assert len(np.unique(self.N_O[ind_advance])) == 1
        
        if self.Y is not None:
            n_o = self.N_O[ind_advance].min()
            Y = self.Y[ind_advance,:,:n_o]
        else:
            assert len(np.unique(self.N_O[ind_advance])) == 1
            n_o = self.N_O[ind_advance].max()
            Y = None
        
        if (self.img is not None) and (not ignore_map):
            img          = self.img[ind_advance]
            img_m_per_px = self.img_m_per_px[ind_advance]
        else:
            img          = None
            img_m_per_px = None
            
        # Move used indices from Index_advance[0] to Index_advance[1]
        Index_advance[1] = np.concat(Index_advance[1], ind_advance)
        Index_advance[0] = ind_remain
        
        if len(Index_advance[0]) == 0:
            epoch_done = True
            Index_advance[0] = Index_advance[1]
            Index_advance[1] = np.array([])
            np.random.shuffle(Index_advance[0])
        else:
            epoch_done = False
            
        # check if epoch is completed, if so, shuffle and reset index
        if mode == 'pred':
            Sample_id = self.ID[ind_advance,0,0]
            Agents = np.array(self.input_names_train)
            Agent_id = Agents[self.ID[ind_advance,:,1]]
            return X, T, img, img_m_per_px, n_o, Sample_id, Agent_id, epoch_done    
        else:
            return X, Y, T, img, img_m_per_px, n_o, epoch_done
        
    def create_empty_output_path(self):
        Agents = np.array(self.Output_path_train.columns)
        
        Output_Path = pd.DataFrame(np.empty((len(self.Output_T_pred_test), self.Pred_agents.sum()), np.ndarray), 
                                   columns = Agents[self.Pred_agents])
        return Output_Path
        
    def check_trainability(self):
        if self.get_input_type()['future']: 
            if self.data_set.pov_agent is None:
                return 'the needed pov agent is not included in this dataset.'
            
            if not self.data_set.future_input():
                return 'the required future trajectory of the pov agent contains too many clues.'
            
            if self.get_output_type() == 'path_all_wi_pov':
                return 'the trajectory of the poc agent that has to be predicted should be used as input.'
        else:
            if self.get_output_type() == 'path_all_wo_pov': 
                return 'the trajectory of the poc agent is not predicted, although it is not know a priori.'
        
        if self.get_output_type() == 'path_all_wo_pov':
            if len(self.data_set.needed_agents) == 1:
                return 'there is no agent of which a trajectory can be predicted.'
        
        if self.get_input_type()['past'] in ['general', 'both']:
            if not self.data_set.general_input_available:
                return 'this dataset cannot provide the required 1D inputs.'
            
        if self.get_output_type() in ['class', 'class_and_time']:
            if not self.data_set.classification_useful:
                return 'a classification model cannot be trained on a classless dataset.'
              
        return self.check_trainability_method()
    
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                            Model dependend functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
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
        # return output

    def check_trainability_method(self):
        # checks if current environment (i.e, number of agents, number of input timesteps) make this trainable.
        # If not, it also retuns the reason in form of a string.
        # If it is trainable, return None
        raise AttributeError('Has to be overridden in actual model.')
        
            
    def get_output_type(self = None):
        # Should return 'class', 'class_and_time', 'path_all_wo_pov', 'path_all_wi_pov'
        # the same as above, only this time callable from class
        raise AttributeError('Has to be overridden in actual model.')
        
    def get_input_type(self = None):
        # Should return a dictionary, with the first key being 'past', 
        # where the value is either 'general' (using self.Input_prediction_train) 
        # or 'path' (using self.Input_path_train) or 'both'
        # The second key is 'future', which is either False or True, with pov being
        # the role of the actor whose future input is used. If this is not None, than 
        # the output type must not be 'path_all_wi_pov'
        # the same as above, only this time callable from class
        raise AttributeError('Has to be overridden in actual model.')
        

    def get_name(self = None):
        # Provides a dictionary with the different names of the dataset:
        # Name = {'print': 'printable_name', 'file': 'name_used_in_files', 'latex': r'latex_name'}
        # If the latex name includes mathmode, the $$ has to be included
        # Here, it has to be noted that name_used_in_files will be restricted in its length.
        # For models, this length is 10 characters, without a '-' inside
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def save_params_in_csv(self = None):
        # Returns true or false, depending on if the params of the trained model should be saved as a .csv file or not
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def requires_torch_gpu(self = None):
        # Returns true or false, depending if the model does calculations on the gpu
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def provides_epoch_loss(self = None):
        # Returns true or false, depending on if the model provides losses or not
        raise AttributeError('Has to be overridden in actual model.')
        
        