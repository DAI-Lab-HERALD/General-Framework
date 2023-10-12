import pandas as pd
import numpy as np
import os
import torch
import warnings
from Prob_function import OPTICS_GMM

class model_template():
    def __init__(self, data_set, splitter, evaluate_on_train_set, behavior = None): 
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
        
        self.dt = data_set.dt
        self.has_map = self.data_set.includes_images()
        
        self.num_timesteps_in = data_set.num_timesteps_in_real
        self.num_timesteps_out = data_set.num_timesteps_out_real
        
        self.agents_to_predict = data_set.agents_to_predict
        self.general_input_available = self.data_set.general_input_available
        
        self.input_names_train = np.array(data_set.Input_path.columns)
        
        self.t_e_quantile = self.data_set.p_quantile
            
        
        self.num_samples_path_pred = self.data_set.num_samples_path_pred
        self.evaluate_on_train_set = evaluate_on_train_set
        
        self.setup_method()

        if behavior == None:
            self.is_data_transformer = False
            
            if hasattr(self.splitter, 'Train_index'):
                self.Index_train = self.splitter.Train_index
                
                if self.evaluate_on_train_set:
                    # self.Index_test = np.arange(len(self.data_set.Output_T))
                    self.Index_test = np.unique(np.concatenate((self.splitter.Test_index, 
                                                                self.splitter.Train_index)))
                else:
                    self.Index_test = self.splitter.Test_index
            
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
                self.simply_load_results = False
            else:
                self.simply_load_results = True
            
        else:
            self.is_data_transformer = True
            self.simply_load_results = False
            
            self.Index_train = np.where(data_set.Output_A[behavior])[0]
            self.Index_test = np.arange(len(data_set.Output_A))
            
            if len(self.Index_train) == 0:
                # There are no samples of the required behavior class
                # Use those cases where the other behaviors are the furthers away
                num_long_index = int(len(data_set.Output_A) / len(data_set.Behaviors))
                self.Index_train = np.argsort(data_set.Output_T_E)[-num_long_index :]
            
            self.model_file = data_set.data_file[:-4] + '-transform_path_(' + behavior + ').npy'
            self.pred_file = self.model_file[:-4] + '-pred_tra_wip_.npy'
        
        # Set trained to flase, this prevents a prediction on an untrained model
        self.trained = False
        self.extracted_data = False
    
    
    def train(self):
        assert not self.simply_load_results, 'This model instance is nonly for loading results.'
        self.model_mode = 'train'
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
        
        
    def predict(self):
        assert not self.simply_load_results, 'This model instance is nonly for loading results.'
        
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
            
            
        # check if prediction can be loaded
        self.model_mode = 'pred'
        # perform prediction
        if self.get_output_type()[:4] == 'path':
            test_file = self.pred_file[:-4] + '00.npy'
            if os.path.isfile(test_file) and not self.data_set.overwrite_results:
                output = np.load(test_file, allow_pickle = True)
                
                Pred_index = [output[0]]
                Output_path = [output[1]]
                
                save = 1 
                new_test_file = self.pred_file[:-4] + str(save).zfill(2)+ '.npy'
                while os.path.isfile(new_test_file):
                    output = np.load(new_test_file, allow_pickle = True)
                    Pred_index.append(output[0])
                    Output_path.append(output[1])
                    
                    save += 1 
                    new_test_file = self.pred_file[:-4] + str(save).zfill(2)+ '.npy'
                
                # Concatenate files
                Pred_index = np.concatenate(Pred_index, axis = 0)
                Output_path = pd.concat(Output_path)
                
                # If loaded data is only test set, but train set is required
                if not ((len(Pred_index) < len(self.data_set.Output_T)) and
                        self.evaluate_on_train_set):
                
                    return [Pred_index, Output_path]
            
        else:
            if os.path.isfile(self.pred_file) and not self.data_set.overwrite_results:
                output = list(np.load(self.pred_file, allow_pickle = True)[:-1])
                return output
        
        # create predictions, as no save file available
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
        
        os.makedirs(os.path.dirname(self.pred_file), exist_ok=True)
        
        if self.get_output_type()[:4] != 'path':
            #0 is there to avoid some numpy load and save errros
            save_data = np.array(output + [0], object)
            np.save(self.pred_file, save_data)
        else:
            print('Saving predicted trajectories.', flush = True)
            
            # Get number of predicted agents per sample:
            agent_bool = self.data_set.Type.to_numpy().astype(str) != 'nan'
            num_timesteps = self.data_set.N_O_data_orig + self.num_timesteps_in
            savable_timesteps = agent_bool.sum(1) * num_timesteps
            savable_timesteps_total = savable_timesteps.sum()
            savable_timesteps_total = max(savable_timesteps_total, 2 ** 27) # 2 ** 27 should be 1 GB
            
            # Get number of timesteps per sample that should be saved
            self.data_set._determine_pred_agents(pred_pov = self.get_output_type() == 'path_all_wi_pov')
            pred_agent_bool = self.data_set.Pred_agents_pred[self.Index_test]
            num_timesteps_pred = self.data_set.N_O_pred_orig[self.Index_test]
            to_save_timesteps = pred_agent_bool.sum(1) * num_timesteps_pred * self.num_samples_path_pred
            
            Unsaved_indices = np.arange(len(self.Index_test))
            save = 0
            while len(Unsaved_indices) > 0:
                to_save_timesteps_cum = np.cumsum(to_save_timesteps[Unsaved_indices])
                if to_save_timesteps_cum[-1] <= savable_timesteps_total:
                    num_saved = len(Unsaved_indices)
                else:
                    num_saved = np.where(to_save_timesteps_cum > savable_timesteps_total)[0][0]

                save_indices = Unsaved_indices[:num_saved]
                save_data = np.array([self.Index_test[save_indices], 
                                      self.Output_path_pred.iloc[save_indices], 
                                      0], object)
                
                save_file = self.pred_file[:-4] + str(save).zfill(2) + '.npy'
                np.save(save_file, save_data)
                
                save += 1
                Unsaved_indices = Unsaved_indices[num_saved:]
                
                print('Saved part {} of predicted trajectories'.format(save), flush = True)
                
        print('')
        print('The model ' + self.get_name()['print'] + ' successfully made predictions.', flush = True)
        print('')
        return output
    
    #%%     
    def prepare_batch_generation(self):
        # Required attributes of the model
        # self.min_t_O_train: How many timesteps do we need for training
        # self.max_t_O_train: How many timesteps do we allow training for
        # self.predict_single_agent: Are joint predictions not possible
        # self.can_use_map: Can use map or not
        # If self.can_use_map, the following is also required
        # self.target_width:
        # self.target_height:
        # self.grayscale: Are image required in grayscale
        
        if not self.extracted_data:
            # Extract old trajectories
            self.data_set._extract_original_trajectories()
            
            # Get required timesteps
            N_O_pred = self.data_set.N_O_pred_orig.copy()
            N_O_data = np.minimum(self.data_set.N_O_data_orig, self.max_t_O_train)
            
            # Get positional data
            X = self.data_set.X_orig
            Y = self.data_set.Y_orig[:,:,:N_O_data.max()]
            
            # Get metadata
            domain_old = self.data_set.Domain
            
            # Determine needed agents
            self.data_set._determine_pred_agents(pred_pov = self.get_output_type() == 'path_all_wi_pov')
            
            # Get agent types
            T = self.data_set.Type.to_numpy()
            T = T.astype(str)
            T[T == 'nan'] = '0'
            
            # Determine map use
            use_map = self.has_map and self.can_use_map
            
            # Reorder agents to save data
            if self.predict_single_agent or (self.data_set.Pred_agents_pred.sum(1) == 1).all():
                num_pred_agents = self.data_set.Pred_agents_pred.sum(axis = 1)
                
                N_O_data = N_O_data.repeat(num_pred_agents)
                N_O_pred = N_O_pred.repeat(num_pred_agents)
                
                # set agent to be predicted into first location
                sample_id, pred_agent_id = np.where(self.data_set.Pred_agents_pred)
                
                # Get sample id
                num_agents = self.data_set.Pred_agents_pred.shape[1]
                Sample_id = np.tile(sample_id[:,np.newaxis], (1, num_agents))
                
                # Roll agents so that pred agent is first
                Agent_id = np.tile(np.arange(num_agents)[np.newaxis,:], (len(sample_id), 1))
                Agent_id = Agent_id + pred_agent_id[:,np.newaxis]
                Agent_id = np.mod(Agent_id, num_agents) 
                
                # Project out the sample ID
                X = X[Sample_id, Agent_id]
                
                # Find closest distance between agents during past observation
                D = ((X[:,[0]] - X) ** 2).sum(-1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category = RuntimeWarning)
                    D = np.nanmin(D, axis = -1)
                Agent_sorted_id = np.argsort(D, axis = 1)
                
                Sample_id_sorted = np.tile(np.arange(len(X))[:,np.newaxis], (1, X.shape[1])) 
                
                X = X[Sample_id_sorted, Agent_sorted_id] 
                Sample_id = Sample_id[Sample_id_sorted, Agent_sorted_id]
                Agent_id  = Agent_id[Sample_id_sorted, Agent_sorted_id] 
                
                # remove nan columns
                use_agents = np.where(~np.isnan(X).all((0,2,3)))[0]
                
                # Set agents to nan that are to far away from the predicted agent
                num_agent = self.data_set.max_num_agents
                if num_agent is not None:
                    use_agents = use_agents[:num_agent]
                
                X         = X[:,use_agents]
                Sample_id = Sample_id[:,use_agents]
                Agent_id  = Agent_id[:,use_agents]
                
                self.Pred_agents = self.data_set.Pred_agents_pred[Sample_id, Agent_id]
                self.Pred_agents[:,1:] = False
                
            else:
                # Sort agents to move the absent ones to the behind, and pred agents first
                Value = - np.isfinite(X).sum((2,3)) - np.isfinite(Y).sum((2,3)) - self.data_set.Pred_agents_pred.astype(int)
                
                Agent_id = np.argsort(Value, axis = 1)
                Sample_id = np.tile(np.arange(len(X))[:,np.newaxis], (1, Value.shape[1]))
                
                
                X = X[Sample_id, Agent_id]
                
                # remove nan columns
                missing_agents = np.isnan(X).all((0,2,3))
                X = X[:,~missing_agents]
                
                Sample_id = Sample_id[:,~missing_agents]
                Agent_id  = Agent_id[:,~missing_agents]
                self.Pred_agents = self.data_set.Pred_agents_pred[Sample_id, Agent_id]
                
            # remove nan columns
            self.X = X.astype(np.float32) # num_samples, num_agents, num_timesteps, 2
            self.Y = Y[Sample_id, Agent_id].astype(np.float32) # num_samples, num_agents, num_timesteps, 2
            
            self.T = T[Sample_id, Agent_id] # num_samples, num_agents
            
            self.N_O_pred = N_O_pred
            self.N_O_data = N_O_data
            
            self.ID = np.stack((Sample_id, Agent_id), -1)  
            # Get images
            if use_map:
                if self.predict_single_agent:
                    centre = X[:,0,-1,:] #x_t.squeeze(-2)
                    x_rel = centre - X[:,0,-2,:]
                    rot = np.angle(x_rel[:,0] + 1j * x_rel[:,1]) 
            
                    domain_repeat = domain_old.loc[domain_old.index.repeat(num_pred_agents)]
                    
                    img, img_m_per_px = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                                          target_height = self.target_height, 
                                                                          target_width = self.target_width, 
                                                                          grayscale = self.grayscale, 
                                                                          return_resolution = True)
            
                    img          = img[:,np.newaxis]
                    img_m_per_px = img_m_per_px[:,np.newaxis]
                else:
                    Img_needed = self.T != '0'
                    
                    
                    centre = X[Img_needed, -1,:]
                    x_rel = centre - X[Img_needed, -2,:]
                    rot = np.angle(x_rel[:,0] + 1j * x_rel[:,1]) 
                
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
    
                self.img = img  # num_samples, num_agents, height, width, channels
                self.img_m_per_px = img_m_per_px # num_samples, num_agents
            else:
                self.img = None
                self.img_m_per_px = None
        
        self.extracted_data = True
    
    
    def provide_all_included_agent_types(self):
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
        T = self.data_set.Type.to_numpy()
        T = T.astype(str)
        T[T == 'nan'] = '0'
        
        T_all = np.unique(T)
        T_all = T_all[T_all != '0']
        return T_all
    
    def _extract_useful_training_samples(self):
        I_train = np.where(np.in1d(self.ID[:,0,0], self.Index_train))[0]
        
        remain_samples = self.N_O_data[I_train] >= self.min_t_O_train
        
        # Only use samples with enough timesteps for training
        I_train = I_train[remain_samples]
        
        return I_train
    
    
    def save_predicted_batch_data(self, Pred, Sample_id, Agent_id, Pred_agents = None):
        r'''

        Parameters
        ----------
        Pred : np.ndarray
            This is the predicted future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{I} \times 2\}` dimensional numpy array with float values. 
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
                self.Output_path_pred.loc[i_sample][agent] = Pred[i, j,:, :, :].astype('float32')
    
    def provide_all_training_trajectories(self):
        r'''
        This function provides trajectroy data an associated metadata for the training of model
        during prediction and training. It returns the whole training set (including validation set)
        in one go


        Returns
        -------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
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
        
        assert self.get_output_type()[:4] == 'path'
        
        self.prepare_batch_generation()
        
        I_train = self._extract_useful_training_samples()
        
        X_train = self.X[I_train]
        Y_train = self.Y[I_train]
        T_train = self.T[I_train]
        Pred_agents_train = self.Pred_agents[I_train]
        
        if self.img is not None:
            img_train = self.img[I_train]
            img_m_per_px_train = self.img_m_per_px[I_train]
        else:
            img_train = None
            img_m_per_px_train = None
        
        Sample_id_train = self.ID[I_train,0,0]
        Agents = np.array(self.input_names_train)
        Agent_id_train = Agents[self.ID[I_train,:,1]]
        
        
        return [X_train, Y_train, T_train, img_train, img_m_per_px_train, 
                Pred_agents_train, Sample_id_train, Agent_id_train]
        
        
    
    
    def provide_batch_data(self, mode, batch_size, val_split_size = 0.0, ignore_map = False):
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
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
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
                
            N_O = self.N_O_data
            Ind_advance = self.Ind_val
            
        elif mode == 'train':
            assert self.model_mode == 'train', 'During training, the non-validation part of the training set should be called.'
            if not hasattr(self, 'Ind_train'):
                I_train = self._extract_useful_training_samples()
                num_train = int(len(I_train) * (1 - val_split_size))
                self.Ind_train = [I_train[:num_train], np.array([], int)]
            
            N_O = self.N_O_data
            Ind_advance = self.Ind_train
        
        else:
            raise TypeError("Unknown mode.")
        
        # Find identical agents
        N_O_advance = N_O[Ind_advance[0]]
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
            T_advance = self.T[Ind_advance[0],0]
            Use_candidate = Use_candidate & (T_advance == T_advance[0])
             
        # Find in remaining samples those whose type corresponds to that of the first
        Ind_candidates = np.where(Use_candidate)[0]
        
        # Get the final indices to be returned
        ind_advance = Ind_advance[0][Ind_candidates[:batch_size]]
        
        # Get the indices that will remain
        ind_remain = np.setdiff1d(Ind_advance[0], ind_advance)
        
        # get num_steps
        num_steps = N_O[ind_advance].min()
        
        X = self.X[ind_advance]
        T = self.T[ind_advance]
        Y = self.Y[ind_advance,:,:num_steps]
        Pred_agents = self.Pred_agents[ind_advance]
        
        if self.predict_single_agent:
            assert len(np.unique(T[:,0])) == 1
        
        if mode == 'pred':
            assert len(np.unique(N_O[ind_advance])) == 1
        
        if (self.img is not None) and (not ignore_map):
            img          = self.img[ind_advance]
            img_m_per_px = self.img_m_per_px[ind_advance]
        else:
            img          = None
            img_m_per_px = None
            
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
            
        # check if epoch is completed, if so, shuffle and reset index
        if mode == 'pred':
            Sample_id = self.ID[ind_advance,0,0]
            Agents = np.array(self.input_names_train)
            Agent_id = Agents[self.ID[ind_advance,:,1]]
            return X,    T, img, img_m_per_px, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
        else:
            return X, Y, T, img, img_m_per_px, Pred_agents, num_steps,                      epoch_done
    
    
    def get_classification_data(self, train = True):
        r'''
        This function retuns inputs and outputs for classification models.

        Parameters
        ----------
        train : bool, optional
            This discribes whether one wants to generate training or testing data. The default is True.

        Returns
        -------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with 
            float values. If an agent is fully or or some timesteps partially not observed, then this can 
            include np.nan values.
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
        
        T = self.data_set.Type.to_numpy()
        T = T.astype(str)
        T[T == 'nan'] = '0'
        
        # Determine map use
        self.data_set._extract_original_trajectories()
        
        # Extract data from original number a samples
        if self.general_input_available:
            D_help = self.data_set.Input_prediction.to_numpy()
            D = np.ones(list(D_help.shape) + [self.num_timesteps_in], dtype = np.float32) * np.nan
            for i_sample in range(D.shape[0]):
                for i_dist in range(D.shape[1]):
                    D[i_sample, i_dist] = D_help[i_sample, i_dist].astypt(np.float32)
        else:
            D = np.zeros((len(T),0), dtype = np.float32)
        
        # Select current samples
        if train:
            assert self.model_mode == 'train', 'During training, training set should be called.'
            Index = self.Index_train
        else:
            assert self.model_mode == 'pred', 'During training, training set should be called.'
            Index = self.Index_test
            
        X = self.data_set.X_orig[Index]
        T = T[Index]
        D = D[Index]
        P = self.data_set.Output_A.to_numpy().astpye(np.float32)[Index]
        DT = self.data_set.Output_T_E.astype(np.float32)[Index]
        
        class_names = self.data_set.Output_A.columns
        agent_names = self.data_set.Input_paths.columns
        dist_names = self.data_set.Input_prediction.columns
        
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
        assert P.shape == self.Output_T_E_pred[class_names].to_numpy.shape
        for i in range(len(P)):
            for name, j in enumerate(class_names):
                self.Output_A_pred.iloc[i][name] = P[i,j]
        
        
        if self.get_output_type() == 'class_and_time':
            assert DT.shape[:2] == self.Output_T_E_pred[class_names].to_numpy.shape
            assert DT.shape[2] == len(self.t_e_quantile)
            for i in range(len(DT)):
                for name, j in enumerate(class_names):
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
    def _transform_predictions_to_numpy(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Path_pred') and hasattr(self, 'Path_true') and hasattr(self, 'Pred_step'):
            if self.excluded_ego == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego = exclude_ego
        
        # Check if data is there
        self.data_set._extract_original_trajectories()
        self.data_set._determine_pred_agents(eval_pov = ~exclude_ego)
        self.prepare_batch_generation()
        
        # Initialize output
        num_samples = len(Output_path_pred)
        
        # Get predicted timesteps
        nto = self.num_timesteps_out
        Nto_i = np.minimum(nto, self.N_O_data[Pred_index])
        
        # get predicted agents
        Pred_agents = self.data_set.Pred_agents_eval[Pred_index]
        
        # Get pred agents
        max_num_pred_agents = Pred_agents.sum(1).max()
        
        i_agent_sort = np.argsort(-Pred_agents.astype(float))
        i_agent_sort = i_agent_sort[:,:max_num_pred_agents]
        i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_pred_agents))
        
        # Get predicted timesteps
        self.Pred_step = Nto_i[:,np.newaxis] > np.arange(nto)[np.newaxis]
        self.Pred_step = self.Pred_step[:,np.newaxis] & Pred_agents[i_sampl_sort, i_agent_sort, np.newaxis]
        
        # Get true predictions
        self.Path_true = self.data_set.Y_orig[Pred_index]
        self.Path_true = self.Path_true[i_sampl_sort, i_agent_sort, :nto]
        self.Path_true[~self.Pred_step] = 0.0
        self.Path_true = self.Path_true[:,np.newaxis]
        
        # Get predicted trajectories
        self.Path_pred = np.zeros((self.num_samples_path_pred, num_samples,
                                   max_num_pred_agents, nto, 2), dtype = np.float32)
        
        for i in range(num_samples):
            nto_i = Nto_i[i]
            
            pred_agents = np.where(self.Pred_step[i].any(-1))[0]
            pred_agents_id = i_agent_sort[i, pred_agents]

            path_pred_orig = Output_path_pred.iloc[i, pred_agents_id]
            path_pred = np.stack(path_pred_orig.to_numpy(), axis = 1)
            
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
    
    
    def _get_joint_KDE_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_joint_pred') and hasattr(self, 'Log_prob_joint_true'):
            if self.excluded_ego == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego = exclude_ego
        
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
        
        for subgroup in np.unique(Subgroups):
            subgroup_index = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[subgroup_index], axis = 0)) == 1
            pred_agents = Pred_agents[subgroup_index[0]]
            
            assert len(np.unique(self.T_pred[subgroup_index], axis = 0)) == 1
            agent_types = self.T_pred[subgroup_index[0]]
            
            std = 1 + (agent_types[pred_agents] != 'P') * 79
            std = std[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] 
            
            nto_subgroup = Num_steps[subgroup_index]
            
            for nto in np.unique(nto_subgroup):
                nto_index = subgroup_index[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = self.Path_true[nto_index][:,:,pred_agents,:nto]
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                
                paths_true = paths_true / std
                paths_pred = paths_pred / std
                        
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
                max_preds = max(2 * len(paths_true_comp), 5 * self.num_samples_path_pred)
                kde = OPTICS_GMM().fit(paths_pred_comp[use_preds[:max_preds]])
                
                # Score samples
                log_prob_true = kde.score_samples(paths_true_comp)
                log_prob_pred = kde.score_samples(paths_pred_comp)
                
                self.Log_prob_joint_true[nto_index] = log_prob_true.reshape(*paths_true.shape[:2])
                self.Log_prob_joint_pred[nto_index] = log_prob_pred.reshape(*paths_pred.shape[:2])
            
            
    def _get_indep_KDE_probabilities(self, Pred_index, Output_path_pred, exclude_ego = False):
        if hasattr(self, 'Log_prob_indep_pred') and hasattr(self, 'Log_prob_indep_true'):
            if self.excluded_ego == exclude_ego:
                return
        
        # Save last setting 
        self.excluded_ego = exclude_ego
        
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
        
        for subgroup in np.unique(Subgroups):
            subgroup_index = np.where(Subgroups == subgroup)[0]
            
            assert len(np.unique(Pred_agents[subgroup_index], axis = 0)) == 1
            pred_agents = Pred_agents[subgroup_index[0]]
            pred_agents_id = np.where(pred_agents)[0]
            
            assert len(np.unique(self.T_pred[subgroup_index], axis = 0)) == 1
            agent_types = self.T_pred[subgroup_index[0]]
            
            std = 1 + (agent_types[pred_agents] != 'P') * 79
            std = std[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] 
            
            nto_subgroup = Num_steps[subgroup_index]
            
            for nto in np.unique(nto_subgroup):
                nto_index = subgroup_index[np.where(nto == nto_subgroup)[0]]
                
                # Should be shape: num_subgroup_samples x num_preds x num_agents x num_T_O x 2
                paths_true = self.Path_true[nto_index][:,:,pred_agents,:nto]
                paths_pred = self.Path_pred[nto_index][:,:,pred_agents,:nto]
                
                paths_true = paths_true / std
                paths_pred = paths_pred / std
                
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
                    max_preds = max(2 * len(paths_true_agent_comp), 5 * self.num_samples_path_pred)
                    kde = OPTICS_GMM().fit(paths_pred_agent_comp[use_preds[:max_preds]])
                    
                    # Score samples
                    log_prob_true_agent = kde.score_samples(paths_true_agent_comp)
                    log_prob_pred_agent = kde.score_samples(paths_pred_agent_comp)
                    
                    self.Log_prob_indep_true[nto_index,:,i_agent_orig] = log_prob_true_agent.reshape(*paths_true.shape[:2])
                    self.Log_prob_indep_pred[nto_index,:,i_agent_orig] = log_prob_pred_agent.reshape(*paths_pred.shape[:2])
                
    
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
        