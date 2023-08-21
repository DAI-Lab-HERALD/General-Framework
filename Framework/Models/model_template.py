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
            
            self.Index_train = splitter.Train_index
            
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
            self.Index_train = np.where(data_set.Output_A[behavior])[0]
            if len(self.Index_train) == 0:
                # There are no samples of the required behavior class
                # Use those cases where the other behaviors are the furthers away
                num_long_index = int(len(data_set.Output_A) / len(data_set.Behaviors))
                self.Index_train = np.argsort(data_set.Output_T_E)[-num_long_index :]
            
            save_file = data_set.data_file[:-4] + '-transform_path_(' + behavior + ').npy'
            self.model_file = save_file
            self.pred_file = self.model_file[:-4] + '-pred_tra_wi_pov.npy'
        
        self.data_set = data_set
        
        self.dt = data_set.dt
        self.has_map = self.data_set.includes_images()
        
        self.num_timesteps_in = data_set.num_timesteps_in_real
        self.num_timesteps_out = data_set.num_timesteps_out_real
        
        self.dynamic_prediction_agents = data_set.dynamic_prediction_agents
        self.general_input_available = self.data_set.general_input_available
        
        self.input_names_train = np.array(data_set.Input_path.columns)
        
        self.t_e_quantile = self.data_set.p_quantile
            
        # check if model is allowed
        self.Domain_train            = data_set.Domain.iloc[self.Index_train]
        
        self.num_samples_path_pred = self.data_set.num_samples_path_pred
        
        self.setup_method()
        
        # Set trained to flase, this prevents a prediction on an untrained model
        self.trained = False
        self.extracted_data = False
    
    
    def train(self):
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
        # perform prediction
        if os.path.isfile(self.pred_file) and not self.data_set.overwrite_results:
            output = list(np.load(self.pred_file, allow_pickle = True)[:-1])
        else:
            # apply model to test samples
            if self.get_output_type()[:4] == 'path':
                self.create_empty_output_path()
                self.predict_method()
                output = [self.Output_path_pred]
            elif self.get_output_type() == 'class':
                self.create_empty_output_A()
                self.predict_method()
                output = [self.Output_A_pred]
            elif self.get_output_type() == 'class_and_time':
                self.create_empty_output_A()
                self.create_empty_output_T()
                self.predict_method()
                output = [self.Output_A_pred, self.Output_T_E_pred]
            else:
                raise TypeError("This output type for models is not implemented.")
            
            save_data = np.array(output + [0], object) #0 is there to avoid some numpy load and save errros
            
            os.makedirs(os.path.dirname(self.pred_file), exist_ok=True)
            np.save(self.pred_file, save_data)
                        
        print('')
        print('The model ' + self.get_name()['print'] + ' successfully made predictions.')
        print('')
        return output
    
    #%% 
    def _determine_pred_agents(self, data_set, Recorded, dynamic):
        Agents = np.array(Recorded.columns)
        Pred_agents = np.array([agent in data_set.needed_agents for agent in Agents])
        Pred_agents = np.tile(Pred_agents[np.newaxis], (len(Recorded), 1))
        
        if dynamic:
            for i_sample in range(len(Recorded)):
                R = Recorded.iloc[i_sample]
                for i_agent, agent in enumerate(Agents):
                    if isinstance(R[agent], np.ndarray):
                        Pred_agents[i_sample, i_agent] = np.all(R[agent])
        
        # NuScenes exemption:
        if self.data_set.get_name()['print'] == 'NuScenes':
            if self.num_timesteps_in == 4 and self.num_timesteps_out == 12:
                if self.dt == 0.5 and self.data_set.t0_type == 'all':
                    Pred_agents_N = np.zeros(Pred_agents.shape, bool)
                    PA = self.data_set.Domain.pred_agents
                    PT = self.data_set.Domain.pred_timepoints
                    T0 = self.data_set.Domain.t_0
                    for i_sample in range(len(Recorded)):
                        pt = PT.iloc[i_sample]
                        t0 = T0.iloc[i_sample]
                        i_time = np.argmin(np.abs(t0 - pt))
                        pa = np.stack(PA.iloc[i_sample].to_numpy().tolist(), 1)
                        
                        Pred_agents_N[i_sample, :pa.shape[1]] = pa[i_time]
                    
                    return Pred_agents_N
        return Pred_agents
    
    
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
        
        
        
        N_O_data = np.zeros(len(self.data_set.Output_T), int)
        N_O_pred = np.zeros(len(self.data_set.Output_T), int)
        for i_sample in range(self.data_set.Output_T.shape[0]):
            N_O_data[i_sample] = len(self.data_set.Output_T[i_sample])
            N_O_pred[i_sample] = len(self.data_set.Output_T_pred[i_sample])
        
        X_help = self.data_set.Input_path.to_numpy()
        Y_help = self.data_set.Output_path.to_numpy()
        
        T = self.data_set.Type.to_numpy()
        Recorded_old = self.data_set.Recorded
        domain_old = self.data_set.Domain
        
        # Determine needed agents
        Agents = np.array(self.input_names_train)
        Pred_agents = self._determine_pred_agents(self.data_set, Recorded_old, self.dynamic_prediction_agents)
        
        # Determine map use
        use_map = self.has_map and self.can_use_map
            
        X = np.ones(list(X_help.shape) + [self.num_timesteps_in, 2], dtype = np.float32) * np.nan
        Y = np.ones(list(Y_help.shape) + [N_O_data.max(), 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample in range(X.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if isinstance(X_help[i_sample, i_agent], float):
                    assert not Pred_agents[i_sample, i_agent], 'A needed agent is not given.'
                else:    
                    n_time = N_O_data[i_sample]
                    X[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                    Y[i_sample, i_agent, :n_time] = Y_help[i_sample, i_agent][:n_time].astype(np.float32)
        
        
        if self.predict_single_agent or (Pred_agents.sum(1) == 1).all():
            N_O_data = N_O_data.repeat(Pred_agents.sum(axis = 1))
            N_O_pred = N_O_pred.repeat(Pred_agents.sum(axis = 1))
            
            # set agent to be predicted into first location
            ID = []
            sample_id, pred_agent_id = np.where(Pred_agents)
            
            # Get sample id
            Sample_id = np.tile(sample_id[:,np.newaxis], (1, Pred_agents.shape[1]))
            
            # Roll agents so that pred agent is first
            Agent_id = np.tile(np.arange(Pred_agents.shape[1])[np.newaxis,:], (len(sample_id), 1))
            Agent_id = Agent_id + pred_agent_id[:,np.newaxis]
            Agent_id = np.mod(Agent_id, Pred_agents.shape[1]) 
                
            # Project out the sample ID
            X = X[Sample_id, Agent_id]
            T = T[Sample_id, Agent_id]
            
            # Find closest distance between agents during past observation
            D = np.nanmin(((X[:,[0]] - X) ** 2).sum(-1), axis = -1)
            Agents_sorted_id = np.argsort(D, axis = 1)
            
            Sample_id_sorted = np.tile(np.arange(len(X))[:,np.newaxis], (1, X.shape[1])) 
            
            X = X[Sample_id_sorted, Agents_sorted_id] 
            T = T[Sample_id_sorted, Agents_sorted_id] 
            
            # Set agents to nan that are to far away from the predicted agent
            num_agent = self.data_set.max_num_agents
            if num_agent is not None:
                X[:, num_agent:] = np.nan
                T[:, num_agent:] = np.nan
            
            Agent_id = Agent_id[Sample_id_sorted, Agents_sorted_id] 
            
            ID = np.stack((Sample_id, Agent_id), axis = -1)
            
            Y = Y[Pred_agents].reshape(-1, 1, *Y.shape[-2:])
            
            if use_map:
                centre = X[:,0,-1,:] #x_t.squeeze(-2)
                x_rel = centre - X[:,0,-2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 

                domain_repeat = domain_old.loc[domain_old.index.repeat(Pred_agents.sum(axis = 1))]
                
                img, img_m_per_px = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                                      target_height = self.target_height, 
                                                                      target_width = self.target_width, 
                                                                      grayscale = self.grayscale, 
                                                                      return_resolution = True)

                img          = img[:,np.newaxis]
                img_m_per_px = img_m_per_px[:,np.newaxis]
            
            # Overwrite Pred agents with new length
            Pred_agents = np.zeros(T.shape, bool)
            Pred_agents[:,0] = True
            
        else:
            
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
            
            # Sort agents to move the absent ones to the behind, and pred agents first
            Value = - np.isfinite(X).sum((2,3)) - np.isfinite(Y).sum((2,3)) - Pred_agents.astype(int)
            
            Agent_id = np.argsort(Value, axis = 1)
            Sample_id = np.tile(np.arange(len(X))[:,np.newaxis], (1, len(Agents)))
            
            ID = np.stack((Sample_id, Agent_id), -1)
            
            X = X[Sample_id, Agent_id]
            Y = Y[Sample_id, Agent_id]
            T = T[Sample_id, Agent_id]
            
            img = img[Sample_id, Agent_id]
            img_m_per_px = img_m_per_px[Sample_id, Agent_id]
            
            
                
                
        self.Pred_agents = Pred_agents   
        
        self.X = X.astype(np.float32) # num_samples, num_agents, num_timesteps, 2
        self.Y = Y.astype(np.float32) # num_samples, num_agents, num_timesteps, 2
        
        self.T = T # num_samples, num_agents
        
        self.N_O_pred = N_O_pred
        self.N_O_data = N_O_data
        
        self.ID = ID
        
        if use_map:
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
        T = self.data_set.Type.to_numpy()
        T[T == np.nan] = '0'
        T_all = np.unique(T.astype(str))
        T_all = T_all[T_all != 'nan']
        return T_all
    
    def _extract_useful_training_samples(self):
        I_train = self.Index_train
        N_O = np.minimum(self.N_O_data, self.max_t_O_train)
        
        remain_samples = N_O[I_train] >= self.min_t_O_train
        
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
        assert Pred.shape[[2,4]] == [self.num_samples_path_pred, 2]
        
        for i, i_sample in enumerate(Sample_id):
            for j, j_agent in enumerate(Agent_id[i]):
                if not Pred_agents[i,j]:
                    continue
                self.Output_path_pred.iloc[i_sample, j_agent] = Pred[i, j,:, :, :].astype('float32')
                
    
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
            If an agent is not observed at all, the value will instead be np.nan.
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
            
            If only one agent has to be predicted per sample, for **Y**, **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
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
        
        if not self.extracted_data:
            self.prepare_batch_generation()
        
        if mode == 'pred':
            if not hasattr(self, 'Ind_pred'):
                self.Ind_pred = [np.arange(len(self.X)), np.array([], int)]
            N_O = self.N_O_pred
            Ind_advance = self.Ind_pred
            
        elif mode == 'val':
            if not hasattr(self, 'Ind_val'):
                I_train = self._extract_useful_training_samples()
                num_train = int(len(I_train) * (1 - val_split_size))
                self.Ind_val = [I_train[num_train:], np.array([], int)]
                
            N_O = self.N_O_data
            Ind_advance = self.Ind_val
            
        elif mode == 'train':
            if not hasattr(self, 'Ind_train'):
                I_train = self._extract_useful_training_samples()
                num_train = int(len(I_train) * (1 - val_split_size))
                self.Ind_train = [I_train[:num_train], np.array([], int)]
            
            N_O = self.N_O_data
            Ind_advance = self.Ind_train
        
        else:
            raise TypeError("Unknown mode.")
        
        
        # TODO: add method that allows one to extract the whole dataset in one batch 
        # For classification models
        
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
            for available types). If an agent is not observed at all, the value will instead be np.nan.
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
        
        X_help = self.data_set.Input_path.to_numpy()
        D_help = self.data_set.Input_prediction.to_numpy()
        
        T = self.data_set.Type.to_numpy()
        
        # Determine needed agents
        Agents = np.array(self.input_names_train)
        
        # Determine map use
        X = np.ones(list(X_help.shape) + [self.num_timesteps_in, 2], dtype = np.float32) * np.nan
        if self.data_set.general_input_available:
            D = np.ones(list(X_help.shape) + [self.timesteps], dtype = np.float32) * np.nan
        else:
            D = None
        
        # Extract data from original number a samples
        for i_sample in range(X.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if not isinstance(X_help[i_sample, i_agent], float):
                    X[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                if self.data_set.general_input_available:
                    D[i_sample, i_agent] = D_help[i_sample, i_agent].astypt(np.float32)
        
        P = self.data_set.Output_A.to_numpy().astpye(np.float32)
        DT = self.data_set.Output_T_E.astype(np.float32)
        
        class_names = self.data_set.Output_A.columns
        agent_names = self.data_set.Input_paths.columns
        dist_names = self.data_set.Input_prediction.columns
        
        if train:
            Index = self.Index_train
        else:
            Index = np.arange(len(X))
            
        X = X[Index]
        T = T[Index]
        D = D[Index]
        P = P[Index]
        
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
        Agents = np.array(self.Output_path_train.columns)
        self.Output_path_pred = pd.DataFrame(np.empty((len(self.data_set.Output_T), len(Agents)), np.ndarray), columns = Agents)
        
    
    
    def create_empty_output_A(self):
        Behaviors = np.array(self.data_set.Behaviors)
        self.Output_A_pred = pd.DataFrame(np.zeros((len(self.data_set.Output_T), len(Behaviors)), float), columns = Behaviors)
        
    
    def create_empty_output_T(self):
        Behaviors = np.array(self.data_set.Behaviors)
        self.Output_T_E_pred = pd.DataFrame(np.empty((len(self.data_set.Output_T), len(Behaviors)), np.ndarray), columns = Behaviors)
        for i in range(len(self.Output_T_E_pred)):
            for j in range(self.Output_T_E_pred.shape[1]):
                self.Output_T_E_pred.iloc[i,j] = np.ones(len(self.t_e_quantile), float) * np.nan
        
        
    def check_trainability(self):
        if self.get_output_type() == 'path_all_wo_pov':
            if len(self.data_set.needed_agents) == 1:
                return 'there is no agent of which a trajectory can be predicted.'
            
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
        