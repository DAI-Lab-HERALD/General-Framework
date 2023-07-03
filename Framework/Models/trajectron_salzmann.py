from model_template import model_template
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import random
from Trajectron.trajec_model.trajectron import Trajectron
from Trajectron.trajec_model.model_registrar import ModelRegistrar
from Trajectron.trajec_model.datawrapper import AgentBatch, AgentType
import json
import os

class trajectron_salzmann(model_template):
    
    def setup_method(self, seed = 0):
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Get params
        self.num_timesteps_in = len(self.Input_path_train.to_numpy()[0,0])
        self.num_timesteps_out = np.zeros(len(self.Output_T_train), int)
        for i_sample in range(self.Output_T_train.shape[0]):
            len_use = len(self.Output_T_train[i_sample])
            if self.data_set.num_timesteps_out_real == len_use:
                self.num_timesteps_out[i_sample] = len_use
            else:
                self.num_timesteps_out[i_sample] = len_use - np.mod(len_use - self.data_set.num_timesteps_out_real, 5)
        
        self.remain_samples = self.num_timesteps_out >= 5
        self.num_timesteps_out = np.minimum(self.num_timesteps_out[self.remain_samples], 100)
        
        self.use_map = self.data_set.includes_images()
        self.target_width = 175
        self.target_height = 100
        
        config_path = os.sep.join(os.path.dirname(self.model_file).split(os.sep)[:-3])
        config_path += os.sep + 'Models' + os.sep + 'Trajectron' + os.sep + 'config' + os.sep
        
        
        if (np.array([name[0] for name in np.array(self.input_names_train)]) == 'P').all():
            config_file = config_path + 'pedestrian.json' 
            with open(config_file) as json_file:
                hyperparams = json.load(json_file)
        else:
            config_file = config_path + 'nuScenes.json' 
            with open(config_file) as json_file:
                hyperparams = json.load(json_file)
            
        hyperparams["dec_final_dim"]                 = 32
        
        hyperparams["map_encoding"]                  = self.use_map
        hyperparams["incl_robot_node"]               = False
        
        hyperparams["edge_encoding"]                 = True
        hyperparams["edge_influence_combine_method"] = "attention"
        hyperparams["edge_state_combine_method"]     = "sum"
        hyperparams["adaptive"]                      = False
        hyperparams["dynamic_edges"]                 = "yes"
        hyperparams["edge_addition_filter"]          = [0.25, 0.5, 0.75, 1.0]
        
        hyperparams["single_mode_multi_sample"]      = False
        hyperparams["single_mode_multi_sample_num"]  = 50
        
        self.std_pos_ped = 1
        self.std_vel_ped = 2
        self.std_acc_ped = 1
        self.std_pos_veh = 80
        self.std_vel_veh = 15
        self.std_acc_veh = 4
        
        # Set time step
        self.dt = self.Input_T_train[0][-1] - self.Input_T_train[0][-2]
        
        # Prepare models
        model_registrar = ModelRegistrar(None, self.device)
        self.trajectron = Trajectron(model_registrar, hyperparams, None, self.device)
        self.trajectron.set_environment()
        self.trajectron.set_annealing_params()
        
    def extract_data(self, train = True):
        attention_radius = dict()
        
        if (np.array([name[0] for name in np.array(self.input_names_train)]) == 'P').all():
            attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 3.0
        else:
            # for inD/rounD
            attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
            attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 25.0
            attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 25.0
            attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 50.0
            
        
        if train:
            X_help = self.Input_path_train.to_numpy()
            Y_help = self.Output_path_train.to_numpy() 
            
            X_help = X_help[self.remain_samples]
            Y_help = Y_help[self.remain_samples]
            self.domain_old = self.Domain_train.iloc[self.remain_samples]
        else:
            X_help = self.Input_path_test.to_numpy()
            self.domain_old = self.Domain_test
        
        Agents = np.array([name[2:] for name in np.array(self.input_names_train)])
        Types  = np.array([AgentType.PEDESTRIAN if name[0] == 'P' else AgentType.VEHICLE  
                           for name in np.array(self.input_names_train)], dtype = AgentType)
        
        # Extract predicted agents
        Pred_agents = np.array([agent in self.data_set.needed_agents for agent in Agents])
        assert Pred_agents.sum() > 0, "nothing to predict"
        
        # Prepare numpy position array
        X = np.ones(list(X_help.shape) + [self.num_timesteps_in, 2], dtype = np.float32) * np.nan
        if train:
            Y = np.ones(list(Y_help.shape) + [self.num_timesteps_out.max(), 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample in range(X.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if isinstance(X_help[i_sample, i_agent], float):
                    assert not Pred_agents[i_agent], 'A needed agent is not given.'
                else:    
                    X[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                    if train:
                        n_time = self.num_timesteps_out[i_sample]
                        Y[i_sample, i_agent, :n_time] = Y_help[i_sample, i_agent][:n_time].astype(np.float32)
        
        # get velocities
        V = (X[...,1:,:] - X[...,:-1,:]) / self.dt
        V = np.concatenate((V[...,[0],:], V), axis = -2)
        
        # get accelaration
        A = (V[...,1:,:] - V[...,:-1,:]) / self.dt
        A = np.concatenate((A[...,[0],:], A), axis = -2)
        
        H = np.angle(V[...,0] + 1j * V[...,1])[...,np.newaxis]
        
        Sin = np.sin(H)
        Cos = np.cos(H)
        
        #final state S
        S = np.concatenate((X, V, A, Sin, Cos), axis = -1).astype(np.float32)
        
        Ped_agents = Types == AgentType.PEDESTRIAN
        
        S_st = S.copy()
        S_st[:,Ped_agents,:,0:2]  /= self.std_pos_ped
        S_st[:,~Ped_agents,:,0:2] /= self.std_pos_veh
        S_st[:,Ped_agents,:,2:4]  /= self.std_vel_ped
        S_st[:,~Ped_agents,:,2:4] /= self.std_vel_veh
        S_st[:,Ped_agents,:,4:6]  /= self.std_acc_ped
        S_st[:,~Ped_agents,:,4:6] /= self.std_acc_veh
        
        D = np.min(np.sqrt(np.sum((X[:,:,np.newaxis] - X[:,np.newaxis]) ** 2, axis = -1)), axis = - 1)
        D_max = np.array([[attention_radius[(Types[j_v], Types[i_v])] 
                           for j_v in range(X.shape[1])] for i_v in range(X.shape[1])])
        
        Neighbor_bool = D < D_max[np.newaxis]
        
        # Get Neighbor for each pred value
        Neigh      = np.nan * np.ones((X.shape[0], Pred_agents.sum(), *S.shape[1:]), dtype = np.float32)
        Neigh_num  = np.zeros((X.shape[0], Pred_agents.sum()), dtype = np.int64)
        Neigh_type = np.zeros((X.shape[0], Pred_agents.sum(), X.shape[1]), dtype = int)
        Neigh_len  = np.zeros((X.shape[0], Pred_agents.sum(), X.shape[1]), dtype = int)

        i_pred_agent = 0
        for i_agent, agent in enumerate(Agents):
            avoid_self = (np.arange(len(Agents)) != i_agent)
            if Pred_agents[i_agent]:
                feasible_goals = avoid_self & Neighbor_bool[:, i_agent]
                Neigh_num[:, i_pred_agent] = feasible_goals.sum(-1) 
                for i_sample in range(X.shape[0]):
                    Neigh[i_sample, i_pred_agent, 
                          :Neigh_num[i_sample, i_pred_agent]] = S_st[i_sample, feasible_goals[i_sample]]
                    Neigh_type[i_sample, i_pred_agent, 
                               :Neigh_num[i_sample, i_pred_agent]] = Types[feasible_goals[i_sample]].astype(int)
                    Neigh_len[i_sample, i_pred_agent, 
                              :Neigh_num[i_sample, i_pred_agent]] = self.num_timesteps_in
                i_pred_agent += 1
                
                        
                        
        if self.use_map:
            centre = X[:,Pred_agents,-1,:].reshape(-1, 2)
            x_rel = centre - X[:,Pred_agents,-2,:].reshape(-1, 2)
            rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 
            domain_repeat = self.domain_old.loc[self.domain_old.index.repeat(Pred_agents.sum())]
            
            img, img_m_per_px = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                                  target_height = self.target_height, 
                                                                  target_width = self.target_width, 
                                                                  grayscale = False, return_resolution = True)
            
            img = img[:,:,75:].transpose(0,3,1,2).reshape(X.shape[0], Pred_agents.sum(), 3, 
                                                          self.target_height, self.target_width - 75)
            
            img_m_per_px = img_m_per_px.reshape(X.shape[0], Pred_agents.sum()).astype('float32')
        else:
            img = None
            img_m_per_px = None
            
            
        
        if train:
            Y_st = Y.copy()
            Y_st[:,Ped_agents]  /= self.std_pos_ped
            Y_st[:,~Ped_agents] /= self.std_pos_veh
            return Pred_agents, Agents, Types, S, S_st, Neigh, Neigh_num, Neigh_type, Neigh_len, img, img_m_per_px, Y, Y_st
        else:
            return Pred_agents, Agents, Types, S, S_st, Neigh, Neigh_num, Neigh_type, Neigh_len, img, img_m_per_px
    
    def prepare_model_training(self, Pred_types):
        optimizer = dict()
        lr_scheduler = dict()
        for node_type in Pred_types:
            if node_type.name in self.trajectron.hyperparams['pred_state']:
                optimizer[node_type] = optim.Adam([{'params': self.trajectron.model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                                   {'params': self.trajectron.model_registrar.get_name_match('map_encoder').parameters(), 
                                                    'lr': self.trajectron.hyperparams['map_enc_learning_rate']}], 
                                                  lr = self.trajectron.hyperparams['learning_rate'])
                # Set Learning Rate
                if self.trajectron.hyperparams['learning_rate_style'] == 'const':
                    gamma = 1.0
                elif self.trajectron.hyperparams['learning_rate_style'] == 'exp':
                    gamma = self.trajectron.hyperparams['learning_decay_rate'] 
                    
                lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                            gamma = gamma)
            else:
                raise TypeError('Missing model componenets')
        return optimizer, lr_scheduler

    def train_method(self, epochs = 100): 
        # Classify agents
        (Pred_agents, Agents, Types, S, S_st, 
         Neigh, Neigh_num, Neigh_type, Neigh_len, 
         img, img_m_per_px, Y, Y_st) = self.extract_data(train = True)
        Pred_types = np.unique(Types[Pred_agents])
        
        # Get gradient clipping values              
        clip_value_final = self.trajectron.hyperparams['grad_clip']
        
        # prepare training
        optimizer, lr_scheduler = self.prepare_model_training(Pred_types)
        
        # Generate training batches
        batch_size = self.trajectron.hyperparams['batch_size']   
        num, count = np.unique(self.num_timesteps_out, return_counts = True)
        num_batches = count // batch_size + 1
        batch_index_num = np.repeat(np.arange(len(num_batches)), num_batches)
        Index_num = [np.where(self.num_timesteps_out == n)[0] for n in num]
        Index_num_start = np.zeros(len(Index_num), int)
        
        # Get the current iteration of the model
        curr_iter = 0
        # go over epochs
        for epoch in range(1, epochs + 1):
            # print current epoch
            rjust_epoch = str(epoch).rjust(len(str(epochs)))
            print('Train trajectron: Epoch ' + rjust_epoch + '/{}'.format(epochs))
            
            # Move model to gpu
            self.trajectron.model_registrar.to(self.trajectron.device)
            
            # Randomly shuffle batches
            Index_num_start[:] = 0
            np.random.shuffle(batch_index_num)
            for i in range(len(Index_num)):
                np.random.shuffle(Index_num[i])
            
            for i_batch, batch in enumerate(batch_index_num):
                rjust_batch = str(i_batch + 1).rjust(len(str(len(batch_index_num))))
                # Get number of output steps in this batch
                num_steps = num[batch]
                if num_steps > 1:
                    Index_use = Index_num[batch][Index_num_start[batch]:Index_num_start[batch] + batch_size]
                    
                    if len(Index_use) > 1:
                        state_len = torch.from_numpy(np.ones(len(Index_use)) * S_st.shape[2])
                        fut_len   = torch.from_numpy(np.ones(len(Index_use)) * num_steps)
                        Index_num_start[batch] += 50
                        
                        print('Train trajectron: Epoch ' + rjust_epoch + '/{} - Batch '.format(epochs) + 
                              rjust_batch + '/{}'.format(len(batch_index_num)))
                        
                        
                        self.trajectron.set_curr_iter(curr_iter)
                        self.trajectron.step_annealers()
                        
                        i_pred_agent = 0
                        for i_agent, agent in enumerate(Agents):
                            if Pred_agents[i_agent]:
                                node_type = Types[i_agent]
                                optimizer[node_type].zero_grad()
                                
                                S_batch    = torch.from_numpy(S[Index_use,i_agent])
                                S_st_batch = torch.from_numpy(S_st[Index_use,i_agent])
                                Y_batch    = torch.from_numpy(Y[Index_use,i_agent,:num_steps])
                                Y_st_batch = torch.from_numpy(Y_st[Index_use,i_agent,:num_steps])
                                
                                if self.use_map:
                                    img_batch = torch.from_numpy(img[Index_use, i_pred_agent].astype(np.float32))
                                    img_batch = img_batch.to(device = self.trajectron.device) / 255
                                    res_batch = 1 / torch.from_numpy(img_m_per_px[Index_use, i_pred_agent])
                                else:
                                    img_batch = None
                                    res_batch = None
                                    
                                # Get batch data
                                Neigh_batch       = torch.from_numpy(Neigh[Index_use, i_pred_agent])
                                Neigh_types_batch = torch.from_numpy(Neigh_type[Index_use, i_pred_agent])
                                Neigh_num_batch   = torch.from_numpy(Neigh_num[Index_use, i_pred_agent])
                                Neigh_len_batch   = torch.from_numpy(Neigh_len[Index_use, i_pred_agent])
                                
                                Weights = list(self.trajectron.model_registrar.parameters())
                                model = self.trajectron.node_models_dict[node_type.name]
                                
                                # Built Agent_batch
                                batch = AgentBatch(dt = torch.ones(len(Index_use), dtype = torch.float32) * self.dt, 
                                                   agent_name = agent, 
                                                   agent_type = node_type, 
                                                   agent_hist = S_st_batch, 
                                                   agent_hist_len = state_len.to(dtype = torch.int64), 
                                                   agent_fut = Y_batch,
                                                   agent_fut_len = fut_len.to(dtype = torch.int64), 
                                                   robot_fut = None,
                                                   robot_fut_len = None,
                                                   num_neigh = Neigh_num_batch, 
                                                   neigh_types = Neigh_types_batch, 
                                                   neigh_hist = Neigh_batch, 
                                                   neigh_hist_len = Neigh_len_batch.to(dtype = torch.int64), 
                                                   maps = img_batch, 
                                                   maps_resolution = 1 / res_batch)
                                
                                # Run forward pass
                                batch.to(device = self.trajectron.device) 
                                train_loss = model.train_loss(batch = batch)
                
                                assert train_loss.isfinite().all(), "The overall loss of the model is nan"
                
                                train_loss.backward()
                 
                                gradients_good = all([(weights.grad.isfinite().all() if weights.grad is not None else True) 
                                                      for weights in Weights])
                                
                                if gradients_good:
                                    if self.trajectron.hyperparams['grad_clip'] is not None:
                                        nn.utils.clip_grad_value_(self.trajectron.model_registrar.parameters(), clip_value_final)
                    
                                    optimizer[node_type].step()
                                    lr_scheduler[node_type].step()
                                    curr_iter += 1
                                else:
                                    print('Too many output timesteps lead to exploding gradients => weights not updated')
                                
                                
                                i_pred_agent += 1
                    
                else:
                    print("Not enough output timesteps => no loss can be calculated")
                
                   
                    
                
        
        
        
        
        # save weigths 
        # after checking here, please return num_epochs to 100 and batch size to 
        Weights = list(self.trajectron.model_registrar.parameters())
        self.weights_saved = []
        for weigths in Weights:
            self.weights_saved.append(weigths.detach().cpu().numpy())
        
        
    def load_method(self, l2_regulization = 0):
        Weights = list(self.trajectron.model_registrar.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_saved):
                Weights[i][:] = torch.from_numpy(weights)[:]
        
    def predict_method(self):
        # get desired output length
        self.num_timesteps_out_test = np.zeros(len(self.Output_T_pred_test), int)
        for i_sample in range(len(self.Output_T_pred_test)):
            self.num_timesteps_out_test[i_sample] = len(self.Output_T_pred_test[i_sample])
            
        # Classify agents
        (Pred_agents, Agents, Types, S, S_st, 
         Neigh, Neigh_num, Neigh_type, Neigh_len, 
         img, img_m_per_px) = self.extract_data(train = False)
        
        
        Path_names = np.array([name for name in self.Output_path_train.columns])
        
        Output_Path = pd.DataFrame(np.empty((S.shape[0], Pred_agents.sum()), np.ndarray), 
                                   columns = Path_names[Pred_agents])
        
        nums = np.unique(self.num_timesteps_out_test)
        batch_size = self.trajectron.hyperparams['batch_size']
        
        samples_done = 0
        calculations_done = 0
        samples_all = len(S)
        calculations_all = np.sum(self.num_timesteps_out_test)
        for num in nums:
            Index_num = np.where(self.num_timesteps_out_test == num)[0]
            needed_max = 200
            
            batch_size_real = int(np.floor((batch_size * needed_max) / (num * self.num_samples_path_pred)))
            
            if batch_size_real > len(Index_num):
                Index_uses = [Index_num]
            else:
                Index_uses = [Index_num[i * batch_size_real : (i + 1) * batch_size_real] 
                              for i in range(int(np.ceil(len(Index_num)/ batch_size_real)))] 
            
            for Index_use in Index_uses:
                state_len = torch.from_numpy(np.ones(len(Index_use)) * S_st.shape[2])
                i_pred_agent = 0
                for i_agent, agent in enumerate(Agents):
                    if Pred_agents[i_agent]:
                        node_type = Types[i_agent]
                        
                        S_batch = torch.from_numpy(S[Index_use,i_agent])
                        S_st_batch = torch.from_numpy(S_st[Index_use,i_agent])
                            
                        if self.use_map:
                            img_batch = torch.from_numpy(img[Index_use, i_pred_agent].astype(np.float32))
                            img_batch = img_batch.to(device = self.trajectron.device) / 255
                            res_batch = 1 / torch.from_numpy(img_m_per_px[Index_use, i_pred_agent])
                        else:
                            img_batch = None
                            res_batch = None
                            
                        # Get batch data
                        Neigh_batch       = torch.from_numpy(Neigh[Index_use, i_pred_agent])
                        Neigh_types_batch = torch.from_numpy(Neigh_type[Index_use, i_pred_agent])
                        Neigh_num_batch   = torch.from_numpy(Neigh_num[Index_use, i_pred_agent])
                        Neigh_len_batch   = torch.from_numpy(Neigh_len[Index_use, i_pred_agent])
                        
                        batch = AgentBatch(dt = torch.ones(len(Index_use), dtype = torch.float32) * self.dt, 
                                           agent_name = agent, 
                                           agent_type = node_type, 
                                           agent_hist = S_st_batch, 
                                           agent_hist_len = state_len.to(dtype = torch.int64),
                                           agent_fut = None,
                                           agent_fut_len = None,  
                                           robot_fut = None,
                                           robot_fut_len = None,
                                           num_neigh = Neigh_num_batch, 
                                           neigh_types = Neigh_types_batch, 
                                           neigh_hist = Neigh_batch, 
                                           neigh_hist_len = Neigh_len_batch.to(dtype = torch.int64), 
                                           maps = img_batch, 
                                           maps_resolution = 1 / res_batch)
                        
                        batch.to(self.trajectron.device)
                        # Run prediction pass
                        model = self.trajectron.node_models_dict[node_type.name]
                        self.trajectron.model_registrar.to(self.trajectron.device)
                        
                        
                        with torch.no_grad(): # Do not build graph for backprop
                            predictions = model.predict(batch              = batch,
                                                        prediction_horizon = num,
                                                        num_samples        = self.num_samples_path_pred)
                        
                        Pred = predictions.detach().cpu().numpy()
                        torch.cuda.empty_cache()
                        for i, i_sample in enumerate(Index_use):
                            index = Types[i_agent].name[0] + '_' + Agents[i_agent]
                            Output_Path.iloc[i_sample][index] = Pred[:, i, :, :].astype('float32')
                        i_pred_agent += 1
                        
                samples_done += len(Index_use)
                calculations_done += len(Index_use) * num
                
                samples_perc = 100 * samples_done / samples_all
                calculations_perc = 100 * calculations_done / calculations_all
                
                print('Predict trajectron: ' + 
                      format(samples_perc, '.2f').rjust(len('100.00')) + 
                      '% of samples, ' + 
                      format(calculations_perc, '.2f').rjust(len('100.00')) +
                      '% of calculations')
                
                

        return [Output_Path]
    
    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
        
    def get_input_type(self = None):
        input_info = {'past': 'path',
                      'future': False}
        return input_info
    
    def get_name(self = None):
        names = {'print': 'Trajectron ++',
                 'file': 'trajectron',
                 'latex': r'\emph{T++}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True 