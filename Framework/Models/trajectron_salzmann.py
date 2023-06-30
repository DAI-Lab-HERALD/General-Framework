from model_template import model_template
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import random
from Trajectron.trajec_model.trajectron import Trajectron
from Trajectron.trajec_model.model_registrar import ModelRegistrar
from Trajectron.environment.environment import Environment
from attrdict import AttrDict

class trajectron_salzmann(model_template):
    
    def setup_method(self, seed = 0):
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        batch_size = 256
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
        self.target_width = 180
        self.target_height = 100
        
        
        if (np.array([name[0] for name in np.array(self.input_names_train)]) == 'P').all():
        
            hyperparams = {'batch_size': batch_size,
                           'grad_clip': 1.0,
                           'learning_rate_style': 'exp',
                           'learning_rate': 0.001, # for ETH: 0.001 for inD/rounD: 0.003,
                           'min_learning_rate': 1e-05,
                           'learning_decay_rate': 0.9999,
                           'prediction_horizon': self.num_timesteps_out.max(),
                           'minimum_history_length': 2,
                           'maximum_history_length': self.num_timesteps_in - 1,
                           'map_encoder': {'PEDESTRIAN': {'heading_state_index': 6,
                                                          'patch_size': [50, 10, 50, 90],
                                                          'map_channels': 3,
                                                          'hidden_channels': [10, 20, 10, 1],
                                                          'output_size': 32,
                                                          "masks": [5, 5, 5, 5], 
                                                          "strides": [1, 1, 1, 1], 
                                                          'dropout': 0.5}},
                           'k': 1,
                           'k_eval': 25,
                           'kl_min': 0.07,
                           'kl_weight': 100.0,
                           'kl_weight_start': 0,
                           'kl_decay_rate': 0.99995,
                           'kl_crossover': 400,
                           'kl_sigmoid_divisor': 4,
                           'rnn_kwargs': {'dropout_keep_prob': 0.75},
                           'MLP_dropout_keep_prob': 0.9,
                           'enc_rnn_dim_edge': 32,
                           'enc_rnn_dim_edge_influence': 32,
                           'enc_rnn_dim_history': 32,
                           'enc_rnn_dim_future': 32,
                           'dec_rnn_dim': 128,
                           'q_z_xy_MLP_dims': None,
                           'p_z_x_MLP_dims': 32,
                           'GMM_components': 1,
                           'log_p_yt_xz_max': 6,
                           'N': 1, # numbver of states per dimension of conditional distribution
                           'K': 25, # number of dimension of conditional distribution
                           'tau_init': 2.0,
                           'tau_final': 0.05,
                           'tau_decay_rate': 0.997,
                           'use_z_logit_clipping': True,
                           'z_logit_clip_start': 0.05,
                           'z_logit_clip_final': 5.0,
                           'z_logit_clip_crossover': 300,
                           'z_logit_clip_divisor': 5,
                           "dynamic": {"PEDESTRIAN": {"name": "SingleIntegrator",
                                                      "distribution": True,
                                                      "limits": {}}}, 
                           "state": {"PEDESTRIAN": {"position": ["x", "y"],
                                                    "velocity": ["x", "y"], 
                                                    "acceleration": ["x", "y"]}}, 
                           "pred_state": {"PEDESTRIAN": {"position": ["x", "y"]}},
                           'log_histograms': False,
                           'dynamic_edges': 'yes',
                           'edge_state_combine_method': 'sum',
                           'edge_influence_combine_method': 'attention',
                           'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
                           'edge_removal_filter': [1.0, 0.0],
                           'offline_scene_graph': 'yes',
                           'incl_robot_node': False,
                           'node_freq_mult_train': False,
                           'node_freq_mult_eval': False,
                           'scene_freq_mult_train': False,
                           'scene_freq_mult_eval': False,
                           'scene_freq_mult_viz': False,
                           'edge_encoding': True,
                           'use_map_encoding': self.use_map,
                           'augment': True,
                           'override_attention_radius': []}
        else:
            hyperparams = {'batch_size': batch_size,
                           'grad_clip': 1.0,
                           'learning_rate_style': 'exp',
                           'learning_rate': 0.003,
                           'min_learning_rate': 1e-05,
                           'learning_decay_rate': 0.9999,
                           'prediction_horizon': self.num_timesteps_out.max(),
                           'minimum_history_length': 2,
                           'maximum_history_length': self.num_timesteps_in - 1,
                           'map_encoder': {'VEHICLE': {'heading_state_index': 6,
                                                       'patch_size': [50, 10, 50, 90],
                                                       'map_channels': 3,
                                                       'hidden_channels': [10, 20, 10, 1],
                                                       'output_size': 32,
                                                       'masks': [5, 5, 5, 3],
                                                       'strides': [2, 2, 1, 1],
                                                       'dropout': 0.5}},
                           'k': 1,
                           'k_eval': 25,
                           'kl_min': 0.07,
                           'kl_weight': 100.0,
                           'kl_weight_start': 0,
                           'kl_decay_rate': 0.99995,
                           'kl_crossover': 400,
                           'kl_sigmoid_divisor': 4,
                           'rnn_kwargs': {'dropout_keep_prob': 0.75},
                           'MLP_dropout_keep_prob': 0.9,
                           'enc_rnn_dim_edge': 32,
                           'enc_rnn_dim_edge_influence': 32,
                           'enc_rnn_dim_history': 32,
                           'enc_rnn_dim_future': 32,
                           'dec_rnn_dim': 128,
                           'q_z_xy_MLP_dims': None,
                           'p_z_x_MLP_dims': 32,
                           'GMM_components': 1,
                           'log_p_yt_xz_max': 6,
                           'N': 1, # numbver of states per dimension of conditional distribution
                           'K': 25, # number of dimension of conditional distribution
                           'tau_init': 2.0,
                           'tau_final': 0.05,
                           'tau_decay_rate': 0.997,
                           'use_z_logit_clipping': True,
                           'z_logit_clip_start': 0.05,
                           'z_logit_clip_final': 5.0,
                           'z_logit_clip_crossover': 300,
                           'z_logit_clip_divisor': 5,
                           'dynamic': {'PEDESTRIAN': {'name': 'SingleIntegrator',
                                                      'distribution': True,
                                                      'limits': {}},
                                       'VEHICLE': {'name': 'Unicycle',
                                                   'distribution': True,
                                                   'limits': {'max_a': 4,
                                                              'min_a': -5,
                                                              'max_heading_change': 0.7,
                                                              'min_heading_change': -0.7}}},
                           'state': {'PEDESTRIAN': {'position': ['x', 'y'],
                                                    'velocity': ['x', 'y'],
                                                    'acceleration': ['x', 'y']},
                                     'VEHICLE': {'position': ['x', 'y'],
                                                 'velocity': ['x', 'y'],
                                                 'acceleration': ['x', 'y'],
                                                 'heading': ['°', 'd°']}},
                           'pred_state': {'VEHICLE': {'position': ['x', 'y']},
                                          'PEDESTRIAN': {'position': ['x', 'y']}},
                           'log_histograms': False,
                           'dynamic_edges': 'yes',
                           'edge_state_combine_method': 'sum',
                           'edge_influence_combine_method': 'attention',
                           'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
                           'edge_removal_filter': [1.0, 0.0],
                           'offline_scene_graph': 'yes',
                           'incl_robot_node': False,
                           'node_freq_mult_train': False,
                           'node_freq_mult_eval': False,
                           'scene_freq_mult_train': False,
                           'scene_freq_mult_eval': False,
                           'scene_freq_mult_viz': False,
                           'edge_encoding': True,
                           'use_map_encoding': self.use_map,
                           'augment': True,
                           'override_attention_radius': []}
        
        self.std_pos_ped = 1
        self.std_vel_ped = 2
        self.std_acc_ped = 1
        self.std_pos_veh = 80
        self.std_vel_veh = 15
        self.std_acc_veh = 4
        self.std_hea_veh = np.pi
        self.std_d_h_veh = 1
        
        
        # Set time step
        self.dt = self.Input_T_train[0][-1] - self.Input_T_train[0][-2]
        
        # Offline Calculate Scene Graph
        model_registrar = ModelRegistrar(None, self.device)
        self.trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     None,
                                     self.device)
        
        # Set train environment
        scenes = [AttrDict({'dt': self.dt})]
        
        
        if self.data_set.get_name()['file'][:3] == 'ETH':
            node_type_list = ['PEDESTRIAN']
        else:
            node_type_list = ['PEDESTRIAN', 'VEHICLE']
        
        train_env = Environment(node_type_list = node_type_list,
                                standardization = None,
                                scenes = scenes,
                                attention_radius = None, 
                                robot_type = None)
        
        # Prepare models
        self.trajectron.set_environment(train_env)
        self.trajectron.set_annealing_params()
        
    def extract_data(self, DIM, train = True):
        attention_radius = dict()

        # for ETH
        if self.data_set.get_name()['file'][:3] == 'ETH':
            attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 3.0
        else:
            # for inD/rounD
            attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
            attention_radius[('PEDESTRIAN', 'VEHICLE')] = 50.0
            attention_radius[('VEHICLE', 'PEDESTRIAN')] = 25.0
            attention_radius[('VEHICLE', 'VEHICLE')] = 150.0
            
        
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
        Types  = np.array(['PEDESTRIAN' if name[0] == 'P' else 'VEHICLE'  for name in np.array(self.input_names_train)])
        
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
        
        H = np.angle(V[...,0] + 1j * V[...,1])
        
        DH = np.unwrap(H, axis = -1) 
        DH = (DH[:,:,1:] - DH[:,:,:-1]) / self.dt
        DH = np.concatenate((DH[:,:,[0]], DH), axis = -1)
        
        #final state S
        S = np.concatenate((X, V, A, H[...,np.newaxis], DH[...,np.newaxis]), axis = -1).astype(np.float32)
        
        D = np.min(np.sqrt(np.sum((X[:,:,np.newaxis] - X[:,np.newaxis]) ** 2, axis = -1)), axis = - 1)
        D_max = np.array([[attention_radius[(Types[j_v], Types[i_v])] 
                           for j_v in range(X.shape[1])] for i_v in range(X.shape[1])])
        
        Neighbor_bool = D < D_max[np.newaxis]
        
        Neighbors = dict()
        Neighbor_edge_values = dict()
        
        # Get Neighbor for each pred value
        
        for i_agent, agent in enumerate(Agents):
            avoid_self = (np.arange(len(Agents)) != i_agent)
            if Pred_agents[i_agent]:
                node_pred = Types[i_agent]
                
                Neighbors[agent] = dict()
                Neighbor_edge_values[agent] = dict()
                for node_goal in DIM.keys():
                    feasible_goals = (Types == node_goal) & avoid_self
                    
                    key = (node_pred, node_goal)
                    Neighbors[agent][key] = []
                    Neighbor_edge_values[agent][key] = []
                    Dim = DIM[node_goal]
                    
                    for i_sample in range(S.shape[0]):
                        Neighbors[agent][key].append([])
                        
                        I_agent_goal = np.where(Neighbor_bool[i_sample, i_agent] & feasible_goals)[0]
                        
                        Neighbor_edge_values[agent][key].append(torch.from_numpy(np.ones(len(I_agent_goal), np.float32)))
                        for i_agent_goal in I_agent_goal:
                            Neighbors[agent][key][i_sample].append(torch.from_numpy(S[i_sample, i_agent_goal, :, :Dim]))
        
        if self.use_map:
            centre = X[:,Pred_agents,-1,:].reshape(-1, 2)
            x_rel = centre - X[:,Pred_agents,-2,:].reshape(-1, 2)
            rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 
            domain_repeat = self.domain_old.loc[self.domain_old.index.repeat(Pred_agents.sum())]
            
            img = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                    target_height = self.target_height, 
                                                    target_width = self.target_width, grayscale = False)
            
            img = img[:,:,80:].transpose(0,3,1,2).reshape(X.shape[0], Pred_agents.sum(), 3, 
                                                          self.target_height, self.target_width - 80)
            
        else:
            img = None
            
            
        Ped_agents = Types == 'PEDESTRIAN'
        
        S_st = S.copy()
        S_st[:,Ped_agents,:,0:2]  /= self.std_pos_ped
        S_st[:,~Ped_agents,:,0:2] /= self.std_pos_veh
        S_st[:,Ped_agents,:,2:4]  /= self.std_vel_ped
        S_st[:,~Ped_agents,:,2:4] /= self.std_vel_veh
        S_st[:,Ped_agents,:,4:6]  /= self.std_acc_ped
        S_st[:,~Ped_agents,:,4:6] /= self.std_acc_veh
        S_st[:,~Ped_agents,:,6]   /= self.std_hea_veh
        S_st[:,~Ped_agents,:,7]   /= self.std_d_h_veh
        
        if train:
            Y_st = Y.copy()
            Y_st[:,Ped_agents]  /= self.std_pos_ped
            Y_st[:,~Ped_agents] /= self.std_pos_veh
            return Pred_agents, Agents, Types, S, S_st, Neighbors, Neighbor_edge_values, img, Y, Y_st
        else:
            return Pred_agents, Agents, Types, S, S_st, Neighbors, Neighbor_edge_values, img
    
    def prepare_model_training(self, Pred_types):
        optimizer = dict()
        lr_scheduler = dict()
        for node_type in Pred_types:
            if node_type  in self.trajectron.hyperparams['pred_state']:
                optimizer[node_type] = optim.Adam([{'params': self.trajectron.model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                                   {'params': self.trajectron.model_registrar.get_name_match('map_encoder').parameters(), 
                                                    'lr': 0.0008}], 
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
        # prepare input data
        
        DIM = {'VEHICLE': 8, 'PEDESTRIAN': 6}
        
        # Classify agents
        Pred_agents, Agents, Types, S, S_st, Neighbors, Neighbor_edge_values, img, Y, Y_st = self.extract_data(DIM, train = True)
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
                        first_h = torch.from_numpy(np.zeros(len(Index_use), np.int32))
                        Index_num_start[batch] += 50
                        
                        print('Train trajectron: Epoch ' + rjust_epoch + '/{} - Batch '.format(epochs) + 
                              rjust_batch + '/{}'.format(len(batch_index_num)))
                        
                        
                        self.trajectron.set_curr_iter(curr_iter)
                        
                        i_pred_agent = 0
                        for i_agent, agent in enumerate(Agents):
                            if Pred_agents[i_agent]:
                                node_type = str(Types[i_agent])
                                self.trajectron.step_annealers(node_type)
                                optimizer[node_type].zero_grad()
                                
                                S_batch    = torch.from_numpy(S[Index_use,i_agent,:,:DIM[node_type]])
                                S_st_batch = torch.from_numpy(S_st[Index_use,i_agent,:,:DIM[node_type]])
                                Y_batch    = torch.from_numpy(Y[Index_use,i_agent,:num_steps])
                                Y_st_batch = torch.from_numpy(Y_st[Index_use,i_agent,:num_steps])
                                
                                if self.use_map:
                                    img_batch = torch.from_numpy(img[Index_use, i_pred_agent].astype(np.float32))
                                    img_batch = img_batch.to(device = self.trajectron.device) / 255
                                else:
                                    img_batch = None
                                    
                                # Get batch data
                                Neighbor_batch = {}
                                Neighbor_edge_value_batch = {}
                                for node_goal in DIM.keys():
                                    key = (node_type, node_goal)
                                    Neighbor_batch[key] = []
                                    Neighbor_edge_value_batch[key] = []
                                    
                                    for i_sample in Index_use:
                                        Neighbor_batch[key].append(Neighbors[agent][key][i_sample])
                                        Neighbor_edge_value_batch[key].append(Neighbor_edge_values[agent][key][i_sample]) 
                                
                                # Get Weights and model
                                Weights = list(self.trajectron.model_registrar.parameters())
                                model = self.trajectron.node_models_dict[node_type]
                                
                                # Run forward pass
                                train_loss = model.train_loss(inputs                = S_batch.to(self.trajectron.device),
                                                              inputs_st             = S_st_batch.to(self.trajectron.device),
                                                              first_history_indices = first_h.to(self.trajectron.device),
                                                              labels                = Y_batch.to(self.trajectron.device),
                                                              labels_st             = Y_st_batch.to(self.trajectron.device),
                                                              neighbors             = Neighbor_batch,
                                                              neighbors_edge_value  = Neighbor_edge_value_batch,       
                                                              robot                 = None,
                                                              map                   = img_batch,
                                                              prediction_horizon    = num_steps)
                
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
            
        DIM = {'VEHICLE': 8, 'PEDESTRIAN': 6}
        Pred_agents, Agents, Types, S, S_st, Neighbors, Neighbor_edge_values, img = self.extract_data(DIM, train = False)
        
        
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
                i_pred_agent = 0
                for i_agent, agent in enumerate(Agents):
                    if Pred_agents[i_agent]:
                        node_type = str(Types[i_agent])
                        
                        S_batch = torch.from_numpy(S[Index_use,i_agent,:,:DIM[node_type]])
                        S_st_batch = torch.from_numpy(S_st[Index_use,i_agent,:,:DIM[node_type]])
                            
                        if self.use_map:
                            img_batch = torch.from_numpy(img[Index_use, i_pred_agent].astype(np.float32))
                            img_batch = img_batch.to(device = self.trajectron.device) / 255
                        else:
                            img_batch = None
                            
                        # Get batch data
                        Neighbor_batch = {}
                        Neighbor_edge_value_batch = {}
                        for node_goal in DIM.keys():
                            key = (node_type,node_goal)
                            Neighbor_batch[key] = []
                            Neighbor_edge_value_batch[key] = []
                            
                            for i_sample in Index_use:
                                Neighbor_batch[key].append(Neighbors[agent][key][i_sample])
                                Neighbor_edge_value_batch[key].append(Neighbor_edge_values[agent][key][i_sample]) 
                        
                        first_h = torch.from_numpy(np.zeros(len(Index_use), np.int32))
                        # Run prediction pass
                        model = self.trajectron.node_models_dict[node_type]
                        
                        self.trajectron.model_registrar.to(self.trajectron.device)
                        with torch.no_grad(): # Do not build graph for backprop
                            predictions = model.predict(inputs                = S_batch.to(self.trajectron.device),
                                                        inputs_st             = S_st_batch.to(self.trajectron.device),
                                                        first_history_indices = first_h.to(self.trajectron.device),
                                                        neighbors             = Neighbor_batch,
                                                        neighbors_edge_value  = Neighbor_edge_value_batch,
                                                        robot                 = None,
                                                        map                   = img_batch,
                                                        prediction_horizon    = num,
                                                        num_samples           = self.num_samples_path_pred)
                        
                        Pred = predictions.detach().cpu().numpy()
                        torch.cuda.empty_cache()
                          
                        for i, i_sample in enumerate(Index_use):
                            index = Types[i_agent][0] + '_' + Agents[i_agent]
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