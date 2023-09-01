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

class trajectron_salzmann_unicycle(model_template):
    '''
    This is the updated version of Trajectron++, a single agent prediction model
    that is mainly based on LSTM cells. In its decoder, it just uses a unicycle model 
    to transform predicted control inputs into trajectories, identically to 
    the original model.
    
    The code was taken from https://github.com/NVlabs/adaptive-prediction/tree/main/src/trajectron
    and the model is published under the following citation:
        
    Ivanovic, B., Harrison, J., & Pavone, M. (2023, May). Expanding the deployment envelope 
    of behavior prediction via adaptive meta-learning. In 2023 IEEE International Conference 
    on Robotics and Automation (ICRA) (pp. 7786-7793). IEEE.
    '''
    def setup_method(self, seed = 0):
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Required attributes of the model
        self.min_t_O_train = 3
        self.max_t_O_train = 100
        self.predict_single_agent = True
        self.can_use_map = True
        # If self.can_use_map = True, the following is also required
        self.target_width = 175
        self.target_height = 100
        self.grayscale = False
        
        config_path = self.data_set.path + os.sep + 'Models' + os.sep + 'Trajectron' + os.sep + 'config' + os.sep
        
        if (self.provide_all_included_agent_types() == 'P').all():
            config_file = config_path + 'pedestrian.json' 
            with open(config_file) as json_file:
                hyperparams = json.load(json_file)
        else:
            config_file = config_path + 'nuScenes.json' 
            with open(config_file) as json_file:
                hyperparams = json.load(json_file)
            
        hyperparams["dec_final_dim"]                 = 32
        
        hyperparams["map_encoding"]                  = self.can_use_map
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
        
        # Prepare models
        model_registrar = ModelRegistrar(None, self.device)
        self.trajectron = Trajectron(model_registrar, hyperparams, None, self.device)
        self.trajectron.set_environment()
        self.trajectron.set_annealing_params()
      
        
    def rotate_pos_matrix(self, M, rot_angle):
        assert M.shape[-1] == 2
        assert M.shape[0] == len(rot_angle)
        
        R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                      [np.sin(rot_angle),  np.cos(rot_angle)]]).transpose(2,0,1)
        R = R[:,np.newaxis]
        
        M_r = np.matmul(M, R)
        return M_r
        
    
    def extract_data_batch(self, X, T, Y = None, img = None, img_m_per_px = None, num_steps = 10):
        attention_radius = dict()
        
        if (self.provide_all_included_agent_types() == 'P').all():
            attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 3.0
        else:
            attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 5.0
            attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)]    = 25.0
            attention_radius[(AgentType.PEDESTRIAN, AgentType.BICYCLE)]    = 10.0
            attention_radius[(AgentType.PEDESTRIAN, AgentType.MOTORCYCLE)] = 25.0
            attention_radius[(AgentType.VEHICLE,    AgentType.PEDESTRIAN)] = 25.0
            attention_radius[(AgentType.VEHICLE,    AgentType.VEHICLE)]    = 75.0
            attention_radius[(AgentType.VEHICLE,    AgentType.BICYCLE)]    = 50.0
            attention_radius[(AgentType.VEHICLE,    AgentType.MOTORCYCLE)] = 75.0
            attention_radius[(AgentType.BICYCLE,    AgentType.PEDESTRIAN)] = 10.0
            attention_radius[(AgentType.BICYCLE,    AgentType.VEHICLE)]    = 50.0
            attention_radius[(AgentType.BICYCLE,    AgentType.BICYCLE)]    = 10.0
            attention_radius[(AgentType.BICYCLE,    AgentType.MOTORCYCLE)] = 50.0
            attention_radius[(AgentType.MOTORCYCLE, AgentType.PEDESTRIAN)] = 25.0
            attention_radius[(AgentType.MOTORCYCLE, AgentType.VEHICLE)]    = 75.0
            attention_radius[(AgentType.MOTORCYCLE, AgentType.BICYCLE)]    = 50.0
            attention_radius[(AgentType.MOTORCYCLE, AgentType.MOTORCYCLE)] = 75.0
            
        Types = np.empty(T.shape, dtype = AgentType)
        Types[T == 'P'] = AgentType.PEDESTRIAN
        Types[T == 'V'] = AgentType.VEHICLE
        Types[T == 'B'] = AgentType.BICYCLE
        Types[T == 'M'] = AgentType.MOTORCYCLE
        
        center_pos = X[:,0,-1]
        delta_x = center_pos - X[:,0,-2]
        rot_angle = np.angle(delta_x[:,0] + 1j * delta_x[:,1])

        center_pos = center_pos[:,np.newaxis,np.newaxis]        
        X_r = self.rotate_pos_matrix(X - center_pos, rot_angle)
        
        
        V = (X_r[...,1:,:] - X_r[...,:-1,:]) / self.dt
        V = np.concatenate((V[...,[0],:], V), axis = -2)
       
        # get accelaration
        A = (V[...,1:,:] - V[...,:-1,:]) / self.dt
        A = np.concatenate((A[...,[0],:], A), axis = -2)
       
        H = np.angle(V[...,0] + 1j * V[...,1])[...,np.newaxis]
       
        Sin = np.sin(H)
        Cos = np.cos(H)
       
        #final state S
        S = np.concatenate((X_r, V, A, Sin, Cos), axis = -1).astype(np.float32)
        
        Ped_agents = Types == AgentType.PEDESTRIAN
        
        S_st = S.copy()
        S_st[Ped_agents,:,0:2]  /= self.std_pos_ped
        S_st[~Ped_agents,:,0:2] /= self.std_pos_veh
        S_st[Ped_agents,:,2:4]  /= self.std_vel_ped
        S_st[~Ped_agents,:,2:4] /= self.std_vel_veh
        S_st[Ped_agents,:,4:6]  /= self.std_acc_ped
        S_st[~Ped_agents,:,4:6] /= self.std_acc_veh
        
        D = np.min(np.sqrt(np.sum((X[:,[0]] - X) ** 2, axis = -1)), axis = - 1)
        D_max = np.zeros_like(D)
        for i_sample in range(len(D)):
            for i_v in range(X.shape[1]):
                if Types[i_sample, i_v] is None:
                    continue
                D_max[i_sample, i_v] = attention_radius[(Types[i_sample, 0], Types[i_sample, i_v])]
        
        Neighbor_bool = D < D_max
        
        # Get Neighbor for each pred value
        Neigh      = np.nan * np.ones((X.shape[0], *S.shape[1:]), dtype = np.float32)
        Neigh_num  = np.zeros(X.shape[0], dtype = np.int64)
        Neigh_type = np.zeros((X.shape[0], X.shape[1]), dtype = int)
        Neigh_len  = np.zeros((X.shape[0], X.shape[1]), dtype = int)

        avoid_self = np.arange(X.shape[1]) != 0
        feasible_goals = avoid_self[np.newaxis] & Neighbor_bool
        Neigh_num = feasible_goals.sum(-1) 
        for i_sample in range(X.shape[0]):
            Neigh[i_sample, :Neigh_num[i_sample]] = S_st[i_sample, feasible_goals[i_sample]]
            Neigh_type[i_sample, :Neigh_num[i_sample]] = Types[i_sample, feasible_goals[i_sample]].astype(int)
            Neigh_len[i_sample, :Neigh_num[i_sample]] = self.num_timesteps_in
        
        num_batch_samples = len(X)
        
        node_type = Types[0,0]
        
        if img is not None:
            img_batch = img[:,0,:,75:].astype(np.float32) / 255 # Cut of image behind vehicle
            img_batch = img_batch.transpose(0,3,1,2) # put channels first
            img_batch = torch.from_numpy(img_batch).to(dtype = torch.float32)
            res_batch = 1 / torch.from_numpy(img_m_per_px[:,0])
        else:
            img_batch = None
            res_batch = None
            
        
        if node_type == AgentType.PEDESTRIAN:
            pos_to_vel_fac = self.std_vel_ped / self.std_pos_ped
        elif node_type == AgentType.VEHICLE:
            pos_to_vel_fac = self.std_vel_veh / self.std_pos_veh    
        else:
            pos_to_vel_fac = self.std_vel_veh / self.std_pos_veh 
            
        # Only take out prediction agent
        S_st_tar = S_st[:,0]
        
        if Y is None:
            batch = AgentBatch(dt              = torch.ones(num_batch_samples, dtype = torch.float32) * self.dt, 
                               agent_type      = node_type,
                               pos_to_vel_fac  = pos_to_vel_fac,
                               agent_hist      = torch.from_numpy(S_st_tar).to(dtype = torch.float32), 
                               agent_hist_len  = torch.ones(num_batch_samples).to(dtype = torch.int64) * S_st_tar.shape[2], 
                               agent_fut       = None,
                               agent_fut_len   = None, 
                               robot_fut       = None,
                               robot_fut_len   = None,
                               num_neigh       = torch.from_numpy(Neigh_num).to(dtype = torch.int64), 
                               neigh_types     = torch.from_numpy(Neigh_type).to(dtype = torch.int64), 
                               neigh_hist      = torch.from_numpy(Neigh).to(dtype = torch.float32), 
                               neigh_hist_len  = torch.from_numpy(Neigh_len).to(dtype = torch.int64), 
                               maps            = img_batch, 
                               maps_resolution = res_batch)
        else:
            Y_st = self.rotate_pos_matrix(Y - center_pos, rot_angle).copy()
            Y_st[Ped_agents[:,0]]  /= self.std_pos_ped
            Y_st[~Ped_agents[:,0]] /= self.std_pos_veh
        
            # Only take out prediction agent
            Y_st = Y_st[:,0]
            
            batch = AgentBatch(dt              = torch.ones(num_batch_samples, dtype = torch.float32) * self.dt, 
                               agent_type      = node_type,
                               pos_to_vel_fac  = pos_to_vel_fac,
                               agent_hist      = torch.from_numpy(S_st_tar).to(dtype = torch.float32), 
                               agent_fut       = torch.from_numpy(Y_st).to(dtype = torch.float32),
                               # Todo: Generate better occupancy maps
                               agent_hist_len  = torch.ones(num_batch_samples).to(dtype = torch.int64) * S_st_tar.shape[2], 
                               agent_fut_len   = torch.ones(num_batch_samples).to(dtype = torch.int64) * num_steps, 
                               robot_fut       = None,
                               robot_fut_len   = None,
                               num_neigh       = torch.from_numpy(Neigh_num).to(dtype = torch.int64), 
                               neigh_types     = torch.from_numpy(Neigh_type).to(dtype = torch.int64), 
                               neigh_hist      = torch.from_numpy(Neigh).to(dtype = torch.float32), 
                               neigh_hist_len  = torch.from_numpy(Neigh_len).to(dtype = torch.int64), 
                               maps            = img_batch, 
                               maps_resolution = res_batch)
        return batch, node_type, center_pos, rot_angle
    
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
        # setup train_loss
        self.train_loss = np.zeros((1, epochs))
        
        
        T_all = self.provide_all_included_agent_types()
        Pred_types = np.empty(T_all.shape, dtype = AgentType)
        Pred_types[T_all == 'P'] = AgentType.PEDESTRIAN
        Pred_types[T_all == 'V'] = AgentType.VEHICLE
        Pred_types[T_all == 'B'] = AgentType.BICYCLE
        Pred_types[T_all == 'M'] = AgentType.MOTORCYCLE
        
        # Get gradient clipping values              
        clip_value_final = self.trajectron.hyperparams['grad_clip']
        
        # prepare training
        optimizer, lr_scheduler = self.prepare_model_training(Pred_types)
        
        # Generate training batches
        batch_size = self.trajectron.hyperparams['batch_size']   
        
        # Move model to gpu
        self.trajectron.model_registrar.to(self.trajectron.device)
        
        # Get the current iteration of the model
        curr_iter = 0
        
        for epoch in range(1, epochs + 1):
            # print current epoch
            rjust_epoch = str(epoch).rjust(len(str(epochs)))
            print('Train trajectron: Epoch ' + rjust_epoch + '/{}'.format(epochs))
            
            epoch_loss = 0.0
            epoch_done = False
            
            batch_number = 0
            while not epoch_done:
                batch_number += 1
                print('Train trajectron: Epoch ' + rjust_epoch + '/{} - Batch {}'.format(epochs, batch_number))
                X, Y, T, img, img_m_per_px, _, num_steps, epoch_done = self.provide_batch_data('train', batch_size)
                
                batch, node_type, _, _ = self.extract_data_batch(X, T, Y, img, img_m_per_px, num_steps)
                
                batch.to(device = self.trajectron.device)
                
                self.trajectron.set_curr_iter(curr_iter)
                self.trajectron.step_annealers()
                
                optimizer[node_type].zero_grad()
                
                # Run forward pass
                model = self.trajectron.node_models_dict[node_type.name]
                train_loss = model.train_loss(batch = batch)
                
                # Calculate gradients
                assert train_loss.isfinite().all(), "The overall loss of the model is nan"
                train_loss.backward()
                
                if self.trajectron.hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(self.trajectron.model_registrar.parameters(), clip_value_final)
                
                optimizer[node_type].step()
                lr_scheduler[node_type].step()
                curr_iter += 1
                
                epoch_loss += train_loss.detach().cpu().numpy()
  
            self.train_loss[0, epoch - 1] = epoch_loss       
                    
              
        # save weigths 
        
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
        batch_size = max(1, int(self.trajectron.hyperparams['batch_size'] / 10))
        
        prediction_done = False
        
        batch_number = 0
        while not prediction_done:
            batch_number += 1
            print('Predict trajectron: Batch {}'.format(batch_number))
            X, T, img, img_m_per_px, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', batch_size)
            batch, node_type, center_pos, rot_angle = self.extract_data_batch(X, T, None, img, img_m_per_px, num_steps)
        
            
            torch.cuda.empty_cache()
            batch.to(self.trajectron.device)
            # Run prediction pass
            model = self.trajectron.node_models_dict[node_type.name]
            self.trajectron.model_registrar.to(self.trajectron.device)
            
            with torch.no_grad():
                predictions = model.predict(batch              = batch,
                                            prediction_horizon = num_steps,
                                            num_samples        = self.num_samples_path_pred)
            
            Pred = predictions.detach().cpu().numpy()
            if node_type == AgentType.PEDESTRIAN:
                Pred *= self.std_pos_ped
            elif node_type == AgentType.VEHICLE:
                Pred *= self.std_pos_veh
            else:
                raise TypeError('The agent type ' + str(node_type.name) + ' is currently not implemented.')
                
            
            
            # set batchsize first
            Pred = Pred.transpose(1,0,2,3)
            
            # reverse rotation
            Pred_r = self.rotate_pos_matrix(Pred, -rot_angle)
            
            # reverse translation
            Pred_t = Pred_r + center_pos
            
            self.save_predicted_batch_data(Pred_t, Sample_id, Agent_id)
    
    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_name(self = None):
        names = {'print': 'Trajectron ++ (Dynamic model: Unicycle)',
                 'file': 'traject_UC',
                 'latex': r'\emph{T++}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True 
        
    def provides_epoch_loss(self = None):
        return True