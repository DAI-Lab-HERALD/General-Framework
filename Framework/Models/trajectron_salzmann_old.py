from model_template import model_template
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import random
from Trajectron_old.trajec_model.trajectron import Trajectron
from Trajectron_old.trajec_model.model_registrar import ModelRegistrar
from Trajectron_old.environment.environment import Environment
from attrdict import AttrDict
import warnings

class trajectron_salzmann_old(model_template):
    '''
    This is the orignial version of Trajectron++, a single agent prediction model
    that is mainly based on LSTM cells.
    
    The code was taken from https://github.com/StanfordASL/Trajectron-plus-plus/tree/master
    and the model is published under the following citation:
        
    Salzmann, T., Ivanovic, B., Chakravarty, P., & Pavone, M. (2020). Trajectron++: Dynamically-
    feasible trajectory forecasting with heterogeneous data. In Computer Vision–ECCV 2020: 
    16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVIII 16 
    (pp. 683-700). Springer International Publishing.
    '''
    def define_default_kwargs(self):
        if not('seed' in self.model_kwargs.keys()):
            self.model_kwargs['seed'] = 0

        if not('predict_ego' in self.model_kwargs.keys()):
            self.model_kwargs['predict_ego'] = True
    
    def setup_method(self):
        self.define_default_kwargs()
        # set random seeds
        seed = self.model_kwargs['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        batch_size = 256
        
        # Required attributes of the model
        self.min_t_O_train = 3
        self.max_t_O_train = 100
        self.predict_single_agent = True
        self.can_use_map = True
        self.can_use_graph = False
        # If self.can_use_map = True, the following is also required
        self.target_width = 175
        self.target_height = 100
        self.grayscale = False
        
        self.use_map = self.can_use_map and self.has_map
        
        if (self.provide_all_included_agent_types() == 'P').all():
            hyperparams = {'batch_size': batch_size,
                           'grad_clip': 1.0,
                           'learning_rate_style': 'exp',
                           'learning_rate': 0.001, # for ETH: 0.001 for inD/rounD: 0.003,
                           'min_learning_rate': 1e-05,
                           'learning_decay_rate': 0.9999,
                           'prediction_horizon': self.num_timesteps_out,
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
                           'prediction_horizon': self.num_timesteps_out,
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
        
        
        # Offline Calculate Scene Graph
        model_registrar = ModelRegistrar(None, self.device)
        self.trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     None,
                                     self.device)
        
        # Set train environment
        scenes = [AttrDict({'dt': self.dt})]
        
        
        if (self.provide_all_included_agent_types() == 'P').all():
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
    
    

    # Data extraction in numpy
    def rotate_pos_matrix(self, M, rot_angle):
        assert M.shape[-1] == 2
        assert M.shape[0] == len(rot_angle)
        
        R = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                      [np.sin(rot_angle),  np.cos(rot_angle)]]).transpose(2,0,1)
        R = R[:,np.newaxis]
        
        M_r = np.matmul(M, R)
        return M_r
    
    
    def extract_data_batch(self, X, T, Y = None, img = None, num_steps = 10):
        attention_radius = dict()
        DIM = {'VEHICLE': 8, 'PEDESTRIAN': 6}
        
        if (self.provide_all_included_agent_types() == 'P').all():
            attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 3.0
        else:
            attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
            attention_radius[('PEDESTRIAN', 'VEHICLE')]    = 50.0
            attention_radius[('VEHICLE',    'PEDESTRIAN')] = 25.0
            attention_radius[('VEHICLE',    'VEHICLE')]    = 150.0
            
        Types = np.empty(T.shape, dtype = object)
        Types[T == 'P'] = 'PEDESTRIAN'
        Types[T == 'V'] = 'VEHICLE'
        Types[T == 'B'] = 'VEHICLE'
        Types[T == 'M'] = 'VEHICLE'
        Types = Types.astype(str)
        
        center_pos = X[:,0,-1]
        delta_x = center_pos - X[:,0,-2]
        rot_angle = np.angle(delta_x[:,0] + 1j * delta_x[:,1])

        center_pos = center_pos[:,np.newaxis,np.newaxis]        
        X_r = self.rotate_pos_matrix(X - center_pos, rot_angle)
        
        
        V = (X_r[...,1:,:] - X_r[...,:-1,:]) / self.dt
        V = np.concatenate((V[...,[0],:], V), axis = -2)

        overwrite_V = np.isnan(V).all(-1) & (~np.isnan(X_r).all(-1))
        assert overwrite_V.sum(-1).max() <= 1, "Velocity interpolation failed."

        if overwrite_V.any():
            OV_s, OV_a, OV_t = np.where(overwrite_V)
            OV_use = OV_t < V.shape[2] - 1
            V[OV_s[OV_use], OV_a[OV_use], OV_t[OV_use]] = V[OV_s[OV_use], OV_a[OV_use], OV_t[OV_use] + 1]
            V[OV_s[~OV_use], OV_a[~OV_use], OV_t[~OV_use]] = 0.0
       
        # get accelaration
        A = (V[...,1:,:] - V[...,:-1,:]) / self.dt
        A = np.concatenate((A[...,[0],:], A), axis = -2)

        overwrite_A = np.isnan(A).all(-1) & (~np.isnan(V).all(-1))
        assert overwrite_A.sum(-1).max() <= 1, "Acceleration interpolation failed."

        if overwrite_A.any():
            OA_s, OA_a, OA_t = np.where(overwrite_A)
            OA_use = OA_t < A.shape[2] - 1
            A[OA_s[OA_use], OA_a[OA_use], OA_t[OA_use]] = A[OA_s[OA_use], OA_a[OA_use], OA_t[OA_use] + 1]
            A[OA_s[~OA_use], OA_a[~OA_use], OA_t[~OA_use]] = 0.0
       
        H = np.arctan2(V[:,:,:,1], V[:,:,:,0])
        
        DH = H.copy()
        DH[np.isfinite(H)] = np.unwrap(H[np.isfinite(H)], axis = -1) 
        DH = (DH[:,:,1:] - DH[:,:,:-1]) / self.dt
        DH = np.concatenate((DH[...,[0]], DH), axis = -1)

        overwrite_DH = np.isnan(DH) & (~np.isnan(H))
        assert overwrite_DH.sum(-1).max() <= 1, "Heading change interpolation failed."

        if overwrite_DH.any():
            ODH_s, ODH_a, ODH_t = np.where(overwrite_DH)
            ODH_use = ODH_t < DH.shape[2] - 1
            DH[ODH_s[ODH_use], ODH_a[ODH_use], ODH_t[ODH_use]] = DH[ODH_s[ODH_use], ODH_a[ODH_use], ODH_t[ODH_use] + 1]
            DH[ODH_s[~ODH_use], ODH_a[~ODH_use], ODH_t[~ODH_use]] = 0.0
       
        # final state S
        S = np.concatenate((X_r, V, A, H[...,np.newaxis], DH[...,np.newaxis]), axis = -1).astype(np.float32)
        
        Ped_agents = Types == 'PEDESTRIAN'
        
        S_st = S.copy()
        S_st[Ped_agents,:,0:2]  /= self.std_pos_ped
        S_st[~Ped_agents,:,0:2] /= self.std_pos_veh
        S_st[Ped_agents,:,2:4]  /= self.std_vel_ped
        S_st[~Ped_agents,:,2:4] /= self.std_vel_veh
        S_st[Ped_agents,:,4:6]  /= self.std_acc_ped
        S_st[~Ped_agents,:,4:6] /= self.std_acc_veh
        S_st[~Ped_agents,:,6] /= self.std_hea_veh
        S_st[~Ped_agents,:,7] /= self.std_d_h_veh
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            D = np.nanmin(np.sqrt(np.sum((X[:,[0]] - X) ** 2, axis = -1)), axis = - 1)

        D_max = np.zeros_like(D)
        for i_sample in range(len(D)):
            for i_v in range(X.shape[1]):
                if not Types[i_sample, i_v] == 'None':
                    D_max[i_sample, i_v] = attention_radius[(Types[i_sample, 0], Types[i_sample, i_v])]
        
        # Oneself cannot be own neighbor
        D_max[:,0] = -10
        
        Neighbor_bool = D < D_max
        
        # Get Neighbor for each pred value
        Neighbor = {}
        Neighbor_edge = {}
        
        node_type = str(Types[0, 0])
        for node_goal in DIM.keys():
            Dim = DIM[node_goal]
            
            key = (node_type, str(node_goal))
            Neighbor[key] = []
            Neighbor_edge[key] = []
            
            for i_sample in range(S.shape[0]):
                I_agent_goal = np.where(Neighbor_bool[i_sample] & 
                                        (Types[i_sample] == node_goal))[0]
                
                Neighbor[key].append([])
                Neighbor_edge[key].append(torch.from_numpy(np.ones(len(I_agent_goal), np.float32))) 
                for i_agent_goal in I_agent_goal:
                    Neighbor[key][i_sample].append(torch.from_numpy(S[i_sample, i_agent_goal, :, :Dim]))
        

        if img is not None:
            img_batch = img[:,0,:,75:].astype(np.float32) / 255 # Cut of image behind VEHICLE'
            img_batch = img_batch.transpose(0,3,1,2) # put channels first
            img_batch = torch.from_numpy(img_batch).to(dtype = torch.float32)
        else:
            img_batch = None
        
        # Get the first non nan_indices
        exists = np.isfinite(X[:,0]).all(-1)
        first_h = np.argmax(exists, axis = -1)
        first_h = torch.from_numpy(first_h.astype(np.int32))
        
        dim = DIM[node_type]
        S = torch.from_numpy(S[...,:dim]).to(dtype = torch.float32)
        S_st = torch.from_numpy(S_st[...,:dim]).to(dtype = torch.float32)
        
        if Y is None:
            return S, S_st, first_h, Neighbor, Neighbor_edge, img_batch, node_type, center_pos, rot_angle
        else:
            Y = self.rotate_pos_matrix(Y - center_pos, rot_angle).copy()
            
            Y_st = Y.copy()
            Y_st[Ped_agents]  /= self.std_pos_ped
            Y_st[~Ped_agents] /= self.std_pos_veh
        
            Y = torch.from_numpy(Y).to(dtype = torch.float32)
            Y_st = torch.from_numpy(Y_st).to(dtype = torch.float32)
            return S, S_st, first_h, Y, Y_st, Neighbor, Neighbor_edge, img_batch, node_type
    

    # Data extraction in torch
    def rotate_pos_matrix_tensor(self, M, rot_angle):
        assert M.shape[-1] == 2
        assert M.shape[0] == len(rot_angle)


        rot_angle_tensor = rot_angle.to(dtype=torch.float32)
        # rot_angle_tensor = torch.tensor(rot_angle, dtype=torch.float32)
        cos_rot = torch.cos(rot_angle_tensor)
        sin_rot = torch.sin(rot_angle_tensor)
        
        R = torch.stack([torch.stack([cos_rot, -sin_rot], dim=-1),
                        torch.stack([sin_rot, cos_rot], dim=-1)], dim=-2)
        
        R = R.transpose(1, 2)  
        R = R.unsqueeze(1)     
        
        M_r = torch.matmul(M, R)
        return M_r

    def unwrap_phase(self,input_tensor, dim=-1):
        unwrapped = torch.atan2(torch.sin(input_tensor), torch.cos(input_tensor))
        cumulative_diff = torch.cumsum((input_tensor - unwrapped + torch.pi) % (2 * torch.pi) - torch.pi, dim=dim)
        return unwrapped + cumulative_diff

    def extract_data_batch_tensor(self, X, T, Y = None, img = None, num_steps = 10):
        attention_radius = {}
        DIM = {'VEHICLE': 8, 'PEDESTRIAN': 6}
        
        if (self.provide_all_included_agent_types() == 'P').all():
            attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 3.0
        else:
            attention_radius[('PEDESTRIAN', 'PEDESTRIAN')] = 10.0
            attention_radius[('PEDESTRIAN', 'VEHICLE')]    = 50.0
            attention_radius[('VEHICLE',    'PEDESTRIAN')] = 25.0
            attention_radius[('VEHICLE',    'VEHICLE')]    = 150.0

        Types = np.empty(T.shape, dtype = object)

        Types[T == 'P'] = 'PEDESTRIAN'
        Types[T == 'V'] = 'VEHICLE'
        Types[T == 'B'] = 'VEHICLE'
        Types[T == 'M'] = 'VEHICLE'
        Types = Types.astype(str)

        # X = torch.from_numpy(X).to(dtype = torch.float32) #uncomment
        
        center_pos = X[:,0,-1]
        delta_x = center_pos - X[:,0,-2]
        rot_angle = torch.angle(delta_x[:,0] + 1j * delta_x[:,1])

        center_pos = center_pos[:,None,None]        
        X_r = self.rotate_pos_matrix_tensor(X - center_pos, rot_angle) 

        V = (X_r[...,1:,:] - X_r[...,:-1,:]) / self.dt
        zero_mask = (V[...,0] == 0)

        V[zero_mask] = 1e-10

        V = torch.cat((V[...,[0],:], V), dim = -2)
       
        # get accelaration
        A = (V[...,1:,:] - V[...,:-1,:]) / self.dt
        A = torch.cat((A[...,[0],:], A), dim = -2)

        H = torch.atan2(V[:,:,:,1], V[:,:,:,0])
        
        DH = self.unwrap_phase(H, dim=-1) 
        DH = (DH[:,:,1:] - DH[:,:,:-1]) / self.dt
        DH = torch.cat((DH[...,[0]], DH), dim = -1)
       
        # final state S
        S = torch.cat((X_r, V, A, H[...,None], DH[...,None]), dim = -1).to(dtype=torch.float32)
 
        Ped_agents = Types == 'PEDESTRIAN'

        S_st = S.clone()

        S_st[Ped_agents,:,:,0:2]  /= self.std_pos_ped
        S_st[~Ped_agents,:,:,0:2] /= self.std_pos_veh
        S_st[Ped_agents,:,:,2:4]  /= self.std_vel_ped
        S_st[~Ped_agents,:,:,2:4] /= self.std_vel_veh
        S_st[Ped_agents,:,:,4:6]  /= self.std_acc_ped
        S_st[~Ped_agents,:,:,4:6] /= self.std_acc_veh
        S_st[~Ped_agents,:,:,6] /= self.std_hea_veh
        S_st[~Ped_agents,:,:,7] /= self.std_d_h_veh
        
        D ,_ = torch.min(torch.sqrt(torch.sum((X[:,[0]] - X) ** 2, dim = -1)), dim = - 1)
        D_max = torch.zeros_like(D)
        for i_sample in range(len(D)):
            for i_v in range(X.shape[1]):
                if not Types[i_sample, i_v] == 'None':
                    D_max[i_sample, i_v] = attention_radius[(Types[i_sample, 0], Types[i_sample, i_v])]
        
        # Oneself cannot be own neighbor
        D_max[:,0] = -10

        if D.is_cuda:
            D = D.cpu()
        if D_max.is_cuda:
            D_max = D_max.cpu()
        # if Types.is_cuda:
        #     Types = Types.cpu()
        
        Neighbor_bool = D < D_max
        
        # Get Neighbor for each pred value
        Neighbor = {}
        Neighbor_edge = {}
        
        node_type = str(Types[0, 0])
        for node_goal in DIM.keys():
            Dim = DIM[node_goal]
            
            key = (node_type, str(node_goal))
            Neighbor[key] = []
            Neighbor_edge[key] = []
            
            for i_sample in range(S.shape[0]):
                I_agent_goal = torch.where(Neighbor_bool[i_sample] & 
                                        (Types[i_sample] == node_goal))[0]
                
                Neighbor[key].append([])
                Neighbor_edge[key].append(torch.ones(len(I_agent_goal), dtype=torch.float32))
                for i_agent_goal in I_agent_goal:
                    Neighbor[key][i_sample].append(S[i_sample, i_agent_goal, :, :Dim])
        

        if img is not None:
            img_batch = img[:,0,:,75:].astype(np.float32) / 255 # Cut of image behind VEHICLE'
            img_batch = img_batch.transpose(0,3,1,2) # put channels first
            img_batch = torch.from_numpy(img_batch).to(dtype = torch.float32)
        else:
            img_batch = None
            
        first_h = torch.zeros(len(X), dtype=torch.int32)
        
        dim = DIM[node_type]
        S = S[...,:dim].to(dtype = torch.float32)
        S_st = S_st[...,:dim].to(dtype = torch.float32)
        
        if Y is None:
            return S, S_st, first_h, Neighbor, Neighbor_edge, img_batch, node_type, center_pos, rot_angle
        else:
            Y = torch.from_numpy(Y).to(dtype = torch.float32)
            Y = self.rotate_pos_matrix_tensor(Y - center_pos, rot_angle).clone()
            
            Y_st = Y.clone()
            Y_st[Ped_agents]  /= self.std_pos_ped
            Y_st[~Ped_agents] /= self.std_pos_veh
        
            Y_st = Y_st.to(dtype = torch.float32)
            return S, S_st, first_h, Y, Y_st, Neighbor, Neighbor_edge, img_batch, node_type











    def prepare_model_training(self, Pred_types):
        optimizer = dict()
        lr_scheduler = dict()
        for node_type in Pred_types:
            if node_type in self.trajectron.hyperparams['pred_state']:
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
        # setup train_loss
        self.train_loss = np.zeros((1, epochs))
        
        
        T_all = self.provide_all_included_agent_types()
        Pred_types = np.empty(T_all.shape, dtype = object)
        Pred_types[T_all == 'P'] = 'PEDESTRIAN'
        Pred_types[T_all == 'V'] = 'VEHICLE'
        Pred_types[T_all == 'B'] = 'VEHICLE'
        Pred_types[T_all == 'M'] = 'VEHICLE'
        Pred_types = np.unique(Pred_types.astype(str))
        
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
                X, Y, T, img, img_m_per_px, _, _, num_steps, _, _, epoch_done = self.provide_batch_data('train', batch_size)
                
                S, S_St, first_h, Y, Y_st, Neighbor, Neighbor_edge, img, node_type = self.extract_data_batch(X, T, Y, img, num_steps)
                
                # Move img to device
                if img is not None:
                    img = img.to(self.trajectron.device)
                
                self.trajectron.set_curr_iter(curr_iter)
                self.trajectron.step_annealers()
                
                optimizer[node_type].zero_grad()
                
                # Run forward pass
                model = self.trajectron.node_models_dict[node_type]
                train_loss = model.train_loss(inputs                = S[:,0].to(self.trajectron.device),
                                              inputs_st             = S_St[:,0].to(self.trajectron.device),
                                              first_history_indices = first_h.to(self.trajectron.device),
                                              labels                = Y[:,0].to(self.trajectron.device),
                                              labels_st             = Y_st[:,0].to(self.trajectron.device),
                                              neighbors             = Neighbor,
                                              neighbors_edge_value  = Neighbor_edge,       
                                              robot                 = None,
                                              map                   = img,
                                              prediction_horizon    = num_steps)
                
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
            ii = 0
            for i, weights in enumerate(self.weights_saved):
                weights_torch = torch.from_numpy(weights)
                if Weights[ii].shape == weights_torch.shape:
                    Weights[ii][:] = weights_torch[:]
                    ii += 1
        
    def predict_method(self):
        batch_size = max(1, int(self.trajectron.hyperparams['batch_size'] / 10))
        
        prediction_done = False
        
        batch_number = 0
        while not prediction_done:
            batch_number += 1
            print('Predict trajectron: Batch {}'.format(batch_number))
            X, T, img, img_m_per_px, _, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', batch_size)
            S, S_St, first_h, Neighbor, Neighbor_edge, img, node_type, center_pos, rot_angle = self.extract_data_batch(X, T, None, img, num_steps)
            
            # Move img to device
            if img is not None:
                img = img.to(self.trajectron.device)
                
            torch.cuda.empty_cache()
            # Run prediction pass
            model = self.trajectron.node_models_dict[node_type]
            self.trajectron.model_registrar.to(self.trajectron.device)
            
            with torch.no_grad():
                predictions = model.predict(inputs                = S[:,0].to(self.trajectron.device),
                                            inputs_st             = S_St[:,0].to(self.trajectron.device),
                                            first_history_indices = first_h.to(self.trajectron.device),
                                            neighbors             = Neighbor,
                                            neighbors_edge_value  = Neighbor_edge,
                                            robot                 = None,
                                            map                   = img,
                                            prediction_horizon    = num_steps,
                                            num_samples           = self.num_samples_path_pred)
            
            Pred = predictions.detach().cpu().numpy()
                
            # set batchsize first
            Pred = Pred.transpose(1,0,2,3)
            
            # reverse rotation
            Pred_r = self.rotate_pos_matrix(Pred, -rot_angle)
            
            # reverse translation
            Pred_t = Pred_r + center_pos
            
            self.save_predicted_batch_data(Pred_t, Sample_id, Agent_id)
    

    def predict_batch_tensor(self,X,T,Domain,img, img_m_per_px,num_steps,num_samples = 20):

        X = X.to(self.trajectron.device)
        self.trajectron.model_registrar.to(self.trajectron.device)
    
        S, S_St, first_h, Neighbor, Neighbor_edge, img, node_type, center_pos, rot_angle = self.extract_data_batch_tensor(X, T, None, img, num_steps)
        
        # Move img to device
        if img is not None:
            img = img.to(self.trajectron.device)
            
        torch.cuda.empty_cache()
        # Run prediction pass
        model = self.trajectron.node_models_dict[node_type]
        self.trajectron.model_registrar.to(self.trajectron.device)
        
        
        predictions = model.predict(inputs                = S[:,0].to(self.trajectron.device),
                                    inputs_st             = S_St[:,0].to(self.trajectron.device),
                                    first_history_indices = first_h.to(self.trajectron.device),
                                    neighbors             = Neighbor,
                                    neighbors_edge_value  = Neighbor_edge,
                                    robot                 = None,
                                    map                   = img,
                                    prediction_horizon    = num_steps,
                                    num_samples           = num_samples)
        

        Pred = predictions.permute(1,0,2,3)
        # reverse rotation
        Pred_r = self.rotate_pos_matrix_tensor(Pred, -rot_angle)
        # reverse translation
        Pred_t = Pred_r + center_pos

        return Pred_t

    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        # get default kwargs
        if hasattr(self, 'model_kwargs'):
            self.define_default_kwargs()
            if self.model_kwargs['predict_ego']:
                return 'path_all_wi_pov'
            else:
                return 'path_all_wo_pov'
        else:
            return 'path_all_wi_pov'
    
    def get_name(self = None):
        self.define_default_kwargs()

        names = {'print': 'Trajectron ++ (Old_version)',
                 'file': 't_pp_old_' + str(self.model_kwargs['seed']) + '_' + str(int(self.model_kwargs['predict_ego'])),
                 'latex': r'\emph{T++}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True 
        
    def provides_epoch_loss(self = None):
        return True