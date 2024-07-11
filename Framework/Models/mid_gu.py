import numpy as np
import os
import pickle
import random
import torch

from model_template import model_template
from MID.evaluation import *
from MID import mid_model
from MID.environment import *
from easydict import EasyDict

from MID.environment.scene import Scene
from MID.environment.node import Node
from MID.environment.node_type import NodeType
from MID.environment import DoubleHeaderNumpyArray



class mid_gu(model_template):
    '''
    The Motion Indeterminacy Diffusion (MID) prediction model originially designed
    for pedestrian predictionsm, in which indeterminacy from all the walkable areas is 
    progressively discarded until reaching the desired trajectory.

    The code it addapted from https://github.com/Gutianpei/MID

    Gu, T., Chen, G., Li, J., Lin, C., Rao, Y., Zhou, J., & Lu, J. (2022). Stochastic trajectory 
    prediction via motion indeterminacy diffusion. In Proceedings of the IEEE/CVF Conference on 
    Computer Vision and Pattern Recognition (pp. 17113-17122).
    '''

    def define_default_kwargs(self):
        if not('seed' in self.model_kwargs.keys()):
            self.model_kwargs['seed'] = 0

    def get_name(self = None):
        self.define_default_kwargs()
        names = {'print': 'MDI',
                    'file': 'MDI_' + str(self.model_kwargs['seed']),
                    'latex': r'\emph{MDI}'}

        return names

    def requires_torch_gpu(self = None):
        return True

    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def check_trainability_method(self):
        return None
    
    def get_config_dict(self, use_map = False):

        self.config_dict = {}
        self.config_dict['lr'] = 0.001
        self.config_dict['data_dir'] = 'processed_data'
        self.config_dict['diffnet'] = 'TransformerConcatLinear'
        if use_map:
            self.config_dict['map_encoding'] = True
            self.config_dict['encoder_dim'] = 288
        else:
            self.config_dict['map_encoding'] = False
            self.config_dict['encoder_dim'] = 256
        self.config_dict['tf_layer'] = 3
        self.config_dict['epochs'] = 2000
        self.config_dict['batch_size'] = 256
        self.config_dict['eval_batch_size'] = 256
        self.config_dict['k_eval'] = 25
        self.config_dict['seed'] = 123
        self.config_dict['eval_every'] = 10
        self.config_dict['eval_at'] = self.config_dict['epochs']
        self.config_dict['eval_mode'] = False
        self.config_dict['sampling'] = 'ddpm'
        self.config_dict['exp_name'] = 'MID_testing'
        self.config_dict['dataset'] = self.data_set.get_name()['file']
        
        self.config_dict['conf'] = None
        self.config_dict['debug'] = False
        self.config_dict['preprocess_workers'] = 0
        self.config_dict['offline_scene_graph'] = 'yes'
        self.config_dict['dynamic_edges'] = 'yes'
        self.config_dict['edge_state_combine_method'] = 'sum'
        self.config_dict['edge_influence_combine_method'] = 'attention'
        self.config_dict['edge_addition_filter'] = [0.25, 0.5, 0.75, 1.0]
        self.config_dict['edge_removal_filter'] = [1.0, 0.0]
        self.config_dict['override_attention_radius'] = []
        self.config_dict['incl_robot_node'] = False
        self.config_dict['augment'] = True
        self.config_dict['node_freq_mult_train'] = False
        self.config_dict['node_freq_mult_eval'] = False
        self.config_dict['scene_freq_mult_train'] = False
        self.config_dict['scene_freq_mult_eval'] = False
        self.config_dict['scene_freq_mult_viz'] = False
        self.config_dict['no_edge_encoding'] = False
        # Data Parameters:
        self.config_dict['device'] = 'cuda'
        self.config_dict['eval_device'] = None

        self.config_dict = EasyDict(self.config_dict)
    
    def setup_method(self):    
        self.define_default_kwargs()
        # set random seeds
        seed = self.model_kwargs['seed']    
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = True        
        self.can_use_map = True
        self.can_use_graph = False
        # If self.can_use_map = True, the following is also required
        self.target_width = 180
        self.target_height = 100
        self.grayscale = False
        
        self.use_map = self.can_use_map and self.has_map        


        batch_size = 256

        if (self.provide_all_included_agent_types() == 'P').all():
            self.hyperparams = {'batch_size': batch_size,
                           'grad_clip': 1.0,
                           'learning_rate_style': 'exp',
                           'learning_rate': 0.01, 
                           'min_learning_rate': 1e-05,
                           'learning_decay_rate': 0.9999,
                           'prediction_horizon': self.num_timesteps_out,
                           'minimum_history_length': 1,
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
                           'enc_rnn_dim_edge': 128,
                           'enc_rnn_dim_edge_influence': 128,
                           'enc_rnn_dim_history': 128,
                           'enc_rnn_dim_future': 128,
                           'dec_rnn_dim': 128,
                           'q_z_xy_MLP_dims': None,
                           'p_z_x_MLP_dims': 32,
                           'GMM_components': 1,
                           'log_p_yt_xz_max': 6,
                           'N': 1, # numbver of states per dimension of conditional distribution
                           'K': 80, # number of dimension of conditional distribution
                           'tau_init': 2.0,
                           'tau_final': 0.05,
                           'tau_decay_rate': 0.997,
                           'use_z_logit_clipping': True,
                           'z_logit_clip_start': 0.05,
                           'z_logit_clip_final': 5.0,
                           'z_logit_clip_crossover': 300,
                           'z_logit_clip_divisor': 5,
                           "dynamic": {"PEDESTRIAN": {"name": "SingleIntegrator",
                                                      "distribution": False,
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
                           'override_attention_radius': [],
                           'npl_rate': 0.8,
                           'tao': 0.4
                           }
        else:
            self.hyperparams = {'batch_size': batch_size,
                           'grad_clip': 1.0,
                           'learning_rate_style': 'exp',
                           'learning_rate': 0.01,
                           'min_learning_rate': 1e-05,
                           'learning_decay_rate': 0.9999,
                           'prediction_horizon': self.num_timesteps_out,
                           'minimum_history_length': 1,
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
                           'enc_rnn_dim_edge': 128,
                           'enc_rnn_dim_edge_influence': 128,
                           'enc_rnn_dim_history': 128,
                           'enc_rnn_dim_future': 128,
                           'dec_rnn_dim': 128,
                           'q_z_xy_MLP_dims': None,
                           'p_z_x_MLP_dims': 32,
                           'GMM_components': 1,
                           'log_p_yt_xz_max': 6,
                           'N': 1, # numbver of states per dimension of conditional distribution
                           'K': 80, # number of dimension of conditional distribution
                           'tau_init': 2.0,
                           'tau_final': 0.05,
                           'tau_decay_rate': 0.997,
                           'use_z_logit_clipping': True,
                           'z_logit_clip_start': 0.05,
                           'z_logit_clip_final': 5.0,
                           'z_logit_clip_crossover': 300,
                           'z_logit_clip_divisor': 5,
                           'dynamic': {'PEDESTRIAN': {'name': 'SingleIntegrator',
                                                      'distribution': False,
                                                      'limits': {}},
                                       'VEHICLE': {'name': 'Unicycle',
                                                   'distribution': False,
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
                           'override_attention_radius': [],
                           'npl_rate': 0.8,
                           'tao': 0.4
                           }
        
        self.std_pos_ped = 1
        self.std_vel_ped = 2
        self.std_acc_ped = 1
        self.std_pos_veh = 80
        self.std_vel_veh = 15
        self.std_acc_veh = 4
        self.std_hea_veh = np.pi
        self.std_d_h_veh = 1

        if (self.provide_all_included_agent_types() == 'P').all():
            node_list = [Node(node_type = 'PEDESTRIAN', node_id = 'EMPTY', data = None)]
        else:
            node_list = [Node(node_type = 'PEDESTRIAN', node_id = 'EMPTY', data = None), 
                              Node(node_type = 'VEHICLE', node_id = 'EMPTY', data = None)
                              ]
            
        scene = Scene(dt = self.dt, timesteps = self.num_timesteps_in + self.num_timesteps_out)
        scene.nodes = node_list
        scenes = [scene]
        
        
            
        
        if (self.provide_all_included_agent_types() == 'P').all():
            node_type_list = ['PEDESTRIAN']
        else:
            node_type_list = ['PEDESTRIAN', 'VEHICLE']
        
        train_env = Environment(node_type_list = node_type_list,
                                standardization = None,
                                scenes = scenes,
                                attention_radius = None, 
                                robot_type = None)
        

        self.get_config_dict(use_map = self.use_map)
        self.MID = mid_model.MID(config=self.config_dict, train_env=train_env, hyperparams=self.hyperparams)
        self.MID._build()


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

        # set rot angle
        if Y is None:
            rot_angle = np.zeros_like(delta_x[:,0])
        else:
            rot_angle = np.random.rand(*delta_x[:,0].shape) * 2 * np.pi

        center_pos = center_pos[:,np.newaxis,np.newaxis]        
        X_r = self.rotate_pos_matrix(X - center_pos, rot_angle)
        
        V = (X_r[...,1:,:] - X_r[...,:-1,:]) / self.dt
        V = np.concatenate((V[...,[0],:], V), axis = -2)
       
        # get accelaration
        A = (V[...,1:,:] - V[...,:-1,:]) / self.dt
        A = np.concatenate((A[...,[0],:], A), axis = -2)
       
        H = np.arctan2(V[:,:,:,1], V[:,:,:,0])
        
        DH = np.unwrap(H, axis = -1) 
        DH = (DH[:,:,1:] - DH[:,:,:-1]) / self.dt
        DH = np.concatenate((DH[...,[0]], DH), axis = -1)
       
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
        
        D = np.min(np.sqrt(np.sum((X[:,[0]] - X) ** 2, axis = -1)), axis = - 1)
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
            img_batch = img[:,0,:,80:].astype(np.float32) / 255 # Cut of image behind VEHICLE'
            img_batch = img_batch.transpose(0,3,1,2) # put channels first
            img_batch = torch.from_numpy(img_batch).to(dtype = torch.float32)
        else:
            img_batch = None
            
        first_h = torch.from_numpy(np.zeros(len(X), np.int32)).to(dtype = torch.int64)
        
        dim = DIM[node_type]
        S = torch.from_numpy(S[...,:dim]).to(dtype = torch.float32)
        S_st = torch.from_numpy(S_st[...,:dim]).to(dtype = torch.float32)
        
        if Y is None:
            return S, S_st, first_h, Neighbor, Neighbor_edge, center_pos, img_batch, node_type
        else:
            Y = self.rotate_pos_matrix(Y - center_pos, rot_angle).copy()

            Y_st = Y.copy()
            Y_st[Ped_agents]  /= self.std_pos_ped
            Y_st[~Ped_agents] /= self.std_pos_veh
        
            Y = torch.from_numpy(Y).to(dtype = torch.float32)
            Y_st = torch.from_numpy(Y_st).to(dtype = torch.float32)
            # return first_h, S, Y, S_st, Y_st, Neighbor, Neighbor_edge, img_batch, node_type
            return S, S_st, first_h, Y, Y_st, Neighbor, Neighbor_edge, img_batch, node_type
        

    def train_method(self):

        T_all = self.provide_all_included_agent_types()
        Pred_types = np.empty(T_all.shape, dtype = object)
        Pred_types[T_all == 'P'] = 'PEDESTRIAN'
        Pred_types[T_all == 'V'] = 'VEHICLE'
        Pred_types[T_all == 'B'] = 'VEHICLE'
        Pred_types[T_all == 'M'] = 'VEHICLE'
        Pred_types = np.unique(Pred_types.astype(str))

        # prepare training
        for epoch in range(1, self.config_dict['epochs'] + 1):
            self.train_dataset_augment = self.config_dict['augment']
            # print current epoch
            print('')

            epoch_done = False
            
            batch_number = 0
            loss_sum = []
            num_samples = 0
            while not epoch_done:
                batch_number += 1

                print(f"Epoch {epoch} - Batch {batch_number}", flush = True)
                X, Y, T, img, _, _, num_steps, epoch_done = self.provide_batch_data('train', self.hyperparams['batch_size'], 
                                                                                               val_split_size = 0.1)
                
                S, S_St, first_h, Y, Y_st, Neighbor, Neighbor_edge, img, node_type = self.extract_data_batch(X, T, Y, img, num_steps)

                if node_type == 'PEDESTRIAN': 
                    value = 1
                elif node_type == 'VEHICLE': 
                    value = 2
                else: 
                    value = 3

                node_type = NodeType(name=node_type, value=value)
                
                # Move img to device
                if img is not None:
                    img = img.to(self.device)
                
                self.MID.optimizer.zero_grad()
                
                # Run forward pass
                batch = (first_h, S[:,0], Y[:,0], S_St[:,0], Y_st[:,0], Neighbor, Neighbor_edge, None, img)
                train_loss = self.MID.model.get_loss(batch, node_type)
                
                loss_sum.append(train_loss.item() * len(X))
                num_samples += len(X)
                train_loss.backward()
                self.MID.optimizer.step()
            
            
            epoch_loss = np.sum(loss_sum) / num_samples
            print(f"Epoch {epoch} MSE: {epoch_loss:.2f}")
            
            self.train_dataset_augment = False
            if epoch % self.config_dict['eval_every'] == 0:
                self.MID.model.eval()

                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.hyperparams['prediction_horizon']
                max_hl = self.hyperparams['maximum_history_length']

                eval_done = False
                i = 0
                while not eval_done:

                    X, Y, T, img, _, _, num_steps, eval_done = self.provide_batch_data('val', self.hyperparams['batch_size'], 
                                                                                        val_split_size = 0.1)
            
                    S, S_St, first_h, Y, Y_st, Neighbor, Neighbor_edge, img, node_type = self.extract_data_batch(X, T, Y, img, num_steps)

                    test_batch = (first_h, S[:,0], Y[:,0], S_St[:,0], Y_st[:,0], Neighbor, Neighbor_edge, None, img)#, num_steps)

                    if test_batch is None:
                        continue

                    if node_type == 'PEDESTRIAN': 
                        value = 1
                    elif node_type == 'VEHICLE': 
                        value = 2
                    else: 
                        value = 3

                    node_type = NodeType(name=node_type, value=value)
                    
                    traj_pred = self.MID.model.generate(test_batch, node_type, num_points=self.num_timesteps_out, sample=20,
                                                        bestof=True) # B * 20 * self.num_timesteps_out * 2
                    timesteps_o = traj_pred.shape[1]

                    predictions = traj_pred
                    predictions_dict = {}
                    for i in range(timesteps_o):#, ts in enumerate(timesteps_o):
                        node_data = torch.concat((S[i,0,:,:2], Y[i,0]))
                        node_data = DoubleHeaderNumpyArray(node_data.detach().cpu().numpy(),
                                                           [tuple(states) for states in list(self.hyperparams['pred_state'][node_type.name].values())])
                        node = Node(node_type = node_type, node_id = 'EMPTY', data = node_data)
                        if i not in predictions_dict.keys():
                            predictions_dict[i] = dict()
                        predictions_dict[i][node] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))

                    batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                           self.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=[node_type.name],
                                                                           kde=False,
                                                                           map=None,
                                                                           best_of=True,
                                                                           prune_ph_to_future=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type.name]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type.name]['fde']))

                    i += 1



                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)


                print(f"Epoch evaluation {epoch} Best Of 20: ADE: {ade} FDE: {fde}")

                # Saving model
                # checkpoint = {
                #     'encoder': self.MID.registrar.model_dict,
                #     'ddpm': self.MID.model.state_dict()
                # }
                # torch.save(checkpoint, osp.join(self.model_dir, f"{self.dataset}_epoch{epoch}.pt")) # TODO check how to do this nicely

                self.MID.model.train()

            
        # save weigths 
        Weights = list(self.MID.registrar.parameters())
        self.weights_saved = []
        for weigths in Weights:
            self.weights_saved.append(weigths.detach().cpu().numpy())


    def load_method(self):
        Weights = list(self.MID.registrar.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_saved):
                Weights[i][:] = torch.from_numpy(weights)[:]


    def predict_method(self):
        batch_size = max(1, int(self.hyperparams['batch_size'] / 10))
        self.MID.model.eval()
        
        prediction_done = False
        
        batch_number = 0
        while not prediction_done:
            batch_number += 1
            print('Predict MID: Batch {}'.format(batch_number))
            X, T, img, _, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', batch_size)
            S, S_St, first_h, Neighbor, Neighbor_edge, center_pos, img, node_type = self.extract_data_batch(X, T, None, img, num_steps)
                
            # Move img to device
            if img is not None:
                img = img.to(self.device)
                
            torch.cuda.empty_cache()
            # Run prediction pass
            self.MID.registrar.to(self.device)
            test_batch = (first_h, S[:,0], None, S_St[:,0], None, Neighbor, Neighbor_edge, None, img)

            if node_type == 'PEDESTRIAN': 
                value = 1
            elif node_type == 'VEHICLE': 
                value = 2
            else: 
                value = 3

            node_type = NodeType(name=node_type, value=value)
            
            num_steps_pred = min(num_steps, 100)

            traj_pred = self.MID.model.generate(test_batch, node_type, num_points=num_steps_pred, sample=self.num_samples_path_pred,
                                                bestof=True, sampling='ddim', step=100//5) # B * 20 * 12 * 2
                
            # set batchsize first
            Pred_r = traj_pred.transpose(1,0,2,3)

            # reverse translation
            Pred_t = Pred_r + center_pos

            # Extrapolate if necessary
            if num_steps_pred < num_steps:
                Pred_last_vel = Pred_t[...,-1:,:] - Pred_t[...,-2:-1,:]
                t_extra = np.arange(1, num_steps - num_steps_pred + 1)[np.newaxis,np.newaxis]
                Pred_extrap = Pred_t[...,-1:,:] + Pred_last_vel * t_extra
                Pred_t = np.concatenate((Pred_t, Pred_extrap), axis = -2)
            
            self.save_predicted_batch_data(Pred_t, Sample_id, Agent_id)
            
    def save_params_in_csv(self = None):
        return False
    

    def provides_epoch_loss(self = None):
        return False
