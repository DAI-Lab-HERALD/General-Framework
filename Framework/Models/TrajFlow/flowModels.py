import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from TrajFlow.spline_flow import NeuralSplineFlow
from TrajFlow.SocialEncodingModule import SocialInterGNN, TrajRNN
from torch_geometric.data import Data

import numpy as np

class RNN(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=16, hs=16, nl=3, device=0):
        
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden
    

class Scene_Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=4, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5, stride=4, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 32, 5, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True)#,
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 * 2 * 3, 128), 
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class FloMo(nn.Module):

    def __init__(self, pred_steps, alpha, beta, gamma, scene_encoder=None, B=15., 
                use_map=False, interactions=False, rel_coords=True, norm_rotation=False, device=0,
                obs_encoding_size=16, scene_encoding_size=8, n_layers_rnn=3, es_rnn=16, hs_rnn=16):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_size = pred_steps * 2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation

        # core modules
        self.obs_encoding_size = obs_encoding_size
        self.scene_encoding_size = scene_encoding_size
        self.n_layers_rnn = n_layers_rnn
        self.es_rnn = es_rnn
        self.hs_rnn = hs_rnn

        self.obs_encoder = RNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)

        if use_map:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size+self.scene_encoding_size, n_layers=10, K=8,
                                                B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)            
            if scene_encoder is None:
                self.scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
            else:
                self.scene_encoder = scene_encoder
        else:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size, n_layers=10, K=8,
                                    B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)    
            
            self.scene_encoder = scene_encoder
        
        

        # move model to specified device
        self.device = device
        self.to(device)

    def _encode_conditionals(self, x, scene=None):
        # encode observed trajectory
        x_in = x
        if self.rel_coords:
            x_in = x[:,1:] - x[:,:-1] # convert to relative coords
        x_enc, hidden = self.obs_encoder(x_in) # encode relative histories
        x_enc = x_enc[:,-1]
        if scene is not None:
            scene_enc = self.scene_encoder(scene)
            x_enc = torch.concat((x_enc,scene_enc), dim=1)

        x_enc_context = x_enc
        return x_enc_context

    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[...,1:,:] = (y_rel[...,1:,:] - y_rel[...,:-1,:]) # steps relative to each other
        y_rel = y_rel * self.alpha # scale up for numeric reasons
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = y_rel / self.alpha
        return torch.cumsum(y_abs, dim=-2) + x_t 
    

    def _rotate(self, x, x_t, angles_rad):
        # Fix alingment issues
        len_x = len(x.shape)
        len_r = len(angles_rad.shape)
        for i in range(len_x - len_r - 1):
            angles_rad = angles_rad.unsqueeze(-1)
            
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        x_centered = x - x_t # translate
        x_vals, y_vals = x_centered[...,0], x_centered[...,1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        x_centered[...,0] = new_x_vals
        x_centered[...,1] = new_y_vals
        return x_centered + x_t # translate back

    def _normalize_rotation(self, x, y_true=None):
        x_t = x[...,-1:,:]
        if len(x_t.shape) > 3:
            x_t = x_t[:,[0]]
        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[...,[-1],:] - x[...,[-2],:]
        if len(x_t_rel.shape) > 3:
            x_t_rel = x_t_rel[:,[0]]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[...,1], x_t_rel[...,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad # inverse
        else:
            return x, rot_angles_rad # forward pass

    def _inverse(self, y_true, x, scene=None):
        if self.norm_rotation:
            x, y_true, angle = self._normalize_rotation(x, y_true)

        x_t = x[...,-1:,:]
        x_enc = self._encode_conditionals(x, scene) # history encoding
        y_rel = self._abs_to_rel(y_true, x_t)
        y_rel_flat = torch.flatten(y_rel, start_dim=1)

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)
            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        
        z, jacobian_det = self.flow.inverse(torch.flatten(y_rel_flat, start_dim=1), x_enc)
        return z, jacobian_det

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError

    def sample(self, n, x, scene=None):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x, scene) # history encoding
            if scene is not None:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size+self.scene_encoding_size)
            else:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.pred_steps, 2) # predict n trajectories input

            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))

            x_t = x[...,-1:,:].unsqueeze(dim=1).repeat(1, n, 1, 1)
            samples_abs = self._rel_to_abs(samples_rel, x_t)

            # invert rotation normalization
            if self.norm_rotation:
                x_t_all = x[...,-1,:]
                for i in range(len(samples_abs)):
                    pred_trajs = samples_abs[i]
                    samples_abs[i] = self._rotate(pred_trajs, x_t_all[i], -1 * rot_angles_rad[i].repeat(pred_trajs.size(0)))

            return samples_abs, log_probs

    def predict_expectation(self, n, x):
        samples = self.sample(n, x)
        y_pred = samples.mean(dim=1, keepdim=True)
        return y_pred

    def log_prob(self, y_true, x, scene=None):
        z, log_abs_jacobian_det = self._inverse(y_true, x, scene)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det
    
class FloMo_I(FloMo):

    def __init__(self, pred_steps, alpha, beta, gamma, scene_encoder=None, B=15., 
                use_map=False, interactions=False, rel_coords=True, norm_rotation=False, device=0,
                obs_encoding_size=16, scene_encoding_size=8, n_layers_rnn=3, es_rnn=16, hs_rnn=16,
                n_layers_gnn=4, es_gnn=64, T_all = None):
        nn.Module.__init__(self)
        self.pred_steps = pred_steps
        self.output_size = pred_steps * 2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation

        # core modules
        self.obs_encoding_size = obs_encoding_size
        self.scene_encoding_size = scene_encoding_size
        self.n_layers_rnn = n_layers_rnn
        self.es_rnn = es_rnn
        self.hs_rnn = hs_rnn

        self.es_gnn = es_gnn
        self.n_layers_gnn = n_layers_gnn
        
        # observation encoders
        self.obs_encoder = nn.ModuleDict({})
        self.tar_obs_encoder = nn.ModuleDict({})
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        for t in self.t_unique:
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            self.obs_encoder[t_key] = TrajRNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, 
                                              hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)
            self.obs_encoder[t_key].to(device)
            
            self.tar_obs_encoder[t_key] = RNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, 
                                              hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)
            self.tar_obs_encoder[t_key].to(device)
        
        self.edge_dim_gnn = len(self.t_unique) * 2 + 1
        
        self.GNNencoder = SocialInterGNN(num_layers=n_layers_gnn, emb_dim=es_gnn,
                                         in_dim=self.obs_encoding_size, edge_dim=self.edge_dim_gnn,
                                         device = device)
        if use_map:
            if scene_encoder is None:
                self.scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
            else:
                self.scene_encoder = scene_encoder
        else:
                self.scene_encoder = None
                
        if use_map:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size+self.es_gnn+self.scene_encoding_size, 
                                         n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)     
        else:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size+self.es_gnn, 
                                         n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device) 
        
        

        # move model to specified device
        self.device = device
        self.to(device)

    def _encode_conditionals(self, x, T, scene=None):
        x_in = x
        if self.rel_coords:
            x_in = x[...,1:,:] - x[...,:-1,:]
        
        x_enc     = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1, self.obs_encoding_size), device = self.device)
        x_tar_enc = torch.zeros((x.shape[0], x.shape[2] - 1, self.obs_encoding_size), device = self.device)
        for t in self.t_unique:
            assert t in T
            t_in = T == t
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            
            x_enc[t_in], _ = self.obs_encoder[t_key](x_in[t_in])
            # target agent is always first agent
            x_tar_enc[t_in[:,0]], _ = self.tar_obs_encoder[t_key](x_in[[t_in[:,0],0]])
            
            
        
        # TODO: Maybe put all the outputs here, and try to punish changes between timesteps
        # To prevent sudden fluctuation
        x_enc     = x_enc[...,-1,:]
        x_tar_enc = x_tar_enc[...,-1,:]
        
        # Deal with autoencoder here
        num_nodes = x_enc.size(dim=1)
        row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
        col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        graph_edge_node = torch.stack([row, col], dim=0).unsqueeze(0).repeat((len(x_enc),1,1))
        # Combine batch into one matrix
        sample_enc = torch.arange(len(x_enc), device = self.device).unsqueeze(1).repeat((1,num_nodes ** 2))
        graph_edge_index = graph_edge_node + sample_enc.unsqueeze(1) * num_nodes
        # Combine
        graph_edge_index = graph_edge_index.permute(1,0,2).reshape(2,-1)
        
        T_one_hot = (T.unsqueeze(-1) == self.t_unique.unsqueeze(0).unsqueeze(0)).float().reshape(-1, len(self.t_unique))
        
        class_in_out = T_one_hot[graph_edge_index, :].permute(1,0,2).reshape(-1, 2 * len(self.t_unique))
        
        D = torch.sqrt(torch.sum((x[:,None,:,-1] - x[:,:,None,-1]) ** 2, dim = -1))
        dist_in_out = D[sample_enc, graph_edge_node[:,0], graph_edge_node[:,1]]
        dist_in_out = dist_in_out.reshape(-1,1)
         
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        
        graph_batch = torch.arange((len(x_enc)), dtype=torch.long, device=self.device).repeat_interleave(num_nodes)
        
        socialData = Data(x=x_enc.reshape(-1, self.obs_encoding_size), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=graph_batch)
        
        social_enc = self.GNNencoder(socialData)
        
        if scene is not None:
            scene_enc = self.scene_encoder(scene)
            total_enc = torch.concat((x_tar_enc, social_enc, scene_enc), dim=1)
        else:
            total_enc = torch.concat((x_tar_enc, social_enc), dim=1)
        return total_enc
    
    def _inverse(self, y_true, x, T, scene=None):
        if self.norm_rotation:
            x, y_true, angle = self._normalize_rotation(x, y_true)

        x_t = x[:,[0],-1:,:]
        x_enc = self._encode_conditionals(x, T, scene) # history encoding
        y_rel = self._abs_to_rel(y_true, x_t)
        y_rel_flat = torch.flatten(y_rel, start_dim=1)

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)
            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        z, jacobian_det = self.flow.inverse(y_rel_flat, x_enc)
        return z, jacobian_det

    def sample(self, n, x, T, scene=None):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x, T, scene) # history encoding
            if scene is not None:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size+self.es_gnn+self.scene_encoding_size)
            else:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size+self.es_gnn)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.pred_steps, 2) # predict n trajectories input

            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))

            x_t = x[:,[0],-1:,:].repeat(1, n, 1, 1)
            samples_abs = self._rel_to_abs(samples_rel, x_t)

            # invert rotation normalization
            samples_abs = self._rotate(samples_abs, x_t, -1 * rot_angles_rad)

            return samples_abs, log_probs

    def log_prob(self, y_true, x, T, scene=None):
        z, log_abs_jacobian_det = self._inverse(y_true, x, T, scene)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det    

class TrajFlow(nn.Module):

    def __init__(self, pred_steps, alpha, beta, gamma, scene_encoder=None, B=15., 
                use_map=False, interactions=False, rel_coords=True, norm_rotation=False, device=0,
                obs_encoding_size=16, scene_encoding_size=8, n_layers_rnn=3, es_rnn=16, hs_rnn=16):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_size = pred_steps 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation

        # core modules
        self.obs_encoding_size = obs_encoding_size
        self.scene_encoding_size = scene_encoding_size
        self.n_layers_rnn = n_layers_rnn
        self.es_rnn = es_rnn
        self.hs_rnn = hs_rnn
        
        self.obs_encoder = RNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)

        if use_map:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size+self.scene_encoding_size, n_layers=10, K=8,
                                                B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)            
            if scene_encoder is None:
                self.scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
            else:
                self.scene_encoder = scene_encoder
        else:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size, n_layers=10, K=8,
                                    B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)    
        
            self.scene_encoder = scene_encoder

        # move model to specified device
        self.device = device
        self.to(device)

    def _encode_conditionals(self, x, scene=None):
        # encode observed trajectory
        x_in = x
        if self.rel_coords:
            x_in = x[:,1:] - x[:,:-1] # convert to relative coords
        x_enc, hidden = self.obs_encoder(x_in) # encode relative histories
        x_enc = x_enc[:,-1]
        if scene is not None:
            scene_enc = self.scene_encoder(scene)
            x_enc = torch.concat((x_enc,scene_enc), dim=1)

        x_enc_context = x_enc
        return x_enc_context

    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[...,1:,:] = (y_rel[...,1:,:] - y_rel[...,:-1,:]) # steps relative to each other
        y_rel = y_rel * self.alpha # scale up for numeric reasons
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = y_rel / self.alpha
        return torch.cumsum(y_abs, dim=-2) + x_t 

    def _rotate(self, x, x_t, angles_rad):
        # Fix alingment issues
        len_x = len(x.shape)
        len_r = len(angles_rad.shape)
        for i in range(len_x - len_r - 1):
            angles_rad = angles_rad.unsqueeze(-1)
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        x_centered = x - x_t # translate
        x_vals, y_vals = x_centered[...,0], x_centered[...,1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        x_centered[...,0] = new_x_vals
        x_centered[...,1] = new_y_vals
        return x_centered + x_t # translate back

    def _normalize_rotation(self, x, y_true=None):
        x_t = x[...,-1:,:]
        if len(x_t.shape) > 3:
            x_t = x_t[:,[0]]
        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[...,[-1],:] - x[...,[-2],:]
        if len(x_t_rel.shape) > 3:
            x_t_rel = x_t_rel[:,[0]]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[...,1], x_t_rel[...,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad # inverse
        else:
            return x, rot_angles_rad # forward pass

    def _inverse(self, y_true, x, scene=None):
        if self.norm_rotation:
            x, angle = self._normalize_rotation(x)

        x_enc = self._encode_conditionals(x, scene) # history encoding
        y_rel_flat = y_true

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)

            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        
        z, jacobian_det = self.flow.inverse(torch.flatten(y_rel_flat, start_dim=1), x_enc)
        return z, jacobian_det

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError

    def sample(self, n, x, scene=None):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x, scene) # history encoding
            
            if scene is not None:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size+self.scene_encoding_size)
            else:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.pred_steps) # predict n trajectories input
            
            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))
            log_probs = log_probs.view(*(x.size(0), n))

            return samples_rel, log_probs

    def predict_expectation(self, n, x):
        samples = self.sample(n, x)
        y_pred = samples.mean(dim=1, keepdim=True)
        return y_pred

    def log_prob(self, y_true, x, scene=None):
        z, log_abs_jacobian_det = self._inverse(y_true, x, scene)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det


class TrajFlow_I(TrajFlow):

    def __init__(self, pred_steps, alpha, beta, gamma, scene_encoder=None, B=15., 
                use_map=False, interactions=False, rel_coords=True, norm_rotation=False, device=0,
                obs_encoding_size=16, scene_encoding_size=8, 
                n_layers_rnn=3, es_rnn=16, hs_rnn=16,
                n_layers_gnn=4, es_gnn=64, T_all = None):
        nn.Module.__init__(self)
        self.pred_steps = pred_steps
        self.output_size = pred_steps 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation
        

        # core modules
        self.obs_encoding_size = obs_encoding_size
        self.scene_encoding_size = scene_encoding_size
        self.n_layers_rnn = n_layers_rnn
        self.es_rnn = es_rnn
        self.hs_rnn = hs_rnn
        
        self.es_gnn = es_gnn
        self.n_layers_gnn = n_layers_gnn
        
        # observation encoders
        self.obs_encoder = nn.ModuleDict({})
        self.tar_obs_encoder = nn.ModuleDict({})
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        for t in self.t_unique:
            
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            self.obs_encoder[t_key] = TrajRNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, 
                                          hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)
            self.obs_encoder[t_key].to(device)
            
            self.tar_obs_encoder[t_key] = RNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, 
                                              hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)
            self.tar_obs_encoder[t_key].to(device)
        
        self.obs_encoder     = nn.ModuleDict(self.obs_encoder)
        self.tar_obs_encoder = nn.ModuleDict(self.tar_obs_encoder)
        
        self.edge_dim_gnn = len(self.t_unique) * 2 + 1
        
        self.GNNencoder = SocialInterGNN(num_layers=n_layers_gnn, emb_dim=es_gnn,
                                         in_dim=self.obs_encoding_size, edge_dim=self.edge_dim_gnn,
                                         device = device)
        if use_map:
            if scene_encoder is None:
                self.scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
            else:
                self.scene_encoder = scene_encoder
        else:
                self.scene_encoder = None
            
        
        if use_map:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size+self.es_gnn+self.scene_encoding_size, 
                                         n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)  
        else:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size+self.es_gnn, 
                                         n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)    

        # move model to specified device
        self.device = device
        self.to(device)

    def _encode_conditionals(self, x, T, scene=None):
        x_in = x
        if self.rel_coords:
            x_in = x[...,1:,:] - x[...,:-1,:]
            
        x_enc     = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1, self.obs_encoding_size), device = self.device)
        x_tar_enc = torch.zeros((x.shape[0], x.shape[2] - 1, self.obs_encoding_size), device = self.device)
        for t in self.t_unique:
            assert t in T
            t_in = T == t
            
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            x_enc[t_in], _ = self.obs_encoder[t_key](x_in[t_in])
            # target agent is always first agent
            x_tar_enc[t_in[:,0]], _ = self.tar_obs_encoder[t_key](x_in[[t_in[:,0],0]])
            
            
        
        # TODO: Maybe put all the outputs here, and try to punish changes between timesteps
        # To prevent sudden fluctuation
        x_enc     = x_enc[...,-1,:]
        x_tar_enc = x_tar_enc[...,-1,:]
        
        # Deal with autoencoder here
        num_nodes = x_enc.size(dim=1)
        row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
        col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        graph_edge_node = torch.stack([row, col], dim=0).unsqueeze(0).repeat((len(x_enc),1,1))
        # Combine batch into one matrix
        sample_enc = torch.arange(len(x_enc), device = self.device).unsqueeze(1).repeat((1,num_nodes ** 2))
        graph_edge_index = graph_edge_node + sample_enc.unsqueeze(1) * num_nodes
        # Combine
        graph_edge_index = graph_edge_index.permute(1,0,2).reshape(2,-1)
        
        T_one_hot = (T.unsqueeze(-1) == self.t_unique.unsqueeze(0).unsqueeze(0)).float().reshape(-1, len(self.t_unique))
        
        class_in_out = T_one_hot[graph_edge_index, :].permute(1,0,2).reshape(-1, 2 * len(self.t_unique))
        
        D = torch.sqrt(torch.sum((x[:,None,:,-1] - x[:,:,None,-1]) ** 2, dim = -1))
        dist_in_out = D[sample_enc, graph_edge_node[:,0], graph_edge_node[:,1]]
        dist_in_out = dist_in_out.reshape(-1,1)
         
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        
        graph_batch = torch.arange((len(x_enc)), dtype=torch.long, device=self.device).repeat_interleave(num_nodes)
        
        socialData = Data(x=x_enc.reshape(-1, self.obs_encoding_size), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=graph_batch)
        
        social_enc = self.GNNencoder(socialData)
        
        if scene is not None:
            scene_enc = self.scene_encoder(scene)
            total_enc = torch.concat((x_tar_enc, social_enc, scene_enc), dim=1)
        else:
            total_enc = torch.concat((x_tar_enc, social_enc), dim=1)
        return total_enc

    def log_prob(self, y_true, x, T, scene=None):
        z, log_abs_jacobian_det = self._inverse(y_true, x, T, scene)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det

    def _inverse(self, y_true, x, T, scene=None):
        if self.norm_rotation:
            x, angle = self._normalize_rotation(x)

        x_enc = self._encode_conditionals(x, T, scene) # history encoding
        y_rel_flat = y_true

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)

            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        
        z, jacobian_det = self.flow.inverse(y_rel_flat, x_enc)
        return z, jacobian_det
    
    def sample(self, n, x, T, scene=None):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x, T, scene) # history encoding
            
            if scene is not None:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size + self.es_gnn + self.scene_encoding_size)
            else:
                x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size + self.es_gnn)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.pred_steps) # predict n trajectories input
            
            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), n))
            
            return samples_rel, log_probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    
class Future_Encoder(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=4, hs=4, nl=3, device=0):
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden
    
    
class Future_Decoder(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, es=4, hs=4, nl=3, device=0):
        super().__init__()
        self.output = nn.Linear(es, nin)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.device = device
        self.cuda(self.device)
        
        self.embedding = nn.Linear(es, es)
        self.nl = nl

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.output(x)
        return x, hidden
    
    
class Future_Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Future_Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, traj):
        batch_size = traj.shape[0]
        
        target_length = traj.shape[1]
        
        outputs = torch.zeros(batch_size, target_length, 2).to(device)
        
        x_in = traj
        
        x_enc, hidden = self.encoder(x_in) # encode relative histories
        out = x_enc
                
        hidden = torch.tile(out[:,-1].unsqueeze(0), (self.decoder.nl,1,1))
        
        # Decoder part
        x = out[:,-1].unsqueeze(1)
                
        for t in range(0, target_length):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output.squeeze()
            
            x = hidden[-1].unsqueeze(1)
        
        return outputs, x_in

    
    
    