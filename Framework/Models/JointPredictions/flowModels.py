import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from TrajFlow.spline_flow import NeuralSplineFlow
from JointPredictions.modules import PastSceneEncoder, TrajRNNEncoder, StaticEnvCNN


class InteractiveFlow(nn.Module):

    def __init__(self, pred_steps, alpha, beta, gamma, B=15., 
                use_map=False, interactions=False, 
                rel_coords=True, norm_rotation=False, device=0,
                obs_encoding_size=16, n_layers_rnn=3, es_rnn=16, hs_rnn=16,
                envCNNparams=None,
                socialGNNparams=None, T_all = None):
        
        nn.Module.__init__(self)
        self.pred_steps = pred_steps
        self.output_size = pred_steps 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation

        self.use_map = use_map
        self.interactions = interactions
        

        # core modules
        self.obs_encoding_size = obs_encoding_size
        self.scene_encoding_size = envCNNparams['enc_dim']
        self.n_layers_rnn = n_layers_rnn
        self.es_rnn = es_rnn
        self.hs_rnn = hs_rnn
        
        self.es_gnn = socialGNNparams['emb_dim']
        self.n_layers_gnn = socialGNNparams['num_layers']
        
        # observation encoders
        self.obs_encoder = nn.ModuleDict({})
        
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        for t in self.t_unique:
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            self.obs_encoder[t_key] = TrajRNNEncoder(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, 
                                          hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)
            self.obs_encoder[t_key].to(device)
            
        self.obs_encoder     = nn.ModuleDict(self.obs_encoder)
        

        self.edge_dim_gnn = len(self.t_unique) * 2 + 1
        
        self.Socialencoder = PastSceneEncoder(socialGNNparams=socialGNNparams, envCNNparams=envCNNparams, T_all=T_all, device = device)
                    
        flow_nc = self.es_gnn
        
        self.flow = NeuralSplineFlow(nin=self.output_size, nc=flow_nc, 
                                        n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)    

        # move model to specified device
        self.device = device
        self.to(device)

    def _encode_conditionals(self, x, T, scene=None):
        x_in = x
        if self.rel_coords:
            x_in = x[...,1:,:] - x[...,:-1,:]
            
        x_enc     = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1, self.obs_encoding_size), device = self.device)
        for t in self.t_unique:
            # assert t in T
            t_in = T == t
            
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            x_enc[t_in], _ = self.obs_encoder[t_key](x_in[t_in])            
            
               
        social_enc = self.Socialencoder(x_enc, x[...,-1,:], T, scene)

        return social_enc
    
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
        # TODO check if this needs to be put back
        # if len(x_t.shape) > 3:
        #     x_t = x_t[:,[0]]
        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[...,[-1],:] - x[...,[-2],:]
        # TODO check if this needs to be put back
        # if len(x_t_rel.shape) > 3:
        #     x_t_rel = x_t_rel[:,[0]]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[...,1], x_t_rel[...,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad # inverse
        else:
            return x, rot_angles_rad # forward pass

    def log_prob(self, y_true, x, T, scene=None):
        z, log_abs_jacobian_det = self._inverse(y_true, x, T, scene)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det

    def _inverse(self, y_true, x, T, scene=None):
        if self.norm_rotation:
            x, _ = self._normalize_rotation(x)

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
    
    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError


    def sample(self, n, x, T, scene=None):
        with torch.no_grad():

            # assert False
            if self.norm_rotation:
                x, _ = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x, T, scene) # history encoding
            
            if scene is not None:
                x_enc_expanded = x_enc.repeat_interleave(n, dim=0) #self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size + self.es_gnn + self.scene_encoding_size)
            else:
                x_enc_expanded = x_enc.repeat_interleave(n, dim=0) #self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size + self.es_gnn)
            n_total = n * x.size(0)
            # output_shape = (x.size(0), n, self.pred_steps) # predict n trajectories input TODO see if needed
            output_shape = (x.size(0)*n, self.pred_steps)
            
            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            # samples_rel = samples_rel.view(*output_shape) # TODO see if needed
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det)#.view((x.size(0), n)) # TODO see if needed
            
            return samples_rel, log_probs
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    
    
    
    