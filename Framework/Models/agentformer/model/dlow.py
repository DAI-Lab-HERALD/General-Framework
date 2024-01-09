import torch
from torch import nn
from torch.nn import functional as F
from agentformer.utils.torch import *
from agentformer.utils.config import Config
from .common.mlp import MLP
from .common.dist import *
from . import model_lib


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist_dlow'].kl(data['p_z_dist_infer']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['agent_num']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def diversity_loss(data, cfg):
    loss_unweighted = 0
    fut_motions = data['infer_dec_motion'].view(data['batch_size'] * data['agent_num'],
                                                data['infer_dec_motion'].shape[2], -1)
    diff = fut_motions.unsqueeze(1) - fut_motions.unsqueeze(2)
    dist = diff.pow(2).sum(dim = -1) 
    div = (- dist / cfg['d_scale']).exp()
    loss_unweighted = (div.sum(dim = (1,2)) - div.shape[-1])/(div.shape[-1] * (div.shape[-1] - 1))
    loss_unweighted = loss_unweighted.sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def recon_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'][:,:,:data['infer_dec_motion'].shape[3]].unsqueeze(2)
    if cfg.get('mask', True):
        mask = data['fut_mask'][:,:,:data['infer_dec_motion'].shape[3]].unsqueeze(2).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=2)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.sum() / diff.shape[1]
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


loss_func = {
    'kld': compute_z_kld,
    'diverse': diversity_loss,
    'recon': recon_loss,
}


""" DLow (Diversifying Latent Flows)"""
class DLow(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg
        self.nk = nk = cfg.sample_k
        self.nz = nz = cfg.nz
        self.share_eps = cfg.get('share_eps', True)
        self.train_w_mean = cfg.get('train_w_mean', False)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())

        pred_cfg = Config(cfg.pred_cfg, tmp=False, create_dirs=False)
           
        pred_cfg.yml_dict["use_map"] = cfg.yml_dict["use_map"]
        
        if cfg.yml_dict["use_map"]:
            pred_cfg.yml_dict["input_type"]               = cfg.yml_dict["input_type"]
            pred_cfg.yml_dict.map_encoder["map_channels"] = cfg.yml_dict.map_encoder["map_channels"]
        
        pred_cfg.yml_dict["past_frames"] = cfg.yml_dict["past_frames"]
        pred_cfg.yml_dict["min_past_frames"] = cfg.yml_dict["min_past_frames"]
              
        pred_cfg.yml_dict["future_frames"] = cfg.yml_dict["future_frames"]
        pred_cfg.yml_dict["min_future_frames"] = cfg.yml_dict["min_future_frames"]
        
        pred_cfg.yml_dict["sample_k"] = cfg.yml_dict["sample_k"]
        pred_cfg.yml_dict["loss_cfg"]["sample"]["k"] = cfg.yml_dict["sample_k"]
        
        
        pred_model = model_lib.model_dict[pred_cfg.model_id](pred_cfg)
        self.pred_model_dim = pred_cfg.tf_model_dim
        cp_path = cfg.yml_dict['model_path']
        model_cp = torch.load(cp_path, map_location='cpu')
        pred_model.load_state_dict(model_cp['model_dict'])
        pred_model.eval()
        self.pred_model = [pred_model]

        # Dlow's Q net
        self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, nk * nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, nk * nz)
        
    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data

    def main(self, mean=False, need_weights=False, sample_num = None):
        pred_model = self.pred_model[0]
        if hasattr(pred_model, 'use_map') and pred_model.use_map:
            Maps = self.data['agent_maps'].reshape(-1, 3, 100,100)
            D_maps = pred_model.map_encoder(Maps)
            self.data['map_enc'] = D_maps.reshape(self.data['batch_size'],-1,D_maps.shape[1])
        pred_model.context_encoder(self.data)

        if not mean:
            if self.share_eps:
                eps = torch.randn([self.data['batch_size'], 1, self.nz]).to(self.device)
                eps = eps.repeat((1, self.data['agent_num'] * self.nk, 1))
            else:
                eps = torch.randn([self.data['batch_size'], self.data['agent_num'], self.nz]).to(self.device)
                eps = eps.repeat_interleave(self.nk, dim=1)

        qnet_h = self.q_mlp(self.data['agent_context'])
        A = self.q_A(qnet_h).view(self.data['batch_size'], -1, self.nz)
        b = self.q_b(qnet_h).view(self.data['batch_size'], -1, self.nz)

        z = b if mean else A*eps + b
        logvar = (A ** 2 + 1e-8).log()
        self.data['q_z_dist_dlow'] = Normal(mu=b, logvar=logvar)
        
        if sample_num is None:
            pred_model.future_decoder(self.data, mode='infer', sample_num=self.nk, autoregress=True, z=z, need_weights=need_weights)
        else:
            pred_model.future_decoder(self.data, mode='infer', sample_num=sample_num, autoregress=True, z=z, need_weights=need_weights)
        return self.data
    
    def forward(self):
        return self.main(mean=self.train_w_mean)

    def inference(self, mode, sample_num, need_weights=False):
        self.main(mean=True, need_weights=need_weights, sample_num = sample_num)
        res = self.data[f'infer_dec_motion']
        if mode == 'recon':
            res = res[:, :, 0]
        return res, self.data

    def compute_loss(self):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_dict, loss_unweighted_dict

    def step_annealer(self):
        pass