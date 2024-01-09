

def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'][:,:,:data['train_dec_motion'].shape[2]] - data['train_dec_motion']
    if cfg.get('mask', True):
        mask = data['fut_mask'][:,:,:data['train_dec_motion'].shape[2]]
        diff *= mask.unsqueeze(-1)
    loss_unweighted = diff.pow(2).sum() 
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[1]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['agent_num']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
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
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss
}