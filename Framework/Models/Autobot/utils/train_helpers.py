import torch
from scipy import special
import numpy as np
import torch.distributions as D
from torch.distributions import MultivariateNormal, Laplace


# ==================================== AUTOBOT-EGO STUFF ====================================

def get_BVG_distributions(pred):
    B = pred.size(0)
    T = pred.size(1)
    mu_x = pred[:, :, 0].unsqueeze(2)
    mu_y = pred[:, :, 1].unsqueeze(2)
    sigma_x = pred[:, :, 2]
    sigma_y = pred[:, :, 3]
    rho = pred[:, :, 4]

    cov = torch.zeros((B, T, 2, 2)).to(pred.device)
    cov[:, :, 0, 0] = sigma_x ** 2
    cov[:, :, 1, 1] = sigma_y ** 2
    cov[:, :, 0, 1] = rho * sigma_x * sigma_y
    cov[:, :, 1, 0] = rho * sigma_x * sigma_y

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist(pred):
    return Laplace(pred[:, :, :2], pred[:, :, 2:4])


def nll_pytorch_dist(pred, data, rtn_loss=True):
    # biv_gauss_dist = get_BVG_distributions(pred)
    biv_gauss_dist = get_Laplace_dist(pred)
    if rtn_loss:
        # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(-1).sum(1)  # Laplace
    else:
        # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
        return (-biv_gauss_dist.log_prob(data)).sum(dim=(1, 2))  # Laplace


def nll_loss_multimodes(pred, data, modes_pred, entropy_weight=1.0, kl_weight=1.0, use_FDEADE_aux_loss=True):
    """NLL loss multimodes for training. MFP Loss function
    Args:
      pred: [K, T, B, 5]
      data: [B, T, 5]
      modes_pred: [B, K], prior prob over modes
      noise is optional
    """
    modes = len(pred)
    nSteps, batch_sz, dim = pred[0].shape

    # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=False)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = modes_pred.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(data.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist(pred[kk].transpose(0, 1), data, rtn_loss=True) * post_pr[:, kk]
        loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions(pred[kk]).entropy())
    entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
    entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
    loss += entropy_weight * entropy_loss

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(modes_pred), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    if use_FDEADE_aux_loss:
        adefde_loss = l2_loss_fde(pred, data)
    else:
        adefde_loss = torch.tensor(0.0).to(data.device)

    return loss, kl_loss, post_entropy, adefde_loss


def l2_loss_fde(pred, data):
    fde_loss = torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1)
    ade_loss = torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1).mean(dim=2).transpose(0, 1)
    loss, min_inds = (fde_loss + ade_loss).min(dim=1)
    return 100.0 * loss.mean()


# ==================================== AUTOBOT-JOINT STUFF ====================================


def get_BVG_distributions_joint(pred):
    B = pred.size(0)
    T = pred.size(1)
    N = pred.size(2)
    mu_x = pred[:, :, :, 0].unsqueeze(3)
    mu_y = pred[:, :, :, 1].unsqueeze(3)
    sigma_x = pred[:, :, :, 2]
    sigma_y = pred[:, :, :, 3]
    rho = pred[:, :, :, 4]

    cov = torch.zeros((B, T, N, 2, 2)).to(pred.device)
    cov[:, :, :, 0, 0] = sigma_x ** 2
    cov[:, :, :, 1, 1] = sigma_y ** 2
    cov_val = rho * sigma_x * sigma_y
    cov[:, :, :, 0, 1] = cov_val
    cov[:, :, :, 1, 0] = cov_val

    biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
    return biv_gauss_dist


def get_Laplace_dist_joint(pred):
    return Laplace(pred[:, :, :, :2], pred[:, :, :, 2:4])


def nll_pytorch_dist_joint(pred, data, agents_masks):
    # biv_gauss_dist = get_BVG_distributions_joint(pred)
    biv_gauss_dist = get_Laplace_dist_joint(pred)
    num_active_agents_per_timestep = agents_masks.sum(2)
    loss = ((-biv_gauss_dist.log_prob(data).sum(-1) * agents_masks).sum(2) / num_active_agents_per_timestep).sum(1)
    return loss


def nll_loss_multimodes_joint(pred, ego_data, agents_data, ego_past, agents_past, mode_probs, 
                              entropy_weight=1.0, kl_weight=1.0, use_FDEADE_aux_loss=True, agent_types=None, predict_yaw=False):
    """
    Args:
      pred: [c, T, B, M, 5]
      ego_data: [B, T, 6]
      ego_past: [B, Tp, 6]
      agents_data: [B, T, M, 6]
      agents_past: [B, Tp, M, 6]
      mode_probs: [B, c], prior prob over modes
    """
    gt_agents = torch.cat((ego_data.unsqueeze(2), agents_data), dim=2) # [B, T, M, 5]
    modes, nSteps, batch_sz = pred.shape[:3]
    agents_masks = gt_agents[..., -1] # [B, T, M]

    gt_pos = gt_agents[..., :2] # [B, T, M, 2]
    gt_pos_past = torch.cat((ego_past.unsqueeze(2), agents_past), dim=2)[..., :2] # [B, Tp, M, 2]
    gt_pos_past = gt_pos_past[:,[-1]] # [B, 1, M, 2]

    # compute posterior probability based on predicted prior and likelihood of predicted scene.
    log_lik = np.zeros((batch_sz, modes))
    with torch.no_grad():
        for kk in range(modes):
            nll = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_pos, agents_masks)
            log_lik[:, kk] = -nll.cpu().numpy()

    priors = mode_probs.detach().cpu().numpy()
    log_posterior_unnorm = log_lik + np.log(priors)
    log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=1).reshape((batch_sz, 1))
    post_pr = np.exp(log_posterior)
    post_pr = torch.tensor(post_pr).float().to(gt_agents.device)
    post_entropy = torch.mean(D.Categorical(post_pr).entropy()).item()

    # Compute loss.
    nll_loss = 0.0
    for kk in range(modes):
        nll_k = nll_pytorch_dist_joint(pred[kk].transpose(0, 1), gt_pos, agents_masks) * post_pr[:, kk]
        nll_loss += nll_k.mean()

    # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
    entropy_vals = []
    for kk in range(modes):
        entropy_vals.append(get_BVG_distributions_joint(pred[kk]).entropy())
    entropy_vals_tensor = torch.stack(entropy_vals).permute(2, 0, 3, 1) # [B, c, M, T]
    
    # Multiply with agents_masks to get entropy for active agents only
    entropy_loss_time = (entropy_vals_tensor * agents_masks.permute(0,2,1).unsqueeze(1)).sum(-1) # [B, c, M]

    # Take mean over agents
    entropy_losses = entropy_loss_time.mean(-1) # [B, c]

    # Take max over modes, and then mean over batch
    entropy_loss = entropy_weight * entropy_losses.max(1).values.mean()

    # KL divergence between the prior and the posterior distributions.
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
    kl_loss = kl_weight*kl_loss_fn(torch.log(mode_probs), post_pr)

    # compute ADE/FDE loss - L2 norms with between best predictions and GT.
    pred_pos = pred[..., :2] # [c, T, B, M, 2]
    ade_loss, sde_loss, fde_loss, yaw_loss = l2_loss_fde_joint(pred_pos, gt_pos, gt_pos_past, agents_masks, agent_types, predict_yaw)

    return nll_loss, entropy_loss, kl_loss, post_entropy, ade_loss, sde_loss, fde_loss, yaw_loss


def l2_loss_fde_joint(pred_pos, gt_pos, gt_pos_past, agents_masks, agent_types, predict_yaw):
    # pred_pos: [c, T, B, M, 2]
    # gt_pos: [B, T, M, 2]
    # gt_pos_past: [B, 1, M, 2]
    # agents_masks: [B, T, M]
    # agent_types: [B, M, 5]

    assert pred_pos.shape[1] == gt_pos.shape[1]
    assert pred_pos.shape[2] == gt_pos.shape[0]
    assert pred_pos.shape[3] == gt_pos.shape[2]


    diff = pred_pos.transpose(1, 2) - gt_pos.unsqueeze(0) # [c, B, T, M, 2]
    dist = torch.norm(diff, 2, dim=-1) # [c, B, T, M]
    dist_mean = (dist * agents_masks.unsqueeze(0)).mean(-1) # [c, B, T]
    # invert first two dimensions
    dist_mean = dist_mean.transpose(0, 1) # [B, c, T]

    fde_loss = dist_mean[..., -1] # [B, c]
    sde_loss = dist_mean[..., 0] # [B, c]
    ade_loss = dist_mean.mean(dim=2) # [B, c]

    # Extend positions with last true value
    last_past_pos = gt_pos_past.unsqueeze(0).repeat_interleave(pred_pos.size(0), dim=0) # [c, B, 1, M, 5]

    # Combine with predicted positions
    positions = torch.concat((last_past_pos, pred_pos.transpose(1,2)), dim=2) # [c, B, T+1, M, 2]

    # get true positions
    positions_gt = torch.concat((gt_pos_past, gt_pos), dim=1) # [B, T+1, M, 2]

    # Get angle based on positions
    diff = positions[:,:,1:] - positions[:,:,:-1] # [c, B, T, M, 2]
    angle_pred = torch.atan2(diff[..., 1], diff[..., 0]) # [c, B, T, M]

    # Get angle based on GT
    diff_gt = positions_gt[:,1:] - positions_gt[:,:-1] # [B, T, M, 2]
    angle_gt = torch.atan2(diff_gt[..., 1], diff_gt[..., 0]) # [B, T, M]

    diff_yaw = angle_pred - angle_gt.unsqueeze(0) # [c, B, T, M]

    # Allow for periodicity of angles
    diff_yaw = torch.fmod(diff_yaw + np.pi, 2*np.pi) - np.pi

    dist_yaw = torch.abs(diff_yaw) # [c, B, T, M]
    yaw_loss = (dist_yaw * agents_masks.unsqueeze(0)).mean(2) # across time, [c, B, M]
    
    # Take mean over vehicles only
    vehicles_only = (agent_types[..., 1] == 1).unsqueeze(0) # [1, B, M]
    yaw_loss = (yaw_loss * vehicles_only).mean(-1).transpose(0, 1)  # [B, c]

    combined_loss = fde_loss + ade_loss # [B, c]
    if predict_yaw:
        combined_loss += yaw_loss
    
    assert combined_loss.shape == (pred_pos.shape[2], pred_pos.shape[0])

    mode_ind = combined_loss.min(dim=1).indices # [B]
    batch_ind = torch.arange(combined_loss.size(0)).to(mode_ind.device)

    ade_loss_final = ade_loss[batch_ind, mode_ind].mean()
    sde_loss_final = sde_loss[batch_ind, mode_ind].mean()
    fde_loss_final = fde_loss[batch_ind, mode_ind].mean()
    yaw_loss_final = yaw_loss[batch_ind, mode_ind].mean()

    return ade_loss_final, sde_loss_final, fde_loss_final, yaw_loss_final