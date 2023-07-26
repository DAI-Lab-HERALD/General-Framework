import torch

def abs_to_rel(y, x_t, alpha):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[...,1:,:] = (y_rel[...,1:,:] - y_rel[...,:-1,:]) # steps relative to each other
        y_rel = y_rel * alpha # scale up for numeric reasons
        return y_rel

def rel_to_abs(y_rel, x_t, alpha):
    y_abs = y_rel / alpha
    return torch.cumsum(y_abs, dim=-2) + x_t 

def rotate(x, x_t, angles_rad):
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

def normalize_rotation(x, y_true=None):
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
    x = rotate(x, x_t, rot_angles_rad)
    
    if y_true != None:
        y_true = rotate(y_true, x_t, rot_angles_rad)
        return x, y_true, rot_angles_rad # inverse
    else:
        return x, rot_angles_rad # forward pass