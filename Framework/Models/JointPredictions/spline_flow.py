
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from TrajFlow.rqs import *


class FCN(nn.Module):
    """ Simple fully connected network. """

    def __init__(self, nin, nout, nh=[24, 24, 24], device=0):
        super().__init__()
        if type(nh) != list:
            nh = [nh] * 3
        self.layers = [nin] + nh + [nout]
        self.net = []
        for (l1, l2) in zip(self.layers, self.layers[1:]):
            self.net.extend([nn.Linear(l1, l2), nn.ELU()])
        self.net.pop() # remove last activation
        self.net = nn.Sequential(*self.net)
        self.device = device
        self.cuda(self.device)

    def forward(self, x):
        return self.net(x)

class FlowSequential(nn.Sequential):
    """ Container for normalizing flow layers. """
    
    def forward(self, x, c=None):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, c)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, c=None):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, c)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

class InvertiblePermutation(nn.Module):
    """
    Randomly permutes inputs and implments the reverse operation.
    Returns 0 for log absolute jacobian determinant, as the jacobian
    determinant of a permutation is 1 (orthogonal matrix) and log(1)
    is 0.
    """

    def __init__(self, dim, reverse_only=False):
        super().__init__()
        self.dim = dim
        self.reverse_only = reverse_only
        self.register_buffer('perm_idx', torch.randperm(dim))
        self.register_buffer('perm_idx_inv', torch.zeros_like(self.perm_idx))
        
        # initialize perm_idx_inv to reverse perm_idx sorting
        for i, j in zip(self.perm_idx, torch.arange(self.dim)):
            self.perm_idx_inv[i] = j

    def forward(self, x, c=None):
        if self.reverse_only:
            return x.flip(-1), 0
        x = x[..., self.perm_idx]
        return x, 0

    def inverse(self, x, c=None):
        if self.reverse_only:
            return x.flip(-1), 0
        x = x[..., self.perm_idx_inv]
        return x, 0

class CouplingNSF(nn.Module):
    """
    Neural spline flow with coupling conditioner [Durkan et al. 2019].
    """
    def __init__(self, dim, dimc=0, K=5, B=3, hidden_dim=8, device=0):
        super().__init__()
        self.dim = dim
        self.dimc = dimc # conditionals dim
        self.K = K # number of knots
        self.B = B # spline support
        # output: for each input dim params that define one spline
        self.conditioner = FCN(dim // 2 + self.dimc, (3 * K - 1) * (self.dim - (self.dim // 2)), hidden_dim)
        self.device = device
        self.to(self.device)

    def _get_spline_params(self, x1, c=None):
        x = x1 if c == None else torch.cat((x1, c), -1) # concat inputs
        out = self.conditioner(x).view(-1, self.dim - (self.dim // 2), 3 * self.K - 1) # calls f(x_1:d), arange spline params by input dim
        W, H, D = torch.split(out, self.K, dim = 2) # get knot width, height, derivatives
        return W, H, D

    def forward(self, x, c=None):
        # compute spline parameters
        x1, x2 = x[:, :self.dim // 2], x[:, self.dim // 2:] # split input
        W, H, D = self._get_spline_params(x1, c)
        # apply spline transform
        x2, ld = unconstrained_RQS(x2, W, H, D, inverse=False, tail_bound=self.B)
        log_det = torch.sum(ld, dim = 1)
        return torch.cat([x1, x2], dim = 1), log_det

    def inverse(self, z, c=None):
        # compute spline parameters
        z1, z2 = z[:, :self.dim // 2], z[:, self.dim // 2:]
        W, H, D = self._get_spline_params(z1, c)
        # apply spline transform
        z2, ld = unconstrained_RQS(z2, W, H, D, inverse=True, tail_bound=self.B)
        log_det = torch.sum(ld, dim = 1)
        return torch.cat([z1, z2], dim = 1), log_det

class NeuralSplineFlow(nn.Module):

    def __init__(self, nin, nc=0, n_layers=5, K=5, B=3, hidden_dim=8, device=0):
        super().__init__()
        self.nin = nin
        self.nc = nc # size of conditionals
        self.n_layers = n_layers
        
        self.net = []
        for i in range(self.n_layers):
            self.net.append(CouplingNSF(self.nin, self.nc, K, B, hidden_dim=hidden_dim, device=device))
            self.net.append(InvertiblePermutation(self.nin, reverse_only=False))
        self.net.pop()
        self.net = FlowSequential(*self.net)

        self.device = device
        self.to(self.device)

    def forward(self, z, c=None):
        return self.net(z, c)

    def inverse(self, x, c=None):
        return self.net.inverse(x, c)

    def sample(self, n, c=None):
        with torch.no_grad():
            if c != None:
                assert(c.size(0) == n)
            samples = torch.zeros([n, self.nin]).to(self.device)
            z = torch.randn_like(samples)
            x, log_det = self.forward(z, c)
            return x
    
    def log_prob(self, x, c=None):
        """
        Computes the log likelihood of the input. The likelihood of z can be
        evaluted as a log sum, because the product of univariate normals gives
        a multivariate normal with diagonal covariance.
        """
        z, log_abs_jacobian_det = self.inverse(x, c)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det