import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))

def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]

class GMM2D(td.Distribution):
    r"""
    Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """
    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM2D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:])
        self.components = log_pis.shape[-1]
        self.dimensions = 2
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = self.reshape_to_components(mus)         # [..., N, 2]
        self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 2]
        self.sigmas = torch.exp(self.log_sigmas)                       # [..., N, 2]
        self.one_minus_rho2 = 1 - corrs**2                        # [..., N]
        self.one_minus_rho2 = torch.clamp(self.one_minus_rho2, min=1e-5, max=1)  # otherwise log can be nan
        self.corrs = corrs  # [..., N]

        self.L = torch.stack([torch.stack([self.sigmas[..., 0], torch.zeros_like(self.log_pis)], dim=-1),
                              torch.stack([self.sigmas[..., 1] * self.corrs,
                                           self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2)],
                                          dim=-1)],
                             dim=-2)

        self.pis_cat_dist = td.Categorical(logits=log_pis)

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        corrs_sigma12 = cov_mats[..., 0, 1]
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs = corrs_sigma12 / (torch.prod(sigmas, dim=-1))
        return cls(log_pis, mus, log_sigmas, corrs)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = (self.mus +
                       torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(size=sample_shape + self.mus.shape, device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        return torch.sum(mvn_samples*selector, dim=-2)

    def traj_sample(self, sample_shape=torch.Size()):
        mvn_samples = self.mus
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        return torch.sum(mvn_samples * selector, dim=-2)

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 2]
        dx = value - self.mus                       # [..., N, 2]

        exp_nominator = ((torch.sum((dx/self.sigmas)**2, dim=-1)  # first and second term of exp nominator
                          - 2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)))    # [..., N]

        component_log_p = -(2*np.log(2*np.pi)
                            + torch.log(self.one_minus_rho2)
                            + 2*torch.sum(self.log_sigmas, dim=-1)
                            + exp_nominator/self.one_minus_rho2) / 2

        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def get_for_node_at_time(self, n, t):
        return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                              self.log_sigmas[:, n:n+1, t:t+1], self.corrs[:, n:n+1, t:t+1])

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min()
                    x_max = self.mus[:, n, t, :, 0].max()
                    y_min = self.mus[:, n, t, :, 1].min()
                    y_max = self.mus[:, n, t, :, 1].max()
                    search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max, 0.01),
                                                              torch.arange(y_min, y_max, 0.01)]), dim=2
                                              ).view(-1, 2).float().to(self.device)

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov], dim=-1),
                         torch.stack([cov, self.sigmas[..., 1]**2], dim=-1)],
                        dim=-2)
        return E



class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, model_registrar, xz_size, node_type):
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        self.model_registrar = model_registrar
        self.node_type = node_type
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con
        #print(f"initial_conditions set as {self.initial_conditions}")

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError

    def create_graph(self, xz_size):
        pass


class Unicycle(Dynamic):
    def init_constants(self):
        self.F_s = torch.eye(4, device=self.device, dtype=torch.float32)
        self.F_s[0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * self.dt
        self.F_s_t = self.F_s.transpose(-2, -1)

    def create_graph(self, xz_size):
        model_if_absent = nn.Linear(xz_size + 1, 1)
        self.p0_model = self.model_registrar.get_model(f"{self.node_type}/unicycle_initializer", model_if_absent)

    def dynamic(self, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        x_p = x[0]
        y_p = x[1]
        phi = x[2]
        v = x[3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        d1 = torch.stack([(x_p
                           + (a / dphi) * dcos_domega
                           + v * dsin_domega
                           + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt),
                          (y_p
                           - v * dcos_domega
                           + (a / dphi) * dsin_domega
                           - (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt),
                          phi + dphi * self.dt,
                          v + a * self.dt], dim=0)
        d2 = torch.stack([x_p + v * torch.cos(phi) * self.dt + (a / 2) * torch.cos(phi) * self.dt ** 2,
                          y_p + v * torch.sin(phi) * self.dt + (a / 2) * torch.sin(phi) * self.dt ** 2,
                          phi * torch.ones_like(a),
                          v + a * self.dt], dim=0)
        return torch.where(~mask, d1, d2)

    def integrate_samples(self, control_samples, x=None):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        ph = control_samples.shape[-2]
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        v_0 = self.initial_conditions['vel'].unsqueeze(1)
        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])

        # phi_0 = phi_0 + torch.tanh(self.p0_model(torch.cat((x, phi_0), dim=-1)))

        u = torch.stack([control_samples[..., 0], control_samples[..., 1]], dim=0)
        x = torch.stack([p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim = 0).squeeze(dim=-1)

        mus_list = []
        for t in range(ph):
            x = self.dynamic(x, u[..., t])
            mus_list.append(torch.stack((x[0], x[1]), dim=-1))

        pos_mus = torch.stack(mus_list, dim=2)
        return pos_mus

    def compute_control_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        F = torch.zeros(sample_batch_dim + [components, 4, 2],
                        device=self.device,
                        dtype=torch.float32)

        phi = x[2]
        v = x[3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = ((v / dphi) * torch.cos(phi_p_omega_dt) * self.dt
                        - (v / dphi) * dsin_domega
                        - (2 * a / dphi ** 2) * torch.sin(phi_p_omega_dt) * self.dt
                        - (2 * a / dphi ** 2) * dcos_domega
                        + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt ** 2)
        F[..., 0, 1] = (1 / dphi) * dcos_domega + (1 / dphi) * torch.sin(phi_p_omega_dt) * self.dt

        F[..., 1, 0] = ((v / dphi) * dcos_domega
                        - (2 * a / dphi ** 2) * dsin_domega
                        + (2 * a / dphi ** 2) * torch.cos(phi_p_omega_dt) * self.dt
                        + (v / dphi) * torch.sin(phi_p_omega_dt) * self.dt
                        + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt ** 2)
        F[..., 1, 1] = (1 / dphi) * dsin_domega - (1 / dphi) * torch.cos(phi_p_omega_dt) * self.dt

        F[..., 2, 0] = self.dt

        F[..., 3, 1] = self.dt

        F_sm = torch.zeros(sample_batch_dim + [components, 4, 2],
                           device=self.device,
                           dtype=torch.float32)

        F_sm[..., 0, 1] = (torch.cos(phi) * self.dt ** 2) / 2

        F_sm[..., 1, 1] = (torch.sin(phi) * self.dt ** 2) / 2

        F_sm[..., 3, 1] = self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def compute_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        one = torch.tensor(1)
        F = torch.zeros(sample_batch_dim + [components, 4, 4],
                        device=self.device,
                        dtype=torch.float32)

        phi = x[2]
        v = x[3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = one
        F[..., 1, 1] = one
        F[..., 2, 2] = one
        F[..., 3, 3] = one

        F[..., 0, 2] = v * dcos_domega - (a / dphi) * dsin_domega + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt
        F[..., 0, 3] = dsin_domega

        F[..., 1, 2] = v * dsin_domega + (a / dphi) * dcos_domega + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt
        F[..., 1, 3] = -dcos_domega

        F_sm = torch.zeros(sample_batch_dim + [components, 4, 4],
                           device=self.device,
                           dtype=torch.float32)

        F_sm[..., 0, 0] = one
        F_sm[..., 1, 1] = one
        F_sm[..., 2, 2] = one
        F_sm[..., 3, 3] = one

        F_sm[..., 0, 2] = -v * torch.sin(phi) * self.dt - (a * torch.sin(phi) * self.dt ** 2) / 2
        F_sm[..., 0, 3] = torch.cos(phi) * self.dt

        F_sm[..., 1, 2] = v * torch.cos(phi) * self.dt + (a * torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 3] = torch.sin(phi) * self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def integrate_distribution(self, control_dist_dphi_a, x):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        sample_batch_dim = list(control_dist_dphi_a.mus.shape[0:2])
        ph = control_dist_dphi_a.mus.shape[-3]
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        v_0 = self.initial_conditions['vel'].unsqueeze(1)
        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])

        phi_0 = phi_0 + torch.tanh(self.p0_model(torch.cat((x, phi_0), dim=-1)))

        dist_sigma_matrix = control_dist_dphi_a.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(sample_batch_dim + [control_dist_dphi_a.components, 4, 4],
                                              device=self.device)

        u = torch.stack([control_dist_dphi_a.mus[..., 0], control_dist_dphi_a.mus[..., 1]], dim=0)
        x = torch.stack([p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim=0)

        pos_dist_sigma_matrix_list = []
        mus_list = []
        for t in range(ph):
            F_t = self.compute_jacobian(sample_batch_dim, control_dist_dphi_a.components, x, u[:, :, :, t])
            G_t = self.compute_control_jacobian(sample_batch_dim, control_dist_dphi_a.components, x, u[:, :, :, t])
            dist_sigma_matrix_t = dist_sigma_matrix[:, :, t]
            pos_dist_sigma_matrix_t = (F_t.matmul(pos_dist_sigma_matrix_t.matmul(F_t.transpose(-2, -1)))
                                       + G_t.matmul(dist_sigma_matrix_t.matmul(G_t.transpose(-2, -1))))
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t[..., :2, :2])

            x = self.dynamic(x, u[:, :, :, t])
            mus_list.append(torch.stack((x[0], x[1]), dim=-1))

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        pos_mus = torch.stack(mus_list, dim=2)
        return GMM2D.from_log_pis_mus_cov_mats(control_dist_dphi_a.log_pis, pos_mus, pos_dist_sigma_matrix)
