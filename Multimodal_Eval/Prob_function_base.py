import numpy as np
import scipy as sp
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#%% First baseline
class MP_Windows():
    def __init__(self, principal_directions = None, min_std = 0.01):
        self.fitted = False

        # Principal directions
        self.principal_directions = principal_directions
            
        self.min_std = max(0.0, np.double(min_std))
        
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_samples, self.num_features = X.shape
        
        # Set principal directions
        if self.principal_directions is None:
            self.principal_directions = self.num_features
        else:
            self.principal_directions = min(int(self.principal_directions), self.num_features)

        # Set up kernels
        self.V = np.zeros((self.num_features, self.num_features, self.principal_directions))
        self.L = np.zeros((self.num_features, self.principal_directions))

        # Get k-closest neighbors
        balltree = BallTree(X)
        num_neighbours = max(self.num_features, int(np.sqrt(self.num_samples) / 3))
        dist, idx = balltree.query(X, num_neighbours)

        # Center points
        M = X[idx] - X[:, np.newaxis]

        # Compute covariance matrices
        _, S, Vi = np.linalg.svd(M, full_matrices = False)

        self.X = X
        self.V = Vi[:, :self.principal_directions, :].transpose((0, 2, 1))
        self.L = S[:, :self.principal_directions] ** 2 / self.num_samples + self.min_std ** 2

        # Compute prob normalisation
        self.R  = self.num_features * np.log(2 * np.pi) + np.sum(np.log(self.L), axis = 1)
        self.R += (self.num_features - self.principal_directions) * np.log(self.min_std)

        self.fitted = True

        return self
        
        
    def prob(self, X, return_log = False):
        assert self.fitted, 'The model was not fitted yet'
        
        assert len(X.shape) == 2
        assert X.shape[1] == self.num_features

        # Compute kernel distances
        DX = X[np.newaxis] - self.X[:, np.newaxis]

        # Compute probabilities
        Q  = (1.0 / self.min_std ** 2) * np.sum(DX ** 2, axis = -1)
        Q += np.sum((1.0 / self.L[:,np.newaxis] - 1 / self.min_std ** 2) * np.matmul(DX, self.V) ** 2, axis = -1)
        # Q.shape: num_samples x num_test_samples

        # Sum over kernels
        Log_probs = -0.5 * (self.R[:, np.newaxis] + Q).T
        log_probs = sp.special.logsumexp(Log_probs, axis = -1) - np.log(self.num_samples)
        
        # Deal with overflow
        if return_log:
            return log_probs
        else:
            prob = np.exp(log_probs)
            return prob
    
    
    def score_samples(self, X):
        return self.prob(X, return_log = True)
        
    
    def sample(self, num_samples = 1, random_state = 0):
        assert self.fitted, 'The model was not fitted yet'
        # Determine Kernel belonging
        np.random.seed(random_state)

        # Samples kernels
        num_full = num_samples // self.num_samples
        kernels = np.repeat(np.arange(self.num_samples), num_full)

        remaining = num_samples - num_full * self.num_samples
        if remaining > 0:
            kernels_extra = np.random.choice(np.arange(self.num_samples), remaining, replace = False,
                                             p = np.ones(self.num_samples) / self.num_samples)
            kernels = np.concatenate((kernels, kernels_extra))
        
        # Get helper function
        identity_matrix = np.eye(self.num_features)[np.newaxis]

        # Get needed kernels
        kernels, num_samples_kernel = np.unique(kernels, return_counts = True)

        # Extract eigen value matrix
        L_adj = 1.0 / self.L[kernels] - 1.0 / self.min_std ** 2
        L_adj = L_adj[:, :, np.newaxis] * identity_matrix

        Sigma_inv = np.matmul(self.V[kernels].transpose(0, 2, 1), np.matmul(L_adj, self.V[kernels])) + identity_matrix / self.min_std ** 2
        Sigma = np.linalg.inv(Sigma_inv)

        # Get mean value
        Mu = self.X[kernels]

        samples = []
        # Sample from kernels
        for i, num in enumerate(num_samples_kernel):
            samples.append(np.random.multivariate_normal(Mu[i], Sigma[i], num))

        # Concatenate samples
        samples = np.concatenate(samples, axis = 0)

        # Shuffle samples
        np.random.shuffle(samples)

        return samples
    
class MPS_Windows(MP_Windows):
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_samples, self.num_features = X.shape
        
        # Set principal directions
        if self.principal_directions is None:
            self.principal_directions = self.num_features
        else:
            self.principal_directions = min(int(self.principal_directions), self.num_features)

        # Set up kernels
        self.V = np.zeros((self.num_features, self.num_features, self.principal_directions))
        self.L = np.zeros((self.num_features, self.principal_directions))

        # Get k-closest neighbors
        balltree = BallTree(X)
        num_neighbours = max(self.num_features, int(np.sqrt(self.num_samples)))
        dist, idx = balltree.query(X, num_neighbours)

        # Center points
        M = X[idx] - X[:, np.newaxis]

        # Compute covariance matrices
        _, S, Vi = np.linalg.svd(M, full_matrices = False)

        self.X = X
        self.V = Vi[:, :self.principal_directions, :].transpose((0, 2, 1))
        self.L = S[:, :self.principal_directions] ** 2 / self.num_samples + self.min_std ** 2

        # Compute prob normalisation
        self.R  = self.num_features * np.log(2 * np.pi) + np.sum(np.log(self.L), axis = 1)
        self.R += (self.num_features - self.principal_directions) * np.log(self.min_std)

        self.fitted = True
        return self
    
class MPK_Windows(MP_Windows):
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_samples, self.num_features = X.shape
        
        # Set principal directions
        if self.principal_directions is None:
            self.principal_directions = self.num_features
        else:
            self.principal_directions = min(int(self.principal_directions), self.num_features)

        # Set up kernels
        self.V = np.zeros((self.num_features, self.num_features, self.principal_directions))
        self.L = np.zeros((self.num_features, self.principal_directions))

        # Get k-closest neighbors
        balltree = BallTree(X)
        num_neighbours = max(self.num_features, int(np.sqrt(self.num_samples) / 1.5))
        dist, idx = balltree.query(X, num_neighbours)

        # Center points
        M = X[idx] - X[:, np.newaxis]

        # Compute covariance matrices
        _, S, Vi = np.linalg.svd(M, full_matrices = False)

        self.X = X
        self.V = Vi[:, :self.principal_directions, :].transpose((0, 2, 1))
        self.L = S[:, :self.principal_directions] ** 2 / num_neighbours + self.min_std ** 2

        # Compute prob normalisation
        self.R  = self.num_features * np.log(2 * np.pi) + np.sum(np.log(self.L), axis = 1)
        self.R += (self.num_features - self.principal_directions) * np.log(self.min_std)

        self.fitted = True
        return self
            

#%% Second baseline         
class KDevine():
    def __init__(self, min_std = 0.01):
        self.fitted = False

            
        
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_features = X.shape[1]

        U = np.zeros_like(X)
        if self.num_features > 1:
            # Fit marginal distributions
            self.marg_dist = []
            for j in range(self.num_features):
                Xj = X[:, [j]]
                Xj_stand = Xj / Xj.std()

                # Use Grid search to select bandwidth
                # Get possible bandwidths
                b_silverman = (len(Xj_stand) * 3 / 4) ** ( -1 / 5)
                b_min = b_silverman / 20
                b_max = b_silverman * 5
                bandwidths = np.logspace(np.log10(b_min), np.log10(b_max), 100)

                # Perform cross-validation to find the optimal bandwidth
                grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv = 20)
                grid.fit(Xj_stand)

                # Extract the optimal bandwidth
                optimal_bandwidth = grid.best_params_['bandwidth']   

                kde_pdf_j = KernelDensity(kernel = 'gaussian', bandwidth = optimal_bandwidth).fit(Xj_stand)
                kde_cdf_j = KDE1_cdf(Xj_stand[:,0], optimal_bandwidth)

                dist = {'model_pdf': kde_pdf_j, 'model_cdf': kde_cdf_j, 'std': Xj.std()}
                self.marg_dist.append(dist)

                U[:, j] = kde_cdf_j.eval(Xj_stand)

            # Fit copula
            self.copula = Cupola_Vines().fit(U)
        
            self.fitted = True

        else:
            X_stand = X / X.std()

            # Use Grid search to select bandwidth
            # Get possible bandwidths
            b_silverman = (len(X_stand) * 3 / 4) ** ( -1 / 5)
            b_min = b_silverman / 20
            b_max = b_silverman * 5
            bandwidths = np.logspace(np.log10(b_min), np.log(b_max), 100)

            # Perform cross-validation to find the optimal bandwidth
            grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv = 20)
            grid.fit(X_stand)

            # Extract the optimal bandwidth
            optimal_bandwidth = grid.best_params_['bandwidth']   

            self = KernelDensity(kernel = 'gaussian', bandwidth = optimal_bandwidth).fit(X_stand)

        return self
        
        
    def prob(self, X, return_log = False):
        assert self.fitted, 'The model was not fitted yet'
        
        assert len(X.shape) == 2
        assert X.shape[1] == self.num_features

        # Transform to cupola
        U = np.zeros_like(X) 
        Log_probs = np.zeros_like(X)

        for j in range(self.num_features):
            # Standardize input
            Xj = X[:, [j]]
            Xj_stand = Xj / self.marg_dist[j]['std']

            U[:, j] = self.marg_dist[j]['model_cdf'].eval(Xj_stand)
            Log_probs[:, j]  = self.marg_dist[j]['model_pdf'].score_samples(Xj_stand)
            Log_probs[:, j] -= np.log(self.marg_dist[j]['std'])

        # Get log probs
        log_probs = Log_probs.sum(1) + self.copula.score_samples(U)

        # Deal with overflow
        if return_log:
            return log_probs
        else:
            prob = np.exp(log_probs).sum(1)
            return prob
    
    
    def score_samples(self, X):
        return self.prob(X, return_log = True)
        
    
    def sample(self, num_samples = 1, random_state = 0):
        assert self.fitted, 'The model was not fitted yet'

        # Sample from copula
        U_samples = self.copula.sample(num_samples, random_state = random_state)
        
        # Transform to marginals
        samples = np.zeros_like(U_samples)
        for j in range(self.num_features):
            samples[:, j] = self.marg_dist[j]['model_cdf'].inv(U_samples[:, j]) * self.marg_dist[j]['std']

        if not np.isfinite(samples).all():
            raise ValueError('Samples contain inf or nan')

        return samples
            
class KDE1_cdf():
    def __init__(self, Xj, b):
        if len(Xj.shape) > 1:
            assert len(Xj.shape) == 2
            assert Xj.shape[1] == 1
            Xj = Xj[:,0] 

        self.b = b
        self.Xj = np.sort(Xj)
        Uj = self.eval(self.Xj)
        self.inverter = sp.interpolate.interp1d(Uj, self.Xj, fill_value = 'extrapolate')

    def eval(self, X):
        if len(X.shape) > 1:
            assert len(X.shape) == 2
            assert X.shape[1] == 1
            X = X[:,0]   

        Z = (X[:, np.newaxis] - self.Xj[np.newaxis, :]) / self.b
        CDF = sp.stats.norm.cdf(Z)
        return CDF.mean(1)

    
    def inv(self, U):
        # Linear interpolation
        X = self.inverter(U)
        return X
             
class Cupola_Vines:
    def __init__(self):
        self.fitted = False
        
    def fit(self, U):
        assert len(U.shape) == 2
        
        self.num_features = U.shape[1]
        
        self.fitted = True

        # Get Vine tree
        # self.Tree is a list of self.num_feature - 1 lists
        # Each list contains a list of self.num_feature - layer -1 tuples
        # each tuple contains the edge nodes, as well as the parent nodes in the connection 

        self.BCupolas = {}

        Uhats = {}
        for j in range(self.num_features):
            Uhats[(j, ())] = U[:, j]            

        self.Tree = []

        for m in range(self.num_features - 1):
            # Get tree layer
            if self.num_features == 2:
                tree_layer = [(0, 1, ())]
            elif self.num_features > 2:
                tree_layer = []
                # Get somehow that viable nodes
                kendell_tau = np.zeros((self.num_features - m, self.num_features - m))
                Edge_candidates = {}
                if m == 0:
                    for i in range(self.num_features):
                        for j in range(i, self.num_features):
                            if i == j:
                                kendell_tau[i, j] = 1.0
                                continue
                            tau_value = np.abs(sp.stats.kendalltau(U[:, i], U[:, j])[0])           
                            kendell_tau[i, j] = tau_value
                            kendell_tau[j, i] = tau_value
                            Edge_candidates[(i, j)] = (i, j, ())

                else:
                    for i in range(self.num_features - m):
                        for j in range(i, self.num_features - m):
                            if i == j:
                                kendell_tau[i, j] = 1.0
                                continue
                            
                            i_edge = self.Tree[m - 1][i]
                            j_edge = self.Tree[m - 1][j]

                            i_set = set((i_edge[0], i_edge[1], *i_edge[2]))
                            j_set = set((j_edge[0], j_edge[1], *j_edge[2]))
                            
                            i_differ = tuple(i_set - j_set)
                            j_differ = tuple(j_set - i_set)

                            if not (len(i_differ) == 1 and len(j_differ) == 1):
                                continue

                            parents = tuple(np.sort(np.array(tuple(i_set.intersection(j_set)))))
                            # Check if edges are compatible

                            Ui = Uhats[(i_differ[0], parents)]
                            Uj = Uhats[(j_differ[0], parents)]


                            tau_value = max(0.00001, np.abs(sp.stats.kendalltau(Ui, Uj)[0]))           
                            kendell_tau[i, j] = tau_value
                            kendell_tau[j, i] = tau_value

                            Edge_candidates[(i, j)] = (i_differ[0], j_differ[0], parents)
                # Get max spanning tree            
                K = sp.sparse.csr_matrix(-kendell_tau)
                max_span_tree = sp.sparse.csgraph.minimum_spanning_tree(K)

                # Get edges
                edges = np.stack(max_span_tree.nonzero(), 1)
                assert len(edges) == self.num_features - m - 1, "In layer {} there are {} edges".format(m, len(edges))
                for edge in edges:
                    tree_layer.append(Edge_candidates[(min(edge), max(edge))])
            else:
                raise ValueError('Number of features should be greater than 1')

            assert len(tree_layer) == self.num_features - m - 1

            self.Tree.append(tree_layer)


            self.BCupolas[m] = {}

            # Go through each edge
            for edge in tree_layer:
                # Get edge
                i, j, parents = edge

                assert len(parents) == m
                if m > 1:
                    parents = tuple(np.sort(np.array(parents)))

                # Get conditional U
                U_i = Uhats[(i, parents)]
                U_j = Uhats[(j, parents)]

                U_ij = np.stack((U_i, U_j), axis = 1)
                self.BCupolas[m][edge] = KDE2().fit(U_ij)

                # Get conditional distribution
                _, Uhat = self.BCupolas[m][edge].score_samples(U_ij)
                i_parents = tuple(np.sort(np.array(parents + (j,))))
                j_parents = tuple(np.sort(np.array(parents + (i,))))

                Uhats[(i, i_parents)] = Uhat[:,0]
                Uhats[(j, j_parents)] = Uhat[:,1]

        return self
        
        
    def prob(self, U, return_log = False):
        assert self.fitted, 'The model was not fitted yet'
        
        assert len(U.shape) == 2
        assert U.shape[1] == self.num_features
        
        Uhats = {}
        for j in range(self.num_features):
            Uhats[(j, ())] = U[:, j]            

        log_probs = np.zeros(U.shape[0])
        for m in range(self.num_features - 1):
            tree = self.Tree[m]
            assert len(tree) == self.num_features - m - 1

            # Go through each edge
            for edge in tree:
                # Get edge
                i, j, parents = edge

                assert len(parents) == m
                if m > 1:
                    parents = tuple(np.sort(np.array(parents)))

                # Score conditional bivariate distribution
                U_i = Uhats[(i, parents)]
                U_j = Uhats[(j, parents)]

                U_ij = np.stack((U_i, U_j), axis = 1)

                edge_prob, Uhat = self.BCupolas[m][edge].score_samples(U_ij)
                log_probs += edge_prob

                # Get conditional distribution
                i_parents = tuple(np.sort(np.array(parents + (j,))))
                j_parents = tuple(np.sort(np.array(parents + (i,))))

                Uhats[(i, i_parents)] = Uhat[:,0]
                Uhats[(j, j_parents)] = Uhat[:,1]


        # Deal with overflow
        if return_log:
            return log_probs
        else:
            prob = np.exp(log_probs).sum(1)
            return prob
    
    
    def score_samples(self, U):
        return self.prob(U, return_log = True)
        
    
    def sample(self, num_samples = 1, random_state = 0):
        assert self.fitted, 'The model was not fitted yet'
        # Determine cluster belonging
        np.random.seed(random_state)
        # Go through tree from top to bottom
        
        num_samples_safe = 5 * num_samples
        
        Uhats = {}
        for m in np.arange(self.num_features - 2, -1, -1):
            tree = self.Tree[m]
            assert len(tree) == self.num_features - m - 1
            # Go through each edge
            for edge in tree:
                # Get edge
                i, j, parents = edge

                # Get conditional distribution
                i_parents = tuple(np.sort(np.array(parents + (j,))))
                j_parents = tuple(np.sort(np.array(parents + (i,))))

                # Get potential Uhats case
                i_test = (i, i_parents)
                j_test = (j, j_parents)
                
                i_there = i_test in Uhats.keys()
                j_there = j_test in Uhats.keys()

                if i_there and j_there:
                    
                        Usampled = self.BCupolas[m][edge].sample(num_samples_safe, random_state = random_state, 
                                                                 Uhats_i = Uhats[i_test], Uhats_j = Uhats[j_test])
                elif not i_there and not j_there:
                    # Generate samples from bivariate distribution
                    Usampled = self.BCupolas[m][edge].sample(num_samples_safe)
                else:
                    if not i_there:
                        Usampled = self.BCupolas[m][edge].sample(num_samples_safe, random_state = random_state, Uhats_j = Uhats[j_test])
                    
                    if not j_there:
                        Usampled = self.BCupolas[m][edge].sample(num_samples_safe, random_state = random_state, Uhats_i = Uhats[i_test])

                Uhats[(i, parents)] = Usampled[:,0]
                Uhats[(j, parents)] = Usampled[:,1]
                


        # Shuffle samples
        samples = np.zeros((num_samples_safe, self.num_features))
        for m in range(self.num_features):
            samples[:, m] = Uhats[(m, ())]
        
        useful_samples = np.isfinite(samples).all(1)
        samples = samples[useful_samples]
        
        if len(samples) == 0:
            raise ValueError('No good samples are found')
        elif len(samples) >= num_samples:
            np.random.shuffle(samples)
            samples = samples[:num_samples]
        else:
            sample_ind = np.random.choice(len(samples), num_samples)
            samples = samples[sample_ind]
            
        return samples


class KDE2():
    def __init__(self):
        self.fitted = False
        
    def fit(self, U):
        assert len(U.shape) == 2
        assert U.shape[1] == 2

        # Transform to standard normal
        self.N = sp.stats.norm.ppf(1e-5 + (1 - 2 * 1e-5) * U)# Test something.py

        # Get bandwidth (Kernel methods for vine copula estimation (2014))
        # This source is identical to silverman's rule of thumb
        self.bandwidth = (1 / len(U)) ** (1 / 6)
        self.KDE = KernelDensity(kernel = 'gaussian', bandwidth = self.bandwidth).fit(self.N)
        
        self.fitted = True

        # Prepare for inverse h

        return self
        
    
    def score_samples(self, U):
        assert self.fitted, 'The model was not fitted yet'
        
        assert len(U.shape) == 2
        assert U.shape[1] == 2

        N = sp.stats.norm.ppf(1e-5 + (1 - 2 * 1e-5) * U)

        CN = sp.stats.norm.pdf(N[:,np.newaxis] - self.N[np.newaxis,:], scale = self.bandwidth)
        NN = sp.stats.norm.pdf(N)
        CN_log = np.log(CN)
        NN_log = np.log(NN)

        log_probs  = sp.special.logsumexp(CN_log.sum(2), axis = 1) - np.log(len(self.N))
        log_probs -= NN_log.sum(1)

        # Get rvalues
        RN = CN / NN[:,np.newaxis]
        RN_norm = RN / RN.sum(1, keepdims = True)
        
        # Get Integral values
        IN = sp.stats.norm.cdf(N[:,np.newaxis] - self.N[np.newaxis,:], scale = self.bandwidth)

        Uhats = (IN * RN_norm[:,:,[1,0]]).sum(1)
        if np.isnan(log_probs).any():
            raise ValueError('Log probs contain nan')
        return log_probs, Uhats
        
    
    def sample(self, num_samples = 1, random_state = 0, Uhats_i = None, Uhats_j = None):
        assert self.fitted, 'The model was not fitted yet'
        # Determine cluster belonging
        np.random.seed(random_state)

        if Uhats_i is None and Uhats_j is None:
            samples = self.KDE.sample(num_samples)
        elif Uhats_i is not None and Uhats_j is not None:
            # Use 50 x 50 gridf for interpolation
            Uh = np.stack((Uhats_i, Uhats_j), axis = 1)
            n = 51
            samples1 = sp.stats.norm.ppf(np.linspace(1e-6, 1 - 1e-6, n))
            samples2 = sp.stats.norm.ppf(np.linspace(1e-6, 1 - 1e-6, n))

            S1, S2 = np.meshgrid(samples1, samples2, indexing = 'ij')
            N = np.stack((S1, S2), axis = 2)

            CN = sp.stats.norm.pdf(N[:,:,np.newaxis] - self.N[np.newaxis,np.newaxis,:], scale = self.bandwidth)
            NN = sp.stats.norm.pdf(N)

            # Get rvalues
            RN = CN / NN[:,:,np.newaxis]
            RN_norm = RN / RN.sum(2, keepdims = True)
            
            # Get Integral values
            IN = sp.stats.norm.cdf(N[:,:,np.newaxis] - self.N[np.newaxis,np.newaxis,:], scale = self.bandwidth)

            UHp = (IN * RN_norm[:,:,:,[1,0]]).sum(2)

            interpolator = sp.interpolate.LinearNDInterpolator(UHp.reshape(-1,2), N.reshape(-1,2))

            samples = interpolator(Uh)

            # Check for convex hull problems
            if np.isnan(samples).any():
                # Get convex hull of test points
                hull = sp.spatial.ConvexHull(UHp.reshape(-1,2))
                Hull_points = hull.points[hull.simplices]
            
                repeat = 0
                while np.isnan(samples).any() and repeat < 20:
                    repeat += 1
                    # Find problem points
                    problem = np.where(np.isnan(samples).any(1))[0]
    
                    # Get problem points
                    Up = Uh[problem]
    
                    i_points = np.argmin(np.sqrt(np.sum((Up[np.newaxis,np.newaxis]-Hull_points[:,:,np.newaxis])**2, axis = -1)).sum(1), axis = 0)
                    hull_points = Hull_points[i_points]
    
                    l = np.sum((hull_points[:,1]-hull_points[:,0])**2, axis = -1)
                    t = np.maximum(0., np.minimum(1., ((Up-hull_points[:,0]) * 
                                                       (hull_points[:,1]-hull_points[:,0])).sum(-1)/l))
                    proj_points = hull_points[:,0] + t[:,np.newaxis]*(hull_points[:,1]-hull_points[:,0]) 
                    # Move from hull into hull
                    proj_points = Up + 1.0001 * (proj_points - Up)
                    samples[problem] = interpolator(proj_points)

        else:
            # Samples kernels
            if Uhats_j is None:
                idx_h = 0
                Uh = Uhats_i

                idx_n = 1 
            else:
                idx_h = 1
                Uh = Uhats_j

                idx_n = 0

            samples = self.KDE.sample(num_samples)

            # interpolate with 30 points
            samples2 = sp.stats.norm.ppf(np.linspace(1e-6, 1 - 1e-6, 31))

            CN = sp.stats.norm.pdf(samples[:,[idx_n]] - self.N[np.newaxis,:,idx_n], scale = self.bandwidth)
            NN = sp.stats.norm.pdf(samples[:,[idx_n]])

            # Get rvalues
            RN = CN / NN
            RN_norm = RN / RN.sum(1, keepdims = True)
            
            # Get Integral values
            INh = sp.stats.norm.cdf(samples2[:,np.newaxis] - self.N[np.newaxis,:,idx_h], scale = self.bandwidth)

            Uhp = (INh[np.newaxis] * RN_norm[:,np.newaxis]).sum(2)
            Uhp[:,0]  = 0.0
            Uhp[:,-1] = 1.0

            for i in range(num_samples):
                samples[i, idx_h] = sp.interpolate.interp1d(Uhp[i], samples2, assume_sorted = True, 
                                                            fill_value = 'extrapolate')(Uh[i])
        
        U_samples = sp.stats.norm.cdf(samples)
        return U_samples
        
