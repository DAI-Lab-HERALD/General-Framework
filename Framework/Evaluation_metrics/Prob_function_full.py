import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.cluster._optics import cluster_optics_dbscan, cluster_optics_xi
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy as sp


class OPTICS_GMM():
    '''
    This is a method for creating a point density invariant probability density
    function using nonparametrics methods.
    
    Spcecifically, it involves a two step process, where first clustering is
    performed using teh OPTICS algorithm.
    
    For each cluster, one then fits a multivariate Gaussian function to form
    a Gaussian Multi Mixture model (GMM), according to which one can calcualte
    probability density values or sample from.
    '''
    def __init__(self, use_cluster = True, use_PCA = True, 
                 use_std = True, use_KDE = True, min_std = 0.1):
        self.fitted = False
        
        # Get design opportunities
        self.use_cluster = use_cluster
        self.use_PCA     = use_PCA
        self.use_std     = use_std
        self.use_KDE     = use_KDE
        
        # Get minimum std values
        self.min_std = min_std
        
        # Avoid unneeded combinations
        if self.use_KDE:
            if not self.use_std:
                if not self.use_PCA:
                    raise ValueError("KDE is invariant to solely rotating samples, " +
                                     "making not using PCA without standardization unneeded.")
        
        else:
            if self.use_PCA or self.use_std:
                raise ValueError("GMM is invariant to rotation and stretching, " + 
                                 "which are consequenlty useless.")
            
        
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_features = X.shape[1]
        
        
        if self.use_cluster and len(X) >= 5:  
            num_min_samples = X.shape[0] / 20
            num_min_samples = int(np.clip(num_min_samples, min(5, X.shape[0]), 20))     

            # Get reachability plot
            optics = OPTICS(min_samples = num_min_samples, 
                            min_cluster_size = 5)
            optics.fit(X)
                
            reachability = optics.reachability_[np.isfinite(optics.reachability_)] 
            
            # Test potential cluster extractions
            self.Eps = np.linspace(reachability.min(), reachability.max(), 100)
            self.Xi = np.linspace(0.01, 0.99, 99)

            Method = np.repeat(np.array(['Eps', 'Xi']), (len(self.Eps), len(self.Xi)))
            Params = np.concatenate((self.Eps, self.Xi), axis = 0)

            # Intialize best score (worst score of valid clustering is -1)
            best_score = -1.1

            # Iterate over all potential cluster extractions
            for method, param in zip(Method, Params):
                # Cluster using dbscan
                if method == 'Eps':
                    eps = param
                    test_labels = cluster_optics_dbscan(reachability   = optics.reachability_,
                                                        core_distances = optics.core_distances_,
                                                        ordering       = optics.ordering_,
                                                        eps            = eps)
                # Cluster using xi
                elif method == 'Xi':
                    xi = param
                    test_labels, _ = cluster_optics_xi(reachability           = optics.reachability_,
                                                       predecessor            = optics.predecessor_,
                                                       ordering               = optics.ordering_,
                                                       min_samples            = num_min_samples,
                                                       min_cluster_size       = 2,
                                                       xi                     = xi,
                                                       predecessor_correction = optics.predecessor_correction)
                else:
                    raise ValueError('Clustering method not recognized')    
                
                # Check for improvement
                if len(np.unique(test_labels)) > 1: 
                    # Check if there are lusters of size one
                    test_clusters, test_size = np.unique(test_labels, return_counts = True)
                    cluster_size_one = test_size[test_clusters >= 0].min() < 2
                    
                    # Avoid clusters consisting of only one sample
                    if not cluster_size_one:
                        # Evaluate clustering
                        test_score = silhouette_score(X, test_labels)

                        # Check for improvement
                        if test_score > best_score:
                            best_score = test_score
                            self.cluster_labels = test_labels
            
            # If no viable clustering method was found
            if not hasattr(self, 'cluster_labels'):
                self.cluster_labels = np.zeros(len(X))
        else:
            self.cluster_labels = np.zeros(len(X))
            
        unique_labels, cluster_size = np.unique(self.cluster_labels, return_counts = True)
            
        # Fit distribution to each cluster of data
        self.Models = [None] * len(unique_labels)
        
        # initialise rotation matrix for PCA
        self.means = np.zeros((len(unique_labels), self.num_features))
        self.T_mat = np.zeros((len(unique_labels), self.num_features, self.num_features))
        
        # Get probability adjustment
        self.log_det_T_mat = np.zeros(len(unique_labels))
        
        Stds = np.zeros((len(unique_labels), self.num_features)) 

        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            # Get cluster data
            X_label = X[self.cluster_labels == label]
            assert len(X_label) == cluster_size[i]

            # Get mean and std
            self.means[i] = X_label.mean(0)
            Stds[i]       = X_label.std(0)

            # Shift coordinate system origin to mean
            X_label_stand = (X_label - self.means[[i]])
            
            if self.use_PCA:
                # Repeat data if not enough samples are available
                if len(X_label) < self.num_features:
                    c = np.tile(X_label_stand, (int(np.ceil(self.num_features/len(X_label))),1))
                else:
                    c = X_label_stand.copy()
    
                # calculate PCA on X_label_stand -> get rot matrix and std
                # Stabalize correlation matrix if necessary
                attempt = 0
                successful_pca = False
                while not successful_pca:
                    try:
                        pca = PCA(random_state = 0).fit(c)
                        successful_pca = True
                    except:
                        e_fac = 10 ** (0.5 * attempt - 6)
                        c[:self.num_features] += np.eye(self.num_features) * e_fac 
                                
                        # Prepare next attempt
                        attempt += 1
                    
                    if not successful_pca:
                        print('PCA failed, was done again with different random start.')
                    
                # Exctract components std
                pca_std = np.sqrt(pca.explained_variance_)

                # Extract roation matrix
                self.T_mat[i]  = pca.components_.T
            else:
                pca_std = Stds[i].copy()
                self.T_mat[i]  = np.eye(self.num_features)
            
            # Initiallize probability adjustment
            self.log_det_T_mat[i] = np.log(np.abs(np.linalg.det(self.T_mat[i])))
        
            # Apply standardization
            if self.use_std:
                # Apply minimum std levels
                pca_std = pca_std * (pca_std.max() - self.min_std) / pca_std.max() + self.min_std
                
                # Adjust T_mat accordingly
                self.T_mat[i]         /= pca_std[np.newaxis]
                self.log_det_T_mat[i] -= np.log(pca_std).sum()
            
            # Apply transformation matrix
            X_label_pca = X_label_stand @ self.T_mat[i] # @ is matrix multiplication
            
            # Fit Surrogate distribution
            if self.use_KDE:
                model = KernelDensity(kernel = 'gaussian', bandwidth = 'silverman').fit(X_label_pca)
            else:
                reg_covar = max(1e-6, self.min_std ** 2)
                model = GaussianMixture(reg_covar = reg_covar).fit(X_label_pca)
                
            self.Models[i] = model
            
        # consider noise values
        if unique_labels[0] == -1:
            X_noise = X[self.cluster_labels == -1]
            # set noise std
            
            # assume that no rotation is necessary rot_mat_pca[0] to be an identity matrix
            if len(self.T_mat) > 1 and self.use_std:
                stds = np.maximum(Stds[1:].mean(0), self.min_std)
                self.T_mat[0] = np.diag(1 / stds)
                self.log_det_T_mat[0] = -np.sum(np.log(stds))
            else:
                self.T_mat[0]         = np.eye(self.num_features)
                self.log_det_T_mat[0] = 0.0

            # Apply transformation matrix
            X_noise_stand = (X_noise - self.means[[0]]) @ self.T_mat[0] 
            
            # Fit Surrogate distribution 
            # We assume each noise point is its own cluster, and therefore, using kde or a number
            # of GMMs is more or less equivalent
            
            # calculate silverman rule assuming only 1 sample 
            bandwidth = ((self.num_features + 2) / 4) ** ( -1 / (self.num_features + 4))
            model_noise = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(X_noise_stand)
            
            self.Models[0] = model_noise
        
        # Get cluster probabilities
        self.probs = cluster_size / cluster_size.sum()
        self.log_probs = np.log(self.probs)
        
        self.fitted = True

        return self
        
        
    def prob(self, X, return_log = False):
        assert self.fitted, 'The model was not fitted yet'
        
        assert len(X.shape) == 2
        assert X.shape[1] == self.num_features
        
        # calculate logarithmic probability
        prob = np.zeros(len(X))
        
        log_probs = np.zeros((len(X), len(self.Models)), dtype = np.float64)
        
        for i, model in enumerate(self.Models):
            X_stand = (X - self.means[[i]]) @ self.T_mat[i]

            # Get in model log probabilty
            log_probs[:,i] = model.score_samples(X_stand)

            # adjust log prob for transformation
            log_probs[:,i] += self.log_det_T_mat[i] 

            # adjust log prob for cluster likelihood
            log_probs[:,i] += self.log_probs[i]
        
        # Deal with overflow
        if return_log:
            return log_probs
        else:
            prob = np.exp(log_probs).sum(1)
            return prob
    
    
    def score_samples(self, X):
        log_probs = self.prob(X, return_log = True)
        l_probs = sp.special.logsumexp(log_probs, axis = -1)
        return l_probs
        
    
    def sample(self, num_samples = 1, random_state = 0):
        assert self.fitted, 'The model was not fitted yet'
        
        # Determine cluster belonging
        np.random.seed(random_state)
        labels = np.random.choice(np.arange(len(self.Models)), num_samples, p = self.probs)
        
        samples = []
        
        # generate from different clusters
        for label in np.unique(labels):
            # Get number of samples from cluster
            num = (label == labels).sum()

            X_label_stand = self.Models[label].sample(num, random_state)
                
            # Apply inverse transformation to get original coordinate samples
            X_label = X_label_stand @ np.linalg.inv(self.T_mat[label]) + self.means[[label]]
            
            # Add samples to output set
            samples.append(X_label)
            
        samples = np.concatenate(samples, axis = 0)
        
        # Shuffle samples
        np.random.shuffle(samples)

        return samples
            
            
        
        
        
        
        
