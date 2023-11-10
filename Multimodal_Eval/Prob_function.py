import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.cluster._optics import cluster_optics_dbscan, cluster_optics_xi
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
import scipy as sp
from scipy.sparse.csgraph import minimum_spanning_tree 


def DBCV_multiple_clusterings(X, clusterings, optics = None):
    num_samples, num_features = X.shape
    assert len(X.shape) == 2
    assert len(clusterings.shape) == 2

    # each row in clusterings is a different approach
    assert clusterings.shape[0] == num_samples

    # Get number of noise samples
    Num_noise_samples = (clusterings == -1).sum(0)
    Num_clusters = clusterings.max(0) + 1

    # Get valid clusterings
    useful = (Num_clusters + (Num_noise_samples > 0).astype(int)) > 1

    # get distances
    Dist = np.sqrt(((X[:, np.newaxis] - X[np.newaxis]) ** 2).sum(-1))

    # Get core distances
    Dist_sorted = np.sort(Dist, axis = 1)[:,1:optics.min_samples * 2]
    adist = (Dist_sorted ** (-1 * num_features)).mean(1) ** (-1 / num_features)

    # Get reachability
    Adist = np.maximum(adist[:, np.newaxis], adist[np.newaxis])
    Dreach = np.maximum(Dist, Adist)

    # Intialize best score (worst score of valid clustering is -1)
    values = -1.1 * np.ones(clusterings.shape[1])

    for i in range(clusterings.shape[1]):
        if not useful[i]:
            continue
        
        # Get cluster spread
        test_labels = clusterings[:,i].copy()
        test_clusters, test_size = np.unique(test_labels, return_counts = True)
        
        # Remove noise samples
        if test_clusters[0] == -1:
            test_clusters = test_clusters[1:]
            test_size = test_size[1:]

        num_clusters = len(test_clusters)

        # Get internal DSC (Density Sparseness of a Cluster)
        DSC = np.ones(num_clusters)
        # Get external DSPC (Density Separation of a Pair of Clusters)
        DSPC = np.ones(num_clusters)
        for label in test_clusters:
            cluster_samples = test_labels == label
            Dreach_cluster = Dreach[cluster_samples]

            # Get internal connections
            Dreach_int = Dreach_cluster[:, cluster_samples]

            # Get internal dispersion
            max_span_tree = minimum_spanning_tree(sp.sparse.csr_matrix(Dreach_int)).data.max()
            max_dist = Dist[cluster_samples][:, cluster_samples].max()
            DSC[label] = min(max_span_tree, max_dist)

            # Get external non noise points
            external = (test_labels >= 0) & (~cluster_samples)

            if external.any():
                # Get external connections
                Dreach_ext = Dreach_cluster[:, external]

                # Get DSCP min 
                DSPC[label] = Dreach_ext.min()
            else:
                DSPC[label] = DSC[label]

        # Get cluster validity
        CV = (DSPC - DSC) / np.maximum(np.maximum(DSC, DSPC), 1e-6)

        # Get weighted average
        values[i] = (CV * test_size).sum() / num_samples

        if False:
            import matplotlib.pyplot as plt
            import seaborn as sns

            colors = sns.color_palette("husl", Num_clusters[i])
            colors.append((0.0, 0.0, 0.0))

            data_colors = [colors[j] for j in clusterings[:,i]]

            if optics is not None:
                data_colors_ordered = [data_colors[j] for j in optics.ordering_]
                reachability = optics.reachability_[optics.ordering_]
                reachability[0] = reachability[1]

                plt.figure()
                plt.plot(np.arange(num_samples), adist[optics.ordering_], c = 'k')
                for j in range(num_samples - 1):
                    plt.plot([j, j + 1], [reachability[j], reachability[j + 1]], c = data_colors_ordered[j])
                plt.show()

            plt.figure()
            plt.scatter(X[:,0], X[:,1], s = 1, c = data_colors, alpha = 0.9)
            plt.axis('equal')
            plt.show()


    
    # Get best clustering
    best_cluster = np.argmax(values)
    return clusterings[:, best_cluster]



def silhouette_multiple_clusterings(X, clusterings, num_min_samples = 5):
    num_samples, num_features = X.shape
    assert len(X.shape) == 2
    assert len(clusterings.shape) == 2

    # each row in clusterings is a different approach
    assert clusterings.shape[0] == num_samples

    # Get valid clusterings
    useful = (clusterings.max(0) - clusterings.min(0)) > 0

    # get distances
    Dist = np.sqrt(((X[:, np.newaxis] - X[np.newaxis]) ** 2).sum(-1))

    # Intialize best score (worst score of valid clustering is -1)
    values = - 1.1 * np.ones(clusterings.shape[1]) 

    for i in range(clusterings.shape[1]):
        if not useful[i]:
            continue

        test_labels = clusterings[:,i]
        num_noise_samples = (test_labels == -1).sum()
        silhouette_labels = test_labels.copy()
        silhouette_labels[test_labels == -1] = test_labels.max() + 1 + np.arange(num_noise_samples)

        # Treat noise as separate cluster
        test_score_noise_separate = silhouette_score(Dist, silhouette_labels, metric = 'precomputed')
        test_score_noise_combined = silhouette_score(Dist, test_labels, metric = 'precomputed')

        noise_fac = num_noise_samples / len(X)
        values[i] = noise_fac * test_score_noise_separate + (1 - noise_fac) * test_score_noise_combined
    
    # Get best clustering
    best_cluster = np.argmax(values)
    return clusterings[:, best_cluster]



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
    def __init__(self, use_cluster = 'silhouette', use_PCA = True, 
                 use_std = True, estimator = 'KDE', min_std = 0.1):
        self.fitted = False
        
        # Get design opportunities
        self.use_cluster = use_cluster
        self.use_PCA     = use_PCA
        self.use_std     = use_std
        self.estimator   = estimator

        assert self.use_cluster in ['silhouette', 'DBCV', 'None'], "Cluster method not recognized"
        assert self.estimator in ['KDE', 'GMM', 'KNN'], "Estimator not recognized"
        
        # Get minimum std values
        self.min_std = min_std
        
        # Avoid unneeded combinations
        if self.estimator in ['KDE', 'KNN']:
            if not self.use_std:
                if not self.use_PCA:
                    raise ValueError("KDE is invariant to solely rotating samples, " +
                                     "making not using PCA without standardization unneeded.")
        
        else:
            if self.use_PCA or self.use_std:
                raise ValueError("GMM is invariant to rotation and stretching, " + 
                                 "which are consequenlty useless.")
            
        
    def fit(self, X, clusters = None):
        assert len(X.shape) == 2
        
        self.num_features = X.shape[1]
        
        if clusters is None:
            if (self.use_cluster != 'None') and len(X) >= 5:  
                num_min_samples = X.shape[0] * self.num_features / 400 
                num_min_samples = int(np.clip(num_min_samples, min(5, X.shape[0]), 20))     

                # Get reachability plot
                optics = OPTICS(min_samples = num_min_samples) 
                optics.fit(X)
                    
                reachability = optics.reachability_[np.isfinite(optics.reachability_)] 
                
                # Potential plotting
                # Test potential cluster extractionssomething like
                self.Eps = np.linspace(0, 1, 100) ** 2
                self.Eps = self.Eps * (reachability.max() - reachability.min()) + reachability.min()
                self.Xi = np.linspace(0.01, 0.99, 99)

                Method = np.repeat(np.array(['Eps', 'Xi']), (len(self.Eps), len(self.Xi)))
                Params = np.concatenate((self.Eps, self.Xi), axis = 0)
                
                # Initializes empty clusters
                Clustering = np.zeros((len(X), len(Method)), int)

                # Iterate over all potential cluster extractions
                for i in range(len(Method)):
                    method = Method[i]
                    param  = Params[i]
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

                        noise_clusters = test_clusters[test_size == 1]
                        test_labels[np.isin(test_labels, noise_clusters)] = -1
                        test_labels[test_labels > -1] = np.unique(test_labels[test_labels > -1], return_inverse = True)[1]

                        Clustering[:, i] = test_labels

                if self.use_cluster == 'silhouette':
                    self.cluster_labels = silhouette_multiple_clusterings(X, Clustering)
                elif self.use_cluster == 'DBCV':
                    self.cluster_labels = DBCV_multiple_clusterings(X, Clustering, optics)  
                else:
                    raise ValueError('Clustering method not recognized')
            else:
                self.cluster_labels = np.zeros(len(X))
        else:
            self.cluster_labels = clusters.copy()
            
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
            num_samples = len(X_label)
            assert num_samples == cluster_size[i]

            # Get mean and std
            self.means[i] = X_label.mean(0)
            Stds[i]       = X_label.std(0)

            # Shift coordinate system origin to mean
            X_label_stand = (X_label - self.means[[i]])
            
            if self.use_PCA:
                # Repeat data if not enough samples are available
                if num_samples < self.num_features:
                    c = np.tile(X_label_stand, (int(np.ceil(self.num_features / num_samples)),1))
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
            if self.estimator == 'KDE':
                model = KernelDensity(kernel = 'gaussian', bandwidth = 'silverman').fit(X_label_pca)
            elif self.estimator == 'GMM':
                reg_covar = max(1e-6, self.min_std ** 2)
                model = GaussianMixture(reg_covar = reg_covar).fit(X_label_pca)
            elif self.estimator == 'KNN':
                # get num neighbors
                num_neighbours = max(min(num_samples, 3), int(np.sqrt(num_samples)))
                # Fit BallTree
                ball_tree = BallTree(X_label_pca)
                # Get volume factor of hypersphere
                volume_unit_hypersphere = np.pi**(self.num_features / 2) / sp.special.gamma(self.num_features / 2 + 1)
                # Get standard probability adjustment
                log_adjustment = np.log(num_neighbours) - np.log(volume_unit_hypersphere) - np.log(num_samples)

                model = (ball_tree, log_adjustment, num_neighbours)
            else:
                raise ValueError('Estimator not recognized')
                
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
            if isinstance(model, KernelDensity) or isinstance(model, GaussianMixture):
                log_probs[:,i] = model.score_samples(X_stand)
            elif isinstance(model, tuple):
                # Load model
                (ball_tree, log_adjustment, num_neighbours) = model

                # Get distances to nearest neighbours
                dist, _ = ball_tree.query(X_stand, num_neighbours)

                # Get radius
                radius = dist.max(axis = -1)

                # Calculate log not recognizedprob values
                log_probs[:,i] = log_adjustment - np.log(radius) * self.num_features
            else:
                raise ValueError('Estimator not recognized')

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

            # Reset radnom seed to be sure
            np.random.seed(random_state)

            # Sample transformed samples from model
            if isinstance(self.Models[label], KernelDensity):
                X_label_stand = self.Models[label].sample(num, random_state)
            elif isinstance(self.Models[label], GaussianMixture):
                X_label_stand = self.Models[label].sample(num)[0]
            else:
                raise ValueError('Estimator cannot generate samples.')
                
            # Apply inverse transformation to get original coordinate samples
            X_label = X_label_stand @ np.linalg.inv(self.T_mat[label]) + self.means[[label]]
            
            # Add samples to output set
            samples.append(X_label)
            
        samples = np.concatenate(samples, axis = 0)
        
        # Shuffle samples
        np.random.shuffle(samples)

        return samples
            
            
        
        
        
        
        