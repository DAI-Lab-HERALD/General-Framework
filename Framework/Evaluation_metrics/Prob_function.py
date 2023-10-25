import numpy as np
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.cluster._optics import cluster_optics_xi, cluster_optics_dbscan
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import scipy as sp


from sklearn.metrics import pairwise_distances

def dunn_index(X, labels):
    # Calculate cluster centroids and their pairwise distances
    cluster_distances = []
    cluster_centers = []
    for label in np.unique(labels[labels >= 0]):
        cluster_points = X[labels == label]
        cluster_centroid = cluster_points.mean(axis=0, keepdims = True)
        cluster_centers.append(cluster_centroid)
        centroid_distance = pairwise_distances(cluster_centroid, cluster_points)
        cluster_distances.append(np.mean(centroid_distance))

    # Calculate the minimum inter-cluster distance
    max_intra_cluster_distance = np.max(cluster_distances)

    # Calculate the maximum intra-cluster distance
    cluster_centers = np.concatenate(cluster_centers, axis = 0)
    noise_points = X[labels == -1]
    
    if len(noise_points) > 0:
        noise_cluster_ditances = pairwise_distances(noise_points, cluster_centers)
        min_noise_cluster_distance = np.min(noise_cluster_ditances)
    else:
        min_noise_cluster_distance = 0.0
    
    if len(cluster_centers) > 1:
        inter_cluster_ditances = pairwise_distances(cluster_centers)
        min_inter_cluster_distance = np.min(inter_cluster_ditances[inter_cluster_ditances > 0])
        
        fac = (labels == -1).mean()
        
        min_inter_distance = fac * min_noise_cluster_distance + (1 - fac) * min_inter_cluster_distance
    else:
        min_inter_distance = min_noise_cluster_distance
    
    # Compute the Dunn Index
    dunn_index = min_inter_distance / max_intra_cluster_distance

    return dunn_index


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
    def __init__(self):
        self.fitted = False
        
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_features = X.shape[1]
        
        num_min_samples = X.shape[0] / 20
        self.num_min_samples = int(np.clip(num_min_samples, min(5, X.shape[0]), 20))
        
        if len(X) > self.num_min_samples:
            # Get clusters using the OPTICS
            # self.Xi = np.linspace(0.01, 0.99, 99)
            # self.Num_cluster = np.zeros(len(self.Xi))
            # self.score       = np.ones(len(self.Xi)) * np.nan
            # self.dunn_score  = np.ones(len(self.Xi)) * np.nan
            # for i, xi in enumerate(self.Xi):
            #     if i == 0:
            #         self.optics = OPTICS(min_samples = self.num_min_samples, 
            #                              min_cluster_size = 5, xi = xi)
            #         self.optics.fit(X)
            #     else:
            #         self.optics.labels_, _ = cluster_optics_xi(reachability           = self.optics.reachability_,
            #                                                    predecessor            = self.optics.predecessor_,
            #                                                    ordering               = self.optics.ordering_,
            #                                                    min_samples            = self.optics.min_samples,
            #                                                    min_cluster_size       = 5,
            #                                                    xi                     = xi,
            #                                                    predecessor_correction = self.optics.predecessor_correction)
                    
            #     self.cluster_labels = self.optics.labels_
            #     self.Num_cluster[i] = self.cluster_labels.max() + 1
            #     if len(np.unique(self.cluster_labels)) > 1:
            #         self.dunn_score[i] = dunn_index(X, self.cluster_labels)
            #         self.score[i]      = silhouette_score(X, self.cluster_labels)
            
            # reachability = self.optics.reachability_[np.isfinite(self.optics.reachability_)] 
            # self.Eps = np.linspace(reachability.min(), reachability.max(), 200)
            # self.Eps_Num_cluster = np.zeros(len(self.Eps))
            # self.Eps_score       = np.ones(len(self.Eps)) * np.nan
            # self.Eps_dunn_score  = np.ones(len(self.Eps)) * np.nan
            # for i, eps in enumerate(self.Eps):
            #     self.cluster_labels = cluster_optics_dbscan(reachability   = self.optics.reachability_,
            #                                                 core_distances = self.optics.core_distances_,
            #                                                 ordering       = self.optics.ordering_,
            #                                                 eps            = eps)
                
            #     self.Eps_Num_cluster[i] = self.cluster_labels.max() + 1
            #     if len(np.unique(self.cluster_labels)) > 1:
            #         self.Eps_dunn_score[i] = dunn_index(X, self.cluster_labels)
            #         self.Eps_score[i]      = silhouette_score(X, self.cluster_labels)
            
            # Get reachability plot
            optics = OPTICS(min_samples = self.num_min_samples, 
                            min_cluster_size = 5)
            optics.fit(X)
            self.cluster_labels = optics.labels_
            
            if len(np.unique(self.cluster_labels)) > 1:
                best_score = silhouette_score(X, self.cluster_labels)
            else:
                best_score = -1
                
            reachability = optics.reachability_[np.isfinite(optics.reachability_)] 
            
            self.Eps = np.linspace(reachability.min(), reachability.max(), 100)
            for i, eps in enumerate(self.Eps):
                test_labels = cluster_optics_dbscan(reachability   = optics.reachability_,
                                                    core_distances = optics.core_distances_,
                                                    ordering       = optics.ordering_,
                                                    eps            = eps)
            
                if len(np.unique(test_labels)) > 1:
                    # Avoid clusters with size 1
                    test_size = np.unique(test_labels, return_counts = True)[1]
                    if test_labels.min() == -1:
                        cluster_size_one = test_size[1:].min() < 2
                    else:
                        cluster_size_one = test_size.min() < 2
                    
                    if not cluster_size_one:
                        test_score = silhouette_score(X, test_labels)
                        if test_score > best_score:
                            best_score = test_score
                            self.cluster_labels = test_labels
        else:
            self.cluster_labels = -1 * np.ones(len(X))
            
        unique_labels, cluster_size = np.unique(self.cluster_labels, return_counts = True)
            
        # Fit GMM to each model
        self.KDEs = [None] * len(unique_labels)
        
        # initialise rotation matrix for PCA
        self.means = np.zeros((len(unique_labels), self.num_features))
        self.T_mat = np.zeros((len(unique_labels), self.num_features, self.num_features))
        
        # Get probability adjustment
        self.log_det_T_mat = np.zeros(len(unique_labels))
        
        Stds = np.zeros((len(unique_labels), self.num_features)) 
        min_std = 0.1

        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            # Get cluster data
            X_label = X[self.cluster_labels == label]
            assert len(X_label) == cluster_size[i]

            
            self.means[i] = X_label.mean(0)
            Stds[i]       = X_label.std(0)

            
            X_label_stand = (X_label - self.means[[i]])

            if len(X_label) < self.num_features:
                c = np.tile(X_label_stand, (int(np.ceil(self.num_features/len(X_label))),1))

            else:
                c = X_label_stand.copy()

            # calculate PCA on X_label_stand -> get rot matrix and std
            attempt = 0
            successful_pca = False
            while not successful_pca:
                try:
                    pca = PCA(random_state = 0).fit(c)
                    successful_pca = True
                except:
                    e_fac = (0.5 * attempt - 6) ** 10
                    attempt += 1
                    c += np.eye(self.num_features) * e_fac 
                
                if not successful_pca:
                    print('PCA failed, was done again with different random start.')
                
            # Apply minimum standard deviation
            pca_std = np.sqrt(pca.explained_variance_)
            pca_std = min_std + pca_std * (pca_std.max() - min_std) / pca_std.max()
            
            self.T_mat[i]         = pca.components_.T / pca_std[np.newaxis]
            self.log_det_T_mat[i] = (np.log(np.abs(np.linalg.det(pca.components_.T))) -  
                                     np.log(pca_std).sum())
            
            # Apply transformation matrix
            X_label_pca = X_label_stand @ self.T_mat[i] # @ is matrix multiplication
            
            # Fit GMM distribution
            kde = KernelDensity(kernel = 'gaussian', bandwidth = 'silverman').fit(X_label_pca)
            self.KDEs[i] = kde
            
        # consider noise values
        if unique_labels[0] == -1:
            X_noise = X[self.cluster_labels == -1]
            # set noise std
            
            # set rot_mat_pca[0] to be an identity matrix
            if len(self.T_mat) > 1:
                stds = np.maximum(Stds[1:].mean(0), min_std)
                self.T_mat[0] = np.diag(1 / stds)
                self.log_det_T_mat[0] = -np.sum(np.log(stds))
            else:
                self.T_mat[0]         = np.eye(self.num_features)
                self.log_det_T_mat[0] = 0.0


            X_noise_stand = (X_noise - self.means[[0]]) @ self.T_mat[0] 
            
            # Fit GMM distribution
            # calculate silverman rule assuming only 1 sample 
            bandwidth = ((self.num_features + 2) / 4) ** ( -1 / (self.num_features + 4))
            try:
                kde_noise = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(X_noise_stand)
            except:
                print('Error in fitting noise KDE')
                print('X_noise_stand.shape', X_noise_stand.shape)
                print('X_noise_stand is finite:', np.isfinite(X_noise_stand).all())
                print('Num clusters', len(unique_labels) + 1)
                print('T_mat finite:', np.isfinite(self.T_mat[0]).all())
                if len(self.T_mat) > 1:
                    print('Std finite:', np.isfinite(Stds[1:]).all())
                    print('Std not zero:', (Stds[1:] > 0).all()) 
                print('bandwidth', bandwidth)
                assert False
            self.KDEs[0] = kde_noise
            
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
        
        log_probs = np.zeros((len(X), len(self.KDEs)), dtype = np.float64)
        
        for i, KDE in enumerate(self.KDEs):
            X_stand = (X - self.means[[i]]) @ self.T_mat[i]

            # get adjusted log prob values
            log_probs[:,i] = self.log_probs[i] + KDE.score_samples(X_stand) + self.log_det_T_mat[i] 
        
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
        
    
    def sample(self, num_samples = 1):
        assert self.fitted, 'The model was not fitted yet'
        
        labels = np.random.choice(np.arange(len(self.KDEs)), num_samples, p = self.probs)
        
        samples = []
        
        for label in np.unqiue(labels):
            num = (label == labels).sum()
            X_label_stand = self.KDEs[label].sample(num)
            
            X_label = X_label_stand @ np.linalg.inv(self.T_mat[label]) + self.means[[label]]
            
            samples.append(X_label)
            
        samples = np.concatenate(samples, axis = 0)
        
        np.random.shuffle(samples)

        return samples
            
            
        
        
        
        
        