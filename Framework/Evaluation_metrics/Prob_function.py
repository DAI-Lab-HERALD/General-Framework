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
            self.optics = OPTICS(min_samples = self.num_min_samples, 
                                          min_cluster_size = 5)
            self.optics.fit(X)
            self.cluster_labels = self.optics.labels_
            
            if len(np.unique(self.cluster_labels)) > 1:
                best_score = silhouette_score(X, self.cluster_labels)
            else:
                best_score = -1
                
            reachability = self.optics.reachability_[np.isfinite(self.optics.reachability_)] 
            
            self.Eps = np.linspace(reachability.min(), reachability.max(), 100)
            for i, eps in enumerate(self.Eps):
                test_labels = cluster_optics_dbscan(reachability   = self.optics.reachability_,
                                                    core_distances = self.optics.core_distances_,
                                                    ordering       = self.optics.ordering_,
                                                    eps            = eps)
            
                if len(np.unique(test_labels)) > 1:
                    test_score = silhouette_score(X, test_labels)
                    if test_score > best_score:
                        best_score = test_score
                        self.cluster_labels = test_labels
        else:
            self.cluster_labels = -1 * np.ones(len(X))
            
        unique_labels, cluster_size = np.unique(self.cluster_labels, return_counts = True)
            
        # Fit GMM to each model
        self.KDEs = [None] * len(unique_labels)
        self.means = np.zeros((len(unique_labels), self.num_features))
        self.stds  = np.zeros((len(unique_labels), self.num_features))

        # initialise rotation matrix for PCA
        self.rot_mat_pca = np.zeros((len(unique_labels), self.num_features, self.num_features))
        # initialise STDs along the principle components
        self.stds_pca  = np.zeros((len(unique_labels), self.num_features))

        # check that self.num_features <  len(X_labels) else rot matrix is identity; and self.stds_pca is ones        
        if self.num_features < len(X):

            for i, label in enumerate(unique_labels):
                if label == -1:
                    continue
                # Get cluster data
                X_label = X[self.cluster_labels == label]
                assert len(X_label) == cluster_size[i]
                
                self.means[i] = X_label.mean(0)
                self.stds[i]  = X_label.std(0) + 0.001 * X_label.std(0).max() + 1e-6

                X_label_stand = (X_label - self.means[[i]]) / self.stds[[i]]

                # calculate PCA on X_label_stand -> get rot matrix and std
                pca = PCA()
                pca.fit(X_label_stand)
                self.rot_mat_pca[i] = pca.components_
                pca_std = np.sqrt(pca.explained_variance_)
                self.stds_pca[i] = pca_std + 0.01 * pca_std.max() + 1e-6

                # Apply rot_matrix on X_label_stand to get X_label_pca
                X_label_pca = X_label_stand @ self.rot_mat_pca[i].T # @ is matrix multiplication

                # Normalise along principle axes to get X_label_pca_stand
                X_label_pca_stand = X_label_pca / self.stds_pca[[i]]
                
                # Fit GMM distribution
                kde = KernelDensity(kernel = 'gaussian', bandwidth = 'silverman').fit(X_label_pca_stand) # TODO change to X_label_pca_stand
                self.KDEs[i] = kde
        else:
            self.rot_mat_pca = np.tile(np.eye(self.num_features), (len(unique_labels), 1, 1))
            self.stds_pca = np.ones((len(unique_labels), self.num_features))
            
        # consider noise values
        if unique_labels[0] == -1:
            X_noise = X[self.cluster_labels == -1]
            # set noise std
            self.stds[0] = self.stds[1:].min(0)
            
            # set rot_mat_pca[0] to be an identity matrix
            self.rot_mat_pca[0] = np.eye(self.num_features)
            # set std_pca[0] to be 1 
            self.stds_pca[0] = np.ones(self.num_features)


            X_noise_stand = (X_noise - self.means[[0]]) / self.stds[[0]]
            
            # Fit GMM distribution
            # calculate silverman rule assuming only 1 sample 
            bandwidth = ((self.num_features + 2) / 4) ** ( -1 / (self.num_features + 4))
            kde_noise = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(X_noise_stand)
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
            X_stand = (X - self.means[[i]]) / self.stds[[i]]

            # calculate PCA on X_stand -> get rot matrix and std 
            # NOTE: it should not be needed to calculate PCA on X_stand; 
            # it was already calculated during the fitting process (?)
            

            # Apply rot_matrix on X_label_stand to get X_pca
            X_pca = X_stand @ self.rot_mat_pca[i].T

            # Normalise along principle axes to get X_label_pca_stand
            X_pca_stand = X_pca / self.stds_pca[[i]]

            # calculate std_adjusts
            std_adjust     = np.log(self.stds[i]).sum()
            std_pca_adjust = np.log(self.stds_pca[i]).sum()

            # calculate rot_mat_adjust = -log(det(rot_matrix))
            rot_mat_adjust = -np.log(np.abs(np.linalg.det(self.rot_mat_pca[i])))

            # TODO subtract the additional adjustment values
            log_probs[:,i] = self.log_probs[i] \
                            + KDE.score_samples(X_pca_stand) \
                            - std_adjust - std_pca_adjust - rot_mat_adjust # TODO: change X_stand to X_pca_stand
        
        # Deal with overflow
        prob = np.exp(log_probs).sum(1)
        if return_log:
            return prob, log_probs
        else:
            return prob
    
    
    def score_samples(self, X):
        probs, log_probs = self.prob(X, return_log = True)
        l_probs = sp.special.logsumexp(log_probs, axis = -1)
        return l_probs
        
    
    def sample(self, num_samples = 1):
        assert self.fitted, 'The model was not fitted yet'
        
        labels = np.random.choice(np.arange(len(self.KDEs)), num_samples, p = self.probs)
        
        samples = []
        
        for label in np.unqiue(labels):
            num = (label == labels).sum()
            X_label_pca_stand = self.KDEs[label].sample(num)

            # invert the normalisation with std_pca
            X_label_pca = X_label_pca_stand * self.stds_pca[[label]]

            # multiply with inverse of rot_mat
            X_label_stand = X_label_pca @ self.rot_mat_pca[[label]]
            
            assert len(X_label_stand.shape) == 2
            assert X_label_stand.shape[1] == self.num_features
            
            X_label = X_label_stand * self.stds[[label]] + self.means[[label]]
            
            
            samples.append(X_label)
            
        samples = np.concatenate(samples, axis = 0)
        
        np.random.shuffle(samples)

        return samples
            
            
        
        
        
        
        