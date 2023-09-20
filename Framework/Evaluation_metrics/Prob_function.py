import numpy as np
from sklearn.cluster import OPTICS
from sklearn.neighbors import KernelDensity
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
    def __init__(self):
        self.fitted = False
        
    def fit(self, X):
        assert len(X.shape) == 2
        
        self.num_features = X.shape[1]
        
        num_min_samples = X.shape[0] / 20
        self.num_min_samples = int(np.clip(num_min_samples, min(5, X.shape[0]), 20))
        
        if len(X) > self.num_min_samples:
            # Get clusters using the OPTICS
            optics = OPTICS(min_samples = self.num_min_samples, 
                            min_cluster_size = 5)
            optics.fit(X)
            self.cluster_labels = optics.labels_
        
        else:
            self.cluster_labels = np.zeros(len(X))
            
        unique_labels, cluster_size = np.unique(self.cluster_labels, return_counts = True)
        
        assert False
            
        # Fit GMM to each model
        self.KDEs = [None] * len(unique_labels)
        self.means = np.zeros((len(unique_labels), self.num_features))
        self.stds  = np.zeros((len(unique_labels), self.num_features))
        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            # Get cluster data
            X_label = X[self.cluster_labels == label]
            assert len(X_label) == cluster_size[i]
            
            self.means[i] = X_label.mean(0)
            self.stds[i]  = X_label.std(0) + 0.001 * X_label.std(0).max() + 1e-6
            
            X_label_stand = (X_label - self.means[[i]]) / self.stds[[i]]
            
            # Fit GMM distribution
            kde = KernelDensity(kernel = 'gaussian', bandwidth = 'silverman').fit(X_label_stand)
            self.KDEs[i] = kde
            
        # consider noise values
        if unique_labels[0] == -1:
            X_noise = X[self.cluster_labels == -1]
            # set noise std
            self.stds[0] = self.stds[1:].min(0)
            
            
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
        
        for i, GMM in enumerate(self.KDEs):
            X_stand = (X - self.means[[i]]) / self.stds[[i]]
            log_probs[:,i] = self.log_probs[i] + GMM.score_samples(X_stand) - np.log(self.stds[i]).sum()
        
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
            X_label_stand = self.KDEs[label].sample(num)
            
            assert len(X_label_stand.shape) == 2
            assert X_label_stand.shape[1] == self.num_features
            
            X_label = X_label_stand * self.stds[[label]] + self.means[[label]]
            
            
            samples.append(X_label)
            
        samples = np.concatenate(samples, axis = 0)
        
        np.random.shuffle(samples)

        return samples
            
            
        
        
        
        
        