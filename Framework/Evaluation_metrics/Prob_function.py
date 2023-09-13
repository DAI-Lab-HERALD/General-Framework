import numpy as np
from sklearn.cluster import OPTICS
from sklearn.neighbors import KernelDensity

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
        
        num_min_samples = 10
        
        if len(X) > num_min_samples:
            # Get clusters using the OPTICS
            optics = OPTICS(min_samples = num_min_samples, min_cluster_size = num_min_samples)
            optics.fit(X)
            cluster_labels = optics.labels_
        
        else:
            cluster_labels = np.zeros(len(X))
            
        unique_labels, cluster_size = np.unique(cluster_labels, return_counts = True)
        # Check for noise points
        if unique_labels.min() == -1:
            assert unique_labels[0] == -1
            unique_labels = unique_labels[1:]
            cluster_size  = cluster_size[1:]
            
        # Fit GMM to each model
        self.KDEs = []
        self.means = np.zeros((len(unique_labels), self.num_features))
        self.stds  = np.zeros((len(unique_labels), self.num_features))
        for i, label in enumerate(unique_labels):
            # Get cluster data
            X_label = X[cluster_labels == label]
            assert len(X_label) == cluster_size[i]
            
            self.means[i] = X_label.mean(0)
            self.stds[i]  = X_label.std(0) + 0.001 * X_label.std(0).max() + 1e-6
            
            X_label_stand = (X_label - self.means[[i]]) / self.stds[[i]]
            
            # Fit GMM distribution
            kde = KernelDensity(kernel = 'gaussian', bandwidth = 'silverman').fit(X_label_stand)
            self.KDEs.append(kde)
            
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
        
        l_probs = - np.ones(probs.shape, dtype = np.float32) * np.inf
        
        # get useful probabilities
        useful = (probs > 0) & (probs < np.inf) 
        l_probs[useful] = np.log(probs[useful])
        
        # Get edge cases
        l_probs = np.maximum(l_probs, log_probs.max(1))
        
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
            
            
        
        
        
        
        