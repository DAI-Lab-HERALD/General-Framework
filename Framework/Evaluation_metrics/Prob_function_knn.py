import numpy as np
from sklearn.neighbors import BallTree
import scipy as sp


class KNN_PDF():
    '''
    This is a method for creating a point density invariant probability density
    function using nonparametrics methods.
    
    Spcecifically, it uses the k-nearest neighbour method to estimate the 
    probability density.
    '''
    def __init__(self, smooting = 0.0, num_neighbours = 5):
        self.fitted = False
        
        # Get design opportunities
        self.smooting = smooting
        self.num_neighbours = num_neighbours
            
        
    def fit(self, X):
        assert len(X.shape) == 2
        self.num_samples, self.num_features = X.shape
        
        self.BallTree = BallTree(X)

        # Get volume factor of hypersphere
        self.volume_ratio = np.pi**(self.num_features / 2) / sp.special.gamma(self.num_features / 2 + 1)
        
        self.fitted = True

        return self
        
        
    def prob(self, X, return_log = False):
        assert self.fitted, 'The model was not fitted yet'
        
        assert len(X.shape) == 2
        assert X.shape[1] == self.num_features
        
        # Get distances to nearest neighbours
        dist, _ = self.BallTree.query(X, self.num_neighbours)

        # Get radius
        radius = dist.max(axis = -1)

           # Deal with overflow
        if return_log:
            # Calculate log volume
            log_volume = np.log(radius) * self.num_features + np.log(self.volume_ratio)

            # Calculate log prob values
            log_prob = np.log(self.num_neighbours) - np.log(self.num_samples) - log_volume
            return log_prob
        else:
            # Get volume of hypersphere
            volume = radius ** self.num_features * self.volume_ratio

            # calculate logarithmic probability
            prob = self.num_neighbours / (self.num_samples * volume)
            return prob 
    
    
    def score_samples(self, X):
        return self.prob(X, return_log = True)
        
    
    def sample(self, num_samples = 1, random_state = 0):
        raise AttributeError("This technique cannot generate samples")

        return samples
            
            
        
        
        
        
        