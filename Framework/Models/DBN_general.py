import numpy as np
import pandas as pd
from DBN import DBN

class DBN_general(DBN):
    def get_data(self, train = True):
        if train:
            _, _, _, X, _, class_names, P, _ = self.get_classification_data(train)
        else:
            _, _, _, X, _, class_names = self.get_classification_data(train)
            P = None
        X = X.reshape(X.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        if not hasattr(self, 'mean'):
            assert train, "This should not be possible, loading failed"
            self.mean = np.nanmean(X, axis = 0, keepdims = True)
            self.xmax = np.nanmax(X, axis = 0, keepdims = True)
            self.xmin = np.nanmin(X, axis = 0, keepdims = True)
        
        X = X - self.mean
        X[np.isnan(X)] = 0
        X = X + self.mean
        
        X = (X - self.xmin) / (self.xmax - self.xmin + 1e-5)
        return X, P, class_names

    
    def get_input_type(self = None):
        input_info = {'past': 'general',
                      'future': None}
        return input_info
    
    
    def get_name(self = None):
        names = {'print': 'Deep belief network (1D inputs)',
                 'file': 'Deep_BN_1D',
                 'latex': r'$\text{\emph{DBN}}_{1D}$'}
        return names   