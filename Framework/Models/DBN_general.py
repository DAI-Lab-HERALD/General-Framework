import numpy as np
import pandas as pd
from DBN import DBN

class DBN_general(DBN):
    def get_data(self, train = True):
        if train:
            Input_array = self.Input_prediction_train.to_numpy()
        else:
            Input_array = self.Input_prediction_test.to_numpy()
        X = np.ones([Input_array.shape[0], Input_array.shape[1], self.timesteps]) * np.nan
        for i in range(len(X)):
            for j in range(Input_array.shape[1]):
                n_help_1 = max(0, self.timesteps - len(Input_array[i,j]))
                n_help_2 = max(0, len(Input_array[i,j]) - self.timesteps)
                X[i, j, n_help_1:] = Input_array[i,j][n_help_2:]
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
        return X

    
    def get_input_type(self = None):
        input_info = {'past': 'general',
                      'future': None}
        return input_info
    
    
    def get_name(self = None):
        names = {'print': 'Deep belief network (1D inputs)',
                 'file': 'Deep_BN_1D',
                 'latex': r'$\text{\emph{DBN}}_{1D}$'}
        return names   