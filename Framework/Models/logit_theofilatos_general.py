import numpy as np
import pandas as pd
from logit_theofilatos import logit_theofilatos
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos_general(logit_theofilatos):
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
        
        X = X - self.mean
        X[np.isnan(X)] = 0
        return X

    def check_input_names_method(self, names, train = True):
        return True


    def check_trainability_method(self):
        return None
        
    
    def get_input_type(self = None):
        input_info = {'past': 'general',
                      'future': None}
        return input_info
    
    
    def get_name(self = None):
        names = {'print': 'Logistic regression (1D inputs)',
                 'file': 'log_reg_1D',
                 'latex': r'$\text{\emph{LR}}_{1D}$'}
        return names   
    
    def requires_torch_gpu(self = None):
        return False

        
        
        
        
        
    
        
        
        