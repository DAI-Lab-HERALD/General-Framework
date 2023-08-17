import numpy as np
import pandas as pd
from logit_theofilatos import logit_theofilatos
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos_general(logit_theofilatos):
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
        
        X = X - self.mean
        X[np.isnan(X)] = 0
        return X, P, class_names

    
    def get_input_type(self = None):
        input_info = {'past': 'general',
                      'future': None}
        return input_info
    
    
    def get_name(self = None):
        names = {'print': 'Logistic regression (1D inputs)',
                 'file': 'log_reg_1D',
                 'latex': r'$\text{\emph{LR}}_{1D}$'}
        return names   