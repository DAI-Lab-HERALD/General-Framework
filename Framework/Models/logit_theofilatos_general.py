import numpy as np
import pandas as pd
from logit_theofilatos import logit_theofilatos
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos_general(logit_theofilatos):
    '''
    The logistic regression model is a standard classification model used in many 
    settings.
    
    This instance takes as input extracted, scenario dependent distances, and its 
    training is based on the following citation:
        
    Theofilatos, A., Ziakopoulos, A., Oviedo-Trespalacios, O., & Timmis, A. (2021). 
    To cross or not to cross? Review and meta-analysis of pedestrian gap acceptance 
    decisions at midblock street crossings. Journal of Transport & Health, 22, 101108.
    '''
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
    
    
    def get_name(self = None):
        names = {'print': 'Logistic regression (1D inputs)',
                 'file': 'log_reg_1D',
                 'latex': r'$\text{\emph{LR}}_{1D}$'}
        return names   
    
    def check_trainability_method(self):
        if not self.general_input_available:
            return " there is no generalized input data available."
        return None