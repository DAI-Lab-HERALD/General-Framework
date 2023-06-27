import numpy as np
import pandas as pd
import os
from evaluation_template import evaluation_template 
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

class ROC_curve(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self, plot = False):
        '''
        It has to be noted here that it would be ideal if one could calcualte the VUS instead of AUC
        However, we instead used this expansion here instead.
        '''
        A_pred = self.Output_A_pred.to_numpy()
        AA = self.Output_A.to_numpy()
        
        num_c = AA.shape[1]
        
        # Necessary adjustment to allow auc = 1 as a result, this might have to be reconsidered
        A_pred_adj = np.copy(A_pred)
        for c in range(num_c):
            A_pred_adj[:,c] = A_pred[:,c] / (A_pred[:,c] + A_pred[:, np.arange(num_c) != c].max(axis = 1))
        
        # rank each sample by likelyhood of this class being chosen
        R = np.zeros(AA.shape, float)
        C_ind = np.tile(np.arange(num_c)[np.newaxis], (len(AA),1))
        R[np.argsort(A_pred_adj, axis = 0), C_ind] = np.arange(1, len(AA) + 1)[:,np.newaxis]
        
        N = AA.sum(axis = 0).astype(float)
        Ns = 0.5 * N * (N + 1)
        
        L = (R * AA).sum(axis = 0) - Ns
        
        # No idea what this here is
        auc1 = L.sum() / (N.sum() ** 2 - (N ** 2).sum())
        # this is the weighted mean of 1 vs rest AUC
        auc2 = (L / (N.sum() - N)).sum() / N.sum()
        
        # method 2
        assert auc1 != np.nan
        assert auc2 != np.nan
        return [auc1, auc2]
    
    def main_result_idx(self = None):
        return 1
    
    def create_plot(self, results, test_file):
        pass
        
    
    def get_output_type(self = None):
        return 'class' 
    
    def get_opt_goal(self = None):
        return 'maximize'
    
    def get_name(self = None):
        names = {'print': 'AUC (ROC curve - expanded)',
                 'file': 'AUC',
                 'latex': r'\emph{AUC}'}
        return names
    
    def requires_preprocessing(self):
        return False
    
    def is_log_scale(self = None):
        return False
    
    def allows_plot(self):
        return False
    
    def check_applicability(self):
        if not self.data_set.classification_useful:
            return 'because a classification metric requires more than one available class.'
        return None