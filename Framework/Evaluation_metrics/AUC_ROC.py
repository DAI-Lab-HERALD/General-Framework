import numpy as np
import pandas as pd
import os
from evaluation_template import evaluation_template 
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

class AUC_ROC(evaluation_template):
    r'''
    The value :math:`F` of the Area Under Curve of the Receiver Operator Curve is calculated in the following way:
        
    .. math::
        F =  {1 \over{N_{samples}}} \sum\limits_{k} {L_k \over {N_{samples} - N_k }} 
        
    with 
    
    .. math::
        & N_k & = \sum\limits_{i = 1}^{N_{samples}} p_{i,k} \\
        & L_k & = \left(\sum\limits_{i = 1}^{N_{samples}} r_{i,k} p_{i,k} \right) - {1\over{2}} N_k (N_k + 1) 
               
    Here, for a specific sample :math:`i \in \{1, ..., N_{samples}\}` and 
    classifcation :math:`k` out of :math:`N_{classes}` possible, :math:`p` is the actually observed and :math:`p_{pred}` 
    the predicted probability for a classification to be observed. The sum of all these values over :math:`k\}` should always 
    be equal to :math:`1`.
    
    Meanwhile, the :math:`r_{i,k} \in \{1, ..., N_{samples}\}` is the likelihood rank of each sample, where
        
    .. math::
        r_{i_1,k} > r_{i_2,k} \Rightarrow 
        {p_{pred,i_1,k} \over {p_{pred,i_1,k} + \underset{\widehat{k} \neq k}{\max} p_{pred,i_1,\widehat{k}}}} \geq
        {p_{pred,i_2,k} \over {p_{pred,i_2,k} + \underset{\widehat{k} \neq k}{\max} p_{pred,i_2,\widehat{k}}}}\, .
        
    It has to be noted that the AUC is normally defined using an integral, but the analytical solution above is much more efficient.
        
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self, plot = False):
        '''
        It has to be noted here that it would be ideal if one could calcualte the VUS instead of AUC
        However, we instead used this expansion here instead.
        '''
        P_true, P_pred, _ = self.get_true_and_predicted_class_probabilities()
        
        num_c = P_true.shape[1]
        
        # Necessary adjustment to allow auc = 1 as a result, this might have to be reconsidered
        P_pred_adj = np.copy(P_pred)
        for c in range(num_c):
            P_pred_adj[:,c] = P_pred[:,c] / (P_pred[:,c] + P_pred[:, np.arange(num_c) != c].max(axis = 1))
        
        # rank each sample by likelyhood of this class being chosen
        R = np.zeros(P_true.shape, float)
        C_ind = np.tile(np.arange(num_c)[np.newaxis], (len(P_true),1))
        R[np.argsort(P_pred_adj, axis = 0), C_ind] = np.arange(1, len(P_true) + 1)[:,np.newaxis]
        
        N = P_true.sum(axis = 0).astype(float)
        Ns = 0.5 * N * (N + 1)
        
        L = (R * P_true).sum(axis = 0) - Ns
        
        # this is the weighted mean of 1 vs rest AUC
        auc = (L / (N.sum() - N)).sum() / N.sum() 
        assert auc
        
        return [auc]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[0]
    
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
        if not self.data_set.classification_possible:
            return 'because a classification metric requires more than one available class.'
        return None
    
    def metric_boundaries(self = None):
        return [0.0, 1.0]
