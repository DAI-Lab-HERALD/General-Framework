import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

from Prob_function import OPTICS_GMM

class KDE_NLL_joint(evaluation_template):
    r'''
    The value :math:`F` of the Negative Log Likelihood (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = - {1 \over{\sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}}
            \sum\limits_{i = 1}^{N_{samples}} \ln \left( P_{KDE,i} \left(\{\{\{x_{i,j} (t), y_{i,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \, | \; \forall \, j \} \right)\right)
            
    Here, :math:`P_{KDE,i}` is a sample and agent specific gaussian Kernel Density Estimate trained on all predictions (:math:`p \in P`)
    for sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`
    
    .. math::
        \{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \, | \; \forall \, j \}
    
    For each prediction timestep in :math:`T_{O,i}`, :math:`x` and :math:`y` are the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Types = self.get_true_and_predicted_paths(return_types = True)
        Pred_agents = Pred_steps.any(-1) 
        Num_agents = Pred_agents.sum(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        
        NLL = 0
        
        for i_sample in range(len(Path_true)):
            pred_agents = Pred_agents[i_sample]
            std = 1 + (Types[i_sample, pred_agents] != 'P') * 79
            std = std[np.newaxis, :, np.newaxis, np.newaxis]
            
            nto = Num_steps[i_sample]
            n_agents = Num_agents[i_sample]
            
            path_true = Path_true[i_sample][:,pred_agents,:nto]
            path_pred = Path_pred[i_sample][:,pred_agents,:nto]
            
            path_true_comp = (path_true / std).reshape(-1, n_agents * nto * 2)
            path_pred_comp = (path_pred / std).reshape(-1, n_agents * nto * 2)
            
            # kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(path_pred_comp)
            kde = OPTICS_GMM().fit(path_pred_comp)
                
            log_prob_true = kde.score_samples(path_true_comp)
            
            NLL += np.log(Num_steps.max() / nto) - log_prob_true[0]
        
        Error = NLL / Pred_agents.sum()
        return [Error]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KDE_NLL (joint prediction)',
                 'file': 'KDE_NLL_joint',
                 'latex': r'\emph{KDE$_{NLL, joint}$}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return False
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
