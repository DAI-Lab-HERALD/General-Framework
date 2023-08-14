import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class KDE_NLL_indep(evaluation_template):
    r'''
    The value :math:`F` of the Negative Log Likelihood (assuming :math:`N_{agents,i}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = - {1 \over{\sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}}  \sum\limits_{i = 1}^{N_{samples}}  \sum\limits_{j = 1}^{N_{agents,i}}
        \ln \left( P_{KDE,i,j} \left(\{\{x_{i,j} (t), y_{i,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \right)\right)
            
    Here, :math:`P_{KDE,i,j}` is a sample and agent specific gaussian Kernel Density Estimate trained on all predictions (:math:`p \in P`)
    for sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`
    
    .. math::
        \{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\}
    
    For each prediction timestep in :math:`T_{O,i}`, :math:`x` and :math:`y` are the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Types = self.get_true_and_predicted_paths(return_types = True)
        Pred_agents = Pred_steps.any(-1)
        
        # Combine agent and sample dimension
        Path_true  = Path_true.transpose(0,2,1,3,4)[Pred_agents]
        Path_pred  = Path_pred.transpose(0,2,1,3,4)[Pred_agents]
        Pred_steps = Pred_steps[Pred_agents]
        Types      = Types[Pred_agents]
        
        Num_steps = Pred_steps.sum(-1)
            
        NLL = 0
        
        for i_case in range(len(Path_true)):
            std = 1 + (Types[i_case] != 'P') * 79
            
            nto = Num_steps[i_case]
            
            path_true = Path_true[i_case,:,:nto]
            path_pred = Path_pred[i_case,:,:nto]
            
            path_true_comp = (path_true / std).reshape(-1, nto * 2)
            path_pred_comp = (path_pred / std).reshape(-1, nto * 2)
                    
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(path_pred_comp)
            
            log_prob_true = kde.score_samples(path_true_comp)
            
            NLL += np.log(Num_steps.max() / nto) - log_prob_true[0]
        
        Error = NLL / Pred_agents.sum()
        return [Error]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KDE_NLL (independent prediction)',
                 'file': 'KDE_NLL_indep',
                 'latex': r'\emph{KDE$_{NLL, indep}$}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return False
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
