import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

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
        _, _, Pred_steps = self.get_true_and_predicted_paths()
        
        # Get likelihood of having higher probability
        KDE_log_prob_true, KDE_log_prob_pred = self.get_KDE_probabilities(joint_agents = True)
        
        # Scale with number of timesteps
        Num_steps = Pred_steps.sum(-1).max(-1)
        NLL = np.log(Num_steps.max() / Num_steps) - KDE_log_prob_true[:,0,0] 
        
        Error = NLL.mean()
        return [Error]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[1]  
    
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
    
    def metric_boundaries(self = None):
        return [None, None]
