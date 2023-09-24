import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

from Prob_function import OPTICS_GMM

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
        _, _, Pred_steps = self.get_true_and_predicted_paths()
        Pred_agents = Pred_steps.any(-1)
        
        # Get likelihood of having higher probability
        KDE_log_prob_true, _ = self.get_KDE_probabilities(joint_agents = False)
        
        # Combine agent and sample dimension
        Pred_steps = Pred_steps[Pred_agents]
        PLL        = KDE_log_prob_true[:,0][Pred_agents]
        
        # Scale with number of timesteps
        Num_steps = Pred_steps.sum(-1)
        NLL = np.log(Num_steps.max() / Num_steps) - PLL 
        
        Error = NLL.mean()
        return [Error, PLL, KDE_log_prob_pred.max(1)[Pred_agents], KDE_log_prob_pred.min(1)[Pred_agents]]
    
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
