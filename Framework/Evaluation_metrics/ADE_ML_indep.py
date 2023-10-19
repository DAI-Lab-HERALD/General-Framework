import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

from Prob_function import OPTICS_GMM

class ADE_ML_indep(evaluation_template):
    r'''
    The value :math:`F` of the most likely Average Displacement Error (assuming :math:`N_{agents,i}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{\sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}}  
            \sum\limits_{i = 1}^{N_{samples}}  \sum\limits_{j = 1}^{N_{agents,i}} {1\over{| T_{O,i} | }} \sum\limits_{t \in T_{O,i}}
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p^*_{i,j},j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p^*_{i,j},j} (t) \right)^2}
            
    Here, for each specific sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`
    
    .. math::
            p^*_{i,j} = \underset{p \in P}{\text{arg} \min} P_{KDE,i,j} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_O\} \right) , 
    
    where :math:`P_{KDE,i,j}`, a sample and agent specific gaussian Kernel Density Estimate trained on all predictions :math:`p \in P`, returns the
    likelihood for trajectories predicted at timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths()
        Pred_agents = Pred_steps.any(-1)
        
        _, KDE_log_prob_pred = self.get_KDE_probabilities(joint_agents = False)
        
        num_samples, num_agents = Path_pred.shape[0], Path_pred.shape[2]
        
        # Get indices for most likely predictions
        sample_index = np.tile(np.arange(num_samples)[:,np.newaxis], (1, num_agents))
        ml_pred_index = np.argmax(KDE_log_prob_pred, axis = 1)
        agent_index  = np.tile(np.arange(num_agents)[np.newaxis], (num_samples, 1))
        
        Path_pred_ml = Path_pred[sample_index, ml_pred_index, agent_index]
        
        # Combine agent and sample dimension
        Path_true    = Path_true[:,0][Pred_agents]
        Path_pred_ml = Path_pred_ml[Pred_agents]
        Pred_steps   = Pred_steps[Pred_agents]
        
        # Get num steps
        Num_steps = Pred_steps.sum(-1)
            
        # Get squared distance    
        Diff = ((Path_true - Path_pred_ml) ** 2).sum(-1)
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Mean over timesteps
        Diff = Diff.sum(-1) / Num_steps
        
        # Mean over predicted agents
        Error = Diff.mean()
        return [Error]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'ADE_ML (independent prediction)',
                 'file': 'ADE_ML_indep',
                 'latex': r'\emph{ADE$_{ML, indep}$ [m]}'}
        return names
    
    def is_log_scale(self = None):
        return True
    
    def check_applicability(self):
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]
