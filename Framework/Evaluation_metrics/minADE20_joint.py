import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class minADE20_joint(evaluation_template):
    r'''
    The value :math:`F` of the minimum Average Displacement Error (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples} }} \sum\limits_{i = 1}^{N_{samples}}  \underset{p \in P_{20}}{\min} 
            \left( {1\over{| T_{O,i} |}} \sum\limits_{t \in T_{O,i}}\sqrt{{1\over{N_{agents,i}}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2} \right)
        
    Here, :math:`P_{20} \subset P` are 20 randomly selected instances of the set of predictions :math:`P` made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths(20)
        Pred_agents = Pred_steps.any(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        Num_agents = Pred_agents.sum(-1)
        
        # Get squared distance
        Diff = ((Path_true - Path_pred) ** 2).sum(-1)
        
        # Get mean over agents
        Diff = Diff.sum(2) / Num_agents[:,np.newaxis,np.newaxis]
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Get mean over timesteps
        Diff = Diff.sum(-1) / Num_steps[:,np.newaxis]
        
        # Get min over predictions        
        Diff = Diff.min(1)
        
        # Get mean over samples
        Error = Diff.mean()
        
        return [Error]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'min ADE (20 samples, joint prediction)',
                 'file': 'minADE20_joint',
                 'latex': r'\emph{min ADE$_{20, joint}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]
