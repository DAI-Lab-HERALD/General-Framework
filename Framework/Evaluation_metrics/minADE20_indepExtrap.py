import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class minADE20_indepExtrap(evaluation_template):
    r'''
    The value :math:`F` of the minimum Average Displacement Error (assuming :math:`N_{agents,i}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{\sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \sum\limits_{i = 1}^{N_{samples}} \sum\limits_{j = 1}^{N_{agents,i}}  
            \underset{p \in P_{20}}{\min} \left( {1\over{| T_{O,i} |}} \sum\limits_{t \in T_{O,i}} 
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2} \right)
        
    Here, :math:`P_{20} \subset P` are 20 randomly selected instances of the set of predictions :math:`P` made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths(20)

        Path_true = Path_true[..., self.model.num_timesteps_out:, :]
        Path_pred = Path_pred[..., self.model.num_timesteps_out:, :]
        Pred_steps = Pred_steps[..., self.model.num_timesteps_out:]

        useful = Pred_steps[:, 0, :].any(-1)
        Path_true = Path_true[useful]
        Path_pred = Path_pred[useful]
        Pred_steps = Pred_steps[useful]

        Pred_agents = Pred_steps.any(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        Num_agents = Pred_agents.sum(-1)
        
        # Get squared distance
        Diff = ((Path_true - Path_pred) ** 2).sum(-1)
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Mean over timesteps
        Diff = Diff.sum(-1) / Num_steps[:,np.newaxis,np.newaxis]
        
        # Mean over predictions
        Diff = Diff.min(1)
        
        # Mean over samples and agents
        Error = Diff.sum() / Num_agents.sum()
        
        return [Error]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'min ADE (20 samples, independent prediction extrap.)',
                 'file': 'minADE20_indepExtrap',
                 'latex': r'\emph{min ADE$_{20, indepExtrap}$ [m]}'}
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
