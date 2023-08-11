import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class FDE_indep(evaluation_template):
    r'''
    The value :math:`F` of the Final Displacement Error (assuming :math:`N_{agents}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples}|P| N_{agents}}} \sum\limits_{i = 1}^{N_{samples}}  
            \sum\limits_{p \in P} \sum\limits_{j = 1}^{N_{agents}} 
            \sqrt{\left( x_{i,j}(\max T_O) - x_{pred,i,p,j} (\max T_O) \right)^2 + \left( y_{i,j}(\max T_O) - y_{pred,i,p,j} (\max T_O) \right)^2}
        
    Here, :math:`P` are the set of predictions made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_O`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        nto = self.data_set.num_timesteps_out_real
        
        Error = 0
        
        idx_l = np.arange(self.data_set.num_samples_path_pred)
        for i_sample in range(len(self.Output_path_pred)):
            # sample_pred.shape = num_path x num_agents x num_timesteps_out x 2
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 1)[idx_l,:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[np.newaxis,:,:nto]
            
            diff = (sample_pred - sample_true) ** 2
            # sum over dimension
            diff = diff.sum(3)
            diff = np.sqrt(diff)
            
            # mean over predicted samples and agents
            diff = diff.mean((0,1))
            diff = diff[-1]
            
            Error += diff

        E = Error / len(self.Output_path)
        return [E]
        
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'FDE (independent predictions)',
                 'file': 'FDE_indep',
                 'latex': r'\emph{FDE$_{indep}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
