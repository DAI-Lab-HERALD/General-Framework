import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class ADE(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        
        
        nto = self.data_set.num_timesteps_out_real
        
        Error = np.zeros(nto)
        Samples = np.zeros(nto, int)
        
        for i_sample in range(len(self.Output_path_pred)):
            # sample_pred.shape = num_path x num_timesteps_out x 2 x num_agents
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = -1)[:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = -1)[np.newaxis,:nto]
            
            diff = (sample_pred - sample_true) ** 2
            # sum over dimension and number agents
            diff = diff.sum((2,3))
            diff = np.sqrt(diff)
            # mean over predicted samples
            diff = diff.mean(0)

            Error[:len(diff)] += diff
            Samples[:len(diff)] += 1

        E = Error / Samples 
        return [E.mean()]
        
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'ADE',
                 'file': 'ADE',
                 'latex': r'\emph{ADE [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
