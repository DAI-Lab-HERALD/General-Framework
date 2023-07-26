import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class ADE_joint(evaluation_template):
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
            # sum over dimension and mean over number agents
            diff = diff.sum(3).mean(1)
            diff = np.sqrt(diff)
            
            # mean over predicted samples and timesteps
            diff = diff.mean((0, 1))

            Error += diff

        E = Error / len(self.Output_path)
        return [E]
        
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'ADE (joint predictions)',
                 'file': 'ADE_joint',
                 'latex': r'\emph{ADE$_{joint}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False