import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class minADE20_indep(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        '''
        Calculates the minADE20 for the predicted paths.
        '''
        num_samples_needed = 20
        num_samples = len(self.Output_path_pred.iloc[0,0])
        if num_samples >= num_samples_needed:
            idx_l = np.random.permutation(num_samples)[:num_samples_needed]#
        else:
            idx_l = np.random.randint(0, num_samples, num_samples_needed)
            
        nto = self.data_set.num_timesteps_out_real
        
        Error = 0
        
        for i_sample in range(len(self.Output_path_pred)):
            # sample_pred.shape = num_path x num_agents x num_timesteps_out x 2
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 1)[idx_l,:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[np.newaxis,:,:nto]
            
            diff = (sample_pred - sample_true) ** 2
            # sum over dimension
            diff = diff.sum(3)
            diff = np.sqrt(diff)
            
            # mean over timesteps
            diff = diff.mean(2)
        
            # take best sample for each agent
            idx = np.argsort(diff, 0)
            diff = diff[idx[0], np.arange(diff.shape[1])]
            diff = diff.mean()
            
            Error += diff
        
        E = Error / len(self.Output_path)
        return [E]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'min ADE (20 samples, independent prediction)',
                 'file': 'minADE20_indep',
                 'latex': r'\emph{min ADE$_{20, indep}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
