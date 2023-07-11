import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class KDE_NLL(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        
        NLL = 0
        nto = self.data_set.num_timesteps_out_real
        
        for i_sample in range(len(self.Output_path_pred)):
            std = 1 + (np.array(self.Type.iloc[i_sample]) == 'V') * 79
            std = std[np.newaxis, np.newaxis, np.newaxis]
            
            samples = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = -1)[:,:nto]
            samples_comp = (samples / std).reshape(self.data_set.num_samples_path_pred, -1)
            
            sample_gt = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = -1)[np.newaxis,:nto]
            sample_gt = (sample_gt / std).reshape(1,-1)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(samples_comp)

            log_prob = kde.score_samples(sample_gt)[0]
            
            NLL += log_prob*(-1)
        
        return [NLL / len(self.Output_path_pred)]
    
    def main_result_idx(self = None):
        return 0
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KDE_NLL',
                 'file': 'KDE_NLL',
                 'latex': r'\emph{KDE$_{NLL}$}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return False
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False