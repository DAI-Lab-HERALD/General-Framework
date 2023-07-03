import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class FDE_ML(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):

        Error = 0
        
        nto = self.data_set.num_timesteps_out_real

        for i_sample in range(len(self.Output_path_pred)):
            std = np.array([1 if name[0] == 'P' else 80 for name in self.Output_path_pred.columns])
            std = std[np.newaxis, np.newaxis, np.newaxis]
            
            samples = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = -1)[:,:nto]
            samples_comp = (samples / std).reshape(self.data_set.num_samples_path_pred, -1)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(samples_comp)

            log_prob = kde.score_samples(samples_comp)

            i_ml = np.argmax(log_prob)

            sample_ml = samples[i_ml]            

            sample_gt = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = -1)[:nto]

            diff = np.sqrt(np.sum((sample_ml - sample_gt) ** 2, axis = (1, 2)))
            
            
            Error += diff[-1]
        
        return [Error / len(self.Output_path_pred)]
    
    def main_result_idx(self = None):
        return 0
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'FDE_ML',
                 'file': 'FDE_ML',
                 'latex': r'\emph{FDE$_{ML}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False