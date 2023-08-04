import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class KDE_NLL_joint(evaluation_template):
    r'''
    The value :math:`F` of the Negative Log Likelihood (assuming :math:`N_{agents}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = - {1 \over{N_{samples} N_{agents}}}
            \sum\limits_{i = 1}^{N_{samples}} \ln \left( P_{KDE,i} \left(\{\{\{x_{i,j} (t), y_{i,j} (t) \} \, | \; \forall\, t \in T_O\} \, | \; \forall \, j \} \right)\right)
            
    Here, :math:`P_{KDE,i}` is a sample and agent specific gaussian Kernel Density Estimate trained on all predictions (:math:`p \in P`)
    for sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`
    
    .. math::
        \{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_O\} \, | \; \forall \, j \}
    
    For each prediction timestep in :math:`T_O`, :math:`x` and :math:`y` are the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        nto = self.data_set.num_timesteps_out_real
        
        num_samples_needed = self.data_set.num_samples_path_pred
        num_samples = len(self.Output_path_pred.iloc[0,0])
        if num_samples >= num_samples_needed:
            idx_l = np.random.permutation(num_samples)[:num_samples_needed]#
        else:
            idx_l = np.random.randint(0, num_samples, num_samples_needed)
            
        NLL = 0
        
        for i_sample in range(len(self.Output_path_pred)):
            std = 1 + (np.array(self.Type.iloc[i_sample]) == 'V') * 79
            std = std[np.newaxis, :, np.newaxis, np.newaxis]
            
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 1)[idx_l,:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[np.newaxis,:,:nto]
            
            samples_pred_comp = (sample_pred / std).reshape(num_samples_needed, -1)
            samples_true_comp = (sample_true / std).reshape(1, -1)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(samples_pred_comp)

            log_prob = kde.score_samples(samples_true_comp)[0]
            
            # mean over agents
            num_agents = sample_pred.shape[1]
            log_prob = log_prob / num_agents
            
            NLL -= log_prob
        
        E = NLL / len(self.Output_path) 
        return [E]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KDE_NLL (joint prediction)',
                 'file': 'KDE_NLL_joint',
                 'latex': r'\emph{KDE$_{NLL, joint}$}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return False
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
