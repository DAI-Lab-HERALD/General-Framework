import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class ADE_ML_indep(evaluation_template):
    r'''
    The value :math:`F` of the most likely Average Displacement Error (assuming :math:`N_{agents}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples} N_{agents} | T_O |}}  \sum\limits_{j = 1}^{N_{agents}} 
            \sum\limits_{i = 1}^{N_{samples}} \sum\limits_{t \in T_O}
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p^*_{i,j},j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p^*_{i,j},j} (t) \right)^2}
            
    Here, for each specific sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`
    
    .. math::
            p^*_{i,j} = \underset{p \in P}{\text{arg} \min} P_{KDE,i,j} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_O\} \right) , 
    
    where :math:`P_{KDE,i,j}`, a sample and agent specific gaussian Kernel Density Estimate trained on all predictions :math:`p \in P`, returns the
    likelihood for trajectories predicted at timesteps :math:`T_O`. :math:`x` and :math:`y` are here the actual observed positions, while 
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
        Error = 0

        for i_sample in range(len(self.Output_path_pred)):
            std = 1 + (np.array(self.Type.iloc[i_sample]) == 'V') * 79
            std = std[np.newaxis, :, np.newaxis, np.newaxis]
            
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 1)[idx_l,:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[np.newaxis,:,:nto]
            
            num_agents = sample_pred.shape[1]
            for i_agent in range(num_agents):
                samples_pred_comp = (sample_pred[:,i_agent] / std[:,i_agent]).reshape(num_samples_needed, -1)
                
                kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(samples_pred_comp)
    
                log_prob = kde.score_samples(samples_pred_comp)
    
                i_ml = np.argmax(log_prob)
    
                sample_pred_ml = sample_pred[i_ml, i_agent]       
                
                diff = (sample_pred_ml - sample_true[0, i_agent]) ** 2
                # sum over dimension
                diff = diff.sum(1)
                diff = np.sqrt(diff)
                
                # mean over timesteps
                diff = diff.mean(0)
                
                # mean over agents
                diff = diff / num_agents
                
                Error += diff
        
        E = Error / len(self.Output_path)
        return [E]
    
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
