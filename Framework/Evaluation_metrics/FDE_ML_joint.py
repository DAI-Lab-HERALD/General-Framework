import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class FDE_ML_joint(evaluation_template):
    r'''
    The value :math:`F` of the most likely Final Displacement Error (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples}}} \sum\limits_{i = 1}^{N_{samples}} 
            \sqrt{{1\over{N_{agents,i}}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \left( x_{i,j}(\max T_{O,i}) - x_{pred,i,p^*_i,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - y_{pred,i,p^*_i,j} (\max T_{O,i}) \right)^2}
            
    Here, for each specific sample :math:`i \in \{1, ..., N_{samples}\}`
    
    .. math::
            p^*_{i} = \underset{p \in P}{\text{arg} \min} P_{KDE,i} \left(\{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \, | \; \forall \, j \} \right) , 
    
    where :math:`P_{KDE,i}`, a sample specific gaussian Kernel Density Estimate trained on all predictions :math:`p \in P`, returns the
    likelihood for trajectories predicted at timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Types = self.get_true_and_predicted_paths(return_types = True)
        Pred_agents = Pred_steps.any(-1) 
        Num_agents = Pred_agents.sum(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        
        
        Path_pred_ml = np.zeros(Path_pred[:,0].shape)
        for i_sample in range(len(self.Output_path_pred)):
            pred_agents = Pred_agents[i_sample]
            std = 1 + (Types[i_sample, pred_agents] != 'P') * 79
            std = std[np.newaxis, :, np.newaxis, np.newaxis]
            
            nto = Num_steps[i_sample]
            n_agents = Num_agents[i_sample]
            
            path_pred = Path_pred[i_sample][:,pred_agents,:nto]
            
            path_pred_comp = (path_pred / std).reshape(-1, n_agents * nto * 2)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(path_pred_comp)

            log_prob = kde.score_samples(path_pred_comp)
            p_ml = np.argmax(log_prob)

            Path_pred_ml[i_sample, pred_agents, :nto] = path_pred[p_ml]    
            
        # Get squared distance
        Diff = ((Path_true[:,0] - Path_pred_ml) ** 2).sum(-1)
        
        # Get mean over agents
        Diff = Diff.sum(1) / Num_agents[:,np.newaxis]
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Take last timestep
        Diff = Diff[np.arange(len(Diff)),Num_steps - 1]
        
        # Get mean over samples        
        Error = Diff.mean()
        return [Error]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'FDE_ML (joint prediction)',
                 'file': 'FDE_ML_joint',
                 'latex': r'\emph{FDE$_{ML, joint}$ [m]}'}
        return names
    
    def is_log_scale(self = None):
        return True
    
    def check_applicability(self):
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
