import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class KLD_traj_joint(evaluation_template):
    r'''
    The value :math:`F` of the Kullback-Leibler divergence , 
    is calculated in the following way:
        
    .. math::
        F = {1 \over {|S|}}\sum\limits_{s = 1}^{|S|} D_{KL,s}
    
    Here, :math:`S` is the set of subsets :math:`S_s`, which contain the samples :math:`i`
    for which the initial input to the models was identical.
    
    Each value :math:`D_{KL,s}` is then calcualted in the following way
    (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j` for each sample :math:`i`):
    
    .. math::
        D_{KL,s} = {1 \over{|S_s|}} \sum\limits_{i \in S_s} \ln 
                    \left({ P_{KDE,s} \left(\{\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \, | \; \forall\, j \} \right)   
                           \over{P_{KDE,pred,s} \left(\{\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \, | \; \forall\, j \} \right) }} \right)
                    
    Here, :math:`P_{KDE,s}` is a subset and timepoint specific gaussian Kernel Density Estimate trained on all samples
    in each subset (:math:`i \in S_s`):
    
    .. math::
        \{\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \, | \; \forall\, j \}
    
    
    while :math:`P_{KDE,pred,s}` is trained on all predictions (:math:`p \in P`) for all subset samples (:math:`i \in S_s`):
    
    .. math::
        \{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \, | \; \forall\, j \}
    
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        _, subgroups = self.get_true_prediction_with_same_input()
        
        KDE_pred_log_prob_true, _ = self.get_KDE_probabilities(joint_agents = True)
        KDE_true_log_prob_true, _ = self.get_true_likelihood(joint_agents = True)
        
        KLD = 0
        unique_subgroups = np.unique(subgroups)
        
        for subgroup in np.unique(subgroups):
            indices = np.where(subgroup == subgroups)[0]
            
            # Get the likelihood of the the true samples according to the
            # true samples. This reflects the fact that we take the 
            # expected value over the true distribtuion
            log_like_true = KDE_true_log_prob_true[indices,0,0]
            log_like_pred = KDE_pred_log_prob_true[indices,0,0]
            
            kld_subgroup = np.mean((log_like_true-log_like_pred))
            
            KLD += kld_subgroup
        
        # Average KLD over subgroups
        KLD /= len(unique_subgroups)
        
        return [KLD]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KLD (Trajectories, joint predictions)',
                 'file': 'KLD_traj_joint',
                 'latex': r'\emph{KLD$_{\text{Traj}, joint}$}'}
        return names
    
    
    def is_log_scale(self = None):
        return False
    
    def check_applicability(self):
        if not self.data_set.enforce_num_timesteps_out:
            return "the metric requires that ouput trajectories are fully observed."
            
        return None
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]
