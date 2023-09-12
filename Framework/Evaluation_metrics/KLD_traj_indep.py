import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde 

from Prob_function import OPTICS_GMM

class KLD_traj_indep(evaluation_template):
    r'''
    TODO: 
    The value :math:`F` of the Kullback-Leibler divergence , 
    is calculated in the following way:
        
    .. math::
        F = {1 \over {|S|}}\sum\limits_{s = 1}^{|S|} D_{KL,s}
    
    Here, :math:`S` is the set of subsets :math:`S_s`, which contain the agents :math:`j`
    for which the initial input to the models was identical.
    
    Each value :math:`D_{KL,s}` is then calcualted in the following way
    (assuming :math:`N_{agents,i}` independent agents :math:`j` for each sample :math:`i`):
    
    .. math::
        D_{KL,s} = {1 \over{|T_{O,s}|}} \sum\limits_{t \in T_{O,s}} {1 \over{|S_s|}}
                    \sum\limits_{(i, j) \in S_s} \ln 
                    \left({ P_{KDE,s, t} \left(\{x_{i,j} (t), y_{i,j} (t)\} \right)   
                           \over{P_{KDE,pred,s,t} \left(\{x_{i,j} (t), y_{i,j} (t)\} \right) }} \right)
                    
    Here, :math:`P_{KDE,s,t}` is a subset and timepoint specific gaussian Kernel Density Estimate trained on all 
    predicted agents in each subset (:math:`(i, j) \in S_s`):
    
    .. math::
        \{x_{i,j} (t), y_{i,j} (t) \} 
    
    
    while :math:`P_{KDE,pred,s,t}` is trained on all predictions (:math:`p \in P`) for all subset samples (:math:`(i, j) \in S_s`):
    
    .. math::
        \{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \}
    
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Types = self.get_true_and_predicted_paths(return_types = True)
        
        Path_true_all, subgroups = self.get_true_prediction_with_same_input()
        
        # Consider agents separately
        Pred_agents = Pred_steps.any(-1)
        
        # Get maximum number representing a distribution
        max_samples = Path_true_all.shape[1]
        
        # Combine agent and samples separately (differentiate between agents)
        agent_indicator = np.linspace(0.1,0.9, Pred_agents.shape[1])
        subgroups = subgroups[:,np.newaxis] + agent_indicator[np.newaxis,:]
        subgroups = subgroups[Pred_agents]
        
        # Combine agent and sample dimension
        Path_pred     = Path_pred.transpose(0,2,1,3,4)[Pred_agents]
        Pred_steps    = Pred_steps[Pred_agents]
        Types         = Types[Pred_agents]
        
        KLD = 0
        unique_subgroups = np.unique(subgroups)
        for subgroup_agent in np.unique(subgroups):
            indices = np.where(subgroup_agent == subgroups)[0]
            
            # Get the predicted agent and subgroup of current selection
            subgroup = int(np.floor(subgroup_agent))
            agent_id = np.argmin(np.abs(agent_indicator - np.mod(subgroup_agent, 1)))
            
            # Get the true futures of the samples with similar input
            samples_true = Path_true_all[subgroup, :, agent_id]
            
            # get the predicted paths for all samples with similar input
            samples_pred = Path_pred[indices]
            
            # Combine samples and prediction into one dimension
            samples_pred = samples_pred.reshape(-1, *samples_pred.shape[2:]) 
            
            # Select number of representative samples
            if len(samples_pred) > max_samples:
                idx = np.random.choice(len(samples_pred), max_samples, replace = False)
                samples_pred = samples_pred[idx]
            
            # Combine agents and dimensions
            samples_true = samples_true.reshape(len(samples_true), -1)
            samples_pred = samples_pred.reshape(len(samples_pred), -1)
            
            # Remove samples that are filler
            samples_true = samples_true[np.isfinite(samples_true).all(1)]
            
            # Train kde models on true and predicted samples
            # kde_true = KernelDensity(kernel='gaussian', bandwidth = 0.2).fit(samples_true)
            # kde_pred = KernelDensity(kernel='gaussian', bandwidth = 0.2).fit(samples_pred)

            kde_true = OPTICS_GMM().fit(samples_true)
            kde_pred = OPTICS_GMM().fit(samples_pred)
            
            # Get the likelihood of the the true samples according to the
            # true samples. This reflects the fact that we take the 
            # expected value over the true distribtuion
            log_like_true = kde_true.score_samples(samples_true)
            log_like_pred = kde_pred.score_samples(samples_true)
            
            KLD += np.mean((log_like_true-log_like_pred))
        
        # Average KLD over subgroups
        KLD /= len(unique_subgroups)
        
        return [KLD]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KLD (Trajectories, independent predictions)',
                 'file': 'KLD_traj_indep',
                 'latex': r'\emph{KLD$_{\text{Traj}, indep}$}'}
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
