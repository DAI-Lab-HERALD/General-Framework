import numpy as np
import pandas as pd
from scipy.special import logsumexp
from evaluation_template import evaluation_template 

class JSD_traj_indep(evaluation_template):
    r'''
    TODO: 
    The value :math:`F` of the Jensen-Shannon divergence , 
    is calculated in the following way:
        
    .. math::
        F = {1 \over {|S|\ln(2)}}\sum\limits_{s = 1}^{|S|} D_{JS,s}
    
    Here, :math:`S` is the set of subsets :math:`S_s`, which contain the agents :math:`j`
    for which the initial input to the models was identical. The division by :math:`\ln(2)`
    is used to normalise the output values inbetween 0 (identical distributions) and 1.
    
    Each value :math:`D_{JS,s}` is then calcualted in the following way
    (assuming :math:`N_{agents,i}` independent agents :math:`j` for each sample :math:`i`):
    
    .. math::
        D_{JS,s} & = {1 \over {2}} \left(D_{KL,s} + D_{KL,s,pred}\right) \\
        D_{KL,s} & = {1 \over{|S_s|}} \sum\limits_{(i, j) \in S_s} \ln 
                    \left({ P_{KDE,s} \left(\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)   
                           \over{{1 \over {2}} \left( P_{KDE,pred,s} \left(\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right) 
                                                     + P_{KDE,s} \left(\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)
                                                     \right)}} \right) \\
        D_{KL,s, pred} & = {1 \over{|S_s| |P|}} \sum\limits_{(i, j) \in S_s}\sum\limits_{p \in P} \ln 
                    \left({ P_{KDE,pred,s} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)   
                           \over{{1 \over {2}} \left( P_{KDE,pred,s} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right) 
                                                     + P_{KDE,s} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)
                                                     \right)}} \right)
                    
    Here, :math:`P_{KDE,s}` is a subset and timepoint specific gaussian Kernel Density Estimate trained on all 
    predicted agents in each subset (:math:`(i, j) \in S_s`):
    
    .. math::
        \{\{x_{i,j} (t), y_{i,j} (t)\} \vert \forall t \in T_{O,s}\}
    
    
    while :math:`P_{KDE,pred,s}` is trained on all predictions (:math:`p \in P`) for all subset samples (:math:`(i, j) \in S_s`):
    
    .. math::
        \{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \vert \forall t \in T_{O,s}\}
    
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        _, _, Pred_steps = self.get_true_and_predicted_paths()
        
        _, subgroups = self.get_true_prediction_with_same_input()
        
        # Get likelihood of true samples according to prediction
        KDE_pred_log_prob_true, KDE_pred_log_prob_pred = self.get_KDE_probabilities(joint_agents = False)
        KDE_true_log_prob_true, KDE_true_log_prob_pred = self.get_true_likelihood(joint_agents = False)
        
        # Consider agents separately
        Pred_agents = Pred_steps.any(-1)
        
        # Combine agent and samples separately (differentiate between agents)
        agent_indicator = np.linspace(0.1,0.9, Pred_agents.shape[1])
        subgroups = subgroups[:,np.newaxis] + agent_indicator[np.newaxis,:]
        subgroups = subgroups[Pred_agents]
        
        # Combine agent and sample dimension
        Log_like_predTrue = KDE_pred_log_prob_true[:,0][Pred_agents] # P(X)
        Log_like_trueTrue = KDE_true_log_prob_true[:,0][Pred_agents] # Q(X)
        Log_like_predPred = KDE_pred_log_prob_pred.transpose(0,2,1)[Pred_agents] # P(\hat{X})
        Log_like_truePred = KDE_true_log_prob_pred.transpose(0,2,1)[Pred_agents] # Q(\hat{X})
        
        JSD = 0
        unique_subgroups = np.unique(subgroups)
        for subgroup_agent in np.unique(subgroups):
            indices = np.where(subgroup_agent == subgroups)[0]
            
            # Get the likelihood of the the true samples according to the
            # true samples. This reflects the fact that we take the 
            # expected value over the true distribtuion
            log_like_trueTrue = Log_like_trueTrue[indices]
            log_like_predTrue = Log_like_predTrue[indices]
            log_like_truePred = Log_like_truePred[indices]
            log_like_predPred = Log_like_predPred[indices]

            log_like_combinedTrue = logsumexp(np.stack([log_like_trueTrue, log_like_predTrue], axis = 0), axis = 0) - np.log(2)
            log_like_combinedPred = logsumexp(np.stack([log_like_truePred, log_like_predPred], axis = 0), axis = 0) - np.log(2)
            
            kld_subgroupTrue = np.mean(log_like_trueTrue-log_like_combinedTrue)
            kld_subgroupPred = np.mean(log_like_predPred-log_like_combinedPred)
            
            JSD += 0.5*kld_subgroupPred + 0.5*kld_subgroupTrue
        
        # Average JSD over subgroups
        JSD /= len(unique_subgroups)
        JSD /= np.log(2)
        
        return [JSD]
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'JSD (Trajectories, independent predictions)',
                 'file': 'JSD_traj_indep',
                 'latex': r'\emph{JSD$_{\text{Traj}, indep}$}'}
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
