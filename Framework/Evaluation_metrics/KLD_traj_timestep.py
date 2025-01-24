import numpy as np
from evaluation_template import evaluation_template 

class KLD_traj_timestep(evaluation_template):
    r'''
    The value :math:`F` of the Kullback-Leibler divergence , 
    is calculated in the following way:
        
    .. math::
        F = {1 \over {|S|}}\sum\limits_{s = 1}^{|S|} D_{KL,s}
    
    Here, :math:`S` is the set of subsets :math:`S_s`, which contain the agents :math:`j`
    for which the initial input to the models was identical.
    
    Each value :math:`D_{KL,s}` is then calcualted in the following way
    (assuming :math:`N_{agents,i}` independent agents :math:`j` and :math:`T_{O,i}` independent timesteps :math:`t` for each sample :math:`i`):
    
    .. math::
        D_{KL,s} = {1 \over{|S_s|}} \sum\limits_{(i, j, t) \in S_s} \ln 
                    \left({ P_{KDE,s} \left(\{x_{i,j} (t), y_{i,j} (t)\} \right)   
                           \over{P_{KDE,pred,s} \left(\{x_{i,j} (t), y_{i,j} (t)\} \right) }} \right)
                    
    Here, :math:`P_{KDE,s}` is a subset and timepoint specific gaussian Kernel Density Estimate trained on all 
    predicted agents in each subset (:math:`(i, j, t) \in S_s`):
    
    .. math::
        \{x_{i,j} (t), y_{i,j} (t)\}
    
    
    while :math:`P_{KDE,pred,s}` is trained on all predictions (:math:`p \in P`) for all subset samples (:math:`(i, j, t) \in S_s`):
    
    .. math::
        \{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\}
    
    '''
    def set_default_kwargs(self):
        if 'include_pov' not in self.metric_kwargs:
            self.metric_kwargs['include_pov'] = True

    def setup_method(self):
        self.set_default_kwargs()
     
    def evaluate_prediction_method(self):
        _, _, Pred_steps = self.get_true_and_predicted_paths()
        
        _, subgroups = self.get_true_prediction_with_same_input()
        
        # Get likelihood of true samples according to prediction
        KDE_pred_log_prob_true, _ = self.get_KDE_probabilities(joint_timesteps=False, joint_agents = False)
        KDE_true_log_prob_true, _ = self.get_true_likelihood(joint_timesteps=False, joint_agents = False)
        
        # Consider agents separately
        Pred_agents = Pred_steps.any(-1)
        
        # Combine agent and samples separately (differentiate between agents)
        agent_indicator = np.linspace(0.1,0.9, Pred_agents.shape[1])
        subgroups = subgroups[:,np.newaxis] + agent_indicator[np.newaxis,:]
        subgroups = subgroups[Pred_agents]
        
        # Combine agent and sample dimension
        Log_like_pred = KDE_pred_log_prob_true[:,0][Pred_agents]
        Log_like_true = KDE_true_log_prob_true[:,0][Pred_agents]
        
        KLD = 0
        unique_subgroups = np.unique(subgroups)
        for subgroup_agent in np.unique(subgroups):
            indices = np.where(subgroup_agent == subgroups)[0]
            
            # Get the likelihood of the the true samples according to the
            # true samples. This reflects the fact that we take the 
            # expected value over the true distribtuion
            log_like_true = Log_like_true[indices]
            log_like_pred = Log_like_pred[indices]
            
            kld_subgroup = np.mean((log_like_true-log_like_pred))
            
            KLD += kld_subgroup
        
        # Average KLD over subgroups
        KLD /= len(unique_subgroups)
        
        return [KLD]
    
    def partial_calculation(self = None):
        options = ['No', 'Subgroups', 'Sample', 'Subgroup_pred_agents', 'Pred_agents']
        return options[3]  
    
    def get_output_type(self = None):
        self.set_default_kwargs()
        if self.metric_kwargs['include_pov']:
            return 'path_all_wi_pov'
        else:
            return 'path_all_wo_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        self.set_default_kwargs()
        if self.metric_kwargs['include_pov']:
            P_p = ''
            P_f = ''
            P_l = ''
        else:
            P_p = ', exclude POV'
            P_f = 'nP'
            P_l = 'nP, '
            
        names = {'print': 'KLD (Trajectories, independent agent and timestep prediction' + P_p + ')',
                 'file': 'KLD_traj_timestep' + P_f,
                 'latex': r'\emph{KLD$_{' + P_l + r'\text{Traj}, timestep}$}'}
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
