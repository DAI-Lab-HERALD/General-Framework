import numpy as np
from evaluation_template import evaluation_template 
class KDE_NLL_timestep(evaluation_template):
    r'''
    The value :math:`F` of the Negative Log Likelihood (assuming :math:`N_{agents,i}` independent agents :math:`j`, and :math:`T_{O,i}` independent timesteps :math:`t`), 
    is calculated in the following way:
        
    .. math::
        F = - {1 \over{\sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \sum\limits_{i = 1}^{N_{samples}} \sum\limits_{j = 1}^{N_{agents,i}} \sum\limits_{t = 1}^{T_{O,i}}
        \ln \left( P_{KDE,i,j,t} \left(\{x_{i,j} (t), y_{i,j} (t) \} \right)\right)
            
    Here, :math:`P_{KDE,i,j,t}` is a sample and agent specific gaussian Kernel Density Estimate per timestep t, trained on all predictions (:math:`p \in P`)
    for sample :math:`i \in \{1, ..., N_{samples}\}`, agent :math:`j`, and timestep :math:`t \in T_{O,i}`
    
    .. math::
        \{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} 
    
    For each prediction timestep in :math:`T_{O,i}`, :math:`x` and :math:`y` are the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    def set_default_kwargs(self):
        if 'include_pov' not in self.metric_kwargs:
            self.metric_kwargs['include_pov'] = True

    def setup_method(self):
        self.set_default_kwargs()
     
    def evaluate_prediction_method(self):
        _, _, Pred_steps = self.get_true_and_predicted_paths()
        Pred_agents = Pred_steps.any(-1)
        
        # Get likelihood of having higher probability
        KDE_log_prob_true, KDE_log_prob_pred = self.get_KDE_probabilities(joint_timesteps = False, joint_agents = False)
        
        # Combine agent and sample dimension
        Pred_steps = Pred_steps[Pred_agents]
        PLL        = KDE_log_prob_true[:,0][Pred_agents]
        
        # Scale with number of timesteps
        Num_steps = Pred_steps.sum(-1)
        NLL = np.log(Num_steps.max() / Num_steps) - PLL 
        
        Error = NLL.mean()
        return [Error]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[2]  
    
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
            
        names = {'print': 'KDE_NLL (independent agent and timestep prediction' + P_p + ')',
                 'file': 'KDE_NLL_timestep' + P_f,
                 'latex': r'\emph{KDE$_{' + P_l + r'NLL, timestep}$}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return False
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [None, None]
