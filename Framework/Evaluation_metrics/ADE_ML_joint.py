import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 



class ADE_ML_joint(evaluation_template):
    r'''
    The value :math:`F` of the most likely Average Displacement Error (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples}}} \sum\limits_{i = 1}^{N_{samples}} 
            {1\over{| T_{O,i} | }} \sum\limits_{t \in T_{O,i}}\sqrt{{1\over{N_{agents,i}}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \left( x_{i,j}(t) - x_{pred,i,p^*_i,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p^*_i,j} (t) \right)^2}
            
    Here, for each specific sample :math:`i \in \{1, ..., N_{samples}\}`
    
    .. math::
            p^*_{i} = \underset{p \in P}{\text{arg} \min} P_{KDE,i} \left(\{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_O\} \, | \; \forall \, j \} \right) , 
    
    where :math:`P_{KDE,i}`, a sample specific gaussian Kernel Density Estimate trained on all predictions :math:`p \in P`, returns the
    likelihood for trajectories predicted at timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.

    Here, the number of predictions :math:`|P|` can be set using the kwargs, under the key 'num_preds'. If not set, None is assumed.
    '''
    def set_default_kwargs(self):
        if 'num_preds' not in self.metric_kwargs:
            self.metric_kwargs['num_preds'] = None
            
        if 'include_pov' not in self.metric_kwargs:
            self.metric_kwargs['include_pov'] = True

    def setup_method(self):
        self.set_default_kwargs()
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths(self.metric_kwargs['num_preds'])
        Pred_agents = Pred_steps.any(-1) 
        
        _, KDE_log_prob_pred = self.get_KDE_probabilities(joint_agents = True)
        ml_pred_index = np.argmax(KDE_log_prob_pred, axis = 1)
        
        num_samples = Path_pred.shape[0]
        
        # Get indices for most likely predictions
        sample_index = np.arange(num_samples)
        ml_pred_index = np.argmax(KDE_log_prob_pred[...,0], axis = 1)
        
        Path_pred_ml = Path_pred[sample_index, ml_pred_index]
        
        # Get num steps and num agents
        Num_agents = Pred_agents.sum(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        
        # Get squared distance
        Diff = ((Path_true[:,0] - Path_pred_ml) ** 2).sum(-1)
        
        # Get mean over agents
        Diff = Diff.sum(1) / Num_agents[:,np.newaxis]
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Get mean over timesteps
        Diff = Diff.sum(-1) / Num_steps
        
        # Get mean over samples        
        Error = Diff.mean()
        return [Error]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[1]
    
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
        if self.metric_kwargs['num_preds'] == None:
            N_p = ''
            N_f = ''
            N_l = ''
        else:
            N_p = str(self.metric_kwargs['num_preds']) + ' samples, '
            N_f = str(self.metric_kwargs['num_preds'])
            N_l = str(self.metric_kwargs['num_preds']) + ', '
        
        if self.metric_kwargs['include_pov']:
            P_p = ''
            P_f = ''
            P_l = ''
        else:
            P_p = ', exclude POV'
            P_f = 'nP'
            P_l = 'nP, '
            
            
        names = {'print': 'ADE_ML (' + N_p + 'joint prediction' + P_p + ')',
                'file': 'ADE_ML' + N_f + '_joint' + P_f,
                'latex': r'\emph{ADE$_{ML, ' + N_l + P_l + r'joint}$ [m]}'}
        return names
    
    def is_log_scale(self = None):
        return True
    
    def check_applicability(self):
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]
