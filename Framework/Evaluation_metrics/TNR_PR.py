import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class TNR_PR(evaluation_template):
    r'''
    The true negative rate under perfect recall is designed to specifically for gap acceptance problems,
    where one predicts the probability :math:`p_{accepted,i}` for a specific 
    sample :math:`i \in \{1, ..., N_{samples}\}` that a gap offered (by an AV) will be acceoted.
    It assumes that a false negative prediction has to be avoided under all circumstances (perfect recall),
    as this would lead potentially to a collision. It than rates the likelihood :math:`L_{brake}` of models to make false positive 
    predictions, leading to unnecessary braking.
    
    Firstly, the needed decision threshold to ensure perfect recall is calculated:
        
    .. math::
        p_{accepted,T} = \underset{j \in \{i \, | \, p_{accepted,true,i} = 1\}}{\min} p_{accepted,pred,j} 
    
    Then, the likelihood :math:`L_{brake}` can be calculated:
        
    .. math::
        L_{brake} = {1 \over{|\{i \, | \, p_{accepted,true,i} = 0\}|}} 
        \sum\limits_{j \in \{i \, | \, p_{accepted,true,i} = 0\}}  p_{accepted,pred,i}
        
    '''
    
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        P_true, P_pred, Class_names = self.get_true_and_predicted_class_probabilities()
        
        i_accepted = np.where(Class_names == 'accepted')[0][0]
        
        P_accepted_true = P_true[:,i_accepted].astype(bool)
        P_accepted_pred = P_pred[:,i_accepted]
        
        Threshold = np.min(P_accepted_pred[P_accepted_true])
        Result = np.mean(P_accepted_pred[~P_accepted_true] < Threshold)
        return [Result]
    
    def get_output_type(self = None):
        return 'class'
    
    def get_t0_type(self = None):
        return 'crit'
    
    def get_opt_goal(self = None):
        return 'maximize'
    
    def get_name(self = None):
        names = {'print': 'TNR (under perfect recall)',
                 'file': 'TNR',
                 'latex': r'\emph{TNR-PR}'}
        return names
    
    def requires_preprocessing(self):
        return False
    
    def is_log_scale(self = None):
        return False
    
    def allows_plot(self):
        return False
        
    def check_applicability(self):
        if self.data_set.scenario_name != 'Gap acceptance problem':
            return 'this makes sense in the context of gap acceptance problems.'
        
        if self.data_set.t0_type != 'Crit':
            return 'this metric is meaningless for t0_types outside "crit".'
        return None
    
    def metric_boundaries(self = None):
        return [0.0, 1.0]
