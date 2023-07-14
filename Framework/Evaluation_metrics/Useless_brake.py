import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Useless_brake(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Threshold = np.min(self.Output_A_pred.accepted[self.Output_A.accepted])
        Result = np.mean(self.Output_A_pred.accepted[self.Output_A.rejected] < Threshold)
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
        if self.data_set.scenario.get_name() != 'Gap acceptance problem':
            return 'this makes sense in the context of gap acceptance problems.'
        
        if self.data_set.t0_type != 'Crit':
            return 'this metric is meaningless for t0_types outside "crit".'
        return None
