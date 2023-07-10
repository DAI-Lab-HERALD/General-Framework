from scenario_template import scenario_template
import numpy as np

class scenario_gap_acceptance(scenario_template):
    def give_default_classification(self = None):
        return 'rejected'
    
    def give_classifications(self = None):
        Class = {'rejected': 0,
                 'accepted': 1}
        return Class, 2
        
    def get_name(self = None):
        return 'Gap acceptance problem'
        
    def can_provide_general_input(self = None):
        return ['D_1', 'D_2', 'D_3', 'L_e', 'L_t']
        
    def pov_agent(self = None):
        return 'ego'
        
    def classifying_agents(self = None):
        return ['tar']
    
    
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        a_brake = 4
        t_D_rej = t_D_class.rejected
        t_D_rej[np.abs(t_D_rej) < 1e-3] = 1e-3
        t_brake = 0.5 * D_class.rejected / (a_brake * t_D_rej + 1e-6)
        return t_brake