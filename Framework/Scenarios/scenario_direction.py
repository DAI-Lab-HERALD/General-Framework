import numpy as np

class scenario_direction():
    def __init__(self):
        pass

    def get_name(self = None):
        return 'Direction prediction at intersection'
    
    def give_classifications(self = None):
        Class = {'staying':  0,
                 'right':    1,
                 'straight': 2,
                 'left':     3}
        return Class, 4
        
    def give_default_classification(self = None):
        return 'staying'
    
    def classifying_agents(self = None):
        return ['tar']
    
    def pov_agent(self = None):
        return None
        
    def can_provide_general_input(self = None):
        return ['D_decision']
        
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        t_D_default = np.minimum(1000, t_D_class[data_set.behavior_default])
        vehicle_len = 5
        D_crit = data_set.calculate_additional_distances(path, None, domain).D_decision - vehicle_len
        t_safe_action = t_D_default - D_crit
        # Delta_tD = t_D_default - t_safe_action
        return t_safe_action #TODO: if Anna wants to allow later prediction, change this
