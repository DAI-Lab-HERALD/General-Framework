import numpy as np

class scenario_direction_highway():
    '''
    This secnario is defined at highways. Realistically, it should have three options to
    advance (going straight, left, or right). 
    '''
    def __init__(self):
        pass

    def get_name(self = None):
        return 'Direction prediction on highways'
    
    def give_classifications(self = None):
        Class = {'right':    0,
                 'straight': 1,
                 'left':     2}
        return Class, 3
        
    def give_default_classification(self = None):
        return 'straight'
    
    def classifying_agents(self = None):
        return ['tar']
    
    def pov_agent(self = None):
        return None
        
    def can_provide_general_input(self = None):
        return []
        
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        return None
