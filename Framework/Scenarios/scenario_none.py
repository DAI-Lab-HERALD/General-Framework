
class scenario_none():
    def __init__(self):
        pass
    
    def give_default_classification(self = None):
        return 'normal'
    
    def give_classifications(self = None):
        Class = {'normal': 0}
        return Class, 1
        
    def get_name(self = None):
        return 'No specific scenario'
        
    def can_provide_general_input(self = None):
        return False
        
    def pov_agent(self = None):
        return None
        
    def classifying_agents(self = None):
        return ['tar']
    
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        return None