
class scenario_none():
    def __init__(self):
        pass
            
    def get_name(self = None):
        return 'No specific scenario'
        
    def give_classifications(self = None):
        Class = {'normal': 0}
        return Class, 1
    
    def give_default_classification(self = None):
        return 'normal'
        
    def classifying_agents(self = None):
        return ['tar']
            
    def pov_agent(self = None):
        return None

    def can_provide_general_input(self = None):
        return None

    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        return None
