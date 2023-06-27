from scenario_template import scenario_template

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
        return True
        
    def pov_agent(self = None):
        return 'ego'
        
    def classifying_agents(self = None):
        return ['tar']
    
    
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        a_brake = 4
        t_brake = 0.5 * D_class.rejected / (a_brake * t_D_class.rejected)
        return t_brake