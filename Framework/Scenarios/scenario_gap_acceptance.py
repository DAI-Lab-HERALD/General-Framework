import numpy as np

class scenario_gap_acceptance():
    def __init__(self):
        pass
    
    def give_default_classification(self = None):
        return 'rejected'
    
    def give_classifications(self = None):
        Class = {'rejected': 0,
                 'accepted': 1}
        return Class, 2
        
    def get_name(self = None):
        return 'Gap acceptance problem'
        
    def can_provide_general_input(self = None):
        '''
        D_1: Distance between ego vehicle and vehicle that it is following.
        D_2: Distance between ego vehicle and vehicle that it is folowed by.
        D_3: The desitance between tar vehicle and the vehicle it is following.
        L_e: The size of the constested space along the path of the ego vehicle.
        L_t: The size of the constested space along the path of the tar vehicle.

        '''
        return ['D_1', 'D_2', 'D_3', 'L_e', 'L_t']
        
    def pov_agent(self = None):
        # The ego vehicle is the vehicle offering the gap
        return 'ego'
        
    def classifying_agents(self = None):
        # Tar is the vehicle that has to accept or reject the gap
        return ['tar']
    
    
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        a_brake = 4
        t_D_rej = t_D_class.rejected
        t_D_rej[np.abs(t_D_rej) < 1e-3] = 1e-3
        t_brake = 0.5 * D_class.rejected / (a_brake * t_D_rej + 1e-6)
        return t_brake