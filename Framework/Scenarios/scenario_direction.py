import numpy as np

class scenario_direction():
    '''
    This secnario is defined at intersection. Mainly, it is assumed that a vehicle
    is trying to enter a intersection. Realistically, it should have three options to
    advance (going straight, left, or right). Additionally, as the observation time
    is not infinte, it is also possible that none of those maneuvers is observed,
    in which case, one would say that the vehicle has stayed (U turns, which are technically 
    possible, would also be classified this way).
    
    There are no other vehicles needed to make this classification, so only it is only
    essential for one vehicle to be predicted.
    
    Generally, predictions are only sensible if the vehicle has not yet entered the
    intersection. Consequently, in this instance, the last safe prediction is can be made
    at the point where the vehicle is just 5 m away from the intersection. 
    '''
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
        delta_t_useful = t_D_default - D_crit
        return delta_t_useful #TODO: if Anna wants to allow later prediction, change this
