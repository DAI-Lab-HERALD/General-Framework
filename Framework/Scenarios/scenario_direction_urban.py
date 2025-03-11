import numpy as np

class scenario_direction_urban():
    '''
    This secnario is defined for urban driving. Realistically, it should have six options to
    advance (going straight, left/right lane change, left/right turn, u-turn). Additionally, 
    as the observation time is not infinte, it is also possible that none of those maneuvers 
    is observed, in which case, one would say that the vehicle has stayed.
    
    There are no other vehicles needed to make this classification, so it is only
    essential for one vehicle to be predicted.
    '''
    def __init__(self):
        pass

    def get_name(self = None):
        return 'Direction prediction for urban driving'
    
    def give_classifications(self = None):
        Class = {'staying':           0,
                 'right_turn':        1,
                 'straight':          2,
                 'left_turn':         3,
                 'u_turn':            4,
                 'right_lane_change': 5,
                 'left_lane_change':  6}
        return Class, 7
        
    def give_default_classification(self = None):
        return 'staying'
    
    def classifying_agents(self = None):
        return ['tar']
    
    def pov_agent(self = None):
        return None
        
    def can_provide_general_input(self = None):
        return []
        
    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        return None #TODO: if Anna wants to allow later prediction, change this
