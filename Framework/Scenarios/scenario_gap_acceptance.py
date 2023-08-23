import numpy as np

class scenario_gap_acceptance():
    '''
    Gap acceptance problems are a type of traffic interaction that involves a 
    space-sharing conflict between two agents with intersecting paths, such as 
    intersections, pedestrian crossings, and lane changes on highways. There, 
    these two agents can be differentiated by the possession of the right of way, 
    with the vehicle with priority being referred to as the ego vehicle. In such 
    a situation, the other agent, designated as the target vehicle, must then decide 
    whether to cross ego vehicles path in front of it (i.e., accepting the offered
    gap) or to wait until the ego vehicle has passed, thereby rejecting the gap. 
    
    For example, if the target vehicle approaches an intersection via a secondary road, 
    it needs to decide whether the gap to the vehicle coming from the perpendicular 
    direction is large enough to cross the intersection without waiting for that car
    to pass.
    
    Consequently, the ego and target vehicle are the two agents whose predictions are
    absolutely necessary to predict gap acceptance behavior. If none of those two reaches 
    the contested space in the observed window, then it assumed that the gap can be 
    considered rejected.
    
    In this scenario, a prediction is considered to be useful if the ego vehicle could
    still react to a sudden gap acceptance maneuver by the target vehicle by safely braking
    (to prevent cases of rear ending, safe braking assumes decellerations of 4 m/s^2, and 
     as it can be assumed that the ego vehicle is an AV, reaction times are neglected) and 
    coming to a standstill before the contested space.
    '''
    def __init__(self):
        pass
            
    def get_name(self = None):
        return 'Gap acceptance problem'
        
    def give_classifications(self = None):
        Class = {'rejected': 0,
                 'accepted': 1}
        return Class, 2
    
    def give_default_classification(self = None):
        return 'rejected'
        
    def classifying_agents(self = None):
        # Tar is the vehicle that has to accept or reject the gap
        return ['tar']
            
    def pov_agent(self = None):
        # The ego vehicle is the vehicle offering the gap
        return 'ego'

    def can_provide_general_input(self = None):
        '''
        D_1: Distance between ego vehicle and vehicle that it is following.
        D_2: Distance between ego vehicle and vehicle that it is folowed by.
        D_3: The desitance between tar vehicle and the vehicle it is following.
        L_e: The size of the constested space along the path of the ego vehicle.
        L_t: The size of the constested space along the path of the tar vehicle.

        '''
        return ['D_1', 'D_2', 'D_3', 'L_e', 'L_t']

    def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain):
        a_brake = 4
        t_D_rej = t_D_class.rejected
        t_D_rej[np.abs(t_D_rej) < 1e-3] = 1e-3
        delta_t_useful = 0.5 * D_class.rejected / (a_brake * t_D_rej + 1e-6)
        return delta_t_useful
