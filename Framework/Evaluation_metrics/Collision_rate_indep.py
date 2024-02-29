import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Collision_rate_indep(evaluation_template):
    r'''
    The value :math:`F` of the Collision_rate gives the probability, that a predicted agent,
    interacting with the ground truth of other agents, will collide with at least one of them.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Type_pred = self.get_true_and_predicted_paths(return_types = True)
        Path_other, Type_other = self.get_other_agents_paths()

        # Reform to treat each agent separately
        Pred_agents = Pred_steps.any(-1)
        num_samples, num_agent_pred = Pred_agents.shape
        sample_pred, agent_pred = np.where(Pred_agents)

        sample_pred_tiles = np.tile(sample_pred[:,np.newaxis], (1, num_agent_pred - 1)).T
        agent_pred_tiles = np.tile(np.arange(len(num_agent_pred - 1))[np.newaxis], (len(sample_pred), 1))
        agent_pred_tiles[agent_pred_tiles >= agent_pred[:,np.newaxi]] += 1

        # get all combinations
        path_pred = Path_pred[sample_pred, :, agent_pred][:, :, np.newaxis] # num_pred_agents x num_pred x 1 x n_O x 2
        path_other = Path_other[sample_pred] # num_pred_agents x 1 x other_agents x n_O x 2
        path_pred_other = Path_true[sample_pred_tiles, :, agent_pred_tiles].transpose(0, 2, 1, 3, 4) # num_pred_agents x 1 x other_pred_agents x n_O x 2

        path_other = np.concatenate([path_other, path_pred_other], axis = 2) # num_pred_agents x 1 x all_other_agents x n_O x 2

        # Get all types
        type_pred = Type_pred[sample_pred, agent_pred]
        type_other = Type_other[sample_pred]
        type_pred_other = Type_pred[sample_pred_tiles, agent_pred_tiles]
        type_other = np.concatenate([type_other, type_pred_other], axis = 2)

        # Check for collisions
        Dist = np.linalg.norm(path_pred - path_other, axis = -1) # num_pred_agents x num_pred x all_other_agents x n_O

        # Min dist over timesteps
        Dist = np.nanmin(Dist, axis = 1) # num_pred_agents x num_pred x all_other_agents

        # Get comparison distance
        Dist_comp = np.zeros(type_other.shape)
        Dist_comp[type_pred == 'P'] += 0.2
        Dist_comp[type_pred == 'V'] += 1
        Dist_comp[type_pred == 'B'] += 0.5
        Dist_comp[type_pred == 'M'] += 0.6
        Dist_comp[type_other == 'P'] += 0.2
        Dist_comp[type_other == 'V'] += 1
        Dist_comp[type_other == 'B'] += 0.5
        Dist_comp[type_other == 'M'] += 0.6

        # Exclude non existent agents
        Dist_comp[type_pred == '0'] += np.inf
        Dist_comp[type_other == '0'] += np.inf

        # Check for collisions
        Collisions = np.any((Dist < Dist_comp[:,np.newaxis]) & np.isfinite(Dist), axis = -1)

        # Get collision rate 
        Collision_rate = Collisions.mean()

        return [Collision_rate]
        
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'Collision rate (independent predictions)',
                 'file': 'CR_indep',
                 'latex': r'\emph{CR$_{indep}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return False
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, 1.0]
