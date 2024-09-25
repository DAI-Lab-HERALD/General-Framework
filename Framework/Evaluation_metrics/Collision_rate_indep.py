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
        Path_true, Path_pred, Pred_steps, _, Size_pred = self.get_true_and_predicted_paths(return_types = True)
        Path_other, _, Size_other = self.get_other_agents_paths(return_types = True)

        # Check that Path_other does not has to many timesteps
        if Path_other.shape[-2] > Path_pred.shape[-2]:
            Path_other = Path_other[...,:Path_pred.shape[-2],:]

        num_pred_agents = Path_true.shape[2]
        # Information
        # Shape of Path_true:  (num_samples, 1, num_pred_agents, n_O, 2)
        # Shape of Path_pred:  (num_samples, P, num_pred_agents, n_O, 2)
        # Shape of Pred_steps: (num_samples, num_pred_agents, n_O)
        # Shape of Size_pred:  (num_samples, num_pred_agents, 2)

        # Positions where Pred_steps == Flase, Path_true and Path_pred are 0.0

        # Shape of Path_other: (num_pred_agents, 1, num_other_agents, n_O, 2)
        # Shape of Size_other: (num_pred_agents, num_other_agents, 2)

        # Check for each predicted agent against the ground truth of other agents
        pred_sample, pred_agent = np.where(Pred_steps.any(-1)) # Assume there are N pred agents

        Path_pred_agent = Path_pred[pred_sample, :, pred_agent] # Shape: (N, P, n_O, 2)
        Size_pred_agent = Size_pred[pred_sample, pred_agent] # Shape: (N, 2)
        N = len(Path_pred_agent)
        # Get the ground truth of all other agents
        num_pred_agents = Path_pred.shape[2]
        Idx_sample = np.repeat(np.arange(N)[:, np.newaxis], num_pred_agents - 1, axis = 1)
        Idx_agents = np.repeat(np.arange(1, num_pred_agents)[np.newaxis], N, axis = 0)
        Idx_agents = np.mod(Idx_agents + pred_agent[:, np.newaxis], num_pred_agents)

        # Apply the indexing to the ground truth of other agents
        Path_pred_other = Path_true.squeeze(1)[pred_sample] # Shape: (N, num_pred_agents, n_O, 2)
        Path_pred_other = Path_pred_other[Idx_sample, Idx_agents] # Shape: (N, num_pred_agents - 1, n_O, 2)
        Size_pred_other = Size_pred[pred_sample][Idx_sample, Idx_agents] # Shape: (N, num_pred_agents - 1, 2)

        Path_other = Path_other.squeeze(1)[pred_sample, :, :Path_pred_other.shape[-2]] # Shape: (N, num_other_agents, n_O, 2)
        Size_other = Size_other[pred_agent] # Shape: (N, num_other_agents, 2)

        # Concatenate the ground truth of other agents
        Path_other = np.concatenate([Path_pred_other, Path_other], axis = 1) # Shape: (N, num_agents, n_O, 2)
        Size_other = np.concatenate([Size_pred_other, Size_other], axis = 1) # Shape: (N, num_agents, 2)

        # Fit range so that types are N, P, num agents, and paths are N, P, num agents, n_O, 2
        P = Path_pred_agent.shape[1]
        num_agents = Path_other.shape[1]
        Path_pred_agent = np.repeat(Path_pred_agent[:,:,np.newaxis], num_agents, axis = 2) # Shape: (N, P, num_agents, n_O, 2)
        Size_pred_agent = np.repeat(np.repeat(Size_pred_agent[:,np.newaxis,np.newaxis], P, axis = 1), num_agents, axis = 2) # Shape: (N, P, num_agents, n_O, 2)

        Path_other = np.repeat(Path_other[:,np.newaxis], P, axis = 1) # Shape: (N, P, num_agents, n_O, 2)
        Size_other = np.repeat(Size_other[:,np.newaxis], P, axis = 1) # Shape: (N, P, num_agents, 2)

        # Compute the collision rate
        Collided = self._check_collisions(Path_pred_agent, Path_other, Size_pred_agent, Size_other) # Shape: (N, P, num_agents)

        # A single collision with other agent is enough to count as a collision
        Collided = Collided.any(-1) # Shape: (N, P)

        # Get probability of collision
        Prob_collision = Collided.mean(-1) # Shape: (N) 

        # Get the mean value over all samples
        Collision_rate = Prob_collision.mean()
        return [Collision_rate]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[2]
        
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
