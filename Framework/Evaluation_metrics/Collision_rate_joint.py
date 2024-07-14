import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Collision_rate_joint(evaluation_template):
    r'''
    The value :math:`F` of the Collision_rate gives the probability, that a predicted agent,
    interacting with the ground truth of other agents or the predicted future of predicted 
    agents, will collide with at least one of them.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Type_pred = self.get_true_and_predicted_paths(return_types = True)
        Path_other, Type_other = self.get_other_agents_paths(return_types = True)

        # Check that Path_other does not has to many timesteps
        if Path_other.shape[-2] > Path_pred.shape[-2]:
            Path_other = Path_other[...,:Path_pred.shape[-2],:]

        num_pred_agents = Path_true.shape[2]
        # Information
        # Shape of Path_true:  (num_samples, 1, num_pred_agents, n_O, 2)
        # Shape of Path_pred:  (num_samples, P, num_pred_agents, n_O, 2)
        # Shape of Pred_steps: (num_samples, num_pred_agents, n_O)
        # Shape of Type_pred:  (num_samples, num_pred_agents)

        # Positions where Pred_steps == Flase, Path_true and Path_pred are 0.0

        # Shape of Path_other: (num_pred_agents, 1, num_other_agents, n_O, 2)
        # Shape of Type_other: (num_pred_agents, num_other_agents)

        # Check for each predicted agent against the ground truth of other agents
        pred_sample, pred_agent = np.where(Pred_steps.any(-1)) # Assume there are N pred agents

        Path_pred_agent = Path_pred[pred_sample, :, pred_agent] # Shape: (N, P, n_O, 2)
        Type_pred_agent = Type_pred[pred_sample, pred_agent] # Shape: (N,)
        N = len(Path_pred_agent)
        P = Path_pred_agent.shape[1]
        num_agents = Path_other.shape[1]
        # Get the ground truth of all other agents
        num_pred_agents = Path_pred.shape[2]
        Idx_sample = np.repeat(np.arange(N)[:, np.newaxis], num_pred_agents - 1, axis = 1)
        Idx_agents = np.repeat(np.arange(1, num_pred_agents)[np.newaxis], N, axis = 0)
        Idx_agents = np.mod(Idx_agents + pred_agent[:, np.newaxis], num_pred_agents)

        # Apply the indexing to the ground truth of other agents
        Path_pred_other = Path_true[pred_sample].transpose(0,2,1,3,4) # Shape: (N, num_pred_agents, P, n_O, 2)
        Path_pred_other = Path_pred_other[Idx_sample, Idx_agents] # Shape: (N, num_pred_agents - 1, P, n_O, 2)
        Type_pred_other = Type_pred[pred_sample][Idx_sample, Idx_agents] # Shape: (N, num_pred_agents - 1)

        Path_other = np.repeat(Path_other[pred_sample, :, :Path_pred_other.shape[-2]].transpose(0,2,1,3,4), P, axis = 2)  # Shape: (N, num_other_agents, P, n_O, 2)
        Type_other = Type_other[pred_agent] # Shape: (N, num_other_agents)

        # Concatenate the ground truth of other agents
        Path_other = np.concatenate([Path_pred_other, Path_other], axis = 1).transpose(0,2,1,3,4) # Shape: (N, num_agents, P, n_O, 2)
        Type_other = np.concatenate([Type_pred_other, Type_other], axis = 1) # Shape: (N, num_agents)

        # Fit range so that types are N, P, num agents, and paths are N, P, num agents, n_O, 2
        Path_pred_agent = np.repeat(Path_pred_agent[:,:,np.newaxis], num_agents, axis = 2)
        Type_pred_agent = np.repeat(np.repeat(Type_pred_agent[:,np.newaxis,np.newaxis], P, axis = 1), num_agents, axis = 2)

        Type_other = np.repeat(Type_other[:,np.newaxis], P, axis = 1)

        # Compute the collision rate
        Collided = self._check_collisions(Path_pred_agent, Path_other, Type_pred_agent, Type_other) # Shape: (N, P, num_agents)

        # A single collision with other agent is enough to count as a collision
        Collided = Collided.any(-1) # Shape: (N, P)

        # Trasnform back into the orignal shape 
        Collided_new = np.zeros(Path_pred.shape[:-2], bool) # Shape: (num_samples, P, num_pred_agents)
        Collided_new[pred_sample, :, pred_agent] = Collided

        # A single collision of any agent is enough to count as a collision
        Collided = Collided_new.any(-1) # Shape: (num_samples, P)

        # Get probability of collision
        Prob_collision = Collided.mean(-1) # Shape: (num_samples) 

        # Get the mean value over all samples
        Collision_rate = Prob_collision.mean()
        return [Collision_rate]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[1]

    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'Collision rate (joint predictions)',
                 'file': 'CR_joint',
                 'latex': r'\emph{CR$_{joint}$ [m]}'}
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
