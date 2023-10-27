import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.stats import wasserstein_distance

class Wasserstein_indep(evaluation_template):
    # TODO: Rewrite description
    r'''
    The value :math:`F` of the Wasserstein (assuming :math:`N_{agents, i}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{|P| \sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \sum\limits_{i = 1}^{N_{samples}}  
            \sum\limits_{p \in P} {1\over{| T_{O,i} | }} \sum\limits_{t \in T_{O,i}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
        
    Here, :math:`P` are the set of predictions made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths()
        _, subgroups = self.get_true_prediction_with_same_input()

        # Get predicted agents
        Pred_agents = Pred_steps.any(-1)

        # Get number of steps
        Num_steps = Pred_steps.sum(-1).max(-1)

        Wd = []
        # Iterate over subgroups
        for subgroup in np.unique(subgroups):
            # Get indices of subgroup
            indices = np.where(subgroup == subgroups)[0]

            # Ensure similar pred agents in subgroup
            assert len(np.unique(Pred_agents[indices], axis = 0)) == 1
            pred_agents = Pred_agents[indices[0]]
            num_agents = pred_agents.sum()

            # Get subgroup paths
            path_true = Path_true[indices]
            path_pred = Path_pred[indices]

            # Get similar number of steps
            num_steps = Num_steps[indices]

            wd = []

            for nto in np.unique(num_steps):
                n_ind = np.where(nto == num_steps)[0]

                # Get flattened paths
                path_true_nto = path_true[n_ind,:,:nto][:,pred_agents,:].reshape(-1, num_agents, nto, 2).reshape(-1, num_agents * nto * 2)
                path_pred_nto = path_pred[n_ind,:,:nto][:,pred_agents,:].reshape(-1, num_agents, nto, 2).reshape(-1, num_agents * nto * 2)

                # Apply Wasserstein distance
                wd.append(wasserstein_distance(path_true_nto, path_pred_nto))

            Wd.append(np.mean(wd))

        # Mean over subgroups
        Error = np.mean(Wd)
        
        return [Error]
        
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'ADE (independent predictions)',
                 'file': 'ADE_indep',
                 'latex': r'\emph{ADE$_{indep}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return False
    
    def metric_boundaries(self = None):
        return [0.0, None]
