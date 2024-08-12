import numpy as np
from evaluation_template import evaluation_template 

class Miss_rate_joint(evaluation_template):
    r'''
    The value :math:`F` of the Agent-wise Miss rate (assuming :math:`N_{agents,i}` independent agents :math:`j`) is the 
    percentage of sample for which for at least one agent, thelast position of all of the predicted trjacetories :math:`p \in P`is 
    at least :math:`\epsilon = 2 m` away from the true trajectory. 
    It is calculated in the following way:

    .. math ::
        F = {1 \over{N_{samples}} \sum\limits_{i = 1}^{N_{samples}}  \underset{j\in \{1, \hdots, N_{agents,i}\}}{\max} \begin{cases} 1 & \sqrt{\left( x_{i,j}(\max T_{O,i}) - x_{pred,i,p,j} (\max T_{O,i}) \right)^2 + 
             \left( y_{i,j}(\max T_{O,i}) - y_{pred,i,p,j} (\max T_{O,i}) \right)^2} > \epsilon \forall p \in P \\ 0 & \text{otherwise} \end{cases}

    Here, :math:`P` are the set of predictions made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths()
        Pred_agents = Pred_steps.any(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        Num_agents = Pred_agents.sum(-1)
        
        # Get squared distance
        Diff = ((Path_true - Path_pred) ** 2).sum(-1)
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Take last timestep
        Diff = Diff[np.arange(len(Diff)),:,:,Num_steps - 1]
        
        # Mean over predictions
        Missed = (Diff > 2.0).all(1)

        # Get max over agents (i.e., any)
        Missed = Missed.any(-1)

        # Mean over samples
        Error = Missed.mean()
        
        return [Error]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[1]   
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'Miss rate (joint predictions)',
                 'file': 'MR_joint',
                 'latex': r'\emph{MR$_{joint}$ [m]}'}
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