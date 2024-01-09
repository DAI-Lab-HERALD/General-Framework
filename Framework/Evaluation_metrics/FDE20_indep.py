import numpy as np
from FDE_indep import FDE_indep 

class FDE20_indep(FDE_indep):
    r'''
    The value :math:`F` of the Final Displacement Error (assuming :math:`N_{agents,i}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{|P| \sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \sum\limits_{i = 1}^{N_{samples}}  
            \sum\limits_{p \in P_{20}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \sqrt{\left( x_{i,j}(\max T_{O,i}) - x_{pred,i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - y_{pred,i,p,j} (\max T_{O,i}) \right)^2}
        
    Here, :math:`P_{20} \subset P` are 20 randomly selected instances of the set of predictions :math:`P` made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths(20)
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
        Diff = Diff.mean(1)
        
        # Mean over samples and agents
        Error = Diff.sum() / Num_agents.sum()
        
        return [Error]
    
    
    def get_name(self = None):
        names = {'print': 'FDE (20 samples, independent prediction)',
                 'file': 'FDE20_indep',
                 'latex': r'\emph{FDE$_{20, indep}$ [m]}'}
        return names
    
    