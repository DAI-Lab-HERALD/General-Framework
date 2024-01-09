import numpy as np
from ADE_indep import ADE_indep 

class ADE50_indep(ADE_indep):
    r'''
    The value :math:`F` of the Average Displacement Error (assuming :math:`N_{agents,i}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{|P_{50}| \sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \sum\limits_{i = 1}^{N_{samples}}  
            \sum\limits_{p \in P_{50}} {1\over{| T_{O,i} | }} \sum\limits_{t \in T_{O,i}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
        
    Here, :math:`P_{50} \subset P` are 50 randomly selected instances of the set of predictions :math:`P` made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_O`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths(50)
        Pred_agents = Pred_steps.any(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        Num_agents = Pred_agents.sum(-1)
        
        # Get squared distance
        Diff = ((Path_true - Path_pred) ** 2).sum(-1)
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Mean over timesteps
        Diff = Diff.sum(-1) / Num_steps[:,np.newaxis,np.newaxis]
        
        # Mean over predictions
        Diff = Diff.mean(1)
        
        # Mean over samples and agents
        Error = Diff.sum() / Num_agents.sum()
        
        return [Error]
    
    
    def get_name(self = None):
        names = {'print': 'ADE (50 samples, independent prediction)',
                 'file': 'ADE50_indep',
                 'latex': r'\emph{ADE$_{50, indep}$ [m]}'}
        return names
    
    