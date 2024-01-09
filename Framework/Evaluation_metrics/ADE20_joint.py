import numpy as np
from ADE_joint import ADE_joint 

class ADE20_joint(ADE_joint):
    r'''
    The value :math:`F` of the Average Displacement Error (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples}  |P_{20}|}} \sum\limits_{i = 1}^{N_{samples}}  \sum\limits_{p \in P_{20}} 
            {1\over{T_{O,i}}}\sum\limits_{t \in T_{O,i}}\sqrt{{1\over{N_{agents,i}}} \sum\limits_{j = 1}^{N_{agents,i}} 
            \left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
        
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
        
        # Get mean over agents
        Diff = Diff.sum(2) / Num_agents[:,np.newaxis,np.newaxis]
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Get mean over timesteps
        Diff = Diff.sum(-1) / Num_steps[:,np.newaxis]
        
        # Get mean over predictions and samples        
        Error = Diff.mean()
        
        return [Error]
    
    
    def get_name(self = None):
        names = {'print': 'ADE (20 samples, joint prediction)',
                 'file': 'ADE20_joint',
                 'latex': r'\emph{ADE$_{20, joint}$ [m]}'}
        return names
    
    