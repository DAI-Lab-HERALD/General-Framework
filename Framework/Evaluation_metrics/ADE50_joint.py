import numpy as np
from ADE_joint import ADE_joint 

class ADE50_joint(ADE_joint):
    r'''
    The value :math:`F` of the Average Displacement Error (assuming :math:`N_{agents}` jointly predicted agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples} | T_O | |P_{50}|}} \sum\limits_{i = 1}^{N_{samples}}  \sum\limits_{p \in P_{50}} 
            \sum\limits_{t \in T_O}\sqrt{{1\over{N_{agents}}} \sum\limits_{j = 1}^{N_{agents}} 
            \left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
        
    Here, :math:`P_{50} \subset P` are 50 randomly selected instances of the set of predictions :math:`P` made for a specific sample :math:`i \in \{1, ..., N_{samples}\}`
    at the predicted timesteps :math:`T_O`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
     
    def evaluate_prediction_method(self):
        num_samples_needed = 50
        num_samples = len(self.Output_path_pred.iloc[0,0])
        if num_samples >= num_samples_needed:
            idx_l = np.random.permutation(num_samples)[:num_samples_needed]#
        else:
            idx_l = np.random.randint(0, num_samples, num_samples_needed)
            
        nto = self.data_set.num_timesteps_out_real
        
        Error = 0
        
        for i_sample in range(len(self.Output_path_pred)):
            # sample_pred.shape = num_path x num_agents x num_timesteps_out x 2
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 1)[idx_l,:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[np.newaxis,:,:nto]
            
            diff = (sample_pred - sample_true) ** 2
            # sum over dimension and mean over number agents
            diff = diff.sum(3).mean(1)
            diff = np.sqrt(diff)
            
            # mean over predicted samples and timesteps
            diff = diff.mean((0, 1))

            Error += diff
        
        E = Error / len(self.Output_path)
        return [E]
    
    
    def get_name(self = None):
        names = {'print': 'ADE (50 samples, joint prediction)',
                 'file': 'ADE50_joint',
                 'latex': r'\emph{ADE$_{50, joint}$ [m]}'}
        return names
    
    