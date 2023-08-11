import numpy as np
from ADE_indep import ADE_indep 

class ADE50_indep(ADE_indep):
    r'''
    The value :math:`F` of the Average Displacement Error (assuming :math:`N_{agents}` independent agents :math:`j`), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples} | T_O | |P_{50}| N_{agents}}} \sum\limits_{i = 1}^{N_{samples}}  
            \sum\limits_{p \in P_{50}}  \sum\limits_{t \in T_O} \sum\limits_{j = 1}^{N_{agents}} 
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
        
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
            # sum over dimension
            diff = diff.sum(3)
            diff = np.sqrt(diff)
            
            # mean over predicted samples, agents, and timesteps
            diff = diff.mean((0,1,2))

            Error += diff
        
        E = Error / len(self.Output_path)
        return [E]
    
    
    def get_name(self = None):
        names = {'print': 'ADE (50 samples, independent prediction)',
                 'file': 'ADE50_indep',
                 'latex': r'\emph{ADE$_{50, indep}$ [m]}'}
        return names
    
    