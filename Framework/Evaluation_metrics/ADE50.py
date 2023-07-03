import numpy as np
from ADE import ADE 

class ADE50(ADE):
     
    def evaluate_prediction_method(self):
        num_samples_needed = 50
        num_samples = len(self.Output_path_pred.iloc[0,0])
        if num_samples >= num_samples_needed:
            idx = np.random.permutation(num_samples)[:num_samples_needed]#
        else:
            idx = np.random.randint(0, num_samples, num_samples_needed)
            
        nto = self.data_set.num_timesteps_out_real
        
        Error = np.zeros(nto)
        Samples = np.zeros(nto, int)
        
        for i_sample in range(len(self.Output_path_pred)):
            # sample_pred.shape = num_path x num_timesteps_out x 2 x num_agents
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = -1)[idx,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = -1)[np.newaxis,:nto]
            
            diff = (sample_pred - sample_true) ** 2
            # sum over dimension and number agents
            diff = diff.sum((2,3))
            diff = np.sqrt(diff)
            # mean over predicted samples
            diff = diff.mean(0)

            Error[:len(diff)] += diff
            Samples[:len(diff)] += 1
        
        E = Error / Samples 
        return [E.mean()]
    
    
    def get_name(self = None):
        names = {'print': 'ADE50',
                 'file': 'ADE50',
                 'latex': r'\emph{ADE$_{50}$ [m]}'}
        return names
    
    