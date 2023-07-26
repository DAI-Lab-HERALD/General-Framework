import numpy as np
from ADE_indep import ADE_indep 

class ADE20_indep(ADE_indep):
     
    def evaluate_prediction_method(self):
        num_samples_needed = 20
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
        names = {'print': 'ADE (20 samples, independent prediction)',
                 'file': 'ADE20_indep',
                 'latex': r'\emph{ADE$_{20, indep}$ [m]}'}
        return names
    
    