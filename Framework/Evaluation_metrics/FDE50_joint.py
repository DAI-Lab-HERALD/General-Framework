import numpy as np
from FDE_joint import FDE_joint 

class FDE50_joint(FDE_joint):
     
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
            
            # mean over predicted samples
            diff = diff.mean(0)
            diff = diff[-1]

            Error += diff
        
        E = Error / len(self.Output_path)
        return [E]
    
    
    def get_name(self = None):
        names = {'print': 'FDE (50 samples, joint prediction)',
                 'file': 'FDE50_joint',
                 'latex': r'\emph{FDE$_{50, joint}$ [m]}'}
        return names
    
    