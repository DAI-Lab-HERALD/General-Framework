import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Oracle(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        '''
        Calculates the Oracle 10% for the predicted paths.
        '''
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
            
            diff_idx = np.argsort(diff.mean(axis = 1))
            diff = diff[diff_idx[:5], :].mean(axis = 0)
            
            Error[:len(diff)] += diff
            Samples[:len(diff)] += 1
        E = Error / Samples 
        return [E.mean(), E, np.arange(len(E)) * self.data_set.dt]
    
    def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
        plt_label =  model_class.get_name()['latex']
        ax.plot(np.concatenate(([0], results[2] + results[2][1])), np.concatenate(([0], results[1])), label = plt_label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Oracle 10%')
        ax.set_xlim([0, results[2].max() + results[2][1]])
        max_val = 1.05 * np.array(results[1]).max()
        if hasattr(self, 'max_val'):
            self.max_val = max(self.max_val, max_val)
        else:
            self.max_val = max_val
            
        ax.set_ylim([0, self.max_val])
        fig.set_figwidth(5)
        fig.set_figheight(2.5)
        #ax.set_title('Oracle 10%')
        
        # ax.axis('off')

        if save:
            # ax.legend()
            fig.show()
            num = 16 + len(self.get_name()['file'])
            fig.savefig(test_file[:-num] + 'Oracle_test.pdf', bbox_inches='tight')  
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'Oracle',
                 'file': 'Oracle',
                 'latex': r'\emph{Oracle [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
