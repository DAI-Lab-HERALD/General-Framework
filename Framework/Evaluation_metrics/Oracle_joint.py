import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Oracle_joint(evaluation_template):
    r'''
    The value :math:`F` of Oracle 10 (the Average Displacement Error of the best 10\% of predictions (assuming :math:`N_{agents}` jointly predicted agents :math:`j`)), is calculated in the following way:
        
    .. math::
        F = {1 \over{N_{samples} | T_O | |P^*_{i}|}} \sum\limits_{i = 1}^{N_{samples}}  \sum\limits_{p \in P^*_{i}} 
            \sum\limits_{t \in T_O}\sqrt{{1\over{N_{agents}}} \sum\limits_{j = 1}^{N_{agents}} 
            \left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
            
    Here, for each specific sample :math:`i \in \{1, ..., N_{samples}\}`, :math:`P^*_{i} \subset P_{50}` are 
    the 5 values of :math:`p \in P_{50}` (:math:`5` is 10\% of :math:`50 = | P_{50}|`), where the term
    
    .. math::
            {1 \over{| T_O |}} 
            \sum\limits_{t \in T_O}\sqrt{{1\over{N_{agents}}} \sum\limits_{j = 1}^{N_{agents}} 
            \left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}  
    
    is smallest.
    
    Here, :math:`P_{50} \subset P` are 50 randomly selected instances of the set of predictions :math:`P` made 
    at the predicted timesteps :math:`T_O`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        '''
        Calculates the Oracle 10% for the predicted paths.
        '''
        num_samples_needed = 50
        num_samples = len(self.Output_path_pred.iloc[0,0])
        if num_samples >= num_samples_needed:
            idx_l = np.random.permutation(num_samples)[:num_samples_needed]#
        else:
            idx_l = np.random.randint(0, num_samples, num_samples_needed)
            
        nto = self.data_set.num_timesteps_out_real
        
        Error = np.zeros(nto)
        Samples = np.zeros(nto, int)
        
        for i_sample in range(len(self.Output_path_pred)):
            # sample_pred.shape = num_path x num_agents x num_timesteps_out x 2
            sample_pred = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 1)[idx_l,:,:nto]
            sample_true = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[np.newaxis,:,:nto]
            
            diff = (sample_pred - sample_true) ** 2
            # sum over dimension and mean over number agents
            diff = diff.sum(3).mean(1)
            diff = np.sqrt(diff)
            
            # Find best five values for each sample
            idx = np.argsort(diff.mean(axis = 1))
            diff = diff[idx[:5], :]
            
            # Mean over 5 best samples
            diff = diff.mean((0))
            
            Error[:len(diff)] += diff
            Samples[:len(diff)] += 1
        E = Error / Samples 
        return [E.mean(), np.concatenate(([0], E)), np.arange(len(E) + 1) * self.data_set.dt]
    
    def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
        plt_label = model_class.get_name()['latex']
        ax.plot(results[2], results[1], label = plt_label)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Oracle 10% [m]')
        
        max_val = 1.05 * np.array(results[1]).max()
        ax.set_ylim([0, max_val])
        ax.set_xlim([0, results[2].max()])
        
        fig.set_figwidth(5)
        fig.set_figheight(2.5)
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
        names = {'print': 'Oracle (joint predictions)',
                 'file': 'Oracle_joint',
                 'latex': r'\emph{Oracle$_{joint}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
