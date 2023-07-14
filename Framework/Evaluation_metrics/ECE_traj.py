import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class ECE_traj(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        
        M = []
        
        nto = self.data_set.num_timesteps_out_real
        
        for i_sample in range(len(self.Output_path_pred)):
            if np.mod(i_sample, 100) == 0:
                print('Sample {}/{}'.format(i_sample + 1, len(self.Output_path_pred)))

            std = 1 + (np.array(self.Type.iloc[i_sample]) == 'V') * 79
            std = std[:,np.newaxis, np.newaxis, np.newaxis]
            
            
            # samples.shape = num_path x num_timesteps_out x 2 x num_agents
            samples = np.stack(self.Output_path_pred.iloc[i_sample].to_numpy(), axis = 0)[:,:,:nto]
            samples_comp = (samples / std).reshape(samples.shape[0], samples.shape[1], -1)
            
            sample_gt = np.stack(self.Output_path.iloc[i_sample].to_numpy(), axis = 0)[:,np.newaxis,:nto]
            sample_gt_comp = (sample_gt / std).reshape(sample_gt.shape[0], sample_gt.shape[1], -1)
            
            for i_agent in range(len(samples)):
                kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(samples_comp[i_agent])
                    
                log_prob_other = kde.score_samples(samples_comp[i_agent])
                log_prob_true  = kde.score_samples(sample_gt_comp[i_agent])[0]
                
                M.append((log_prob_other > log_prob_true).mean())
        
        M = np.array(M)
        
        T = np.linspace(0,1,201)
        
        ECE = (M[np.newaxis] > T[:,np.newaxis]).mean(-1)
        
        ece = np.abs(ECE - (1 - T)).sum() / (len(T) - 1)
        
        return [ece, T, ECE]
   
    def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
        idx = np.argsort(results[1])
        plt_label = 'ECE = {:0.2f} ('.format(results[0]) + model_class.get_name()['latex'] + ')'
        
        ax.plot(results[1][idx], results[2][idx], label = plt_label, linewidth=2)
        ax.plot([0,1], [1,0], c = 'black', linestyle = '--')
        ax.set_xlabel('Threshold T')
        ax.set_ylabel('Probability')
        ax.set_ylim([-0.01,1.01])
        ax.set_xlim([-0.01,1.01])
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        if save:
            fig.set_figwidth(3.5)
            fig.set_figheight(3.5)
            ax.legend(loc='lower left')
            fig.show()
            num = 16 + len(self.get_name()['file'])
            fig.savefig(test_file[:-num] + 'ECE_traj_test.pdf', bbox_inches='tight')
            
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'ECE (Trajectories)',
                 'file': 'ECE_traj',
                 'latex': r'\emph{ECE$_{\text{Traj}}$}'}
        return names
    
    
    def is_log_scale(self = None):
        return False
    
    def check_applicability(self):
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
