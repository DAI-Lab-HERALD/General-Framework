import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class KLD_traj_indep(evaluation_template):
    r'''
    TODO: 
    The value :math:`F` of the Kullback-Leibler divergence (assuming :math:`N_{agents,i}` independent agents :math:`j`), 
    is calculated in the following way:
        
    
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Types = self.get_true_and_predicted_paths(return_types = True)
        
        Path_true_all = self.get_true_prediction_with_same_input()
        
        assert False
        
        Pred_agents = Pred_steps.any(-1)
        
        # Combine agent and sample dimension
        Path_true  = Path_true.transpose(0,2,1,3,4)[Pred_agents]
        Path_pred  = Path_pred.transpose(0,2,1,3,4)[Pred_agents]
        Pred_steps = Pred_steps[Pred_agents]
        Types      = Types[Pred_agents]
        
        Num_steps = Pred_steps.sum(-1)
        
        
        return ...
   
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
            fig.savefig(test_file, bbox_inches='tight')
            
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'KLD (Trajectories, independent predictions)',
                 'file': 'KLD_traj_indep',
                 'latex': r'\emph{KLD$_{\text{Traj}, indep}$}'}
        return names
    
    
    def is_log_scale(self = None):
        return False
    
    def check_applicability(self):
        # TODO: Check if there are enough similar predictions
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
