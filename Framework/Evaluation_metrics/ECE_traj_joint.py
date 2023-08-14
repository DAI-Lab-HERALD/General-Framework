import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 
from sklearn.neighbors import KernelDensity

class ECE_traj_joint(evaluation_template):
    r'''
    The value :math:`F` of the Trajectory Expected Calibration Error (assuming :math:`N_{agents,i}` jointly predicted agents :math:`j`), is calculated in the following way:
    
    .. math::
        F = {1\over{201}} \sum\limits_{k = 0}^{200} \left| \left({1\over{N_{samples}}} \left| \left\{i \, | \, m_{i} > {k\over{200}}  \right\} \right| \right) + {k\over{200}} - 1 \right|
    
    Here, :math:`m_{i}` is calculated for each sample :math:`i \in \{1, ..., N_{samples}\}` with
    
    .. math::
        m_{i} = {1\over{|P|}} \left|\left\{ p \in P \, | \, L_{pred,i,p} > L_{i} \right\} \right|,
        
    where the true and predicted likelihoods for each prediction :math:`p \in P`
    
    .. math::
        & L_{i} & = P_{KDE,i} \left(\{\{\{x_{i,j} (t), y_{i,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \, | \; \forall \, j \}\right) \\
        & L_{pred,i,p} & = P_{KDE,i} \left(\{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \, | \; \forall \, j \}\right)
    
    
    are calculated using a sample specific gaussian Kernel Density Estimate :math:`P_{KDE,i}` trained on 
    all predictions (:math:`p \in P`) for sample :math:`i \in \{1, ..., N_{samples}\}`
    
    .. math::
        \{\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \, | \; \forall \, j \}
    
    For each prediction timestep in :math:`T_{O,i}`, :math:`x` and :math:`y` are the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps, Types = self.get_true_and_predicted_paths(return_types = True)
        Pred_agents = Pred_steps.any(-1) 
        Num_agents = Pred_agents.sum(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        
        M = []
        
        for i_sample in range(len(self.Output_path_pred)):
            std = 1 + (Types[i_sample, Pred_agents[i_sample]] != 'P') * 79
            std = std[np.newaxis, :, np.newaxis, np.newaxis]
            
            nto = Num_steps[i_sample]
            n_agents = Num_agents[i_sample]
            
            path_true = Path_true[i_sample,:,Pred_agents[i_sample],:nto]
            path_pred = Path_pred[i_sample,:,Pred_agents[i_sample],:nto]
            
            path_true_comp = (path_true / std).reshape(-1, n_agents * nto * 2)
            path_pred_comp = (path_pred / std).reshape(-1, n_agents * nto * 2)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(path_pred_comp)
                
            log_prob_pred = kde.score_samples(path_pred_comp)
            log_prob_true = kde.score_samples(path_true_comp)
            
            M.append((log_prob_pred > log_prob_true).mean())
        
        M = np.array(M)
        
        T = np.linspace(0,1,201)
        
        # Mean over samples
        ECE = (M[np.newaxis] > T[:,np.newaxis]).mean(-1)
        
        ece = np.abs(ECE - (1 - T)).mean()
        
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
        names = {'print': 'ECE (Trajectories, joint predictions)',
                 'file': 'ECE_traj_joint',
                 'latex': r'\emph{ECE$_{\text{Traj}, joint}$}'}
        return names
    
    
    def is_log_scale(self = None):
        return False
    
    def check_applicability(self):
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
