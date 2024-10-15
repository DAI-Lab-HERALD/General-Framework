import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class ECE_traj_indep(evaluation_template):
    r'''
    The value :math:`F` of the Trajectory Expected Calibration Error (assuming :math:`N_{agents,i}` independent agents :math:`j`), is calculated in the following way:
    
    .. math::
        F = {1\over{201}} \sum\limits_{k = 0}^{200} \left| \left({1\over{\sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \left| \left\{i,j \, | \, m_{i,j} > {k\over{200}}  \right\} \right| \right) + {k\over{200}} - 1 \right|
    
    Here, :math:`m_{i,j}` is calculated for each sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j` with
    
    .. math::
        m_{i,j} = {1\over{|P|}} \left|\left\{ p \in P \, | \, L_{pred,i,p,j} > L_{i,j} \right\} \right|,
        
    where the true and predicted likelihoods for each prediction :math:`p \in P`
    
    .. math::
        & L_{i,j} & = P_{KDE,i,j} \left(\{\{x_{i,j} (t), y_{i,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \right) \\
        & L_{pred,i,p,j} & = P_{KDE,i,j} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\} \right)
    
    
    are calculated using a sample and agent specific gaussian Kernel Density Estimate :math:`P_{KDE,i,j}` trained on 
    all predictions (:math:`p \in P`) for sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`
    
    .. math::
        \{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t) \} \, | \; \forall\, t \in T_{O,i}\}
    
    For each prediction timestep in :math:`T_{O,i}`, :math:`x` and :math:`y` are the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    def set_default_kwargs(self):
        if 'include_pov' not in self.metric_kwargs:
            self.metric_kwargs['include_pov'] = True

    def setup_method(self):
        self.set_default_kwargs()
     
    def evaluate_prediction_method(self):
        # Get predicted agents
        _, _, Pred_steps = self.get_true_and_predicted_paths()
        Pred_agents = Pred_steps.any(-1)
        num_agents = Pred_agents.shape[1]
        
        # Get identical input subgroups
        _, subgroups = self.get_true_prediction_with_same_input()
        
        # Get likelihood of having higher probability
        KDE_log_prob_true, KDE_log_prob_pred = self.get_KDE_probabilities(joint_agents = False)
        
        M = np.zeros(KDE_log_prob_true.shape)[:,0]
        
        unique_subgroups = np.unique(subgroups)
        
        for subgroup in unique_subgroups:
            indices = np.where(subgroup == subgroups)[0]
            
            LP_pred = KDE_log_prob_pred[indices].reshape(1, -1, num_agents) 
            LP_true = KDE_log_prob_true[indices]
            
            M[indices] = (LP_true < LP_pred).mean(1)
        
        # Align all agents from all samples
        M = M[Pred_agents]
        
        # Compare to expectation
        T = np.linspace(0,1,201)
        
        # Mean over predicted agents
        ECE = (M[np.newaxis] > T[:,np.newaxis]).mean(-1)
        
        ece = np.abs(ECE - (1 - T)).mean()
        
        # shape of the different results
        # results[1].shape = (201,)
        # results[2].shape = (201,)
        return [ece, T, ECE]
    
    def combine_results(self, result_lists, weights):
        # Get combined ECE values
        ECE = []
        for i in range(len(result_lists)):
            ECE.append(result_lists[i][2])
        
        ECE = np.stack(ECE, axis = 0)

        ECE = np.average(ECE, axis = 0, weights = np.array(weights))

        # Get combined T values
        T = result_lists[0][1]

        # ece value
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
            fig.savefig(test_file, bbox_inches='tight')
    
    def partial_calculation(self = None):
        return 'Sample'
    
    def get_output_type(self = None):
        self.set_default_kwargs()
        if self.metric_kwargs['include_pov']:
            return 'path_all_wi_pov'
        else:
            return 'path_all_wo_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        self.set_default_kwargs()
        if self.metric_kwargs['include_pov']:
            P_p = ''
            P_f = ''
            P_l = ''
        else:
            P_p = ', exclude POV'
            P_f = 'nP'
            P_l = 'nP, '
            
        names = {'print': 'ECE (Trajectories, independent prediction' + P_p + ')',
                 'file': 'ECE_traj_indep' + P_f,
                 'latex': r'\emph{ECE$_{' + P_l + r'\text{Traj}, indep}$}'}
        return names
    
    
    def is_log_scale(self = None):
        return False
    
    def check_applicability(self):
        return None
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
    
    def metric_boundaries(self = None):
        return [0.0, 0.5]
