import numpy as np
import pandas as pd
from evaluation_template import evaluation_template 

class Oracle_indep(evaluation_template):
    r'''
    The value :math:`F` of Oracle 10 (the Average Displacement Error of the best 10\% of predictions (assuming :math:`N_{agents,i}` independent agents :math:`j`)), is calculated in the following way:
        
    .. math::
        F = {1 \over{|P^*_{i,j}| \sum\limits_{i = 1}^{N_{samples}} N_{agents, i}}} \sum\limits_{i = 1}^{N_{samples}}  
            \sum\limits_{j = 1}^{N_{agents,i}} \sum\limits_{p \in P^*_{i,j}} {1\over{| T_{O,i} | }} \sum\limits_{t \in T_{O,i}} 
            \sqrt{\left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}
            
    Here, for each specific sample :math:`i \in \{1, ..., N_{samples}\}` and agent :math:`j`, :math:`P^*_{i,j} \subset P_{50}` are 
    the 5 values of :math:`p \in P_{50}` (:math:`5` is 10\% of :math:`50 = | P_{50}|`), where the term
    
    .. math::
            {1 \over{| T_{O,i} |}} \sum\limits_{t \in T_{O,i}} \sqrt{\left( x_{i,j}(t) - x_{pred,i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - y_{pred,i,p,j} (t) \right)^2}   
    
    is smallest.
    
    Here, :math:`P_{50} \subset P` are 50 randomly selected instances of the set of predictions :math:`P` made 
    at the predicted timesteps :math:`T_{O,i}`. :math:`x` and :math:`y` are here the actual observed positions, while 
    :math:`x_{pred}` and :math:`y_{pred}` are those predicted by a model.
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths(50)
        Pred_agents = Pred_steps.any(-1)
        Num_steps = Pred_steps.sum(-1).max(-1)
        Num_agents = Pred_agents.sum(-1)
        
        # Get squared distance
        Diff = ((Path_true - Path_pred) ** 2).sum(-1)
        
        # Get absolute distance
        Diff = np.sqrt(Diff)
        
        # Best 5 over predictions
        idx = np.argsort(Diff.sum(-1), axis = 1)
        idx_p = idx[:,:5]
        idx_i = np.tile(np.arange(len(Diff))[:,np.newaxis,np.newaxis], (1,5,Pred_agents.shape[1]))
        idx_j = np.tile(np.arange(Pred_agents.shape[1])[np.newaxis,np.newaxis,:], (len(Diff),5,1))
        Diff = Diff[idx_i, idx_p, idx_j,:]
        
        # Mean over predictions
        Diff = Diff.mean(1)
        
        # Mean over samples and agents
        Step_error = Diff.sum((0,1)) / Pred_steps.sum((0,1))
        
        # Mean over timesteps
        Diff = Diff.sum(-1) / Num_steps[:,np.newaxis]
        
        # Mean over samples and agents
        Error = Diff.sum() / Num_agents.sum()
        
        return [Error, np.concatenate(([0], Step_error)), np.arange(len(Step_error) + 1) * self.data_set.dt]
    
    def partial_calculation(self = None):
        options = ['No', 'Sample', 'Pred_agents']
        return options[2]  
    
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
            fig.savefig(test_file, bbox_inches='tight')  
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'Oracle (independent predictions)',
                 'file': 'Oracle_indep',
                 'latex': r'\emph{Oracle$_{indep}$ [m]}'}
        return names
    
    
    def check_applicability(self):
        return None
    
    def is_log_scale(self = None):
        return True
    
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
    
    def metric_boundaries(self = None):
        return [0.0, None]
