import numpy as np
import pandas as pd
import os
from evaluation_template import evaluation_template 
import matplotlib.pyplot as plt
from scipy.special import loggamma

plt.rcParams['text.usetex'] = False

class ECE(evaluation_template):
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self, plot = False):
        p_t = self.Output_A.to_numpy().reshape(-1).astype(float)
        p_pred = self.Output_A_pred.to_numpy().reshape(-1)
        Idx = np.argsort(p_pred)
        
        p_pred = p_pred[Idx]
        p_t = p_t[Idx]

        B_num = np.arange(1, 101, 3)
        
        P_t_adjusted = np.zeros((len(p_pred), len(B_num)), float)
        model_b_log_likelihood = np.zeros(len(B_num))
        
        for b_ind, b_num in enumerate(B_num):
            p_size = 1 / b_num + 1e-4
            for i in range(b_num):
                p_pred_min = p_size * i
                p_pred_max = p_size * (i + 1)
                idx = np.where((p_pred_min <= p_pred) & (p_pred < p_pred_max))[0]
                if len(idx) == 0:
                    continue
                P_t_adjusted[idx,b_ind] = p_t[idx].mean()
                
                # calculate model likelihood
                N_hat = 2
                alpha_b = N_hat * p_pred[idx].mean() / b_num
                beta_b  = N_hat * (1 - p_pred[idx].mean()) / b_num
                
                N_b = len(idx)
                a_b = len(idx) * p_t[idx].mean()
                b_b = len(idx) * (1 - p_t[idx].mean())
                
                L_b = (loggamma(N_hat / b_num + 1e-6) - loggamma(N_b + N_hat / b_num + 1e-6) +
                       loggamma(a_b + alpha_b + 1e-6) - loggamma(alpha_b + 1e-6) +
                       loggamma(b_b + beta_b + 1e-6)  - loggamma(beta_b + 1e-6))
                assert np.isfinite(L_b)
                
                model_b_log_likelihood[b_ind] += L_b
        
        model_b_likelihood = np.exp(model_b_log_likelihood - model_b_log_likelihood.max())
        model_b_likelihood += 0.0033 * model_b_likelihood.max()
        model_b_likelihood /= model_b_likelihood.sum()
        
        p_t_adjusted = (P_t_adjusted * model_b_likelihood[np.newaxis]).sum(axis = 1)
        ece = np.abs(p_pred - p_t_adjusted).mean() 
        assert np.isfinite(ece)
        return [ece, p_pred, p_t_adjusted]
    
    def main_result_idx(self = None):
        return 0
    
    def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
        P = np.unique(np.stack((results[1], results[2])), axis = 1)
        
        plt_label = 'ECE = ' + str(np.round(results[0], 3)) + ' (' + model_class.get_name()['print'] + ')'
        
        ax.plot(P[0], P[1], label = plt_label)
        ax.plot([0,1], [0,1], c = 'black', linestyle = '--')
        ax.set_ylabel('True probability')
        ax.set_xlabel('Predicted probability')
        ax.set_title('ECE')
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        ax.set_aspect('equal', 'box')

        if save:
            ax.legend()
            fig.show()
            num = 16 + len(self.get_name()['file'])
            fig.savefig(test_file[:-num] + 'ECE_test.pdf', bbox_inches='tight')
        
    
    def get_output_type(self = None):
        return 'class' 
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'ECE (Expected Calibration Error on classes)',
                 'file': 'ECE',
                 'latex': r'\emph{ECE}'}
        return names
    
    def requires_preprocessing(self):
        return False
    
    def is_log_scale(self = None):
        return False
    
    def allows_plot(self):
        return True
    
    def check_applicability(self):
        if not self.data_set.classification_useful:
            return 'because a classification metric requires more than one available class.'
        return None