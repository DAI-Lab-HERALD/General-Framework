import numpy as np
import pandas as pd
import os
from evaluation_template import evaluation_template 
import matplotlib.pyplot as plt
from scipy.special import loggamma

plt.rcParams['text.usetex'] = False

class ECE_class(evaluation_template):
    r'''
    The value :math:`F` of the Expected Calibration Error is calculated in the following way:
        
    .. math::
        F = {1 \over{ N_{samples} N_{classes}}} \sum\limits_{i = 1}^{N_{samples}}  \sum\limits_{k} 
            \left| \widehat{p}_{i,k} - p_{pred,i,k} \right|
               
    Here, for a specific sample :math:`i \in \{1, ..., N_{samples}\}` and 
    classifcation :math:`k \in \{1, ..., N_{classes}\}`, :math:`p` is the actually observed and :math:`p_{pred}` 
    the predicted probability for a classification to be observed. The sum of all these values over :math:`k` should always 
    be equal to :math:`1`.
    
    Meanwhile, the adjusted true probability :math:`\widehat{p}_{i,k}` is an weighted average of a bin probability 
    :math:`\widetilde{p}_{i,k, b}`, where for calibration model :math:`M_b` with bin number 
    :math:`N_b \in B = \{\lfloor 0.1 \, (N_{samples}N_{classes})^{1\over{3}} \rfloor, ..., \lfloor 10 \,  (N_{samples}N_{classes})^{1\over{3}} \rfloor\}`, 
    a likelihood :math:`L_{b}` is assigned:
        
    .. math::
        \widehat{p}_{i,k} = {\sum\limits_{b \in B} L_{b} \widetilde{p}_{i,k, b} \over {\sum\limits_{b \in B} L_{b}}} 
        
        
    Given a number :math:`N_b`, a calibration model :math:`M_b` is built by firstly setting the bin size
    
    .. math::
        s_b = {N_{samples} N_{classes} \over{N_b}}
        
    Then, we sort the predicted probabilites (using the bijective mapping :math:`\mathcal{S}: i,k \rightarrow j`) so that 
    
    .. math::
        j_1 > j_2 \Rightarrow  p_{pred,i_1,k_1} = p_{pred, \mathcal{S}^{-1}(j_1)} \geq p_{pred,i_2, k_2} = p_{pred, \mathcal{S}^{-1}(j_2)}
        
    For the model :math:`M_b`, one can then set a number of bins, where the bin :math:`J_{b, n_b}` with
    :math:`n_b \in \{1, ..., N_b\}` will be the following:
        
    .. math::
        J_{b, n_b} = \left\{ \lfloor s_b (n_b - 1) \rfloor + 1, ..., \lfloor s_b n_b \rfloor + 1  \right\}
        
    with 
    
    .. math::
        &\widehat{\widetilde{p}}_{b,n_b} & = {1\over{| J_{b,n_b} |}} \sum\limits_{j \in J_{b,n_b}} p_{\mathcal{S}^{-1}(j)} \\
        &\widehat{\widetilde{p}}_{pred, b,n_b} & = {1\over{| J_{b,n_b} |}} \sum\limits_{j \in J_{b,n_b}} p_{pred, \mathcal{S}^{-1}(j)}     
        
    One then can get with :math:`\mathcal{S}(i,k) \in J_{b,l}` the searched for value
    
    .. math:: 
        \widetilde{p}_{i,k, b} = \widehat{\widetilde{p}}_{b,l}
        
    The likelihood :math:`L_{b}` can be computed in the following way:
        
    .. math::
        L_{b} = \prod\limits_{n_b = 1}^{N_b} 
        {\Gamma \left({2\over{N_b}}\right) \over{\Gamma \left(| J_{b,n_b} | + {2\over{N_b}}\right)}} 
        {\Gamma \left(| J_{b,n_b} |  \widehat{\widetilde{p}}_{b,l} + {2\over{N_b}} \widehat{\widetilde{p}}_{pred, b,n_b}\right) \over{\Gamma \left({2\over{N_b}} \widehat{\widetilde{p}}_{pred, b,n_b}\right)}}
        {\Gamma \left(| J_{b,n_b} |  (1 - \widehat{\widetilde{p}}_{b,l}) + {2\over{N_b}}(1-  \widehat{\widetilde{p}}_{pred, b,n_b})\right) \over{\Gamma \left({2\over{N_b}} (1 -  \widehat{\widetilde{p}}_{pred, b,n_b})\right)}}
        
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self, plot = False):
        P_true, P_pred, _ = self.get_true_and_predicted_class_probabilities()
        
        # Sort probabilities
        P_true = P_true.reshape(-1)
        P_pred = P_pred.reshape(-1)
        
        Idx = np.argsort(P_pred)
        
        P_pred = P_pred[Idx]
        P_true = P_true[Idx]
        
        # Average the true p_t values over p_pred of same value
        P_true_similar = np.zeros_like(P_true)
        for p in np.unique(P_pred):
            use = p == P_pred
            P_true_similar[use] = P_true[use].mean()
        
        
        N = len(P_pred)
        C = 10
        
        B_num = np.arange(int(N ** (1/3) / C), int(N ** (1/3) * C))
        P_tilde = np.zeros((len(P_pred), len(B_num)), float)
        model_b_log_likelihood = np.zeros(len(B_num))
        
        N_hat = 2
        
        for b_ind, B in enumerate(B_num):
            b_size = N / B
            for i in range(B):
                idx = np.arange(int(b_size * i), int(b_size * (i + 1)))
                
                bin_p_true = P_true_similar[idx].mean()
                p_b = P_pred[idx].mean()
                
                P_tilde[idx, b_ind] = bin_p_true
                
                # calculate model likelihood
                alpha_b = N_hat * p_b / B
                beta_b  = N_hat * (1 - p_b) / B
                
                N_b = len(idx)
                m_b = len(idx) * bin_p_true
                n_b = len(idx) * (1 - bin_p_true)
                
                L_b = (loggamma(N_hat / B + 1e-6) - loggamma(N_b + N_hat / B + 1e-6) +
                       loggamma(m_b + alpha_b + 1e-6) - loggamma(alpha_b + 1e-6) +
                       loggamma(n_b + beta_b + 1e-6)  - loggamma(beta_b + 1e-6))
                assert np.isfinite(L_b)
                
                model_b_log_likelihood[b_ind] += L_b
        
        model_b_likelihood = np.exp(model_b_log_likelihood - model_b_log_likelihood.max())
        model_b_likelihood /= model_b_likelihood.sum()
        
        P_hat = (P_tilde * model_b_likelihood[np.newaxis]).sum(axis = 1)
        ece = np.abs(P_pred - P_hat).mean() 
        assert np.isfinite(ece)
        return [ece, P_pred, P_hat]
    
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
