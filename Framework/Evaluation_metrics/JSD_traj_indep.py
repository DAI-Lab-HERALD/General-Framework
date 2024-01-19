import numpy as np
import pandas as pd
from scipy.special import logsumexp
from evaluation_template import evaluation_template 
import os
from matplotlib import cm

class JSD_traj_indep(evaluation_template):
    r'''
    The value :math:`F` of the Jensen-Shannon divergence , 
    is calculated in the following way:
        
    .. math::
        F = {1 \over {|S|\ln(2)}}\sum\limits_{s = 1}^{|S|} D_{JS,s}
    
    Here, :math:`S` is the set of subsets :math:`S_s`, which contain the agents :math:`j`
    for which the initial input to the models was identical. The division by :math:`\ln(2)`
    is used to normalise the output values inbetween 0 (identical distributions) and 1.
    
    Each value :math:`D_{JS,s}` is then calcualted in the following way
    (assuming :math:`N_{agents,i}` independent agents :math:`j` for each sample :math:`i`):
    
    .. math::
        D_{JS,s} & = {1 \over {2}} \left(D_{KL,s} + D_{KL,s,pred}\right) \\
        D_{KL,s} & = {1 \over{|S_s|}} \sum\limits_{(i, j) \in S_s} \ln 
                    \left({ P_{KDE,s} \left(\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)   
                           \over{{1 \over {2}} \left( P_{KDE,pred,s} \left(\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right) 
                                                     + P_{KDE,s} \left(\{\{x_{i,j} (t), y_{i,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)
                                                     \right)}} \right) \\
        D_{KL,s, pred} & = {1 \over{|S_s| |P|}} \sum\limits_{(i, j) \in S_s}\sum\limits_{p \in P} \ln 
                    \left({ P_{KDE,pred,s} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)   
                           \over{{1 \over {2}} \left( P_{KDE,pred,s} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right) 
                                                     + P_{KDE,s} \left(\{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \, | \; \forall t \in T_{O,s}\} \right)
                                                     \right)}} \right)
                    
    Here, :math:`P_{KDE,s}` is a subset and timepoint specific gaussian Kernel Density Estimate trained on all 
    predicted agents in each subset (:math:`(i, j) \in S_s`):
    
    .. math::
        \{\{x_{i,j} (t), y_{i,j} (t)\} \vert \forall t \in T_{O,s}\}
    
    
    while :math:`P_{KDE,pred,s}` is trained on all predictions (:math:`p \in P`) for all predicted agents (:math:`(i, j) \in S_s`):
    
    .. math::
        \{\{x_{pred,i,p,j} (t), y_{pred,i,p,j} (t)\} \vert \forall t \in T_{O,s}\}
    
    '''
    
    def setup_method(self):
        pass
     
    def evaluate_prediction_method(self):
        Path_true, Path_pred, Pred_steps = self.get_true_and_predicted_paths()
        
        _, subgroups = self.get_true_prediction_with_same_input()
        
        # Get likelihood of true samples according to prediction
        KDE_pred_log_prob_true, KDE_pred_log_prob_pred = self.get_KDE_probabilities(joint_agents = False)
        KDE_true_log_prob_true, KDE_true_log_prob_pred = self.get_true_likelihood(joint_agents = False)
        
        # Consider agents separately
        Pred_agents = Pred_steps.any(-1)
        
        # Combine agent and samples separately (differentiate between agents)
        agent_indicator = np.linspace(0.1,0.9, Pred_agents.shape[1])
        subgroups = subgroups[:,np.newaxis] + agent_indicator[np.newaxis,:]
        subgroups = subgroups[Pred_agents]
        
        # Combine agent and sample dimension
        Log_like_predTrue = KDE_pred_log_prob_true.transpose(0,2,1)[Pred_agents] # P(X)
        Log_like_trueTrue = KDE_true_log_prob_true.transpose(0,2,1)[Pred_agents] # Q(X)
        Log_like_predPred = KDE_pred_log_prob_pred.transpose(0,2,1)[Pred_agents] # P(\hat{X})
        Log_like_truePred = KDE_true_log_prob_pred.transpose(0,2,1)[Pred_agents] # Q(\hat{X})
        
        JSD = 0
        unique_subgroups = np.unique(subgroups)
        for subgroup_agent in np.unique(subgroups):
            indices = np.where(subgroup_agent == subgroups)[0]
            
            # Get the likelihood of the the true samples according to the
            # true samples. This reflects the fact that we take the 
            # expected value over the true distribtuion
            log_like_trueTrue = Log_like_trueTrue[indices]
            log_like_predTrue = Log_like_predTrue[indices]
            log_like_truePred = Log_like_truePred[indices]
            log_like_predPred = Log_like_predPred[indices]
            
            log_like_trueComb = np.concatenate((np.tile(log_like_trueTrue, (1, log_like_truePred.shape[1])), 
                                                log_like_truePred), axis = 0)
            log_like_predComb = np.concatenate((np.tile(log_like_predTrue, (1, log_like_truePred.shape[1])), 
                                                log_like_predPred), axis = 0)
            
            log_like_combComb = logsumexp(np.stack([log_like_trueComb, log_like_predComb], axis = 0), axis = 0) - np.log(2)
            # log_like_combTrue = logsumexp(np.stack([log_like_trueTrue, log_like_predTrue], axis = 0), axis = 0) - np.log(2)
            # log_like_combPred = logsumexp(np.stack([log_like_truePred, log_like_predPred], axis = 0), axis = 0) - np.log(2)
            
            # kld_subgroupTrue = np.mean(log_like_trueTrue-log_like_combTrue)
            # kld_subgroupPred = np.mean(log_like_predPred-log_like_combPred)
            
            helpTrue = log_like_trueComb-log_like_combComb
            kld_subgroupTrue = np.mean(np.exp(helpTrue) * helpTrue)
            
            helpPred = log_like_predComb-log_like_combComb
            kld_subgroupPred = np.mean(np.exp(helpPred) * helpPred)
            
            
            JSD += 0.5*kld_subgroupPred + 0.5*kld_subgroupTrue
        
        # Average JSD over subgroups
        JSD /= len(unique_subgroups)
        JSD /= np.log(2)
        
        # prepare to save data
        if len(unique_subgroups) == 1 and Path_true.shape[2] == 1:
            input_path = self.data_set.Input_path.iloc[self.splitter.Test_index[0]]
            useful_agents = [isinstance(p, np.ndarray) for p in input_path]
            
            input_path = np.stack(input_path[useful_agents].to_numpy(), axis = 0)[0]
            
            return [JSD, Path_true, KDE_true_log_prob_true, Path_pred, KDE_pred_log_prob_pred, Pred_steps, input_path]
        else:
            return [JSD]
    
    def create_plot(self, results, test_file, fig, ax, save, model):
        if len(results) > 1:
            # Delete previous model
            ax.clear()
            
            # Get plot boundaries to be constant over all predictions
            Path_true = results[1]
            Path_in   = results[6]
            
            Path_combo = np.concatenate((np.tile(Path_in[np.newaxis, np.newaxis], (len(Path_true),1,1,1)), 
                                         Path_true[:,0]),  axis = -2)
            
            min_bound = Path_combo.min(axis = tuple(np.arange(Path_combo.ndim - 1)))
            max_bound = Path_combo.max(axis = tuple(np.arange(Path_combo.ndim - 1)))
            
            interval = max_bound - min_bound
            
            min_bound = np.floor(min_bound - 0.25 * interval)
            max_bound = np.ceil(max_bound + 0.25 * interval)
            
            x_lim = [min_bound[0], max_bound[0]]
            y_lim = [min_bound[1], max_bound[1]]
            
            # Get number of plottable samples
            max_samples = Path_true.shape[0]
            
            # check if current file has been saved (i.e, the ground truth)
            if not os.path.isfile(test_file):
                # Laad data
                Log_true  = results[2]
                Pred_step = results[5]
                
                # Plot results
                self.plot_results(Path_in, Path_true, Log_true, Pred_step, ax, x_lim, y_lim, max_samples)
                fig.show()
                
                # Save results
                fig.savefig(test_file, bbox_inches='tight') 
                
                # Clear figure
                ax.clear()
            
            # get model specific test file.
            num = 4 + len(self.get_name()['file'])
            model_test_file = test_file[:-num] + model.get_name()['file'] + '--' + self.get_name()['file'] + '.pdf'
            
            # Laad data
            Path_pred = results[3]
            Log_pred  = results[4]
            Pred_step = results[5]
            
            # Plot results
            self.plot_results(Path_in, Path_pred, Log_pred, Pred_step, ax, x_lim, y_lim, max_samples)
            fig.show()
            
            # Save results
            fig.savefig(model_test_file, bbox_inches='tight')
        
    
    def plot_results(self, Path_in, Path_out, Log, Pred_step, ax, x_lim, y_lim, max_samples):
        # Combine samples and predictions
        Path_out = Path_out.reshape(-1, *Path_out.shape[2:])[:,0]
        Log      = Log.reshape(-1, *Log.shape[2:])[:,0]
        
        # Path_in.shape  = n_I x 2
        # Path_out.shape = (n_samples * n_preds) x n_O x 2
        
        # plot input
        ax.plot(Path_in[:,0], Path_in[:,1], linewidth = 1, c = 'k')
        
        # concatenate output
        Path_out = np.concatenate((np.tile(Path_in[np.newaxis,[-1]], (len(Path_out), 1, 1)), 
                                   Path_out), axis = -2)
        
        
        # Get random order
        np.random.seed(0)
        Indices = np.arange(len(Log))
        np.random.shuffle(Indices)
        Indices = Indices[:max_samples]
        
        # Sort by log prob values        
        Log_plot  = Log[Indices]
        Path_plot = Path_out[Indices]
        
        I_sort = np.argsort(Log_plot)
        
        Log_plot  = Log_plot[I_sort]
        Path_plot = Path_plot[I_sort]
        
        # Get colors
        viridis = cm.get_cmap('viridis', 100)
        Log_adj = (Log_plot - 37.5) / 2
        col_val = 1 / (1 + np.exp(-Log_adj))
        
        # Probabil = np.exp(Log - Log.max())
        # Prab_adj = Probabil / Probabil.sum()
        # col_val = 10 * Prab_adj / len(Prab_adj) 
        
        for i, path_out in enumerate(Path_plot):#[:10]:
            col = np.array(viridis(col_val[i]))
            ax.plot(path_out[:,0], path_out[:,1], color = col, linewidth = 0.25, alpha = 0.5)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('$x$ [$m$]')
        ax.set_ylabel('$y$ [$m$]')
        ax.set_title('$\ln (p)$')
        # ax.set_axis_off()
        # ax.set_title('$\ln (p) - {:0.2f}$'.format(Log_plot.mean()))
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_opt_goal(self = None):
        return 'minimize'
    
    def get_name(self = None):
        names = {'print': 'JSD (Trajectories, independent predictions)',
                 'file': 'JSD_traj_indep',
                 'latex': r'\emph{JSD$_{\text{Traj}, indep}$}'}
        return names
    
    
    def is_log_scale(self = None):
        return False
    
    def check_applicability(self):
        if not self.data_set.enforce_num_timesteps_out:
            return "the metric requires that ouput trajectories are fully observed."
        return None
    
    def requires_preprocessing(self):
        return False
    
    def allows_plot(self):
        return True
    
    def metric_boundaries(self = None):
        return [0.0, 1.0]
