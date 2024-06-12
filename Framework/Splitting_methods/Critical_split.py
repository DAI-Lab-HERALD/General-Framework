import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Critical_split(splitting_template):
    '''
    In gap acceptance scenarios (see scenario_gap_acceptance.py), one has the ability 
    to roughly rate the difficulty of prediction based on the assumption that 
    unintuitive behavior is more difficult to predict.
    
    One can classify here beahvior to be more unintuitive, the smaller the difference 
    in time between the target vehicle entering the contested space (accepting the gap) 
    and the ego vehicle entering the gap, or the larger a rejected gap is.
    
    However, this method then allows only one splitting repetition.
    '''
    
    def split_data_method(self):
        # Check assumptions
        assert self.repetition == [0,]
        
        Accepted = np.zeros(len(self.Domain), bool)
        T_decision = np.zeros(len(self.Domain), bool)
        for file_index in range(len(self.data_set.Files)):
            data_file = self.data_set.Files[file_index] + '.npy'
            path_file_addition = self.data_set.Files[file_index][-8:-4]
            
            used = self.Domain.file_index == file_index
            used_index = np.where(used)[0]
            
            # Get Output_A and Output_T_E
            [_, _, _, _, _, _, Output_A, Output_T_E, _] = np.load(data_file, allow_pickle=True)
            
            # Get the actual indices of Output_A
            ind_saved = self.Domain[used].Index_saved
            Output_A   = Output_A.loc[ind_saved]
            Output_T_E = Output_T_E[ind_saved]
            
            # Get accepted/rejected indices
            accepted = Output_A.accepted
            rejected = Output_A.rejected
            used_index_accepted = used_index[accepted]
            used_index_rejected = used_index[rejected]
            
            # Get the decision
            Accepted[used_index] = accepted
            
            # Get the gap siye of rejected gaps
            T_decision[used_index_rejected] = Output_T_E[rejected]
            
            # Get gap size at T_A for the accepted gap
            Domain_accepted = self.Domain.iloc[used_index_accepted]
            Path_id_accepted = Domain_accepted.Path_ID.to_numpy()
            
            # Get scenario name
            scenario_name = np.unique(Domain_accepted.Scenario)
            assert len(scenario_name) == 1, 'Scenario should be the same for a single loaded file'
            scenario_name = scenario_name[0]
            
            # Get path_addition
            time_file = data_set.file_path + '--all_time_points' + path_file_addition + '.npy'
            assert os.path.isfile(time_file), "Something went wrong during dataset extraction."
            [local_id, local_t, _, _, local_T_D_class, _, _, local_t_decision, _, _] = np.load(time_file, allow_pickle=True) 
            
            # find the data_set in self.data_set.Datasets
            for data_set in self.data_set.Datasets.values():
                if scenario_name.startswith(data_set.get_name()['print']):
                    # Check for potential perturbation
                    if ('(Pertubation_' in scenario_name) == data_set.is_perturbed:
                        # Correct data_set was found
                        break 
            
            # Accepted: gap size at acceptance
            for i in range(len(used_index_accepted)):
                i_old = np.searchsorted(local_id, Path_id_accepted[i])
                
                # get relevant data
                ta = local_t_decision[i_old]
                t = local_t[i_old]
                tcpre = local_T_D_class[i_old] + t
                
                # find gap size at ta
                dta = t - ta
                ind = np.where(dta <= 0)[0][-1]
                tcpre_ta = (tcpre[ind] * dta[ind + 1] - tcpre[ind + 1] * dta[ind]) / (dta[ind + 1] - dta[ind])
                
                assert tcpre_ta > ta, "Gap should be open when accepted"
                T_decision[used_index_accepted[i]] = tcpre_ta - ta
            
        # For accepted select the shortest T_decision for each unique scenario
        # For rejected, select the longest T_decision for each unique scenario
        Scenario_ind = np.unique(self.Domain['Scenario'], return_inverse = True)[1]
        Index = np.arange(len(self.Domain))
        
        Train_index = []
        Test_index = []
        for i in np.unique(Scenario_ind):
            index = Index[i == Scenario_ind]
            accepted = Accepted[index]
            t_decision = T_decision[index]
            
            
            index_accepted = index[accepted]
            index_rejected = index[~accepted]
            
            dt_accepted = t_decision[accepted]
            tc_rejected = t_decision[~accepted]
            
            # Get the longest rejected gaps
            num_test_rejected = int(self.test_part * len(index_rejected))
            sorted_rejected = np.argsort(-tc_rejected) 
            index_rejected_sorted = index_rejected[sorted_rejected]
            Test_index.append(index_rejected_sorted[:num_test_rejected])
            Train_index.append(index_rejected_sorted[num_test_rejected:])
            
            # Get the shortest accepted gaps
            num_test_accepted = int(self.test_part * len(index_accepted))
            sorted_accepted = np.argsort(dt_accepted)
            index_accepted_sorted = index_accepted[sorted_accepted]
            Test_index.append(index_accepted_sorted[:num_test_accepted])
            Train_index.append(index_accepted_sorted[num_test_accepted:])
            
        # Transform list to array
        Train_index = np.array(Train_index)
        Test_index  = np.array(Test_index)
        
        return Train_index, Test_index
        
    def get_name(self):
        names = {'print': 'Critical splitting',
                 'file': 'critic_split',
                 'latex': r'Critical split'}
        return names
    
    def check_splitability_method(self):
        if self.data_set.scenario_name != 'Gap acceptance problem':
            return 'this splitting method can only work on gap acceptance problems.'
        else:
            return None
    
    
    def repetition_number(self):
        return 1
    
    
    def can_process_str_repetition(self = None):
        return False


