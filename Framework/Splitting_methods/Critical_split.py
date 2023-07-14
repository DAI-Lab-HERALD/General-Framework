import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Critical_split(splitting_template):
    def split_data_method(self):
        Index = np.arange(len(self.data_set.Output_A))
        Index_accepted = Index[self.data_set.Output_A.accepted]
        Index_rejected = Index[self.data_set.Output_A.rejected]
        
        num_test_accepted = int(self.test_part * len(Index_accepted))
        num_test_rejected = int(self.test_part * len(Index_rejected))
        
        # Test: longest reejcted gaps
        self.data_set.extract_time_points()
        
        Tc_rejected = self.data_set.Output_T_E[Index_rejected]
        Sorted_rejected = np.argsort(- Tc_rejected) 
        Index_rejected_test = Index_rejected[Sorted_rejected][:num_test_rejected]
        Index_rejected_train = Index_rejected[Sorted_rejected][num_test_rejected:]
        
        # Accepted: shortest gap size at acceptance
        Path_id_accepted = self.data_set.Domain.iloc[Index_accepted].Path_ID.to_numpy()
        DT_accepted = np.zeros(len(Index_accepted))
        for i in range(len(Index_accepted)):
            i_old = np.searchsorted(self.data_set.id, Path_id_accepted[i])
            
            # get relevant data
            ta = self.data_set.t_decision[i_old]
            t = self.data_set.t[i_old]
            tcpre = self.data_set.T_D_class.rejected[i_old] + t
            
            # find gap size at ta
            dta = t - ta
            ind = np.where(dta <= 0)[0][-1]
            tcpre_ta = (tcpre[ind] * dta[ind + 1] - tcpre[ind + 1] * dta[ind]) / (dta[ind + 1] - dta[ind])
            
            assert tcpre_ta > ta, "Gap should be open when accepted"
            DT_accepted[i] = tcpre_ta - ta
        
        Sorted_accepted = np.argsort(DT_accepted)
        Index_accepted_test = Index_accepted[Sorted_accepted][:num_test_accepted]
        Index_accepted_train = Index_accepted[Sorted_accepted][num_test_accepted:]
        
        self.Train_index = np.sort(np.concatenate((Index_accepted_train, Index_rejected_train), axis = 0))
        self.Test_index  = np.sort(np.concatenate((Index_accepted_test, Index_rejected_test), axis = 0))
        
    def get_name(self):
        names = {'print': 'Critical splitting',
                 'file': 'critic_split',
                 'latex': r'Critical split'}
        return names
    
    def check_splitability_method(self):
        if self.data_set.scenario.get_name() != 'Gap acceptance problem':
            return 'this splitting method can only work on gap acceptance problems.'
        else:
            return None
    
    
    def repetition_number(self):
        return 1
    
        



