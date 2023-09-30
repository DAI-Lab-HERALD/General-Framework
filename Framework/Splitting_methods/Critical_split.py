import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Critical_split(splitting_template):
    '''
    In gap acceptance scenarios, one has the ability to roughly rate the difficulty 
    of prediction based on the assumption that unintuitive behavior is more
    difficult to predict.
    
    One can classify here beahvior to be more unintuitive, the smaller the difference 
    in time between the target vehicle entering the contested space (accepting the gap) 
    and the ego vehicle entering the gap, or the larger a rejected gap is.
    
    However, this method then allows only one splitting repetition.
    '''
    
    def split_data_method(self):
        # Check assumptions
        assert self.repetition == [0,]
        
        # Get possible indices
        Index = np.arange(len(self.data_set.Output_A))
        
        # TODO: Redo all this.
        Train_index = []
        Test_index = []
        
        for data_set_name in self.data_set.Datasets:
            data_set = self.data_set.Datasets[data_set_name]
            
            Ind_data_set = np.where(self.Domain['Scenario'] == data_set.get_name()['print'])[0]
        
            # Test: longest reejcted gaps
            data_set.extract_time_points()
            
            Index = np.arange(len(data_set.Output_A))
            Index_accepted = Index[data_set.Output_A.accepted]
            Index_rejected = Index[data_set.Output_A.rejected]
            
            num_test_accepted = int(self.test_part * len(Index_accepted))
            num_test_rejected = int(self.test_part * len(Index_rejected))
        
            Tc_rejected = data_set.Output_T_E[Index_rejected]
            Sorted_rejected = np.argsort(- Tc_rejected) 
            Index_rejected_test = Index_rejected[Sorted_rejected][:num_test_rejected]
            Index_rejected_train = Index_rejected[Sorted_rejected][num_test_rejected:]
            
            # Accepted: shortest gap size at acceptance
            Path_id_accepted = data_set.Domain.iloc[Index_accepted].Path_ID.to_numpy()
            DT_accepted = np.zeros(len(Index_accepted))
            for i in range(len(Index_accepted)):
                i_old = np.searchsorted(data_set.id, Path_id_accepted[i])
                
                # get relevant data
                ta = data_set.t_decision[i_old]
                t = data_set.t[i_old]
                tcpre = data_set.T_D_class.rejected[i_old] + t
                
                # find gap size at ta
                dta = t - ta
                ind = np.where(dta <= 0)[0][-1]
                tcpre_ta = (tcpre[ind] * dta[ind + 1] - tcpre[ind + 1] * dta[ind]) / (dta[ind + 1] - dta[ind])
                
                assert tcpre_ta > ta, "Gap should be open when accepted"
                DT_accepted[i] = tcpre_ta - ta
            
            Sorted_accepted = np.argsort(DT_accepted)
            Index_accepted_test = Index_accepted[Sorted_accepted][:num_test_accepted]
            Index_accepted_train = Index_accepted[Sorted_accepted][num_test_accepted:]
            
            Train_index.append(Ind_data_set[Index_accepted_train])
            Train_index.append(Ind_data_set[Index_rejected_train])
            
            Test_index.append(Ind_data_set[Index_accepted_test])
            Test_index.append(Ind_data_set[Index_rejected_test])
        
        Train_index = np.concatenate(Train_index, axis = 0)
        Test_index  = np.concatenate(Test_index, axis = 0)
        
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


