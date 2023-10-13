import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template


class Random_split(splitting_template):
    '''
    The easiest form of splitting data into training and testing sets is surely
    a random splitting. As this does not rest on any assumptions, the number
    of repetitions is potentially limitless.
    '''
    def split_data_method(self):
        
        self.data_set._group_indentical_inputs()
        Subgroups = self.data_set.Subgroups - 1
        
        # Get Behaviors, with non appearing ones being neglected
        Behaviors = np.unique(self.data_set.Output_A.to_numpy().argmax(1), return_inverse = True)[1]
        
        # Get unique subgroups
        uni_subgroups, uni_subgroups_samples = np.unique(Subgroups)
        
        # Get number of behaviors for each subgroup
        uni_subgroups_beh = np.zeros((len(uni_subgroups), Behaviors.max() + 1))
        for ind, subgroup in enumerate(uni_subgroups):
            subgroup_beh = Behaviors[Subgroups == subgroup]
            beh_included, beh_num = np.unique(subgroup_beh, return_counts = True)
            
            uni_subgroups_beh[ind, beh_included] = beh_num
        
        desired_beh = uni_subgroups_beh.sum(0) * self.test_part
        
        Test_ind = []
        for rep in self.repetition:
            np.random.seed(rep)
            sort_ind = np.arange(len(uni_subgroups_beh))
            np.random.shuffle(sort_ind)
            
            current_beh = np.zeros_like(desired_beh)
            test_ind = []
            
            for ind in sort_ind:
                current_loss = ((desired_beh - current_beh) ** 2).sum()
                
                test_beh = current_beh + uni_subgroups_beh[ind]
                test_loss = ((desired_beh - test_beh) ** 2).sum()
                
                if test_loss < current_loss:
                    current_beh = test_beh
                    test_ind.append(ind)
                    
            Test_ind.append(np.array(test_ind))
        
        Test_ind = np.unique(np.concatenate(Test_ind), axis = 0)
        
        Index = np.arange(len(self.data_set.Output_A))
        Test_index = []
        for ind in Test_ind:
            subgroup = uni_subgroups[ind]
            Test_index.append(Index[Subgroups == subgroup])
            
        Test_index = np.unique(np.concatenate(Test_index), axis = 0)  
        Train_index_bool = ~np.in1d(Index, Test_index, assume_unique = True)
        Train_index = Index[Train_index_bool]
        
        return Train_index, Test_index
        
    
    def get_name(self):
        rep_str = str(self.repetition)[1:-1]
        names = {'print': 'Random splitting (random seed = ' + rep_str + ')',
                 'file': 'random_split',
                 'latex': r'Random split'}
        return names
        
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        return None
    
    
    def can_process_str_repetition(self = None):
        return False
    
        
        
        
    
        



