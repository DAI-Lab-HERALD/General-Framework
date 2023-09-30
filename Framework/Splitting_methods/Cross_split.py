import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template


class Cross_split(splitting_template):
    '''
    This splitting method implements one of the standard methods of evaluating
    prediction model, themethod of crossvalidation. Here, the dataset is split
    into a number of roughly similar sized partitions, which take turns as the
    testing set, while the respective other partitions are used as the 
    training set. 
    
    The number of possible repetitions depends on the chosen size of each partitition,
    with smaller test sets allowing for more repetitions.
    '''
    def split_data_method(self):
        num_splits = int(np.ceil(1 / self.test_part))
        test_parts = 1 / num_splits

        Index = np.arange(len(self.data_set.Output_A))
        
        Index_test = []
        for beh in self.data_set.Behaviors:
            Index_beh = Index[self.data_set.Output_A[beh]]
            np.random.seed(0)
            np.random.shuffle(Index_beh)
            
            roll_value = int(test_parts * len(Index_beh))
            
            for rep in self.repetition:
                roll_value_rep = roll_value * rep
                Index_beh_rolled = np.roll(Index_beh, roll_value_rep)
                Index_test.append(Index_beh_rolled[:roll_value])
        
        Test_index = np.concatenate(Index_test, axis = 0)
        Test_index = np.unique(Test_index)
        
        Train_index_bool = ~np.in1d(Index, Test_index, assume_unique = True)
        Train_index = Index[Train_index_bool]
        
        return Train_index, Test_index
    
    def get_name(self):
        num_splits = int(np.ceil(1 / self.test_part))
        rep_str = str(self.repetition)[1:-1]
        names = {'print': '{} fold Cross validation (Splits '.format(num_splits) + rep_str + ')',
                 'file': 'crossv_split',
                 'latex': r'CV - Fold ' + rep_str}
        return names
        
    def check_splitability_method(self):
        return None
    
    
    def repetition_number(self):
        num_splits = int(np.ceil(1 / self.test_part))
        return num_splits
    
    
    def can_process_str_repetition(self = None):
        return False
        
        
        
    
        



