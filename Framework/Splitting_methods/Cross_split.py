import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template


class Cross_split(splitting_template):
    def split_data_method(self):
        np.random.seed(0)
        
        num_splits = int(np.ceil(1 / self.test_part))
        test_parts = 1 / num_splits

        Index = np.arange(len(self.data_set.Output_A))
        Index_train = []
        Index_test = []
        for beh in self.data_set.Behaviors:
            Index_beh = Index[self.data_set.Output_A[beh]]
            np.random.shuffle(Index_beh)
            
            fold_number = self.repetition
            roll_value = int(test_parts * len(Index_beh))
            Index_beh = np.roll(Index_beh, fold_number * roll_value)
            
            Index_train.append(Index_beh[roll_value:])
            Index_test.append(Index_beh[:roll_value])
        
        self.Train_index = np.concatenate(Index_train, axis = 0)
        self.Test_index  = np.concatenate(Index_test, axis = 0)
    
    def get_name(self):
        num_splits = int(np.ceil(1 / self.test_part))
        names = {'print': '{} fold Cross validation (Split {})'.format(num_splits, self.repetition + 1),
                 'file': 'crossv_{}_split'.format(self.repetition),
                 'latex': r'CV - Fold {}'.format(self.repetition + 1)}
        return names
        
    def check_splitability_method(self):
        return None
    
    
    def repetition_number(self):
        num_splits = int(np.ceil(1 / self.test_part))
        return num_splits
        
        
        
    
        



