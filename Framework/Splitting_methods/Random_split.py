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
        
        Index = np.arange(len(self.data_set.Output_A))
        Index_test = []
        for beh in self.data_set.Behaviors:
            for rep in self.repetition:
                np.random.seed(rep)
                Index_beh = Index[self.data_set.Output_A[beh]]
                np.random.shuffle(Index_beh)
                num_train_beh = int((1 - self.test_part) * len(Index_beh))
                Index_test.append(Index_beh[num_train_beh:])
        
        Test_index = np.concatenate(Index_test, axis = 0)
        Test_index = np.unique(Test_index)
        
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
    
        
        
        
    
        



