import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template


class Random_split(splitting_template):
    def split_data_method(self):
        np.random.seed(self.repetition)
        
        Index = np.arange(len(self.data_set.Output_A))
        Index_train = []
        Index_test = []
        for beh in self.data_set.Behaviors:
            Index_beh = Index[self.data_set.Output_A[beh]]
            np.random.shuffle(Index_beh)
            
            num_train_beh = int((1 - self.test_part) * len(Index_beh))
            Index_train.append(Index_beh[:num_train_beh])
            Index_test.append(Index_beh[num_train_beh:])
        
        Train_index = np.concatenate(Index_train, axis = 0)
        Test_index  = np.concatenate(Index_test, axis = 0)
        
        return Train_index, Test_index
        
    
    def get_name(self):
        names = {'print': 'Random splitting (random seed = {})'.format(self.repetition + 1),
                 'file': 'random_split',
                 'latex': r'Random split'}
        return names
        
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        return None
    
        
        
        
    
        



