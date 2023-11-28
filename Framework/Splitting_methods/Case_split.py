import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Case_split(splitting_template):
    '''
    In this splitting method, the whole dataset is used as both the training
    and the testing set. It therefore only test how well a model can reproduce 
    its training data.
    
    As there is only one possible split here, this method onlye has one possible 
    repetition.
    '''
    def split_data_method(self):
        # Check assumptions
        self.data_set._group_indentical_inputs()
        Subgroups = self.data_set.Subgroups - 1
        repetition = np.array(self.repetition)
        
        Index = np.arange(len(self.Domain))
        
        Train_index = Index[np.in1d(Subgroups, repetition)] # how will this work? what is in Subgroups[:, np.newaxis]; shouldn't it be np.where or smth similar?
        Test_index  = Train_index
        
        return Train_index, Test_index
        
    def get_name(self):
        names = {'print': 'Identical input splitting',
                 'file': 'iden_i_split',
                 'latex': r'II split'}
        return names
    
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        if self.data_set is None:
            return 1000
        else:
            self.data_set._group_indentical_inputs()
            return self.data_set.Subgroups.max()
            # return 1000
    
    
    def can_process_str_repetition(self = None):
        return False
        
    
        



