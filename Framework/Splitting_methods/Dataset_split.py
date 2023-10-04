import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Dataset_split(splitting_template):
    '''
    When models are applied to combined datasets, it is possible to test
    their generalizability by testing them on one dataset and letting the models
    learn on the other datasets.
    
    Here, the number of possible repetitions depends on the number of datasets.
    '''
    def split_data_method(self):
        Situations = self.Domain['Scenario']
        Situation, Situation_type = np.unique(Situations.to_numpy().astype('str'), return_inverse = True, axis = 0)
        
        Situations_test = (Situation_type[:,np.newaxis] == np.array(self.repetition)[np.newaxis]).any(1)
        
        Index = np.arange(len(Situations))
        Train_index = Index[~Situations_test]
        Test_index  = Index[Situations_test]
        
        return Train_index, Test_index
        
    def get_name(self):
        Situations = self.Domain['Scenario']
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        scene = ', '.join(Situation[np.array(self.repetition)])
        rep_str = str(self.repetition)[1:-1]
        
        names = {'print': 'Dataset splitting (Testing on dataset ' + scene + ')',
                 'file': 'scenes_split',
                 'latex': r'Datasets ' + rep_str}
        return names
    
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        Situations = self.Domain['Scenario']
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        num_rep = len(Situation) 
        if num_rep == 1:
            return 0
        else:
            return num_rep
    
    def can_process_str_repetition(self = None):
        return True
        
    def tranform_str_to_number(self, rep_str):
        Situations = self.Domain['Scenario']
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        
        if rep_str in Situation:
            return list(np.where(rep_str == Situation)[0])
        else:
            return []
        
        



