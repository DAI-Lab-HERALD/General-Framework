import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Dataset_split(splitting_template):
    def split_data_method(self):
        Situations = self.data_set.Domain['Scenario']
        Situation, Situation_type = np.unique(Situations.to_numpy().astype('str'), return_inverse = True, axis = 0)
        
        Situations_test = Situation_type == self.repetition
        
        Index = np.arange(len(Situations))
        self.Train_index = Index[~Situations_test]
        self.Test_index  = Index[Situations_test]
        
    def get_name(self):
        Situations = self.data_set.Domain['Scenario']
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        scene = Situation[self.repetition]
        
        names = {'print': 'Dataset splitting (Testing on dataset ' + scene + ')',
                 'file': 'scenes_split',
                 'latex': r'Dataset split {}'.format(self.repetition + 1)}
        return names
    
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        Situations = self.data_set.Domain['Scenario']
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        num_rep = len(Situation) 
        if num_rep == 1:
            return 0
        else:
            return num_rep
        
    
        



