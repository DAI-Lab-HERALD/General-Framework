import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Location_split(splitting_template):
    def split_data_method(self):
        Situations = self.data_set.Domain[['Scenario', 'location']]
        Situation, Situation_type = np.unique(Situations.to_numpy().astype('str'), return_inverse = True, axis = 0)
        
        Situations_test = Situation_type == self.repetition
        
        Index = np.arange(len(Situations))
        self.Train_index = Index[~Situations_test]
        self.Test_index  = Index[Situations_test]
        
    def get_name(self):
        Situations = self.data_set.Domain[['Scenario', 'location']]
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        scene, loc = Situation[self.repetition]
        
        names = {'print': 'Location splitting (Testing on location ' + loc + ' in dataset ' + scene + ')',
                 'file': 'locals_split_{}'.format(self.repetition),
                 'latex': r'Location split {}'.format(self.repetition + 1)}
        return names
    
    def check_splitability_method(self):
        if not hasattr(self.data_set.Domain, 'location'):
            return 'this splitting method can only work on problems divisible by location.'
        
        location_splits = len(np.unique(self.data_set.Domain[['Scenario', 'location']].to_numpy().astype('str'), axis = 0))
        datasets_splits = len(np.unique(self.data_set.Domain['Scenario'].to_numpy().astype('str'), axis = 0))
        
        if location_splits == datasets_splits:
            return ('location splitting would be identical to dataset splitting.')
        
        return None
    
    def repetition_number(self):
        Situations = self.data_set.Domain[['Scenario', 'location']]
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        num_rep = len(Situation) 
        if num_rep == 1:
            return 0
        else:
            return num_rep
        
    
        


