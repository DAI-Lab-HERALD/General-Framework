import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class Location_split(splitting_template):
    '''
    To test how well models generalize to new scenes, one can test them on a 
    recording location held out from the training set. 
    
    The number of possible repetitions of this split is therefore depending
    on the number of different recording locations.
    
    If ones passes a repetition as a string, then there are two options.
    Either, one simly passes the location name (see self.Domain.location), in which 
    case it might be possible that multiple locations with equal names across multiple 
    datasets are selected. Alternatively, one can specify the dataset by passing the 
    following string: *<dataset_name> -:- <location_name>*.
    '''
    def split_data_method(self):
        Situations = self.Domain[['Scenario', 'location']]
        Situation, Situation_type = np.unique(Situations.to_numpy().astype('str'), return_inverse = True, axis = 0)
        
        Situations_test = (Situation_type[:,np.newaxis] == np.array(self.repetition)[np.newaxis]).any(1)
        
        Index = np.arange(len(Situations))
        Train_index = Index[~Situations_test]
        Test_index  = Index[Situations_test]
        
        return Train_index, Test_index
        
    def get_name(self):
        Situations = self.Domain[['Scenario', 'location']]
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        scenes, locs = Situation[np.array(self.repetition)].T
        
        Locs = np.core.defchararray.add(locs, ' (')
        Locs = np.core.defchararray.add(Locs, scenes)
        Locs = np.core.defchararray.add(Locs, ')')
        
        Locs_str = ', '.join(Locs)
        
        rep_str = str(self.repetition)[1:-1]
        
        names = {'print': 'Location splitting (Testing on locations ' + Locs_str + ')',
                 'file': 'locals_split',
                 'latex': r'Locations ' + rep_str}
        return names
    
    def check_splitability_method(self):
        if not hasattr(self.Domain, 'location'):
            return 'this splitting method can only work on problems divisible by location.'
        
        location_splits = len(np.unique(self.Domain[['Scenario', 'location']].to_numpy().astype('str'), axis = 0))
        datasets_splits = len(np.unique(self.Domain['Scenario'].to_numpy().astype('str'), axis = 0))
        
        if location_splits == datasets_splits:
            return ('location splitting would be identical to dataset splitting.')
        
        return None
    
    def repetition_number(self):
        Situations = self.Domain[['Scenario', 'location']]
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        num_rep = len(Situation) 
        if num_rep == 1:
            return 0
        else:
            return num_rep
    
    def can_process_str_repetition(self = None):
        return True
        
    def tranform_str_to_number(self, rep_str):
        Situations = self.Domain[['Scenario', 'location']]
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        
        info = rep_str.split(' -:- ')
        if len(info) > 1:
            dataset_str = info[0]
            location = info[1]
            index = np.where((Situation[:,0] == dataset_str) &
                             (Situation[:,1] == location))[0]
        else:
            location = info[0]
            index = np.where(Situation[:,1] == location)[0]
            
        return list(index)
        
    
        



