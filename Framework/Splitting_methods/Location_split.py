import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template

from no_split import no_split



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
    

    def alternative_train_split_file(self):
        # If there is only one dataset that is part of the training set, 
        # the model might allready be trained using no split / location split

        # Check if this is a connected dataset
        if len(self.data_set.Datasets) == 1:
            return None
            
        # Check how many datasets are part of the training set
        Situations = self.Domain['Scenario'].to_numpy().astype('str')
        Situations_train = Situations[self.Train_index]
        Situation = np.unique(Situations_train)
        
        if len(Situation) > 1:
            return None
        
        # Get training dataset
        Dataset_used = Situation[0]
        data_set_used = None
        for data_set in self.data_set.Datasets.values():
            data_set_name = data_set.get_name()['print']
            # Check if Dataset_used starts with data_set_name
            if Dataset_used.startswith(data_set_name):
                data_set_used = data_set
                break
        
        assert data_set_used is not None

        ## Get locations in the testset that are in data_set_used
        # Get locations in data_set_used
        all_locations = np.unique(data_set_used.Domain.location)

        # Get training locations
        train_locations = np.unique(self.Domain.location.iloc[self.Train_index])

        # Get the test locations
        test_locations = []
        for location in all_locations:
            if location not in train_locations:
                test_locations.append(str(location))
                    

        # Get the corresponding splitting method
        if len(test_locations) == 0:
            # The whole data_set_used is part of the training domain
            split_alternative = no_split(data_set_used, 
                                        test_part = self.test_part,
                                        repetition = (0,), 
                                        train_pert = self.train_pert, 
                                        test_pert = self.train_pert, 
                                        train_on_test = False 
                                        )

        else:
            # Some parts of data_set_used are part of the test domain
            split_alternative = Location_split(data_set_used, 
                                            test_part = self.test_part, 
                                            repetition = tuple(test_locations),
                                            train_pert = self.train_pert, 
                                            test_pert = self.train_pert,
                                            train_on_test = False 
                                            )
        
        # Do the actual test
        split_alternative.set_file_name()

        return split_alternative.split_file



            

        
    
        



