import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template
from Predefined_split import Predefined_split



class Dataset_predefined_split(splitting_template):
    '''
    When models are applied to combined datasets, it is possible to test
    their generalizability by testing them on one dataset and letting the models
    learn on the other datasets.
    
    Here, the number of possible repetitions depends on the number of datasets.
    
    When determining the desired repetition, the user can either input the string 
    of the dataset (see the get_name()['print'] method of the dataset), or the number
    corresponding to the position in a alphabetically sorted list of the dataset strings.
    '''
    def split_data_method(self):
        # Split by datasets
        Situations = self.Domain['Scenario']
        Situation, Situation_type = np.unique(Situations.to_numpy().astype('str'), return_inverse = True, axis = 0)
        
        Situations_test = (Situation_type[:,np.newaxis] == np.array(self.repetition)[np.newaxis]).any(1)
        
        Index = np.arange(len(Situations))
        
        # Check predefinition status
        Belonging = self.Domain['splitting'].to_numpy()
        assert np.all(np.unique(Belonging) == np.array(['test', 'train'])), 'Some of the datasets included are not predefined'
        
        Train_potential = (~Situations_test) & (Belonging == 'train')
        Test_potential  = (Situations_test) & (Belonging == 'test')

        Train_index = Index[Train_potential]
        Test_index  = Index[Test_potential]
        
        return Train_index, Test_index
        
    def get_name(self):
        Situations = self.Domain['Scenario']
        Situation = np.unique(Situations.to_numpy().astype('str'), axis = 0)
        scene = ', '.join(Situation[np.array(self.repetition)])
        rep_str = str(self.repetition)[1:-1]
        
        names = {'print': 'Dataset splitting using predefined (Testing on dataset ' + scene + ')',
                 'file': 'sc_pre_split',
                 'latex': r'Datasets (predefined) ' + rep_str}
        return names
    
    def check_splitability_method(self):
        if not hasattr(self.Domain, 'splitting'):
            return 'this dataset has no predefined splitting.'
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
    
    def alternative_train_split_file(self):
        # If there is only one dataset that is part of the training set, 
        # the model might allready be trained using predefined split

        # Check if this is a connected dataset
        if len(self.data_set.Datasets) > 1:
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
        for data_set in self.data_set.Datasets:
            data_set_name = data_set.get_name()['print']
            # Check if Dataset_used starts with data_set_name
            if Dataset_used.startswith(data_set_name):
                data_set_used = data_set
                break
        
        assert data_set_used is not None

        # Get the corresponding splitting method
        split_alternative = Predefined_split(data_set_used, 
                                             test_part = self.test_part, 
                                             repetition = (0), 
                                             train_pert = self.train_pert, 
                                             test_pert = self.test_pert, 
                                             train_on_test = False 
                                             )
        
        # Do the actual test
        split_alternative.set_file_name()

        return split_alternative.file_name
        




        
        



