import pandas as pd
import numpy as np
import os
import scipy as sp




class splitting_template():
    def __init__(self, data_set, test_partition = 0.2, repetition = 0):
        # Save data_set location
        self.data_set = data_set
        
        # Set the part of datasets sorted into test sets
        self.test_part = test_partition
        
        assert repetition < 10, "No more than 10 repetitions of the same splitting type are possible."
        self.repetition = int(max(0,repetition))
       
    def split_data(self):
        self.split_file = self.data_set.change_result_directory(self.data_set.data_file, 'Splitting', 
                                                                self.get_name()['file'] + '_{}'.format(self.repetition))
        if os.path.isfile(self.split_file):
            [self.Train_index, 
             self.Test_index,
             sim_all,
             Sim_any, _] = np.load(self.split_file, allow_pickle = True)
        else:
            self.split_data_method()
             
            # check if split was successful
            if not all([hasattr(self, attr) for attr in ['Train_index', 'Test_index']]):
                 raise AttributeError("The splitting into train and test set was unsuccesful")
            
            np.random.shuffle(self.Train_index)
            np.random.shuffle(self.Test_index)
            
            save_data = np.array([self.Train_index, 
                                  self.Test_index,
                                  0], object) #0 is there to avoid some numpy load and save errors
            
            
            os.makedirs(os.path.dirname(self.split_file), exist_ok=True)
            np.save(self.split_file, save_data)
        

    def check_splitability(self):
        max_rep = self.repetition_number()
        if max_rep is not None:
            if self.repetition >= max_rep:
                return 'this splitting method only allows for {} repetitions.'.format(max_rep)
        
        return self.check_splitability_method()  
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                      Splitting method dependend functions                         ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################

    def split_data_method(self):
        # this function takes the given input and then creates a 
        # split according to a desied method
        # creates:
            # self.Train_index -    A 1D numpy including the samples IDs of the training set
            # self.Test_index -     A 1D numpy including the samples IDs of the test set
        raise AttributeError('Has to be overridden in actual method.')
    
    
    def get_name(self):
        # Provides a dictionary with the different names of the dataset:
        # Name = {'print': 'printable_name', 'file': 'name_used_in_files', 'latex': r'latex_name'}
        # If the latex name includes mathmode, the $$ has to be included
        # Here, it has to be noted that name_used_in_files will be restricted in its length.
        # For datasets, this length is 14 characters, without a '-' inside
        raise AttributeError('Has to be overridden in actual method.')
        
        
    
    def check_splitability_method(self):
        # Provides feedback on if a splitting method can be used, as it might be
        # related to only specifc datasets/scenarios/etc.
        raise AttributeError('Has to be overridden in actual method.') 
        
    def repetition_number(self):
        # Provides the umber of repetitions that the model requires
        raise AttributeError('Has to be overridden in actual method.') 
        
        



