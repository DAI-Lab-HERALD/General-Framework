import pandas as pd
import numpy as np
import os


class splitting_template():
    def __init__(self, data_set, test_partition = 0.2, repetition = (0)):
        # Save data_set location
        self.data_set = data_set
        if self.data_set is not None:
            self.Domain = self.data_set.Domain
        
        # Set the part of datasets sorted into test sets
        self.test_part = min(1.0, max(0.0, test_partition))
        
        
        # Filter out usable repetitions
        assert len(repetition) > 0
        self.repetition = []
        for rep in repetition:
            # Avoid duplications
            if rep in self.repetition:
                continue
            
            # Check type 
            if isinstance(rep, int):
                self.repetition.append(max(0,rep))
            elif isinstance(rep, str):
                if self.can_process_str_repetition():
                    rep_numbers = self.tranform_str_to_number(rep)
                    error_str = 'self.transform_str_to_number should return a list of integers.'
                    assert all((isinstance(x, np.int64) or isinstance(x, int)) for x in rep_numbers), error_str
                    self.repetition += rep_numbers
            else:
                raise TypeError('Repetitions must be integers or strings.')
        
        # Get max number of repetition 
        self.max_max_rep = 1000
        max_rep = self.repetition_number()
        if max_rep is None:
            max_rep = self.max_max_rep
        else:
            max_rep = min(self.max_max_rep, max_rep)
            
        self.repetition = np.unique(self.repetition).astype(int)
        self.repetition = list(self.repetition[self.repetition < max_rep])
       
    def split_data(self):
        self.split_file = self.data_set.change_result_directory(self.data_set.data_file, 'Splitting', 
                                                                self.get_name()['file'] + self.get_rep_str())
        
        if os.path.isfile(self.split_file):
            [self.Train_index, 
             self.Test_index, _] = np.load(self.split_file, allow_pickle = True)
        else:
            self.Train_index, self.Test_index = self.split_data_method()
             
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
    
    def get_rep_str(self):
        dig = len(str(self.max_max_rep - 1))
        rep_str = '_'.join([str(rep).zfill(dig) for rep in self.repetition])
        
        return '_' + rep_str

    def check_splitability(self):            
        if len(self.repetition) == 0:
            return 'this splitting method has no viable repetitions.'
        
        return self.check_splitability_method()  
    
    
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                      Splitting method dependend functions                         ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    def get_name(self):
        r'''
        Provides a dictionary with the different names of the splitting method
            
        Returns
        -------
        names : dict
          The first key of names ('print')  will be primarily used to refer to the splitting method in console outputs. 
                
          The 'file' key has to be a string with exactly **12 characters**, that does not include any folder separators 
          (for any operating system), as it is mostly used to indicate that certain result files belong to this splitting
          method. 
                
          The 'latex' key string is used in automatically generated tables and figures for latex and can include 
          latex commands - such as using '$$' for math notation.
            
        '''
        raise AttributeError('Has to be overridden in actual method.')
        
    def split_data_method(self):
        # this function takes the given input and then creates a 
        # split according to a desied method
        # creates:
            # self.Train_index -    A 1D numpy including the samples IDs of the training set
            # self.Test_index -     A 1D numpy including the samples IDs of the test set
        raise AttributeError('Has to be overridden in actual method.')
    
    def check_splitability_method(self):
        # Provides feedback on if a splitting method can be used, as it might be
        # related to only specifc datasets/scenarios/etc.
        raise AttributeError('Has to be overridden in actual method.') 
        
    def repetition_number(self):
        # Provides the number of repetitions that the model requires
        raise AttributeError('Has to be overridden in actual method.') 
        
    def can_process_str_repetition(self = None):
        # Decides whether the splitting method can take string repetitions
        raise AttributeError('Has to be overridden in actual method.') 
        
    def tranform_str_to_number(self, rep_str):
        # Given a string, returns the actual number that would have been
        # used if the string was not given
        raise AttributeError('Has to be overridden in actual method.') 
        
        
        



