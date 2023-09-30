import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template


class Predefined_split(splitting_template):
    '''
    Especially for large dataset, it is often only viable to use a single test 
    set. In this case, it is often common for samples to be already
    devided into train and test sets by their creators.
    '''
    def split_data_method(self):
        # Check assumptions
        assert self.repetition == [0,]
        
        Belonging = self.Domain['splitting'].to_numpy()
        
        assert np.all(np.unqiue(Belonging) == np.array(['test', 'train'])), ''
        
        Index = np.arange(len(Belonging))
        Train_index = Index[Belonging == 'train']
        Test_index  = Index[Belonging == 'test']
        
        return Train_index, Test_index
        
    def get_name(self):
        names = {'print': 'Predefined splitting',
                 'file': 'predef_split',
                 'latex': r'Predefined split'}
        return names
    
    def check_splitability_method(self):
        if not hasattr(self.Domain, 'splitting'):
            return 'this dataset has no predefined splitting.'
        
        return None
    
    def repetition_number(self):
        return 1
    
    
    def can_process_str_repetition(self = None):
        return False
        
    
        



