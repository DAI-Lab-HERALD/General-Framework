import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class no_split(splitting_template):
    def split_data_method(self):
        Index = np.arange(len(self.Domain))
        
        Train_index = Index
        Test_index  = Index
        
        return Train_index, Test_index
        
    def get_name(self):
        names = {'print': 'No splitting',
                 'file': 'identi_split',
                 'latex': r'No split'}
        return names
    
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        return 1
        
    
        



