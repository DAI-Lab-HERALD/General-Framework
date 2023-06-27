import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template



class no_split(splitting_template):
    def split_data_method(self):
        Index = np.arange(len(self.data_set.Output_A))
        
        self.Train_index = Index
        self.Test_index  = Index
        
    def get_name(self):
        names = {'print': 'No splitting',
                 'file': 'identic__split',
                 'latex': r'No split'}
        return names
    
    def check_splitability_method(self):
        return None
    
    def repetition_number(self):
        return 1
        
    
        



