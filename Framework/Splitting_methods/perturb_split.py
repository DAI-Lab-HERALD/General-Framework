import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template

class perturb_split(splitting_template):
    '''
    In this splitting method, the whole dataset is used as both the training
    and the testing set. It therefore only test how well a model can reproduce 
    its training data.
    
    As there is only one possible split here, this method onlye has one possible 
    repetition.
    '''
    def split_data_method(self):
        # Check assumptions
        assert self.repetition == [0,]

        index_perturbed = self.Domain['perturbation']
        
        Train_index = np.where(~index_perturbed)[0]
        Test_index  = np.where(index_perturbed)[0]
        
        return Train_index, Test_index
        
    def get_name(self):
        names = {'print': 'Perturbation splitting',
                 'file': 'pertur_split',
                 'latex': r'Pert split'}
        return names
    
    def check_splitability_method(self):
        # Check if train and test pert are the same
        if self.train_pert:
            return 'training set must be unperturbed.'
        if not self.test_pert:
            return 'testing set must be perturbed.'
        if 'perturbation' not in self.Domain.columns:
            return 'dataset contains no perturbed data'
        else:
            if self.Domain['perturbation'].sum() == 0:
                return 'dataset contains no perturbed data'
        return None
    
    def repetition_number(self):
        return 1
    
    
    def can_process_str_repetition(self = None):
        return False
        
    
        