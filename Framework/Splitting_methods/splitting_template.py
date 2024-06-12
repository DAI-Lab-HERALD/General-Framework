import pandas as pd
import numpy as np
import os


class splitting_template():
    def __init__(self, data_set, test_part = 0.2, repetition = (0), train_pert = False, test_pert = False):
        # Save data_set location
        self.data_set = data_set
        if self.data_set is not None:
            self.Domain = self.data_set.Domain
        
        # Set the part of datasets sorted into test sets
        self.test_part = min(1.0, max(0.0, test_part))
        
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
        self.max_max_rep = 10
        max_rep = self.repetition_number()
        if max_rep is None:
            max_rep = self.max_max_rep
        else:
            max_rep = min(self.max_max_rep, max_rep)
            
        self.repetition = np.unique(self.repetition).astype(int)
        self.repetition = list(self.repetition[self.repetition < max_rep])

        # Check for pertubation if they exist
        if train_pert or test_pert:
            if not self.Domain.perturbation.any():
                raise AttributeError('The domain does not contain any perturbed data.')
            
        self.train_pert = train_pert
        self.test_pert = test_pert
       
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
            
            # Overwrite the respective File location for the perturbed data
            perturbed = self.Domain['perturbation']
            overwrite_with_unperturbed = np.zeros(len(perturbed), dtype = bool)
            
            if not self.train_pert:
                overwrite_with_unperturbed |= perturbed & np.in1d(np.arange(len(perturbed)), self.Train_index)
            if not self.test_pert:
                overwrite_with_unperturbed |= perturbed & np.in1d(np.arange(len(perturbed)), self.Test_index)
            
            if np.any(overwrite_with_unperturbed):
                index_overwrite = np.where(overwrite_with_unperturbed)[0]
                
                # Overwrite the respective file location
                Files_used = self.Domain.file_index.iloc[index_overwrite]
                
                for file_index in np.unique(Files_used):
                    # Get indices
                    index_overwrite_local = np.where(Files_used == file_index)[0]
                    index_local = index_overwrite[index_overwrite_local]
                    
                    # Get the current file name
                    file = self.Files[file_index]
                    
                    # Check that this is indeed a perturbed file
                    assert '--Pertubation' in file, 'This file is not a perturbed file.'
                    
                    pre_perturbation, post_perturbation = file.split('--Pertubation')
                    file_unperturbed = pre_perturbation + post_perturbation[4:]
                    
                    # Check if unperturbed file is in file list
                    if file_unperturbed in self.Domain.Files:
                        file_index_unperturbed = self.Domain.Files.index(file_unperturbed)
                        self.Domain.file_index.iloc[index_local] = file_index_unperturbed
                    else:
                        self.Domain.Files.append(file_unperturbed)
                        self.Domain.file_index.iloc[index_local] = len(self.Domain.Files) - 1
                    
                    # TODO: Check if self.data_set.Domain was also overwritten
                        
                    # If we can load complete dataset, we will also overwrite this
                    if self.data_set.data_in_one_piece:
                        # Load the respective input and output data
                        data_file_unperturbed = file_unperturbed + '.npy'
                        [_, Input_path, _, Output_path, _, _, _, _, _] = np.load(data_file_unperturbed, allow_pickle=True)
                        
                        # Apply the local index
                        used_index = self.Domain.iloc[index_local].Index_saved
                        Input_path  = Input_path.loc[used_index]
                        Output_path = Output_path.loc[used_index]
                        
                        # Overwrite the data
                        index_loca_loc = self.Domain.iloc[index_local].index
                        self.data_set.Input_path.loc[index_loca_loc, Input_path.columns]   = Input_path
                        self.data_set.Output_path.loc[index_loca_loc, Output_path.columns] = Output_path
                        
            # Set overwritten files to status unperturbed
            self.Domain['perturbation'].iloc[overwrite_with_unperturbed] = False

    
    def get_rep_str(self):
        dig = len(str(self.max_max_rep - 1))
        rep_str = '_'.join([str(rep).zfill(dig) for rep in self.repetition])
        rep_str += '_pert=' + str(int(self.train_pert)) + str(int(self.test_pert))
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