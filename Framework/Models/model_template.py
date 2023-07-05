import pandas as pd
import numpy as np
import os
import torch
import warnings

class model_template():
    def __init__(self, data_set, splitter, behavior = None): 
        # Load gpu
        if self.requires_torch_gpu():
            if torch.cuda.is_available():
                self.device = torch.device('cuda', index=0) 
                torch.cuda.set_device(0)
            else:
                warnings.warn('''No GPU could be found. Program is proccessed on the CPU.
                              
                This might lead either to a model failure or significantly decreased
                training and prediction speed.''')
                self.device = torch.device('cpu')
        
        
        
        if behavior == None:
            self.is_data_transformer = False
            
            Index = splitter.Train_index
            
            self.model_file = data_set.change_result_directory(splitter.split_file,
                                                               'Models', self.get_name()['file'])
            
            
            if self.get_output_type() == 'path_all_wi_pov':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_tra_wi_pov')
            
            elif self.get_output_type() == 'path_all_wo_pov':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_tra_wo_pov')
                
            elif self.get_output_type() == 'class':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_classified')
                
            elif self.get_output_type() == 'class_and_time':
                self.pred_file = data_set.change_result_directory(self.model_file,
                                                                  'Predictions', 'pred_class_time')
            
            else:
                raise AttributeError("This type of prediction is not implemented")
            
            
        else:
            self.is_data_transformer = True
            Index = np.where(data_set.Output_A[behavior])[0]
            if len(Index) == 0:
                # There are no samples of the required behavior class
                # Use those cases where the other behaviors are the furthers away
                num_long_index = int(len(data_set.Output_A) / len(data_set.Behaviors))
                Index = np.argsort(data_set.Output_T_E)[-num_long_index :]
            
            save_file = data_set.data_file[:-4] + '-transform_path_(' + behavior + ').npy'
            self.model_file = save_file
            self.pred_file = self.model_file[:-4] + '-pred_tra_wi_pov.npy'
        
        self.data_set = data_set
        
        self.input_names_train = data_set.Input_path.columns
        # Only provide one kind of input (if model cheats and tries to use both)
        if self.get_input_type()['past'] == 'general':
            self.Input_prediction_train = data_set.Input_prediction.iloc[Index]
            self.Input_path_train       = None
            
        elif self.get_input_type()['past'] == 'path':
            self.Input_prediction_train = None
            self.Input_path_train       = data_set.Input_path.iloc[Index]
            
        elif self.get_input_type()['past'] == 'both':
            self.Input_prediction_train = data_set.Input_prediction.iloc[Index]
            self.Input_path_train       = data_set.Input_path.iloc[Index]
        else:
            raise AttributeError("This kind of past input information is not implemented.")
            
        
        if self.get_input_type()['future']:
            Pov_ind = np.array([name[2:] == self.data_set.pov_agent for name in self.input_names_train])
            self.Input_future_train = data_set.Output_path.iloc[Index, Pov_ind]
        else:
            self.Input_future_train = data_set.Output_path.iloc[Index, np.zeros(len(self.input_names_train), bool)]
            
        # check if model is allowed
        
        self.Input_T_train           = data_set.Input_T[Index]
        
        self.Output_path_train       = data_set.Output_path.iloc[Index]
        self.Output_T_train          = data_set.Output_T[Index]
        self.Output_T_pred_train     = data_set.Output_T_pred[Index]
        self.Output_A_train          = data_set.Output_A.iloc[Index]
        self.Output_T_E_train        = data_set.Output_T_E[Index]
        
        self.Domain_train            = data_set.Domain.iloc[Index]
        
        self.num_samples_path_pred = self.data_set.num_samples_path_pred
        
        self.num_samples_train = len(Index)
        
        self.setup_method()
        
        # Set trained to flase, this prevents a prediction on an untrained model
        self.trained = False
        
    
    def train(self):
        if os.path.isfile(self.model_file) and not self.data_set.overwrite_results:
            self.weights_saved = list(np.load(self.model_file, allow_pickle = True)[:-1])
            self.load_method()
        
        else:
            # creates /overrides self.weights_saved
            self.train_method() 
            #0 is there to avoid some numpy load and save errros
            save_data = np.array(self.weights_saved + [0], object) 
            
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            np.save(self.model_file, save_data)
            
            if self.save_params_in_csv():
                for i, param in enumerate(self.weights_saved):
                    param_name = [name for name in vars(self) if np.array_equal(getattr(self, name), param)][0]
                    if i == 0:
                        np.savetxt(self.model_file[:-4] + '.txt', param,
                                   fmt       = '%10.5f',
                                   delimiter = ', ',
                                   header    = param_name + ':',
                                   footer    = '\n \n',
                                   comments  = '')
                    else:
                        with open(self.model_file[:-4] + '.txt', 'a') as f:
                            np.savetxt(f, param,
                                       fmt       = '%10.5f',
                                       delimiter = ', ',
                                       header    = param_name + ':',
                                       footer    = '\n \n',
                                       comments  = '')
            
            if self.provides_epoch_loss():
                self.loss_file = self.model_file[:-4] + '--train_loss.npy'
                assert hasattr(self, 'train_loss'), "No train loss is provided."
                assert isinstance(self.train_loss, np.ndarray), "The train loss should be a numpy array."
                assert len(self.train_loss.shape) == 2, "The train loss should be a 2D numpy array."
                np.save(self.loss_file, self.train_loss.astype(np.float32))
           
        self.trained = True
        
        print('')
        print('The model ' + self.get_name()['print'] + ' was successfully trained.')
        print('')
        return self.model_file
        
        
    def predict(self):
        # perform prediction
        if os.path.isfile(self.pred_file) and not self.data_set.overwrite_results:
            output = list(np.load(self.pred_file, allow_pickle = True)[:-1])
        else:
            if self.get_input_type()['past'] == 'general':
                self.Input_prediction_test = self.data_set.Input_prediction
                self.Input_path_test       = None
                
            elif self.get_input_type()['past'] == 'path':
                self.Input_prediction_test = None
                self.Input_path_test       = self.data_set.Input_path
                
            elif self.get_input_type()['past'] == 'both':
                self.Input_prediction_test = self.data_set.Input_prediction
                self.Input_path_test       = self.data_set.Input_path
                
            if self.get_input_type()['future']:
                Pov_ind = np.array([name[2:] == self.data_set.pov_agent for name in self.input_names_train])
                self.Input_future_test = self.data_set.Output_path.iloc[:, Pov_ind]
            
            # Make predictions on all samples, test and training samples
            self.Input_T_test             = self.data_set.Input_T
            self.Output_T_pred_test       = self.data_set.Output_T_pred

            self.Domain_test              = self.data_set.Domain            
            # save number of training samples
            self.num_samples_test = len(self.Input_T_test)
            
            # apply model to test samples
            output = self.predict_method() # output needs to be a list of components
            
            save_data = np.array(output + [0], object) #0 is there to avoid some numpy load and save errros
            
            os.makedirs(os.path.dirname(self.pred_file), exist_ok=True)
            np.save(self.pred_file, save_data)
                        
        print('')
        print('The model ' + self.get_name()['print'] + ' successfully made predictions.')
        print('')
        return output
    
        
    def check_trainability(self):
        if self.get_input_type()['future']: 
            if self.data_set.pov_agent is None:
                return 'the needed pov agent is not included in this dataset.'
            
            if not self.data_set.future_input():
                return 'the required future trajectory of the pov agent contains too many clues.'
            
            if self.get_output_type() == 'path_all_wi_pov':
                return 'the trajectory of the poc agent that has to be predicted should be used as input.'
        else:
            if self.get_output_type() == 'path_all_wo_pov': 
                return 'the trajectory of the poc agent is not predicted, although it is not know a priori.'
        
        if self.get_output_type() == 'path_all_wo_pov':
            if len(self.data_set.needed_agents) == 1:
                return 'there is no agent of which a trajectory can be predicted.'
        
        if self.get_input_type()['past'] in ['general', 'both']:
            if not self.data_set.general_input_available:
                return 'this dataset cannot provide the required 1D inputs.'
            
        if self.get_output_type() in ['class', 'class_and_time']:
            if not self.data_set.classification_useful:
                return 'a classification model cannot be trained on a classless dataset.'
              
        return self.check_trainability_method()
    
    
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                            Model dependend functions                              ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    
    def setup_method(self):
        # setsup the model, but only use the less expensive calculations.
        # Anything more expensive, especially if only needed for training,
        # should be done in train_method or load_method instead.
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def train_method(self):
        # Uses input data to train the model, saving the weights needed to fully
        # describe the trained model in self.weight_saved
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def load_method(self):
        # Builds the models using loaded weights in self.weights_saved
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def predict_method(self):
        # takes test input and uses that to predict the output
        raise AttributeError('Has to be overridden in actual model.')
        # return output

    def check_trainability_method(self):
        # checks if current environment (i.e, number of agents, number of input timesteps) make this trainable.
        # If not, it also retuns the reason in form of a string.
        # If it is trainable, return None
        raise AttributeError('Has to be overridden in actual model.')
        
            
    def get_output_type(self = None):
        # Should return 'class', 'class_and_time', 'path_all_wo_pov', 'path_all_wi_pov'
        # the same as above, only this time callable from class
        raise AttributeError('Has to be overridden in actual model.')
        
    def get_input_type(self = None):
        # Should return a dictionary, with the first key being 'past', 
        # where the value is either 'general' (using self.Input_prediction_train) 
        # or 'path' (using self.Input_path_train) or 'both'
        # The second key is 'future', which is either False or True, with pov being
        # the role of the actor whose future input is used. If this is not None, than 
        # the output type must not be 'path_all_wi_pov'
        # the same as above, only this time callable from class
        raise AttributeError('Has to be overridden in actual model.')
        

    def get_name(self = None):
        # Provides a dictionary with the different names of the dataset:
        # Name = {'print': 'printable_name', 'file': 'name_used_in_files', 'latex': r'latex_name'}
        # If the latex name includes mathmode, the $$ has to be included
        # Here, it has to be noted that name_used_in_files will be restricted in its length.
        # For models, this length is 10 characters, without a '-' inside
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def save_params_in_csv(self = None):
        # Returns true or false, depending on if the params of the trained model should be saved as a .csv file or not
        raise AttributeError('Has to be overridden in actual model.')
        
        
    def requires_torch_gpu(self = None):
        # Returns true or false, depending if the model does calculations on the gpu
        raise AttributeError('Has to be overridden in actual model.')
        
    
    def provides_epoch_loss(self = None):
        # Returns true or false, depending on if the model provides losses or not
        raise AttributeError('Has to be overridden in actual model.')
        
        