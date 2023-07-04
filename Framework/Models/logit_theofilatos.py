import numpy as np
import pandas as pd
from model_template import model_template
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos(model_template):
    def setup_method(self, l2_regulization = 0.1):
        self.timesteps = max([len(T) for T in self.Input_T_train])
    
        self.model = LR(C = 1/(l2_regulization+1e-7), 
                        max_iter = 100000, 
                        class_weight = 'balanced',
                        multi_class = 'multinomial')
    
    def get_data(self, train = True):
        if train:
            Input_array = self.Input_path_train.to_numpy()
        else:
            Input_array = self.Input_path_test.to_numpy()
            
        # Extract predicted agents
        Agents = np.array([name[2:] for name in np.array(self.input_names_train)])
        Pred_agents = np.array([agent in self.data_set.needed_agents for agent in Agents])
        
        X = np.ones([Input_array.shape[0], Input_array.shape[1], self.timesteps, 2]) * np.nan
        for i_sample in range(len(X)):
            for i_agent in range(Input_array.shape[1]):
                if isinstance(Input_array[i_sample, i_agent], float):
                    assert not Pred_agents[i_agent], 'A needed agent is not given.'
                else:   
                    len_given = len(Input_array[i_sample,i_agent])
                    n_help_1 = max(0, self.timesteps - len_given)
                    n_help_2 = max(0, len_given - self.timesteps)
                    X[i_sample, i_agent, n_help_1:] = Input_array[i_sample,i_agent][n_help_2:]
        X = X.reshape(X.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        if not hasattr(self, 'mean'):
            assert train, "This should not be possible, loading failed"
            self.mean = np.nanmean(X, axis = 0, keepdims = True)
        
        X = X - self.mean
        X[np.isnan(X)] = 0
        return X
        
        
    def train_method(self):
        # Multiple timesteps have to be flattened
        X = self.get_data(train = True)
        # Train model
        true_labels = self.data_set.Behaviors[self.Output_A_train.to_numpy().argmax(axis = 1)]
        self.model.fit(X, true_labels)
        # Test if something else might be feasible
        self.weights_saved = [self.mean, self.model]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.model] = self.weights_saved
        
        
    def predict_method(self):
        X = self.get_data(train = False)
        Probs = pd.DataFrame(self.model.predict_proba(X), columns = self.model.classes_)
        
        # Fill in required classes that have been missing in the training data
        missing_classes = self.data_set.Behaviors[(self.data_set.Behaviors[:, np.newaxis] != 
                                                   np.array(Probs.columns)[np.newaxis]).all(1)]
        Probs[missing_classes] = 0.0
        
        Probs = Probs[self.data_set.Behaviors]
        return [Probs]
    
    def check_trainability_method(self):
        return None
        
    def get_output_type(self = None):
        # Logit model only produces class outputs
        return 'class'
        
    def get_input_type(self = None):
        input_info = {'past': 'path',
                      'future': False}
        return input_info
    
    def get_name(self = None):
        names = {'print': 'Logistic regression (2D inputs)',
                 'file': 'log_reg_2D',
                 'latex': r'$\text{\emph{LR}}_{2D}$'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return False
        
    def provides_epoch_loss(self = None):
        return False
        
        
        
        
        
    
        
        
        