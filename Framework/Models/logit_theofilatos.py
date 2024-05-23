import numpy as np
import pandas as pd
from model_template import model_template
from sklearn.linear_model import LogisticRegression as LR

class logit_theofilatos(model_template):
    '''
    The logistic regression model is a standard classification model used in many 
    settings.
    
    This instance takes as input normal position data, and its training is based 
    on the following citation:
        
    Theofilatos, A., Ziakopoulos, A., Oviedo-Trespalacios, O., & Timmis, A. (2021). 
    To cross or not to cross? Review and meta-analysis of pedestrian gap acceptance 
    decisions at midblock street crossings. Journal of Transport & Health, 22, 101108.
    '''
    def setup_method(self, l2_regulization = 0.1):
        self.timesteps = self.dt
    
        self.model = LR(C = 1/(l2_regulization+1e-7), 
                        max_iter = 100000, 
                        class_weight = 'balanced',
                        multi_class = 'multinomial')
    
    def get_data(self, train = True):
        if train:
            X, _, _, _, _, class_names, P, _ = self.get_classification_data(train)
        else:
            X, _, _, _, _, class_names = self.get_classification_data(train)
            P = None
        X = X.reshape(X.shape[0], -1)
        
        # Normalize data, so no input can be set to zero
        if not hasattr(self, 'mean'):
            assert train, "This should not be possible, loading failed"
            self.mean = np.nanmean(X, axis = 0, keepdims = True)
        
        X = X - self.mean
        X[np.isnan(X)] = 0
        return X, P, class_names
        
        
    def train_method(self):
        # Multiple timesteps have to be flattened
        X, P, class_names = self.get_data(train = True)
        # Train model
        true_labels = np.array(class_names)[P.argmax(axis = 1)]
        
        self.model.fit(X, true_labels)
        # Test if something else might be feasible
        self.weights_saved = [self.mean, self.model]
        
        
    def load_method(self, l2_regulization = 0):
        [self.mean, self.model] = self.weights_saved
        
        
    def predict_method(self):
        X, _, class_names = self.get_data(train = False)
        Probs = pd.DataFrame(self.model.predict_proba(X), columns = self.model.classes_)
        
        # Fill in required classes that have been missing in the training data
        cn = np.array(class_names)
        missing_classes = list(cn[(cn[:, np.newaxis] != np.array(Probs.columns)[np.newaxis]).all(1)])
        Probs[missing_classes] = 0.0
        Probs = Probs[class_names]
        self.save_predicted_classifications(class_names, Probs.to_numpy())
    
    def check_trainability_method(self):
        return None
        
    def get_output_type(self = None):
        # Logit model only produces class outputs
        return 'class'
    
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
        
        
        
        
        
    
        
        
        