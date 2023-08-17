import numpy as np
import pandas as pd
from model_template import model_template
import torch
import torch.nn as nn
import torch.optim as optim

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, learning_rate=0.001, momentum=0.95, xavier_const=1.0):
        super(RBM, self).__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')
        
        def xavier_init(fan_in, fan_out, const=1.0):
            k = const * np.sqrt(6.0 / (fan_in + fan_out))
            return torch.Tensor(fan_in, fan_out).uniform_(-k, k)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.w = nn.Parameter(xavier_init(self.n_visible, self.n_hidden, const=xavier_const))
        self.visible_bias = nn.Parameter(torch.zeros(self.n_visible))
        self.hidden_bias = nn.Parameter(torch.zeros(self.n_hidden))

        self.delta_w = torch.zeros(self.n_visible, self.n_hidden)
        self.delta_visible_bias = torch.zeros(self.n_visible)
        self.delta_hidden_bias = torch.zeros(self.n_hidden)
        
    def step(self, x):
        
        hidden_p = torch.sigmoid(torch.matmul(x, self.w) + self.hidden_bias)
        hidden_sample = torch.bernoulli(hidden_p)
        visible_recon_p = torch.sigmoid(torch.matmul(hidden_sample, self.w.t()) + self.visible_bias)
        hidden_recon_p = torch.sigmoid(torch.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = torch.matmul(x.t(), hidden_p)
        negative_grad = torch.matmul(visible_recon_p.t(), hidden_recon_p)
        
        device = positive_grad.device
        
        self.delta_w = self._apply_momentum(self.delta_w.to(device = device), 
                                            positive_grad - negative_grad)
        self.delta_visible_bias = self._apply_momentum(self.delta_visible_bias.to(device = device), 
                                                       torch.mean(x - visible_recon_p, dim=0))
        self.delta_hidden_bias = self._apply_momentum(self.delta_hidden_bias.to(device = device), 
                                                      torch.mean(hidden_p - hidden_recon_p, dim=0))

        self.w.data.add_(self.learning_rate * self.delta_w)
        self.visible_bias.data.add_(self.learning_rate * self.delta_visible_bias)
        self.hidden_bias.data.add_(self.learning_rate * self.delta_hidden_bias)
        
        return torch.square(x - visible_recon_p).mean(1).sum(0)

    def _apply_momentum(self, old, new):
        m = self.momentum
        return m * old + (1 - m) * new

    def fit(self, X, epochs=10, batch_size=10):
        assert epochs >= 0, "Number of epochs must be positive"
        index = torch.arange(len(X))
        for epoch in range(epochs):
            torch.randperm(len(X), out=index)
            self.learning_rate *= 0.98
            epoch_loss = 0.0
            for batch in range(len(X) // batch_size):
                torch.cuda.empty_cache()
                xb = X[index[batch * batch_size: (batch + 1) * batch_size]]
                with torch.no_grad():
                    loss = self.step(xb)
                    
                batch_loss = loss.detach().cpu().numpy()
                epoch_loss += batch_loss
            
            epoch_loss /= len(X)
            if np.mod(epoch + 1, 10) == 0:
                print('RBM MSE at epoch {:3.0f}: {:10.6f}'.format(epoch + 1, epoch_loss))

    def get_state(self):
        return {'w': self.w,
                'visible_bias': self.visible_bias,
                'hidden_bias': self.hidden_bias,
                'delta_w': self.delta_w,
                'delta_visible_bias': self.delta_visible_bias}

class Classifier(nn.Module):
    def __init__(self, structure):
        super(Classifier, self).__init__()
        self.structure = structure
        
        self.class_layers = nn.ModuleList()
        self.class_layers.append(nn.Linear(self.structure[-2], self.structure[-1]))
        self.class_layers.append(nn.Softmax(dim=1))
    
    def forward(self, x):
        for layer in self.class_layers:
            x = layer(x)
        return x
    
class Pre_classifier(nn.Module):
    def __init__(self, structure, number_RBM):
        super(Pre_classifier, self).__init__()
        self.structure = structure
        self.number_RBM = number_RBM

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.structure[0], self.structure[1]))
        self.layers.append(nn.Sigmoid())
        
        for i in range(2, len(self.structure) - 1):
            self.layers.append(nn.Linear(self.structure[i - 1], self.structure[i]))
            self.layers.append(nn.Sigmoid())
   
    def set_RBM_layers(self, W_RBM, B_RBM):
        for i in range(self.number_RBM):
            self.layers[i * 2].weight.data = torch.from_numpy(W_RBM[i]).T
            self.layers[i * 2].bias.data = torch.from_numpy(B_RBM[i])
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class DBN_instance(nn.Module):
    def __init__(self, pre_classifier, classifier):
        super(DBN_instance, self).__init__()
        
        self.pre_classifier = pre_classifier
        self.classifier     = classifier

    def forward(self, x):
        x = self.pre_classifier(x)
        x = self.classifier(x)
        return x
    
    
    
    

class deep_belief_network():
    def __init__(self, structure, device = 'cpu'):
        self.structure = structure
        self.device = device
    
    def train(self, X, Y, RBM_weights_old, batch_size = 10, epochs_pre = 100, epochs = 100):
        if not isinstance(X,np.ndarray):
            raise TypeError("Training data is supposed to be numpy array")
        if X.ndim != 2:
            raise ValueError("Training data is supposed to be a 2D numpy array")
        if X.shape[1] != self.structure[0]:
            raise ValueError("Traing data does not match network input size")
            
        if not isinstance(Y,np.ndarray):
            raise TypeError("Training data is supposed to be numpy array")
        if Y.ndim != 2:
            raise ValueError("Training data is supposed to be a 2D numpy array")
        if Y.shape[1] != self.structure[-1]:
            raise ValueError("Traing data does not match network output size") 
        
        # Pretrain network
        number_RBM = len(self.structure) - 2
        
        if number_RBM == len(RBM_weights_old):
            number_RBM_pretrained = len(RBM_weights_old) - 1
            RBM_weights_old = RBM_weights_old[:-1]
        else:
            number_RBM_pretrained = len(RBM_weights_old)
        
        W_RBM = []
        B_RBM = []
        RBM_weights_new = RBM_weights_old
        X_i = torch.from_numpy(np.copy(X)).to(device = self.device, dtype = torch.float32)
        for i in range(number_RBM):
            if i < number_RBM_pretrained:
                W_i, b_i = RBM_weights_old[i]
                
                assert W_i.shape == (self.structure[i], self.structure[i+1])
                assert b_i.size == self.structure[i+1]
            else:
                RBM_i = RBM(self.structure[i],self.structure[i+1])
                RBM_i.to(device = self.device)
                
                RBM_i.fit(X_i, epochs_pre, batch_size)
                data_i = RBM_i.get_state()
                W_i = data_i['w'].detach().cpu().numpy()
                b_i = data_i['hidden_bias'].detach().cpu().numpy()
                
                RBM_weights_new.append([W_i, b_i])
                
            Xl = np.dot(X_i.detach().cpu().numpy(),W_i) + b_i
            X_i = nn.Sigmoid()(torch.from_numpy(Xl).to(device = self.device, dtype = torch.float32))
            W_RBM.append(W_i)
            B_RBM.append(b_i)
        
        # Train combined network
        criterion = nn.BCELoss()
        
        self.DBN_classifier = Classifier(self.structure)
        self.DBN_classifier.to(device = self.device)
        
        self.DBN_RBMs = Pre_classifier(self.structure, number_RBM)
        self.DBN_RBMs.set_RBM_layers(W_RBM, B_RBM)
        self.DBN_RBMs.to(device = self.device)
        
        Xt = torch.from_numpy(X).to(device = self.device, dtype = torch.float32)
        Yt = torch.from_numpy(Y).to(device = self.device, dtype = torch.float32)
        
        with torch.no_grad():
            Xi = self.DBN_RBMs(Xt) 
        
        dataset_class = torch.utils.data.TensorDataset(Xi, Yt)
        dataloader_class = torch.utils.data.DataLoader(dataset_class, batch_size=batch_size, shuffle=True)
       
        optimizer_class = optim.Adam(self.DBN_classifier.parameters(), lr = 0.01)
        
        # Train outer layers
        for epoch in range(epochs_pre):
            epoch_loss = 0.0
            for inputs, targets in dataloader_class:
                optimizer_class.zero_grad()
                outputs = self.DBN_classifier(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                epoch_loss += loss.detach().cpu().numpy() * len(inputs)
                optimizer_class.step()

            if np.mod(epoch + 1, 10) == 0:
                print('Outer Layer training: Loss at epoch {:3.0f}: {:10.6f}'.format(epoch + 1, epoch_loss / len(Xi)))
        
        # Finetune model
        
        self.DBN = DBN_instance(self.DBN_RBMs, self.DBN_classifier)
        self.DBN.to(device = self.device)

        dataset = torch.utils.data.TensorDataset(Xt, Yt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer    = optim.Adam(self.DBN.parameters(), lr = 0.01)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.975)
        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.DBN(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                epoch_loss += loss.detach().cpu().numpy() * len(inputs)
                optimizer.step()
            lr_scheduler.step()
            
            if np.mod(epoch + 1, 10) == 0:
                print('Finetuning: Loss at epoch {:3.0f}: {:10.6f}'.format(epoch + 1, epoch_loss / len(Xt)))
        
        torch.cuda.empty_cache()
        return RBM_weights_new
        
    def predict(self, X):
        X_tensor = torch.from_numpy(X).to(device = self.device, dtype = torch.float32)
        return self.DBN(X_tensor).detach().cpu().numpy()



class DBN(model_template):
    def setup_method(self, l2_regulization = 0.1):
        self.timesteps = max([len(T) for T in self.Input_T_train])
    
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
            self.xmax = np.nanmax(X, axis = 0, keepdims = True)
            self.xmin = np.nanmin(X, axis = 0, keepdims = True)
        
        X = X - self.mean
        X[np.isnan(X)] = 0
        X = X + self.mean
        
        X = (X - self.xmin) / (self.xmax - self.xmin + 1e-5)
        return X, P, class_names
        
        
    def train_method(self):
        # Multiple timesteps have to be flattened
        X, Y, class_names = self.get_data(train = True)
        
        input_dim = X.shape[1]
        output_dim = Y.shape[1]
        
        hidden_shape = []        
        
        # Initialize error terms for number layers
        E_num_layers_old = 10001
        E_num_layers = 10000 

        print('DBN')
        RBM_weights_old = []
        DBN_num_layers = None  
        
        while E_num_layers_old > E_num_layers:
            # Save old network and corresponding error
            E_num_layers_old = E_num_layers
            DBN_num_layers_old = DBN_num_layers
            
            # Add new layer to the DBN
            hidden_shape = hidden_shape + [0]
            
            print('DBN - {} hidden layers'.format(len(hidden_shape)))
            
            ## Create and train new network with optimal number of nodes in new layer
            # Initialize error terms for number nodes
            E_num_nodes_old = 10001
            E_num_nodes = 10000 
            
            # Initialize new network
            DBN_num_nodes = None
            RBM_weights_new = RBM_weights_old
            
            if len(hidden_shape) <= 1:
                hidden_states_step = min(5, max(1, int(input_dim / 20)))
            else:
                hidden_states_step = min(5, max(1, int(hidden_shape[-2] / 20)))
                
            
            while E_num_nodes_old > E_num_nodes:                
                # Save old network and corresponding error
                E_num_nodes_old = E_num_nodes
                DBN_num_nodes_old = DBN_num_nodes
                RBM_weights = RBM_weights_new
                
                # create structure for new model by adding a node to last hidden layer
                hidden_shape[-1] = max(2, hidden_shape[-1] + hidden_states_step)
                structure = [input_dim] + hidden_shape + [output_dim]
                
                
                print('DBN - {} hidden layers - {} nodes in last layer'.format(len(hidden_shape),hidden_shape[-1]))
                # Create and train new network
                DBN_num_nodes = deep_belief_network(structure, self.device)
                RBM_weights_new = DBN_num_nodes.train(X, Y, RBM_weights_old, int(min(len(X), 128)), 50, 200)  
                
                # Evaluate the error corresponding to the new network
                E_num_nodes = np.mean((Y - DBN_num_nodes.predict(X))**2)
                
                print('DBN - {} hidden layers - {} nodes in last layer - loss: {:0.4f}'.format(len(hidden_shape), hidden_shape[-1], E_num_nodes))
            # Use this new network
            DBN_num_layers = DBN_num_nodes_old
            RBM_weights_old = RBM_weights 
            hidden_shape[-1] = hidden_shape[-1] - hidden_states_step
            # Get the error corresponding to the new network
            E_num_layers = E_num_nodes_old
            print('DBN - {} hidden layers -                       - loss: {:0.4f}'.format(len(hidden_shape), E_num_layers))
            
        
        
        self.DBN = DBN_num_layers_old
        hidden_shape = hidden_shape[:-1]
        
        self.structure = [input_dim] + hidden_shape + [output_dim]
    
        print('Final DBN structure: {} - loss: {:0.4f}'.format(self.structure,E_num_layers_old))
        
        DBN_weights = []
        for param in self.DBN.DBN.parameters():
            DBN_weights.append(param.data.detach().cpu().numpy())
        
        self.weights_saved = [self.mean, self.xmin, self.xmax, self.structure, DBN_weights]
        
        
    def load_method(self):
        [self.mean, self.xmin, self.xmax, self.structure, DBN_weights] = self.weights_saved
        
        self.DBN = deep_belief_network(self.structure, self.device) 
        
        self.DBN.DBN = DBN_instance(self.structure, len(self.structure) - 2)
        
        for param, weight in zip(self.DBN.DBN.parameters(), DBN_weights):
            param.data = torch.from_numpy(weight).to(dtype = torch.float32)
        
        self.DBN.DBN.to(device = self.device)
        
        
    def predict_method(self):
        X, _, class_names = self.get_data(train = False)
        
        Probs = self.DBN.predict(X)
        
        self.save_predicted_classifications(Probs, class_names)
    
    
    def check_trainability_method(self):
        return None
    
    
    def get_output_type(self = None):
        # Logit model only produces class outputs
        return 'class'
    
    def get_name(self = None):
        names = {'print': 'Deep belief network (2D inputs)',
                 'file': 'Deep_BN_2D',
                 'latex': r'$\text{\emph{DBN}}_{2D}$'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True
        
    def provides_epoch_loss(self = None):
        return False