# import horovod.torch as hvd
import numpy as np
import os
import pickle
import random
import sys
import time
import torch
from omegaconf import OmegaConf

from model_template import model_template
from Autobot.models.autobot_joint import AutoBotJoint
from Autobot.utils.train_helpers import nll_loss_multimodes_joint
from torch.distributions import Laplace


class autobot_girgis_test(model_template):
    '''
    This is the version of Autobot-Joint, the joint prediction version of AutoBot. 
    The code was taken from https://github.com/roggirg/AutoBots/tree/master, and
    the work should be cited as:
        
    Girgis, R., Golemo, F., Codevilla, F., Weiss, M., D'Souza, J. A., Kahou, S. E., ... & Pal, C. 
    Latent Variable Sequential Set Transformers for Joint Multi-Agent Motion Prediction. 
    In International Conference on Learning Representations.
    '''
    def get_name(self = None):
        r'''
        Provides a dictionary with the different names of the model
            
        Returns
        -------
        names : dict
        The first key of names ('print')  will be primarily used to refer to the model in console outputs. 
                
        The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
        (for any operating system), as it is mostly used to indicate that certain result files belong to this model. 
                
        The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
        latex commands - such as using '$$' for math notation.
            
        '''
        self.define_default_kwargs()

        # TODO: Implement
        kwargs_str = ''
        seed_str = str(self.model_kwargs['seed'])

        model_str = 'Autobot' + kwargs_str + '_seed' + seed_str

        names = {'print': model_str,
                'file': model_str,
                'latex': r'\emph{%s}' % model_str}

        return names
  
    def requires_torch_gpu(self = None):
        r'''
        If True, then the model will use pytorch on the gpu.
            
        Returns
        -------
        pytorch_decision : bool
            
        '''
        return True

    def get_output_type(self = None):
        r'''
        This returns a string with the output type:
        The possibilities are:
        'path_all_wo_pov' : This returns the predicted trajectories of all agents except the pov agent (defined
        in scenario), if this is for example assumed to be an AV.
        'path_all_wi_pov' : This returns the predicted trajectories of all designated agents, including the
        pov agent.
        'class' : This returns the predicted probability that some class of behavior will be observable
        in the future.
        'class_and_time' : This predicted both the aforementioned probabilities, as well as the time at which
        the behavior will become observable.
            
        Returns
        -------
        output_type : str
            
        '''
        return 'path_all_wi_pov'

    def check_trainability_method(self):
        r'''
        This function potentially returns reasons why the model is not applicable to the chosen scenario.
            
        Returns
        -------
        reason : str
        This str gives the reason why the model cannot be used in this instance. If the model is usable,
        return None instead.
            
        '''
        return None
    
    def define_default_kwargs(self):
        if not ('seed' in self.model_kwargs.keys()):
            self.model_kwargs["seed"] = 42
        
        # Go through the cfg file in autobot.cfg.autobot.yaml
        # Check if the key is in the model_kwargs, if not, add it, and set 
        # the value to the one in the .yaml file
        
        # Get path to this file
        path = os.path.dirname(os.path.abspath(__file__))

        # Load yaml file
        if not hasattr(self, 'cfg'):
            cfg_path = path + os.sep + 'Autobot' + os.sep + 'cfg' + os.sep + 'autobot.yaml'
            self.cfg = OmegaConf.load(cfg_path)

        # Go through the cfg file
        for key in self.cfg.keys():
            if not (key in self.model_kwargs.keys()):
                self.model_kwargs[key] = self.cfg[key]
            
            else:
                self.cfg[key] = self.model_kwargs[key]
        

        

        

    def setup_method(self):
        self.define_default_kwargs()

        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = False
        self.can_use_map = False
        self.can_use_graph = True
        self.sceneGraph_radius = self.model_kwargs['map_range']

        seed = self.model_kwargs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


        # add required keys to config file
        self.cfg['past_len'] = self.num_timesteps_in
        self.cfg['future_len'] = self.num_timesteps_out

        # Check if map is available
        self.cfg['use_map'] = self.can_use_graph and self.has_graph

        # get number of possible config types
        self.cfg['num_agent_types'] = 5 # Pedestrian, Cyclist, Motorcylce, Vehicle, None

        # Initialize the model
        self.model = AutoBotJoint(self.cfg)


    def extract_data(self, X, T, graph):
        # X.shape = (batch_size, num_agents, num_steps_in, num_features)
        # T.shape = (batch_size, num_agents)
        
        # Only take last timestep
        X = X[..., :2] # (batch_size, num_agents, num_timesteps 2)
        
        # Transform to torch tensor
        X = torch.from_numpy(X).float().to(self.device)
        
        # Find vehicles
        vehicles = T == 'V'
        vehicles = torch.from_numpy(vehicles).to(self.device) # (batch_size, num_agents)
        
        max_dist = []
        # Go through samples
        for i in range(X.shape[0]):
            # Get data
            X_i = X[i] # (num_agents, num_timesteps, 2)
            vehicles_i = vehicles[i] # (num_agents)
            
            # Get vehicle psoitions
            X_vehicles = X_i[vehicles_i] # (num_vehicles, num_timesteps, 2)
            
            # Get the centerlines
            graph_i = graph[i]
            centerlines = graph_i['centerlines']
            
            # Concatenate the centerlines
            centerlines = np.concatenate(centerlines, axis=0) # (num_points, 2)
            
            # To torch tensor
            centerlines = torch.from_numpy(centerlines).float().to(self.device).unsqueeze(0) # (1, num_points, 2)
            
            # Get extent of centerlines
            centerlines_min = centerlines.min(dim=1, keepdim=True).values # (1, 1, 2)
            centerlines_max = centerlines.max(dim=1, keepdim=True).values # (1, 1, 2)
            
            # Get useful X_positions
            pos_useful = ((X_vehicles >= centerlines_min) & (X_vehicles <= centerlines_max)).all(dim=-1) # (num_vehicles, num_timesteps)
            
            # Get for each vehicle closest centerline point
            Dist = torch.cdist(X_vehicles, centerlines) # (num_vehicles, num_timesteps, num_points)
            
            # Get the minimum distance
            min_dist, _ = torch.min(Dist, dim=-1) # (num_vehicles, num_timesteps)
            
            # Get the maximum distance
            max_dist_i = torch.max(min_dist[pos_useful]).item() # (1)
            max_dist.append(max_dist_i)
            
            
            if max_dist_i > 5:
                
                import matplotlib.pyplot as plt
                plt.figure()
                c = centerlines[0].cpu().numpy()
                plt.scatter(c[:,0], c[:,1], c = 'k', marker='o', s=0.1)
                
                x = X_vehicles[pos_useful].cpu().numpy() # (n, 2)
                plt.scatter(x[:,0], x[:,1], c = 'r', marker='o', s=0.1)
                    
                plt.axis('equal')
                plt.show()

        return np.array(max_dist)
    

    def train_method(self):
        # initialize optimizer
        [optimizer], [scheduler] = self.model.configure_optimizers()

        # Put model to device
        self.model.train()
        self.model.to(self.device)

        # load model
        intermediate_file = self.model_file[:-4] + '_intermediate.npy'
        if os.path.exists(intermediate_file):
            data = np.load(intermediate_file, allow_pickle=True)

            Model_weights = data[0]
            self.train_loss = data[1]
            completed_epochs = data[2]

            # Overwrite models parameters
            Weights = list(self.model.parameters())
            with torch.no_grad():
                for i, weights_loaded in enumerate(Model_weights):
                    weights_loaded_torch = torch.from_numpy(weights_loaded)
                    assert Weights[i].shape == weights_loaded_torch.shape, "Model weights do not match"
                    Weights[i][:] = weights_loaded_torch[:]

            # Check if train loss needs to be appended
            if self.train_loss.shape[1] < self.model_kwargs['max_epochs']:
                self.train_loss = np.concatenate([self.train_loss, np.ones((8, self.model_kwargs['max_epochs'] - self.train_loss.shape[1])) * np.nan], axis=1)

            # Load data if needed
            if completed_epochs >= self.model_kwargs['max_epochs']:
                self.weights_saved = Model_weights
                return
        
        else:
            completed_epochs = 0
            self.train_loss = np.ones((8, self.model_kwargs['max_epochs'])) * np.nan

        for epoch in range(1, 2):
            print('    Autobot: Training', flush=True)

            if epoch <= completed_epochs:
                scheduler.step()
                continue

            batch_size = 32

            epoch_done = False

            # Reset the optimizer
            optimizer.zero_grad()
            accumulated_batch_size = 0
            max_dist_epoch = 0
            while not epoch_done:


                X, Y, T, _, _, _, graph, Pred_agents, num_steps, Sample_id, _, epoch_done = self.provide_batch_data('train', batch_size, val_split_size=0.0)
                assert num_steps == self.cfg['future_len']

                accumulated_batch_size += len(X)

                # Transfrom the data to the correct format
                max_dist = self.extract_data(X, T, graph)
                    
                
                Locations = self.data_set.Domain.iloc[Sample_id].location
                print('        Max_dist at location {}: {} m'.format(Locations.iloc[0], max_dist.max()))
                
                max_dist_epoch = max(max_dist_epoch, max_dist.max())

            # Get epoch loss
            self.train_loss[:, epoch - 1] = 0.0
            
            # Run scheduler
            scheduler.step()

            # Save intermediate model
            Weights = list(self.model.parameters())
            Weights_saved = []
            for weigths in Weights:
                Weights_saved.append(weigths.detach().cpu().numpy())

            os.makedirs(os.path.dirname(intermediate_file), exist_ok=True)
            np.save(intermediate_file, [Weights_saved, self.train_loss, epoch])
        
            # Save model
            self.weights_saved = Weights_saved
        


    def save_params_in_csv(self = None):
        r'''
        If True, then the model's parameters will be saved to a .csv file for easier access.
            
        Returns
        -------
        csv_decision : bool
            
        '''
        return False

    def provides_epoch_loss(self = None):
        r'''
        If True, then the model's epoch loss will be saved.
            
        Returns
        -------
        loss_decision : bool
            
        '''
        return True

    def load_method(self):
        # Overwrite models parameters
        Weights = list(self.model.parameters())
        with torch.no_grad():
            for i, weights_loaded in enumerate(self.weights_saved):
                weights_loaded_torch = torch.from_numpy(weights_loaded)
                assert Weights[i].shape == weights_loaded_torch.shape, "Model weights do not match"
                Weights[i][:] = weights_loaded_torch[:]

    def predict_method(self):
        prediction_done = False
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():     
            ind_batch = 0    
            while not prediction_done:
                ind_batch = ind_batch + 1
                print('    Autobot: Predicting batch {}'.format(ind_batch))

                X, T, _, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.model_kwargs['eval_batch_size'])
                max_dist = self.extract_data(X, T, graph)

                Locations = self.data_set.Domain.iloc[Sample_id].location
                print('        Max_dist at location {}: {} m'.format(Locations.iloc[0], max_dist))

            print('    Autobot: Prediction done')
            print('')


