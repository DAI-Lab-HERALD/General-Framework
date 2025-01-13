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

from Autobot_unitraj.model.autobot import AutoBotEgo
from torch.distributions import MultivariateNormal




class autobot_unitraj(model_template):
    '''
    This is the version of Autobot-Ego. The code was taken from its implementation
    in the Unitraj prediction framework 
    (https://github.com/vita-epfl/UniTraj/tree/main/unitraj/models/autobot).
    
    The original paper is cited as:
        
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

        model_str = 'Autobot_Ego' + kwargs_str + '_seed' + seed_str

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
            cfg_path = path + os.sep + 'Autobot_unitraj' + os.sep + 'cfg' + os.sep + 'autobot.yaml'
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
        self.predict_single_agent = True
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

        # Initialize the model
        self.model = AutoBotEgo(self.cfg)


    def extract_data(self, X, Y, graph):
        # X.shape = (batch_size, num_agents, num_steps_in, num_features)
        # Y.shape = (batch_size, num_agents, num_steps_out, num_features)
        # T.shape = (batch_size, num_agents)
        # S.shape = (batch_size, num_agents)
        # Pred_agents.shape = (batch_size, num_agents)

        Obj_trajs = torch.from_numpy(X[...,:2]).float().to(self.device)
        Obj_trajs_mask = torch.isfinite(Obj_trajs).all(dim=-1)

        
        Track_index_to_predict = torch.zeros(Obj_trajs.shape[0], dtype=torch.long).to(self.device)

        if Y is not None:
            Center_GT_trajs = torch.from_numpy(Y[:,0,...,:2]).float().to(self.device) # Shape (B, num_steps_out, 2)
            Center_GT_trajs_mask = torch.isfinite(Center_GT_trajs).all(dim=-1)
        else:
            Center_GT_trajs = None
            Center_GT_trajs_mask = None

        # Get graphs
        if graph is None:
            # Assume num_roads = 0, and num_steps_in = 0
            map_polylines = torch.zeros((Obj_trajs.shape[0], 0, 0, 2)).float().to(self.device)
            map_polylines_mask = torch.zeros((Obj_trajs.shape[0], 0, 0)).bool().to(self.device)
        else:
            map_polylines = []
            max_len_batch = 0
            for i in range(len(graph)):
                centerlines = graph[i].centerlines
                max_len_scene = max([len(centerline) for centerline in centerlines])
                c = np.stack([np.pad(centerline, ((0,max_len_scene - len(centerline)),(0,0)), constant_values=np.nan) for centerline in centerlines], axis=0)

                max_len_batch = max(max_len_batch, max_len_scene)
                map_polylines.append(c)

            max_num_roads = max([len(centerlines) for centerlines in map_polylines])
            for i in range(len(map_polylines)):
                num_roads, len_roads = map_polylines[i].shape[:2]
                map_polylines[i] = np.pad(map_polylines[i], ((0,max_num_roads - num_roads),(0,max_len_batch - len_roads),(0,0)), constant_values=np.nan)
            
            map_polylines = torch.from_numpy(np.stack(map_polylines, axis=0)).float().to(self.device)
            # Use the midpoint petween points instead of points
            map_polylines = (map_polylines[:,:,1:] + map_polylines[:,:,:-1]) / 2
            map_polylines_mask = torch.isfinite(map_polylines).all(dim=-1)

        # Get rotation center and angle
        rot_center = Obj_trajs[:,0,-1,:2].clone() # Shape (batch_size, 2)
        rot_angle = torch.atan2(rot_center[:,1] - Obj_trajs[:,0,-2,1], rot_center[:,0] - Obj_trajs[:,0,-2,0]) # Shape (batch_size)

        # Translate all the positions to the rotation center
        Obj_trajs = Obj_trajs - rot_center.unsqueeze(1).unsqueeze(1)
        if graph is not None:
            map_polylines = map_polylines - rot_center.unsqueeze(1).unsqueeze(1)

        # Rotate all the positions
        c, s = torch.cos(rot_angle), torch.sin(rot_angle)
        R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)

        Obj_trajs = torch.matmul(Obj_trajs, R.unsqueeze(1))
        if graph is not None:
            map_polylines = torch.matmul(map_polylines, R.unsqueeze(1))

        # Remove nan values
        Obj_trajs = torch.nan_to_num(Obj_trajs, nan=0.0)
        if Center_GT_trajs is not None:
            Center_GT_trajs = Center_GT_trajs - rot_center.unsqueeze(1)
            Center_GT_trajs = torch.matmul(Center_GT_trajs, R)
            Center_GT_trajs = torch.nan_to_num(Center_GT_trajs, nan=0.0)
        map_polylines = torch.nan_to_num(map_polylines, nan=0.0)

        inputs = {
            'obj_trajs': Obj_trajs, # Should torch.tensor of shape (batch_size, num_agents, num_steps_in, 2) 
            'obj_trajs_mask': Obj_trajs_mask, # Should torch.tensor of shape (batch_size, num_agents, num_steps_in)
            'track_index_to_predict': Track_index_to_predict, # Should be torch.tensor of shape (batch_size)
            'center_gt_trajs': Center_GT_trajs, # Should torch.tensor of shape (batch_size, num_steps_out, 2)
            'center_gt_trajs_mask': Center_GT_trajs_mask, # Should torch.tensor of shape (batch_size, num_steps_out)
            'map_polylines': map_polylines, # Should torch.tensor of shape (batch_size, num_road_segemnts, num_pts_per_segment, 2)
            'map_polylines_mask': map_polylines_mask, # Should torch.tensor of shape (batch_size, num_road_segemnts, num_pts_per_segment)
        }


        batch = {'input_dict': inputs}
        return batch, rot_angle, rot_center
    
    
    def train_method(self):
        # initialize optimizer
        [optimizer], [scheduler] = self.model.configure_optimizers()

        # Put model to device
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
        
        else:
            completed_epochs = 0
            self.train_loss = np.ones((1, self.model_kwargs['max_epochs'])) * np.nan

        for epoch in range(1, self.model_kwargs['max_epochs'] + 1):
            print('    Autobot: Training epoch {}'.format(epoch))

            if epoch <= completed_epochs:
                scheduler.step()
                continue

            batch_size = self.cfg['train_batch_size']

            epoch_done = False
            batch_train_loss = []
            num_samples = 0
            while not epoch_done:
                X, Y, _, _, _, _, graph, _, num_steps, _, _, epoch_done = self.provide_batch_data('train', batch_size, val_split_size=0.0)
                assert num_steps == self.cfg['future_len']

                # Transfrom the data to the correct format
                batch, _, _ = self.extract_data(X, Y, graph)

                # Calculate loss
                _, loss = self.model(batch)

                # Backprop
                optimizer.zero_grad()
                loss.backward()

                # Clip gradient vis norms with self.cfg['grad_clip_norm]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_clip_norm'])

                # Update weights
                optimizer.step()

                # Save loss
                batch_train_loss.append(loss.item() * len(X))
                num_samples += len(X)

            # Get epoch loss
            epoch_loss = np.sum(batch_train_loss) / num_samples
            self.train_loss[0, epoch - 1] = epoch_loss

            print('    Autobot: Epoch loss: {}'.format(epoch_loss))
            print('')
            
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

                X, _, _, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.model_kwargs['eval_batch_size'])
                batch, rot_angle, rot_center = self.extract_data(X, None, graph)

                assert (batch['input_dict']['obj_trajs_mask'].sum(1) > 0).all(), "At least one agent should be present in the scene"

                output_dist = self.model(batch, eval = True)

                # Sample trajectories
                mode_prob = output_dist['predicted_probability'] # batch_size x num_modes
                mode_dist = output_dist['predicted_trajectory'] # batch_size x num_modes x num_steps_out x 5
                # The five values are mu_x, mu_y, sigma_x, sigma_y, rho_xy

                # Build GMM (use multivaraiate normal distribution)
                Pred = []
                for i in range(len(mode_prob)):
                    num_modes = mode_prob[i].shape[0]
                    Pred.append([])
                    for j in range(num_modes):
                        mode_dist_ij = mode_dist[i,j]
                        mu_x = mode_dist_ij[...,0]
                        mu_y = mode_dist_ij[...,1]
                        sigma_x = mode_dist_ij[...,2]
                        sigma_y = mode_dist_ij[...,3]
                        rho_xy = mode_dist_ij[...,4]

                        cov = torch.zeros((len(mu_x),2,2), dtype = torch.float32, device = self.device)
                        cov[...,0,0] = sigma_x ** 2
                        cov[...,1,1] = sigma_y ** 2
                        cov[...,0,1] = sigma_x * sigma_y * rho_xy
                        cov[...,1,0] = sigma_x * sigma_y * rho_xy

                        dist = MultivariateNormal(torch.stack([mu_x, mu_y], dim=-1), cov)
                        Pred[-1].append(dist.sample([self.num_samples_path_pred]))
                    Pred[-1] = torch.stack(Pred[-1], dim=0)
                
                Pred = torch.stack(Pred, dim=0) # Shape (batch_size, num_modes, num_paths, num_steps_out, 2)

                # Randomly select one of the predicted paths alonf the num_modes dimension, according to given probabilites
                indices = torch.multinomial(mode_prob, self.num_samples_path_pred, replacement=True) # Shape (batch_size, num_samples_path_pred)

                batch_ind = torch.arange(Pred.shape[0], device = self.device).unsqueeze(-1).repeat(1, self.num_samples_path_pred)
                paths_ind = torch.arange(self.num_samples_path_pred, device = self.device).unsqueeze(0).repeat(Pred.shape[0], 1)
                Pred = Pred[batch_ind, indices, paths_ind] # Shape (batch_size, num_samples_path_pred, num_steps_out, 2)
                
                # Rotate the predicted paths back
                c, s = torch.cos(-rot_angle), torch.sin(-rot_angle)
                R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2) # Shape (batch_size, 2, 2)
                
                Pred = torch.matmul(Pred, R.unsqueeze(1))
                Pred = Pred + rot_center.unsqueeze(1).unsqueeze(1)

                Pred = Pred.cpu().numpy()

                num_step_pred = Pred.shape[-2]
                if num_steps <= num_step_pred:
                    Pred = Pred[..., :num_steps, :]
                else: 
                    # use linear extrapolation
                    last_vel = Pred[..., [-1],:] - Pred[..., [-2],:] # Shape (batch_size, num_paths, 1, 2)
                    steps = np.arange(1, num_steps - num_step_pred + 1).reshape(1, 1, -1, 1)

                    Pred_exp = Pred[..., [-1],:] + last_vel * steps
                    Pred = np.concatenate([Pred, Pred_exp], axis=-2)
                
                # save predictions
                self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)
            
            print('    Autobot: Prediction done')
            print('')



