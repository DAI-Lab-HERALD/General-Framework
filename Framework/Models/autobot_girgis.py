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


class autobot_girgis(model_template):
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


    def rearange_input_data(self, data):
        # data.shape = bs x num_agents x num_timesteps x 2

        data_old = data.copy()
        assert np.array_equal(self.input_data_type[:2], ['x', 'y']), "Input data type does not start with ['x', 'y']"

        data = np.concatenate([data[...,:2], np.full(data.shape[:3] + (3,), np.nan)], axis = -1)
        missing_timesteps_start = np.isfinite(data[...,:2]).all(-1).argmax(-1) # Num agents x num samples 
        missing_timesteps_end = missing_timesteps_start + np.isfinite(data[...,:2]).all(-1).sum(-1)

        if 'v_x' in self.input_data_type:
            v_x_id = self.input_data_type.index('v_x')
            data[...,2] = data_old[...,v_x_id]
        else:
            for (missing_timestep_start, missing_timestep_end) in np.unique(np.stack([missing_timesteps_start, missing_timesteps_end], -1).reshape(-1,2), axis = 0):
                if np.abs(missing_timestep_end - missing_timestep_start) > 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start:missing_timestep_end, 2] = np.gradient(data[mask, missing_timestep_start:missing_timestep_end, 0], axis = -1)
                elif np.abs(missing_timestep_end - missing_timestep_start) == 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start, 2] = 0
            # Use differentation along 'x' axis to get v_x
            # data[...,2] = np.gradient(data[...,0], axis = -1)
        
        if 'v_y' in self.input_data_type:
            v_y_id = self.input_data_type.index('v_y')
            data[...,3] = data_old[...,v_y_id]
        else:
            for (missing_timestep_start, missing_timestep_end) in np.unique(np.stack([missing_timesteps_start, missing_timesteps_end], -1).reshape(-1,2), axis = 0):
                if np.abs(missing_timestep_end - missing_timestep_start) > 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start:missing_timestep_end, 3] = np.gradient(data[mask, missing_timestep_start:missing_timestep_end, 1], axis = -1)
                elif np.abs(missing_timestep_end - missing_timestep_start) == 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start, 3] = 0
            # Use differentation along 'y' axis to get v_y
            # data[...,3] = np.gradient(data[...,1], axis = -1)
        
        if 'theta' in self.input_data_type:
            theta_id = self.input_data_type.index('theta')
            data[...,4] = data_old[...,theta_id]
        else:
            # Use atan2 between v_y and v_x to get theta
            data[...,4] = np.arctan2(data[...,3], data[...,2])

        # Ensure position data is there
        missing_position = np.isnan(data[...,:2]).any(-1)
        data[missing_position] = np.nan

        # Check if there is additional data missing at available position data
        finite_data = np.isfinite(data)
        any_finite = finite_data.any(-1)
        num_any_finite = any_finite.sum()
        num_all_finite = finite_data.all(-1).sum()
        if num_any_finite != num_all_finite:
            print('    Autobot: Rearranging input data failed')
            failed_cases = (num_any_finite - num_all_finite) / num_any_finite
            print('    Autobot: In {:0.2f}% of cases, there where mixed nan/finite values'.format(100 * failed_cases))

            # Check if positions are always available
            assert np.isfinite(data[any_finite][...,:2]).all(), "There are finite values in v_x or v_y where x or y is nan"

        return data


    def extract_data(self, X, Y, T, graph, Pred_agents):
        # X.shape = (batch_size, num_agents, num_steps_in, num_features)
        # Y.shape = (batch_size, num_agents, num_steps_out, num_features)
        # T.shape = (batch_size, num_agents)
        # S.shape = (batch_size, num_agents)
        # Pred_agents.shape = (batch_size, num_agents)

        Obj_trajs = self.rearange_input_data(X) # Shape (B, num_agents, num_steps_in, 5)
        if Y is not None:
            Obj_trajs_out = self.rearange_input_data(Y) # Shape (B, num_agents, num_steps_out, 5)

        # Find the agent closest to center of the scene
        Pos_mean = np.nanmean(Obj_trajs[:,:,-1,:2], axis=1) # Shape (B, 2)

        Diff = Obj_trajs[:,:,-1,:2] - Pos_mean[:,np.newaxis] # Shape (B, num_agents, 2)
        Dist = np.linalg.norm(Diff, axis=-1) # Shape (B, num_agents)
        Dist[~Pred_agents] += 200

        # Get max num agents
        max_num_agents = np.isfinite(Dist).sum(-1).max()

        ind_agent = np.argsort(Dist, axis=-1) # Shape (B, num_agents)
        ind_sample = np.arange(Dist.shape[0])[:,np.newaxis].repeat(Dist.shape[1], axis=-1) # Shape (B, num_agents)

        ind_agent = ind_agent[:,:max_num_agents]
        ind_sample = ind_sample[:,:max_num_agents]

        Obj_trajs = Obj_trajs[ind_sample, ind_agent] # Shape (B, num_agents, num_steps_in, 5)
        Obj_types = T[ind_sample, ind_agent] # Shape (B, num_agents)
        if Y is not None:
            Obj_trajs_out = Obj_trajs_out[ind_sample, ind_agent]

        # Transform to torch tensors
        Obj_trajs = torch.from_numpy(Obj_trajs).float().to(self.device)
        Obj_trajs_mask = torch.isfinite(Obj_trajs).all(dim=-1).float()
        if Y is not None:
            Obj_trajs_out = torch.from_numpy(Obj_trajs_out).float().to(self.device)
            Obj_trajs_out_mask = torch.isfinite(Obj_trajs_out).all(dim=-1).float()
            # Ensure that only pred agents are used
            Pred_agents_used = torch.from_numpy(Pred_agents[ind_sample, ind_agent]).to(self.device)
            Obj_trajs_out_mask *= Pred_agents_used.unsqueeze(-1).float()

        else:
            Obj_trajs_out = None
            Obj_trajs_out_mask = None

        # get one hot encoding of agent types
        Obj_types_hot = np.zeros((Obj_types.shape[0], Obj_types.shape[1], self.cfg['num_agent_types']))
        Obj_types_hot[Obj_types == '0', 0] = 1
        Obj_types_hot[Obj_types == 'V', 1] = 1
        Obj_types_hot[Obj_types == 'M', 2] = 1
        Obj_types_hot[Obj_types == 'B', 3] = 1
        Obj_types_hot[Obj_types == 'P', 4] = 1 

        Obj_types_hot = torch.from_numpy(Obj_types_hot).float().to(self.device)

        assert (Obj_types_hot.sum(-1) == 1).all(), "One hot encoding failed"

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
            map_polylines = (map_polylines[:,:,1:] + map_polylines[:,:,:-1]) / 2 # Shape (B, num_roads, num_steps_in, map_attr)
            map_polylines_mask = torch.isfinite(map_polylines).all(dim=-1) # Shape (B, num_roads, num_steps_in)


        # Get rotation center and angle
        rot_center = Obj_trajs[:,0,-1,:2].clone() # Shape (batch_size, 2)
        # Translate all the positions to the rotation center
        Obj_trajs[...,:2] = Obj_trajs[...,:2] - rot_center.unsqueeze(1).unsqueeze(1)
        if graph is not None:
            map_polylines = map_polylines - rot_center.unsqueeze(1).unsqueeze(1)
        if Y is not None:
            Obj_trajs_out[...,:2] = Obj_trajs_out[...,:2] - rot_center.unsqueeze(1).unsqueeze(1)

        # Get rotation angle
        if Y is not None:
            rot_angle = 2 * torch.pi * torch.rand((len(Obj_trajs))).to(device = Obj_trajs.device) - torch.pi # Shape (batch_size)

            # Rotate all the positions
            c, s = torch.cos(rot_angle), torch.sin(rot_angle)
            R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)

            Obj_trajs[...,:2]  = torch.matmul(Obj_trajs[...,:2], R.unsqueeze(1))
            Obj_trajs[...,2:4] = torch.matmul(Obj_trajs[...,2:4], R.unsqueeze(1))
            Obj_trajs[...,4]   = Obj_trajs[...,4] - rot_angle.unsqueeze(-1).unsqueeze(-1)

            if graph is not None:
                map_polylines = torch.matmul(map_polylines, R.unsqueeze(1))

            Obj_trajs_out[...,:2]  = torch.matmul(Obj_trajs_out[...,:2], R.unsqueeze(1))
            Obj_trajs_out[...,2:4] = torch.matmul(Obj_trajs_out[...,2:4], R.unsqueeze(1))
            Obj_trajs_out[...,4]   = Obj_trajs_out[...,4] - rot_angle.unsqueeze(-1).unsqueeze(-1)

        # Remove nan values
        Obj_trajs = torch.nan_to_num(Obj_trajs, nan=0.0)
        map_polylines = torch.nan_to_num(map_polylines, nan=0.0)
        if Y is not None:
            Obj_trajs_out = torch.nan_to_num(Obj_trajs_out, nan=0.0)
        
        # Put mask onto the last value
        Obj_trajs = torch.cat([Obj_trajs, Obj_trajs_mask.unsqueeze(-1)], dim=-1)
        if Y is not None:
            Obj_trajs_out = torch.cat([Obj_trajs_out, Obj_trajs_out_mask.unsqueeze(-1)], dim=-1)

        # Divide into ego and agents
        Ego_in = Obj_trajs[:,0] # Shape (B, num_steps_in, 5)
        Agents_in = Obj_trajs[:,1:].permute(0,2,1,3) # Shape (B, num_steps_in, num_agents - 1, 5)
        if Y is not None:
            Ego_out = Obj_trajs_out[:,0]
            Agents_out = Obj_trajs_out[:,1:].permute(0,2,1,3)
        else:
            Ego_out = None
            Agents_out = None

        # Put mask onto polylines
        map_polylines = torch.cat([map_polylines, map_polylines_mask.unsqueeze(-1)], dim=-1)

        # Add agent_dim to polylines
        map_polylines = map_polylines.unsqueeze(1) # Shape (B, 1, num_roads, num_steps_in, map_attr + 1)

        return Ego_in, Ego_out, Agents_in, Agents_out, map_polylines, Obj_types_hot, rot_center, ind_sample, ind_agent
    

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

        for epoch in range(1, self.model_kwargs['max_epochs'] + 1):
            print('    Autobot: Training epoch {}'.format(epoch), flush=True)

            if epoch <= completed_epochs:
                scheduler.step()
                continue

            batch_size = 4
            batch_size_train = self.model_kwargs['train_batch_size']

            epoch_done = False
            batch_train_loss = []
            num_samples = 0
            ind_batch = 0

            # Reset the optimizer
            optimizer.zero_grad()
            accumulated_batch_size = 0
            while not epoch_done:


                X, Y, T, _, _, _, graph, Pred_agents, num_steps, Sample_id, _, epoch_done = self.provide_batch_data('train', batch_size, val_split_size=0.0)
                assert num_steps == self.cfg['future_len']

                accumulated_batch_size += len(X)

                # Transfrom the data to the correct format
                Ego_in, Ego_out, Agents_in, Agents_out, map_polylines, Obj_types_hot, _, _, _ = self.extract_data(X, Y, T, graph, Pred_agents)

                # Set number of agents
                self.model._M = Agents_in.shape[2]

                # Calculate loss
                out_dists, mode_probs = self.model(Ego_in, Agents_in, map_polylines, Obj_types_hot)
                # output_dists.shape: (num_modes, num_time_outputs, batch_size, num_agents, 5)
                # mode_probs.shape: (batch_size, num_modes)

                assert out_dists.isfinite().all(), "Output distribution contains nan values"

                nll_loss, entropy_loss, kl_loss, _, ade_loss, sde_loss, fde_loss, yaw_loss =  nll_loss_multimodes_joint(
                    out_dists, Ego_out, Agents_out, Ego_in, Agents_in, mode_probs,
                    entropy_weight      = self.cfg['entropy_weight'],
                    kl_weight           = self.cfg['kl_weight'],
                    use_FDEADE_aux_loss = True,
                    agent_types         = Obj_types_hot,
                    predict_yaw         = self.cfg['predict_yaw'])
                loss = nll_loss + entropy_loss + kl_loss

                if self.cfg['use_FDEADE_aux_loss']:
                    loss += 100 * (ade_loss + fde_loss)
                    if epoch > 0.8 * self.model_kwargs['max_epochs']:
                        loss += 300 * sde_loss

                    if self.cfg['predict_yaw']:
                        loss += 100 * yaw_loss

                # Get number of used agents
                perc_pred_state = (Ego_out[...,-1].sum() + Agents_out[...,-1].sum()) / (torch.numel(Ego_out[...,-1]) + torch.numel(Agents_out[...,-1]))
                perc_available_agents = (self.model._M + 1) / X.shape[1]
                loss = loss * perc_available_agents

                # Backprop
                loss.backward()

                # Check if there are nan/inf values in the gradients
                for i, param in enumerate(self.model.parameters()):
                    if not torch.isfinite(param.grad).all():
                        print('    Autobot: Gradient contains nan values')
                        assert False

                if epoch_done or (accumulated_batch_size >= batch_size_train):
                    ind_batch += 1
                    print('    Autobot: Training epoch {} - batch {}'.format(epoch, ind_batch), flush=True)

                    # Clip gradient vis norms with self.cfg['grad_clip_norm]
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_clip_norm'])

                    # Update weights
                    optimizer.step()

                    # Reset optimizer
                    optimizer.zero_grad()

                    # Reset accumulated batch size
                    accumulated_batch_size = 0

                # Save loss
                train_loss = np.array([loss.item() / float(perc_available_agents), nll_loss.item(), entropy_loss.item(), kl_loss.item(), 
                                       ade_loss.item(), sde_loss.item(), fde_loss.item(), yaw_loss.item()])
                train_loss /= float(perc_pred_state) # avergae out number of agents
                train_loss *= len(X) * float(perc_available_agents) # avergae out batch size
                batch_train_loss.append(train_loss)
                num_samples += len(X) * float(perc_available_agents)

            # Get epoch loss
            batch_train_loss = np.stack(batch_train_loss, axis=0) # Shape (num_batches, 5)
            epoch_loss = np.sum(batch_train_loss, axis = 0) / num_samples
            self.train_loss[:, epoch - 1] = epoch_loss

            print('    Autobot: Epoch loss: {:0.4f}'.format(epoch_loss[0]))
            print('    Autobot: Total loss: NLL: {:0.4f}, Entropy: {:0.4f}, KL: {:0.4f}, ADE: {:0.4f}, SDE: {:0.4f}, FDE: {:0.4f}, Yaw: {:0.4f}'.format(*epoch_loss[1:]))
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

                X, T, _, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.model_kwargs['eval_batch_size'])

                Ego_in, _, Agents_in, _, map_polylines, Obj_types_hot, rot_center, ind_sample, ind_agent = self.extract_data(X, None, T, graph, Pred_agents)

                # Set number of agents
                self.model._M = Agents_in.shape[2]

                # Adjust Agent_id to reordering
                Agent_id    = Agent_id[ind_sample, ind_agent]
                Pred_agents = Pred_agents[ind_sample, ind_agent]

                out_dists, mode_prob = self.model(Ego_in, Agents_in, map_polylines, Obj_types_hot)
                # output_dists.shape: (num_modes, num_time_outputs, batch_size, num_agents, 5)
                # mode_prob.shape: (batch_size, num_modes)

                out_dists = out_dists.permute(2, 0, 3, 1, 4) # Shape (batch_size, num_modes, num_agents, num_time_outputs, 5)
                # The five values are mu_x, mu_y, sigma_x, sigma_y, rho_xy

                # Build GMM (use multivaraiate normal distribution)
                Pred = []
                for i in range(len(mode_prob)):
                    num_modes = mode_prob[i].shape[0]
                    Pred.append([])
                    for j in range(num_modes):
                        mode_dist_ij = out_dists[i,j] # Shape (num_agents, num_steps_out, 5)

                        dist = Laplace(mode_dist_ij[...,:2], mode_dist_ij[...,2:4])
                        Pred[-1].append(dist.sample([self.num_samples_path_pred]))
                    Pred[-1] = torch.stack(Pred[-1], dim=0)
                
                Pred = torch.stack(Pred, dim=0) # Shape (batch_size, num_modes, num_paths, num_agents, num_steps_out, 2)

                # Randomly select one of the predicted paths alonf the num_modes dimension, according to given probabilites
                indices = torch.multinomial(mode_prob, self.num_samples_path_pred, replacement=True) # Shape (batch_size, num_samples_path_pred)

                batch_ind = torch.arange(Pred.shape[0], device = self.device).unsqueeze(-1).repeat(1, self.num_samples_path_pred)
                paths_ind = torch.arange(self.num_samples_path_pred, device = self.device).unsqueeze(0).repeat(Pred.shape[0], 1)
                Pred = Pred[batch_ind, indices, paths_ind] # Shape (batch_size, num_samples_path_pred, num_agents, num_steps_out, 2)
                
                
                # Translate the predicted paths back
                Pred = Pred + rot_center.unsqueeze(1).unsqueeze(1).unsqueeze(1)

                Pred = Pred.cpu().numpy()

                num_step_pred = Pred.shape[-2]
                if num_steps <= num_step_pred:
                    Pred = Pred[..., :num_steps, :]
                else: 
                    # use linear extrapolation
                    last_vel = Pred[..., [-1],:] - Pred[..., [-2],:] # Shape (batch_size, num_paths, num_agents, 1, 2)
                    steps = np.arange(1, num_steps - num_step_pred + 1).reshape(1, 1, -1, 1)

                    Pred_exp = Pred[..., [-1],:] + last_vel * steps
                    Pred = np.concatenate([Pred, Pred_exp], axis=-2)
                
                # Permute paths and agents dimension
                Pred = Pred.transpose(0, 2, 1, 3, 4) # Shape (batch_size, num_agents, num_samples_path_pred, num_steps_out, 2)

                # save predictions
                self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)

            print('    Autobot: Prediction done')
            print('')


