from model_template import model_template
import numpy as np
import pandas as pd
import torch
import random
import scipy
from TrajFlow.flowModels import *
import pickle
from torch.utils.data import TensorDataset, DataLoader
import os
from mpmath import exp

class flomo_schoeller(model_template):
    '''
    FloMo is a single agent prediction model using Normalizing Flows as its main 
    component.
    
    The code was taken from https://github.com/cschoeller/flomo_motion_prediction
    and the model is published under the following citation:
        
    Schöller, C., & Knoll, A. (2021, September). Flomo: Tractable motion prediction 
    with normalizing flows. In 2021 IEEE/RSJ International Conference on Intelligent 
    Robots and Systems (IROS) (pp. 7977-7984). IEEE.    
    '''
    
    def setup_method(self, seed = 0):
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.batch_size = 128

        # Required attributes of the model
        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = True
        self.can_use_map = True
        # If self.can_use_map, the following is also required
        self.target_width = 257
        self.target_height = 156
        self.grayscale = True
        
        self.norm_rotation = True
        
        self.hs_rnn = 16
        self.n_layers_rnn = 3
        self.fut_enc_sz = 4

        self.scene_encoding_size = 4
        self.obs_encoding_size = 16 
        
        if (self.provide_all_included_agent_types() == 'P').all():
            self.beta_noise = 0.2
            self.gamma_noise = 0.02
            
            self.alpha = 10
            self.s_min = 0.3
            self.s_max = 1.7
            self.sigma = 0.5

        else:
            self.beta_noise = 0.002
            self.gamma_noise = 0.002
            
            self.alpha = 3
            self.s_min = 0.8
            self.s_max = 1.2
            self.sigma = 0.2

        self.flow_epochs = 200
        self.flow_lr = 1e-3
        self.flow_wd = 1e-5

        self.std_pos_ped = 1
        self.std_pos_veh = 1 #80        
        
        
    def extract_batch_data(self, X, T, Y = None, img = None):
        
        # Get type of agents
        T_out = T.astype(str)
        Ped_agents = T_out == 'P'
        
        # Transform types to numbers
        T_out[T_out == 'nan'] = '0'
        T_out = np.fromstring(T_out.reshape(-1), dtype = np.uint32).reshape(*T_out.shape, int(str(T_out.astype(str).dtype)[2:])).astype(np.uint8)[:,:,0]
        T_out = torch.from_numpy(T_out).to(device = self.device)
        
        # Normalize positions
        X = (X - self.min_pos) / (self.max_pos - self.min_pos)
        
        # Standardize positions
        X[Ped_agents]  /= self.std_pos_ped
        X[~Ped_agents] /= self.std_pos_veh
        X = torch.from_numpy(X).float().to(device = self.device)
        
        if Y is not None:
            # Normalize future positions
            Y = (Y - self.min_pos) / (self.max_pos - self.min_pos)
            
            # Standardize future positions
            Y[Ped_agents[:,0]]  /= self.std_pos_ped
            Y[~Ped_agents[:,0]] /= self.std_pos_veh
            Y = torch.from_numpy(Y).float().to(device = self.device)
        
        if img is not None:
            img = torch.from_numpy(img).float().to(device = self.device) / 255
            
        return X, T_out, Y, img
        
    
    

    def train_flow(self, T_all):
        use_map = self.can_use_map and self.has_map
                
        if use_map:
            scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
        else:
            scene_encoder = None
        
        flow_dist = FloMo_I(pred_steps=self.max_t_O_train, alpha=self.alpha, beta=self.beta_noise, 
                            gamma=self.gamma_noise, scene_encoder=scene_encoder, 
                            norm_rotation=self.norm_rotation, device=self.device,
                            obs_encoding_size=self.obs_encoding_size, 
                            scene_encoding_size=self.scene_encoding_size, n_layers_rnn=self.n_layers_rnn, 
                            es_rnn=self.hs_rnn, hs_rnn=self.hs_rnn, use_map=use_map, 
                            n_layers_gnn=4, es_gnn=32, T_all = T_all)
        
        
        flow_dist_file = self.model_file[:-4] + '_NF'

        if os.path.isfile(flow_dist_file) and not self.data_set.overwrite_results:
            flow_dist = pickle.load(open(flow_dist_file, 'rb'))
                          
        else:
            optimizer = torch.optim.AdamW(flow_dist.parameters(), lr=self.flow_lr, weight_decay=self.flow_wd)

            val_losses = []


            for step in range(self.flow_epochs):

                flow_dist.train()
                
                losses_epoch = []
                val_losses_epoch = []
                
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, T, img, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                           val_split_size = 0.1)
                    X, T, Y, img = self.extract_batch_data(X, T, Y, img)
                    
                    # X.shape:   bs x num_agents x num_timesteps_is x 2
                    # Y.shape:   bs x 1 x num_timesteps_is x 2
                    # T.shape:   bs x num_agents
                    # img.shape: bs x 1 x 156 x 257 x 1

                    scaler = torch.tensor(scipy.stats.truncnorm.rvs((self.s_min-1)/self.sigma, (self.s_max-1)/self.sigma,
                                                                     loc=1, scale=self.sigma, size=X.shape[0])).float()
                    
                    scaler = scaler.unsqueeze(1)
                    scaler = scaler.unsqueeze(2)
                    scaler = scaler.unsqueeze(3)
                    scaler = scaler.to(device = self.device)
                    
                    # TODO: check if this should be using all_pos or tar_pos
                    all_pos_past   = X
                    tar_pos_past   = X[:,0]
                    all_pos_future = Y
                    tar_pos_future = Y[:,0]                
                    
                    mean_pos = torch.mean(torch.concat((tar_pos_past, tar_pos_future), dim = 1), dim=1, keepdims = True).unsqueeze(1)
                    
                    shifted_past = all_pos_past - mean_pos
                    shifted_future = all_pos_future - mean_pos
                        
                    past_data = shifted_past * scaler + mean_pos
                    future_data = shifted_future * scaler + mean_pos
                    
                    optimizer.zero_grad()
                    
                    if img is not None:
                        img = img[:,0].permute(0,3,1,2)
                    
                    if img is not None:
                        logprob = flow_dist.log_prob(future_data, past_data, T, img)
                    else:
                        logprob = flow_dist.log_prob(future_data, past_data, T)

                    loss = -torch.mean(logprob) # NLL
                    losses_epoch.append(loss.item())
                    
                    loss.backward()
                    optimizer.step()
                    
                    
                flow_dist.eval()
                with torch.no_grad():
                    val_epoch_done = False
                    while not val_epoch_done:
                        X, Y, T, img, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                val_split_size = 0.1)
                        X, T, Y, img = self.extract_batch_data(X, T, Y, img)
                        
                        past_data_val = X
                        future_data_val = Y
                                               
                        
                        if img is not None:
                            img_val = img[:,0].permute(0,3,1,2)
                            
                        optimizer.zero_grad()

                        if img is not None:
                            log_prob = flow_dist.log_prob(future_data_val, past_data_val, T, img_val)
                        else:
                            log_prob = flow_dist.log_prob(future_data_val, past_data_val, T)
                    
                        val_loss = -torch.mean(log_prob)
                        val_losses_epoch.append(val_loss.item())
                        
                    val_losses.append(np.mean(val_losses_epoch))      
                
                # Check for convergence
                if step > 50:
                    best_val_step = np.argmin(val_losses)
                    if step - best_val_step > 10:
                        print('Converged')
                        print('step: {}, loss:     {}'.format(step, np.mean(losses_epoch)))
                        print('step: {}, val_loss: {}'.format(step, np.mean(val_losses_epoch)))
                        break     
                
                if step % 10 == 0:

                    print('step: {}, loss: {}'.format(step, np.mean(losses_epoch)))
                    print('step: {}, val_loss: {}'.format(step, np.mean(val_losses_epoch)))

            self.train_loss[0, :len(val_losses)] = np.array(val_losses)
            os.makedirs(os.path.dirname(flow_dist_file), exist_ok=True)
            pickle.dump(flow_dist, open(flow_dist_file, 'wb'))

        return flow_dist


    def train_method(self):    
        self.train_loss = np.ones((1, self.flow_epochs)) * np.nan

        # Get needed agent types
        T_all = self.provide_all_included_agent_types().astype(str)
        T_all = np.fromstring(T_all, dtype = np.uint32).reshape(len(T_all), int(str(T_all.astype(str).dtype)[2:])).astype(np.uint8)[:,0]

        # Prepare stuff for Normalization
        if self.data_set.get_name()['file'] == 'Fork_P_Aug':
            X, Y, _, _, _, _, _, _ = self.provide_all_training_trajectories()
            traj_tar = np.concatenate((X[:,0], Y[:,0]), dim = 1)
            self.max_pos = torch.max(traj_tar)
            self.min_pos = torch.min(traj_tar)
        else:
            self.min_pos = 0.0
            self.max_pos = 1.0

        # Train model components 
        self.flow_dist = self.train_flow(T_all)
        
        # save weigths 
        self.weights_saved = [self.min_pos, self.max_pos]
        
        
    def load_method(self):
        self.min_pos, self.max_pos = self.weights_saved

        flow_dist_file = self.model_file[:-4] + '_NF'
        self.flow_dist = pickle.load(open(flow_dist_file, 'rb'))
        
    def _repeat_rowwise(self, x, n):
        org_dim = x.size(-1)
        x = x.repeat(1, 1, n)
        return x.view(-1, n, org_dim)


    def predict_method(self):
        prediction_done = False
        while not prediction_done:
            X, T, img, _, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.batch_size)
            actual_batch_size = len(X)
            Ped_agent = T == 'P'
            
            X, T, _, img = self.extract_batch_data(X, T, img = img)
            # Run prediction pass
            with torch.no_grad():
                                
                if img is not None:
                    img = img[:,0].permute(0,3,1,2)
                else:
                    img = None
                
                if img is not None: 
                    samples, log_probs = self.flow_dist.sample(self.num_samples_path_pred, X, T, img)
                else:
                    samples, log_probs = self.flow_dist.sample(self.num_samples_path_pred, X, T)
                
                samples = samples.squeeze(0)
                        
                Y_pred = samples.detach()
                    
            Pred = Y_pred.detach().cpu().numpy()
            if len(Pred.shape) == 3:
                Pred = Pred[np.newaxis]
            
            Pred[Ped_agent[:,0]]  *= self.std_pos_ped
            Pred[~Ped_agent[:,0]] *= self.std_pos_veh

            if self.data_set.get_name()['file'] == 'Fork_P_Aug':
                Pred = Pred * (self.max_pos - self.min_pos) + self.min_pos
            
            torch.cuda.empty_cache()
            
            assert False
            
            # extrapolate if needed
            if num_steps > Pred.shape[-2]:
                step_delta = Pred[...,-1,:] - Pred[...,-2,:]
                step_delta = step_delta[...,np.newaxis,:]
                
                steps = np.arange(1, num_steps + 1 - Pred.shape[-2])
                steps = steps[np.newaxis,np.newaxis,:,np.newaxis]
                
                Pred_delta = Pred[...,[-1],:] + step_delta * steps
                
                Pred = np.concatenate((Pred, Pred_delta), axis = -2)
            
            # save predictions
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id)
    
    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_name(self = None):
        names = {'print': 'FloMo',
                 'file': 'FloMo_traj',
                 'latex': r'\emph{FM}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True
        
    def provides_epoch_loss(self = None):
        return True