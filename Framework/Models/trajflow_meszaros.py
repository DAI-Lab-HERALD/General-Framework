from model_template import model_template
import numpy as np
import pandas as pd
import torch
import random
import scipy
from TrajFlow.flowModels import TrajFlow_I, TrajFlow, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder
import pickle
from torch.utils.data import TensorDataset, DataLoader
import os

class trajflow_meszaros(model_template):
    
    def setup_method(self, seed = 0):        
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.batch_size = 128
        # Get params
        self.num_timesteps_in = len(self.Input_path_train.to_numpy()[0,0])
        self.num_timesteps_out = np.zeros(len(self.Output_T_train), int)
        for i_sample in range(self.Output_T_train.shape[0]):
            len_use = len(self.Output_T_train[i_sample])
            self.num_timesteps_out[i_sample] = len_use

        self.future_traj_len = self.data_set.num_timesteps_out_real
        self.norm_rotation = True
        
        self.use_map = self.data_set.includes_images()
        self.target_width = 257
        self.target_height = 156
        
        self.past_traj_len = self.data_set.num_timesteps_in_real
        self.remain_samples = self.num_timesteps_out >= self.future_traj_len 
        self.num_timesteps_out = np.minimum(self.num_timesteps_out[self.remain_samples], self.future_traj_len )

        
        self.hs_rnn = 16
        self.n_layers_rnn = 3
        self.fut_enc_sz = 4
        
        if (np.array([name[0] for name in np.array(self.input_names_train)]) == 'P').all():
            self.beta_noise = 0.2
            self.gamma_noise = 0.02
            self.scene_encoding_size = 4
            
            self.obs_encoding_size = 16 
            
            self.alpha = 10
            self.s_min = 0.3
            self.s_max = 1.7
            self.sigma = 0.5

        else:
            self.beta_noise = 0
            self.gamma_noise = 0 
            
            self.scene_encoding_size = 4
            self.obs_encoding_size = 4
            self.alpha = 3
            self.s_min = 0.8
            self.s_max = 1.2
            self.sigma = 0.2


        self.fut_ae_epochs = 1000
        self.fut_ae_lr = 5e-4
        self.fut_ae_wd = 1e-4

        self.flow_epochs = 200
        self.flow_lr = 1e-3
        self.flow_wd = 1e-5

        self.std_pos_ped = 1
        self.std_pos_veh = 80
        
        
        # Set time step
        self.dt = self.Input_T_train[0][-1] - self.Input_T_train[0][-2]
        
        
        
    def extract_data(self, train = True):
        
        if train:
            X_help = self.Input_path_train.to_numpy()
            Y_help = self.Output_path_train.to_numpy() 
            
            X_help = X_help[self.remain_samples]
            Y_help = Y_help[self.remain_samples]

            self.domain_old = self.Domain_train.iloc[self.remain_samples]
        else:
            X_help = self.Input_path_test.to_numpy()
            self.domain_old = self.Domain_test
        
        Agents = np.array([name[2:] for name in np.array(self.input_names_train)])
        Types  = np.array([name[0]  for name in np.array(self.input_names_train)])
        
        # Extract predicted agents
        Pred_agents = np.array([agent in self.data_set.needed_agents for agent in Agents])
        assert Pred_agents.sum() > 0, "nothing to predict"
        
        # Prepare numpy position array
        X = np.ones(list(X_help.shape) + [self.num_timesteps_in, 2], dtype = np.float32) * np.nan
        if train:
            Y = np.ones(list(Y_help.shape) + [self.num_timesteps_out.max(), 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample in range(X.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if isinstance(X_help[i_sample, i_agent], float):
                    assert not Pred_agents[i_agent], 'A needed agent is not given.'
                else:    
                    X[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                    if train:
                        n_time = self.num_timesteps_out[i_sample]
                        Y[i_sample, i_agent, :n_time] = Y_help[i_sample, i_agent][:n_time].astype(np.float32)
        
        #standardize input
        Ped_agents = Types == 'P'
        
                
        if train:
            Xi = X.transpose(0,1,3,2) # num_samples, num_agents, num_timesteps, 2
            Y = Y.transpose(0,1,3,2)
            
            
            # set agent to be predicted into first location
            X = []
            T = []
            for i_agent in np.where(Pred_agents)[0]:
                reorder_index = np.array([i_agent] + list(np.arange(i_agent)) + 
                                         list(np.arange(i_agent + 1, Xi.shape[1])))
                X.append(Xi[:,reorder_index])
                T.append(np.tile(Types[np.newaxis,reorder_index], (len(Xi), 1)))
            X = np.stack(X, axis = 1).reshape(-1, Xi.shape[1], self.num_timesteps_in, 2)
            T = np.stack(T, axis = 1).reshape(-1, len(Types))
            PPed_agents = T == 'P'
            # transform to ascii int:
            T = np.fromstring(T.reshape(-1), dtype = np.uint32).reshape(len(T), -1).astype(np.uint8)

            if self.use_map:
                
                centre = X[:,0,-1,:] #x_t.squeeze(-2)
                x_rel = centre - X[:,0,-2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 

                domain_repeat = self.domain_old.loc[self.domain_old.index.repeat(Pred_agents.sum())]
            
                img = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                        target_height = self.target_height, 
                                                        target_width = self.target_width, grayscale = True)
                
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                Y[:,Ped_agents]  /= self.std_pos_ped
                Y[:,~Ped_agents] /= self.std_pos_veh
                Y = Y[:, Pred_agents].reshape(-1, 1, self.num_timesteps_out.max(), 2)
                
                my_dataset = TensorDataset(torch.tensor(X).to(device=self.device),
                                           torch.tensor(Y).to(device=self.device), 
                                           torch.tensor(img),
                                           torch.tensor(T).to(device=self.device)) # create your datset
                
            else:
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                Y[:,Ped_agents]  /= self.std_pos_ped
                Y[:,~Ped_agents] /= self.std_pos_veh
                Y = Y[:, Pred_agents].reshape(-1, 1, self.num_timesteps_out.max(), 2)
                
                my_dataset = TensorDataset(torch.tensor(X).to(device=self.device),
                                           torch.tensor(Y).to(device=self.device),
                                           torch.tensor(T).to(device=self.device)) # create your datset

            
            train_data, val_data = torch.utils.data.random_split(my_dataset, 
                                                                 [int(np.round(len(my_dataset)*0.9)),
                                                                  int(len(my_dataset) - np.round(len(my_dataset)*0.9))],
                                                                 generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
            return train_loader, val_loader
        else:
            Xi = X.transpose(0,1,3,2) # num_samples, num_agents, num_timesteps, 2
            # set agent to be predicted into first location
            X = []
            T = []
            for i_agent in np.where(Pred_agents)[0]:
                reorder_index = np.array([i_agent] + list(np.arange(i_agent)) + 
                                         list(np.arange(i_agent + 1, Xi.shape[1])))
                X.append(Xi[:,reorder_index])
                T.append(np.tile(Types[np.newaxis,reorder_index], (len(Xi), 1)))
            
            X = np.stack(X, axis = 1).reshape(-1, Xi.shape[1], self.num_timesteps_in, 2)
            T = np.stack(T, axis = 1).reshape(-1, len(Types))
            PPed_agents = T == 'P'
            # transform to ascii int:
            T = np.fromstring(T.reshape(-1), dtype = np.uint32).reshape(len(T), -1).astype(np.uint8)
            
            if self.use_map:
                centre = X[:,0,-1,:] #x_t.squeeze(-2)
                x_rel = centre - X[:,0,-2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 

                domain_repeat = self.domain_old.loc[self.domain_old.index.repeat(Pred_agents.sum())]

                img = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                        target_height = self.target_height, 
                                                        target_width = self.target_width, grayscale = True)
                
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                return Pred_agents, Agents, X, T, PPed_agents, img
            
            else:
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                return Pred_agents, Agents, X, T, PPed_agents
            
        
    
    def train_futureAE(self, train_loader, val_loader):
        hs_rnn = self.hs_rnn
        obs_encoding_size = self.obs_encoding_size # 4
        n_layers_rnn = self.n_layers_rnn
        scene_encoding_size = self.scene_encoding_size # 4

        enc_size = self.fut_enc_sz

        flow_dist_futMdl = TrajFlow(pred_steps=self.future_traj_len, alpha=self.alpha, beta=self.beta_noise, 
                                    gamma=self.gamma_noise, norm_rotation=True, device=self.device, 
                                    obs_encoding_size=obs_encoding_size, scene_encoding_size=scene_encoding_size, 
                                    n_layers_rnn=n_layers_rnn, es_rnn=hs_rnn, hs_rnn=hs_rnn)
        
        num_epochs = self.fut_ae_epochs
        

        enc = Future_Encoder(2, enc_size, enc_size, enc_size)
        dec = Future_Decoder(2, enc_size, enc_size)

        fut_model = Future_Seq2Seq(enc, dec)
        # Use encoder of noninteractive trajflow model if available, as same training stuff is used
        fut_model_file = self.model_file[:-4] + '_AE'
        if os.path.isfile(fut_model_file) and not self.data_set.overwrite_results:
            fut_model = pickle.load(open(fut_model_file, 'rb'))
            print('Future AE model loaded')

        else:

            loss_fn = torch.nn.MSELoss()
            optim = torch.optim.AdamW(fut_model.parameters(), lr=self.fut_ae_lr, weight_decay=self.fut_ae_wd)
            diz_loss = {'train_loss':[],'val_loss':[]}

            val_losses = []
            
            
            converged = False
            for epoch in range(num_epochs):
                print('')
                rjust_epoch = str(epoch).rjust(len(str(num_epochs)))
                print('Train TrajFlow Autoencoder: Epoch ' + rjust_epoch + '/{}'.format(num_epochs), flush = True)
                # Analyize memory:
                fut_model.to(device = self.device)
                fut_model.train()
                train_loss = []
                    
                for _, sample_batched in enumerate(train_loader):
                    
                    scaler = torch.tensor(scipy.stats.truncnorm.rvs((self.s_min-1)/self.sigma, (self.s_max-1)/self.sigma, loc=1, scale=self.sigma, size=sample_batched[0].shape[0]))
                    scaler = scaler.unsqueeze(1)
                    scaler = scaler.unsqueeze(2)
                    scaler_past = torch.tile(scaler, (1, self.past_traj_len, 2)).to(self.device)
                    scaler_future = torch.tile(scaler, (1, self.future_traj_len, 2)).to(self.device)
                    
                    past_pos = sample_batched[0][:,0].float()
                    future_pos = sample_batched[1][:,0].float()
                    
                    
                    mean_pos = torch.mean(torch.concat((sample_batched[0][:,0], sample_batched[1][:,0]), dim=1), dim=1)
                    mean_pos_past = torch.tile(mean_pos.unsqueeze(1), (1,self.past_traj_len,1))
                    mean_pos_future = torch.tile(mean_pos.unsqueeze(1), (1,self.future_traj_len,1))
                    
                    shifted_past = past_pos - mean_pos_past
                    shifted_future = future_pos - mean_pos_future
                        
                    scaled_past = shifted_past*scaler_past + mean_pos_past
                    scaled_future = shifted_future*scaler_future + mean_pos_future
                    
                    past_data = scaled_past.float()
                    future_data = scaled_future.float()
                    
                    past_data = past_data.to(device = self.device)
                    future_data = future_data.to(device = self.device)
                    
                    past_traj, fut_traj, rot_angles_rad = flow_dist_futMdl._normalize_rotation(past_data, future_data)
                    
                    x_t = past_traj[...,-1:,:]
                    y_rel = flow_dist_futMdl._abs_to_rel(fut_traj, x_t)
                    
                    future_traj_hat, y_in = fut_model(y_rel)     
                        
                    optim.zero_grad()
                    loss = torch.sqrt(loss_fn(future_traj_hat, y_in))
                    loss.backward()
                    optim.step()
                        
                    train_loss.append(loss.detach().cpu().numpy())
                                
                fut_model.to(device = self.device)
                fut_model.eval()
                        
                with torch.no_grad():
                    conc_out = []
                    conc_label = []
                    for i_batch, sample_batched in enumerate(val_loader):
                        
                        past_data = sample_batched[0][:,0].float()
                        future_data = sample_batched[1][:,0].float()
                        
                        past_data = past_data.to(device = self.device)
                        future_data = future_data.to(device = self.device)
                        
                        past_traj, fut_traj, rot_angles_rad = flow_dist_futMdl._normalize_rotation(past_data, future_data)
                        
                        x_t = past_traj[...,-1:,:]
                        y_rel = flow_dist_futMdl._abs_to_rel(fut_traj, x_t)
                        
                            
                        future_traj_hat, y_in = fut_model(y_rel)
                            
                        
                        conc_out.append(future_traj_hat.cpu())
                        conc_label.append(y_rel.cpu())
                            
                    conc_out = torch.cat(conc_out)
                    conc_label = torch.cat(conc_label) 
                        
                    val_loss = torch.sqrt(loss_fn(conc_out, conc_label))

                    val_losses.append(val_loss.detach().cpu().numpy())
                    
                    # Check for convergence
                    if epoch > 100:
                        best_val_step = np.argmin(val_losses)
                        if epoch - best_val_step > 10:
                            converged = True
                    
                print('Train loss: {:7.5f}; \t Val loss: {:7.5f}'.format(np.mean(train_loss),val_loss.data))
                diz_loss['train_loss'].append(train_loss)
                diz_loss['val_loss'].append(val_loss)
                if converged:
                    print('Model is converged.')
                    break


            os.makedirs(os.path.dirname(fut_model_file), exist_ok=True)
            pickle.dump(fut_model, open(fut_model_file, 'wb'))

        return fut_model


    def train_flow(self, fut_model, train_loader, val_loader):
        steps = self.flow_epochs

        beta_noise = 0 
        gamma_noise = 0 
        
        if self.use_map:
            scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
        else:
            scene_encoder = None
            
        T_all = np.array([name[0] for name in np.array(self.input_names_train).reshape(-1,2)[:,0]])
        T_all = np.fromstring(T_all, dtype = np.uint32).astype(np.uint8)
        
        # TODO: Set the gnn parameters
        flow_dist = TrajFlow_I(pred_steps=self.fut_enc_sz, alpha=self.alpha, beta=beta_noise, gamma=gamma_noise, 
                               scene_encoder=scene_encoder, norm_rotation=self.norm_rotation, device=self.device,
                               obs_encoding_size=self.obs_encoding_size, scene_encoding_size=self.scene_encoding_size, n_layers_rnn=self.n_layers_rnn, 
                               es_rnn=self.hs_rnn, hs_rnn=self.hs_rnn, use_map=self.use_map, 
                               n_layers_gnn=4, es_gnn=32, T_all = T_all)
        
        for param in fut_model.parameters():
            param.requires_grad = False 
            param.grad = None
            
        
        flow_dist_file = self.model_file[:-4] + '_NF'
        
        if os.path.isfile(flow_dist_file) and not self.data_set.overwrite_results:
            flow_dist = pickle.load(open(flow_dist_file, 'rb'))
                          
        else:
            optimizer = torch.optim.AdamW(flow_dist.parameters(), lr=self.flow_lr, weight_decay=self.flow_wd)

            val_losses = []


            for step in range(steps):

                flow_dist.train()
                fut_model.eval()
                
                losses_epoch = []
                val_losses_epoch = []
                
                for i, data in enumerate(train_loader, 0):
                    # Introduce noise to make model more robust
                    scaler = torch.tensor(scipy.stats.truncnorm.rvs((self.s_min-1)/self.sigma, (self.s_max-1)/self.sigma, loc=1, scale=self.sigma, size=data[0].shape[0]))
                    # scaler = torch.ones(data[0].shape[0])
                    scaler = scaler.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(self.device)
                    
                    past_pos = data[0].to(device = self.device)
                    future_pos = data[1].to(device = self.device)
                    agent_types = data[-1].to(device = self.device)
                    
                    
                    mean_pos = torch.mean(torch.concat((past_pos[:,0], future_pos[:,0]), 
                                                       dim=1), dim=1, keepdims = True).unsqueeze(1)
                    # mean_pos shape: batch_size, dims
                    
                    shifted_past = past_pos - mean_pos
                    shifted_future = future_pos - mean_pos
                        
                    scaled_past = shifted_past * scaler + mean_pos
                    scaled_future = shifted_future * scaler + mean_pos
                    
                    optimizer.zero_grad()
                    
                    past_data = scaled_past.float()
                    
                    future_data = scaled_future.float()
                    
                    past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data, future_data)
                    
                    x_t = past_traj[:,[0],-1:,:]
                    y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                    if self.use_map:
                        img = data[2].float().to(device = self.device)
                        # img.shape = (batch_sz, target_height, target_width, 3)
                        img = img.permute(0,3,1,2)

                    out, _ = fut_model.encoder(y_rel[:,0])
                    out = out[:,-1]
                    # out.shape:       batch size x enc_dims
                    # past_data.shape: btach_size x num_agents x input_timesteps x num_dims
                    
                    if self.use_map:
                        logprob = flow_dist.log_prob(out, past_data, agent_types, img)#prior_logprob + log_det
                    else:
                        logprob = flow_dist.log_prob(out, past_data, agent_types)#prior_logprob + log_det

                    loss = -torch.mean(logprob) # NLL
                    losses_epoch.append(loss.item())
                    
                    # TODO check if flow_dist.zero_grad() was decisive
                    flow_dist.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    
                flow_dist.eval()
                fut_model.eval()
                with torch.no_grad():
                    for j, val in enumerate(val_loader, 0):
                        
                        past_data_val = val[0].float().to(device = self.device)
                        future_data_val = val[1].float().to(device = self.device)  
                        agent_types_val = val[-1].to(device = self.device)
                        
                        past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data_val, future_data_val)
                        
                        x_t = past_traj[:,[0],-1:,:]
                        y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                        
                        if self.use_map:
                            img_val = val[2].float().to(device = self.device)
                            # img.shape = (batch_sz, target_height, target_width, channels)
                            img_val = img_val.permute(0,3,1,2)

                        out, _ = fut_model.encoder(y_rel[:,0])
                        out = out[:, -1]
                        # out.shape: batch size x enc_dims
                            
                        optimizer.zero_grad()

                        if self.use_map:
                            log_prob = flow_dist.log_prob(out, past_data_val, agent_types_val, img_val)
                        else:
                            log_prob = flow_dist.log_prob(out, past_data_val, agent_types_val)
                    
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

                    print('step: {}, loss:     {}'.format(step, np.mean(losses_epoch)))
                    print('step: {}, val_loss: {}'.format(step, np.mean(val_losses_epoch)))

            os.makedirs(os.path.dirname(flow_dist_file), exist_ok=True)
            pickle.dump(flow_dist, open(flow_dist_file, 'wb'))

        return flow_dist


    def train_method(self):    

        train_loader, val_loader = self.extract_data(train = True)
        self.fut_model = self.train_futureAE(train_loader, val_loader)

        self.flow_dist = self.train_flow(self.fut_model, train_loader, val_loader)
        
        # save weigths 
        # after checking here, please return num_epochs to 100 and batch size to 
        self.weights_saved = []
        
        
    def load_method(self):
        fut_model_file = self.model_file[:-4] + '_AE'
        flow_dist_file = self.model_file[:-4] + '_NF'
        self.fut_model = pickle.load(open(fut_model_file, 'rb'))
        self.flow_dist = pickle.load(open(flow_dist_file, 'rb'))
        
    def _repeat_rowwise(self, x, n):
        org_dim = x.size(-1)
        x = x.repeat(1, 1, n)
        return x.view(-1, n, org_dim)
                
    def predict_batch(self, models, test_loader, target_length, batch_sz):
        flow_dist = models[1]
        fut_model = models[0]

        
        for _, sample_batched in enumerate(test_loader):
            
            past=sample_batched[0]
            past=past.float().to(device = self.device)
            agent_types = sample_batched[-1].to(device = self.device)
            
            past_traj, rot_angles_rad = flow_dist._normalize_rotation(past)
            
            
            if self.use_map:
                img = sample_batched[1].float().to(device=self.device)
                # img.shape = (batch_sz, target_height, target_width, channels)
                img = img.permute(0,3,1,2)

                # img[img>0] = 1
                # img = img
            else:
                img = None
            
            

            x_t = past_traj[:,0,-1:,:]
            x_t = self._repeat_rowwise(x_t, self.num_samples_path_pred)
            x_t = x_t.reshape(batch_sz*self.num_samples_path_pred,-1).unsqueeze(1)
            
            rot_angles_rad = rot_angles_rad.repeat_interleave(self.num_samples_path_pred)

            if self.use_map: 
                samples_rel, log_probs = flow_dist.sample(self.num_samples_path_pred, past.float(), agent_types, img)
            else:
                samples_rel, log_probs = flow_dist.sample(self.num_samples_path_pred, past.float(), agent_types)
            
            samples_rel = samples_rel.squeeze(0)
                    
            hidden = torch.tile(samples_rel.reshape(-1, self.fut_enc_sz).unsqueeze(0), (fut_model.decoder.nl,1,1))
            
            # Decoder part
            x = samples_rel.reshape(-1, self.fut_enc_sz).unsqueeze(1)
            
            outputs = torch.zeros(batch_sz*self.num_samples_path_pred, target_length, 2).to(device = self.device)
            for t in range(0, target_length):

                output, hidden = fut_model.decoder(x, hidden)
                
                outputs[:, t, :] = output.squeeze()
                
                x = hidden[-1].unsqueeze(1)
            
            y_hat = flow_dist._rel_to_abs(outputs, x_t)

            # invert rotation normalization
            y_hat = flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad.unsqueeze(1))

            y_hat = y_hat.reshape(batch_sz, self.num_samples_path_pred, target_length, 2)
            
            Y_pred = y_hat.detach()
                
            log_probs = log_probs.detach()
            log_probs[torch.isnan(log_probs)] = -1000
            prob = torch.exp(log_probs)#[exp(x) for x in log_probs]
            prob = torch.tensor(prob)
        
        return Y_pred, prob


    def predict_method(self):
        # get desired output length
        self.num_timesteps_out_test = np.zeros(len(self.Output_T_pred_test), int)
        for i_sample in range(len(self.Output_T_pred_test)):
            self.num_timesteps_out_test[i_sample] = len(self.Output_T_pred_test[i_sample])
            
        if self.use_map:
            Pred_agents, Agents, X, T, PPed_agents, img = self.extract_data(train = False)
        else:
            Pred_agents, Agents, X, T, PPed_agents = self.extract_data(train = False)
        
        
        Path_names = np.array([name for name in self.Output_path_train.columns]).reshape(-1, 2)
        
        # TODO keep in mind since len(test_loader.dataset) might cause issues
        samples_all = int(len(X)/ Pred_agents.sum())

        Output_Path = pd.DataFrame(np.empty((samples_all, Pred_agents.sum() * 2), object), 
                                   columns = Path_names[Pred_agents].reshape(-1))
        
        nums = np.unique(self.num_timesteps_out_test)
        
        
        samples_done = 0
        calculations_done = 0
        calculations_all = np.sum(self.num_timesteps_out_test)
        for num in nums:
            Index_num = np.where(self.num_timesteps_out_test == num)[0]
            
            if self.batch_size > len(Index_num):
                Index_uses = [Index_num]
            else:
                Index_uses = [Index_num[i * self.batch_size : (i + 1) * self.batch_size] 
                              for i in range(int(np.ceil(len(Index_num)/ self.batch_size)))] 
            
            for Index_use in Index_uses:
                for i_agent, agent in enumerate(Agents[Pred_agents]):
                    if self.use_map:
                        Index_use_agent = Index_use * Pred_agents.sum() + i_agent
                        my_dataset = TensorDataset(torch.tensor(X[Index_use_agent]).to(device=self.device),
                                                   torch.tensor(img[Index_use_agent]),
                                                   torch.tensor(T[Index_use_agent]).to(device=self.device)) # create your datset
                    else:
                        Index_use_agent = Index_use * Pred_agents.sum() + i_agent
                        my_dataset = TensorDataset(torch.tensor(X[Index_use_agent]).to(device=self.device),
                                                   torch.tensor(T[Index_use_agent]).to(device=self.device)) # create your datset
                    
                    Ped_agent = PPed_agents[Index_use, 0]
                    test_loader = DataLoader(my_dataset, batch_size=len(Index_use)) # create your dataloader

                    path_names = Path_names[np.where(Pred_agents)[0][i_agent]]
                    
                    # Run prediction pass
                    with torch.no_grad(): # Do not build graph for backprop
                        predictions, predictions_prob = self.predict_batch([self.fut_model, self.flow_dist], test_loader, num, len(Index_use))

                    
                    Pred = predictions.detach().cpu().numpy()
                    if len(Pred.shape) == 3:
                        Pred = Pred[np.newaxis]
                    
                    Pred[Ped_agent]  *= self.std_pos_ped
                    Pred[~Ped_agent] *= self.std_pos_veh
                    
                    
                    torch.cuda.empty_cache()
                    
                    for i, i_sample in enumerate(Index_use):
                        traj = Pred[i, :, :, :]
                        for j, index in enumerate(path_names):
                            Output_Path.iloc[i_sample][index] = traj[:,:,j].astype('float32')
                        
                        
                samples_done += len(Index_use)
                calculations_done += len(Index_use) * num
                
                samples_perc = 100 * samples_done / samples_all
                calculations_perc = 100 * calculations_done / calculations_all
                
                print('Predict TrajFlow: ' + 
                      format(samples_perc, '.2f').rjust(len('100.00')) + 
                      '% of samples, ' + 
                      format(calculations_perc, '.2f').rjust(len('100.00')) +
                      '% of calculations')
                
        return [Output_Path]
    
    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
        
    def get_input_type(self = None):
        input_info = {'past': 'path',
                      'future': False}
        return input_info
    
    def get_name(self = None):
        names = {'print': 'TrajFlow',
                 'file': 'TrajFlow_M',
                 'latex': r'\emph{TF}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True