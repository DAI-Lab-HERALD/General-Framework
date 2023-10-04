from model_template import model_template
import numpy as np
import torch
import random
import scipy
from TrajFlow_LSTM.flowModels import TrajFlow_I, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder
import pickle
import os

class trajflow_meszaros_futEnc12_LSTM_past64_refining(model_template):
    '''
    TrajFlow is a single agent prediction model that combine Normalizing Flows with
    GRU-based autoencoders.
    
    The model was implemented into the framework by its original creators, and 
    the model was first published under:
        
    Mészáros, A., Alonso-Mora, J., & Kober, J. (2023). Trajflow: Learning the 
    distribution over trajectories. arXiv preprint arXiv:2304.05166.
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
        self.fut_enc_sz = 12 ## 4-8

        self.scene_encoding_size = 4
        self.obs_encoding_size = 64
        
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


        self.fut_ae_epochs = 5000
        self.fut_ae_lr = 5e-4
        self.fut_ae_wd = 1e-4

        self.flow_epochs = 200
        self.flow_lr = 1e-3
        self.flow_wd = 1e-5

        self.refine_epochs = 100
        self.refining_lr = 5e-5

        self.std_pos_ped = 1
        self.std_pos_veh = 1 #80

        self.vary_input_length = False
        
    
    def extract_batch_data(self, X, T, Y = None, img = None):
        
        # Get type of agents
        T_out = T.astype(str)
        Ped_agents = T_out == 'P'
        
        # Transform types to numbers
        T_out[T_out == 'nan'] = '0'
        T_out = np.fromstring(T_out.reshape(-1), dtype = np.uint32).reshape(*T_out.shape, int(str(T_out.astype(str).dtype)[2:])).astype(np.uint8)[:,:,0]
        T_out = torch.from_numpy(T_out).to(device = self.device)
        
        # Standardize positions
        X[Ped_agents]  /= self.std_pos_ped
        X[~Ped_agents] /= self.std_pos_veh
        X = torch.from_numpy(X).float().to(device = self.device)
        
        if Y is not None:            
            # Standardize future positions
            Y[Ped_agents]  /= self.std_pos_ped
            Y[~Ped_agents] /= self.std_pos_veh
            Y = torch.from_numpy(Y).float().to(device = self.device)
        
        if img is not None:
            img = torch.from_numpy(img).float().to(device = self.device) / 255
            
        return X, T_out, Y, img
    
    
    def train_futureAE(self, T_all):
        hs_rnn = self.hs_rnn
        obs_encoding_size = self.obs_encoding_size # 4
        n_layers_rnn = self.n_layers_rnn
        scene_encoding_size = self.scene_encoding_size # 4

        enc_size = self.fut_enc_sz

        flow_dist_futMdl = TrajFlow_I(pred_steps=self.num_timesteps_out, alpha=self.alpha, beta=self.beta_noise, 
                                    gamma=self.gamma_noise, norm_rotation=True, device=self.device, 
                                    obs_encoding_size=obs_encoding_size, scene_encoding_size=scene_encoding_size, 
                                    n_layers_rnn=n_layers_rnn, es_rnn=hs_rnn, hs_rnn=hs_rnn, T_all=T_all)

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
            for epoch in range(self.fut_ae_epochs):
                print('')
                rjust_epoch = str(epoch).rjust(len(str(self.fut_ae_epochs)))
                print('Train TrajFlow Autoencoder: Epoch ' + rjust_epoch + '/{}'.format(self.fut_ae_epochs), flush = True)
                # Analyize memory:
                fut_model.to(device = self.device)
                fut_model.train()
                train_loss = []
                
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, T, img, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                           val_split_size = 0.1)
                    X, T, Y, _ = self.extract_batch_data(X, T, Y)
                    
                    # X.shape:   bs x num_agents x num_timesteps_is x 2
                    # Y.shape:   bs x 1 x num_timesteps_is x 2
                    # T.shape:   bs x num_agents
                    # img.shape: bs x 1 x 156 x 257 x 1
                    
                    scaler = torch.tensor(scipy.stats.truncnorm.rvs((self.s_min-1)/self.sigma, (self.s_max-1)/self.sigma, 
                                                                    loc=1, scale=self.sigma, size=X.shape[0])).float()
                    scaler = scaler.unsqueeze(1)
                    scaler = scaler.unsqueeze(2)
                    scaler = scaler.to(device = self.device)

                    scaler = torch.tensor(np.ones_like(scaler.cpu().numpy())).to(device = self.device)
                    
                    tar_pos_past   = X[:,0]
                    tar_pos_future = Y[:,0]
                    
                    mean_pos = torch.mean(torch.concat((tar_pos_past, tar_pos_future), dim = 1), dim=1, keepdims = True)
                    
                    shifted_past = tar_pos_past - mean_pos
                    shifted_future = tar_pos_future - mean_pos
                        
                    past_data = shifted_past * scaler + mean_pos
                    future_data = shifted_future * scaler + mean_pos
                    
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
                    
                    val_epoch_done = False
                    while not val_epoch_done:
                        X, Y, T, _, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                val_split_size = 0.1)
                        
                        X, T, Y, _ = self.extract_batch_data(X, T, Y)
                        
                        past_data = X[:,0]
                        future_data = Y[:,0]
                        
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
                    
                    # Early stopping for AE
                    # Check for convergence
                    if epoch > 200:
                        best_val_step = np.argmin(val_losses)
                        if epoch - best_val_step > 50:
                            converged = True
                    
                print('Train loss: {:7.5f}; \t Val loss: {:7.5f}'.format(np.mean(train_loss),val_loss.data))
                diz_loss['train_loss'].append(train_loss)
                diz_loss['val_loss'].append(val_loss)
                if converged:
                    print('Model is converged.')
                    break

            self.train_loss[0, :len(val_losses)] = np.array(val_losses)
            os.makedirs(os.path.dirname(fut_model_file), exist_ok=True)
            pickle.dump(fut_model, open(fut_model_file, 'wb'))

        return fut_model


    def train_flow(self, fut_model, T_all):
        use_map = self.can_use_map and self.has_map

        self.beta_noise = 0.002
        self.gamma_noise = 0.002

        if self.vary_input_length:
            past_length_options = np.arange(0.5, self.num_timesteps_in*self.dt, 0.5)
            sample_past_length = int(np.ceil(np.random.choice(past_length_options)/self.dt))
        else:
            sample_past_length = self.num_timesteps_in
        
        if use_map:
            scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
        else:
            scene_encoder = None
        # TODO: Set the gnn parameters
        flow_dist = TrajFlow_I(pred_steps=self.fut_enc_sz, alpha=self.alpha, beta=self.beta_noise, gamma=self.gamma_noise, 
                               scene_encoder=scene_encoder, norm_rotation=self.norm_rotation, device=self.device,
                               obs_encoding_size=self.obs_encoding_size, scene_encoding_size=self.scene_encoding_size, n_layers_rnn=self.n_layers_rnn, 
                               es_rnn=self.hs_rnn, hs_rnn=self.hs_rnn, use_map=use_map, 
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


            for step in range(self.flow_epochs):

                flow_dist.train()
                fut_model.eval()
                
                losses_epoch = []
                val_losses_epoch = []
                
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, T, img, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                           val_split_size = 0.1)
                    X, T, Y, img = self.extract_batch_data(X, T, Y, img)
                    
                    # X.shape:   bs x num_agents x num_timesteps_is x 2
                    # Y.shape:   bs x num_agents x num_timesteps_is x 2
                    # T.shape:   bs x num_agents
                    # img.shape: bs x 1 x 156 x 257 x 1
                    
                    scaler = torch.tensor(scipy.stats.truncnorm.rvs((self.s_min-1)/self.sigma, (self.s_max-1)/self.sigma, 
                                                                    loc=1, scale=self.sigma, size=X.shape[0])).float()
                    scaler = scaler.unsqueeze(1)
                    scaler = scaler.unsqueeze(2)
                    scaler = scaler.unsqueeze(3)
                    scaler = scaler.to(device = self.device)
                    
                    scaler = torch.tensor(np.ones_like(scaler.cpu().numpy())).to(device = self.device)

                    X = X[:,:,-sample_past_length:,:]
                    
                    all_pos_past   = X
                    tar_pos_past   = X[:,0]
                    all_pos_future = Y
                    tar_pos_future = Y[:,0]
                    
                    mean_pos = torch.mean(torch.concat((tar_pos_past, tar_pos_future), dim = 1), dim=1, keepdims = True).unsqueeze(1)
                    
                    shifted_past   = all_pos_past - mean_pos
                    shifted_future = all_pos_future - mean_pos
                        
                    past_data   = shifted_past * scaler + mean_pos
                    future_data = shifted_future * scaler + mean_pos
                    
                    optimizer.zero_grad()
                    
                    past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data, future_data)
                    
                    x_t   = past_traj[:,[0],-1:,:]
                    y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                    if img is not None:
                        img = img[:,0].permute(0,3,1,2)

                    out, _ = fut_model.encoder(y_rel[:,0])
                    out = out[:,-1]
                    # out.shape:       batch size x enc_dims
                    
                    if img is not None:
                        logprob = flow_dist.log_prob(out, past_data, T, img) #prior_logprob + log_det
                    else:
                        logprob = flow_dist.log_prob(out, past_data, T) #prior_logprob + log_det

                    loss = -torch.mean(logprob) # NLL
                    losses_epoch.append(loss.item())
                    
                    loss.backward()
                    optimizer.step()
                    
                    
                flow_dist.eval()
                fut_model.eval()
                with torch.no_grad():
                    val_epoch_done = False
                    while not val_epoch_done:
                        X, Y, T, img, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                val_split_size = 0.1)
                        X, T, Y, img = self.extract_batch_data(X, T, Y, img)
                        
                        past_data_val = X
                        future_data_val = Y
                        
                        past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data_val, future_data_val)
                        
                        x_t = past_traj[:,[0],-1:,:]
                        y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                        
                        if img is not None:
                            img_val = img[:,0].permute(0,3,1,2)

                        out, _ = fut_model.encoder(y_rel[:,0])
                        out = out[:, -1]
                        # out.shape: batch size x enc_dims
                            
                        optimizer.zero_grad()

                        if img is not None:
                            log_prob = flow_dist.log_prob(out, past_data_val, T, img_val)
                        else:
                            log_prob = flow_dist.log_prob(out, past_data_val, T)
                    
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

            self.train_loss[1, :len(val_losses)] = np.array(val_losses)
            os.makedirs(os.path.dirname(flow_dist_file), exist_ok=True)
            pickle.dump(flow_dist, open(flow_dist_file, 'wb'))

        return flow_dist
    


    def refine_network(self, fut_model, flow_dist):
        if self.vary_input_length:
            past_length_options = np.arange(0.5, self.num_timesteps_in*self.dt, 0.5)
            sample_past_length = int(np.ceil(np.random.choice(past_length_options)/self.dt))
        else:
            sample_past_length = self.num_timesteps_in
        

        for param in fut_model.parameters():
            param.requires_grad = True 
            
        
        flow_dist_file = self.model_file[:-4] + '_NF'
        ae_file = self.model_file[:-4] + '_AE'
             
        optimizer = torch.optim.AdamW(list(fut_model.parameters()) + list(flow_dist.parameters()), 
                                      lr=self.refining_lr, weight_decay=self.flow_wd)

        val_losses = []


        for step in range(self.refine_epochs):

            flow_dist.train()
            fut_model.train()
            
            losses_epoch = []
            val_losses_epoch = []
            
            train_epoch_done = False
            while not train_epoch_done:
                X, Y, T, img, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                        val_split_size = 0.1)
                X, T, Y, img = self.extract_batch_data(X, T, Y, img)
                
                # X.shape:   bs x num_agents x num_timesteps_is x 2
                # Y.shape:   bs x num_agents x num_timesteps_is x 2
                # T.shape:   bs x num_agents
                # img.shape: bs x 1 x 156 x 257 x 1
                
                scaler = torch.tensor(scipy.stats.truncnorm.rvs((self.s_min-1)/self.sigma, (self.s_max-1)/self.sigma, 
                                                                loc=1, scale=self.sigma, size=X.shape[0])).float()
                scaler = scaler.unsqueeze(1)
                scaler = scaler.unsqueeze(2)
                scaler = scaler.unsqueeze(3)
                scaler = scaler.to(device = self.device)
                
                scaler = torch.tensor(np.ones_like(scaler.cpu().numpy())).to(device = self.device)

                X = X[:,:,-sample_past_length:,:]
                
                all_pos_past   = X
                tar_pos_past   = X[:,0]
                all_pos_future = Y
                tar_pos_future = Y[:,0]
                
                mean_pos = torch.mean(torch.concat((tar_pos_past, tar_pos_future), dim = 1), dim=1, keepdims = True).unsqueeze(1)
                
                shifted_past   = all_pos_past - mean_pos
                shifted_future = all_pos_future - mean_pos
                    
                past_data   = shifted_past * scaler + mean_pos
                future_data = shifted_future * scaler + mean_pos
                
                optimizer.zero_grad()
                
                past_traj, fut_traj, _ = flow_dist._normalize_rotation(past_data, future_data)
                
                x_t   = past_traj[:,[0],-1:,:]
                y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                if img is not None:
                    img = img[:,0].permute(0,3,1,2)

                out, _ = fut_model.encoder(y_rel[:,0])
                out = out[:,-1]
                # out.shape:       batch size x enc_dims
                
                if img is not None:
                    logprob = flow_dist.log_prob(out, past_data, T, img) #prior_logprob + log_det
                else:
                    logprob = flow_dist.log_prob(out, past_data, T) #prior_logprob + log_det

                loss = -torch.mean(logprob) # NLL
                losses_epoch.append(loss.item())
                
                loss.backward()
                optimizer.step()
                
                
            flow_dist.eval()
            fut_model.eval()
            with torch.no_grad():
                val_epoch_done = False
                while not val_epoch_done:
                    X, Y, T, img, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                            val_split_size = 0.1)
                    X, T, Y, img = self.extract_batch_data(X, T, Y, img)
                    
                    past_data_val = X
                    future_data_val = Y
                    
                    past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data_val, future_data_val)
                    
                    x_t = past_traj[:,[0],-1:,:]
                    y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                    
                    if img is not None:
                        img_val = img[:,0].permute(0,3,1,2)

                    out, _ = fut_model.encoder(y_rel[:,0])
                    out = out[:, -1]
                    # out.shape: batch size x enc_dims
                        
                    optimizer.zero_grad()

                    if img is not None:
                        log_prob = flow_dist.log_prob(out, past_data_val, T, img_val)
                    else:
                        log_prob = flow_dist.log_prob(out, past_data_val, T)
                
                    val_loss = -torch.mean(log_prob)
                    val_losses_epoch.append(val_loss.item())
                    
                val_losses.append(np.mean(val_losses_epoch))      
            
            # Check for convergence
            if step > 25:
                best_val_step = np.argmin(val_losses)
                if step - best_val_step > 10:
                    print('Converged')
                    print('step: {}, loss:     {}'.format(step, np.mean(losses_epoch)))
                    print('step: {}, val_loss: {}'.format(step, np.mean(val_losses_epoch)))
                    break

            if step % 10 == 0:

                print('step: {}, loss:     {}'.format(step, np.mean(losses_epoch)))
                print('step: {}, val_loss: {}'.format(step, np.mean(val_losses_epoch)))

            self.train_loss[1, :len(val_losses)] = np.array(val_losses)
            os.makedirs(os.path.dirname(ae_file), exist_ok=True)
            pickle.dump(flow_dist, open(ae_file, 'wb'))
            os.makedirs(os.path.dirname(flow_dist_file), exist_ok=True)
            pickle.dump(flow_dist, open(flow_dist_file, 'wb'))

        return fut_model, flow_dist


    def train_method(self):    
        self.train_loss = np.ones((2, max(self.fut_ae_epochs, self.flow_epochs))) * np.nan
        
        # Get needed agent types
        T_all = self.provide_all_included_agent_types().astype(str)
        T_all = np.fromstring(T_all, dtype = np.uint32).reshape(len(T_all), int(str(T_all.astype(str).dtype)[2:])).astype(np.uint8)[:,0]
                    
        # Train model components        
        self.fut_model = self.train_futureAE(T_all)
        self.flow_dist = self.train_flow(self.fut_model, T_all)
        self.fut_model, self.flow_dist = self.refine_network(self.fut_model, self.flow_dist)
        
        # save weigths 
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
    

    def predict_method(self):
        prediction_done = False
        while not prediction_done:
            X, T, img, _, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.batch_size)
            actual_batch_size = len(X)
            Ped_agent = T == 'P'
            
            X, T, _, img = self.extract_batch_data(X, T, img = img)
            # Run prediction pass
            with torch.no_grad():
                past_traj, rot_angles_rad = self.flow_dist._normalize_rotation(X)
                
                if img is not None:
                    img = img[:,0].permute(0,3,1,2)
                else:
                    img = None
                
                x_t = past_traj[:,0,-1:,:]
                x_t = self._repeat_rowwise(x_t, self.num_samples_path_pred)
                x_t = x_t.reshape(actual_batch_size * self.num_samples_path_pred,-1).unsqueeze(1)
                
                rot_angles_rad = rot_angles_rad.repeat_interleave(self.num_samples_path_pred)

                if img is not None: 
                    samples_rel, log_probs = self.flow_dist.sample(self.num_samples_path_pred, X, T, img)
                else:
                    samples_rel, log_probs = self.flow_dist.sample(self.num_samples_path_pred, X, T)
                
                samples_rel = samples_rel.squeeze(0)
                        
                hidden = torch.tile(samples_rel.reshape(-1, self.fut_enc_sz).unsqueeze(0), (self.fut_model.decoder.nl,1,1))
                
                # Decoder part
                x = samples_rel.reshape(-1, self.fut_enc_sz).unsqueeze(1)
                
                outputs = torch.zeros(actual_batch_size * self.num_samples_path_pred, num_steps, 2).to(device = self.device)
                for t in range(0, num_steps):

                    output, hidden = self.fut_model.decoder(x, hidden)
                    
                    outputs[:, t, :] = output.squeeze()
                    
                    x = hidden[-1].unsqueeze(1)
                
                y_hat = self.flow_dist._rel_to_abs(outputs, x_t)

                # invert rotation normalization
                y_hat = self.flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad.unsqueeze(1))

                y_hat = y_hat.reshape(actual_batch_size, self.num_samples_path_pred, num_steps, 2)
                
                Y_pred = y_hat.detach()
                    
                # This should not be needed
                # log_probs = log_probs.detach()
                # log_probs[torch.isnan(log_probs)] = -1000
                # prob = torch.exp(log_probs)#[exp(x) for x in log_probs]
                # prob = torch.tensor(prob)
                    
                    
            Pred = Y_pred.detach().cpu().numpy()
            if len(Pred.shape) == 3:
                Pred = Pred[np.newaxis]
            
            Pred[Ped_agent[:,0]]  *= self.std_pos_ped
            Pred[~Ped_agent[:,0]] *= self.std_pos_veh
            
            torch.cuda.empty_cache()
            
            # save predictions
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id)
    
    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_name(self = None):
        
        names = {'print': 'TrajFlow_fut12_L64R',
                'file': 'TF_f12L64R',
                'latex': r'\emph{TF}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True
        
    def provides_epoch_loss(self = None):
        return True