from model_template import model_template
import numpy as np
import torch
import random
import scipy
from TrajFlow.flowModels import TrajFlow, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder, Future_Decoder_Control, Future_Seq2Seq_Control
import pickle
import os

class trajflow_meszaros(model_template):
    '''
    TrajFlow is a single agent prediction model that combine Normalizing Flows with
    GRU-based autoencoders.
    
    The model was implemented into the framework by its original creators, and 
    the model was first published under:
        
    Mészáros, A., Alonso-Mora, J., & Kober, J. (2023). Trajflow: Learning the 
    distribution over trajectories. arXiv preprint arXiv:2304.05166.
    '''
    def define_default_kwargs(self):
        if not ('batch_size' in self.model_kwargs.keys()):
            self.model_kwargs['batch_size'] = 128

        if not ('hs_rnn' in self.model_kwargs.keys()):
            self.model_kwargs['hs_rnn'] = 16

        if not ('n_layers_rnn' in self.model_kwargs.keys()):
            self.model_kwargs['n_layers_rnn'] = 3

        if not ('fut_enc_sz' in self.model_kwargs.keys()):
            self.model_kwargs['fut_enc_sz'] = 20

        if not ('scene_encoding_size' in self.model_kwargs.keys()):
            self.model_kwargs['scene_encoding_size'] = 64

        if not ('obs_encoding_size' in self.model_kwargs.keys()):
            self.model_kwargs['obs_encoding_size'] = 64

        # TODO: Add the GNN encoding size (currently 32)

        if not ('beta_noise' in self.model_kwargs.keys()):
            self.model_kwargs['beta_noise'] = 0 # 0.2 (P) / 0.002

        if not ('gamma_noise' in self.model_kwargs.keys()):
            self.model_kwargs['gamma_noise'] = 0 # 0.02 (P) / 0.002

        if not ('alpha' in self.model_kwargs.keys()):
            self.model_kwargs['alpha'] = 10 # 10 (P) / 3

        if not ('s_min' in self.model_kwargs.keys()):
            self.model_kwargs['s_min'] = 0.8 # 0.3 (P) / 0.8 

        if not ('s_max' in self.model_kwargs.keys()):
            self.model_kwargs['s_max'] = 1.2 # 1.7 (P) / 1.2

        if not ('sigma' in self.model_kwargs.keys()):  
            self.model_kwargs['sigma'] = 0.5 # 0.5 (P) / 0.2

        if not ('fut_ae_epochs' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_epochs'] = 5000
        
        if not ('fut_ae_lr' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_lr'] = 5e-4

        if not ('fut_ae_lr_decay' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_lr_decay'] = 1.0

        if not ('fut_ae_wd' in self.model_kwargs.keys()):
            self.model_kwargs['fut_ae_wd'] = 1e-4

        if not ('flow_epochs' in self.model_kwargs.keys()):
            self.model_kwargs['flow_epochs'] = 500

        if not ('flow_lr' in self.model_kwargs.keys()):
            self.model_kwargs['flow_lr'] = 1e-3

        if not ('flow_lr_decay' in self.model_kwargs.keys()):
            self.model_kwargs['flow_lr_decay'] = 0.98

        if not ('flow_wd' in self.model_kwargs.keys()):
            self.model_kwargs['flow_wd'] = 1e-5

        if not ('vary_input_length' in self.model_kwargs.keys()):
            self.model_kwargs['vary_input_length'] = False

        if not ('scale_AE' in self.model_kwargs.keys()):
            self.model_kwargs['scale_AE'] = False

        if not ('scale_NF' in self.model_kwargs.keys()):
            self.model_kwargs['scale_NF'] = False

        if not ('pos_loss' in self.model_kwargs.keys()):
            self.model_kwargs['pos_loss'] = True

        if not("decoder_type" in self.model_kwargs.keys()):
            self.model_kwargs["decoder_type"] = "none"

        if not('seed' in self.model_kwargs.keys()):
            self.model_kwargs['seed'] = 0

        if not('interactions' in self.model_kwargs.keys()):
            self.model_kwargs['interactions'] = True


        self.hs_rnn = self.model_kwargs['hs_rnn']
        self.n_layers_rnn = self.model_kwargs['n_layers_rnn']
        self.fut_enc_sz = self.model_kwargs['fut_enc_sz'] 

        self.scene_encoding_size = self.model_kwargs['scene_encoding_size']
        self.obs_encoding_size = self.model_kwargs['obs_encoding_size'] 
        
        self.beta_noise = self.model_kwargs['beta_noise']
        self.gamma_noise = self.model_kwargs['gamma_noise']
        
        self.alpha = self.model_kwargs['alpha']
        self.s_min = self.model_kwargs['s_min']
        self.s_max = self.model_kwargs['s_max']
        self.sigma = self.model_kwargs['sigma']

        self.fut_ae_epochs = self.model_kwargs['fut_ae_epochs']
        self.fut_ae_lr = self.model_kwargs['fut_ae_lr']
        self.fut_ae_lr_decay = self.model_kwargs['fut_ae_lr_decay']
        self.fut_ae_wd = self.model_kwargs['fut_ae_wd']

        self.flow_epochs = self.model_kwargs['flow_epochs']
        self.flow_lr = self.model_kwargs['flow_lr']
        self.flow_lr_decay = self.model_kwargs['flow_lr_decay']
        self.flow_wd = self.model_kwargs['flow_wd']

        self.vary_input_length = self.model_kwargs['vary_input_length']
        self.scale_AE = self.model_kwargs['scale_AE']
        self.scale_NF = self.model_kwargs['scale_NF']
        self.pos_loss = self.model_kwargs['pos_loss']

        # Get Decoder Type
        self.decoder_type = self.model_kwargs["decoder_type"] 

        self.interactions = self.model_kwargs['interactions']

    
    def setup_method(self):        
        # set random seeds
        self.define_default_kwargs()
        seed = self.model_kwargs['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.batch_size = self.model_kwargs['batch_size']
        
        # Required attributes of the model
        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = 100
        self.predict_single_agent = True
        self.can_use_map = True
        self.can_use_graph = False
        # If self.can_use_map, the following is also required
        self.target_width = 257
        self.target_height = 156
        self.grayscale = True
        
        self.norm_rotation = True

        self.std_pos_ped = 1
        self.std_pos_veh = 1 
        
    
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
        

        # Use encoder of noninteractive trajflow model if available, as same training stuff is used
        fut_model_file = self.model_file[:-4] + '_AE'
        if os.path.isfile(fut_model_file) and not self.model_overwrite:
            fut_model = pickle.load(open(fut_model_file, 'rb'))
            print('Future AE model loaded')

        else:
            # Check if equivalent AE model is available
            eqivalent_file = None

            model_directory = os.path.dirname(self.model_file)

            if os.path.isdir(model_directory):
                # Find all files in Model folder
                model_files = os.listdir(model_directory)

                # Find all AE models
                model_files = [f for f in model_files if f[-3:] == '_AE']

                # Get desired environment name
                env_name_desired = '--'.join(fut_model_file.split(os.sep)[-1].split('--')[:-1])
                model_kwargs_potential = fut_model_file.split(os.sep)[-1].split('--')[-1][3:-3].split('_')

                AE_kwargs = ['seed', 'fut', 'alpha', 'posLoss', 'varyInLen', 'sclAE']
                if 'sclAE' in model_kwargs_potential:
                    AE_kwargs += ['smin', 'smax', 'sigma']

                model_kwargs_desired = []
                for i_kw, kw in enumerate(model_kwargs_potential):
                    for ae_kw in AE_kwargs:
                        if ae_kw in kw:
                            model_kwargs_desired.append(kw)
                            break
                    # Check for lr_decay
                    if kw == 'lrDec':
                        if model_kwargs_potential[i_kw - 1] == "ae":
                            model_kwargs_desired.append(kw)

                # Find all AE models with same parameters
                for file in model_files:
                    # Diveide Model name from rest
                    File_split = file.split('--')
                    
                    # Check env name
                    env_name = '--'.join(File_split[:-1])
                    if env_name != env_name_desired:
                        continue

                    # Get model name
                    model_name = File_split[-1][:-3]

                    # Check for TF model
                    if model_name[:3] != 'TF_':
                        continue

                    # Check for parameters
                    model_kwargs_pot = model_name[3:].split('_')

                    # Find important kwargs
                    model_kwargs = []
                    for kw in model_kwargs_pot:
                        for ae_kw in AE_kwargs:
                            if ae_kw in kw:
                                model_kwargs_desired.append(kw)
                                break
                        
                        # Check for lr_decay
                        if kw == 'lrDec':
                            if model_kwargs_potential[i_kw - 1] == "ae":
                                model_kwargs_desired.append(kw)

                    # Check if AE settings are identical
                    if model_kwargs == model_kwargs_desired:
                        eqivalent_file = file
                        break

            if eqivalent_file is not None and not self.model_overwrite:
                # Load model
                equiv_model_file = os.path.dirname(self.model_file) + os.sep + eqivalent_file
                fut_model = pickle.load(open(equiv_model_file, 'rb'))

                # Load train losses
                if self.provides_epoch_loss():
                    equiv_train_loss_file = equiv_model_file[:-3] + '--train_loss.npy'
                    equiv_train_loss = np.load(equiv_train_loss_file)

                    max_len = min(equiv_train_loss.shape[1], self.train_loss.shape[1])
                    self.train_loss[0, :max_len] = equiv_train_loss[0, :max_len]
            else:
                hs_rnn = self.hs_rnn
                obs_encoding_size = self.obs_encoding_size # 4
                n_layers_rnn = self.n_layers_rnn
                scene_encoding_size = self.scene_encoding_size # 4

                enc_size = self.fut_enc_sz



                flow_dist_futMdl = TrajFlow(pred_steps=self.num_timesteps_out, alpha=self.alpha, beta=self.beta_noise, 
                                            gamma=self.gamma_noise, norm_rotation=True, device=self.device, 
                                            obs_encoding_size=obs_encoding_size, scene_encoding_size=scene_encoding_size, 
                                            interactions=self.interactions ,n_layers_rnn=n_layers_rnn, es_rnn=hs_rnn, hs_rnn=hs_rnn, T_all=T_all)

                enc = Future_Encoder(2, enc_size, enc_size, enc_size)

                if self.decoder_type == "none":
                    dec = Future_Decoder(2, enc_size, enc_size)
                    fut_model = Future_Seq2Seq(enc, dec)
                else:
                    dec = Future_Decoder_Control(2, enc_size, enc_size, decoder_type = self.decoder_type)
                    fut_model = Future_Seq2Seq_Control(enc, dec)



                loss_fn = torch.nn.MSELoss()
                optim = torch.optim.AdamW(fut_model.parameters(), lr=self.fut_ae_lr, weight_decay=self.fut_ae_wd)
                lr_sc = torch.optim.lr_scheduler.ExponentialLR(optim, gamma = self.fut_ae_lr_decay)

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
                    batch = 0
                    while not train_epoch_done:
                        batch += 1
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

                        if not self.scale_AE:
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
                        
                        future_traj_hat, _ = fut_model(y_rel)     
                            
                        optim.zero_grad()
                        if self.pos_loss:
                            y_hat = flow_dist_futMdl._rel_to_abs(future_traj_hat, x_t)
                            loss = torch.sqrt(loss_fn(y_hat, fut_traj))
                        else:
                            loss = torch.sqrt(loss_fn(future_traj_hat, y_rel))

                        loss.backward()
                        # Check for nan gradients
                        grad_is_nan = False
                        for param in fut_model.parameters():
                            if not torch.isfinite(param.grad).all():
                                grad_is_nan = True

                        if grad_is_nan:
                            print('Loss is not finite in batch {}'.format(batch))
                        else:
                            optim.step()
                            
                        train_loss.append(loss.detach().cpu().numpy())
                    
                    # Update learning rate
                    lr_sc.step()
                                    
                    fut_model.to(device = self.device)
                    fut_model.eval()
                            
                    with torch.no_grad():
                        conc_out = []
                        conc_label = []
                        
                        val_epoch_done = False
                        Num_steps = []
                        samples = 0
                        while not val_epoch_done:
                            X, Y, T, _, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                    val_split_size = 0.1)
                            
                            X, T, Y, _ = self.extract_batch_data(X, T, Y)
                            
                            past_data = X[:,0]
                            future_data = Y[:,0]
                            
                            past_traj, fut_traj, rot_angles_rad = flow_dist_futMdl._normalize_rotation(past_data, future_data)
                            
                            x_t = past_traj[...,-1:,:]
                            y_rel = flow_dist_futMdl._abs_to_rel(fut_traj, x_t)
                            
                            future_traj_hat, _ = fut_model(y_rel)
                            
                            if self.pos_loss:
                                y_hat = flow_dist_futMdl._rel_to_abs(future_traj_hat, x_t)
                                conc_out.append(y_hat.cpu())
                                conc_label.append(fut_traj.cpu())
                            else:
                                conc_out.append(future_traj_hat.cpu())
                                conc_label.append(y_rel.cpu())
                            
                            Num_steps.append(num_steps)
                            samples += X.shape[0]
                        
                        # Combine variable length predictions
                        Conc_out = torch.zeros(samples, max(Num_steps), 2).to(device = self.device)
                        Conc_label = torch.zeros(samples, max(Num_steps), 2).to(device = self.device)

                        i_start = 0
                        for i in range(len(conc_out)):
                            i_end = i_start + conc_out[i].shape[0]
                            Conc_out[i_start:i_end, :Num_steps[i]] = conc_out[i]
                            Conc_label[i_start:i_end, :Num_steps[i]] = conc_label[i]
                            i_start = i_end
                            
                        val_loss = torch.sqrt(loss_fn(Conc_out, Conc_label))

                        val_losses.append(val_loss.detach().cpu().numpy())
                        
                        # Early stopping for AE
                        # Check for convergence
                        if epoch > 200:
                            best_val_step = np.argmin(val_losses)
                            if epoch - best_val_step > 50:
                                converged = True
                        
                    print('Train loss: {:7.5f}; \t Val loss: {:7.5f}'.format(np.mean(train_loss),val_loss.data))
                    if converged:
                        print('Model is converged.')
                        break

                self.train_loss[0, :len(val_losses)] = np.array(val_losses)
                os.makedirs(os.path.dirname(fut_model_file), exist_ok=True)

            # Save model    
            pickle.dump(fut_model, open(fut_model_file, 'wb'))

        return fut_model


    def train_flow(self, fut_model, T_all):
        use_map = self.can_use_map and self.has_map

        
        if use_map:
            scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
        else:
            scene_encoder = None
        # TODO: Set the gnn parameters
        flow_dist = TrajFlow(pred_steps=self.fut_enc_sz, alpha=self.alpha, beta=self.beta_noise, gamma=self.gamma_noise, 
                               scene_encoder=scene_encoder, norm_rotation=self.norm_rotation, device=self.device,
                               obs_encoding_size=self.obs_encoding_size, scene_encoding_size=self.scene_encoding_size, n_layers_rnn=self.n_layers_rnn, 
                               es_rnn=self.hs_rnn, hs_rnn=self.hs_rnn, use_map=use_map, 
                               n_layers_gnn=4, es_gnn=32, T_all = T_all)
        
        for param in fut_model.parameters():
            param.requires_grad = False 
            param.grad = None
            
        
        flow_dist_file = self.model_file[:-4] + '_NF'
        
        if os.path.isfile(flow_dist_file) and not self.model_overwrite:
            flow_dist = pickle.load(open(flow_dist_file, 'rb'))
                          
        else:
            optimizer = torch.optim.AdamW(flow_dist.parameters(), lr=self.flow_lr, weight_decay=self.flow_wd)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.flow_lr_decay)

            val_losses = []


            for step in range(self.flow_epochs):

                flow_dist.train()
                fut_model.eval()
                
                losses_epoch = []
                val_losses_epoch = []
                
                train_epoch_done = False
                while not train_epoch_done:

                    if self.vary_input_length:
                        past_length_options = np.arange(0.5, self.num_timesteps_in*self.dt, 0.5)
                        sample_past_length = int(np.ceil(np.random.choice(past_length_options)/self.dt))
                    else:
                        sample_past_length = self.num_timesteps_in
                        
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
                    
                    if not self.scale_NF:
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

                # Update learning rate
                if step < 250:
                    lr_scheduler.step()   
                    
                    
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


    def train_method(self):    
        self.train_loss = np.ones((2, max(self.fut_ae_epochs, self.flow_epochs))) * np.nan
        
        # Get needed agent types
        T_all = self.provide_all_included_agent_types().astype(str)
        T_all = np.fromstring(T_all, dtype = np.uint32).reshape(len(T_all), int(str(T_all.astype(str).dtype)[2:])).astype(np.uint8)[:,0]
        
            
        # Train model components        
        self.fut_model = self.train_futureAE(T_all)
        self.flow_dist = self.train_flow(self.fut_model, T_all)
        
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
                prev_step = torch.tensor([1.0,0.0]).unsqueeze(0).repeat(actual_batch_size * self.num_samples_path_pred,1).to(self.device)
                
                outputs = torch.zeros(actual_batch_size * self.num_samples_path_pred, num_steps, 2).to(device = self.device)
                for t in range(0, num_steps):
                    if self.decoder_type == "none":
                        output, hidden = self.fut_model.decoder(x, hidden)
                    else:
                        output, hidden = self.fut_model.decoder(prev_step, x, hidden)
                    
                    prev_step = output.squeeze()
                    outputs[:, t, :] = prev_step
                    
                    x = hidden[-1].unsqueeze(1)
                
                y_hat = self.flow_dist._rel_to_abs(outputs, x_t)

                # invert rotation normalization
                y_hat = self.flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad.unsqueeze(1))

                y_hat = y_hat.reshape(actual_batch_size, self.num_samples_path_pred, num_steps, 2)
                
                Y_pred = y_hat.detach()
                    
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

        self.define_default_kwargs()

        if self.decoder_type == "none":
            kwargs_str = ""
        else:
            kwargs_str = self.decoder_type

        kwargs_str += 'seed' + str(self.model_kwargs['seed']) + '_' + \
                      'fut' + str(self.model_kwargs['fut_enc_sz']) + '_' + \
                      'sc' + str(self.model_kwargs['scene_encoding_size']) + '_' + \
                      'obs' + str(self.model_kwargs['obs_encoding_size']) + '_' + \
                      'alpha' + str(self.model_kwargs['alpha']) + '_' + \
                      'beta' + str(self.model_kwargs['beta_noise']) + '_' + \
                      'gamma' + str(self.model_kwargs['gamma_noise']) + '_' + \
                      'smin' + str(self.model_kwargs['s_min']) + '_' + \
                      'smax' + str(self.model_kwargs['s_max']) + '_' + \
                      'sigma' + str(self.model_kwargs['sigma']) 
        
        if self.fut_ae_lr_decay != 1.0:
            kwargs_str += '_ae_lrDec' + str(self.fut_ae_lr_decay)
        if self.flow_lr_decay != 1.0:
            kwargs_str += '_nf_lrDec' + str(self.flow_lr_decay)
                     
        if self.model_kwargs['vary_input_length']:
            kwargs_str += '_varyInLen'

        if self.model_kwargs['scale_AE']:
            kwargs_str += '_sclAE'

        if self.model_kwargs['scale_NF']:
            kwargs_str += '_sclNF'

        if self.model_kwargs['pos_loss']:
            kwargs_str += '_posLoss'

        if not self.model_kwargs['interactions']:
            kwargs_str += '_NoInteractions'

        model_str = 'TF_' + kwargs_str
        
        names = {'print': model_str,
                'file': model_str,
                'latex': r'\emph{%s}' % model_str
                }
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True
        
    def provides_epoch_loss(self = None):
        return True