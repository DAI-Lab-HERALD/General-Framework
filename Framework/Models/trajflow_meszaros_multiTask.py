from model_template import model_template
import numpy as np
import torch
import random
from TrajFlow_multiTask.flowModels import TrajFlow, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder
from TrajFlow_multiTask.LaneGCN import from_numpy

from TrajFlow_multiTask.multi_task_utils import get_crossing_trajectories, get_hypothetical_path_crossing, get_closeness

import pickle
import os

class trajflow_meszaros_multiTask(model_template):
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

        if not ('es_gnn' in self.model_kwargs.keys()):
            self.model_kwargs['es_gnn'] = 32

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

        if not ('pos_loss' in self.model_kwargs.keys()):
            self.model_kwargs['pos_loss'] = True

        if not('seed' in self.model_kwargs.keys()):
            self.model_kwargs['seed'] = 0

        if not('interaction_thresh' in self.model_kwargs.keys()):
            self.model_kwargs['interaction_thresh'] = 1/3

        if not('iCL_hdim' in self.model_kwargs.keys()):
            self.model_kwargs['iCL_hdim'] = 128

        if not('crossing_task' in self.model_kwargs.keys()):
            self.model_kwargs['crossing_task'] = False

        if not('hyp_crossing_task' in self.model_kwargs.keys()):
            self.model_kwargs['hyp_crossing_task'] = False

        if not('closeness_task' in self.model_kwargs.keys()):
            self.model_kwargs['closeness_task'] = False

        if not('multiTask_lossWeight' in self.model_kwargs.keys()):
            self.model_kwargs['multiTask_lossWeight'] = 1.0

        
        self.hs_rnn = self.model_kwargs['hs_rnn']
        self.es_gnn = self.model_kwargs['es_gnn']
        self.n_layers_rnn = self.model_kwargs['n_layers_rnn']
        self.fut_enc_sz = self.model_kwargs['fut_enc_sz'] 

        self.scene_encoding_size = self.model_kwargs['scene_encoding_size']
        self.obs_encoding_size = self.model_kwargs['obs_encoding_size'] 
        
        self.beta_noise = self.model_kwargs['beta_noise']
        self.gamma_noise = self.model_kwargs['gamma_noise']
        
        self.alpha = self.model_kwargs['alpha']

        self.fut_ae_epochs = self.model_kwargs['fut_ae_epochs']
        self.fut_ae_lr = self.model_kwargs['fut_ae_lr']
        self.fut_ae_lr_decay = self.model_kwargs['fut_ae_lr_decay']
        self.fut_ae_wd = self.model_kwargs['fut_ae_wd']

        self.flow_epochs = self.model_kwargs['flow_epochs']
        self.flow_lr = self.model_kwargs['flow_lr']
        self.flow_lr_decay = self.model_kwargs['flow_lr_decay']
        self.flow_wd = self.model_kwargs['flow_wd']

        self.vary_input_length = self.model_kwargs['vary_input_length']
        self.pos_loss = self.model_kwargs['pos_loss']

        # Get Decoder Type
        self.interaction_thresh = self.model_kwargs['interaction_thresh']

        self.lanegcn_config = dict()

        self.lanegcn_config["num_scales"] = 5
        self.lanegcn_config["n_actor"] = self.model_kwargs['obs_encoding_size'] 
        self.lanegcn_config["n_map"] = 128
        self.lanegcn_config["actor2map_dist"] = 7.0
        self.lanegcn_config["map2actor_dist"] = 6.0
        self.lanegcn_config["actor2actor_dist"] = 100.0

        self.iCL_hdim = self.model_kwargs['iCL_hdim']

        self.crossing_task = self.model_kwargs['crossing_task']
        self.hyp_crossing_task = self.model_kwargs['hyp_crossing_task']
        self.closeness_task = self.model_kwargs['closeness_task']

    
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
        self.predict_single_agent = False
        self.can_use_map = True
        self.can_use_graph = True
        # If self.can_use_map, the following is also required
        self.target_width = 257
        self.target_height = 156
        self.grayscale = True
        
        self.norm_rotation = True
        
        self.std_pos_ped = 1
        self.std_pos_veh = 1 
        
        
    
    def extract_batch_data(self, X, T, Y = None, img = None, graph = None):
        
        
        if img is not None:
            img = torch.from_numpy(img).float().to(device = self.device) / 255

        data_graphs = []
        if graph is not None:
            for b in range(X.shape[0]):
                orig = X[b,0,0,:2]
                x_t_rel = X[b,0,[-1],:2] - X[b,0,[-2],:2]
                rot_angles_rad = -1 * np.arctan2(x_t_rel[0,1], x_t_rel[0,0])
                
                rot = np.asarray([
                        [np.cos(rot_angles_rad), -np.sin(rot_angles_rad)],
                        [np.sin(rot_angles_rad), np.cos(rot_angles_rad)]], np.float32)
                            
                graph_b = dict(graph[b])

                # process lane centerline in same way as agent trajectories
                centerlines_b = [np.matmul(rot, (centerline - orig.reshape(-1, 2)).T).T for centerline in graph_b['centerlines']]
                left_boundary_b = [np.matmul(rot, (left_boundary - orig.reshape(-1, 2)).T).T for left_boundary in graph_b['left_boundaries']]
                right_boundary_b = [np.matmul(rot, (right_boundary - orig.reshape(-1, 2)).T).T for right_boundary in graph_b['right_boundaries']]
                
                ctrs_b = [np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32) for centerline in centerlines_b]
                ctrs_b = np.concatenate(ctrs_b, axis=0)

                feats_b = [np.asarray(centerline[1:] - centerline[:-1], np.float32) for centerline in centerlines_b]
                feats_b = np.concatenate(feats_b, axis=0)

                graph_b['centerlines'] = centerlines_b
                graph_b['left_boundaries'] = left_boundary_b
                graph_b['right_boundaries'] = right_boundary_b
                graph_b['ctrs'] = ctrs_b
                graph_b['feats'] = feats_b

                graph_b = from_numpy(graph_b)
                graph_b['left'] = graph_b['left'][0]
                graph_b['right'] = graph_b['right'][0]
                
                data_graphs.append(graph_b)

            graph = data_graphs

        if Y is not None:
            Y = Y[..., :2]
        
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
            
        return X, T_out, Y, img, graph
    
    
    def train_futureAE(self, T_all):
        

        # Use encoder of noninteractive trajflow model if available, as same training stuff is used
        fut_model_file = self.model_file[:-4] + '_AE'
        if os.path.isfile(fut_model_file) and not self.model_overwrite:
            fut_model = pickle.load(open(fut_model_file, 'rb'))
            print('Future AE model loaded')

        else:
            dir = os.path.dirname(self.model_file)
            # Check if equivalent AE model is available
            if not os.path.exists(dir):
                os.makedirs(dir)
            # Find all files in Model folder
            model_files = os.listdir(dir)

            # Find all AE models
            model_files = [f for f in model_files if f[-16:] == '--train_loss.npy']

            # Get desired environment name
            env_name_desired = '--'.join(fut_model_file.split(os.sep)[-1].split('--')[:-1])
            model_kwargs_potential = fut_model_file.split(os.sep)[-1].split('--')[-1][3:-3].split('_')

            AE_kwargs = ['fut', 'alpha', 'posLoss', 'varyInLen']

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
            eqivalent_file = None
            for file in model_files:
                # Diveide Model name from rest
                File_split = file.split('--')
                
                # Check env name
                env_name = '--'.join(File_split[:-2])
                if env_name != env_name_desired:
                    continue

                # Get model name
                model_name = File_split[-2][:-3]

                # Check for TF model
                if model_name[:13] != 'TF_multiTask_':
                    continue

                # Check for parameters
                model_kwargs_pot = model_name[13:].split('_')

                # Find important kwargs
                model_kwargs = []
                for kw in model_kwargs_pot:
                    for ae_kw in AE_kwargs:
                        if ae_kw in kw:
                            model_kwargs.append(kw)
                            break
                    
                    # Check for lr_decay
                    if kw == 'lrDec':
                        if model_kwargs_potential[i_kw - 1] == "ae":
                            model_kwargs.append(kw)

                # Check if AE settings are identical
                if model_kwargs == model_kwargs_desired:
                    eqivalent_file = file
                    break

            if eqivalent_file is not None and not self.model_overwrite:
                # Load model
                equiv_model_file = os.path.dirname(self.model_file) + os.sep + eqivalent_file
                equiv_model_file = equiv_model_file[:-16] + '_AE'
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
                                            n_layers_rnn=n_layers_rnn, es_rnn=hs_rnn, hs_rnn=hs_rnn, T_all=T_all)

                enc = Future_Encoder(2, enc_size, enc_size, enc_size)

                dec = Future_Decoder(2, enc_size, enc_size)
                fut_model = Future_Seq2Seq(enc, dec)



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

                        X, Y, T, img, _, _, Pred_agents, num_steps, _, _, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                            val_split_size = 0.1)
                        X, T, Y, _, _ = self.extract_batch_data(X, T, Y)
                        
                        # X.shape:   bs x num_agents x num_timesteps_is x 2
                        # Y.shape:   bs x 1 x num_timesteps_is x 2
                        # T.shape:   bs x num_agents
                        # img.shape: bs x 1 x 156 x 257 x 1     
                        # graph.shape: bs x 1                       
                        past_data = X[..., :2]
                        future_data = Y
                        
                        past_traj, fut_traj, rot_angles_rad = flow_dist_futMdl._normalize_rotation(past_data, future_data)
                        
                        x_t = past_traj[...,-1:,:]
                        y_rel = flow_dist_futMdl._abs_to_rel(fut_traj, x_t)
                        y_rel = y_rel[Pred_agents,:,:]
                        
                        future_traj_hat, _ = fut_model(y_rel)     
                            
                        optim.zero_grad()
                        if self.pos_loss:
                            y_hat = flow_dist_futMdl._rel_to_abs(future_traj_hat, x_t[Pred_agents,:,:])
                            loss = torch.sqrt(loss_fn(y_hat, fut_traj[Pred_agents,:,:]))
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
                            X, Y, T, _, _, _, Pred_agents, num_steps, _, _, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                    val_split_size = 0.1)
                            
                            X, T, Y, _, _ = self.extract_batch_data(X, T, Y)
                            
                            past_data = X[..., :2]
                            future_data = Y
                            
                            past_traj, fut_traj, rot_angles_rad = flow_dist_futMdl._normalize_rotation(past_data, future_data)
                            
                            x_t = past_traj[...,-1:,:]
                            y_rel = flow_dist_futMdl._abs_to_rel(fut_traj, x_t)
                            y_rel = y_rel[Pred_agents,:,:]
                            
                            future_traj_hat, _ = fut_model(y_rel)
                            
                            if self.pos_loss:
                                y_hat = flow_dist_futMdl._rel_to_abs(future_traj_hat, x_t[Pred_agents,:,:])
                                conc_out.append(y_hat.cpu())
                                conc_label.append(fut_traj[Pred_agents,:,:].cpu())
                            else:
                                conc_out.append(future_traj_hat.cpu())
                                conc_label.append(y_rel.cpu())
                            
                            Num_steps.append(num_steps)
                            samples += X[Pred_agents,:,:].shape[0]
                        
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
                        if epoch > 100:
                            best_val_step = np.argmin(val_losses)
                            if epoch - best_val_step > 25:
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
        use_graph = self.can_use_graph and self.has_graph
        
        if use_map:
            scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
        else:
            scene_encoder = None

        # TODO: Set the gnn parameters
        flow_dist = TrajFlow(pred_steps=self.fut_enc_sz, alpha=self.alpha, beta=self.beta_noise, gamma=self.gamma_noise, 
                               scene_encoder=scene_encoder, norm_rotation=self.norm_rotation, device=self.device,
                               obs_encoding_size=self.obs_encoding_size, scene_encoding_size=self.scene_encoding_size, n_layers_rnn=self.n_layers_rnn, 
                               es_rnn=self.hs_rnn, hs_rnn=self.hs_rnn, use_map=use_map, use_graph=use_graph,
                               n_layers_gnn=1, es_gnn=self.es_gnn, T_all = T_all, interaction_thresh=self.interaction_thresh, lanegcn_configs=self.lanegcn_config,
                               iCL_hdim = self.iCL_hdim)
        
        for param in fut_model.parameters():
            param.requires_grad = False 
            param.grad = None
            
        
        flow_dist_file = self.model_file[:-4] + '_NF'
        flow_dist_pt_file = self.model_file[:-4] + '_NF.pt'

        epoch_file = self.model_file[:-4] + '_NF_epoch.pkl'

        start_epoch = 0
        if os.path.exists(epoch_file):
            start_epoch = pickle.load(open(epoch_file, 'rb'))
            start_epoch = start_epoch + 1

        
        if os.path.isfile(flow_dist_file) and not self.model_overwrite:
            flow_dist = pickle.load(open(flow_dist_file, 'rb'))
                          
        else:
            if os.path.isfile(flow_dist_pt_file) and not self.model_overwrite:
                flow_dist = pickle.load(open(flow_dist_pt_file, 'rb'))

            optimizer = torch.optim.AdamW(flow_dist.parameters(), lr=self.flow_lr, weight_decay=self.flow_wd)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.flow_lr_decay)

            multi_task_optim = torch.optim.AdamW(flow_dist.interaction_classifier.parameters(), lr=self.flow_lr, weight_decay=self.flow_wd)

            val_losses = []


            CE_loss = torch.nn.CrossEntropyLoss()

            print('Train Normalizing Flow', flush = True)

            for step in range(start_epoch, self.flow_epochs):
                epoch_string = 'Train Normalizing Flow: Epoch ' + str(step + 1).zfill(len(str(self.flow_epochs)))
                print('', flush = True)
                print(epoch_string, flush=True)

                flow_dist.train()
                fut_model.eval()
                
                losses_epoch = []
                val_losses_epoch = []
                
                train_epoch_done = False
                batch = 0
                while not train_epoch_done:
                    batch += 1
                    print('', flush = True)
                    print(epoch_string + ' - Batch {}'.format(batch), flush = True)

                    if self.vary_input_length:
                        past_length_options = np.arange(0.5, self.num_timesteps_in*self.dt, 0.5)
                        sample_past_length = int(np.ceil(np.random.choice(past_length_options)/self.dt))
                    else:
                        sample_past_length = self.num_timesteps_in
                    
                    print(epoch_string + ' - Batch {} - preprocess data'.format(batch), flush = True)    
                    X, Y, T, img, _, graph, Pred_agents, num_steps, _, _, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                           val_split_size = 0.1)
                    X, T, Y, img, graph = self.extract_batch_data(X, T, Y, img, graph)
                    
                    # X.shape:   bs x num_agents x num_timesteps_is x 2
                    # Y.shape:   bs x num_agents x num_timesteps_is x 2
                    # T.shape:   bs x num_agents
                    # img.shape: bs x 1 x 156 x 257 x 1
                    
                    X = X[:,:,-sample_past_length:,:]
                    
                    past_data   = X[..., :2]
                    future_data = Y
                    
                    optimizer.zero_grad()
                    
                    past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data, future_data)
                    
                    x_t   = past_traj[:,:,-1:,:]
                    y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                    # x_rel = (past_traj[...,1:,:] - past_traj[...,:-1,:])* self.alpha

                    if img is not None:
                        img = img.permute(0,1,4,2,3)

                    y_rel = y_rel.reshape(y_rel.shape[0]*y_rel.shape[1], y_rel.shape[2], y_rel.shape[3])
                    out, _ = fut_model.encoder(y_rel)
                    out = out[:,-1]
                    # out.shape:       batch size x enc_dims
                    
                    print(epoch_string + ' - Batch {} - calculate loss'.format(batch), flush = True)
                    logprob, j_logprob, num_agents_per_sample = flow_dist.log_prob(out, past_data, T, scene=img, scene_graph=graph)

                    loss = -torch.nanmean(j_logprob / num_agents_per_sample) # NLL
                    # loss = -torch.nanmean(logprob) # NLL
                    losses_epoch.append(loss.item())
                    
                    print(epoch_string + ' - Batch {} - backpropagate'.format(batch), flush = True)
                    loss.backward()
                    optimizer.step()

                    if self.crossing_task or self.hyp_crossing_task or self.closeness_task:
                        
                        print(epoch_string + ' - Batch {} - preprocess tasks'.format(batch), flush = True)

                        loss_task1 = 0
                        loss_task2 = 0
                        loss_task3 = 0

                        x_enc = flow_dist._encode_trajectories(past_data, T)               
                            
                        # Define sizes
                        max_num_agents = x_enc.shape[1]
                        

                        D = past_data[:,None,:,-1] - past_data[:,:,None,-1] # shape: num_samples x max_num_agents x max_num_agents x 2

                        ang = flow_dist._get_agent_pair_angles(past_data, D, max_num_agents)
                        
                        # Get agent pair information
                        T_one_hot = (T.unsqueeze(-1) == flow_dist.t_unique.unsqueeze(0)).float() # shape: num_samples x max_num_agents x num_classes

                        pair_info = torch.cat((x_enc[:,None,:].repeat(1,max_num_agents,1,1), x_enc[:,:,None,:].repeat(1,1,max_num_agents,1),
                                T_one_hot[:,None,:].repeat(1,max_num_agents,1,1), T_one_hot[:,:,None].repeat(1,1,max_num_agents,1), 
                                D, ang[:,:,:,None]), dim = -1)
                        
                        print(epoch_string + ' - Batch {} - run tasks'.format(batch), flush = True)
                        
                        if self.crossing_task:
                            
                            crossing_pair_ids, crossing_classes = get_crossing_trajectories(Y=Y.cpu().numpy(), 
                                                                                                T=T.cpu().numpy())
                            
                            classes = np.concatenate(crossing_classes)

                            weights = torch.tensor([(classes==0).sum(axis=0), (classes==1).sum(axis=0), (classes==2).sum(axis=0)])
                            weights = (len(classes)/weights).to(self.device)
                            CE_loss = torch.nn.CrossEntropyLoss(weight=weights)

                            # reduce the number of samples for majority class through random downsampling
                            # if (classes==0).sum() > 2*(classes==1).sum():
                            #     remove_count = (classes==0).sum() - (classes==1).sum()
                            #     remove_indices = np.random.choice(np.where(classes==0)[0], remove_count, replace=False)
                            #     classes = np.delete(classes, remove_indices)

                            classes_one_hot = torch.from_numpy(classes[:,None] == np.array([[0,1,2]])).float().to(self.device)

                            pair_info_t1 = []

                            for i in range(pair_info.shape[0]):
                                pair_info_t1.append(pair_info[i, crossing_pair_ids[i][:,0], crossing_pair_ids[i][:,1], :])

                            pair_info_t1 = torch.concatenate(pair_info_t1)
                            # mask = torch.ones(pair_info_t1.size(0), dtype=torch.bool)
                            # mask[remove_indices] = False
                            # pair_info_t1 = pair_info_t1[mask,:]

                            interaction_class = flow_dist.interaction_classifier(pair_info_t1, task=1)
                            loss_task1 = CE_loss(interaction_class, classes_one_hot)

                        if self.hyp_crossing_task:
                            hyp_crossing_pair_ids, hyp_crossing_classes = get_hypothetical_path_crossing(X.cpu().numpy(), 
                                                                                                             Y.cpu().numpy(), 
                                                                                                             T.cpu().numpy(), 
                                                                                                             self.dt)

                            classes = np.concatenate(hyp_crossing_classes)

                            weights = torch.tensor([(classes==0).sum(axis=0), (classes==1).sum(axis=0), (classes==2).sum(axis=0)])
                            weights = (len(classes)/weights).to(self.device)
                            CE_loss = torch.nn.CrossEntropyLoss(weight=weights)

                            # reduce the number of samples for majority class through random downsampling
                            # if (classes==0).sum() > 2*(classes==1).sum():
                            #     remove_count = (classes==0).sum() - (classes==1).sum()
                            #     remove_indices = np.random.choice(np.where(classes==0)[0], remove_count, replace=False)
                            #     classes = np.delete(classes, remove_indices)
                                
                            classes_one_hot = torch.from_numpy(classes[:,None] == np.array([[0,1,2]])).float().to(self.device)

                            pair_info_t2 = []

                            for i in range(pair_info.shape[0]):
                                pair_info_t2.append(pair_info[i, hyp_crossing_pair_ids[i][:,0], hyp_crossing_pair_ids[i][:,1], :])

                            pair_info_t2 = torch.concatenate(pair_info_t2)
                            # mask = torch.ones(pair_info_t2.size(0), dtype=torch.bool)
                            # mask[remove_indices] = False
                            # pair_info_t2 = pair_info_t2[mask,:]

                            interaction_class = flow_dist.interaction_classifier(pair_info_t2, task=2)
                            loss_task2 = CE_loss(interaction_class, classes_one_hot)

                        if self.closeness_task:
                            closeness_pair_ids, closeness_classes = get_closeness(Y=Y.cpu().numpy())

                            classes = np.concatenate(closeness_classes)
                            classes = torch.from_numpy(classes).float().to(self.device)

                            weight = (classes==0).sum(axis=0)/((classes==1).sum(axis=0) + 1e-6)
                            BCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

                            pair_info_t3 = []

                            for i in range(pair_info.shape[0]):
                                pair_info_t3.append(pair_info[i, closeness_pair_ids[i][:,0], closeness_pair_ids[i][:,1], :])

                            pair_info_t3 = torch.concatenate(pair_info_t3)

                            interaction_class = flow_dist.interaction_classifier(pair_info_t3, task=3)
                            loss_task3 = BCE_loss(interaction_class, classes)

                        loss_multi_task = (loss_task1 + loss_task2 + loss_task3)*self.model_kwargs['multiTask_lossWeight']
                        multi_task_optim.zero_grad()
                        loss_multi_task.backward()
                        multi_task_optim.step()

                print('', flush = True)
                print(epoch_string + ' - Saving', flush = True)    
                pickle.dump(step, open(epoch_file, 'wb'))
                pickle.dump(flow_dist, open(flow_dist_pt_file, 'wb'))

                # Update learning rate
                if step < 250:
                    lr_scheduler.step()   
                    
                
                print('', flush = True)
                print(epoch_string + ' - Validation', flush = True)    
                flow_dist.eval()
                fut_model.eval()
                with torch.no_grad():
                    val_epoch_done = False
                    while not val_epoch_done:
                        X, Y, T, img, _, graph, Pred_agents, num_steps, _, _, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                val_split_size = 0.1)
                        X, T, Y, img, graph = self.extract_batch_data(X, T, Y, img, graph)
                        
                        past_data_val = X[..., :2]
                        future_data_val = Y
                        
                        past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data_val, future_data_val)
                        
                        x_t = past_traj[:,:,-1:,:]
                        y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                        
                        if img is not None:
                            img_val = img.permute(0,1,4,2,3)
                        else:
                            img_val = None

                        y_rel = y_rel.reshape(y_rel.shape[0]*y_rel.shape[1], y_rel.shape[2], y_rel.shape[3])
                        out, _ = fut_model.encoder(y_rel)
                        out = out[:, -1]
                        # out.shape: batch size x enc_dims
                            
                        optimizer.zero_grad()

                        log_prob, j_logprob, num_agents_per_sample = flow_dist.log_prob(out, past_data_val, T, scene=img_val, scene_graph=graph)
                    
                        val_loss = -torch.nanmean(j_logprob / num_agents_per_sample)
                        # val_loss = -torch.nanmean(log_prob)
                        val_losses_epoch.append(val_loss.item())
                        
                    val_losses.append(np.mean(val_losses_epoch))      
                
                # Check for convergence
                if step > 30:
                    best_val_step = np.argmin(val_losses)
                    if step - best_val_step > 10:
                        print('', flush = True)
                        print('Converged', flush = True)
                        print(epoch_string + ' - loss = {:0.4f}; val_loss = {:0.4f}'.format(np.mean(losses_epoch), np.mean(val_losses_epoch)), flush = True)
                        break
                
                print('', flush = True)
                print(epoch_string + ' - loss = {:0.4f}; val_loss = {:0.4f}'.format(np.mean(losses_epoch), np.mean(val_losses_epoch)))

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
        # assert False # Sampling not implemented for factorized prediction
        prediction_done = False
        while not prediction_done:
            X, T, img, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.batch_size)
            actual_batch_size = len(X)
            num_agents = X.shape[1]
            Ped_agent = T == 'P'
            
            X, T, _, img, graph = self.extract_batch_data(X, T, img = img, graph=graph)
            # Run prediction pass
            with torch.no_grad():
                past_traj, rot_angles_rad = self.flow_dist._normalize_rotation(X[..., :2])
                
                if img is not None:
                    img = img.permute(0,1,4,2,3)
                else:
                    img = None
                
                
                x_t = past_traj[:,:,[-1],:]
                x_t = x_t.repeat_interleave(self.num_samples_path_pred, dim=1)

                rot_angles_rad = rot_angles_rad.repeat_interleave(self.num_samples_path_pred, dim=1)
                if rot_angles_rad.shape[0] != actual_batch_size:
                    rot_angles_rad = rot_angles_rad.unsqueeze(0)

                samples_rel, log_probs, j_log_probs, batch_ids, row_ids = self.flow_dist.sample(self.num_samples_path_pred, X[..., :2], T, scene=img, scene_graph=graph)
            

                samples_rel = samples_rel.reshape(-1, samples_rel.shape[-1])
                # row_ids = row_ids.unsqueeze(-1).repeat(1, self.num_samples_path_pred).flatten()
                # batch_ids = batch_ids.unsqueeze(-1).repeat(1, self.num_samples_path_pred)
                # batch_ids += torch.arange(0, self.num_samples_path_pred, device=self.device).unsqueeze(0) * actual_batch_size
                # batch_ids = batch_ids.flatten()
                        
                hidden = torch.tile(samples_rel.unsqueeze(0), (self.fut_model.decoder.nl,1,1))
                
                # Decoder part
                x = samples_rel.unsqueeze(1)#.reshape(-1, self.fut_enc_sz).unsqueeze(1)
                prev_step = torch.tensor([1.0,0.0]).unsqueeze(0).repeat(samples_rel.shape[0],1).to(self.device)

                # agent_exists = T.clone().detach() != 48
                # agent_exists = agent_exists.repeat_interleave(self.num_samples_path_pred, dim=1).flatten()
                # agent_exists = agent_exists.unsqueeze(1).repeat(1, self.num_samples_path_pred, 1)
                # agent_exists = agent_exists.flatten()
                
                outputs = torch.zeros(actual_batch_size, num_agents, self.num_samples_path_pred, num_steps, 2).to(device = self.device)
                for t in range(0, num_steps):
                    output, hidden = self.fut_model.decoder(x, hidden)
                    
                    prev_step = output.squeeze()
                    output = output.reshape(-1, self.num_samples_path_pred, 2)
                    outputs[batch_ids, row_ids, :, t, :] = output
                    
                    x = hidden[-1].unsqueeze(1)
                y_hat = self.flow_dist._rel_to_abs(outputs.reshape(actual_batch_size, -1, num_steps, 2), x_t)

                # invert rotation normalization
                y_hat = self.flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad)

                # Y_pred = torch.zeros(actual_batch_size, num_agents, self.num_samples_path_pred, num_steps, 2)
                # Pred_agents_expanded = 

                y_hat = y_hat.reshape(actual_batch_size, num_agents, self.num_samples_path_pred, num_steps, 2)
                
                Y_pred = y_hat.detach()

            Pred = Y_pred.detach().cpu().numpy()
            if len(Pred.shape) == 3:
                Pred = Pred[np.newaxis]
            
            Pred[Ped_agent]  *= self.std_pos_ped
            Pred[~Ped_agent] *= self.std_pos_veh
            
            torch.cuda.empty_cache()
            
            # save predictions
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)
    
    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_name(self = None):

        self.define_default_kwargs()

        kwargs_str = ''

        kwargs_str += 'seed' + str(self.model_kwargs['seed']) + '_' + \
                      'fut' + str(self.model_kwargs['fut_enc_sz']) + '_' + \
                      'sc' + str(self.model_kwargs['scene_encoding_size']) + '_' + \
                      'obs' + str(self.model_kwargs['obs_encoding_size']) + '_' + \
                      'es_gnn' + str(self.model_kwargs['es_gnn']) + '_' + \
                      'iCL_hdim' + str(self.model_kwargs['iCL_hdim']) + '_' + \
                      'alpha' + str(self.model_kwargs['alpha'])
        
        if self.model_kwargs['vary_input_length']:
            kwargs_str += '_varyInLen'

        if self.model_kwargs['pos_loss']:
            kwargs_str += '_posL'

        if self.model_kwargs['crossing_task']:
            kwargs_str += '_crTask'

        if self.model_kwargs['hyp_crossing_task']:
            kwargs_str += '_hCrTask'

        if self.model_kwargs['closeness_task']:
            kwargs_str += '_clTask'


        model_str = 'TF_multiTask_' + kwargs_str 
        
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