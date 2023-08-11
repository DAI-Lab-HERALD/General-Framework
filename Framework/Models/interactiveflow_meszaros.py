from model_template import model_template
import numpy as np
import pandas as pd
import torch
import random
from JointPredictions.flowModels import InteractiveFlow
import pickle
from torch.utils.data import TensorDataset, DataLoader
from JointPredictions.modules import FutureSceneAE
from JointPredictions.utils import abs_to_rel, rel_to_abs, normalize_rotation, rotate
import os
import torch.nn.functional as F

class interactiveflow_meszaros(model_template):
    
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


        self.ft_enc_dim = 8
        self.pos_emb_dim = 2
        self.fre_nin = 2
        self.fre_nl = 3
        self.fsg_nl = 3
        self.fs_enc_dim = 8
        self.frd_nout = 2
        self.frd_nl = 3

        # TODO dependent on dataset
        if self.data_set.get_name()['file'][:3] == 'ETH':
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

        self.map_encoding_size = 8
        self.fut_ae_epochs = 10000
        self.fut_ae_lr = 1e-3
        self.fut_ae_wd = 1e-3

        self.flow_epochs = 200
        self.flow_lr = 1e-3
        self.flow_wd = 1e-5

        self.std_pos_ped = 1
        self.std_pos_veh = 1
        
        
        # Set time step
        self.dt = self.Input_T_train[0][-1] - self.Input_T_train[0][-2]
        # TODO see if this can be made variable
        self.img_dim = [self.target_height, self.target_width]
        
        
    def extract_data(self, train = True):
        
        if train:
            X_help = self.Input_path_train.to_numpy()
            Y_help = self.Output_path_train.to_numpy() 
            Types  = self.Type_train.to_numpy()
            
            X_help = X_help[self.remain_samples]
            Y_help = Y_help[self.remain_samples]
            Types  = Types[self.remain_samples]

            self.domain_old = self.Domain_train.iloc[self.remain_samples]
        else:
            X_help = self.Input_path_test.to_numpy()
            Types  = self.Type_test.to_numpy()
            self.domain_old = self.Domain_test
        
        Agents = np.array(self.input_names_train)
        
        # Extract predicted agents
        Pred_agents = np.array([agent in self.data_set.needed_agents for agent in Agents])
        assert Pred_agents.sum() > 0, "nothing to predict"
        
        X = np.ones(list(X_help.shape) + [self.num_timesteps_in, 2], dtype = np.float32) * np.nan
        if train:
            Y = np.ones(list(Y_help.shape) + [self.num_timesteps_out.max(), 2], dtype = np.float32) * np.nan
            
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
            Xi = X # num_samples, num_agents, num_timesteps, 2
            T = Types
            
            
            # set agent to be predicted into first location
            # X = []
            # T = []
            # for i_agent in np.where(Pred_agents)[0]:
            #     reorder_index = np.array([i_agent] + list(np.arange(i_agent)) + 
            #                              list(np.arange(i_agent + 1, Xi.shape[1])))
            #     X.append(Xi[:,reorder_index])
            #     T.append(Types[:, reorder_index])
            # X = np.stack(X, axis = 1).reshape(-1, Xi.shape[1], self.num_timesteps_in, 2)
            # T = np.stack(T, axis = 1).reshape(-1, Types.shape[1])
            T = T.astype(str)
            PPed_agents = T == 'P'
            # transform to ascii int:
            T[T == 'nan'] = '0'
            T = np.fromstring(T.reshape(-1), dtype = np.uint32).reshape(*T.shape, 3).astype(np.uint8)[:,:,0]

            if self.use_map:
                
                Img_needed = T != 48
                centre = X[Img_needed, -1,:]
                x_rel = centre - X[Img_needed, -2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 

                domain_index = self.domain_old.index.to_numpy()
                domain_index = domain_index.repeat(Img_needed.sum(1))
                domain_repeat = self.domain_old.loc[domain_index]

                img = np.zeros((X.shape[0], X.shape[1], self.target_height, self.target_width, 1), dtype = 'uint8')
                img[Img_needed] = self.data_set.return_batch_images(domain_repeat, centre, rot, target_height=self.target_height, target_width=self.target_width, grayscale = True)
                
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                Y[Ped_agents]  /= self.std_pos_ped
                Y[~Ped_agents] /= self.std_pos_veh
                # Y = Y[:, Pred_agents].reshape(-1, 1, self.num_timesteps_out.max(), 2)
                
                my_dataset = TensorDataset(torch.tensor(X).to(device=self.device),
                                           torch.tensor(Y).to(device=self.device), 
                                           torch.tensor(img),
                                           torch.tensor(T).to(device=self.device)) # create your datset
                
            else:
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                Y[Ped_agents]  /= self.std_pos_ped
                Y[~Ped_agents] /= self.std_pos_veh
                # Y = Y[:, Pred_agents].reshape(-1, 1, self.num_timesteps_out.max(), 2)
                
                my_dataset = TensorDataset(torch.tensor(X).to(device=self.device),
                                           torch.tensor(Y).to(device=self.device),
                                           torch.tensor(T).to(device=self.device)) # create your datset

            
            train_data, val_data = torch.utils.data.random_split(my_dataset, 
                                                                 [int(np.round(len(my_dataset)*0.9)),
                                                                  int(len(my_dataset) - np.round(len(my_dataset)*0.9))],
                                                                 generator=torch.Generator().manual_seed(42))

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
            
            return train_loader, val_loader, T
        else:
            Xi = X # num_samples, num_agents, num_timesteps, 2
            # set agent to be predicted int
            
            # X = X.transpose(0,1,3,2) # num_samples, num_agents, num_timesteps, 2            
            
            T = Types #np.tile(Types[np.newaxis,:], (len(X), 1))
            T = T.astype(str)
            PPed_agents = T == 'P'
            T[T == 'nan'] = '0'
            # transform to ascii int:
            T = T = np.fromstring(T.reshape(-1), dtype = np.uint32).reshape(*T.shape, 3).astype(np.uint8)[:,:,0] #np.fromstring(T.reshape(-1), dtype = np.uint32).reshape(len(T), -1).astype(np.uint8)

            if self.use_map:
                
                Img_needed = T != 48
                centre = X[Img_needed, -1,:]
                x_rel = centre - X[Img_needed, -2,:]
                rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 

                domain_index = self.domain_old.index.to_numpy()
                domain_index = domain_index.repeat(Img_needed.sum(1))
                domain_repeat = self.domain_old.loc[domain_index]

                img = np.zeros((X.shape[0], X.shape[1], self.target_height, self.target_width, 1), dtype = 'uint8')
                img[Img_needed] = self.data_set.return_batch_images(domain_repeat, centre, rot, target_height=self.target_height, target_width=self.target_width, grayscale = True)
                
                
                # img = img.reshape(len(X), X.shape[1], self.target_height, self.target_width, -1)
                
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                return Pred_agents, Agents, X, T, PPed_agents, img
            
            else:
                X[PPed_agents]   /= self.std_pos_ped
                X[~PPed_agents]  /= self.std_pos_veh
                return Pred_agents, Agents, X, T, PPed_agents


    def train_futureAE(self, train_loader, val_loader, T_all):

        t_unique = torch.unique(torch.from_numpy(T_all).to(self.device))
        t_unique = t_unique[t_unique != 48]
        
        future_traj_enc_dim = self.ft_enc_dim 
        pos_emb_dim = self.pos_emb_dim 
        futureRNNencParams = {'nin': self.fre_nin, 'nout': future_traj_enc_dim, 'es': future_traj_enc_dim, 'hs': future_traj_enc_dim, 'nl': self.fre_nl} 
        futureSocialGNNparams = {'num_layers': self.fsg_nl, 'emb_dim': future_traj_enc_dim, 'in_dim': futureRNNencParams['hs'], 'edge_dim': len(t_unique) * 2 + 1} 
        enc_dim = self.fs_enc_dim 
        futureRNNdecParams = {'nout': self.frd_nout, 'es': future_traj_enc_dim, 'hs': future_traj_enc_dim, 'nl': self.frd_nl}


        future_scene_ae = FutureSceneAE(futureRNNencParams, futureRNNdecParams, futureSocialGNNparams, enc_dim, pos_emb_dim, T_all, device=self.device).to(device = self.device)

        fut_model_file = self.model_file[:-16] + '--InteFlow_M_AE'
        if os.path.isfile(fut_model_file) and not self.data_set.overwrite_results:
            future_scene_ae = pickle.load(open(fut_model_file, 'rb'))
            print('Future AE model loaded')

        else:

            loss_fn = torch.nn.MSELoss() 
            optim = torch.optim.AdamW(future_scene_ae.parameters(), lr=self.fut_ae_lr, weight_decay=self.fut_ae_wd)

            train_loss = []
            val_loss = []
            for epoch in range(self.fut_ae_epochs):
                future_scene_ae.train()
                train_loss_ep = []
                val_loss_ep = []

                converged = False

                print('Training Future Scene AE...')
                for i_batch, data in enumerate(train_loader):


                    past_pos = data[0].to(device = self.device)
                    future_pos = data[1].to(device = self.device)
                    agent_types = data[-1].to(device = self.device)

                    future_traj_orig = future_pos.to(self.device).float()
                    past_traj_orig = past_pos.to(self.device).float()

                    # Loss = []
                    # for i_samples in range(len(past_pos)):
                    
                    # shuffle agents to potentially get different target agents for the decoding (should improve generalization) TODO check
                    shuffle_ids = torch.randperm(past_traj_orig.size()[1])
                    future_traj_orig_sample = future_traj_orig[:,shuffle_ids,:,:]
                    past_traj_orig_sample = past_traj_orig[:,shuffle_ids,:,:]
                    agent_types_sample = agent_types[:,shuffle_ids]

                    # normalise rotation such that trajectory at t=0 is always pointing in the same direction (x-axis)
                    past_traj, future_traj, rot_angles_rad = normalize_rotation(x=past_traj_orig_sample, y_true=future_traj_orig_sample)
                    x_t = past_traj[...,-1:,:]
                    
                    # convert to relative coordinates
                    future_disp = abs_to_rel(future_traj, x_t, alpha=self.alpha)
                    
                    # set agents' current positions relative to a target agent (agent 0) (should ideally set this to ego-vehicle when testing)
                    curr_pos = x_t - x_t.nanmean(1, keepdims=True)#-x_t[:,0].unsqueeze(1)#x_t[:,0].unsqueeze(1)
                    curr_pos = curr_pos.squeeze(2)
                    
                    pred, _ = future_scene_ae(future_disp, curr_pos, agent_types_sample)        

                    pred_abs = rel_to_abs(pred, x_t, alpha=self.alpha)       
                    pred_abs = rotate(pred_abs, x_t, -1 * rot_angles_rad)     

                    # pred_agent_dist = pred_abs - pred_abs[:,0].unsqueeze(1)
                    # future_agent_dist = future_traj_orig_sample - future_traj_orig_sample[:,0].unsqueeze(1)

                    # check where agent is not present in the future
                    # mask = future_traj_orig_sample == [] # TODO check what the placeholder value are
                        
                    # remove nan values from pred
                    mask = torch.isfinite(pred)
                    
                    upper_diag_ids = torch.triu_indices(agent_types.shape[1], agent_types.shape[1])
                    
                    D_true = torch.sqrt(torch.sum((future_disp[:,:,-1,:][:,None,:] - future_disp[:,:,-1,:][:,:,None]) ** 2, dim = -1))
                    D_pred = torch.sqrt(torch.sum((pred[:,:,-1,:][:,None,:] - pred[:,:,-1,:][:,:,None]) ** 2, dim = -1))
                    
                    D_true = D_true[:,upper_diag_ids[0], upper_diag_ids[1]]
                    D_pred = D_pred[:,upper_diag_ids[0], upper_diag_ids[1]]
                    
                    mask_D = torch.isfinite(D_pred)

                    loss_batch = loss_fn(pred[mask], future_disp[mask]) + loss_fn(D_pred[mask_D], D_true[mask_D]) #+ 0.1*loss_fn(pred_agent_dist, future_agent_dist)
                    # Loss.append(loss)
                    
                    # loss_batch = torch.mean(torch.stack(Loss))

                    optim.zero_grad()
                    loss_batch.backward()
                    optim.step()

                    train_loss_ep.append(loss_batch.item())  

                print('Validating...')
                future_scene_ae.eval()
                with torch.no_grad():
                    for _, data_val in enumerate(val_loader):

                        past_pos_val = data_val[0].to(device = self.device)
                        future_pos_val = data_val[1].to(device = self.device)
                        agent_types = data_val[-1].to(device = self.device)

                        # if curr_pos_val_orig.squeeze().size()[0] > 1:
                        future_traj_val_orig = future_pos_val.to(self.device).float()
                        past_traj_val_orig = past_pos_val.to(self.device).float()

                        # Loss_val = []
                        # for i_samples in range(len(past_pos_val)):

                            # shuffle agents to potentially get different target agents for the decoding (should improve generalization) TODO check
                        shuffle_ids_val = torch.randperm(past_traj_val_orig.size()[1])
                        future_traj_val_orig_sample = future_traj_val_orig[:,shuffle_ids_val,:,:]
                        past_traj_val_orig_sample = past_traj_val_orig[:,shuffle_ids_val,:,:]
                        agent_types_sample = agent_types[:,shuffle_ids_val]

                    
                        past_traj_val, future_traj_val, rot_angles_rad_val = normalize_rotation(x=past_traj_val_orig_sample, y_true=future_traj_val_orig_sample)
                        
                        
                        x_t = past_traj_val[...,-1:,:]
                        
                        future_disp_val = abs_to_rel(future_traj_val, x_t, alpha=self.alpha)

                        curr_pos_val = x_t-x_t.nanmean(1, keepdims=True)#x_t[:,0].unsqueeze(1)
                        curr_pos_val = curr_pos_val.squeeze(2)

                        pred_val, _ = future_scene_ae(future_disp_val, curr_pos_val, agent_types_sample)

                        pred_val_abs = rel_to_abs(pred_val, x_t, alpha=self.alpha)       
                        pred_val_abs = rotate(pred_val_abs, x_t, -1 * rot_angles_rad_val)     

                        # pred_val_agent_dist = pred_val_abs - pred_val_abs[:,0].unsqueeze(1)
                        # future_val_agent_dist = future_traj_val_orig_sample - future_traj_val_orig_sample[:,0].unsqueeze(1)
                        
                        mask = torch.isfinite(pred_val)
                        
                        
                        upper_diag_ids = torch.triu_indices(agent_types.shape[1], agent_types.shape[1])
                        D_true_val = torch.sqrt(torch.sum((future_disp_val[:,:,-1,:][:,None,:] - future_disp_val[:,:,-1,:][:,:,None]) ** 2, dim = -1))
                        D_pred_val = torch.sqrt(torch.sum((pred_val[:,:,-1,:][:,None,:] - pred_val[:,:,-1,:][:,:,None]) ** 2, dim = -1))
                        
                        D_true_val = D_true_val[:,upper_diag_ids[0], upper_diag_ids[1]]
                        D_pred_val = D_pred_val[:,upper_diag_ids[0], upper_diag_ids[1]]
                        mask_D = torch.isfinite(D_pred_val)
                        
                        loss_val_batch = loss_fn(pred_val[mask], future_disp_val[mask]) + loss_fn(D_pred_val[mask_D], D_true_val[mask_D]) #+ 0.1*loss_fn(pred_val_agent_dist, future_val_agent_dist)
                        # Loss_val.append(loss_val)
                        # # loss_val = loss_fn(pred_graph_val, y_in_val)
                        # loss_val_batch = torch.mean(torch.stack(Loss_val))

                        val_loss_ep.append(loss_val_batch.item())

                train_loss.append(np.mean(train_loss_ep))
                val_loss.append(np.mean(val_loss_ep))
                
                # Check for convergence
                if epoch > 100:
                    best_val_step = np.argmin(val_loss)
                    if epoch - best_val_step > 100:
                        converged = True

                print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, self.fut_ae_epochs, train_loss[-1], val_loss[-1]))
                
                if converged:
                    print('Model is converged.')
                    break
                
            os.makedirs(os.path.dirname(fut_model_file), exist_ok=True)
            pickle.dump(future_scene_ae, open(fut_model_file, 'wb'))

        return future_scene_ae


    def train_flow(self, fut_model, train_loader, val_loader, T_all):
        steps = self.flow_epochs

        beta_noise = 0 
        gamma_noise = 0 

        self.t_unique = torch.unique(torch.from_numpy(T_all).to(self.device))
        self.t_unique = self.t_unique[self.t_unique != 48]

        if self.use_map:
            gnn_in_dim = self.obs_encoding_size + self.map_encoding_size
        else:
            gnn_in_dim = self.obs_encoding_size

        socialGNNparams = {'num_layers': 4, 'emb_dim': 32, 'in_dim': gnn_in_dim, 'edge_dim': len(self.t_unique) * 2 + 1}
        envCNNparams = {'img_dim':self.img_dim , 'nin': 1, 'enc_dim':self.map_encoding_size}
        
        # TODO: Set the gnn parameters
        flow_dist = InteractiveFlow(pred_steps=self.ft_enc_dim, alpha=self.alpha, beta=beta_noise, gamma=gamma_noise, 
                               norm_rotation=self.norm_rotation, device=self.device,
                               obs_encoding_size=self.obs_encoding_size, n_layers_rnn=self.n_layers_rnn, 
                               es_rnn=self.hs_rnn, hs_rnn=self.hs_rnn, use_map=self.use_map, 
                               envCNNparams=envCNNparams,
                               socialGNNparams=socialGNNparams, T_all = T_all)
        
        for param in fut_model.parameters():
            param.requires_grad = False 
            param.grad = None
            
        
        flow_dist_file = self.model_file[:-16] + '--InteFlow_M_NF'
        
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
                    
                    past_pos = data[0].to(device = self.device)
                    future_pos = data[1].to(device = self.device)
                    agent_types = data[-1].to(device = self.device)
                    
                    
                    optimizer.zero_grad()
                    
                    past_data = past_pos.float()
                    future_data = future_pos.float()
                    
                    past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data, future_data)
                    
                    # x_t = past_traj[:,[0],-1:,:]
                    x_t = past_traj[...,-1:,:]
                    y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                    # set agents' current positions relative to a target agent (agent 0) (should ideally set this to ego-vehicle when testing)
                    curr_pos = x_t-x_t.nanmean(1, keepdims=True)#x_t[:,0].unsqueeze(1)
                    curr_pos = curr_pos.squeeze(2)

                    if self.use_map:
                        img = data[2].float().to(device = self.device)
                        # img.shape = (batch_sz, num_agents, target_height, target_width, channels)
                        img = img.permute(0,1,4,2,3)


                    target_length = y_rel.size(dim=2)
                    #should be fine
                    num_agents = y_rel.size(dim=1)
                    batch_size = y_rel.size(dim=0)

                    agentTrajs_flattened = y_rel.reshape(-1, target_length, 2) # (n_agents * batch_size, seq_len, 2)

                    agentFutureTrajEnc_flattened = torch.zeros((agentTrajs_flattened.shape[0], 
                                                                fut_model.traj_encoder[str(int(self.t_unique[0].detach().cpu().numpy().astype(int)))].rnn.hs),
                                                                device = self.device)
                    
                    T_flattened = agent_types.reshape(-1)
                    for t in self.t_unique:
                        # assert t in T_flattened
                        t_in = T_flattened == t
                        
                        t_key = str(int(t.detach().cpu().numpy().astype(int)))
                        agentFutureTrajEnc_flattened[t_in] = fut_model.traj_encoder[t_key](agentTrajs_flattened[t_in])

                    # agentFutureTrajEnc_flattened = self.traj_encoder(agentTrajs_flattened) # (n_agents * batch_size, seq_len, hs)
                    agentFutureTrajEnc = agentFutureTrajEnc_flattened.reshape(batch_size, num_agents, -1) # (batch_size, n_agents, 1, hs)

                    out = fut_model.scene_encoder(curr_pos, agentFutureTrajEnc, agent_types) # (batch_size, enc_dim)

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
                        
                        # x_t = past_traj[:,[0],-1:,:]
                        x_t = past_traj[...,-1:,:]
                        y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                        # set agents' current positions relative to a target agent (agent 0) (should ideally set this to ego-vehicle when testing)
                        curr_pos = x_t-x_t.nanmean(1, keepdims=True)#x_t[:,0].unsqueeze(1)
                        curr_pos = curr_pos.squeeze(2)
                        
                        if self.use_map:
                            img_val = val[2].float().to(device = self.device)
                            # img.shape = (batch_sz, num_agents, target_height, target_width, channels)
                            img_val = img_val.permute(0,1,4,2,3)



                        target_length = y_rel.size(dim=2)
                        #should be fine
                        num_agents = y_rel.size(dim=1)
                        batch_size = y_rel.size(dim=0)

                        agentTrajs_flattened = y_rel.reshape(-1, target_length, 2) # (n_agents * batch_size, seq_len, 2)

                        agentFutureTrajEnc_flattened = torch.zeros((agentTrajs_flattened.shape[0], 
                                                                    fut_model.traj_encoder[str(int(self.t_unique[0].detach().cpu().numpy().astype(int)))].rnn.hs),
                                                                    device = self.device)
                        
                        T_flattened = agent_types_val.reshape(-1)
                        for t in self.t_unique:
                            # assert t in T_flattened
                            t_in = T_flattened == t
                            
                            t_key = str(int(t.detach().cpu().numpy().astype(int)))
                            agentFutureTrajEnc_flattened[t_in] = fut_model.traj_encoder[t_key](agentTrajs_flattened[t_in])

                        agentFutureTrajEnc = agentFutureTrajEnc_flattened.reshape(batch_size, num_agents, -1) # (batch_size, n_agents, 1, hs)

                        out = fut_model.scene_encoder(curr_pos, agentFutureTrajEnc, agent_types_val) # (batch_size, enc_dim)
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

        train_loader, val_loader, T = self.extract_data(train = True)
        self.fut_model = self.train_futureAE(train_loader, val_loader, T_all=T)

        self.flow_dist = self.train_flow(self.fut_model, train_loader, val_loader, T_all=T)
        
        # save weigths 
        # after checking here, please return num_epochs to 100 and batch size to 
        self.weights_saved = []
        
        
    def load_method(self):
        fut_model_file = self.model_file[:-16] + '--InteFlow_M_AE'
        flow_dist_file = self.model_file[:-16] + '--InteFlow_M_NF'
        self.fut_model = pickle.load(open(fut_model_file, 'rb'))
        self.flow_dist = pickle.load(open(flow_dist_file, 'rb'))
        
    def _repeat_rowwise(self, x, n):
        org_dim = x.size(-1)
        x = x.repeat(1, 1, n)
        return x.view(-1, n, org_dim)
                
    def predict_batch(self, models, test_loader, target_length, batch_sz, T_all):
        flow_dist = models[1]
        fut_model = models[0]

        
        for _, sample_batched in enumerate(test_loader):
            
            past=sample_batched[0]
            past=past.float().to(device = self.device)
            agent_types = sample_batched[-1].to(device = self.device)
            
            past_traj, rot_angles_rad = flow_dist._normalize_rotation(past)
            
            
            if self.use_map:
                img = sample_batched[1].float().to(device=self.device)
                # img.shape = (batch_sz, num_agents, target_height, target_width, channels)
                img = img.permute(0,1,4,2,3)

            else:
                img = None
            
            t_unique = torch.unique(torch.from_numpy(T_all))
            t_unique = t_unique[t_unique != 48]

            num_agents = past.size(dim=1)
            batch_size = past.size(dim=0)
            x_t = past_traj[...,-1:,:]
            x_t = x_t.repeat_interleave(self.num_samples_path_pred, dim=0)
            T = agent_types.repeat_interleave(self.num_samples_path_pred, dim=0)
            
            rot_angles_rad = rot_angles_rad.repeat_interleave(self.num_samples_path_pred, dim=0)

            if self.use_map: 
                samples_rel, log_probs = flow_dist.sample(self.num_samples_path_pred, past.float(), agent_types, img)
            else:
                samples_rel, log_probs = flow_dist.sample(self.num_samples_path_pred, past.float(), agent_types)
            
            samples_rel = samples_rel.squeeze(0)
                    
            agentPos = x_t-x_t.nanmean(1, keepdims=True)#x_t[:,0].unsqueeze(1)
            agentPos = agentPos.squeeze(2)

            pos_emb = F.tanh(fut_model.pos_emb(agentPos)) # (n_agents, enc_dim)
            
            existing_agent = T != 48 # (batch_size, max_num_agents)
            num_existing_agents = existing_agent.sum(axis=1)
            numAgents_emb = F.tanh(fut_model.numAgents_emb(torch.tensor(num_existing_agents).float().to(self.device).unsqueeze(1))) # (n_agents, 1)
            
            # numAgents_emb = F.tanh(fut_model.numAgents_emb(torch.tensor(num_agents).float().to(self.device).unsqueeze(0))) # (1, 1)
            # numAgents_emb = numAgents_emb.repeat(batch_size*self.num_samples_path_pred, 1) # (n_agents, 1)
            
            graphDecoding, existing_agents = fut_model.scene_decoder(agentPos, samples_rel, pos_emb, numAgents_emb, num_agents, T)

            agentFutureTrajDec = torch.zeros((graphDecoding.shape[0], target_length, 2),
                                                    device = self.device)
        
            T_flattened = T.reshape(-1)
            T_flattened = T_flattened[T_flattened != 48]
            for t in t_unique:
                # assert t in T_flattened
                t_in = T_flattened == t
                
                t_key = str(int(t.detach().cpu().numpy().astype(int)))
                agentFutureTrajDec[t_in] = fut_model.traj_decoder[t_key](graphDecoding[t_in], target_length=target_length, batch_size=len(graphDecoding[t_in]))

            # Needed for batch training
            tmp = torch.zeros((batch_size*self.num_samples_path_pred, num_agents, target_length, 2), device = self.device)
            tmp[tmp == 0] = float('nan')
            existing_sample, existing_row = torch.where(existing_agents)

            tmp[existing_sample, existing_row] = agentFutureTrajDec

            agentFutureTrajDec = tmp
            torch.cuda.empty_cache() 
            
            y_hat = flow_dist._rel_to_abs(agentFutureTrajDec, x_t)

            # invert rotation normalization
            y_hat = flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad)#.unsqueeze(1))

            #prediction.shape = (batch_sz, num_agents, self.num_samples_path_pred, target_length, 2)   
            # y_hat = y_hat.reshape(batch_sz, num_agents, self.num_samples_path_pred, target_length, 2)
            tmp = y_hat.reshape(batch_size, self.num_samples_path_pred, num_agents, target_length, 2)
            tmp = tmp.transpose(2,1)
            y_hat = tmp
            
            # y_hat = y_hat.reshape(batch_size, self.num_samples_path_pred, num_agents, target_length, 2)
            # y_hat = y_hat.transpose(2,1)
            
            Y_pred = y_hat.detach()
                
            log_probs = log_probs.reshape(batch_size, self.num_samples_path_pred)
            log_probs = log_probs.detach()
            log_probs[torch.isnan(log_probs)] = -1000
            prob = torch.exp(log_probs)#[exp(x) for x in log_probs]
            prob = torch.tensor(prob)
        
            torch.cuda.empty_cache() 
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
        
        
        Path_names = np.array([name for name in self.Output_path_train.columns])
        
        # TODO keep in mind since len(test_loader.dataset) might cause issues
        samples_all = int(len(X)) 

        Output_Path = pd.DataFrame(np.empty((samples_all, Pred_agents.sum()), object), 
                                   columns = Path_names[Pred_agents].reshape(-1))
        
        nums = np.unique(self.num_timesteps_out_test)
        
        
        samples_done = 0
        calculations_done = 0
        calculations_all = np.sum(self.num_timesteps_out_test)

        self.batch_size = 32

        for num in nums:
            Index_num = np.where(self.num_timesteps_out_test == num)[0]
            
            if self.batch_size > len(Index_num):
                Index_uses = [Index_num]
            else:
                Index_uses = [Index_num[i * self.batch_size : (i + 1) * self.batch_size] 
                              for i in range(int(np.ceil(len(Index_num)/ self.batch_size)))] 
            
            for Index_use in Index_uses:
                
                if self.use_map:
                    
                    my_dataset = TensorDataset(torch.tensor(X[Index_use]).to(device=self.device),
                                                torch.tensor(img[Index_use]),
                                                torch.tensor(T[Index_use]).to(device=self.device)) # create your datset
                else:
                    
                    my_dataset = TensorDataset(torch.tensor(X[Index_use]).to(device=self.device),
                                                torch.tensor(T[Index_use]).to(device=self.device)) # create your datset
                
                test_loader = DataLoader(my_dataset, batch_size=len(Index_use)) # create your dataloader

                
                # Run prediction pass
                with torch.no_grad(): # Do not build graph for backprop
                    predictions, predictions_prob = self.predict_batch([self.fut_model, self.flow_dist], test_loader, num, len(Index_use), T_all=T)

                #prediction.shape = (batch_sz, num_agents, self.num_samples_path_pred, target_length, 2)                
                Pred = predictions.detach().cpu().numpy()

                if len(Pred.shape) == 4:
                    Pred = Pred[np.newaxis]
                
                Pred[PPed_agents[Index_use]]  *= self.std_pos_ped
                Pred[~PPed_agents[Index_use]] *= self.std_pos_veh
                
                
                torch.cuda.empty_cache()
                
                for i, i_sample in enumerate(Index_use):
                    traj = Pred[i, :, :, :]
                    for index in Path_names:
                        j = np.where(index==Agents)[0][0] 
                        if Pred_agents[j]:
                            Output_Path.iloc[i_sample][index] = traj[j,:,:,:].astype('float32')
                        
                        
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
        names = {'print': 'InteFlow',
                 'file': 'InteFlow_M',
                 'latex': r'\emph{IF}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True
    
    def provides_epoch_loss(self = None):
        return True
