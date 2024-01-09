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


        self.ft_enc_dim = 8
        self.pos_emb_dim = 2
        self.fre_nin = 2
        self.fre_nl = 3
        self.fsg_nl = 3
        self.fs_enc_dim = 8
        self.frd_nout = 2
        self.frd_nl = 3


        self.obs_encoding_size = 16 
        self.scene_encoding_size = 4

        # TODO dependent on dataset
        if self.data_set.get_name()['file'][:3] == 'ETH':
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

        self.map_encoding_size = 8
        self.fut_ae_epochs = 10000
        self.fut_ae_lr = 1e-3
        self.fut_ae_wd = 1e-3

        self.flow_epochs = 200
        self.flow_lr = 1e-3
        self.flow_wd = 1e-5

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


    def train_futureAE(self, T_all):

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
        if os.path.isfile(fut_model_file) and not self.model_overwrite:
            future_scene_ae = pickle.load(open(fut_model_file, 'rb'))
            print('Future AE model loaded')

        else:

            loss_fn = torch.nn.MSELoss() 
            optim = torch.optim.AdamW(future_scene_ae.parameters(), lr=self.fut_ae_lr, weight_decay=self.fut_ae_wd)

            train_loss = []
            val_loss = []

            converged = False
            for epoch in range(self.fut_ae_epochs):
                future_scene_ae.train()
                train_loss_ep = []
                val_loss_ep = []

                train_epoch_done = False
                print('Training Future Scene AE...')
                while not train_epoch_done:
                    X, Y, T, img, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                           val_split_size = 0.1)
                    X, T, Y, _ = self.extract_batch_data(X, T, Y)

                    past_pos = X
                    future_pos = Y
                    agent_types = T

                    future_traj_orig = future_pos.to(self.device).float()
                    past_traj_orig = past_pos.to(self.device).float()
                    
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
                    
                    upper_diag_ids = torch.triu_indices(agent_types.shape[1], agent_types.shape[1], offset=1)
                    
                    D_true = torch.sqrt(torch.sum((future_disp[:,:,-1,:][:,None,:] - future_disp[:,:,-1,:][:,:,None]) ** 2, dim = -1)+1e-6)
                    D_pred = torch.sqrt(torch.sum((pred[:,:,-1,:][:,None,:] - pred[:,:,-1,:][:,:,None]) ** 2, dim = -1)+1e-6)
                    
                    D_true = D_true[:,upper_diag_ids[0], upper_diag_ids[1]]
                    D_pred = D_pred[:,upper_diag_ids[0], upper_diag_ids[1]]
                    
                    mask_D = torch.isfinite(D_pred)

                    loss_batch = loss_fn(pred[mask], future_disp[mask]) #+ 0.1*loss_fn(D_pred[mask_D], D_true[mask_D]) #+ 0.1*loss_fn(pred_agent_dist, future_agent_dist)
                    # Loss.append(loss)
                    
                    # loss_batch = torch.mean(torch.stack(Loss))

                    optim.zero_grad()
                    loss_batch.backward(retain_graph=True)


                    # # Set the maximum norm value to 1.0
                    # max_norm = 1.0

                    # # Calculate the norm of the gradients
                    # grad_norm = torch.nn.utils.clip_grad_norm_(future_scene_ae.parameters(), max_norm)

                    optim.step()

                    train_loss_ep.append(loss_batch.item())  

                print('Validating...')
                future_scene_ae.eval()
                with torch.no_grad():

                    val_epoch_done = False
                    while not val_epoch_done:
                        X, Y, T, _, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                val_split_size = 0.1)
                        
                        X, T, Y, _ = self.extract_batch_data(X, T, Y)

                        past_pos_val = X
                        future_pos_val = Y
                        agent_types = T

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
                
            self.train_loss[0, :len(val_loss)] = np.array(val_loss)
            os.makedirs(os.path.dirname(fut_model_file), exist_ok=True)
            pickle.dump(future_scene_ae, open(fut_model_file, 'wb'))

        return future_scene_ae


    def train_flow(self, fut_model, T_all):
        use_map = self.can_use_map and self.has_map
        steps = self.flow_epochs

        beta_noise = 0.002
        gamma_noise = 0.002 

        self.t_unique = torch.unique(torch.from_numpy(T_all).to(self.device))
        self.t_unique = self.t_unique[self.t_unique != 48]

        if use_map:
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
        
        if os.path.isfile(flow_dist_file) and not self.model_overwrite:
            flow_dist = pickle.load(open(flow_dist_file, 'rb'))
                          
        else:
            optimizer = torch.optim.AdamW(flow_dist.parameters(), lr=self.flow_lr, weight_decay=self.flow_wd)

            val_losses = []


            for step in range(steps):

                flow_dist.train()
                fut_model.eval()
                
                losses_epoch = []
                val_losses_epoch = []
                
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, T, img, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.batch_size, 
                                                                                           val_split_size = 0.1)
                    X, T, Y, img = self.extract_batch_data(X, T, Y, img)

                    
                    past_pos = X
                    future_pos = Y
                    agent_types = T
                    
                    
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

                    if img is not None:
                        img = img[:,0].permute(0,3,1,2)


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
                    
                    if img is not None:
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
                    val_epoch_done = False
                    while not val_epoch_done:
                        X, Y, T, img, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.batch_size, 
                                                                                                val_split_size = 0.1)
                        X, T, Y, img = self.extract_batch_data(X, T, Y, img)

                        past_data_val = X
                        future_data_val = Y
                        agent_types_val = T
                        
                        past_traj, fut_traj, rot_angles_rad = flow_dist._normalize_rotation(past_data_val, future_data_val)
                        
                        # x_t = past_traj[:,[0],-1:,:]
                        x_t = past_traj[...,-1:,:]
                        y_rel = flow_dist._abs_to_rel(fut_traj, x_t)

                        # set agents' current positions relative to a target agent (agent 0) (should ideally set this to ego-vehicle when testing)
                        curr_pos = x_t-x_t.nanmean(1, keepdims=True)#x_t[:,0].unsqueeze(1)
                        curr_pos = curr_pos.squeeze(2)
                        
                        
                        if img is not None:
                            img_val = img[:,0].permute(0,3,1,2)


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

            self.train_loss[1, :len(val_losses)] = np.array(val_losses)
            os.makedirs(os.path.dirname(flow_dist_file), exist_ok=True)
            pickle.dump(flow_dist, open(flow_dist_file, 'wb'))

        return flow_dist


    def train_method(self):    
        self.train_loss = np.ones((2, max(self.fut_ae_epochs, self.flow_epochs))) * np.nan

        # Get needed agent types
        T_all = self.provide_all_included_agent_types().astype(str)
        T_all = np.fromstring(T_all, dtype = np.uint32).reshape(len(T_all), int(str(T_all.astype(str).dtype)[2:])).astype(np.uint8)[:,0]

        # Prepare stuff for Normalization
        if self.data_set.get_name()['file'] == 'Fork_P_Aug':
            X, Y, _, _, _, _, _, _ = self.provide_all_training_trajectories()
            traj_tar = np.concatenate((X[:,0], Y[:,0]), axis = 1)
            self.max_pos = np.max(traj_tar)
            self.min_pos = np.min(traj_tar)
        else:
            self.min_pos = 0.0
            self.max_pos = 1.0

        self.fut_model = self.train_futureAE(T_all)
        self.flow_dist = self.train_flow(self.fut_model, T_all)
        
        # save weigths 
        # after checking here, please return num_epochs to 100 and batch size to 
        self.weights_saved = [self.min_pos, self.max_pos]
        
        
    def load_method(self):
        self.min_pos, self.max_pos = self.weights_saved
        
        fut_model_file = self.model_file[:-16] + '--InteFlow_M_AE'
        flow_dist_file = self.model_file[:-16] + '--InteFlow_M_NF'
        self.fut_model = pickle.load(open(fut_model_file, 'rb'))
        self.flow_dist = pickle.load(open(flow_dist_file, 'rb'))
        
    def _repeat_rowwise(self, x, n):
        org_dim = x.size(-1)
        x = x.repeat(1, 1, n)
        return x.view(-1, n, org_dim)
                
    def predict_method(self, T_all):
        prediction_done = False
        
        while not prediction_done:
            X, T, img, _, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.batch_size)
            Ped_agent = T == 'P'
            
            X, T, _, img = self.extract_batch_data(X, T, img = img)

            # Run prediction pass
            with torch.no_grad():
                past = X
                agent_types = T
            
                past_traj, rot_angles_rad = self.flow_dist._normalize_rotation(past)

                if img is not None:
                    img = img[:,0].permute(0,3,1,2)
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

                if img is not None: 
                    samples_rel, log_probs = self.flow_dist.sample(self.num_samples_path_pred, past.float(), agent_types, img)
                else:
                    samples_rel, log_probs = self.flow_dist.sample(self.num_samples_path_pred, past.float(), agent_types)
                
                samples_rel = samples_rel.squeeze(0)
                        
                agentPos = x_t-x_t.nanmean(1, keepdims=True)#x_t[:,0].unsqueeze(1)
                agentPos = agentPos.squeeze(2)

                existing_agent = T != 48 # (batch_size, max_num_agents)
                exist_sample2, exist_row2 = torch.where(existing_agent)

                agentPos_existing = agentPos[exist_sample2, exist_row2] # (num_existing_agents, 2)

                pos_emb = F.tanh(self.fut_model.pos_emb(agentPos_existing)) # (n_agents, enc_dim)
                
                tmp = torch.zeros((batch_size, num_agents, 2), device = self.device)
                tmp[tmp == 0] = float('nan')
                tmp[exist_sample2, exist_row2] = pos_emb

                pos_emb = tmp

                num_existing_agents = existing_agent.sum(axis=1)
                numAgents_emb = F.tanh(self.fut_model.numAgents_emb(torch.tensor(num_existing_agents).float().to(self.device).unsqueeze(1))) # (n_agents, 1)
                
                # numAgents_emb = F.tanh(fut_model.numAgents_emb(torch.tensor(num_agents).float().to(self.device).unsqueeze(0))) # (1, 1)
                # numAgents_emb = numAgents_emb.repeat(batch_size*self.num_samples_path_pred, 1) # (n_agents, 1)
                
                graphDecoding, existing_agents = self.fut_model.scene_decoder(agentPos, samples_rel, pos_emb, numAgents_emb, num_agents, T)

                agentFutureTrajDec = torch.zeros((graphDecoding.shape[0], num_steps, 2),
                                                        device = self.device)
            
                T_flattened = T.reshape(-1)
                T_flattened = T_flattened[T_flattened != 48]
                for t in t_unique:
                    # assert t in T_flattened
                    t_in = T_flattened == t
                    
                    t_key = str(int(t.detach().cpu().numpy().astype(int)))
                    agentFutureTrajDec[t_in] = self.fut_model.traj_decoder[t_key](graphDecoding[t_in], target_length=num_steps, batch_size=len(graphDecoding[t_in]))

                # Needed for batch training
                tmp = torch.zeros((batch_size*self.num_samples_path_pred, num_agents, num_steps, 2), device = self.device)
                tmp[tmp == 0] = float('nan')
                existing_sample, existing_row = torch.where(existing_agents)

                tmp[existing_sample, existing_row] = agentFutureTrajDec

                agentFutureTrajDec = tmp
                torch.cuda.empty_cache() 
                
                y_hat = self.flow_dist._rel_to_abs(agentFutureTrajDec, x_t)

                # invert rotation normalization
                y_hat = self.flow_dist._rotate(y_hat, x_t, -1 * rot_angles_rad)#.unsqueeze(1))

                #prediction.shape = (batch_sz, num_agents, self.num_samples_path_pred, target_length, 2)   
                # y_hat = y_hat.reshape(batch_sz, num_agents, self.num_samples_path_pred, target_length, 2)
                tmp = y_hat.reshape(batch_size, self.num_samples_path_pred, num_agents, num_steps, 2)
                tmp = tmp.transpose(2,1)
                y_hat = tmp
                
                # y_hat = y_hat.reshape(batch_size, self.num_samples_path_pred, num_agents, target_length, 2)
                # y_hat = y_hat.transpose(2,1)
                
                Y_pred = y_hat.detach()
                    
            Pred = Y_pred.detach().cpu().numpy()
            if len(Pred.shape) == 3:
                Pred = Pred[np.newaxis]
            
            Pred[Ped_agent[:,0]]  *= self.std_pos_ped
            Pred[~Ped_agent[:,0]] *= self.std_pos_veh

            if self.data_set.get_name()['file'] == 'Fork_P_Aug':
                Pred = Pred * (self.max_pos - self.min_pos) + self.min_pos

            torch.cuda.empty_cache() 
        
            # save predictions
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id)

    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
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
