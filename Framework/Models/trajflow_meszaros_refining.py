from trajflow_meszaros import trajflow_meszaros
import numpy as np
import torch
import random
import scipy
from TrajFlow.flowModels import TrajFlow_I, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder
import pickle
import os

class trajflow_meszaros_futEnc12_past04_refining(trajflow_meszaros):
    '''
    TrajFlow is a single agent prediction model that combine Normalizing Flows with
    GRU-based autoencoders.
    
    The model was implemented into the framework by its original creators, and 
    the model was first published under:
        
    Mészáros, A., Alonso-Mora, J., & Kober, J. (2023). Trajflow: Learning the 
    distribution over trajectories. arXiv preprint arXiv:2304.05166.
    '''


    def refine_network(self, fut_model, flow_dist):

        if self.vary_input_length:
            past_length_options = np.arange(0.5, self.num_timesteps_in*self.dt, 0.5)
            sample_past_length = int(np.ceil(np.random.choice(past_length_options)/self.dt))
        else:
            sample_past_length = self.num_timesteps_in
        

        for param in fut_model.parameters():
            param.requires_grad = True 
            
        
        flow_dist_file = self.model_file[:-4] + '_NF'
        fut_model_file = self.model_file[:-4] + '_AE'
             
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
            os.makedirs(os.path.dirname(fut_model_file), exist_ok=True)
            pickle.dump(fut_model, open(fut_model_file, 'wb'))
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
        self.fut_model, self.flow_dist = self.refine_network(fut_model = self.fut_model, flow_dist = self.flow_dist)
        
        # save weigths 
        self.weights_saved = []

    
    def get_name(self = None):

        self.define_default_kwargs()

        kwargs_str = 'fut' + str(self.model_kwargs['fut_enc_sz']) + '_' + \
                     'sc' + str(self.model_kwargs['scene_encoding_size']) + '_' + \
                     'obs' + str(self.model_kwargs['obs_encoding_size']) + '_' + \
                     'alpha' + str(self.model_kwargs['alpha']) + '_' + \
                     'beta' + str(self.model_kwargs['beta_noise']) + '_' + \
                     'gamma' + str(self.model_kwargs['gamma_noise']) + '_' + \
                     'smin' + str(self.model_kwargs['s_min']) + '_' + \
                     'smax' + str(self.model_kwargs['s_max']) + '_' + \
                     'sigma' + str(self.model_kwargs['sigma']) 
                     
        if self.model_kwargs['vary_input_length']:
            kwargs_str += '_varyInLen'

        if self.model_kwargs['scale_AE']:
            kwargs_str += '_sclAE'

        if self.model_kwargs['scale_NF']:
            kwargs_str += '_sclNF'

        model_str = 'TF_R_' + kwargs_str
        
        names = {'print': model_str,
                'file': model_str,
                'latex': r'\emph{%s}' % model_str
                }
        return names