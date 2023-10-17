from trajflow_meszaros import trajflow_meszaros
import numpy as np
import torch
import random
import scipy
from TrajFlow.flowModels import TrajFlow_I, Future_Encoder, Future_Decoder, Future_Seq2Seq, Scene_Encoder
import pickle
import os

class trajflow_meszaros_weighting(trajflow_meszaros):
    '''
    TrajFlow is a single agent prediction model that combine Normalizing Flows with
    GRU-based autoencoders.
    
    The model was implemented into the framework by its original creators, and 
    the model was first published under:
        
    Mészáros, A., Alonso-Mora, J., & Kober, J. (2023). Trajflow: Learning the 
    distribution over trajectories. arXiv preprint arXiv:2304.05166.
    '''
    
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
        if os.path.isfile(fut_model_file) and not self.model_overwrite:
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
                    
                    future_traj_hat, y_in = fut_model(y_rel)   

                    # add weighting to the last timestep
                    future_traj_hat_weighted = future_traj_hat.cpu()
                    future_traj_hat_weighted[...,-1:,:] = future_traj_hat_weighted[...,-1:,:]*2
                    future_traj_hat_weighted = future_traj_hat_weighted.to(self.device)

                    y_in_weighted = y_in.cpu()
                    y_in_weighted[...,-1:,:] = y_in_weighted[...,-1:,:]*2
                    y_in_weighted = y_in_weighted.to(self.device)
                        
                    optim.zero_grad()
                    loss = torch.sqrt(loss_fn(future_traj_hat_weighted, y_in_weighted))
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

        model_str = 'TF_W_' + kwargs_str
        
        names = {'print': model_str,
                'file': model_str,
                'latex': r'\emph{%s}' % model_str
                }
        return names