import numpy as np
import pickle
import os
import random
import torch
import torch.nn as nn

from model_template import model_template
from PECNet.social_utils import *
from PECNet.models import *

class pecnet_mangalam(model_template):

    
    def setup_method(self, seed = 0):        
        # set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.hyper_params = {}
        self.hyper_params["adl_reg"] = 1
        self.hyper_params["data_scale"] = 1.86
        self.hyper_params["dataset_type"] = 'image'
        self.hyper_params["dec_size"] = [1024, 512, 1024]
        self.hyper_params['dist_thresh'] = 100
        self.hyper_params['enc_dest_size'] = [8, 16]
        self.hyper_params['enc_latent_size'] = [8, 50]
        self.hyper_params['enc_past_size'] = [512, 256]
        self.hyper_params['non_local_theta_size'] = [256, 128, 64]
        self.hyper_params['non_local_phi_size'] = [256, 128, 64]
        self.hyper_params['non_local_g_size'] = [256, 128, 64]
        self.hyper_params['non_local_dim'] = 128
        self.hyper_params['fdim'] = 16
        self.hyper_params['future_length'] = self.num_timesteps_out
        self.hyper_params['gpu_index'] = 0
        self.hyper_params['kld_reg'] = 1
        self.hyper_params['learning_rate'] = 0.0003
        self.hyper_params['mu'] = 0
        self.hyper_params['n_values'] = 20
        self.hyper_params['nonlocal_pools'] = 3
        self.hyper_params['normalize_type'] = 'shift_origin'
        self.hyper_params['num_epochs'] = 650
        self.hyper_params['num_workers'] = 0
        self.hyper_params['past_length'] = self.num_timesteps_in
        self.hyper_params['predictor_hidden_size'] = [1024, 512, 256]
        self.hyper_params['sigma'] = 1.3
        self.hyper_params['test_b_size'] = 128 # changed from orig 4096
        self.hyper_params['time_thresh'] = 0
        self.hyper_params['train_b_size'] = 128 # changed from orig 512
        self.hyper_params['zdim'] = 16

        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = False
        self.can_use_map = False

        self.verbose = True

    def train_PECNet(self):
        model_file = self.model_file[:-4]

        train_loss = []
        val_loss = []
        total_rcl, total_kld, total_adl = 0, 0, 0

        model = PECNet(self.hyper_params["enc_past_size"], self.hyper_params["enc_dest_size"], 
                       self.hyper_params["enc_latent_size"], self.hyper_params["dec_size"], 
                       self.hyper_params["predictor_hidden_size"], self.hyper_params['non_local_theta_size'], 
                       self.hyper_params['non_local_phi_size'], self.hyper_params['non_local_g_size'], 
                       self.hyper_params["fdim"], self.hyper_params["zdim"], self.hyper_params["nonlocal_pools"], 
                       self.hyper_params['non_local_dim'], self.hyper_params["sigma"], 
                       self.hyper_params["past_length"], self.hyper_params["future_length"], self.verbose)
        model = model.double().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=  self.hyper_params["learning_rate"])
        criterion = nn.MSELoss()


        if os.path.isfile(model_file) and not self.model_overwrite:
            model = pickle.load(open(model_file, 'rb'))
                          
        else:
            for step in range(self.hyper_params['num_epochs']):
                train_epoch_done = False
                model.train()

                train_mu = torch.zeros(0).to(self.device, dtype=torch.double)
                train_var = torch.zeros(0).to(self.device, dtype=torch.double)
            
                t_loss = 0
                while not train_epoch_done:
                    X, Y, _, _, _, _, num_steps, train_epoch_done = self.provide_batch_data('train', self.hyper_params['train_b_size'], 
                                                                                            val_split_size = 0.1)

                    x = torch.DoubleTensor(X)
                    y = torch.DoubleTensor(Y)    

                    # mask is nothing other than an indicator of which agents belong to the same scene row is agent
                    # 1 in each column of the row means that off diagonal agents belong to same scene as diagonal agent
                    # 
                    # NOTE to self, filter out nan agents when flattening
                    # 
                    agent_exists = torch.isfinite(x).any(dim = 3).any(dim = 2)
                    num_agents_exist = agent_exists.sum(1)            

                    x = x[agent_exists]
                    y = y[agent_exists]

                    scene = torch.repeat_interleave(torch.arange(agent_exists.shape[0]), num_agents_exist)
                    mask = scene[None] == scene[:,None]       
                    mask = mask.to(device = self.device, dtype = torch.double) 

                    traj = torch.concat((x, y), dim = 1)
                    traj -= traj[:, :1, :]
                    traj *= self.hyper_params["data_scale"]   

                    x = traj[:, :self.hyper_params["past_length"], :]
                    y = traj[:, self.hyper_params["past_length"]:, :]                                    

                    initial_pos = x[:,-1,:].clone().detach()/1000
                    initial_pos = initial_pos.to(self.device)
                    x = x.contiguous().view(-1, x.shape[1]*x.shape[2]).to(self.device)
                    dest = y[:,-1,:].contiguous().to(self.device)
                    future = y[:,:-1,:].contiguous().view(y.size(0),-1).to(self.device)


                    dest_recon, mu, var, interpolated_future = model.forward(x, initial_pos,
                                                                            dest=dest, mask=mask,
                                                                            device=self.device)

                    optimizer.zero_grad()
                    rcl, kld, adl = calculate_loss(dest, dest_recon, mu, var, criterion, future, interpolated_future)
                    loss = rcl + kld*self.hyper_params["kld_reg"] + adl*self.hyper_params["adl_reg"]
                    loss.backward()

                    t_loss += loss.item()
                    total_rcl += rcl.item()
                    total_kld += kld.item()
                    total_adl += adl.item()
                    optimizer.step()
                    train_mu = torch.concat((train_mu, dest_recon[:,1]))
                    # train_var = torch.concat((train_var, var.mul(0.5).exp_()))

                train_loss.append(t_loss)
                

                val_epoch_done = False
                model.eval()
            
                v_loss = 0
                num_val_samples = 0
                while not val_epoch_done:
                    X, Y, _, _, _, _, num_steps, val_epoch_done = self.provide_batch_data('val', self.hyper_params['train_b_size'], 
                                                                                            val_split_size = 0.1)

                    x = torch.DoubleTensor(X)
                    y = torch.DoubleTensor(Y)    

                    # mask is nothing other than an indicator of which agents belong to the same scene row is agent
                    # 1 in each column of the row means that off diagonal agents belong to same scene as diagonal agent
                    # 
                    # NOTE to self, filter out nan agents when flattening
                    # 
                    agent_exists = torch.isfinite(x).any(dim = 3).any(dim = 2)
                    num_agents_exist = agent_exists.sum(1)            

                    x = x[agent_exists]
                    y = y[agent_exists]

                    scene = torch.repeat_interleave(torch.arange(agent_exists.shape[0]), num_agents_exist)
                    mask = scene[None] == scene[:,None]       
                    mask = mask.to(device = self.device, dtype = torch.double) 

                    traj = torch.concat((x, y), dim = 1)
                    traj -= traj[:, :1, :]
                    traj *= self.hyper_params["data_scale"]   

                    x = traj[:, :self.hyper_params["past_length"], :]
                    y = traj[:, self.hyper_params["past_length"]:, :]                                    

                    initial_pos = x[:,-1,:].clone().detach()/1000
                    initial_pos = initial_pos.to(self.device)
                    x = x.contiguous().view(-1, x.shape[1]*x.shape[2]).to(self.device)
                    dest = y[:,-1,:].contiguous().to(self.device)
                    future = y[:,:-1,:].contiguous().view(y.size(0),-1).to(self.device)

                    all_l2_errors_dest = []
                    all_guesses = []
                    for _ in range(self.hyper_params['n_values']):

                        dest_recon = model.forward(x, initial_pos, device=self.device)
                        dest_recon = dest_recon.cpu().detach().numpy()
                        all_guesses.append(dest_recon)

                        l2error_sample = np.linalg.norm(dest_recon - dest.cpu().detach().numpy(), axis = 1)
                        all_l2_errors_dest.append(l2error_sample)
                    
                    all_l2_errors_dest = np.array(all_l2_errors_dest)
                    all_guesses = np.array(all_guesses)

			        # choosing the best guess
                    indices = np.argmin(all_l2_errors_dest, axis = 0)

                    best_guess_dest = torch.DoubleTensor(all_guesses[indices,np.arange(x.shape[0]),  :]).to(self.device)

                    interpolated_future = model.predict(x, best_guess_dest, mask, initial_pos)
                    predicted_future = torch.concat((interpolated_future, best_guess_dest), dim = 1)
                    predicted_future = predicted_future.reshape((-1, self.hyper_params['future_length'], 2))

                    loss = np.mean(np.linalg.norm(y.numpy() - predicted_future.cpu().detach().numpy(), axis = 2), axis = 1).sum()

                    v_loss += loss
                    num_val_samples += y.shape[0]

                v_loss /= num_val_samples

                print('Epoch: {:7.0f}; \t Train loss: {:7.5f}; \t Val loss: {:7.5f}; \t Mu: {:7.5f}; STD_MU: {:7.5f}'#; \t Var: {:7.5f}; STD_VAR: {:7.5f}'
                      .format(step, np.mean(train_loss[-1]), v_loss, np.mean(train_mu.cpu().detach().numpy()), 
                              np.std(train_mu.cpu().detach().numpy())))#, np.mean(train_var.cpu().detach().numpy()), np.std(train_var.cpu().detach().numpy())))


            self.train_loss[0, :len(train_loss)] = np.array(train_loss)
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            pickle.dump(model, open(model_file, 'wb'))
            pickle.dump(train_mu.cpu().detach().numpy(), open(model_file + '_train_var', 'wb'))
            pickle.dump(train_var.cpu().detach().numpy(), open(model_file + '_train_var', 'wb'))

        return model


    def train_method(self):

        self.train_loss = np.ones((1, self.hyper_params['num_epochs'])) * np.nan
        self.model = self.train_PECNet()
        self.weights_saved = []


    def load_method(self):
        model_file = self.model_file[:-4]
        self.model = pickle.load(open(model_file, 'rb'))


    def predict_method(self):
        self.model.eval()
        assert self.num_samples_path_pred >= 1 and type(self.num_samples_path_pred) == int

        prediction_done = False
        while not prediction_done:
            X, _, _, _, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.hyper_params['test_b_size'])
            x = torch.DoubleTensor(X)

            # mask is nothing other than an indicator of which agents belong to the same scene row is agent
            # 1 in each column of the row means that off diagonal agents belong to same scene as diagonal agent
            # filter out nan agents when flattening

            Pred = np.zeros((X.shape[0], X.shape[1], self.num_samples_path_pred, self.num_timesteps_out, 2))
            
            agent_exists = torch.isfinite(x).any(dim = 3).any(dim = 2)
            num_agents_exist = agent_exists.sum(1)            

            x = x[agent_exists]

            scene = torch.repeat_interleave(torch.arange(agent_exists.shape[0]), num_agents_exist)
            mask = scene[None] == scene[:,None]   
            mask = mask.to(device = self.device, dtype = torch.double)       

            initial_pos = x[:,-1,:].clone().detach()/1000
            initial_pos = initial_pos.to(self.device)

            x0 = x[:, :1, :]

            x -= x0
            x *= self.hyper_params["data_scale"]  
            x = x.contiguous().view(-1, x.shape[1]*x.shape[2]).to(self.device)

            # Note: not taking the best destination guess since in reality this type of info is not present.
            for index in range(self.num_samples_path_pred):
                dest_recon = self.model.forward(x, initial_pos, device=self.device)
                interpolated_future = self.model.predict(x, dest_recon, mask, initial_pos)
                predicted_future = torch.concat((interpolated_future, dest_recon), dim = 1)
                predicted_future = predicted_future.reshape((-1, self.hyper_params['future_length'], 2))
                predicted_future /= self.hyper_params["data_scale"]
                predicted_future += x0.to(self.device)

                Pred[agent_exists.numpy(), index] = predicted_future.cpu().detach().numpy()

            # extrapolate if needed
            if num_steps > Pred.shape[-2]:
                step_delta = Pred[...,-1,:] - Pred[...,-2,:]
                step_delta = step_delta[...,np.newaxis,:]
                
                steps = np.arange(1, num_steps + 1 - Pred.shape[-2])
                steps = steps[np.newaxis,np.newaxis,:,np.newaxis]
                
                Pred_delta = Pred[...,[-1],:] + step_delta * steps
                
                Pred = np.concatenate((Pred, Pred_delta), axis = -2)
                
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)
            
        

    def save_params_in_csv(self = None):
        return False
    

    def provides_epoch_loss(self = None):
        return True
    

    def get_name(self = None):
        names = {'print': 'PECNet',
                    'file': 'PECNet',
                    'latex': r'\emph{PECNet}'}

        return names

    def requires_torch_gpu(self = None):
        return True

    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def check_trainability_method(self):
        return None
