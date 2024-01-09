import numpy as np
import torch 
import torch.nn as nn
import model_template as model_template
import yaml
import torch.optim as optim
from torch.autograd import Variable
import os

from NSP.model_goals import NSP_goals
from NSP.model_nsp_wo import NSP_collisions
from NSP.model_cvae import CVAE
from NSP.utils import new_point, select_para, calculate_loss_cvae

class Neural_Social_Physics(model_template):
    def get_name(self = None):
        names = {'print': 'Neural Social Physics Model',
                 'file': 'NSP_models',
                 'latex': '\emph{NSP}'}
        return names
    
    # Define model interactions with framework
    def requires_torch_gpu(self = None):
        return True
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def check_trainability_method(self):
        # This is a pedestrin model only
        if not (self.provide_all_included_agent_types() == 'P').all():
            return 'the model is only designed to process and predict predestrians.'
        return None
    
    def save_params_in_csv(self = None):
        return False
    
    def provides_epoch_loss(self = None):
        return True
    
    
    # Define model itself
    def setup_method(self):
        # set random seeds
        seed = 0
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Required attributes of the model
        self.min_t_O_train = 2
        self.max_t_O_train = 100
        self.predict_single_agent = True
        self.can_use_map = False
        
        dtype = torch.float32
        torch.set_default_dtype(dtype)
        
        config_path = self.data_set.path + os.sep + 'Models' + os.sep + 'NSP' + os.sep + 'config' + os.sep 
        
        
        # Get goal network
        config_path_goals = config_path + 'sdd_goals.yaml'
        with open(config_path_goals) as file:
            self.params_goal = yaml.load(file, Loader=yaml.FullLoader)
            
        self.model_goal = NSP_goals(self.params_goal["input_size"], self.params_goal["embedding_size"], 
                                    self.params_goal["rnn_size"], self.params_goal["output_size"],  
                                    self.params_goal["enc_dest_state_size"], self.params_goal["dec_tau_size"],
                                    dt = self.dt)
        
        self.model_goal = self.model_goal.float().to(device = self.device)
        
        # Get collisions network
        config_path_collisions = config_path + 'config/sdd_nsp_wo.yaml'
        with open(config_path_collisions) as file:
            self.params_coll = yaml.load(file, Loader=yaml.FullLoader)
            
        self.model_coll = NSP_collisions(self.params_coll["input_size"], self.params_coll["embedding_size"], 
                                         self.params_coll["rnn_size"], self.params_coll["output_size"],  
                                         self.params_coll["enc_size"], self.params_coll["dec_size"],
                                         dt = self.dt)
        
        self.model_coll = self.model_coll.float().to(device = self.device)
        
        # Get CVAE
        config_path_full = config_path + 'sdd_nsp_cvae.yaml'
        with open(config_path_full) as file:
            self.params_full = yaml.load(file, Loader=yaml.FullLoader)
            
        self.model_cvae = CVAE(self.params_full["enc_past_size"], self.params_full["enc_dest_size"], 
                               self.params_full["enc_latent_size"], self.params_full["dec_size"], 
                               self.params_full["fdim"], self.params_full["zdim"], self.params_full["sigma"], 
                               self.num_timesteps_in, self.num_timesteps_out)

        
        
    
    #%% 
    def extract_data_goal_batch(self, X, Y = None, augment = False):  
        if Y is not None:
            Pos = np.concatenate((X[:,0], Y[:,0]), axis = 1)
        else:
            Pos = X[:,0]
        
        Pos = Pos - Pos[:,[0]]
        if augment:
            # rotate by 90, 180, 270 degree
            Pos_090 = np.stack((-Pos[:,:,1], Pos[:,:,0]), -1)
            Pos_180 = np.stack((-Pos[:,:,0], -Pos[:,:,1]), -1)
            Pos_270 = np.stack((Pos[:,:,1], -Pos[:,:,0]), -1)
            
            Pos = np.concatenate([Pos, Pos_090, Pos_180, Pos_270], axis = 0)
            
            # mirror along x-axis
            Pos_mirror = np.stack((-Pos[:,:,0], Pos[:,:,1]), -1)
            
            Pos = np.concatenate([Pos, Pos_mirror], axis = 0)
            
        # calculate velocity
        Vel = (Pos[:,1:] - Pos[:,:-1]) / self.dt
        Vel = np.concatenate((Vel[:,[0]], Vel), axis = 1)
        
        useless_vel = np.linalg.norm(Vel, axis = - 1) < 1e-5
        Vel[useless_vel] = 0.0
        
        S = np.concatenate((Pos, Vel), axis = -1)
        
        S = torch.from_numpy(S).to(device = self.decvice, dtype = torch.float32)
        
        if Y is not None:
            return S[:,:X.shape[2]], S[:,X.shape[2]:]
        else:
            return S
        
    
    def train_loss_goal_batch(self, X0, Y0):
        Y0 = Y0[..., :2].contiguous() #peds*future_length*2
        dest = Y0[:, -1, :]
        
        batch_size = X0.shape[0]
        n_I = X0.shape[1]
        n_O = Y0.shape[1]

        future_vel = (dest - X0[:, - 1, :2]) / (n_O * self.dt) #peds*2
        initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True) #peds
        
        # Encode past behavior
        hidden_states = Variable(torch.zeros(batch_size, self.params_goal['rnn_size']))
        hidden_states = hidden_states.to(device = self.device)
        
        cell_states = Variable(torch.zeros(batch_size, self.params_goal['rnn_size']))
        cell_states = cell_states.to(device = self.device)

        for m in range(1, n_I):
            outputs_features, hidden_states, cell_states = self.model_goal.forward_lstm(X0[:, m], hidden_states, cell_states)
        
        # Predict future behavior
        predictions = torch.zeros(batch_size, n_O, 2).to(device = self.device)
        pred_pos, pred_vel = self.model_goal.forward_next_step(X0[:,-1,:2], X0[:,-1,2:], 
                                                               initial_speeds, dest,
                                                               outputs_features, device = self.device)
        predictions[:, 0, :] = pred_pos

        for t in range(1, n_O):
            input_lstm = torch.cat((pred_pos, pred_vel), dim=1)
            outputs_features, hidden_states, cell_states = self.model_goal.forward_lstm(input_lstm, hidden_states, cell_states)

            future_vel = (dest - pred_pos) / ((n_O - t) * self.dt)  # peds*2
            initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True)  # peds*1

            pred_pos, pred_vel = self.model_goal.forward_next_step(pred_pos, pred_vel, 
                                                                   initial_speeds, dest,
                                                                   outputs_features, device = self.device)
            predictions[:, t, :] = pred_pos
            
        loss = ((predictions - Y0) ** 2).sum(2).sqrt().mean(1)
        
        return loss
    
    
    #%%    
    def extract_data_coll_batch(self, X, Y = None):  
        if Y is not None:
            Pos = np.concatenate((X, Y), axis = 1)
        else:
            Pos = X
        center = X[:,0,0]
        Pos = Pos - center[:,np.newaxis,np.newaxis]
            
        # calculate velocity
        Vel = (Pos[:,1:] - Pos[:,:-1]) / self.dt
        Vel = np.concatenate((Vel[:,[0]], Vel), axis = 1)
        
        useless_vel = np.linalg.norm(Vel, axis = - 1) < 1e-5
        Vel[useless_vel] = 0.0
        
        S = np.concatenate((Pos, Vel), axis = -1)
        
        S = torch.from_numpy(S).to(device = self.decvice, dtype = torch.float32)
        
        supplement = torch.concat([S[:,1:], torch.isfinite(S[:,1:,:,0]).to(dtype = torch.float32).unsqueeze(-1)], dim = -1)
        supplement = torch.permute(0,2,1,3)
        
        supplement_extra = torch.zeros(supplement[:,:,0], dtype = torch.int64).to(device = self.device)
        # Assume that existing agents are named first
        supplement_extra[:,:,1] = (supplement[:,:,:,-1] > 0).any(1).argmin(dim = 1, keepdims = True)
        
        supplement = torch.concat((supplement, supplement_extra.unsqueeze(2)), dim = 2)
        
        supplement = torch.nan_to_num(supplement, 0.0)
        
        if Y is not None:
            return S[:,0,:X.shape[2]], supplement[:,:,:X.shape[2]], S[:,0,X.shape[2]:], supplement[:,:,X.shape[2]:]
        else:
            return S, supplement
        
    
    def train_loss_coll_batch(self, X0, supplement_past, Y0, supplement_fut):
        sigma = torch.tensor(100)
        
        batch_size = X0.shape[0]
        n_I = X0.shape[1]
        n_O = Y0.shape[1]

        dest = Y0[:, -1, :]

        future_vel = (dest - X0[:, - 1, :2]) / (n_O * self.dt) #peds*2
        initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True) # peds * 1

        hidden_states1 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        hidden_states1 = hidden_states1.to(device = self.device)
        
        cell_states1 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        cell_states1 = cell_states1.to(device = self.device)
        
        hidden_states2 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        hidden_states2 = hidden_states2.to(device = self.device)
        
        cell_states2 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        cell_states2 = cell_states2.to(device = self.device)

        for m in range(1, n_I):
            [outputs_features1, hidden_states1, cell_states1, 
             outputs_features2, hidden_states2, cell_states2] = self.model_coll.forward_lstm(X0[:,m], 
                                                                                             hidden_states1, cell_states1, 
                                                                                             hidden_states2, cell_states2)

        predictions = torch.zeros(batch_size, n_O, 2).to(device = self.device)

        coefficients, curr_supp = self.model_coll.forward_coefficient_people(outputs_features2, supplement_past[:, -1, :, :], 
                                                                             X0[:, -1, :2], X0[:, -1, 2:], device = self.device)  # peds*maxpeds*2, peds*(max_peds + 1)*4

        pred_pos, pred_vel = self.model_coll.forward_next_step(X0[:, -1, :2], X0[:, -1, 2:], initial_speeds, dest,
                                                               outputs_features1, coefficients, curr_supp, 
                                                               sigma, device = self.device)
        predictions[:, 0, :] = pred_pos

        for t in range(1, n_O):
            input_lstm = torch.cat((pred_pos, pred_vel), dim=1)  # peds*4
            [outputs_features1, hidden_states1, cell_states1, 
             outputs_features2, hidden_states2, cell_states2] = self.model_coll.forward_lstm(input_lstm, 
                                                                                             hidden_states1, cell_states1, 
                                                                                             hidden_states2, cell_states2)

            future_vel = (dest - pred_pos) / ((n_O - t) * self.dt)  # peds*2
            future_vel_norm = torch.norm(future_vel, dim=-1)  # peds
            initial_speeds = torch.unsqueeze(future_vel_norm, dim=-1)  # peds*1

            coefficients, curr_supp = self.model_coll.forward_coefficient_people(outputs_features2, supplement_fut[:, t, :, :], 
                                                                                 pred_pos, pred_vel, device = self.device)

            pred_pos, pred_vel = self.model_coll.forward_next_step(pred_pos, pred_vel, initial_speeds, dest,
                                                                   outputs_features1, coefficients, curr_supp, 
                                                                   sigma, device = self.device)
            predictions[:, t, :] = pred_pos
            
        loss = ((predictions - Y0) ** 2).sum(2).sqrt().mean(1)
        
        return loss
    
    #%%
    def train_loss_full_batch(self, X0, supplement_past, Y0, supplement_fut, optimizer_full = None):
        # get validation number of predictions
        N_val = self.params_full["n_values"]
        sigma = torch.tensor(100)
        
        if optimizer_full is not None:
            train_loss = 0
        
        T0 = torch.concat((X0, Y0), dim = 1)
        
        batch_size = X0.shape[0]
        n_I = X0.shape[1]
        n_O = Y0.shape[1]

        dest = Y0[:, -1, :]

        future_vel = (dest - X0[:, - 1, :2]) / (n_O * self.dt) #peds*2
        initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True) # peds * 1

        hidden_states1 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        hidden_states1 = hidden_states1.to(device = self.device)
        
        cell_states1 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        cell_states1 = cell_states1.to(device = self.device)
        
        hidden_states2 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        hidden_states2 = hidden_states2.to(device = self.device)
        
        cell_states2 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        cell_states2 = cell_states2.to(device = self.device)
        
        with torch.no_grad():
            for m in range(1, n_I):
                [outputs_features1, hidden_states1, cell_states1, 
                 outputs_features2, hidden_states2, cell_states2] = self.model_coll.forward_lstm(X0[:,m], 
                                                                                                 hidden_states1, cell_states1, 
                                                                                                 hidden_states2, cell_states2)
    
    
            coefficients, curr_supp = self.model_coll.forward_coefficient_people(outputs_features2, supplement_past[:, -1, :, :], 
                                                                                 X0[:, -1, :2], X0[:, -1, 2:], device = self.device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
    
            pred_pos, pred_vel = self.model_coll.forward_next_step(X0[:, -1, :2], X0[:, -1, 2:], initial_speeds, dest,
                                                                   outputs_features1, coefficients, curr_supp, 
                                                                   sigma, device = self.device)
        
        x = T0[:, :n_I, :2].clone()
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])) * self.params_full['data_scale']
        
        if optimizer_full is not None:
            alpha = (Y0[:, 0, :2] - pred_pos) * self.params_full['data_scale']
            alpha_recon, mu, var = self.model_full.forward(x, next_step=alpha, device = self.device)
            
            kld, adl = calculate_loss_cvae(mu, var, nn.MSELoss(), alpha, alpha_recon)
            loss = kld * self.params_full["kld_reg"] + adl
            
            optimizer_full.zero_grad()
            loss.backward()
            optimizer_full.step()
            
            train_loss += loss.detach().cpu().numpy()
        
        else:
            predictions = torch.zeros(batch_size, n_O, 2).to(device = self.device)
            T0_copy = T0.clone()
            alpha_step = torch.zeros(N_val - 5, len(X0), 2).to(device = self.device)
            for i in range(len(alpha_step)):
                alpha_step[i, :, :] = self.model_full.forward(x, device = self.device)
                
            alpha_step[-1,:,:] = 0.0
            prediction_correct = alpha_step / self.params_full['data_scale'] + pred_pos[np.newaxis]
            
            predictions_norm = torch.norm((prediction_correct - Y0[np.newaxis, :, 0, :2]), dim = -1)
            values, indices = torch.min(predictions_norm, dim = 0)  # peds
            
            predictions[:, 0, :] = prediction_correct[indices, np.arange(batch_size), :]
            T0_copy[:, n_I, :2] = predictions[:, 0, :]
            T0_copy[:, n_I, 2:] = (T0_copy[:, n_I, :2] - T0_copy[:, n_I - 1, :2]) / self.dt
            

        for t in range(1, n_O):
            input_lstm = torch.cat((pred_pos, pred_vel), dim=1)  # peds*4
            with torch.no_grad():
                [outputs_features1, hidden_states1, cell_states1, 
                 outputs_features2, hidden_states2, cell_states2] = self.model_coll.forward_lstm(input_lstm, 
                                                                                                 hidden_states1, cell_states1, 
                                                                                                 hidden_states2, cell_states2)
    
                future_vel = (dest - pred_pos) / ((n_O - t) * self.dt)  # peds*2
                initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True) # peds * 1
    
                coefficients, curr_supp = self.model_coll.forward_coefficient_people(outputs_features2, supplement_fut[:, t, :, :], 
                                                                                     pred_pos, pred_vel, device = self.device)
    
                pred_pos, pred_vel = self.model_coll.forward_next_step(pred_pos, pred_vel, initial_speeds, dest,
                                                                       outputs_features1, coefficients, curr_supp, 
                                                                       sigma, device = self.device)
            
            if optimizer_full is not None:
                x = T0[:, t : n_I + t, :2].clone()
            else:
                x = T0_copy[:, t : n_I + t, :2].clone()
                
            x = x - x[:, [0]]
            x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])) * self.params_full['data_scale']
            
            if optimizer_full is not None:
                alpha = (Y0[:, t, :2] - pred_pos) * self.params_full['data_scale']
                alpha_recon, mu, var = self.model_full.forward(x, next_step=alpha, device = self.device)
                
                kld, adl = calculate_loss_cvae(mu, var, nn.MSELoss(), alpha, alpha_recon)
                loss = kld * self.params_full["kld_reg"] + adl
                
                optimizer_full.zero_grad()
                loss.backward()
                optimizer_full.step()
            
                train_loss += loss.detach().cpu().numpy()
            
            else:
                alpha_step = torch.zeros(N_val - 5, len(X0), 2).to(device = self.device)
                for i in range(len(alpha_step)):
                    alpha_step[i, :, :] = self.model_full.forward(x, device = self.device)
                    
                alpha_step[-1,:,:] = 0.0
                prediction_correct = alpha_step / self.params_full['data_scale'] + pred_pos[np.newaxis]
                
                predictions_norm = torch.norm((prediction_correct - Y0[np.newaxis, :, t, :2]), dim = -1)
                values, indices = torch.min(predictions_norm, dim = 0)  # peds
                
                predictions[:, 0, :] = prediction_correct[indices, np.arange(batch_size), :]
                T0_copy[:, n_I + t, :2] = predictions[:, t, :]
                T0_copy[:, n_I + t, 2:] = (T0_copy[:, n_I + t, :2] - T0_copy[:, n_I + t - 1, :2]) / self.dt
                
        if optimizer_full is None:
            train_loss = ((predictions - Y0) ** 2).sum(2).sqrt().mean(1)

        return train_loss
    
    
    def pred_full_batch(self, X0, supplement_past, n_O):
        # get validation number of predictions
        N_val = self.params_full["n_values"]
        sigma = torch.tensor(100)
        
        batch_size = X0.shape[0]
        n_I = X0.shape[1]
        
        T0 = torch.zeros((batch_size, n_I + n_O, 4), device = self.device)
        T0[:,:n_I] = X0

        dest = Y0[:, -1, :]

        future_vel = (dest - X0[:, - 1, :2]) / (n_O * self.dt) #peds*2
        initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True) # peds * 1

        hidden_states1 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        hidden_states1 = hidden_states1.to(device = self.device)
        
        cell_states1 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        cell_states1 = cell_states1.to(device = self.device)
        
        hidden_states2 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        hidden_states2 = hidden_states2.to(device = self.device)
        
        cell_states2 = Variable(torch.zeros(batch_size, self.params_coll['rnn_size']))
        cell_states2 = cell_states2.to(device = self.device)
        
        with torch.no_grad():
            for m in range(1, n_I):
                [outputs_features1, hidden_states1, cell_states1, 
                 outputs_features2, hidden_states2, cell_states2] = self.model_coll.forward_lstm(X0[:,m], 
                                                                                                 hidden_states1, cell_states1, 
                                                                                                 hidden_states2, cell_states2)
    
    
            coefficients, curr_supp = self.model_coll.forward_coefficient_people(outputs_features2, supplement_past[:, -1, :, :], 
                                                                                 X0[:, -1, :2], X0[:, -1, 2:], device = self.device)  # peds*maxpeds*2, peds*(max_peds + 1)*4
    
            pred_pos, pred_vel = self.model_coll.forward_next_step(X0[:, -1, :2], X0[:, -1, 2:], initial_speeds, dest,
                                                                   outputs_features1, coefficients, curr_supp, 
                                                                   sigma, device = self.device)
        
        x = T0[:, :n_I, :2].clone()
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])) * self.params_full['data_scale']
        
        predictions = torch.zeros(batch_size, n_O, 2).to(device = self.device)
        alpha_step = torch.zeros(N_val - 5, len(X0), 2).to(device = self.device)
        for i in range(len(alpha_step)):
            alpha_step[i, :, :] = self.model_full.forward(x, device = self.device)
            
        alpha_step[-1,:,:] = 0.0
        prediction_correct = alpha_step / self.params_full['data_scale'] + pred_pos[np.newaxis]
        
        predictions_norm = torch.norm((prediction_correct - Y0[np.newaxis, :, 0, :2]), dim = -1)
        values, indices = torch.min(predictions_norm, dim = 0)  # peds
        
        predictions[:, 0, :] = prediction_correct[indices, np.arange(batch_size), :]
        T0[:, n_I, :2] = predictions[:, 0, :]
        T0[:, n_I, 2:] = (T0[:, n_I, :2] - T0[:, n_I - 1, :2]) / self.dt
            

        for t in range(1, n_O):
            input_lstm = torch.cat((pred_pos, pred_vel), dim=1)  # peds*4
            with torch.no_grad():
                [outputs_features1, hidden_states1, cell_states1, 
                 outputs_features2, hidden_states2, cell_states2] = self.model_coll.forward_lstm(input_lstm, 
                                                                                                 hidden_states1, cell_states1, 
                                                                                                 hidden_states2, cell_states2)
    
                future_vel = (dest - pred_pos) / ((n_O - t) * self.dt)  # peds*2
                initial_speeds = torch.norm(future_vel, dim = -1, keepdims = True) # peds * 1
    
                coefficients, curr_supp = self.model_coll.forward_coefficient_people(outputs_features2, supplement_fut[:, t, :, :], 
                                                                                     pred_pos, pred_vel, device = self.device)
    
                pred_pos, pred_vel = self.model_coll.forward_next_step(pred_pos, pred_vel, initial_speeds, dest,
                                                                       outputs_features1, coefficients, curr_supp, 
                                                                       sigma, device = self.device)
            
            x = T0[:, t : n_I + t, :2].clone()
            x = x - x[:, [0]]
            x = torch.reshape(x, (-1, x.shape[1] * x.shape[2])) * self.params_full['data_scale']
            
            alpha_step = torch.zeros(N_val - 5, len(X0), 2).to(device = self.device)
            for i in range(len(alpha_step)):
                alpha_step[i, :, :] = self.model_full.forward(x, device = self.device)
                
            alpha_step[-1,:,:] = 0.0
            prediction_correct = alpha_step / self.params_full['data_scale'] + pred_pos[np.newaxis]
            
            predictions_norm = torch.norm((prediction_correct - Y0[np.newaxis, :, t, :2]), dim = -1)
            values, indices = torch.min(predictions_norm, dim = 0)  # peds
            
            predictions[:, 0, :] = prediction_correct[indices, np.arange(batch_size), :]
            T0[:, n_I + t, :2] = predictions[:, t, :]
            T0[:, n_I + t, 2:] = (T0[:, n_I + t, :2] - T0[:, n_I + t - 1, :2]) / self.dt


        return predictions
    #%%
    def train_method(self):
        # Initialize train loss
        self.train_loss = np.zeros((3, max(self.params_goal['num_epochs'],
                                           self.params_coll['num_epochs'],
                                           self.params_full['num_epochs'])))
        batch_size = 256
        
        #######################################################################
        #                     Train goal model                                #
        #######################################################################
        model_goal_path = self.model_file[:4] + '_goal_model.pt'
        if os.path.isfile(model_goal_path):
            self.model_coll.load_state_dict(torch.load(model_goal_path, map_location = self.device))
        
        else:
            # initialize optimizer
            optimizer_goal = optim.Adam(self.model_goal.parameters(), lr = self.params_goal["learning_rate"])
            
            # prepare best val loss
            best_val_loss = 100000
            
            # go through epochs
            for epoch in range(self.params_goal['num_epochs']):
                print('')
                print('Train NSP - Goal model (Epoch {}/{})'.format(epoch + 1, self.params_goal['num_epochs']))
                self.model_goal.train()
                
                # go through train batches
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, _, _, _, _, _, train_epoch_done = self.provide_batch_data('train', int(batch_size / 8), 
                                                                                    val_split_size = 0.25)
                    
                    # only use 5 / 36 of batches
                    if not np.random.rand() < 5 / 36:
                        continue
                    
                    X0, Y0 = self.extract_data_batch_goal(X, Y, augment = True)
                    
                    optimizer_goal.zero_grad()
                    train_loss = self.train_loss_goal_batch(X0, Y0)
                    train_loss /= len(X0)
                    train_loss.backwards()
                    optimizer_goal.step()
                    
                # go through val batches
                self.model_goal.eval()
                
                val_loss_epoch = 0 
                val_epoch_done = False
                while not val_epoch_done:
                    X, Y, _, _, _, _, _, train_epoch_done = self.provide_batch_data('val', batch_size, 
                                                                                    val_split_size = 0.25)
                    
                    X0, Y0 = self.extract_data_batch_goal(X, Y, augment = False)
                    
                    with torch.no_grad():
                        val_loss = self.train_loss_goal_batch(X0, Y0)
                        
                    val_loss_epoch += val_loss.detach().cpu().numpy()
                
                self.train_loss[0, epoch] = val_loss_epoch
                
                print('Validation loss: {:7.3f}'.format(val_loss_epoch))
                # if validation error is improved, save model
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    os.makedirs(model_goal_path, exist_ok = True)
                    torch.save({'model_state_dict': self.model_goal.state_dict(),
                                'optimizer_state_dict': optimizer_goal.state_dict()}, 
                               model_goal_path)
                
            
        #######################################################################
        #                 Train Collision model                               #
        #######################################################################
        model_coll_path = self.model_file[:4] + '_coll_model.pt'
        if os.path.isfile(model_coll_path):
            self.model_coll.load_state_dict(torch.load(model_coll_path, map_location = self.device))
            
        else:
            # Load initial model states
            load_path_ini = os.sep.join(os.path.dirname(self.model_file).split(os.sep)[:-3])
            load_path_ini += os.sep + 'Models' + os.sep + 'NSP' + os.sep + 'saved_models' + os.sep + 'SDD_nsp_wo_ini.pt'
            model_coll_ini = torch.load(load_path_ini, map_location = self.device)
            
            checkpoint_dic = new_point(self.model_goal['model_state_dict'], model_coll_ini['model_state_dict'])
            self.model_coll.load_state_dict(checkpoint_dic)
            
            # set optimizer
            parameter_train = select_para(self.model_coll)

            optimizer_coll = optim.Adam([{'params': parameter_train}], lr = self.params_coll["learning_rate"])
            
            # prepare best val loss
            best_val_loss = 100000
            
            # go through epochs
            for epoch in range(self.params_coll['num_epochs']):
                print('')
                print('Train NSP - Collision model (Epoch {}/{})'.format(epoch + 1, self.params_coll['num_epochs']))
                self.model_coll.train()
                
                # go through train batches
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, _, _, _, _, _, train_epoch_done = self.provide_batch_data('train', int(batch_size), 
                                                                                    val_split_size = 0.25)
                    
                    X0, supp_past, Y0, supp_fut = self.extract_data_batch_coll(X, Y)
                    
                    optimizer_coll.zero_grad()
                    train_loss = self.train_loss_coll_batch(X0, supp_past, Y0, supp_fut)
                    train_loss /= len(X0)
                    train_loss.backwards()
                    optimizer_coll.step()
                    
                # go through val batches
                self.model_coll.eval()
                
                val_loss_epoch = 0 
                val_epoch_done = False
                while not val_epoch_done:
                    X, Y, _, _, _, _, _, train_epoch_done = self.provide_batch_data('val', batch_size, 
                                                                                    val_split_size = 0.25)
                    
                    X0, supp_past, Y0, supp_fut = self.extract_data_batch_coll(X, Y)
                    
                    with torch.no_grad():
                        val_loss = self.train_loss_coll_batch(X0, supp_past, Y0, supp_fut)
                        
                    val_loss_epoch += val_loss.detach().cpu().numpy()
                
                self.train_loss[1, epoch] = val_loss_epoch
                
                print('Validation loss: {:7.3f}'.format(val_loss_epoch))
                # if validation error is improved, save model
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    os.makedirs(model_coll_path, exist_ok = True)
                    torch.save({'model_state_dict': self.model_coll.state_dict(),
                                'optimizer_state_dict': optimizer_coll.state_dict()}, 
                               model_coll_path)
    
                
        #######################################################################
        #                      Train Full model                               #
        #######################################################################
        model_full_path = self.model_file[:4] + '_full_model.pt'
        if os.path.isfile(model_full_path):
            self.model_full.load_state_dict(torch.load(model_full_path, map_location = self.device)  ) 
        else:
            # initialize optimizer
            optimizer_full = optim.Adam(self.model_full.parameters(), lr = self.params_full["learning_rate"])
            
            # Set inner model to eval
            self.model_coll.eval()
            
            # prepare best val loss
            best_val_loss = 100000
            
            # go through epochs
            for epoch in range(self.params_full['num_epochs']):
                print('')
                print('Train NSP - CVAE model (Epoch {}/{})'.format(epoch + 1, self.params_coll['num_epochs']))
                self.model_full.train()
                
                # go through train batches
                train_epoch_done = False
                while not train_epoch_done:
                    X, Y, _, _, _, _, _, train_epoch_done = self.provide_batch_data('train', int(batch_size), 
                                                                                    val_split_size = 0.25)
                    
                    X0, supp_past, Y0, supp_fut = self.extract_data_batch_coll(X, Y)
                    
                    train_loss = self.train_loss_full_batch(X0, supp_past, Y0, supp_fut, optimizer_full)
                    
                # go through val batches
                self.model_full.eval()
                
                val_loss_epoch = 0 
                val_epoch_done = False
                while not val_epoch_done:
                    X, Y, _, _, _, _, _, train_epoch_done = self.provide_batch_data('val', batch_size, 
                                                                                    val_split_size = 0.25)
                    
                    X0, supp_past, Y0, supp_fut = self.extract_data_batch_coll(X, Y)
                    
                    with torch.no_grad():
                        val_loss = self.train_loss_full_batch(X0, supp_past, Y0, supp_fut)
                        
                    val_loss_epoch += val_loss.detach().cpu().numpy()
                
                self.train_loss[2, epoch] = val_loss_epoch
                
                print('Validation loss: {:7.3f}'.format(val_loss_epoch))
                # if validation error is improved, save model
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    os.makedirs(model_full_path, exist_ok = True)
                    torch.save({'model_state_dict': self.model_full.state_dict(),
                                'optimizer_state_dict': optimizer_full.state_dict()}, 
                               model_full_path)
            
            
        
    def load_method(self):
        # Load inner model
        model_coll_path = self.model_file[:4] + '_coll_model.pt'
        self.model_coll.load_state_dict(torch.load(model_coll_path, map_location = self.device))
        
        # Load outer model
        model_coll_path = self.model_file[:4] + '_coll_model.pt'
        self.model_coll.load_state_dict(torch.load(model_coll_path, map_location = self.device))
    
    
    
    
    def predict_method(self):
        batch_size = 256
        prediction_done = False
        while not prediction_done:
            X, T, _, _, _, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', batch_size)
            
            X0, supp_past = self.extract_data_batch_coll(X)
            # Run prediction pass
            with torch.no_grad():
                Y_pred = self.pred_full_batch(X0, supp_past, num_steps)
                    
            Pred = Y_pred.detach().cpu().numpy()
            
            torch.cuda.empty_cache()
            
            # save predictions
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id)