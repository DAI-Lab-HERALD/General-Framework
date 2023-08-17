import os
import numpy as np
import pandas as pd
import torch
from torch import optim
from model_template import model_template

from agentformer.model.model_lib import model_dict
from agentformer.utils.torch import get_scheduler
from agentformer.utils.config import Config

class agent_yuan(model_template):
    
    def setup_method(self, seed = 0):
        # check for gpus
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.set_default_dtype(torch.float32)
        
        torch.cuda.empty_cache()
        # Get params
        # Required attributes of the model
        self.min_t_O_train = 5
        self.max_t_O_train = self.num_timsteps_out
        self.predict_single_agent = False
        self.can_use_map = True
        # If self.can_use_map = True, the following is also required
        self.target_width = 180
        self.target_height = 100
        self.grayscale = True
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        self.batch_size = 2 * total_memory / (len(self.Input_path_train.columns) ** 1.5 * (self.num_timsteps_out + self.num_timesteps_in))
        self.batch_size = max(1, int(np.floor(self.batch_size)))
        
        self.sample_number = 10
        
            
    def extract_data_batch(self, X, T, Pred_agents, Y = None, img = None, img_m_per_px = None, num_steps = 10):  
        # Determine if this is training
        train = Y != None
        
        Data = []
        for i in range(len(X)):
            X_i = torch.from_numpy(X[i]).to(dtype = torch.float32) # num_agents x num_timesteps x 211
            if train:
                Y_i = torch.from_numpy(Y[i]).to(dtype = torch.float32)
            else:
                Y_i = torch.ones((len(X_i), num_steps, 2), dtype = torch.float32) * torch.nan
                
            X_useful = torch.isfinite(X_i).all(-1)
            Y_useful = torch.isfinite(Y_i).all(-1)
            
            X_i[~X_useful] = 0.0
            Y_i[~Y_useful] = 0.0
            
            if not train:
                Y_useful[torch.from_numpy(Pred_agents[i]), :num_steps] = True
            
            pre_motion_3D = list(X_i)
            fut_motion_3D = list(Y_i)
            
            pre_motion_mask = list(X_useful.to(dtype = torch.float32))
            fut_motion_mask = list(Y_useful.to(dtype = torch.float32))
            
            if img is not None:
                img_sample = img[i,:,:,80:] # Cut of behond agent
                img_sample = img_sample.transpose(0,3,1,2) # Put channels first
                
                img_scale = 1 / img_m_per_px[i].mean()
            
            data = {
                'pre_motion_3D': pre_motion_3D,
                'fut_motion_3D': fut_motion_3D,
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                'pre_data': None,
                'fut_data': None,
                'heading': None,
                # Todo: Check what this exactly does
                'valid_id': [1.0, 2.0, 3.0, 4.0, 5.0],
                'pred_mask': None,
                'scene_map': img_sample,
                'traj_scale': img_scale,
                'seq': 'Not_needed',
                'frame': i
            }
            
            Data.append(data)
        
        Data = np.array(Data)  
        return Data
        

    def train_method(self):
        ######################################################################
        ##                Train VAE                                         ##
        ######################################################################
        # load hyperparams and set up model
        cfg = Config('hyperparams_pre', False, create_dirs = False)        
        cfg.yml_dict["past_frames"] = self.num_timesteps_in
        cfg.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg.yml_dict["future_frames"] = self.num_timesteps_out.max()
        cfg.yml_dict["min_future_frames"] = self.num_timesteps_out.min()
        
        cfg.yml_dict["sample_k"] = self.sample_number
        cfg.yml_dict["loss_cfg"]["sample"]["k"] = self.sample_number
        
        if self.use_map:
            cfg.yml_dict["use_map"] = True
            cfg.yml_dict["input_type"] = cfg.yml_dict["input_type"] + ['map']
            if self.grayscale:
                cfg.yml_dict.map_encoder["map_channels"] = 1
            else:
                cfg.yml_dict.map_encoder["map_channels"] = 3
        else:
            cfg.yml_dict["use_map"] = False
        
        model_id = cfg.get('model_id', 'agentformer')
        self.model_vae = model_dict[model_id](cfg)
        
        cp_path = self.model_file[:-4] + '_vae.p'
        
        if not os.path.isfile(cp_path):
            
            optimizer = optim.Adam(self.model_vae.parameters(), lr=cfg.lr)
            scheduler_type = cfg.get('lr_scheduler', 'linear')
            if scheduler_type == 'linear':
                scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
            elif scheduler_type == 'step':
                scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
            else:
                raise ValueError('unknown scheduler type!')
                
            Epoch_loss_vae = []
            start_epoch = 1
            
            # check if partially trained model exists
            cp_path_test = cp_path[:-2]
            files = os.listdir(os.path.dirname(cp_path_test))
            filename = os.path.basename(cp_path_test)
            
            for file_name_candidate in files:
                if filename in file_name_candidate:
                    saved_epoch = int(file_name_candidate[len(filename) + 1:-2])
                    start_epoch = saved_epoch + 1
                    cp_path_epoch = cp_path[:-2] + '_{}.p'.format(saved_epoch)
                    model_cp = torch.load(cp_path_epoch, map_location='cpu')
                    self.model_vae.load_state_dict(model_cp['model_dict'])
                    
                    loss_epoch_path = cp_path[:-2] + '_{}_loss.npy'.format(saved_epoch)
                    Epoch_loss_vae = list(np.load(loss_epoch_path))
                    
                    break
                
            self.model_vae.set_device(self.device)
            self.model_vae.train()
            
            # train vae model
            epochs = cfg.yml_dict["num_epochs"]
            print('')
            
            # Set up scheduler
            for epoch in range(1, start_epoch):
                scheduler.step()
                
            # epochs = 2 # TODO: Remove this line
            for epoch in range(start_epoch, epochs + 1):
                print('Train VAE: Epoch ' + 
                      str(epoch).rjust(len(str(epochs))) + 
                      '/{}'.format(epochs), flush = True)
                
                epoch_loss = 0.0
                epoch_done = False
                
                batch = 0
                samples = 0
                while not epoch_done:
                    batch += 1
                    print('Train VAE: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch {}'.format(epochs, batch), flush = True)
                    
                    X, Y, T, img, img_m_per_px, Pred_agents, num_steps, epoch_done = self.provide_batch_data('train', self.batch_size)
                    data = self.extract_data_batch(X, T, Pred_agents, Y, img, img_m_per_px, num_steps)
                    samples += len(data)
                    # prevent unnecessary simulations
                    self.model_vae.future_decoder.future_frames = num_steps
                    
                    # Give data to model
                    self.model_vae.set_data(data)
                    self.model_vae()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_vae.compute_loss()
                    optimizer.zero_grad()
                    
                    # Update model
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.detach().cpu().numpy()
                    torch.cuda.empty_cache()
                scheduler.step()
                    
                Epoch_loss_vae.append(epoch_loss)
                
                print('Train VAE: Epoch ' + str(epoch).rjust(len(str(epochs))) + 
                      '/{} with loss {:0.3f}'.format(epochs, epoch_loss/samples), flush = True)
                print('', flush = True)
                
                # Save intermediate
                cp_path_epoch = cp_path[:-2] + '_{}.p'.format(epoch)
                model_cp_epoch = {'model_dict': self.model_vae.state_dict()}

                os.makedirs(os.path.dirname(cp_path_epoch), exist_ok=True)
                torch.save(model_cp_epoch, cp_path_epoch)  
                
                loss_epoch_path = cp_path[:-2] + '_{}_loss.npy'.format(epoch)
                np.save(loss_epoch_path, np.array(Epoch_loss_vae))
                
                if epoch >= 2:
                    cp_path_epoch_last   = cp_path[:-2] + '_{}.p'.format(epoch - 1)
                    loss_epoch_path_last = cp_path[:-2] + '_{}_loss.npy'.format(epoch - 1)
                    
                    os.remove(cp_path_epoch_last)
                    os.remove(loss_epoch_path_last)
                    
                 
            Epoch_loss_vae = np.array(Epoch_loss_vae)   
            os.rename(cp_path_epoch, cp_path)  
            # Save intermediate
            model_cp = {'model_dict': self.model_vae.state_dict()}
            
            os.makedirs(os.path.dirname(cp_path), exist_ok=True)
            torch.save(model_cp, cp_path)   
         
        else:
            loss_epoch_path = cp_path[:-2] + '_{}_loss.npy'.format(epochs)
            Epoch_loss_vae = np.load(loss_epoch_path)
            
            model_cp = torch.load(cp_path, map_location='cpu')
            self.model_vae.load_state_dict(model_cp['model_dict'])
        
        
        # save weights
        Weights_vae = list(self.model_vae.parameters())
        self.weights_vae = []
        for weigths in Weights_vae:
            self.weights_vae.append(weigths.detach().cpu().numpy())


        ######################################################################
        ##                Train DLow                                        ##
        ######################################################################
        
        cfg_d = Config('hyperparams', False, create_dirs = False)   
        cfg_d.yml_dict["past_frames"] = self.num_timesteps_in
        cfg_d.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg_d.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg_d.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg_d.yml_dict["sample_k"] = self.sample_number
        cfg_d.yml_dict['model_path'] = cp_path
        
        if self.use_map:
            cfg_d.yml_dict["use_map"] = True
            cfg_d.yml_dict["input_type"] = cfg.yml_dict["input_type"]
            if self.grayscale:
                cfg_d.yml_dict.map_encoder["map_channels"] = 1
            else:
                cfg_d.yml_dict.map_encoder["map_channels"] = 3
        else:
            cfg_d.yml_dict["use_map"] = False
        
        # create model
        model_id = cfg_d.get('model_id', 'dlow')
        self.model_dlow = model_dict[model_id](cfg_d)
        
        
        cp_path_dlow = self.model_file[:-4] + '_dlow.p'
        if not os.path.isfile(cp_path_dlow):
            optimizer = optim.Adam(self.model_dlow.parameters(), lr=cfg_d.lr)
            scheduler_type = cfg_d.get('lr_scheduler', 'linear')
            if scheduler_type == 'linear':
                scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg_d.lr_fix_epochs, nepoch=cfg_d.num_epochs)
            elif scheduler_type == 'step':
                scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg_d.decay_step, decay_gamma=cfg_d.decay_gamma)
            else:
                raise ValueError('unknown scheduler type!')
            
            Epoch_loss_dlow = []
            start_epoch = 1
            
            # check if partially trained model exists
            cp_path_test = cp_path_dlow[:-2]
            files = os.listdir(os.path.dirname(cp_path_test))
            filename = os.path.basename(cp_path_test)
            
            for file_name_candidate in files:
                if filename in file_name_candidate:
                    saved_epoch = int(file_name_candidate[len(filename) + 1:-2])
                    start_epoch = saved_epoch + 1
                    cp_path_epoch = cp_path_dlow[:-2] + '_{}.p'.format(saved_epoch)
                    model_cp = torch.load(cp_path_epoch, map_location='cpu')
                    self.model_dlow.load_state_dict(model_cp['model_dict'])
                    
                    loss_epoch_path = cp_path_dlow[:-2] + '_{}_loss.npy'.format(saved_epoch)
                    Epoch_loss_dlow = list(np.load(loss_epoch_path))
                    
                    break
                
            self.model_dlow.set_device(self.device)
            self.model_dlow.train()
            
            # train dlow model
            epochs = cfg_d.yml_dict["num_epochs"]
            print('')
            
            # Set up scheduler
            for epoch in range(1, start_epoch):
                scheduler.step()
                
            # epochs = 2 # TODO: Remove this line
            for epoch in range(start_epoch, epochs + 1):
                print('Train DLow: Epoch ' + 
                      str(epoch).rjust(len(str(epochs))) + 
                      '/{}'.format(epochs))
                
                epoch_loss = 0.0
                epoch_done = False
                
                batch = 0
                samples = 0
                while not epoch_done:
                    batch += 1
                    print('Train DLow: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch {}'.format(epochs, batch), flush = True)
                    
                    X, Y, T, img, img_m_per_px, Pred_agents, num_steps, epoch_done = self.provide_batch_data('train', self.batch_size)
                    data = self.extract_data_batch(X, T, Pred_agents, Y, img, img_m_per_px, num_steps)
                    samples += len(data)
                    # prevent unnecessary simulations
                    self.model_dlow.pred_model[0].future_decoder.future_frames = num_steps
                    
                    # Give data to model
                    self.model_dlow.set_data(data)
                    self.model_dlow()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_dlow.compute_loss()
                    optimizer.zero_grad()
                    
                    # Update model
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.detach().cpu().numpy()
                    torch.cuda.empty_cache()
                scheduler.step()
                
                Epoch_loss_dlow.append(epoch_loss)
                print('Train DLow: Epoch ' + str(epoch).rjust(len(str(epochs))) + 
                      '/{} with loss {:0.3f}'.format(epochs, epoch_loss/samples), flush = True)
                print('')  
                
                # Save intermediate
                if epoch < epochs:
                    cp_path_epoch = cp_path_dlow[:-2] + '_{}.p'.format(epoch)
                    model_cp_epoch = {'model_dict': self.model_dlow.state_dict()}
                
                    os.makedirs(os.path.dirname(cp_path_epoch), exist_ok=True)
                    torch.save(model_cp_epoch, cp_path_epoch)  
                    
                loss_epoch_path = cp_path_dlow[:-2] + '_{}_loss.npy'.format(epoch)
                np.save(loss_epoch_path, np.array(Epoch_loss_dlow))
                
                if epoch >= 2:
                    cp_path_epoch_last   = cp_path_dlow[:-2] + '_{}.p'.format(epoch - 1)
                    loss_epoch_path_last = cp_path_dlow[:-2] + '_{}_loss.npy'.format(epoch - 1)
                    
                    os.remove(cp_path_epoch_last)
                    os.remove(loss_epoch_path_last)
            
            Epoch_loss_dlow = np.array(Epoch_loss_dlow)
            
        else:
            loss_epoch_path = cp_path_dlow[:-2] + '_{}_loss.npy'.format(epochs)
            Epoch_loss_dlow = np.load(loss_epoch_path)
            
            model_cp = torch.load(cp_path_dlow, map_location='cpu')
            self.model_dlow.load_state_dict(model_cp['model_dict']) 
         
        
        # save weights
        Weights_dlow = list(self.model_dlow.parameters())
        self.weights_dlow = []
        for weigths in Weights_dlow:
            self.weights_dlow.append(weigths.detach().cpu().numpy()) 
        
        if os.path.isfile(cp_path): 
            os.remove(cp_path)
        self.weights_saved = [self.weights_vae, self.weights_dlow]
        
        # save loss
        loss_len = max(len(Epoch_loss_dlow), len(Epoch_loss_vae))
        self.train_loss = np.ones((2, loss_len)) * np.nan
        
        self.train_loss[0,:len(Epoch_loss_vae)]  = Epoch_loss_vae
        self.train_loss[1,:len(Epoch_loss_dlow)] = Epoch_loss_dlow
        
        
    def load_method(self, l2_regulization = 0):
        [self.weights_vae, self.weights_dlow] = self.weights_saved
        
        ######################################################################
        ##                Load VAE                                          ##
        ######################################################################
        
        cfg = Config('hyperparams_pre', False, create_dirs = False)        
        cfg.yml_dict["past_frames"] = self.num_timesteps_in
        cfg.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg.yml_dict["sample_k"] = self.sample_number
        cfg.yml_dict["loss_cfg"]["sample"]["k"] = self.sample_number
        
        if self.use_map:
            cfg.yml_dict["use_map"] = True
            cfg.yml_dict["input_type"] = cfg.yml_dict["input_type"] + ['map']
            if self.grayscale:
                cfg.yml_dict.map_encoder["map_channels"] = 1
            else:
                cfg.yml_dict.map_encoder["map_channels"] = 3
        else:
            cfg.yml_dict["use_map"] = False
        
        # load model
        model_id = cfg.get('model_id', 'agentformer')
        self.model_vae = model_dict[model_id](cfg)
            
        Weights_vae = list(self.model_vae.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_vae):
                Weights_vae[i][:] = torch.from_numpy(weights)[:]
        
        cp_path = self.model_file[:-4] + '_vae.p'
        model_cp = {'model_dict': self.model_vae.state_dict()}
        
        os.makedirs(os.path.dirname(cp_path), exist_ok=True)
        torch.save(model_cp, cp_path)   
        
        ######################################################################
        ##                Load DLow                                         ##
        ######################################################################
        
        cfg_d = Config('hyperparams', False, create_dirs = False)   
        cfg_d.yml_dict["past_frames"] = self.num_timesteps_in
        cfg_d.yml_dict["min_past_frames"] = self.num_timesteps_in
              
        cfg_d.yml_dict["future_frames"] = max(self.num_timesteps_out)
        cfg_d.yml_dict["min_future_frames"] = min(self.num_timesteps_out)
        
        cfg_d.yml_dict["sample_k"] = self.sample_number
        cfg_d.yml_dict['model_path'] = cp_path
        
        if self.use_map:
            cfg_d.yml_dict["use_map"] = True
            cfg_d.yml_dict["input_type"] = cfg.yml_dict["input_type"]
            if self.grayscale:
                cfg_d.yml_dict.map_encoder["map_channels"] = 1
            else:
                cfg_d.yml_dict.map_encoder["map_channels"] = 3
        else:
            cfg_d.yml_dict["use_map"] = False
        
        # create model
        model_id = cfg_d.get('model_id', 'agentformer')
        self.model_dlow = model_dict[model_id](cfg_d)
        
        Weights_dlow = list(self.model_dlow.parameters())
        with torch.no_grad():
            for i, weights in enumerate(self.weights_dlow):
                Weights_dlow[i][:] = torch.from_numpy(weights)[:]
        
        os.remove(cp_path)

        
    def predict_method(self):
        self.model_dlow.set_device(self.device)
        self.model_dlow.eval()
        batch = 0
        prediction_done = False
        while not prediction_done:
            batch += 1
            print('Predict trajectron: Batch {}'.format( batch))
            
            # check if problem was already solved in saved data
            X, T, img, img_m_per_px, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', 2)
            data = self.extract_data_batch(X, T, Pred_agents, img, img_m_per_px, num_steps)
            
            # OOM protection
            splits = int(np.ceil((self.num_samples_path_pred / self.sample_number)))
            
            num_samples_path_pred_max = int(self.sample_number * splits)
            
            Pred = np.zeros((len(data), Pred_agents.shape[1], num_samples_path_pred_max, num_steps, 2), dtype = np.float32)
            
            for i in range(splits):
                # Rewrite random generators
                np.random.seed(i)
                torch.manual_seed(i)
                torch.cuda.manual_seed_all(i)
                
                Index = np.arange(i * self.sample_number, min((i + 1) * self.sample_number, num_samples_path_pred_max))
                with torch.no_grad():
                    self.model_dlow.pred_model[0].future_decoder.future_frames = num_steps
                    self.model_dlow.set_data(data)
                    
                    sample_motion_3D, _ = self.model_dlow.inference(mode = 'infer', sample_num = self.sample_number)
            
                pred = sample_motion_3D.detach().cpu().numpy().astype(np.float32)
                torch.cuda.empty_cache()
                
                Pred[:,:,Index] = pred[:,:,:,:num_steps]
                
            # cut number of predictions down if needed
            Pred = Pred[:, :, :self.num_samples_path_pred]
            
            self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)

    
    def check_trainability_method(self):
        return None
    
    def get_output_type(self = None):
        return 'path_all_wi_pov'
    
    def get_name(self = None):
        names = {'print': 'AgentFormer',
                 'file': 'agent_form',
                 'latex': r'\emph{AF}'}
        return names
        
    def save_params_in_csv(self = None):
        return False
    
    def requires_torch_gpu(self = None):
        return True
    
    def provides_epoch_loss(self = None):
        return True