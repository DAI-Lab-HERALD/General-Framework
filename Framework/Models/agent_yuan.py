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
        self.num_timesteps_in = len(self.Input_path_train.to_numpy()[0,0])
        self.num_timesteps_out = np.zeros(len(self.Output_T_train), int)
        for i_sample in range(self.Output_T_train.shape[0]):
            len_use = len(self.Output_T_train[i_sample])
            if self.data_set.num_timesteps_out_real == len_use:
                self.num_timesteps_out[i_sample] = len_use
            else:
                self.num_timesteps_out[i_sample] = len_use - np.mod(len_use - self.data_set.num_timesteps_out_real, 5)
                
        self.remain_samples = self.num_timesteps_out >= 5
        self.num_timesteps_out = np.minimum(self.num_timesteps_out[self.remain_samples], self.data_set.num_timesteps_out_real)
        
        self.use_map = self.data_set.includes_images()
        self.target_width = 180
        self.target_height = 100
        
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
        self.batch_size = int(np.floor(2 * total_memory / (len(self.Input_path_train.columns) ** 1.5 * 
                                                           (self.num_timesteps_out.max() + 
                                                            self.num_timesteps_in))))
        self.sample_number = 10
        self.grayscale = True
        
        
    def extract_data(self, train = True):
        if train:
            N_O = self.num_timesteps_out
            
            X_help = self.Input_path_train.to_numpy()
            Y_help = self.Output_path_train.to_numpy() 
            Types  = self.Type_train.to_numpy()
            
            X_help = X_help[self.remain_samples]
            Y_help = Y_help[self.remain_samples]
            Types  = Types[self.remain_samples]
            self.domain_old = self.Domain_train.iloc[self.remain_samples]
        else:
            N_O = self.num_timesteps_out_test
            
            X_help = self.Input_path_test.to_numpy()
            Types  = self.Type_test.to_numpy()
            self.domain_old = self.Domain_test
        
        
        
        Agents = np.array(self.input_names_train)
        
        # Extract predicted agents
        Pred_agents = np.array([agent in self.data_set.needed_agents for agent in Agents])
        assert Pred_agents.sum() > 0, "nothing to predict"
        
        # Prepare numpy position array
        X = np.ones(list(X_help.shape) + [self.num_timesteps_in, 2], dtype = np.float32) * np.nan
        if train:
            Y = np.ones(list(Y_help.shape) + [self.num_timesteps_out.max(), 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample in range(X.shape[0]):
            for i_agent, agent in enumerate(Agents):
                if isinstance(X_help[i_sample, i_agent], float):
                    assert not Pred_agents[i_agent], 'A needed agent is not given.'
                else:    
                    X[i_sample, i_agent] = X_help[i_sample, i_agent].astype(np.float32)
                    if train:
                        n_time = self.num_timesteps_out[i_sample]
                        Y[i_sample, i_agent, :n_time] = Y_help[i_sample, i_agent][:n_time].astype(np.float32)
        
        # Transform to torch tensor
        X = torch.from_numpy(X)
        if train:
            Y = torch.from_numpy(Y)
        
        
        if self.use_map:
            centre = X[...,-1,:].reshape(-1, 2).cpu().numpy()
            x_rel = centre - X[...,-2,:].reshape(-1, 2).cpu().numpy()
            rot = np.angle(x_rel[:,0] + 1j*x_rel[:,1]) 
            domain_repeat = self.domain_old.loc[self.domain_old.index.repeat(X.shape[1])]
            img, img_m_per_px = self.data_set.return_batch_images(domain_repeat, centre, rot,
                                                                  target_height = self.target_height, 
                                                                  target_width = self.target_width, 
                                                                  grayscale = False, return_resolution = True)
            if self.grayscale:
                img = img[:,:,80:].transpose(0,3,1,2).reshape(X.shape[0], X.shape[1], 1, 
                                                              self.target_height, self.target_width - 80)
            else:
                img = img[:,:,80:].transpose(0,3,1,2).reshape(X.shape[0], X.shape[1], 3, 
                                                              self.target_height, self.target_width - 80)

            img_scale = 1 / img_m_per_px.reshape(X.shape[0], X.shape[1]).mean(1)
        else:
            img = np.ones(len(self.domain_old)) * np.nan
            img_scale = np.zeros(len(self.domain_old))
        
        Data = []
        
        for i in range(len(X)):
            X_i = X[i].clone()
            useful = torch.isfinite(X_i).all(-1).all(-1)
            X_i[~useful] = 0.0
            useful = useful.to(dtype = torch.float32)
            
            pre_motion_3D = list(X_i)
            pre_motion_mask = [torch.ones(self.num_timesteps_in) * useful[j] for j in range(len(pre_motion_3D))] 
            
            fut_motion_3D = [torch.zeros((N_O.max(),2)) for i in range(len(pre_motion_3D))] 
            fut_motion_mask = [torch.zeros(N_O.max()) for i in range(len(pre_motion_3D))] 
            for i_agent, agent in enumerate(Agents):
                if Pred_agents[i_agent]:
                    if train:
                        Y_fut = Y[i,i_agent,:N_O[i]]
                        assert torch.isfinite(Y_fut).all()
                        fut_motion_3D[i_agent][:N_O[i],:] = Y_fut
                    fut_motion_mask[i_agent][:N_O[i]] = 1.0
            
            data = {
                'pre_motion_3D': pre_motion_3D,
                'fut_motion_3D': fut_motion_3D,
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                'pre_data': None,
                'fut_data': None,
                'heading': None,
                'valid_id': [1.0, 2.0, 3.0, 4.0, 5.0],
                'pred_mask': None,
                'scene_map': img[i],
                'traj_scale': img_scale[i],
                'seq': 'Not_needed',
                'frame': i
            }
            
            Data.append(data)
        
        Data = np.array(Data)    
        
        
        # Get indices
        Index = np.argsort(-N_O)
        
        # Get dlow batches
        if train:
            max_batch_size_dlow = self.batch_size
        else:
            max_batch_size_dlow = self.batch_size
            
        
        Index_batches_dlow = []
        batch_dlow = []
        batch_size_dlow = 0
        
        for i, ind in enumerate(Index):
            batch_dlow.append(ind)
            batch_size_dlow += N_O[ind] / self.num_timesteps_out.max()
            if batch_size_dlow >= max_batch_size_dlow or i == len(Index) - 1:
                Index_batches_dlow.append(batch_dlow)
                batch_dlow = []
                batch_size_dlow = 0
                
        Batches_dlow = np.arange(len(Index_batches_dlow))
        
        
        if train:
            max_batch_size_vae = self.batch_size
            Index_batches_vae = []
            batch_vae = []
            batch_size_vae = 0
            for i, ind in enumerate(Index):
                batch_vae.append(ind)
                batch_size_vae += N_O[ind] / N_O.max()
                if batch_size_vae >= max_batch_size_vae or i == len(Index) - 1:
                    Index_batches_vae.append(batch_vae)
                    batch_vae = []
                    batch_size_vae = 0
                    
            Batches_vae = np.arange(len(Index_batches_vae))
            
        if train:
            return Data, Batches_dlow, Index_batches_dlow, Batches_vae, Index_batches_vae
        else:
            return Data, Batches_dlow, Index_batches_dlow, Pred_agents, Agents
            
        

    def train_method(self):
        # Prepare data preliminary
        Data, Batches_dlow, Index_batches_dlow, Batches_vae, Index_batches_vae = self.extract_data(train = True)
        
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
                np.random.shuffle(Batches_vae)
                epoch_loss = 0.0
                for i, ind in enumerate(Batches_vae):
                    print('Train VAE: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch '.format(epochs) + 
                          str(i + 1).rjust(len(str(len(Batches_vae)))) + 
                          '/{}'.format(len(Batches_vae)), flush = True)
                    data = Data[Index_batches_vae[ind]]
                    # prevent unnecessary simulations
                    self.model_vae.future_decoder.future_frames = self.num_timesteps_out[Index_batches_vae[ind]].max()
                    
                    # Give data to model
                    self.model_vae.set_data(data)
                    self.model_vae()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_vae.compute_loss()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.detach().cpu().numpy()
                    torch.cuda.empty_cache()
                scheduler.step()
                    
                Epoch_loss_vae.append(epoch_loss)
                print('Train VAE: Epoch ' + str(epoch).rjust(len(str(epochs))) + 
                      '/{} with loss {:0.3f}'.format(epochs, epoch_loss/len(Data)), flush = True)
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
                np.random.shuffle(Batches_dlow)
                epoch_loss = 0.0
                for i, ind in enumerate(Batches_dlow):
                    print('Train DLow: Epoch ' + 
                          str(epoch).rjust(len(str(epochs))) + 
                          '/{}, Batch '.format(epochs) + 
                          str(i + 1).rjust(len(str(len(Batches_dlow)))) + 
                          '/{}'.format(len(Batches_dlow)), flush = True)
                    data = Data[Index_batches_dlow[ind]]
                    self.model_dlow.pred_model[0].future_decoder.future_frames = self.num_timesteps_out[Index_batches_dlow[ind]].max()
                    self.model_dlow.set_data(data)
                    self.model_dlow()
                    total_loss, loss_dict, loss_unweighted_dict = self.model_dlow.compute_loss()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.detach().cpu().numpy()
                    torch.cuda.empty_cache()
                scheduler.step()
                
                Epoch_loss_dlow.append(epoch_loss)
                print('Train DLow: Epoch ' + str(epoch).rjust(len(str(epochs))) + 
                      '/{} with loss {:0.3f}'.format(epochs, epoch_loss/len(Data)), flush = True)
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
        self.num_timesteps_out_test = np.zeros(len(self.Output_T_pred_test), int)
        for i_sample in range(len(self.Output_T_pred_test)):
            self.num_timesteps_out_test[i_sample] = len(self.Output_T_pred_test[i_sample])
            
        # get desired output length
        Data, Batches_dlow, Index_batches_dlow, Pred_agents, Agents = self.extract_data(train = False)
        
        Path_names = np.array([name for name in self.Output_path_train.columns]).reshape(-1, 2)
        
        Output_Path = pd.DataFrame(np.empty((len(Data), Pred_agents.sum()), np.ndarray), 
                                   columns = Path_names[Pred_agents])
        
        self.model_dlow.set_device(self.device)
        self.model_dlow.eval()
        
        for batch in Batches_dlow:
            print('Eval DLow: Batch ' + 
                  str(batch + 1).rjust(len(str(len(Batches_dlow)))) + 
                  '/{}'.format(len(Batches_dlow)))
            
            # check if problem was already solved in saved data
            index_batch = Index_batches_dlow[batch]
            
            
            data = Data[index_batch]
            # OOM protection
            splits = int(np.ceil((self.num_samples_path_pred / self.sample_number)))
            
            num_samples_path_pred_max = int(self.sample_number * splits)
            
            Pred = np.zeros((len(data),Pred_agents.sum(), num_samples_path_pred_max, 
                             self.num_timesteps_out_test[index_batch].max(), 2))
            
            for i in range(splits):
                Index = np.arange(i * self.sample_number, min((i + 1) * self.sample_number, num_samples_path_pred_max))
                with torch.no_grad():
                    self.model_dlow.pred_model[0].future_decoder.future_frames = self.num_timesteps_out_test[index_batch].max()
                    self.model_dlow.set_data(data)
                    
                    sample_motion_3D, _ = self.model_dlow.inference(mode = 'infer', sample_num = self.sample_number)
            
                pred = sample_motion_3D.detach().cpu().numpy()
                torch.cuda.empty_cache()
                if pred.shape[1] == len(Pred_agents):
                    Pred[:,:,Index] = pred[:,Pred_agents]
                elif pred.shape[1] == Pred_agents.sum():
                    Pred[:,:,Index] = pred
                else:
                    raise TypeError("Something went wrong")
            # Write the results into the pandas dataframe
            for i, ind in enumerate(index_batch):
                for i_agent, agent in enumerate(Agents):
                    if Pred_agents[i_agent]:
                        Output_Path.iloc[ind][agent] = Pred[i, i_agent, :self.num_samples_path_pred, 
                                                            :self.num_timesteps_out_test[ind]].astype('float32')
                    
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

        
        
        
        
        
    
        
        
      
