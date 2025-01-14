import math
import numpy as np
import os
import random
import torch
# import torch.distributed as dist

# from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from model_template import model_template

from ADAPT.model.adapt import ADAPT
from ADAPT.utils.get_model import save_model
from ADAPT.utils.utils import rotate




class adapt_aydemir(model_template):
    '''
    This is the implementation of the joint prediction model ADAPT. 
    The code was taken from https://github.com/gorkaydemir/ADAPT/tree/main, and
    the work should be cited as:
        
    Aydemir, G\"orkay and Akan, Adil Kaan and G\"uney, Fatma
    ADAPT: Efficient Multi-Agent Trajectory Prediction with Adaptation
    In International Conference on Computer Vision.

    '''
    def get_name(self = None):
        r'''
        Provides a dictionary with the different names of the model
            
        Returns
        -------
        names : dict
        The first key of names ('print')  will be primarily used to refer to the model in console outputs. 
                
        The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
        (for any operating system), as it is mostly used to indicate that certain result files belong to this model. 
                
        The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
        latex commands - such as using '$$' for math notation.
            
        '''
        self.define_default_kwargs()

        kwargs_str = ''
        seed_str = str(self.model_kwargs['seed'])

        model_str = 'ADAPT' + kwargs_str + '_seed' + seed_str

        names = {'print': model_str,
                'file': model_str,
                'latex': r'\emph{%s}' % model_str}

        return names
  
    def requires_torch_gpu(self = None):
        r'''
        If True, then the model will use pytorch on the gpu.
            
        Returns
        -------
        pytorch_decision : bool
            
        '''
        return True

    def get_output_type(self = None):
        r'''
        This returns a string with the output type:
        The possibilities are:
        'path_all_wo_pov' : This returns the predicted trajectories of all agents except the pov agent (defined
        in scenario), if this is for example assumed to be an AV.
        'path_all_wi_pov' : This returns the predicted trajectories of all designated agents, including the
        pov agent.
        'class' : This returns the predicted probability that some class of behavior will be observable
        in the future.
        'class_and_time' : This predicted both the aforementioned probabilities, as well as the time at which
        the behavior will become observable.
            
        Returns
        -------
        output_type : str
            
        '''
        return 'path_all_wi_pov'

    def check_trainability_method(self):
        r'''
        This function potentially returns reasons why the model is not applicable to the chosen scenario.
            
        Returns
        -------
        reason : str
        This str gives the reason why the model cannot be used in this instance. If the model is usable,
        return None instead.
            
        '''
        return None
    
    def define_default_kwargs(self):
        if not ('seed' in self.model_kwargs.keys()):
            self.model_kwargs["seed"] = 42
        
        # Go through the cfg file in autobot.cfg.autobot.yaml
        # Check if the key is in the model_kwargs, if not, add it, and set 
        # the value to the one in the .yaml file
        
        # Get path to this file
        path = os.path.dirname(os.path.abspath(__file__))

        # Load yaml file
        if not hasattr(self, 'cfg'):
            cfg_path = path + os.sep + 'ADAPT' + os.sep + 'cfg' + os.sep + 'adapt.yaml'
            self.cfg = OmegaConf.load(cfg_path)

        # Go through the cfg file
        for key in self.cfg.keys():
            if not (key in self.model_kwargs.keys()):
                self.model_kwargs[key] = self.cfg[key]
            
            else:
                self.cfg[key] = self.model_kwargs[key]

            
        

        

        

    def setup_method(self):
        self.define_default_kwargs()

        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = False
        self.can_use_map = False
        self.can_use_graph = True
        self.sceneGraph_radius = self.model_kwargs['max_distance']

        seed = self.model_kwargs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


        # add required keys to config file
        self.cfg['past_len'] = self.num_timesteps_in
        self.cfg['future_len'] = self.num_timesteps_out

        # Check if map is available
        self.cfg['use_graph'] = self.can_use_graph and self.has_graph

        # set checkpoint path
        self.cfg['checkpoint_path'] = self.model_file[:-4] + os.sep + 'checkpoint.pt'

        # set model save path
        self.cfg['model_save_path'] = self.model_file[:-4]

        # get number of possible config types
        # self.cfg['num_agent_types'] = 5 # Pedestrian, Cyclist, Motorcylce, Vehicle, None




    def extract_data(self, X, Y, graph, Pred_agents, Sample_id, Agent_id):
        # X.shape = (batch_size, num_agents, num_steps_in, num_features)
        # Y.shape = (batch_size, num_agents, num_steps_out, num_features)
        # T.shape = (batch_size, num_agents)
        # S.shape = (batch_size, num_agents)
        # Pred_agents.shape = (batch_size, num_agents)

        # TODO

        batch = []
        entries = 0

        for i in range(X.shape[0]):
            random_scale = 1.0
            if self.cfg['scaling']:
                random_scale = 0.75 + random.random()*0.5

            if Y is None:
                target_agents = [0]
            else:
                target_agents = np.where(Pred_agents[i])[0]

            for j in target_agents:
                sample_id_i = Sample_id[i].copy()
                agent_id_i = Agent_id[i].copy()
                
                batch.append({})

                pos_matrix = X[i,:,:,:2].copy()

                if graph is not None:
                    batch_graph = graph[i].copy()

                if Y is not None:
                    label_matrix = Y[i,:,:,:2].copy()
                    label_matrix_with_nans = Y[i,:,:,:2].copy()


                span = X[i, j, -6:, :2].copy()
                interval = 2

                # Select only the first and second columns (angles as [der[0], der[1]])
                angles = span[interval:] - span[:-interval]
                
                angles = np.array(angles)
                der_x, der_y = np.mean(angles, axis=0)
                angle = -math.atan2(der_y, der_x) + math.radians(90)


                # normalize the pos_matrix around the target agent
                rot_matrix = np.array([[math.cos(angle), math.sin(angle)],
                           [-math.sin(angle), math.cos(angle)]])
                
                pos_matrix -= X[i, j, -1, :2]
                pos_matrix = np.matmul(pos_matrix, rot_matrix)

                if Y is not None:
                    label_matrix -= X[i, j, -1, :2]
                    label_matrix[np.isnan(label_matrix)] = -666
                    label_matrix = np.matmul(label_matrix, rot_matrix)

                    label_matrix[:, :, :2] *= random_scale

                # tensor consists of prev_x prev_y x y timestamp type==AV type==AGENT type==OTHERS agent_id timestep and padding until 128
                tensor = np.zeros((X.shape[1], X.shape[2]-1, 128))
                tensor[:, :, :2] = pos_matrix[:, :-1, :2] # prev_x prev_y
                tensor[:, :, 2:4] = pos_matrix[:, 1:, :2] # x y
                tensor[:, :, 4] = np.tile((np.arange(X.shape[2]-1)+1) * self.dt, (X.shape[1], 1)) # timestamp
                
                tensor[:, :, 5] = np.tile(~self.data_set.Not_pov_agent[[sample_id_i]][:, agent_id_i], (X.shape[2]-1, 1)).T # TODO check if pov agent


                tensor[j, :, 6] = 1 # type==target agent
                tensor[:, :, 7] = 1 # type==others
                tensor[j, :, 7] = 0 # type==others (target agent is not others)
                tensor[:, :, 8] = np.tile(np.arange(X.shape[1]), (X.shape[2]-1, 1)).T # agent_id
                tensor[:, :, 9] = np.tile((np.arange(X.shape[2]-1)+1), (X.shape[1], 1)) # timestep

                if Y is not None:
                    tensor[:, :, :4] *= random_scale

                if Y is not None:
                    complete_trajectory = np.concatenate([X[i, :, :, :2], Y[i, :, :, :2]], axis=1)
                    complete_trajectory_mask = np.isfinite(complete_trajectory).all(-1).all(-1) # shape should be agents
                    missing_agent_mask = np.isnan(complete_trajectory).all(-1).all(-1) # shape should be agents

                    displacement = torch.norm((torch.tensor(tensor[:, -1, 2:4]) - torch.tensor(tensor[:, 0, :2])), dim=-1)
                    drop_candidates = (displacement < 1.0) & (self.cfg['static_agent_drop'])

                    drop = np.zeros(X.shape[1], dtype=bool)
                    drop[drop_candidates] = np.random.rand(drop_candidates.sum()) < 0.1
                    drop[j] = False
                    drop[missing_agent_mask] = True

                    moving = torch.norm(torch.tensor(pos_matrix[:, -1]) - torch.tensor(pos_matrix[:, 0]), dim=-1) > 6.0
                    moving[j] = True # in original code, moving is always True for 0th agent which seems to be the target agent

                    tensor = tensor[~drop]
                    label_matrix = label_matrix[~drop]
                    label_matrix_with_nans = label_matrix_with_nans[~drop]

                    complete_trajectory_mask = complete_trajectory_mask[~drop]
                    moving = moving[~drop]
                    missing_agent_mask = missing_agent_mask[~drop]

                    batch[entries]['consider'] = torch.tensor(np.where(complete_trajectory_mask & moving.numpy())[0])

                else:
                    batch[entries]['consider'] = torch.tensor(np.where(Pred_agents[i])[0])


                batch[entries]['cent_x'] = X[i, j, -1, 0].copy()
                batch[entries]['cent_y'] = X[i, j, -1, 1].copy()

                batch[entries]['angle'] = angle

                existing_timestep_mask = ~np.isnan(tensor).any(-1)
                batch[entries]['agent_data'] = [torch.tensor(tensor[ag][existing_timestep_mask[ag]]).float() for ag in range(tensor.shape[0]) if len(torch.tensor(tensor[ag][existing_timestep_mask[ag]]).float())>0]

                lane_list = []
                hist_len = X.shape[2] 

                if graph is not None:
                    for l in range(len(batch_graph['centerlines'])):
                        lane = batch_graph['centerlines'][l].copy()

                        lane -= X[i, j, -1, :2]
                        lane = np.matmul(lane, rot_matrix)
                        lane = lane[:hist_len]

                        # lane vector
                        # [..., y, x, pre_y, pre_x]
                        lane_vector = np.zeros((lane.shape[0]-1, 128))
                        lane_vector[:, -4] = lane[1:, 1]
                        lane_vector[:, -3] = lane[1:, 0]
                        lane_vector[:, -2] = lane[:-1, 1]
                        lane_vector[:, -1] = lane[:-1, 0]

                        lane_vector[:,-5] = 1
                        lane_vector[:,-6] = np.arange(lane.shape[0]-1)+1
                        lane_vector[:,-7] = l + len(batch[entries]['agent_data']) # lane id, ensuring that there is no overlap with agent ids
                        lane_vector[:,-8] = -1 # has traffic control 
                        lane_vector[:,-9] = 0  # turn direction
                        lane_vector[:,-10] = 1 if batch_graph['lane_type'][l][1] else -1  # is intersection

                        point_pre_pre = np.zeros((lane.shape[0]-1, 2))
                        point_pre_pre[0] = [2 * lane_vector[0, -1] - lane_vector[0, -3], 2 * lane_vector[0, -2] - lane_vector[0, -4]]
                        point_pre_pre[1:] = lane[:-2]
                        
                        lane_vector[:,-17] = point_pre_pre[:, 0]
                        lane_vector[:,-18] = point_pre_pre[:, 1]

                        if Y is not None:
                            lane_vector[:, -4:] *= random_scale
                            lane_vector[:, -18:-16] *= random_scale

                        lane_list.append(torch.tensor(lane_vector).float())

                batch[entries]['lane_data'] = lane_list

                if Y is not None:
                    batch[entries]['labels'] = torch.tensor(label_matrix[:, :, :2]).float()
                    batch[entries]['labels'] = batch[entries]['labels'].transpose(1, 0) # shape should be timestep x agents x 2

                    batch[entries]['origin_labels'] = torch.tensor(Y[i, j, :, :2]).float() # shape should be timestep x 2

                    batch[entries]['label_is_valid'] = torch.tensor(np.isfinite(label_matrix_with_nans[:, :, 0])).float()
                    batch[entries]['label_is_valid'] = batch[entries]['label_is_valid'].transpose(1, 0) # shape should be timestep x agents


                    assert batch[entries]['consider'][-1] <= batch[entries]['labels'].shape[1] - 1

                else:
                    missing_agent_mask = np.isnan(X[i,:,:,:2]).all(-1).all(-1) # shape should be agents

                    batch[entries]['labels'] = torch.zeros((self.num_timesteps_out, (~missing_agent_mask).sum(), 2)).float()
                    batch[entries]['origin_labels'] = torch.zeros((self.num_timesteps_out, 2)).float()
                    batch[entries]['label_is_valid'] = torch.zeros((self.num_timesteps_out, (~missing_agent_mask).sum())).float()

                dpos = tensor[:, -1, 2:4] - tensor[:, -1, :2]
                degree = torch.atan2(torch.tensor(dpos[:,1]), torch.tensor(dpos[:,0])).numpy()
                degree = degree[~missing_agent_mask]
                x = tensor[~missing_agent_mask, -1, 2]
                y = tensor[~missing_agent_mask, -1, 3]
                pre_x = tensor[~missing_agent_mask, -1, 0]
                pre_y = tensor[~missing_agent_mask, -1, 1]
                info = torch.tensor(
                    np.array([degree, x, y, pre_x, pre_y]))#.unsqueeze(dim=0)
                batch[entries]['meta_info'] = info.transpose(1, 0).float()

                entries += 1

        return batch

# batch has 12 dict entries
# "agent_data"
# "lane_data"
# "city_name"
# "file_name"
# "origin_labels"
# "labels"
# "label_is_valid"
# "consider"
# "cent_x"
# "cent_y"
# "angle"
# "meta_info"

    

    def train_method(self):
        if self.device == 'cpu':
            self.cfg['device'] = 'cpu'
        else:
            self.cfg['device'] = 0
        # dist.init_process_group("nccl", rank=0, world_size=0)

        
        self.model = ADAPT(self.cfg)

        if self.cfg['use_checkpoint'] and os.path.exists(self.cfg['checkpoint_path']):
            checkpoint = torch.load(self.cfg['checkpoint_path'])
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.to(self.device)

        # self.model = DDP(self.model, device_ids=[0], find_unused_parameters=True)

        start_epoch = 0
        iter_num = int(self.data_set.Pred_agents_pred[self.splitter.Train_index].sum() /self.cfg['batch_size'] * 0.9) # int(self.Pred_agents.sum()/self.cfg['batch_size']) #100000

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])
        total_cycle = self.cfg['epoch'] * iter_num # TODO add when total num of batches known
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_cycle * 0.7), int(total_cycle * 0.9)], gamma=0.15)

        if self.cfg['use_checkpoint'] and os.path.exists(self.cfg['checkpoint_path']):
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1

        total_loss = 0.0

        for step in range(start_epoch, self.cfg['epoch']):

            epoch_string = 'Train Model: Epoch ' + str(step + 1).zfill(len(str(self.cfg['epoch'])))

            train_epoch_done = False
            batch = 0
            while not train_epoch_done:
                batch += 1
                print('', flush = True)
                print(epoch_string + ' - Batch {}'.format(batch), flush = True)

                X, Y, _, _, _, _, graph, Pred_agents, _, Sample_id, Agent_id, train_epoch_done = self.provide_batch_data('train', self.cfg['batch_size'], 
                                                                                        val_split_size = 0.1)
                
                batch_data = self.extract_data(X=X, Y=Y, graph=graph, Pred_agents=Pred_agents, Sample_id=Sample_id, Agent_id=Agent_id)

                if len(batch_data) >= 2*self.cfg['batch_size']:
                    # break batch into smaller batches of size batch_size
                    for i in range(0, len(batch_data), self.cfg['batch_size']):
                        traj_loss = self.model(batch_data[i:i+self.cfg['batch_size']])
                        loss = traj_loss
                        total_loss += loss.item()

                        loss.backward()
                        lr = optimizer.state_dict()['param_groups'][0]['lr']
                        # loss_desc = f"lr = {lr:.6f} loss = {total_loss/(step+1):.5f}"
                        loss_desc = f"lr = {lr:.6f} loss = {loss:.5f}"
                        print(loss_desc, flush = True)

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                else:

                    traj_loss = self.model(batch_data)
                    loss = traj_loss
                    total_loss += loss.item()

                    loss.backward()
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    # loss_desc = f"lr = {lr:.6f} loss = {total_loss/(step+1):.5f}"
                    loss_desc = f"lr = {lr:.6f} loss = {loss:.5f}"
                    print(loss_desc, flush = True)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            save_model(self.cfg, step, self.model, optimizer, scheduler)

        self.weights_saved = []

        #     dist.barrier()
        # dist.destroy_process_group()

        


    def save_params_in_csv(self = None):
        r'''
        If True, then the model's parameters will be saved to a .csv file for easier access.
            
        Returns
        -------
        csv_decision : bool
            
        '''
        return False

    def provides_epoch_loss(self = None):
        r'''
        If True, then the model's epoch loss will be saved.
            
        Returns
        -------
        loss_decision : bool
            
        '''
        return True

    def load_method(self):

        if self.device == 'cpu':
            self.cfg['device'] = 'cpu'
        else:
            self.cfg['device'] = 0

        self.model = ADAPT(self.cfg)

        if self.cfg['use_checkpoint']:
            assert os.path.exists(self.cfg['checkpoint_path'])
            checkpoint = torch.load(self.cfg['checkpoint_path'])
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        self.model.to(self.device)
        

    def predict_method(self):
        prediction_done = False
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():     
            ind_batch = 0    
            while not prediction_done:
                ind_batch = ind_batch + 1
                print('    ADAPT: Predicting batch {}'.format(ind_batch))

                X, _, _, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.cfg['batch_size'])

                batch_data = self.extract_data(X, None, graph, Pred_agents, Sample_id, Agent_id)
                # TODO
                pred_trajectory, pred_probs, multi_out = self.model(batch_data, True)


                # map proposals to Predictions according to Pred_agents
                sample_number = self.cfg['num_modes']
                
                # OOM protection
                splits = int(np.ceil((self.num_samples_path_pred / sample_number)))
                
                num_samples_path_pred_max = int(sample_number * splits)
                Pred = np.zeros((X.shape[0], X.shape[1], num_samples_path_pred_max, self.num_timesteps_out, 2)) # Shape (batch_size, num_agents, num_paths, num_steps_out, 2)

                pred = np.zeros((X.shape[0], X.shape[1], sample_number, self.num_timesteps_out, 2))

                for i in range(X.shape[0]):
                    mul_out_x, mul_out_y = rotate(multi_out[i][0][:,:,:,0], multi_out[i][0][:,:,:,1], -batch_data[i]['angle']) # in original code, when handling data they seem to rotate by angle when normalizing
                    multi_out[i][0][:,:,:,0] = mul_out_x
                    multi_out[i][0][:,:,:,1] = mul_out_y
                    pred[i, Pred_agents[i]] = multi_out[i][0].detach().cpu().numpy() + np.array([batch_data[i]['cent_x'], batch_data[i]['cent_y']])

                # pred_probs are save in multi_out[batch][1]


                pred = np.tile(pred, (1, 1, splits, 1, 1))
                Pred = pred
                # Pred[:, :,Index] = pred
                                
                torch.cuda.empty_cache()

                Pred = Pred[:, :, :self.num_samples_path_pred]
                
                # save predictions
                self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)

            print('    ADAPT: Prediction done')
            print('')


