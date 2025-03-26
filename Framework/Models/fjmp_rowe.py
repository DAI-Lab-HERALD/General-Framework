# import horovod.torch as hvd
import numpy as np
import os
import pickle
import random
import sys
import time
import torch

from pathlib import Path

try:
    import horovod.torch as hvd
    can_use_hvd = True
except:
    can_use_hvd = False

from mpi4py import MPI
comm = MPI.COMM_WORLD

from model_template import model_template

from FJMP.fjmp_dataloader_utils import get_obj_feats
from FJMP.fjmp import FJMP
from FJMP.fjmp_utils import Logger, gpu, from_numpy


object_type_dict = {
    'V':0,
    'P':1,
    'M':2,
    'B':3,
    'other': -1,
    '0': -1
}

class fjmp_rowe(model_template):
    '''
    This is the implementation of the joint prediction model FJMP. 
    The code was taken from https://github.com/RLuke22/FJMP, and
    the work should be cited as:
        
    Rowe, Luke and Ethier, Martin and Dykhne, Eli-Henry and Czarnecki, Krzysztof
    FJMP: Factorized Joint Multi-Agent Motion Prediction over Learned Directed Acyclic Interaction Graphs
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

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

        kwargs_str = '_train' + str(self.model_kwargs['num_joint_modes'])  + '_pred' + str(self.num_samples_path_pred)
        model_str = 'FJMP' + kwargs_str + '_seed' + str(self.model_kwargs['seed'])

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

        if not ('num_edge_types' in self.model_kwargs.keys()):
            self.model_kwargs["num_edge_types"] = 3
    
        if not ('h_dim' in self.model_kwargs.keys()):
            self.model_kwargs["h_dim"] = 128

        if not ('num_joint_modes' in self.model_kwargs.keys()):
            self.model_kwargs["num_joint_modes"] = 6

        if not ('num_proposals' in self.model_kwargs.keys()):
            self.model_kwargs["num_proposals"] = 15

        if not ('batch_size' in self.model_kwargs.keys()):
            self.model_kwargs["batch_size"] = 16

        if not ('max_epochs' in self.model_kwargs.keys()):
            self.model_kwargs["max_epochs"] = 50 
        
        if not ('lr' in self.model_kwargs.keys()):
            self.model_kwargs["lr"] = 1e-3

        if not ('decoder' in self.model_kwargs.keys()):
            self.model_kwargs["decoder"] = 'dagnn'
    
        if not ('num_heads' in self.model_kwargs.keys()):
            self.model_kwargs["num_heads"] = 1
            
        if not ('learned_relation_header' in self.model_kwargs.keys()):
            self.model_kwargs["learned_relation_header"] = True

        if not ('n_mapnet_layers' in self.model_kwargs.keys()):
            self.model_kwargs["n_mapnet_layers"] = 2
            
        if not ('n_l2a_layers' in self.model_kwargs.keys()):
            self.model_kwargs["n_l2a_layers"] = 2

        if not ('n_a2a_layers' in self.model_kwargs.keys()):
            self.model_kwargs["n_a2a_layers"] = 2
        
        if not ('proposal_coef' in self.model_kwargs.keys()):
            self.model_kwargs["proposal_coef"] = 1
        
        if not ('rel_coef' in self.model_kwargs.keys()):
            self.model_kwargs["rel_coef"] = 100

        if not ('proposal_header' in self.model_kwargs.keys()):
            self.model_kwargs["proposal_header"] = True

        if not ('two_stage_training' in self.model_kwargs.keys()):
            self.model_kwargs["two_stage_training"] = True

        if not ('training_stage' in self.model_kwargs.keys()):
            self.model_kwargs["training_stage"] = 1
            
        if not ('ig' in self.model_kwargs.keys()):
            self.model_kwargs["ig"] = 'sparse'
        
        if not ('focal_loss' in self.model_kwargs.keys()):
            self.model_kwargs["focal_loss"] = True

        if not ('gamma' in self.model_kwargs.keys()):
            self.model_kwargs["gamma"] = 5

        if not ('weight_0' in self.model_kwargs.keys()):
            self.model_kwargs["weight_0"] = 1

        if not ('weight_1' in self.model_kwargs.keys()):
            self.model_kwargs["weight_1"] = 2

        if not ('weight_2' in self.model_kwargs.keys()):
            self.model_kwargs["weight_2"] = 4
            
        if not ('teacher_forcing' in self.model_kwargs.keys()):
            self.model_kwargs["teacher_forcing"] = True

        if not ('scheduled_sampling' in self.model_kwargs.keys()):
            self.model_kwargs["scheduled_sampling"] = True

        if not ('supervise_vehicles' in self.model_kwargs.keys()):
            self.model_kwargs["supervise_vehicles"] = False # seems to be only for INTERACTION

        if not ('train_all' in self.model_kwargs.keys()):
            self.model_kwargs["train_all"] = False
        
        if not ('no_agenttype_encoder' in self.model_kwargs.keys()):
            self.model_kwargs["no_agenttype_encoder"] = False # only for Argoverse 2

        if hasattr(self, 'model_file'):
            if not ('log_path' in self.model_kwargs.keys()):
                self.model_kwargs["log_path"] = Path(self.model_file[:-4] + '_log' + os.sep)

                if not os.path.exists(self.model_kwargs["log_path"]):
                    os.makedirs(self.model_kwargs["log_path"], exist_ok=True)

                log = os.path.join(self.model_kwargs["log_path"], "log")
                # write stdout to log file
                sys.stdout = Logger(log)


        if self.data_set.get_name()['file'] == 'Interaction' or self.data_set.get_name()['file'] == 'Roundabout':
            self.model_kwargs["dataset"] = 'interaction'
        else:
            self.model_kwargs["dataset"] = 'argoverse2'

        if self.model_kwargs["dataset"] == 'interaction':
            self.model_kwargs["switch_lr_1"] = 40
            self.model_kwargs["switch_lr_2"] = 48
            self.model_kwargs["lr_step"] = 1/5
            self.model_kwargs["input_size"] = 5
            self.model_kwargs["prediction_steps"] = self.num_timesteps_out # 30 
            self.model_kwargs["observation_steps"] = self.num_timesteps_in # 10
            # two agent types: "car", and "pedestrian/bicyclist"
            self.model_kwargs["num_agenttypes"] = 2
            # self.model_kwargs['dataset_path'] = 'dataset_INTERACTION'
            # self.model_kwargs['tracks_train_reformatted'] = os.path.join(self.model_kwargs['dataset_path'], 'train_reformatted')
            # self.model_kwargs['tracks_val_reformatted'] = os.path.join(self.model_kwargs['dataset_path'], 'val_reformatted')
            self.model_kwargs['num_scales'] = 4
            self.model_kwargs["map2actor_dist"] = 20.0
            self.model_kwargs["actor2actor_dist"] = 100.0
            # self.model_kwargs['maps'] = os.path.join(self.model_kwargs['dataset_path'], 'maps')
            self.model_kwargs['cross_dist'] = 10
            self.model_kwargs['cross_angle'] = 1 * np.pi
            self.model_kwargs["preprocess"] = True
            self.model_kwargs["val_workers"] = 0
            self.model_kwargs["workers"] = 0

        elif self.model_kwargs["dataset"] == "argoverse2":
            self.model_kwargs["switch_lr_1"] = 32
            self.model_kwargs["switch_lr_2"] = 36
            self.model_kwargs["lr_step"] = 1/10
            self.model_kwargs["input_size"] = 5
            self.model_kwargs["prediction_steps"] = self.num_timesteps_out # 60
            self.model_kwargs["observation_steps"] = self.num_timesteps_in # 50
            self.model_kwargs["num_agenttypes"] = 5
            # self.model_kwargs['dataset_path'] = 'dataset_AV2'
            # self.model_kwargs['files_train'] = os.path.join(self.model_kwargs['dataset_path'], 'train')
            # self.model_kwargs['files_val'] = os.path.join(self.model_kwargs['dataset_path'], 'val')
            self.model_kwargs['num_scales'] = 6 # TODO something is wrong with how I load the dataset probably; this won't work
            self.model_kwargs["map2actor_dist"] = 10.0
            self.model_kwargs["actor2actor_dist"] = 100.0
            self.model_kwargs['cross_dist'] = 6
            self.model_kwargs['cross_angle'] = 0.5 * np.pi
            self.model_kwargs["preprocess"] = True
            self.model_kwargs["val_workers"] = 0
            self.model_kwargs["workers"] = 0
            self.model_kwargs["max_epochs"] = min(36, self.model_kwargs["max_epochs"])
            self.model_kwargs["num_proposals"] = 15
            self.model_kwargs["ig"] = 'dense'
            self.model_kwargs["n_mapnet_layers"] = 4
            self.model_kwargs["focal_loss"] = True
            self.model_kwargs["gamma"] = 5
            self.model_kwargs["weight_0"] = 1
            self.model_kwargs["weight_1"] = 4
            self.model_kwargs["weight_2"] = 4
            self.model_kwargs["learned_relation_header"] = True
            self.model_kwargs["decoder"] = 'dagnn'
            self.model_kwargs["teacher_forcing"] = True

        if self.data_set.get_name()['file'] == 'Roundabout':
            self.model_kwargs["num_agenttypes"] = 4



    def setup_method(self):
        self.define_default_kwargs()

        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = False
        self.can_use_map = False
        self.can_use_graph = True

        seed = self.model_kwargs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def rearange_input_data(self, data):
        # data.shape = bs x num_agents x num_timesteps x 2

        data_old = data.copy()
        assert np.array_equal(self.input_data_type[:2], ['x', 'y']), "Input data type does not start with ['x', 'y']"

        data = np.concatenate([data[...,:2], np.full(data.shape[:3] + (3,), np.nan)], axis = -1)
        missing_timesteps_start = np.isfinite(data[...,:2]).all(-1).argmax(-1) # Num agents x num samples 
        missing_timesteps_end = missing_timesteps_start + np.isfinite(data[...,:2]).all(-1).sum(-1)

        if 'v_x' in self.input_data_type:
            v_x_id = self.input_data_type.index('v_x')
            data[...,2] = data_old[...,v_x_id]
        else:
            for (missing_timestep_start, missing_timestep_end) in np.unique(np.stack([missing_timesteps_start, missing_timesteps_end], -1).reshape(-1,2), axis = 0):
                if np.abs(missing_timestep_end - missing_timestep_start) > 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start:missing_timestep_end, 2] = np.gradient(data[mask, missing_timestep_start:missing_timestep_end, 0], axis = -1)
                elif np.abs(missing_timestep_end - missing_timestep_start) == 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start, 2] = 0
            # Use differentation along 'x' axis to get v_x
            # data[...,2] = np.gradient(data[...,0], axis = -1)
        
        if 'v_y' in self.input_data_type:
            v_y_id = self.input_data_type.index('v_y')
            data[...,3] = data_old[...,v_y_id]
        else:
            for (missing_timestep_start, missing_timestep_end) in np.unique(np.stack([missing_timesteps_start, missing_timesteps_end], -1).reshape(-1,2), axis = 0):
                if np.abs(missing_timestep_end - missing_timestep_start) > 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start:missing_timestep_end, 3] = np.gradient(data[mask, missing_timestep_start:missing_timestep_end, 1], axis = -1)
                elif np.abs(missing_timestep_end - missing_timestep_start) == 1:
                    mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end)
                    data[mask, missing_timestep_start, 3] = 0
            # Use differentation along 'y' axis to get v_y
            # data[...,3] = np.gradient(data[...,1], axis = -1)
        
        if 'theta' in self.input_data_type:
            theta_id = self.input_data_type.index('theta')
            data[...,4] = data_old[...,theta_id]
        else:
            # Use atan2 between v_y and v_x to get theta
            data[...,4] = np.arctan2(data[...,3], data[...,2])

        # Ensure position data is there
        missing_position = np.isnan(data[...,:2]).any(-1)
        data[missing_position] = np.nan

        # Check if there is additional data missing at available position data
        finite_data = np.isfinite(data)
        any_finite = finite_data.any(-1)
        num_any_finite = any_finite.sum()
        num_all_finite = finite_data.all(-1).sum()
        if num_any_finite != num_all_finite:
            print('    Autobot: Rearranging input data failed')
            failed_cases = (num_any_finite - num_all_finite) / num_any_finite
            print('    Autobot: In {:0.2f}% of cases, there where mixed nan/finite values'.format(100 * failed_cases))

            # Check if positions are always available
            assert np.isfinite(data[any_finite][...,:2]).all(), "There are finite values in v_x or v_y where x or y is nan"

        return data



    def extract_batch_data(self, X, T, S, C, Pred_agents, Y = None, graph = None, Sample_id = None, Agent_id = None):
        # X.shape:   bs x num_agents x num_timesteps_is x 2
        # Y.shape:   bs x num_agents x num_timesteps_is x 2
        # T.shape:   bs x num_agents 

        # requires ['x', 'y', 'v_x', 'v_y', 'theta']
        # check again with self.input_data_type with each new framework update
        if not np.array_equal(self.input_data_type, ['x', 'y', 'v_x', 'v_y', 'theta']):
            X = self.rearange_input_data(X)
            if Y is not None:
                Y = self.rearange_input_data(Y)

        
        
        data = {}
        data_feats, data_ctrs, data_orig, data_theta, data_rot = [], [], [], [], []
        data_feat_locs, data_feat_vels, data_feat_psirads = [], [], [],
        data_feat_agenttypes, data_feat_agentcategories, data_feat_shapes =  [], [], []
        data_gt_preds, data_gt_vels, data_gt_psirads, data_has_preds, data_has_obss = [], [], [], [], []
        data_ig_labels_sparse, data_ig_labels_dense, data_ig_labels_m2i = [], [], []
        data_graphs = []

        if C is None:
            C = np.ones(T.shape)
            # Remove missing agents
            C[~np.isfinite(X).all(-1).all(-1)] = 0
            # Make predicted agents have category 2
            C[Pred_agents] = 2

        

        for b in range(X.shape[0]):
            if Y is not None:
                trajs = np.concatenate([X[b,:,:,:2], Y[b,:,:,:2]], axis = 1)

                data['vels'] = np.concatenate([X[b,:,:,2:4], Y[b,:,:,2:4]], axis = 1)

                data['psirads'] = np.concatenate([X[b,:,:,4], Y[b,:,:,4]], axis = 1)

            else:
                trajs = X[b,:,:,:2]
                data['vels'] = X[b,:,:,2:4]
                data['psirads'] = X[b,:,:,4]

            
            # convert array to list with len = num_agents
            data['trajs'] = [trajs[i,~np.isnan(trajs[i,:,0])] for i in range(len(trajs))]
            data['vels'] = [data['vels'][i,~np.isnan(data['vels'][i,:,0])] for i in range(len(data['vels']))]
            data['psirads'] = [data['psirads'][i,~np.isnan(data['psirads'][i,:])][:,np.newaxis] for i in range(len(data['psirads']))]

            steps = np.arange(trajs.shape[1])
            data['steps'] = [steps[~np.isnan(traj[:,0])] for traj in trajs]

            T[T == ''] = '0'
            data['agenttypes'] = [object_type_dict[T[b, t]] * np.ones((len(data['steps'][t]), 1)) for t in range(len(T[b]))]

            data['agentcategories'] = [C[b,c] * np.ones((len(data['steps'][c]), 2)) for c in range(len(C[b]))]

            data['shapes'] = [S[b,s] * np.ones((len(data['steps'][s]), 2)) for s in range(len(S[b]))]

            data['track_ids'] = [Agent_id[b,id] * np.ones((len(data['steps'][id]), 1)) for id in range(len(Agent_id[b]))]


            data = get_obj_feats(train = True, 
                                    data = data, 
                                    idx = b, 
                                    dataset = self.model_kwargs["dataset"], 
                                    num_timesteps_in = self.num_timesteps_in, 
                                    num_timesteps_out = self.num_timesteps_out)
            
            # define scales as the 2^i where i is the number of scales - 1
            # scales = [2**(i+1) for i in range(self.model_kwargs['num_scales']-1)]
            # graph_b = self.add_node_connections(graph=graph[b], scales = scales, device = self.device)
            if graph is not None:
                graph_b = dict(graph[b])

                # process lane centerline in same way as agent trajectories
                centerlines_b = [np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T for centerline in graph_b['centerlines']]
                left_boundary_b = [np.matmul(data['rot'], (left_boundary - data['orig'].reshape(-1, 2)).T).T for left_boundary in graph_b['left_boundaries']]
                right_boundary_b = [np.matmul(data['rot'], (right_boundary - data['orig'].reshape(-1, 2)).T).T for right_boundary in graph_b['right_boundaries']]
                
                ctrs_b = [np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32) for centerline in centerlines_b]
                ctrs_b = np.concatenate(ctrs_b, axis=0)

                feats_b = [np.asarray(centerline[1:] - centerline[:-1], np.float32) for centerline in centerlines_b]
                feats_b = np.concatenate(feats_b, axis=0)

                graph_b['centerlines'] = centerlines_b
                graph_b['left_boundaries'] = left_boundary_b
                graph_b['right_boundaries'] = right_boundary_b
                graph_b['ctrs'] = ctrs_b
                graph_b['feats'] = feats_b


                del graph_b['lane_type']

                graph_b = from_numpy(graph_b)
                graph_b['left'] = graph_b['left'][0]
                graph_b['right'] = graph_b['right'][0]
            else:
                graph_b = None

            data['graph'] = graph_b

            data_feats.append(torch.tensor(data['feats']))#.to(self.device))
            data_ctrs.append(torch.tensor(data['ctrs']))#.to(self.device))
            data_orig.append(torch.tensor(data['orig']))#.to(self.device))
            data_theta.append(data['theta'])
            data_rot.append(torch.tensor(data['rot']))#.to(self.device))
            data_feat_locs.append(torch.tensor(data['feat_locs']))#.to(self.device))
            data_feat_vels.append(torch.tensor(data['feat_vels']))#.to(self.device))
            data_feat_psirads.append(torch.tensor(data['feat_psirads']))#.to(self.device))
            data_feat_agenttypes.append(torch.tensor(data['feat_agenttypes']))#.to(self.device))
            if 'feat_agentcategories' in data.keys():
                data_feat_agentcategories.append(torch.tensor(data['feat_agentcategories']))#.to(self.device))
            if 'feat_shapes' in data.keys():
                data_feat_shapes.append(torch.tensor(data['feat_shapes']))#.to(self.device))
            data_gt_preds.append(torch.tensor(data['gt_preds']))#.to(self.device))
            data_gt_vels.append(torch.tensor(data['gt_vels']))#.to(self.device))
            data_gt_psirads.append(torch.tensor(data['gt_psirads']))#.to(self.device))
            data_has_preds.append(torch.tensor(data['has_preds']))#.to(self.device))
            data_has_obss.append(torch.tensor(data['has_obss']))#.to(self.device))
            data_ig_labels_sparse.append(torch.tensor(data['ig_labels_sparse']))#.to(self.device))
            data_ig_labels_dense.append(torch.tensor(data['ig_labels_dense']))#.to(self.device))
            if 'ig_labels_m2i' in data.keys():
                data_ig_labels_m2i.append(torch.tensor(data['ig_labels_m2i']))#.to(self.device))
            data_graphs.append(data['graph'])

        data['idx'] = Sample_id.tolist()
        data['feats'] = data_feats
        data['ctrs'] = data_ctrs
        data['orig'] = data_orig
        data['theta'] = data_theta
        data['rot'] = data_rot
        data['feat_locs'] = data_feat_locs
        data['feat_vels'] = data_feat_vels
        data['feat_psirads'] = data_feat_psirads
        data['feat_agenttypes'] = data_feat_agenttypes
        if 'feat_agentcategories' in data.keys():
            data['feat_agentcategories'] = data_feat_agentcategories
        if 'feat_shapes' in data.keys():
            data['feat_shapes'] = data_feat_shapes
        data['gt_preds'] = data_gt_preds
        data['gt_vels'] = data_gt_vels
        data['gt_psirads'] = data_gt_psirads
        data['has_preds'] = data_has_preds
        data['has_obss'] = data_has_obss
        data['ig_labels_sparse'] = data_ig_labels_sparse
        data['ig_labels_dense'] = data_ig_labels_dense
        if 'ig_labels_m2i' in data.keys():
            data['ig_labels_m2i'] = data_ig_labels_m2i
        data['graph'] = data_graphs


        del data['trajs']
        del data['steps']
        del data['vels']
        del data['psirads']
        del data['agenttypes']
        del data['agentcategories']
        del data['track_ids']
        del data['is_valid_agent']

            
        return data
    
    def train_fjmp(self, start_epoch = 1, optimizer = None):
        

        m = sum(p.numel() for p in self.model.parameters())

        print("Model: {} parameters".format(m))
        print("Training model...")

        # save stage 1 config
        if self.model.two_stage_training and self.model.training_stage == 1:
            if can_use_hvd:
                if hvd.rank() == 0:
                    with open(os.path.join(self.model_kwargs["log_path"], "config_stage_1.pkl"), "wb") as f:
                        pickle.dump(self.model_kwargs, f)
            else:
                with open(os.path.join(self.model_kwargs["log_path"], "config_stage_1.pkl"), "wb") as f:
                    pickle.dump(self.model_kwargs, f)

        # hvd.broadcast_parameters(self.state_dict(), root_rank=0)
        # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        for epoch in range(start_epoch, self.model_kwargs['max_epochs'] + 1):   
            print("Epoch/Total: {}/{} ".format(epoch, self.model_kwargs['max_epochs']))
            # this shuffles the training set every epoch         
            # train_loader.sampler.set_epoch(int(epoch))
            
            # t_start_epoch = time.time()
            self.model.train()
            
            if self.model_kwargs['scheduled_sampling']:
                prop_ground_truth = 1 - (epoch - 1) / (self.model_kwargs['max_epochs'] - 1)   
            elif self.model_kwargs['teacher_forcing']:
                prop_ground_truth = 1.  
            else:
                prop_ground_truth = 0. 
            
            # set learning rate accordingly
            for e, param_group in enumerate(optimizer.param_groups):
                if epoch == self.model_kwargs['switch_lr_1'] or epoch == self.model_kwargs['switch_lr_2']:
                    param_group["lr"] = param_group["lr"] * (self.model_kwargs['lr_step'])
                
                if e == 0:
                    cur_lr = param_group["lr"]  
            
            tot = 0
            # accum_gradients = {}

            train_epoch_done = False
            i = 0

            while not train_epoch_done:
                X, Y, T, S, C, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, train_epoch_done = self.provide_batch_data('train', self.model_kwargs['batch_size'], 
                                                                                    val_split_size = 0.1, return_categories=True)  
                # X.shape:   bs x num_agents x num_timesteps_is x 2
                # Y.shape:   bs x num_agents x num_timesteps_is x 2
                # T.shape:   bs x num_agents 

                # X[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = np.nan
                # T[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = '0'
                # Y[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = np.nan

                # if S is not None:
                #     S[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = np.nan

                # if C is not None:
                #     C[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = 4


                data = self.extract_batch_data(X=X, T=T, S=S, C=C, Pred_agents = Pred_agents, Y=Y, graph=graph, Sample_id=Sample_id, Agent_id=Agent_id) # TODO check if correct

                # graph_id = self.data_set.Domain.graph_id.iloc[Sample_id].values
                # get data dictionary for processing batch
                dd = self.model.process(data)

                dgl_graph = self.model.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd["agenttypes"], dd['world_locs'], dd['has_preds']).to(self.device)
                # only process observed features
                dgl_graph = self.model.feature_encoder(dgl_graph, dd['feats'][:,:self.num_timesteps_in], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                if self.model_kwargs['two_stage_training'] and self.model_kwargs['training_stage'] == 2:
                    stage_1_graph = self.model.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.num_timesteps_in], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
                else:
                    stage_1_graph = None

                dd = {key:gpu(_data) for key,_data in dd.items()}
              
                ig_dict = {}
                ig_dict["ig_labels"] = dd["ig_labels"] 
                
                # produces dictionary of results
                res = self.model.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=prop_ground_truth)

                loss_dict = self.model.get_loss(dgl_graph, dd['batch_idxs'], res, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'], dd["ig_labels"], epoch)
                
                loss = loss_dict["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                # accum_gradients = accumulate_gradients(accum_gradients, self.model.named_parameters())
                optimizer.step()
                
                if i % 100 == 0:
                    print("Training data: ", "{:.2f}%".format(i * 100), "lr={:.3e}".format(cur_lr), "rel_coef={:.1f}".format(self.model_kwargs['rel_coef']),
                        "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_dict.items()]))

                i = i + 1
                tot += dd['batch_size']
                torch.cuda.empty_cache()
                

            if (not self.model_kwargs['two_stage_training']) or (self.model_kwargs['two_stage_training'] and self.model_kwargs['training_stage'] == 2):
                print("Saving model")
                self.model.save(epoch, optimizer, None, None, None)
            else:
                print("Saving relation header")
                self.model.save_relation_header(epoch, optimizer, None)

            pickle.dump(epoch, open(os.path.join(self.model_kwargs["log_path"], "epoch.pkl"), "wb"))

            # else:
            self.model.eval()

            i = 0
            val_epoch_done = False
            while not val_epoch_done:
                X, Y, T, S, C, img, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, val_epoch_done = self.provide_batch_data('val', self.model_kwargs['batch_size'], 
                                                                                        val_split_size = 0.1, return_categories=True)                

                # X[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = np.nan
                # T[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = '0'
                # Y[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = np.nan

                # if S is not None:
                #     S[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = np.nan

                # if C is not None:
                #     C[(np.isnan(X).any((2,3)) | np.isnan(Y).any((2,3)))] = 4

                data = self.extract_batch_data(X=X, T=T, S=S, C=C, Pred_agents = Pred_agents, Y=Y, graph=graph, Sample_id=Sample_id, Agent_id=Agent_id) # TODO check if correct

                # graph_id = self.data_set.Domain.graph_id.iloc[Sample_id].values
                # get data dictionary for processing batch
                dd = self.model.process(data)

                dgl_graph = self.model.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd["agenttypes"], dd['world_locs'], dd['has_preds']).to(self.device)
                # only process observed features
                dgl_graph = self.model.feature_encoder(dgl_graph, dd['feats'][:,:self.num_timesteps_in], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                if self.model_kwargs['two_stage_training'] and self.model_kwargs['training_stage'] == 2:
                    stage_1_graph = self.model.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.num_timesteps_in], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
                else:
                    stage_1_graph = None

                dd = {key:gpu(_data) for key,_data in dd.items()}
              
                ig_dict = {}
                ig_dict["ig_labels"] = dd["ig_labels"] 
                
                # produces dictionary of results
                res = self.model.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=prop_ground_truth)

                val_loss = self.model.get_loss(dgl_graph, dd['batch_idxs'], res, dd['agenttypes'], dd['has_preds'], dd['gt_locs'], dd['batch_size'], dd["ig_labels"], epoch)
                
                if i % 100 == 0:
                    print("Validation data: ", "{:.2f}%".format(i * 100), "lr={:.3e}".format(cur_lr), "rel_coef={:.1f}".format(self.model_kwargs['rel_coef']),
                        "\t".join([k + ":" + f"{v.item():.3f}" for k, v in val_loss.items()]))
                    
                torch.cuda.empty_cache()
            


        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

        if self.model_kwargs['two_stage_training'] and self.model_kwargs['training_stage'] == 1:
            pickle.dump(self.model, open(self.model_file[:-4] + '_stage_1', 'wb'))
        else:
            pickle.dump(self.model, open(self.model_file[:-4] + '_stage_2', 'wb'))
    
    
    def train_method(self):
        # TODO: Implement

        # Allow for finetuning of model
        if not hasattr(self, 'model'):
            self.model = FJMP(self.model_kwargs)
        else:
            self.model_kwargs['training_stage'] = 1

        if os.path.exists(os.path.join(self.model_kwargs["log_path"], "epoch.pkl")):
            start_epoch = pickle.load(open(os.path.join(self.model_kwargs["log_path"], "epoch.pkl"), "rb"))
        else:
            start_epoch = 0
                
        # initialize optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.model.learning_rate)
        if can_use_hvd:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=self.model.named_parameters()
            ) 


        # Check if stage 1 model exists
        if self.model_kwargs['two_stage_training']:
            print('')
            print("Check if stage 1 exists")
            if self.model_kwargs['training_stage'] == 1 and not os.path.exists(self.model_file[:-4] + '_stage_1'):
                if os.path.exists(str(self.model_kwargs["log_path"]) + 'best_model_relation_header.pt'):
                    self.model.load_relation_header()
                print("Training stage 1")
                self.train_fjmp(start_epoch+1, optimizer)

                start_epoch = 0
                pickle.dump(start_epoch, open(os.path.join(self.model_kwargs["log_path"], "epoch.pkl"), "wb"))

            if self.model_kwargs['training_stage'] == 2 and not os.path.exists(self.model_file[:-4] + '_stage_1'):
                raise ValueError('Stage 1 model does not exist')
            
            if os.path.exists(self.model_file[:-4] + '_stage_1'):
                self.model_kwargs['training_stage'] = 2
                self.model.training_stage = 2
                
                print('Load stage 1 model')
                with open(os.path.join(self.model_kwargs["log_path"], "config_stage_1.pkl"), "rb") as f:
                    config_stage_1 = pickle.load(f) 
                
                self.model = FJMP(self.model_kwargs)
                
                pretrained_relation_header = FJMP(config_stage_1)
                self.model.prepare_for_stage_2(pretrained_relation_header)

                # initialize optimizer
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.model.learning_rate)
                if can_use_hvd:
                    optimizer = hvd.DistributedOptimizer(
                        optimizer, named_parameters=self.model.named_parameters()
                    ) 

                if os.path.exists(str(self.model_kwargs["log_path"]) + 'best_model.pt'):
                    self.model.load_for_train(optimizer)

                print('')
                print("Training stage 2")
                self.train_fjmp(start_epoch+1, optimizer)
            
            else:
                raise ValueError('Stage 1 model does not exist')
        

        self.train_loss = np.ones((1, self.model_kwargs['max_epochs'])) * np.nan

        self.weights_saved = []
        


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
        if os.path.exists(self.model_file[:-4] + '_stage_2'):
            self.model = pickle.load(open(self.model_file[:-4] + '_stage_2', 'rb'))
            self.model_kwargs['training_stage'] = 2
        else:
            raise ValueError('Model file does not exist')

    def predict_method(self):
        prediction_done = False

        # hvd.broadcast_parameters(self.state_dict(), root_rank=0)

        self.model.eval()
        # validation results

        if self.model_kwargs['proposal_header']:
            proposals_all = []
        
        if self.model_kwargs['learned_relation_header']:
            ig_preds = []
            ig_labels_all = []   

        self.model_kwargs['training_stage'] = 2
        
        
        tot = 0
        with torch.no_grad():
            # tot_log = self.num_val_samples // (self.batch_size * hvd.size())            
            while not prediction_done:
                X, T, S, C, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.model_kwargs['batch_size'], return_categories=True)
                
                # X[np.isnan(X).any((2,3))] = np.nan
                # T[np.isnan(X).any((2,3))] = '0'

                # if S is not None:
                #     S[np.isnan(X).any((2,3))] = np.nan

                # if C is not None:
                #     C[np.isnan(X).any((2,3))] = 4
                
                data =  self.extract_batch_data(X=X, T=T, S=S, C=C, Pred_agents = Pred_agents, Y=None, graph=graph, Sample_id=Sample_id, Agent_id=Agent_id)
        
                dd = self.model.process(data)
                
                dgl_graph = self.model.init_dgl_graph(dd['batch_idxs'], dd['ctrs'], dd['orig'], dd['rot'], dd['agenttypes'], dd['world_locs'], dd['has_preds']).to(self.device)
                dgl_graph = self.model.feature_encoder(dgl_graph, dd['feats'][:,:self.num_timesteps_in], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])

                if self.model.two_stage_training and self.model.training_stage == 2:
                    stage_1_graph = self.model.build_stage_1_graph(dgl_graph, dd['feats'][:,:self.num_timesteps_in], dd['agenttypes'], dd['actor_idcs'], dd['actor_ctrs'], dd['lane_graph'])
                else:
                    stage_1_graph = None


                dd = {key:gpu(_data) for key,_data in dd.items()}

                ig_dict = {}
                ig_dict["ig_labels"] = dd["ig_labels"]

                # map proposals to Predictions according to Pred_agents
                sample_number = self.model_kwargs['num_joint_modes']
                
                # OOM protection
                splits = int(np.ceil((self.num_samples_path_pred / sample_number)))
                
                num_samples_path_pred_max = int(sample_number * splits)
                Pred = np.zeros((X.shape[0], X.shape[1], num_samples_path_pred_max, self.num_timesteps_out, 2))

                # for i in range(splits):
                #     res = self.model.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=0.)


                #     Index = np.arange(i * sample_number, min((i + 1) * sample_number, num_samples_path_pred_max))

                #     predicted_agents = np.isfinite(X)[:,:,0,0]

                #     pred = np.zeros((X.shape[0], X.shape[1], sample_number, self.num_timesteps_out, 2))
                #     pred[predicted_agents] = res["loc_pred"].detach().cpu().numpy().transpose(0, 2, 1, 3)
                #     Pred[:, :,Index] = pred

                res = self.model.forward(dd["scene_idxs"], dgl_graph, stage_1_graph, ig_dict, dd['batch_idxs'], dd["batch_idxs_edges"], dd["actor_ctrs"], prop_ground_truth=0.)


                # Index = np.arange(i * sample_number, min((i + 1) * sample_number, num_samples_path_pred_max))

                predicted_agents = np.isfinite(X)[:,:,-1,0]

                pred = np.zeros((X.shape[0], X.shape[1], sample_number, self.num_timesteps_out, 2))
                pred[predicted_agents] = res["loc_pred"].detach().cpu().numpy().transpose(0, 2, 1, 3)
                pred = np.tile(pred, (1, 1, splits, 1, 1))
                Pred = pred
                # Pred[:, :,Index] = pred
                                
                torch.cuda.empty_cache()

                Pred = Pred[:, :, :self.num_samples_path_pred]
                
                # save predictions
                self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)



