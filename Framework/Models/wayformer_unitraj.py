# import horovod.torch as hvd
import numpy as np
import os
import pickle
import random
import sys
import time
import torch
from omegaconf import OmegaConf

from model_template import model_template

from Wayformer_unitraj.model.wayformer import Wayformer
from torch.distributions import MultivariateNormal


def safe_atan2(y, x):
    Evaluate = (y != 0.0) | (x != 0.0) # Shape (batch_size, num_agents, num_steps_in)
    theta = torch.zeros_like(x) # Shape (batch_size, num_agents, num_steps_in)
    theta[Evaluate] = torch.atan2(y[Evaluate], x[Evaluate]) # Shape (batch_size, num_agents, num_steps_in)
    return theta


class wayformer_unitraj(model_template):
    '''
    This is the version of Wayformer, implemented in the Unitraj prediction framework 
    (https://github.com/vita-epfl/UniTraj/tree/main/unitraj/models/wayformer).
    
    The original paper is cited as:
        
    Nayakanti, N., Al-Rfou, R., Zhou, A., Goel, K., Refaat, K. S., & Sapp, B. (2023, May). 
    Wayformer: Motion forecasting via simple & efficient attention networks. 
    In 2023 IEEE International Conference on Robotics and Automation (ICRA) (pp. 2980-2987).
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
        self.define_default_kwargs(name = True)

        # TODO: Implement
        kwargs_str = ''
        seed_str = str(self.model_kwargs['seed'])

        model_str = 'Wayformer' + kwargs_str + '_seed' + seed_str

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
    
    def define_default_kwargs(self, name = False):
        if not ('seed' in self.model_kwargs.keys()):
            self.model_kwargs["seed"] = 42
        
        if name:
            return
        
        # Go through the cfg file in wayformer.cfg.wayformer.yaml
        # Check if the key is in the model_kwargs, if not, add it, and set 
        # the value to the one in the .yaml file
        

        # Load yaml file
        if not hasattr(self, 'cfg'):
            # Get path to this file
            path = os.path.dirname(os.path.abspath(__file__))
            cfg_path = path + os.sep + 'Wayformer_unitraj' + os.sep + 'cfg' + os.sep + 'wayformer.yaml'
            self.cfg = OmegaConf.load(cfg_path)

        # Go through the cfg file
        for key in self.cfg.keys():
            if not (key in self.model_kwargs.keys()):
                self.model_kwargs[key] = self.cfg[key]
            
            else:
                self.cfg[key] = self.model_kwargs[key]

        

    def setup_method(self):

        self.min_t_O_train = self.num_timesteps_out
        self.max_t_O_train = self.num_timesteps_out
        self.predict_single_agent = True
        self.can_use_map = False
        self.can_use_graph = True

        seed = self.model_kwargs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Load yaml file
        path = os.path.dirname(os.path.abspath(__file__))
        cfg_path = path + os.sep + 'Wayformer_unitraj' + os.sep + 'cfg' + os.sep + 'wayformer.yaml'
        self.cfg = OmegaConf.load(cfg_path)

        # Check if map is available
        self.cfg['use_map'] = self.can_use_graph and self.has_graph

        # get number of possible config types
        self.cfg['num_agent_types'] = 5 # Pedestrian, Cyclist, Motorcylce, Vehicle, None

        # add required keys to config file
        self.cfg['past_len'] = self.num_timesteps_in
        self.cfg['future_len'] = self.num_timesteps_out


        # Set the number of agents needed for prediction
        self.prepare_batch_generation()
        self.cfg['max_num_agents'] = self.ID.shape[1]

        self.define_default_kwargs()
        self.sceneGraph_radius = self.model_kwargs['map_range']


        # Initialize the model
        self.model = Wayformer(self.cfg)


    def rearange_input_data(self, data):
        # data.shape = bs x num_agents x num_timesteps x 2

        data_old = data.copy()
        assert np.array_equal(self.input_data_type[:2], ['x', 'y']), "Input data type does not start with ['x', 'y']"

        data = np.concatenate([data[...,:2], np.full(data.shape[:-1] + (3,), np.nan)], axis = -1)
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
            print('    Wayformer: Rearranging input data failed')
            failed_cases = (num_any_finite - num_all_finite) / num_any_finite
            print('    Wayformer: In {:0.2f}% of cases, there where mixed nan/finite values'.format(100 * failed_cases))

            # Check if positions are always available
            assert np.isfinite(data[any_finite][...,:2]).all(), "There are finite values in v_x or v_y where x or y is nan"
            
        return data

    def extract_data_tensor(self, X, T, S, graph):
        # X.shape = (batch_size, num_agents, num_steps_in, num_features)
        # Y.shape = (batch_size, num_agents, num_steps_out, num_features)
        # T.shape = (batch_size, num_agents)
        # S.shape = (batch_size, num_agents)
        # Pred_agents.shape = (batch_size, num_agents)
        num_agents_max = self.model._M - 1
        X = X[:,:num_agents_max]
        T = T[:,:num_agents_max]
        S = S[:,:num_agents_max]

        missing_timesteps_start = torch.isfinite(X).all(-1).float().argmax(-1) # batch_size, num_agents
        missing_timesteps_end = missing_timesteps_start + torch.isfinite(X).all(-1).int().sum(-1) # batch_size, num_agents

        missing_timesteps = torch.stack([missing_timesteps_start, missing_timesteps_end], -1) # batch_size, num_agents, 2
        missing_timesteps = torch.unique(missing_timesteps.reshape(-1,2), dim = 0) # n, 2 

        V_x = torch.zeros_like(X[...,0]) # Shape (batch_size, num_agents, num_steps_in)
        V_y = torch.zeros_like(X[...,0]) # Shape (batch_size, num_agents, num_steps_in)
        for (missing_timestep_start, missing_timestep_end) in missing_timesteps:
            mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end) # batch_size, num_agents
            
            if torch.abs(missing_timestep_end - missing_timestep_start) > 1:
                x_mask = X[mask][..., 0] # Shape (M x num_steps_in)
                x_mask_time = x_mask[:,missing_timestep_start:missing_timestep_end] # Shape (M x num_steps_in)
                vx_mask_time = torch.gradient(x_mask_time, dim = 1)[0] # Shape (M x num_steps_in)
                vx_mask = torch.zeros_like(x_mask) # Shape (M x num_steps_in)
                vx_mask[:,missing_timestep_start:missing_timestep_end] = vx_mask_time
                V_x[mask] = vx_mask
                
                y_mask = X[mask][..., 1] # Shape (M x num_steps_in)
                y_mask_time = y_mask[:,missing_timestep_start:missing_timestep_end] # Shape (M x num_steps_in)
                vy_mask_time = torch.gradient(y_mask_time, dim = 1)[0] # Shape (M x num_steps_in)
                vy_mask = torch.zeros_like(y_mask) # Shape (M x num_steps_in)
                vy_mask[:,missing_timestep_start:missing_timestep_end] = vy_mask_time
                V_y[mask] = vy_mask
        

        # Use atan2 between v_y and v_x to get theta
        Theta = safe_atan2(V_y, V_x) # Shape (batch_size, num_agents, num_steps_in)

        Obj_trajs = torch.cat([X, V_x.unsqueeze(-1), V_y.unsqueeze(-1), Theta.unsqueeze(-1)], dim = -1)

        # Ensure position data is there
        missing_position = torch.isnan(X).any(-1)
        Obj_trajs[missing_position] = torch.nan
        Obj_trajs_mask = torch.isfinite(Obj_trajs).all(dim=-1)

        Track_index_to_predict = torch.zeros(Obj_trajs.shape[0], dtype=torch.long).to(self.device)

        # get one hot encoding of agent types
        Obj_types_hot = np.zeros((T.shape[0], T.shape[1], self.cfg['num_agent_types']))
        Obj_types_hot[T == '0', 0] = 1
        Obj_types_hot[T == 'V', 1] = 1
        Obj_types_hot[T == 'M', 2] = 1
        Obj_types_hot[T == 'B', 3] = 1
        Obj_types_hot[T == 'P', 4] = 1 

        Obj_types_hot = torch.from_numpy(Obj_types_hot).float().to(self.device)

        assert (Obj_types_hot.sum(-1) == 1).all(), "One hot encoding failed"

        # Get sizes
        Obj_sizes = torch.from_numpy(S).float().to(self.device) # Shape (batch_size, num_agents, 2)

        # Get graphs
        if graph is None:
            # Assume  num_road_segments = 0, num_pts_per_segment = 0
            map_polylines = torch.zeros(Obj_trajs.shape[0], 0, 0, 3).to(self.device)
            map_polylines_mask = torch.zeros(Obj_trajs.shape[0], 0, 0).to(self.device)
            map_type_hot = torch.zeros(Obj_trajs.shape[0], 0, 4).to(self.device)
            map_info = torch.zeros(Obj_trajs.shape[0], 0, 1).to(self.device)
        else:
            map_polylines = []
            map_type = []
            map_info = []
            max_len_batch = 0
            for i in range(len(graph)):
                centerlines = graph[i].centerlines
                lane_types = graph[i].lane_type
                max_len_scene = max([len(centerline) for centerline in centerlines])
                c = np.stack([np.pad(centerline, ((0,max_len_scene - len(centerline)),(0,0)), constant_values=np.nan) for centerline in centerlines], axis=0)

                max_len_batch = max(max_len_batch, max_len_scene)
                map_polylines.append(c)

                types = np.array([lane_type[0] for lane_type in lane_types])
                info  = np.array([lane_type[1] for lane_type in lane_types])

                map_type.append(types)
                map_info.append(info)

            max_num_roads = max([len(centerlines) for centerlines in map_polylines])
            for i in range(len(map_polylines)):
                num_roads, len_roads = map_polylines[i].shape[:2]
                map_polylines[i] = np.pad(map_polylines[i], ((0,max_num_roads - num_roads),(0,max_len_batch - len_roads),(0,0)), constant_values=np.nan)
                map_type[i] = np.pad(map_type[i], (0,max_num_roads - num_roads), constant_values='PEDESTRIAN')
                map_info[i] = np.pad(map_info[i], (0,max_num_roads - num_roads), constant_values=0)

            map_type = np.array(map_type) # Shape (batch_size, num_road_segments)
            map_type_hot = np.zeros((map_type.shape[0], map_type.shape[1], 4))
            map_type_hot[map_type == 'VEHICLE', 0] = 1
            map_type_hot[map_type == 'BIKE', 1] = 1
            map_type_hot[map_type == 'BUS', 2] = 1
            map_type_hot[map_type == 'PEDESTRIAN', 3] = 1
            map_type_hot = torch.from_numpy(map_type_hot).float().to(self.device) # Shape (batch_size, num_road_segments, 4)

            map_info = torch.from_numpy(np.array(map_info)).unsqueeze(-1).to(self.device) # Shape (batch_size, num_road_segments, 1)
            
            map_polylines_points = torch.from_numpy(np.stack(map_polylines, axis=0)).float().to(self.device)
            # Use the midpoint petween points instead of points
            map_polylines = (map_polylines_points[:,:,1:] + map_polylines_points[:,:,:-1]) / 2 # Shape (batch_size, num_road_segemnts, num_pts_per_segment, 2)
            map_polylines_mask = torch.isfinite(map_polylines).all(dim=-1)

            # Get the map_polylines heading
            map_polylines_heading = safe_atan2(map_polylines_points[:,:,1:,1] - map_polylines_points[:,:,:-1,1],
                                               map_polylines_points[:,:,1:,0] - map_polylines_points[:,:,:-1,0]) # Shape (batch_size, num_road_segemnts, num_pts_per_segment)

            # Combine map polyline stuff
            map_polylines = torch.cat([map_polylines, map_polylines_heading.unsqueeze(-1)], dim=-1) # Shape (batch_size, num_road_segemnts, num_pts_per_segment, 3)

        # Get rotation center and angle
        rot_center = Obj_trajs[:,0,-1,:2].clone() # Shape (batch_size, 2)
        rot_angle = Obj_trajs[:,0,-1,4].clone()

        # Prepare rotation matrix
        c, s = torch.cos(rot_angle), torch.sin(rot_angle)
        R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)

        # Translate and rotate all the positions to the rotation center
        Obj_trajs[...,:2]  = Obj_trajs[...,:2] - rot_center.unsqueeze(1).unsqueeze(1)
        Obj_trajs[...,:2]  = torch.matmul(Obj_trajs[...,:2], R.unsqueeze(1))
        Obj_trajs[...,2:4] = torch.matmul(Obj_trajs[...,2:4], R.unsqueeze(1))
        Obj_trajs[...,4]   = Obj_trajs[...,4] - rot_angle.unsqueeze(-1).unsqueeze(-1)

        if graph is not None:
            map_polylines[...,:2] = map_polylines[...,:2] - rot_center.unsqueeze(1).unsqueeze(1)
            map_polylines[...,:2] = torch.matmul(map_polylines[...,:2], R.unsqueeze(1))
            map_polylines[...,2]  = map_polylines[...,2] - rot_angle.unsqueeze(-1).unsqueeze(-1)

        # Combine agent information
        Obj_info = torch.cat([Obj_types_hot, Obj_sizes], dim=-1).unsqueeze(-2) # Shape (batch_size, num_agents, 1, 7)
        Map_info = torch.cat([map_type_hot, map_info], dim=-1).unsqueeze(-2) # Shape (batch_size, num_road_segments, 1, 5)

        # Add type and size to the input data
        Obj_trajs = torch.cat([Obj_trajs, Obj_info.repeat_interleave(Obj_trajs.shape[-2], dim = -2)], dim=-1) # Shape (batch_size, num_agents, num_steps_in, 12)
        map_polylines = torch.cat([map_polylines, Map_info.repeat_interleave(map_polylines.shape[-2], dim = -2)], dim=-1) # Shape (batch_size, num_road_segemnts, num_pts_per_segment, 8)

        # Remove nan values
        Obj_trajs = torch.nan_to_num(Obj_trajs, nan=0.0)
        map_polylines = torch.nan_to_num(map_polylines, nan=0.0)

        inputs = {
            'obj_trajs': Obj_trajs, # Should torch.tensor of shape (batch_size, num_agents, num_steps_in, 12) 
            'obj_trajs_mask': Obj_trajs_mask, # Should torch.tensor of shape (batch_size, num_agents, num_steps_in)
            'track_index_to_predict': Track_index_to_predict, # Should be torch.tensor of shape (batch_size)
            'center_gt_trajs': None, # Should torch.tensor of shape (batch_size, num_steps_out, 12)
            'center_gt_trajs_mask': None, # Should torch.tensor of shape (batch_size, num_steps_out)
            'map_polylines': map_polylines, # Should torch.tensor of shape (batch_size, num_road_segemnts, num_pts_per_segment, 8)
            'map_polylines_mask': map_polylines_mask, # Should torch.tensor of shape (batch_size, num_road_segemnts, num_pts_per_segment)
        }


        batch = {'input_dict': inputs}
        return batch, rot_angle, rot_center


    def extract_data(self, X, Y, T, S, Pred_agents, graph):
        # X.shape = (batch_size, num_agents, num_steps_in, num_features)
        # Y.shape = (batch_size, num_agents, num_steps_out, num_features)
        # T.shape = (batch_size, num_agents)
        # S.shape = (batch_size, num_agents)
        # Pred_agents.shape = (batch_size, num_agents)
        # Remove excessive agents
        num_agents_max = self.model._M - 1
        X = X[:,:num_agents_max]
        T = T[:,:num_agents_max]
        S = S[:,:num_agents_max]




        Obj_trajs = self.rearange_input_data(X)
        Obj_trajs = torch.from_numpy(Obj_trajs).float().to(self.device)
        Obj_trajs_mask = torch.isfinite(Obj_trajs).all(dim=-1)

        
        Track_index_to_predict = torch.zeros(Obj_trajs.shape[0], dtype=torch.long).to(self.device)

        if Y is not None:
            Center_GT_trajs = self.rearange_input_data(Y[:,0])
            Center_GT_trajs = torch.from_numpy(Center_GT_trajs).float().to(self.device) # Shape (B, num_steps_out, 2)
            Center_GT_trajs_mask = torch.isfinite(Center_GT_trajs).all(dim=-1)
        else:
            Center_GT_trajs = None
            Center_GT_trajs_mask = None

        # get one hot encoding of agent types
        Obj_types_hot = np.zeros((T.shape[0], T.shape[1], self.cfg['num_agent_types']))
        Obj_types_hot[T == '0', 0] = 1
        Obj_types_hot[T == 'V', 1] = 1
        Obj_types_hot[T == 'M', 2] = 1
        Obj_types_hot[T == 'B', 3] = 1
        Obj_types_hot[T == 'P', 4] = 1 

        Obj_types_hot = torch.from_numpy(Obj_types_hot).float().to(self.device)

        assert (Obj_types_hot.sum(-1) == 1).all(), "One hot encoding failed"

        # Get sizes
        Obj_sizes = torch.from_numpy(S).float().to(self.device) # Shape (batch_size, num_agents, 2)

        # Get graphs
        if graph is None:
            # Assume  num_road_segments = 0, num_pts_per_segment = 0
            map_polylines = torch.zeros(Obj_trajs.shape[0], 0, 0, 3).to(self.device)
            map_polylines_mask = torch.zeros(Obj_trajs.shape[0], 0, 0).to(self.device)
            map_type_hot = torch.zeros(Obj_trajs.shape[0], 0, 4).to(self.device)
            map_info = torch.zeros(Obj_trajs.shape[0], 0, 1).to(self.device)
        else:
            map_polylines = []
            map_type = []
            map_info = []
            max_len_batch = 0
            for i in range(len(graph)):
                centerlines = graph[i].centerlines
                lane_types = graph[i].lane_type
                max_len_scene = max([len(centerline) for centerline in centerlines])
                c = np.stack([np.pad(centerline, ((0,max_len_scene - len(centerline)),(0,0)), constant_values=np.nan) for centerline in centerlines], axis=0)

                max_len_batch = max(max_len_batch, max_len_scene)
                map_polylines.append(c)

                types = np.array([lane_type[0] for lane_type in lane_types])
                info  = np.array([lane_type[1] for lane_type in lane_types])

                map_type.append(types)
                map_info.append(info)

            max_num_roads = max([len(centerlines) for centerlines in map_polylines])
            for i in range(len(map_polylines)):
                num_roads, len_roads = map_polylines[i].shape[:2]
                map_polylines[i] = np.pad(map_polylines[i], ((0,max_num_roads - num_roads),(0,max_len_batch - len_roads),(0,0)), constant_values=np.nan)
                map_type[i] = np.pad(map_type[i], (0,max_num_roads - num_roads), constant_values='PEDESTRIAN')
                map_info[i] = np.pad(map_info[i], (0,max_num_roads - num_roads), constant_values=0)

            map_type = np.array(map_type) # Shape (batch_size, num_road_segments)
            map_type_hot = np.zeros((map_type.shape[0], map_type.shape[1], 4))
            map_type_hot[map_type == 'VEHICLE', 0] = 1
            map_type_hot[map_type == 'BIKE', 1] = 1
            map_type_hot[map_type == 'BUS', 2] = 1
            map_type_hot[map_type == 'PEDESTRIAN', 3] = 1
            map_type_hot = torch.from_numpy(map_type_hot).float().to(self.device) # Shape (batch_size, num_road_segments, 4)

            map_info = torch.from_numpy(np.array(map_info)).unsqueeze(-1).to(self.device) # Shape (batch_size, num_road_segments, 1)
            
            map_polylines_points = torch.from_numpy(np.stack(map_polylines, axis=0)).float().to(self.device)
            # Use the midpoint petween points instead of points
            map_polylines = (map_polylines_points[:,:,1:] + map_polylines_points[:,:,:-1]) / 2 # Shape (batch_size, num_road_segemnts, num_pts_per_segment, 2)
            map_polylines_mask = torch.isfinite(map_polylines).all(dim=-1)

            # Get the map_polylines heading
            map_polylines_heading = torch.atan2(map_polylines_points[:,:,1:,1] - map_polylines_points[:,:,:-1,1], 
                                                map_polylines_points[:,:,1:,0] - map_polylines_points[:,:,:-1,0]) # Shape (batch_size, num_road_segemnts, num_pts_per_segment)
            
            # Combine map polyline stuff
            map_polylines = torch.cat([map_polylines, map_polylines_heading.unsqueeze(-1)], dim=-1) # Shape (batch_size, num_road_segemnts, num_pts_per_segment, 3)

        # Get rotation center and angle
        rot_center = Obj_trajs[:,0,-1,:2].clone() # Shape (batch_size, 2)
        rot_angle = Obj_trajs[:,0,-1,4].clone()

        # Prepare rotation matrix
        c, s = torch.cos(rot_angle), torch.sin(rot_angle)
        R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)

        # Translate and rotate all the positions to the rotation center
        Obj_trajs[...,:2]  = Obj_trajs[...,:2] - rot_center.unsqueeze(1).unsqueeze(1)
        Obj_trajs[...,:2]  = torch.matmul(Obj_trajs[...,:2], R.unsqueeze(1))
        Obj_trajs[...,2:4] = torch.matmul(Obj_trajs[...,2:4], R.unsqueeze(1))
        Obj_trajs[...,4]   = Obj_trajs[...,4] - rot_angle.unsqueeze(-1).unsqueeze(-1)

        if graph is not None:
            map_polylines[...,:2] = map_polylines[...,:2] - rot_center.unsqueeze(1).unsqueeze(1)
            map_polylines[...,:2] = torch.matmul(map_polylines[...,:2], R.unsqueeze(1))
            map_polylines[...,2]  = map_polylines[...,2] - rot_angle.unsqueeze(-1).unsqueeze(-1)


        # Do translation and rotation for gt
        if Center_GT_trajs is not None:
            Center_GT_trajs[...,:2]  = Center_GT_trajs[...,:2] - rot_center.unsqueeze(1)
            Center_GT_trajs[...,:2]  = torch.matmul(Center_GT_trajs[...,:2], R)
            Center_GT_trajs[...,2:4] = torch.matmul(Center_GT_trajs[...,2:4], R)
            Center_GT_trajs[...,4]   = Center_GT_trajs[...,4] - rot_angle.unsqueeze(-1)

        # Combine agent information
        Obj_info = torch.cat([Obj_types_hot, Obj_sizes], dim=-1).unsqueeze(-2) # Shape (batch_size, num_agents, 1, 7)
        Map_info = torch.cat([map_type_hot, map_info], dim=-1).unsqueeze(-2) # Shape (batch_size, num_road_segments, 1, 5)

        # Add type and size to the input data
        Obj_trajs = torch.cat([Obj_trajs, Obj_info.repeat_interleave(Obj_trajs.shape[-2], dim = -2)], dim=-1) # Shape (batch_size, num_agents, num_steps_in, 12)
        map_polylines = torch.cat([map_polylines, Map_info.repeat_interleave(map_polylines.shape[-2], dim = -2)], dim=-1) # Shape (batch_size, num_road_segemnts, num_pts_per_segment, 8)
        if Center_GT_trajs is not None:
            Center_GT_trajs = torch.cat([Center_GT_trajs, Obj_info[:,0].repeat_interleave(Center_GT_trajs.shape[-2], dim = -2)], dim=-1) # Shape (batch_size, num_steps_out, 12)
        
        # Remove nan values
        Obj_trajs = torch.nan_to_num(Obj_trajs, nan=0.0)
        map_polylines = torch.nan_to_num(map_polylines, nan=0.0)
        if Center_GT_trajs is not None:
            Center_GT_trajs = torch.nan_to_num(Center_GT_trajs, nan=0.0)

        inputs = {
            'obj_trajs': Obj_trajs, # Should torch.tensor of shape (batch_size, num_agents, num_steps_in, 12) 
            'obj_trajs_mask': Obj_trajs_mask, # Should torch.tensor of shape (batch_size, num_agents, num_steps_in)
            'track_index_to_predict': Track_index_to_predict, # Should be torch.tensor of shape (batch_size)
            'center_gt_trajs': Center_GT_trajs, # Should torch.tensor of shape (batch_size, num_steps_out, 12)
            'center_gt_trajs_mask': Center_GT_trajs_mask, # Should torch.tensor of shape (batch_size, num_steps_out)
            'map_polylines': map_polylines, # Should torch.tensor of shape (batch_size, num_road_segemnts, num_pts_per_segment, 8)
            'map_polylines_mask': map_polylines_mask, # Should torch.tensor of shape (batch_size, num_road_segemnts, num_pts_per_segment)
        }


        batch = {'input_dict': inputs}
        return batch, rot_angle, rot_center
    
    
    def train_method(self):
        # initialize optimizer
        [optimizer], [scheduler] = self.model.configure_optimizers()

        # Put model to device
        self.model.to(self.device)

        # load model
        intermediate_file = self.model_file[:-4] + '_intermediate.npy'
        if os.path.exists(intermediate_file):
            data = np.load(intermediate_file, allow_pickle=True)

            Model_weights = data[0]
            self.train_loss = data[1]
            completed_epochs = data[2]

            # Overwrite models parameters
            Weights = list(self.model.parameters())
            with torch.no_grad():
                for i, weights_loaded in enumerate(Model_weights):
                    weights_loaded_torch = torch.from_numpy(weights_loaded)
                    assert Weights[i].shape == weights_loaded_torch.shape, "Model weights do not match"
                    Weights[i][:] = weights_loaded_torch[:]
        
        else:
            completed_epochs = 0
            self.train_loss = np.ones((1, self.model_kwargs['max_epochs'])) * np.nan

        for epoch in range(1, self.model_kwargs['max_epochs'] + 1):
            print('    Wayformer: Training epoch {}'.format(epoch))

            if epoch <= completed_epochs:
                scheduler.step()
                continue

            batch_size = self.cfg['train_batch_size']

            epoch_done = False
            batch_train_loss = []
            num_samples = 0
            while not epoch_done:
                X, Y, T, S, _, _, graph, Pred_agents, num_steps, _, _, epoch_done = self.provide_batch_data('train', batch_size, val_split_size=0.0)
                assert num_steps == self.cfg['future_len']

                # Transfrom the data to the correct format
                batch, _, _ = self.extract_data(X, Y,  T, S, Pred_agents, graph)

                # Calculate loss
                _, loss = self.model(batch)

                # Backprop
                optimizer.zero_grad()
                loss.backward()

                # Clip gradient vis norms with self.cfg['grad_clip_norm]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['grad_clip_norm'])

                # Update weights
                optimizer.step()

                # Save loss
                batch_train_loss.append(loss.item() * len(X))
                num_samples += len(X)

            # Get epoch loss
            epoch_loss = np.sum(batch_train_loss) / num_samples
            self.train_loss[0, epoch - 1] = epoch_loss

            print('    Wayformer: Epoch loss: {}'.format(epoch_loss))
            print('')
            
            # Run scheduler
            scheduler.step()

            # Save intermediate model
            Weights = list(self.model.parameters())
            Weights_saved = []
            for weigths in Weights:
                Weights_saved.append(weigths.detach().cpu().numpy())

            os.makedirs(os.path.dirname(intermediate_file), exist_ok=True)
            np.save(intermediate_file, [Weights_saved, self.train_loss, epoch])
        
        # Save model
        self.weights_saved = Weights_saved
        


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
        # Overwrite models parameters
        Weights = list(self.model.parameters())
        with torch.no_grad():
            for i, weights_loaded in enumerate(self.weights_saved):
                weights_loaded_torch = torch.from_numpy(weights_loaded)
                if not (Weights[i].shape == weights_loaded_torch.shape):
                    error_string = []
                    error_string.append('Model weights do not match')
                    error_string.append('Shape loaded: {}'.format(weights_loaded_torch.shape))
                    error_string.append('Shape desired: {}'.format(Weights[i].shape))
                    error_string.append('max_num_agents (cfg): {}'.format(self.cfg['max_num_agents']))
                    error_string.append('data_set.Pred_agents_eval.shape: {}'.format(len(self.data_set.Pred_agents_eval)))
                    if hasattr(self, 'ID'):
                        error_string.append('ID.shape: {}'.format(self.ID.shape))
                    else:
                        error_string.append('ID.shape not available. Why???')
                    error_string = '\n'.join(error_string)
                    assert False, error_string
                Weights[i][:] = weights_loaded_torch[:]

    def predict_method(self):
        prediction_done = False
        self.model.eval()
        self.model.to(self.device)

        # Get range for stds
        log_std_range = (-1.609, 5.0) # i.e. 0.2 - 150 m
        rho_limit = 0.5

        with torch.no_grad():     
            ind_batch = 0    
            while not prediction_done:
                ind_batch = ind_batch + 1
                print('    Wayformer: Predicting batch {}'.format(ind_batch))

                X,  T, S, _, _, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', self.model_kwargs['eval_batch_size'])
                batch, rot_angle, rot_center = self.extract_data(X, None, T, S, Pred_agents, graph)

                assert (batch['input_dict']['obj_trajs_mask'].sum(1) > 0).all(), "At least one agent should be present in the scene"

                output_dist = self.model(batch, eval = True)

                # Sample trajectories
                mode_prob = output_dist['predicted_probability'] # batch_size x num_modes
                mode_dist = output_dist['predicted_trajectory'] # batch_size x num_modes x num_steps_out x 5
                # The five values are mu_x, mu_y, sigma_x, sigma_y, rho_xy

                # Build GMM (use multivaraiate normal distribution)
                Pred = []
                for i in range(len(mode_prob)):
                    num_modes = mode_prob[i].shape[0]
                    Pred.append([])
                    for j in range(num_modes):
                        mode_dist_ij = mode_dist[i,j]
                        mu_x = mode_dist_ij[...,0]
                        mu_y = mode_dist_ij[...,1]

                        # Limit std
                        log_sigma_x = torch.clip(mode_dist_ij[...,2], min=log_std_range[0], max=log_std_range[1])
                        log_sigma_y = torch.clip(mode_dist_ij[...,3], min=log_std_range[0], max=log_std_range[1])
                        sigma_x = torch.exp(log_sigma_x)  # (0.2m to 150m)
                        sigma_y = torch.exp(log_sigma_y)  # (0.2m to 150m)
                        rho_xy = torch.clip(mode_dist_ij[...,4], min = -rho_limit, max = rho_limit)

                        mu = torch.stack([mu_x, mu_y], dim=-1)

                        cov = torch.zeros((len(mu_x),2,2), dtype = torch.float32, device = self.device)
                        cov[...,0,0] = sigma_x ** 2
                        cov[...,1,1] = sigma_y ** 2
                        cov[...,0,1] = sigma_x * sigma_y * rho_xy
                        cov[...,1,0] = sigma_x * sigma_y * rho_xy

                        dist = MultivariateNormal(mu, cov)
                        Pred[-1].append(dist.sample([self.num_samples_path_pred]))
                    Pred[-1] = torch.stack(Pred[-1], dim=0)
                
                Pred = torch.stack(Pred, dim=0) # Shape (batch_size, num_modes, num_paths, num_steps_out, 2)

                # Randomly select one of the predicted paths alonf the num_modes dimension, according to given probabilites
                indices = torch.multinomial(mode_prob, self.num_samples_path_pred, replacement=True) # Shape (batch_size, num_samples_path_pred)

                batch_ind = torch.arange(Pred.shape[0], device = self.device).unsqueeze(-1).repeat(1, self.num_samples_path_pred)
                paths_ind = torch.arange(self.num_samples_path_pred, device = self.device).unsqueeze(0).repeat(Pred.shape[0], 1)
                Pred = Pred[batch_ind, indices, paths_ind] # Shape (batch_size, num_samples_path_pred, num_steps_out, 2)
                
                # Rotate the predicted paths back
                c, s = torch.cos(-rot_angle), torch.sin(-rot_angle)
                R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2) # Shape (batch_size, 2, 2)
                
                Pred = torch.matmul(Pred, R.unsqueeze(1))
                Pred = Pred + rot_center.unsqueeze(1).unsqueeze(1)

                Pred = Pred.cpu().numpy()

                num_step_pred = Pred.shape[-2]
                if num_steps <= num_step_pred:
                    Pred = Pred[..., :num_steps, :]
                else: 
                    # use linear extrapolation
                    last_vel = Pred[..., [-1],:] - Pred[..., [-2],:] # Shape (batch_size, num_paths, 1, 2)
                    steps = np.arange(1, num_steps - num_step_pred + 1).reshape(1, 1, -1, 1)

                    Pred_exp = Pred[..., [-1],:] + last_vel * steps
                    Pred = np.concatenate([Pred, Pred_exp], axis=-2)
                
                # save predictions
                self.save_predicted_batch_data(Pred, Sample_id, Agent_id, Pred_agents)
            
            print('    Wayformer: Prediction done')
            print('')



    def predict_batch_tensor(self, X, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps):
        self.model.to(X.device)
        # Get range for stds
        log_std_range = (-1.609, 5.0) # i.e. 0.2 - 150 m
        rho_limit = 0.5

        batch, rot_angle, rot_center = self.extract_data_tensor(X, T, S, graph)

        assert (batch['input_dict']['obj_trajs_mask'].sum(1) > 0).all(), "At least one agent should be present in the scene"

        output_dist = self.model(batch, eval = True)

        # Sample trajectories
        mode_prob = output_dist['predicted_probability'] # batch_size x num_modes
        mode_dist = output_dist['predicted_trajectory'] # batch_size x num_modes x num_steps_out x 5
        # The five values are mu_x, mu_y, sigma_x, sigma_y, rho_xy

        # Build GMM (use multivaraiate normal distribution)
        Pred = []
        for i in range(len(mode_prob)):
            num_modes = mode_prob[i].shape[0]
            Pred.append([])
            for j in range(num_modes):
                mode_dist_ij = mode_dist[i,j]
                mu_x = mode_dist_ij[...,0]
                mu_y = mode_dist_ij[...,1]

                # Limit std
                log_sigma_x = torch.clip(mode_dist_ij[...,2], min=log_std_range[0], max=log_std_range[1])
                log_sigma_y = torch.clip(mode_dist_ij[...,3], min=log_std_range[0], max=log_std_range[1])
                sigma_x = torch.exp(log_sigma_x)  # (0.2m to 150m)
                sigma_y = torch.exp(log_sigma_y)  # (0.2m to 150m)
                rho_xy = torch.clip(mode_dist_ij[...,4], min = -rho_limit, max = rho_limit)

                mu = torch.stack([mu_x, mu_y], dim=-1)

                cov = torch.zeros((len(mu_x),2,2), dtype = torch.float32, device = self.device)
                cov[...,0,0] = sigma_x ** 2
                cov[...,1,1] = sigma_y ** 2
                cov[...,0,1] = sigma_x * sigma_y * rho_xy
                cov[...,1,0] = sigma_x * sigma_y * rho_xy

                dist = MultivariateNormal(mu, cov)
                Pred[-1].append(dist.sample([self.num_samples_path_pred]))
            Pred[-1] = torch.stack(Pred[-1], dim=0)
        
        Pred = torch.stack(Pred, dim=0) # Shape (batch_size, num_modes, num_paths, num_steps_out, 2)

        # Randomly select one of the predicted paths alonf the num_modes dimension, according to given probabilites
        indices = torch.multinomial(mode_prob, self.num_samples_path_pred, replacement=True) # Shape (batch_size, num_samples_path_pred)

        batch_ind = torch.arange(Pred.shape[0], device = self.device).unsqueeze(-1).repeat(1, self.num_samples_path_pred)
        paths_ind = torch.arange(self.num_samples_path_pred, device = self.device).unsqueeze(0).repeat(Pred.shape[0], 1)
        Pred = Pred[batch_ind, indices, paths_ind] # Shape (batch_size, num_samples_path_pred, num_steps_out, 2)
        
        # Rotate the predicted paths back
        c, s = torch.cos(-rot_angle), torch.sin(-rot_angle)
        R = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2) # Shape (batch_size, 2, 2)
        
        Pred = torch.matmul(Pred, R.unsqueeze(1))
        Pred = Pred + rot_center.unsqueeze(1).unsqueeze(1)

        num_step_pred = Pred.shape[-2]
        if num_steps <= num_step_pred:
            Pred = Pred[..., :num_steps, :]
        else: 
            # use linear extrapolation
            last_vel = Pred[..., [-1],:] - Pred[..., [-2],:] # Shape (batch_size, num_paths, 1, 2)
            steps = torch.arange(1, num_steps - num_step_pred + 1).reshape(1, 1, -1, 1).to(self.device).float()

            Pred_exp = Pred[..., [-1],:] + last_vel * steps
            Pred = torch.cat([Pred, Pred_exp], axis=-2)

        return Pred.unsqueeze(1).repeat_interleave(X.shape[1], dim = 1)
