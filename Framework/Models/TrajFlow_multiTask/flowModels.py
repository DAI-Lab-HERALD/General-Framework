import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from TrajFlow_multiTask.spline_flow import NeuralSplineFlow
from TrajFlow_multiTask.SocialEncodingModule import SocialInterGNN
from torch_geometric.data import Data

from utils.memory_utils import get_total_memory, get_used_memory


import numpy as np
import networkx as nx

from TrajFlow_multiTask.dag_utils import dagification, kahn_toposort
from TrajFlow_multiTask.LaneGCN import M2A, MapNet, graph_gather, gpu, to_long

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)

class RNN(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=16, hs=16, nl=3, device=0):
        
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden
    

class Scene_Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=4, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5, stride=4, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 32, 5, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True)#,
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 * 2 * 3, 128), 
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    

class InteractionClassifier(nn.Module):
    
    def __init__(self, pair_info_dim, num_classes, iCL_hdim=128):
        
        super().__init__()
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(pair_info_dim, iCL_hdim), 
            nn.ReLU(True)
        )
        self.final_class0 = nn.Linear(iCL_hdim, num_classes)
        self.final_class1 = nn.Linear(iCL_hdim, num_classes)
        self.final_class2 = nn.Linear(iCL_hdim, num_classes)
        self.final_binary0 = nn.Linear(iCL_hdim, 3)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, task=0):
        x = self.encoder_lin(x)
        if task == 0:
            x = self.final_class0(x)
            x = self.softmax(x)
        elif task == 1:
            x = self.final_class1(x)
            x = self.softmax(x)
        elif task == 2:
            x = self.final_class2(x)
            x = self.softmax(x)
        elif task == 3:
            x = self.final_binary0(x)
            x = self.sigmoid(x)
        else:
            raise ValueError("Task not recognized")
        return x



class TrajFlow(nn.Module):

    def __init__(self, pred_steps, alpha, beta, gamma, scene_encoder=None, B=15., 
                use_map=False, use_graph=False, rel_coords=True, norm_rotation=False, device=0,
                obs_encoding_size=16, scene_encoding_size=8, 
                n_layers_rnn=3, es_rnn=16, hs_rnn=16,
                n_layers_gnn=4, es_gnn=64, T_all = None, interaction_thresh=1/3, lanegcn_configs=None, iCL_hdim=128):
        
        nn.Module.__init__(self)
        self.pred_steps = pred_steps
        self.output_size = pred_steps 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation
        

        # core modules
        self.obs_encoding_size = obs_encoding_size
        self.scene_encoding_size = scene_encoding_size
        self.n_layers_rnn = n_layers_rnn
        self.es_rnn = es_rnn
        self.hs_rnn = hs_rnn
        
        self.es_gnn = es_gnn
        self.n_layers_gnn = n_layers_gnn

        self.use_map = use_map
        self.use_graph = use_graph
        self.lanegcn_configs = lanegcn_configs
        
        # observation encoders
        self.obs_encoder = nn.ModuleDict({})
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.t_unique = self.t_unique[self.t_unique != 48]

        for t in self.t_unique:
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            
            self.obs_encoder[t_key] = RNN(nin=2, nout=self.obs_encoding_size, es=self.es_rnn, 
                                        hs=self.hs_rnn, nl=self.n_layers_rnn, device=device)
            self.obs_encoder[t_key].to(device)
    
        self.edge_dim_gnn = len(self.t_unique) * 2 + 3 # 2 times number of classes for one hot encoding of agent type, 2 for distance and 1 for angle


        if use_map:
            if scene_encoder is None:
                self.scene_encoder = Scene_Encoder(encoded_space_dim=self.scene_encoding_size)
            else:
                self.scene_encoder = scene_encoder
            
            # self.GNNencoder = SocialInterGNN(num_layers=n_layers_gnn, emb_dim=es_gnn,
            #                                 in_dim=self.obs_encoding_size + self.scene_encoding_size, edge_dim=self.edge_dim_gnn,
            #                                 device = device)
        else:
            self.scene_encoder = None

        self.GNNencoder = SocialInterGNN(num_layers=n_layers_gnn, emb_dim=es_gnn,
                                        in_dim=self.obs_encoding_size, edge_dim=self.edge_dim_gnn,
                                        device = device)
            

        if use_map:
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.es_gnn+self.scene_encoding_size+self.output_size, 
                                    n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)  
        else:    
            self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.es_gnn+self.output_size, 
                                        n_layers=10, K=8, B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)  
            

        if use_graph:
            self.MapNet = MapNet(self.lanegcn_configs)
            self.M2A = M2A(self.lanegcn_configs)
            
        
        self.adjacency_matrix = None
        self.G = None
        self.G_with_self_loops = None
        self.topo_order = None
        self.topo_order_pred = None
        self.iCL_hdim = iCL_hdim

        self.interaction_classifier = InteractionClassifier(pair_info_dim=2 * self.obs_encoding_size + 2 * len(self.t_unique) + 3, num_classes=3, iCL_hdim=self.iCL_hdim)
        # self.interaction_classifier = InteractionClassifier(pair_info_dim=2*self.obs_encoding_size + len(self.t_unique) + 3, num_classes=3, iCL_hdim=self.iCL_hdim)
        # self.interaction_classifier = InteractionClassifier(pair_info_dim=self.obs_encoding_size + len(self.t_unique) + 3, num_classes=3, iCL_hdim=self.iCL_hdim)

        self.interaction_thresh = interaction_thresh#0.5
    

        # move model to specified device
        self.device = device
        self.to(device)


    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[...,1:,:] = (y_rel[...,1:,:] - y_rel[...,:-1,:]) # steps relative to each other
        y_rel = y_rel * self.alpha # scale up for numeric reasons
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = y_rel / self.alpha
        return torch.cumsum(y_abs, dim=-2) + x_t 

    def _rotate(self, x, x_t, angles_rad):
        # Fix alingment issues
        len_x = len(x.shape)
        len_r = len(angles_rad.shape)
        for i in range(len_x - len_r - 1):
            angles_rad = angles_rad.unsqueeze(-1)
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        x_centered = x - x_t # translate
        x_vals, y_vals = x_centered[...,0], x_centered[...,1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        x_centered[...,0] = new_x_vals
        x_centered[...,1] = new_y_vals
        return x_centered + x_t # translate back

    def _normalize_rotation(self, x, y_true=None):
        x_t = x[...,-1:,:]
        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[...,[-1],:] - x[...,[-2],:]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[...,1], x_t_rel[...,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad # inverse
        else:
            return x, rot_angles_rad # forward pass

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError
    

    def _encode_trajectories(self, x, T):

        x_in = x

        if self.norm_rotation:
            x_in, angle = self._normalize_rotation(x_in)

        if self.rel_coords:
            x_in = x[...,1:,:] - x[...,:-1,:]
            x_in[...,-1,:] = torch.nan_to_num(x_in[...,-1,:])
        
        x_enc     = torch.zeros((x.shape[0], x.shape[1], x.shape[2] - 1, self.obs_encoding_size), device = self.device)

         # create mask for entries that start with NaN  
        first_entry_step = x_in.isfinite().all(-1).to(torch.float32).argmax(dim = -1) # Batch_size * num_agents

        sample_ind, agent_ind, step_ind = torch.where(torch.ones(x_in.shape[:3], dtype = torch.bool, device = self.device))

        step_ind_adjust = first_entry_step.flatten().repeat_interleave(x_in.shape[2]) 
        
        # Roll the tensor to the left by step_ind_adjust so that the first non-NaN entry is at the first position
        x_in[sample_ind, agent_ind, step_ind - step_ind_adjust] = x_in[sample_ind, agent_ind, step_ind] 

        # Replace NaN with 0
        x_in = torch.nan_to_num(x_in)
        
        for t in self.t_unique:
            t_in = T == t
            
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            
            x_enc[t_in], _ = self.obs_encoder[t_key](x_in[t_in])
            
        # Extract the corresponding values
        x_enc = torch.gather(x_enc, 2, x_enc.shape[2] - 1 - torch.tile(first_entry_step.unsqueeze(-1).unsqueeze(-1), (1,1,1,x_enc.shape[3])))

        x_enc = x_enc.squeeze(2) 

        return x_enc
    
    def _get_agent_pair_angles(self, x, D, max_num_agents):

        # Determine relative velocity angle between agents
        x_vel = x[:,:,-1,:] - x[:,:,-2,:] 
        # x_vel = x_vel / torch.sqrt(torch.sum(x_vel ** 2, dim = -1, keepdim = True)) # shape: num_samples x max_num_agents x 2
        x_vel = torch.tile(x_vel.unsqueeze(1), (1,max_num_agents,1,1)) # shape: num_samples x max_num_agents x max_num_agents x 2

        # Dot product between the distance and the velocities of all agents
        
        dot_product = torch.sum(D*x_vel, dim = -1) # shape: num_samples x max_num_agents x max_num_agents
        # Magnitude of the velocities of all agents
        vel_mag = torch.sqrt(torch.sum(x_vel ** 2, dim = -1)) # shape: num_samples x max_num_agents x max_num_agents
        dist_mag = torch.sqrt(torch.sum(D ** 2, dim = -1)) # shape: num_samples x max_num_agents x max_num_agents
        mag_product = dist_mag * vel_mag # shape: num_samples x max_num_agents x max_num_agents

        # Angle between the velocities of all agents
        ratio = torch.clip(dot_product / mag_product, -1, 1) # shape: num_samples x max_num_agents x max_num_agents
        ang = torch.acos(ratio) # shape: num_samples x max_num_agents x max_num_agents
        ang[torch.abs(mag_product) <= 1e-6] = 0 # set angle to 0 if the magnitude product is 0

        return ang


    def _encode_conditionals(self, x, T, scene=None, scene_graph=None):
        print('    Encoding trajectories')
        x_enc = self._encode_trajectories(x, T)

        x_t0 = x[:,:,[-1],:]
        present_agent = x_t0.isfinite().all(-1).squeeze(-1) # Shape num_samples x num_agents
            
            
        # Define sizes
        max_num_agents = x_enc.shape[1]
        
        # Deal with autoencoder here
        # Find existing agents (T = 48 means that the agent does not exist)
        existing_agent = T != 48 # shape: num_samples x max_num_agents

        D = x[:,None,:,-1] - x[:,:,None,-1] # shape: num_samples x max_num_agents x max_num_agents x 2

        ang = self._get_agent_pair_angles(x, D, max_num_agents)
        
        # Get agent pair information
        T_one_hot = (T.unsqueeze(-1) == self.t_unique.unsqueeze(0)).float() # shape: num_samples x max_num_agents x num_classes

        # pair_info = torch.cat(((x_enc[:,None,:].repeat(1,max_num_agents,1,1) + x_enc[:,:,None,:].repeat(1,1,max_num_agents,1)),
        #                        (T_one_hot[:,None,:].repeat(1,max_num_agents,1,1) + T_one_hot[:,:,None].repeat(1,1,max_num_agents,1)), 
        #                        D, ang[:,:,:,None]), dim = -1) # shape: num_samples x max_num_agents x max_num_agents x 2 * enc_size + 2 * num_classes + 3
        
        # pair_info = torch.cat(((x_enc[:,None,:].repeat(1,max_num_agents,1,1)+x_enc[:,:,None,:].repeat(1,1,max_num_agents,1)),
        #             (T_one_hot[:,None,:].repeat(1,max_num_agents,1,1)+T_one_hot[:,:,None].repeat(1,1,max_num_agents,1)), 
        #             D, ang[:,:,:,None]), dim = -1)

        pair_info = torch.cat((x_enc[:,None,:].repeat(1,max_num_agents,1,1), x_enc[:,:,None,:].repeat(1,1,max_num_agents,1),
                T_one_hot[:,None,:].repeat(1,max_num_agents,1,1), T_one_hot[:,:,None].repeat(1,1,max_num_agents,1), 
                D, ang[:,:,:,None]), dim = -1)
        
        # get upper triangular part of the pair_info tensor
        pair_info = pair_info[:,torch.triu(torch.ones(max_num_agents, max_num_agents), diagonal = 1) == 1,:] # shape: num_samples x max_num_agents x max_num_agents x 2 * enc_size + 2 * num_classes + 3
        
        interaction_class = self.interaction_classifier(pair_info, task=0) # shape: num_samples x max_num_agents x max_num_agents x 3

        adjacency_matrix = torch.zeros((x.shape[0], max_num_agents, max_num_agents), device = self.device)
        edge_probs = torch.zeros((x.shape[0], max_num_agents, max_num_agents), device = self.device)

        # Set upper triangular part of adjacency matrix to 1 for agents if interaction class argmax is index 1
        # adjacency_matrix[:,torch.triu(torch.ones(max_num_agents, max_num_agents), diagonal = 1) == 1] = (interaction_class.argmax(dim = -1) == 1).float()
        adjacency_matrix[:,torch.triu(torch.ones(max_num_agents, max_num_agents), diagonal = 1) == 1] = ((interaction_class.argmax(dim = -1) == 1) & 
                                                                                                            ((interaction_class.max(dim = -1))[0] > self.interaction_thresh)).float()
        edge_probs[:,torch.triu(torch.ones(max_num_agents, max_num_agents), diagonal = 1) == 1] = interaction_class[:, :, 1]

        # Set lower triangular part of adjacency matrix to 1 for agents if interaction class argmax is index 2
        idx, idy = torch.where(torch.tril(torch.ones(max_num_agents, max_num_agents), diagonal = -1) == 1) 
        idy_sorted, idy_sorted_ids = torch.sort(idy, stable=True)
        idx_sorted = torch.index_select(idx, 0, idy_sorted_ids)

        # adjacency_matrix[:,idx_sorted, idy_sorted] = (interaction_class.argmax(dim = -1) == 2).float()
        adjacency_matrix[:,idx_sorted, idy_sorted] = ((interaction_class.argmax(dim = -1) == 2) & 
                                                        ((interaction_class.max(dim = -1))[0] > self.interaction_thresh)).float()
        edge_probs[:,idx_sorted, idy_sorted] = interaction_class[:, :, 2]

        # Include self loops so that the nodes are at least included in the graph, even if they are not connected to any other node
        # Self loops will later be removed in the dagification step
        adjacency_matrix[:, torch.arange(max_num_agents), torch.arange(max_num_agents)] = existing_agent.float()
        edge_probs[:, torch.arange(max_num_agents), torch.arange(max_num_agents)] = 1

        edge_probs[adjacency_matrix==0] = 0

        Edge_bool = adjacency_matrix.bool()
        self.adjacency_matrix = adjacency_matrix
             
        # Find connection matrix of existing agents      

        # exist_sample2, exist_row2 = torch.where(existing_agent)
        exist_sample2, exist_row2 = torch.where(present_agent)
        exist_sample3, exist_row3, exist_col3 = torch.where(Edge_bool)
        
        # Check if all connections in Edge_bool are actually part of existing agents
        assert present_agent[exist_sample3, exist_row3].all(), 'Edge to nonexisting agent'
        assert present_agent[exist_sample3, exist_col3].all(), 'Edge to nonexisting agent'


        
        # Differentiate between agents of different samples
        node_adder = exist_sample3 * max_num_agents
        graph_edge_index = torch.stack([exist_row3, exist_col3]) + node_adder[None,:] # shape: 2 x num_edges
        
        # Eliminate the holes due to missing agents
        # i.e node ids go from (0,3,4,6,...) to (0,1,2,3,...)
        # graph_edge_index = torch.unique(graph_edge_index, return_inverse = True)[1] 


        ## ADAPTED FROM FJMP - Dagification
        print('    Dagification')
        new_graph_edge_index, new_node_addr, G, G_with_self_loops = dagification(graph_edge_index, edge_probs, node_adder, self.device)

        self.G = G
        self.G_with_self_loops = G_with_self_loops

        exist_row3 = new_graph_edge_index[0] - new_node_addr
        exist_col3 = new_graph_edge_index[1] - new_node_addr

        exist_sample3 = torch.floor_divide(new_node_addr, max_num_agents)

        Edge_bool = torch.zeros((x.shape[0], max_num_agents, max_num_agents), device=self.device)
        Edge_bool[exist_sample3, exist_row3, exist_col3] = 1

        self.adjacency_matrix = Edge_bool
        
        # Differentiate between agents of different samples
        graph_edge_index = new_graph_edge_index # shape: 2 x num_edges
        
        # Eliminate the holes due to missing agents
        # i.e node ids go from (0,3,4,6,...) to (0,1,2,3,...)
        graph_edge_index = torch.unique(graph_edge_index, return_inverse = True)[1] 
        max_id_after = graph_edge_index.max()
        
        # Check if all connections in Edge_bool are actually part of existing agents
        assert present_agent[exist_sample3, exist_row3].all(), 'Edge to nonexisting agent'
        assert present_agent[exist_sample3, exist_col3].all(), 'Edge to nonexisting agent'
                
        # Get type of existing agents
        T_exisiting_agents = T[exist_sample2, exist_row2] # shape: num_nodes
        # Get one hot encodeing of this agent type
        T_one_hot = (T_exisiting_agents.unsqueeze(-1) == self.t_unique.unsqueeze(0)).float() # shape: num_nodes x num_classes
        # Get one hot encoding from start and goal node of each edge
        T_one_hot_edge = T_one_hot[graph_edge_index, :] # shape: 2 x num_edges x num_classes
        # Concatenate in and output type for each edge
        class_in_out = T_one_hot_edge.permute(1,0,2).reshape(-1, 2 * len(self.t_unique)) # shape: num_edges x (2 num_classes)
        
        # Get distance along each edge
        dist_in_out = D[exist_sample3, exist_row3, exist_col3] # shape: num_edges x 2

        ang_in_out = ang[exist_sample3, exist_row3, exist_col3] # shape: num_edges x 1
        
        # Concatenate edge information
        graph_edge_attr = torch.cat((class_in_out, dist_in_out, ang_in_out[:,None]), dim = 1)



        # If graph info present, use LaneGCN to enrich actor features
        if self.use_graph:
            print('    Encoding Graph')
            graph = graph_gather(to_long(gpu(scene_graph)), self.lanegcn_configs)

            nodes, node_idcs, node_ctrs = self.MapNet(graph)


            actor_idcs = exist_sample2*max_num_agents + exist_row2

            # Find the indices where the value is 0
            zero_indices = torch.where(exist_row2 == 0)[0]

            # Append the length of the tensor to handle the last segment
            zero_indices = torch.cat((zero_indices, torch.tensor([exist_row2.size(0)]).to(self.device)))

            # Split the array into sequences
            actor_idcs = [actor_idcs[start:end] for start, end in zip(zero_indices[:-1], zero_indices[1:])]

            x_centered = x-x[:,0,-1].unsqueeze(1).unsqueeze(1)
            # Extract actor centers
            actor_ctrs = x_centered[exist_sample2, exist_row2, -1]
            # Split the array into sequences
            actor_ctrs = [actor_ctrs[start:end] for start, end in zip(zero_indices[:-1], zero_indices[1:])]

            # Extract actor features
            actors = x_enc[exist_sample2, exist_row2]
            # Split the array into sequences
            # actors = [actors[start:end] for start, end in zip(zero_indices[:-1], zero_indices[1:])]

            actors = self.M2A(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)

            x_enc[exist_sample2, exist_row2] = actors

         
        

        print('    Encoding interactions')
        # Get encoded trajectory for each edge
        x_enc_existing_agents = x_enc[exist_sample2, exist_row2] # shape: num_nodes x enc_size   

        
        # Apply the graph neural network to the batch 
        socialData = Data(x          = x_enc_existing_agents, 
                        edge_index = graph_edge_index, 
                        edge_attr  = graph_edge_attr, 
                        batch      = exist_sample2)
        context = self.GNNencoder(socialData)

        if scene is not None:
            print('    Encoding images')
            scene_existing = scene[exist_sample2, exist_row2]
            scene_enc_flattened = self.scene_encoder(scene_existing)
            # scene_enc = scene_enc_flattened.reshape(scene.shape[0], scene.shape[1], -1)

            context = torch.cat((context, scene_enc_flattened), dim=1)

        self.topo_order, self.topo_order_pred = kahn_toposort(self.G, present_agent.sum())
        
        return context, exist_sample2, exist_row2

    def log_prob(self, y_true, x, T, scene=None, scene_graph=None):
        z, log_abs_jacobian_det, batch_ids, row_ids = self._inverse(y_true, x, T, scene, scene_graph)
        normal = Normal(0, 1, validate_args=True)
        num_agents_per_sample = torch.bincount(batch_ids)
        likelihoods = normal.log_prob(z).sum(-1) + log_abs_jacobian_det


        # _, key_counts = np.unique(batch_ids.cpu().numpy(), return_counts=True)
        # joint_likelihoods = np.add.reduceat(likelihoods.flatten().detach().cpu().numpy(), np.cumsum(key_counts) - key_counts)
        # joint_likelihoods = torch.tensor(joint_likelihoods).to(self.device)
        # sum over agents
        joint_likelihoods = torch.zeros(x.shape[0], device=self.device)
        joint_likelihoods.scatter_add_(dim=0, index=batch_ids, src=likelihoods)

        return likelihoods, joint_likelihoods, num_agents_per_sample

    def _inverse(self, y_true, x, T, scene=None, scene_graph=None):
        x_enc, batch_ids, row_ids = self._encode_conditionals(x, T, scene, scene_graph) # history encoding
        y_true = y_true.reshape(x.shape[0], x.shape[1],-1)
        y_true = y_true[batch_ids, row_ids]

        print('    Normalizing Flow')
        max_num_agents = x.shape[1]
        topo_order = self.topo_order.copy()
 
        z = torch.zeros_like(y_true)
        jacobian_det = torch.zeros(y_true.shape[0], device=self.device)
 
        L_given = row_ids + batch_ids * max_num_agents
        L = []
        L_length = []
        L_pred = []
        L_pred_length = []
        J_pred = []
        for i, l in enumerate(topo_order.values()):
            L += l
            L_length.append(len(l))
 
            l_pred_length = 0
            if i in self.topo_order_pred.keys():
                for j, t_pred in enumerate(self.topo_order_pred[i]):
                    L_pred += t_pred
                    l_pred_length += len(t_pred)
                    J_pred += [j] * len(t_pred)
            L_pred_length.append(l_pred_length)
 
        L = torch.tensor(L, device=self.device)
        L_pred = torch.tensor(L_pred, device=self.device)
        J_pred = torch.tensor(J_pred, device=self.device)
 
        # Map L onto L_given
        test_ids, L_ids = torch.where(L.unsqueeze(1) == L_given.unsqueeze(0))
        assert (test_ids == torch.arange(len(L), device=self.device)).all()
 
        # Map L_pred onto L_given
        test_ids, L_pred_ids = torch.where(L_pred.unsqueeze(1) == L_given.unsqueeze(0))
        assert (test_ids == torch.arange(len(L_pred), device=self.device)).all()
 
        # Expand J_pred dimension
        for s in range(1, len(y_true.shape)):
            J_pred = J_pred.unsqueeze(-1).repeat_interleave(y_true.shape[s], dim=-1)
        
        i_start = 0
        i_start_pred = 0
        for i, length in enumerate(L_length):
            # Get the current agents
            use_i = torch.arange(i_start, i_start + length, device=self.device)
            i_start += length
 
            # get corresponding agent information
            l_ids = L_ids[use_i]
            x_enc_relevant = x_enc[l_ids]
            y_true_relevant = y_true[l_ids]
            y_rel_flat = y_true_relevant           
 
            # Get the current pred agents
            length_pred = L_pred_length[i]
            use_pred_i = torch.arange(i_start_pred, i_start_pred + length_pred, device=self.device)
            i_start_pred += length_pred
 
            if i == 0:
                x_enc_relevant = torch.concat((x_enc_relevant, torch.zeros(y_true_relevant.shape, device=self.device)), dim=-1)
 
            else:
                j_pred = J_pred[use_pred_i]
                l_pred_ids = L_pred_ids[use_pred_i]
                y_true_use = y_true[l_pred_ids]
 
                # Sum up y_true_use for each unique j
                predecessors_pred = torch.zeros((j_pred.max() + 1, *y_true_use.shape[1:]), device=self.device)
                predecessors_pred = predecessors_pred.scatter_add(0, j_pred, y_true_use)
 
                x_enc_relevant = torch.concat((x_enc_relevant, predecessors_pred), dim=-1)
 
            if self.training:
                # add noise to zero values to avoid infinite density points
                zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
                noise = torch.randn_like(y_rel_flat) * self.beta
                y_rel_flat = y_rel_flat + (zero_mask * noise)
 
                # minimally perturb remaining motion to avoid x1 = x2 for any values
                noise = torch.randn_like(y_rel_flat) * self.gamma
                y_rel_flat = y_rel_flat + (~zero_mask * noise)
            
            z_relevant, jacobian_det_relevant = self.flow.inverse(y_rel_flat, x_enc_relevant)
 
            z[l_ids] = z_relevant
            jacobian_det[l_ids] = jacobian_det_relevant
 
        return z, jacobian_det, batch_ids, row_ids
    
    def sample(self, n, x, T, scene=None, scene_graph=None):
        with torch.no_grad():
            normal = Normal(0, 1, validate_args=True)

            x_enc, batch_ids, row_ids = self._encode_conditionals(x, T, scene, scene_graph) # history encoding

            max_num_agents = x.shape[1]
            topo_order = self.topo_order.copy()

            z = torch.randn([x.shape[0], x.shape[1], n, self.output_size]).to(self.device)
            z = z[batch_ids,row_ids]
            # initialise samples_rel with zeros
            samples_rel = torch.zeros(x_enc.shape[0], n, self.pred_steps).to(self.device)
            # initialise log_det with zeros
            log_det = torch.zeros(x_enc.shape[0], n).to(self.device)

            for i in range(len(topo_order)):
                L = np.array(topo_order[i])
                L_batch = L//max_num_agents
                L_row = L%max_num_agents

                lookup_matrix = np.array(list(zip(batch_ids.cpu().tolist(), row_ids.cpu().tolist())))
                L_indices = np.array(list(zip(L_batch, L_row)))

                relevant_ids = np.where(np.all(lookup_matrix[:,None,:] == L_indices[None, :, :], axis=2))[0]

                x_enc_relevant = x_enc[relevant_ids]

                x_enc_expanded = self._repeat_rowwise(x_enc_relevant, n)

                if i == 0:
                    x_enc_expanded = torch.concat((x_enc_expanded, torch.zeros((x_enc_expanded.shape[0], 
                                                                                x_enc_expanded.shape[1], 
                                                                                self.output_size), device=self.device)), dim=-1)

                else:
                    predecessors = self.topo_order_pred[i].copy()
                    predecessors_ids = np.array([np.where(np.all(lookup_matrix[:,None,:] == 
                                                                  np.array(list(zip(np.array(pred)//max_num_agents, 
                                                                                    np.array(pred)%max_num_agents)))[None, :, :], 
                                                                  axis=2))[0] for pred in predecessors])

                    predecessors_pred = torch.tensor([torch.sum(samples_rel[predecessor], dim=0).cpu().tolist() for predecessor in predecessors_ids], device=self.device)

                    x_enc_expanded = torch.concat((x_enc_expanded, predecessors_pred), dim=-1)

                if self.use_map:
                    x_enc_expanded = x_enc_expanded.view(-1, self.es_gnn+self.scene_encoding_size+self.output_size)
                else:
                    x_enc_expanded = x_enc_expanded.view(-1, self.es_gnn+self.output_size)
                    
                output_shape = (x_enc_relevant.size(0), n, self.pred_steps) # predict n trajectories input

                # sample and compute likelihoods
                # z_relevant = z[relevant_ids].reshape(-1,self.output_size)
                if i==0:
                    z_relevant = z[relevant_ids].reshape(-1,self.output_size)

                    samples_rel_relevant, log_det_relevant = self.flow(z_relevant, x_enc_expanded)

                else:
                    torch.cuda.empty_cache()
                    # per prediction draw 15 samples in order to then pick the most likely
                    scoring_samples = 1#5

                    z_relevant = torch.randn([x_enc_expanded.shape[0], scoring_samples, self.output_size]).to(self.device) 
                    
                    x_enc_expanded_scoring = self._repeat_rowwise(x_enc_expanded, scoring_samples)
                    x_enc_expanded_scoring = x_enc_expanded_scoring.reshape(-1, x_enc_expanded.shape[-1])

                    samples_rel_relevant, log_det_relevant = self.flow(z_relevant.reshape(-1, self.output_size), x_enc_expanded_scoring)
                    log_det_relevant = log_det_relevant.reshape([x_enc_expanded.shape[0], scoring_samples])
                    samples_rel_relevant = samples_rel_relevant.reshape([x_enc_expanded.shape[0], scoring_samples, self.output_size])

                    log_probs = (normal.log_prob(z_relevant).sum(-1) + log_det_relevant)
                    max_log_ids = log_probs.max(axis=1)[1]

                    log_det_relevant = log_det_relevant[torch.arange(log_det_relevant.size(0)),max_log_ids]
                    samples_rel_relevant = samples_rel_relevant[torch.arange(samples_rel_relevant.size(0)),max_log_ids]
                    z_relevant = z_relevant[torch.arange(z_relevant.size(0)), max_log_ids]

                    z[relevant_ids] = z_relevant.reshape(relevant_ids.shape[0], n, self.output_size)


                samples_rel_relevant = samples_rel_relevant.view(*output_shape)
                log_det_relevant = log_det_relevant.view(*output_shape[:-1])

                log_det[relevant_ids] = log_det_relevant
                samples_rel[relevant_ids] = samples_rel_relevant

            sample_ids = torch.arange(n, device=self.device).repeat(x_enc.shape[0])
            # sample_ids = batch_ids.repeat(n) + torch.arange(n, device=self.device).repeat(x_enc.shape[0])*x.shape[0] #torch.sort(torch.arange(n, device=self.device).repeat(x_enc.shape[0]))[0]*x.shape[0]
            sample_ids = batch_ids.repeat(n) + torch.sort(torch.arange(n, device=self.device).repeat(x_enc.shape[0]))[0]*x.shape[0]

            log_probs = (normal.log_prob(z).sum(-1) + log_det)

            # _, key_counts = np.unique(sample_ids.cpu().numpy(), return_counts=True)
            # joint_likelihoods = np.add.reduceat(log_probs.flatten().cpu().numpy(), np.cumsum(key_counts) - key_counts)
            # joint_likelihoods = torch.tensor(joint_likelihoods.reshape(-1, n)).to(self.device)
            joint_likelihoods = torch.zeros(x.shape[0]*n, device=self.device)
            joint_likelihoods.scatter_add_(dim=0, index=sample_ids, src=log_probs.T.flatten())
            joint_likelihoods = joint_likelihoods.reshape(n, -1).T
            
            return samples_rel, log_probs, joint_likelihoods, batch_ids, row_ids
        

    def predict_expectation(self, n, x):
        samples = self.sample(n, x)
        y_pred = samples.mean(dim=1, keepdim=True)
        return y_pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    
class Future_Encoder(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=4, hs=4, nl=3, device=0):
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden
    
    
class Future_Decoder(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, es=4, hs=4, nl=3, device=0):
        super().__init__()
        self.output = nn.Linear(es, nin)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.device = device
        self.cuda(self.device)
        
        self.embedding = nn.Linear(es, es)
        self.nl = nl

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.output(x)
        return x, hidden


    
    
class Future_Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Future_Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, traj):
        batch_size = traj.shape[0]
        
        target_length = traj.shape[1]
        
        outputs = torch.zeros(batch_size, target_length, 2).to(device)
        
        x_in = traj
        
        x_enc, hidden = self.encoder(x_in) # encode relative histories
        out = x_enc
                
        hidden = torch.tile(out[:,-1].unsqueeze(0), (self.decoder.nl,1,1))
        
        # Decoder part
        x = out[:,-1].unsqueeze(1)
                
        for t in range(0, target_length):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = output.squeeze()
            
            x = hidden[-1].unsqueeze(1)
        
        return outputs, x_in