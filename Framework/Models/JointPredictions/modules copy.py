import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter
from torch_geometric.data import Data

class TrajRNNEncoder(nn.Module):
    """ 
    GRU based recurrent neural network. 
    Used for encoding past trajectories.
    """

    def __init__(self, nin, nout, es, hs, nl, device=0):
        """ 
        Args:
            nin (int): number of input features
            nout (int): number of output features
            es (int): embedding size
            hs (int): hidden size
            nl (int): number of layers
            device (int): device to use
        """

        super().__init__()
        self.device = device
        
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        

    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, nin)
            hidden (torch.Tensor): hidden state of shape (nl, batch_size, hs)
        Returns:
            x (torch.Tensor): output tensor of shape (batch_size, seq_len, nout)
            hidden (torch.Tensor): hidden state of shape (nl, batch_size, hs)
        """
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden
    
class TrajRNNDecoder(nn.Module):
    """ GRU based recurrent neural network for decoding future trajectories. """

    def __init__(self, nout, es=4, hs=4, nl=3, device=0):
        """
        Args:
            nout (int): number of output features
            es (int): embedding size
            hs (int): hidden size
            nl (int): number of layers
            device (int): device to use
        """
        super().__init__()
        self.device = device

        self.embedding = nn.Linear(es, es)
        self.nl = nl
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output = nn.Linear(es, nout)        

    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, 1, es)
            hidden (torch.Tensor): hidden state of shape (nl, batch_size, hs)
        Returns:
            x (torch.Tensor): output tensor of shape (batch_size, 1, nout)
            hidden (torch.Tensor): hidden state of shape (nl, batch_size, hs)
        """
        x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.output(x)
        return x, hidden
       
class StaticEnvCNN(nn.Module):
    """ CNN based static environment encoder. """

    def __init__(self, img_dim, nin, ks=[5,5,5], strides=[4,4,4], paddings=[1,1,0], enc_dim=8, device=0):
        """ 
        Args:
            img_dim (list): image dimensions
            nin (int): number of input features
            ks (int): kernel size
            strides (list): list of strides
            paddings (list): list of paddings
            enc_dim (int): encoding dimension
            device (int): device to use
        """
        super().__init__()
        self.device = device

        ### Convolutional layers
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(nin, 8, kernel_size=ks, stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=ks, stride=strides[1], padding=paddings[1]),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=ks, stride=strides[2], padding=paddings[2]),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        h_out_l1 = int((img_dim[0] + 2*paddings[0]-1*(ks-1)-1)/strides[0] + 1)
        w_out_l1 = int((img_dim[1] + 2*paddings[0]-1*(ks-1)-1)/strides[0] + 1)
        h_out_l2 = int((h_out_l1 + 2*paddings[1]-1*(ks-1)-1)/strides[1] + 1)
        w_out_l2 = int((w_out_l1 + 2*paddings[1]-1*(ks-1)-1)/strides[1] + 1)
        h_out_l3 = int((h_out_l2 + 2*paddings[2]-1*(ks-1)-1)/strides[2] + 1)
        w_out_l3 = int((w_out_l2 + 2*paddings[2]-1*(ks-1)-1)/strides[2] + 1)

        ### Flatten layer
        self.flatten = nn.Flatten()

        ### Fully connected layer
        self.encoder_fc = nn.Sequential(
            nn.Linear(32*h_out_l3*w_out_l3, 128),
            nn.ReLU(),
            nn.Linear(128, enc_dim)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, nin, img_dim[0], img_dim[1])
        Returns:
            x (torch.Tensor): output tensor of shape (batch_size, enc_dim)
        """
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_fc(x)
        return x

# TODO: remove the commented dropout if possible    
class MPNNLayer(MessagePassing):
    """ 
    Message Passing Neural Network Layer.
    Implementation taken from GDL lecture practical 3 (https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR)
    """

    def __init__(self, emb_dim, edge_dim, aggr='add', device=0):
        """Message Passing Neural Network Layer

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            device (int): device to use
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.device = device

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim + edge_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), 
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim), 
            nn.Tanh()
          )

    def forward(self, h, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        As our MPNNLayer class inherits from the PyG MessagePassing parent class,
        we simply need to call the `propagate()` function which starts the 
        message passing procedure: `message()` -> `aggregate()` -> `update()`.
        
        The MessagePassing class handles most of the logic for the implementation.
        To build custom GNNs, we only need to define our own `message()`, 
        `aggregate()`, and `update()` functions (defined subsequently).

        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        """Step (1) Message

        The `message()` function constructs messages from source nodes j 
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take 
        any arguments that were initially passed to `propagate`. Additionally, 
        we can differentiate destination nodes and source nodes by appending 
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`. 
        
        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features
        
        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)
    
    def aggregate(self, inputs, index):
        """Step (2) Aggregate

        The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        """
        Step (3) Update

        The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        `update()` takes the first argument `aggr_out`, the result of `aggregate()`, 
        as well as any optional arguments that were initially passed to 
        `propagate()`. E.g. in this case, we additionally pass `h`.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
      
class SocialInterGNN(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4):
        """Message Passing Neural Network model for encoding social interactions.
        Implementation taken and modified from GDL lecture practical 3 (https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR)

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
        """
        super().__init__()
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = nn.Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        return h_graph
    
class SocialInterDecGNN(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4):
        """Message Passing Neural Network model for encoding social interactions.
        Implementation taken and modified from GDL lecture practical 3 (https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR)

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
        """
        super().__init__()
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = nn.Linear(in_dim, emb_dim)
        
        # Stack of MPNN layers
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)
        
        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = h 

        return h_graph
    
### Start of modules for full pipeline

class PastSceneEncoder(nn.Module):
    """Module for encoding past scene information."""

    def __init__(self, num_layers, emb_dim, in_dim, edge_dim, T_all=None, device=0):
        super().__init__()
        self.device = device
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))

        self.in_dim = in_dim
        # Module for encoding social interactions
        self.encoder = SocialInterGNN(num_layers=num_layers,
                                      emb_dim=emb_dim,
                                      in_dim=in_dim,
                                      edge_dim=edge_dim)

    def forward(self, pos, x_enc, T):
        """
        Args: 
            pos: (batch_size, n_agents, 2) - current position of all agents
            x_enc: (batch_size, n_agents, seq_len, past_nout) - encoded past trajectories of all agents
            T: (batch_size, n_agents) - class labels for all agents
        """
        # print('Past Scene Encoder')
        # print('past trajectory shape: ', pos.shape)
        # print('past trajectory encoding shape: ', x_enc.shape)
        # print('T shape: ', T.shape)
        
        x_enc     = x_enc[...,-1,:]
        num_nodes = x_enc.size(dim=1)
        row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
        col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        graph_edge_node = torch.stack([row, col], dim=0).unsqueeze(0).repeat((len(x_enc),1,1))
        # Combine batch into one matrix
        sample_enc = torch.arange(len(x_enc), device = self.device).unsqueeze(1).repeat((1,num_nodes ** 2)) 
        graph_edge_index = graph_edge_node + sample_enc.unsqueeze(1) * num_nodes # get indices for separating out each sample
        # Combine
        graph_edge_index = graph_edge_index.permute(1,0,2).reshape(2,-1) # (2, n_agents ** 2 * batch_size)

        T_one_hot = (T.unsqueeze(-1) == self.t_unique.unsqueeze(0).unsqueeze(0)).float().reshape(-1, len(self.t_unique))

        class_in_out = T_one_hot[graph_edge_index, :].permute(1,0,2).reshape(-1, 2 * len(self.t_unique))
        
        D = torch.sqrt(torch.sum((pos[:,None,:] - pos[:,:,None]) ** 2, dim = -1))
        dist_in_out = D[sample_enc, graph_edge_node[:,0], graph_edge_node[:,1]]
        dist_in_out = dist_in_out.reshape(-1,1)
         
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        
        graph_batch = torch.arange((len(x_enc)), dtype=torch.long, device=self.device).repeat_interleave(num_nodes)

        socialData = Data(x=x_enc.reshape(-1, self.in_dim), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=graph_batch)
        
        sceneEncoding = self.encoder(socialData) # (batch_size, emb_dim)

        return sceneEncoding


class FutureTrajEncoder(nn.Module):
    
    def __init__(self, futureRNNencParams, device=0):
        super().__init__()
        self.device = device

        # Module for encoding past trajectories of agents and ego vehicle
        self.rnn = TrajRNNEncoder(nin=futureRNNencParams['nin'],
                                  nout=futureRNNencParams['nout'],
                                  es=futureRNNencParams['es'],
                                  hs=futureRNNencParams['hs'],
                                  nl=futureRNNencParams['nl'])

    def forward(self, agentTrajs):
        agentFutureTrajEnc, _ = self.rnn(agentTrajs) # (n_agents, seq_len, hs)
        agentFutureTrajEnc = agentFutureTrajEnc[:, -1, :] # (n_agents, hs)

        return agentFutureTrajEnc
    
class FutureTrajDecoder(nn.Module):

    def __init__(self, futureRNNdecParams, device=0):
        super().__init__()
        self.device = device

        # Module for decoding future trajectories of agents and ego vehicle
        self.rnn = TrajRNNDecoder(futureRNNdecParams['nout'],
                                  futureRNNdecParams['es'],
                                  futureRNNdecParams['hs'],
                                  futureRNNdecParams['nl']).to(device)

    def forward(self, agentFutureTrajEnc, target_length, batch_size):
        outputs = torch.zeros(batch_size, target_length, 2).to(self.device)

        x = agentFutureTrajEnc # (n_agents, emb_dim)
        x = x.unsqueeze(1) # (n_agents, 1, emb_dim)

        hidden = agentFutureTrajEnc # (n_agents, emb_dim)
        hidden = hidden.unsqueeze(0) # (1, n_agents, emb_dim)
        hidden = hidden.repeat(self.rnn.nl, 1, 1) # (nl, n_agents, emb_dim)

        for t in range(target_length):
            output, hidden = self.rnn(x, hidden)
            outputs[:, t, :] = output.squeeze()

            x = hidden[-1].unsqueeze(1) # (n_agents, 1, 2*enc_dim)

        return outputs
        

# class FutureSceneEncoder(nn.Module):
#     """Module for encoding past scene information."""

#     def __init__(self, socialGNNparams, enc_dim, pos_emb_dim, device=0):
#         super().__init__()
#         self.device = device
       
        
#         # Module for encoding social interactions
#         self.encoder = SocialInterGNN(num_layers=socialGNNparams['num_layers'],
#                                       emb_dim=socialGNNparams['emb_dim'],
#                                       in_dim=socialGNNparams['in_dim'] + pos_emb_dim + 1,
#                                       edge_dim=socialGNNparams['edge_dim'])
        
        
#         ### Fully connected layer
#         self.encoder_fc = nn.Sequential(
#             nn.Linear(socialGNNparams['emb_dim'], int(socialGNNparams['emb_dim']/2)),
#             nn.Tanh(),
#             nn.Linear(int(socialGNNparams['emb_dim']/2), enc_dim)
#         )

#     def forward(self, agentFutureTrajEnc, pos_emb, numAgents_emb): #, agentPos):
        
#         agentContext = agentFutureTrajEnc # (n_agents, hs)

#         sceneContext = agentContext # (n_agents, hs)

#         graph_x = sceneContext

#         # print('pos_emb', pos_emb.shape)
#         # print('numAgents_emb', numAgents_emb.shape)
#         # print('graph_x', graph_x.shape)

#         graph_x = torch.cat((pos_emb, numAgents_emb, graph_x), dim=1)

#         #set edge indices; assuming full connectivity
#         num_nodes = sceneContext.size(dim=0)
#         row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
#         col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

#         row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
#         col = col.repeat(num_nodes)
#         graph_edge_index = torch.stack([row, col], dim=0)

#         graph_edge_attr = torch.ones((num_nodes**2, 1), device=self.device)
#         graph_batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

#         socialData = Data(x=graph_x, edge_index=graph_edge_index, edge_attr=graph_edge_attr, batch=graph_batch)

#         graphEncoding = self.encoder(socialData) # (batch_size, emb_dim)
        
#         sceneEncoding = self.encoder_fc(graphEncoding)

#         return sceneEncoding
    
class FutureSceneEncoder(nn.Module):
    """Module for encoding past scene information."""

    def __init__(self, socialGNNparams, enc_dim, pos_emb_dim, T_all=None, device=0):
        super().__init__()
        self.device = device
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.in_dim = socialGNNparams['in_dim']# + pos_emb_dim + 1
       
        
        # Module for encoding social interactions
        self.encoder = SocialInterGNN(num_layers=socialGNNparams['num_layers'],
                                      emb_dim=socialGNNparams['emb_dim'],
                                      in_dim=socialGNNparams['in_dim'],# + pos_emb_dim + 1,
                                      edge_dim=socialGNNparams['edge_dim'])
        
        
        ### Fully connected layer
        self.encoder_fc = nn.Sequential(
            nn.Linear(socialGNNparams['emb_dim'], int(socialGNNparams['emb_dim']/2)),
            nn.Tanh(),
            nn.Linear(int(socialGNNparams['emb_dim']/2), enc_dim)
        )

    def forward(self, pos, x_enc, pos_emb, numAgents_emb, T):

        """
        Args: 
            x: (batch_size, n_agents, 1, 2) - past trajectories of all agents
            x_enc: (batch_size, n_agents, fut_nout) - encoded future trajectories of all agents
            pos_emb: (batch_size, n_agents, pos_emb_dim) - positional encoding of all agents
            numAgents_emb: (batch_size, n_agents, 1) - encoding of number of agents in the scene 
            T: (batch_size, n_agents) - class labels for all agents
        """
        # print('Future Scene Encoder')
        # print('past position shape: ', pos.shape)
        # print('future trajectory encoding shape: ', x_enc.shape)
        # print('positional encoding shape: ', pos_emb.shape)
        # print('number of agents encoding shape: ', numAgents_emb.shape)
        # print('T shape: ', T.shape)
        # print('in dim: ', self.in_dim)
        
        num_nodes = x_enc.size(dim=1)
        row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
        col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        graph_edge_node = torch.stack([row, col], dim=0).unsqueeze(0).repeat((len(x_enc),1,1))
        # Combine batch into one matrix
        sample_enc = torch.arange(len(x_enc), device = self.device).unsqueeze(1).repeat((1,num_nodes ** 2)) 
        graph_edge_index = graph_edge_node + sample_enc.unsqueeze(1) * num_nodes # get indices for separating out each sample
        # Combine
        graph_edge_index = graph_edge_index.permute(1,0,2).reshape(2,-1) # (2, n_agents ** 2 * batch_size)

        T_one_hot = (T.unsqueeze(-1) == self.t_unique.unsqueeze(0).unsqueeze(0)).float().reshape(-1, len(self.t_unique))

        class_in_out = T_one_hot[graph_edge_index, :].permute(1,0,2).reshape(-1, 2 * len(self.t_unique))
        
        # if len(pos.shape) == 4:
        D = torch.sqrt(torch.sum((pos[:,None,:] - pos[:,:,None]) ** 2, dim = -1))
        dist_in_out = D[sample_enc, graph_edge_node[:,0], graph_edge_node[:,1]]
        # else:
        #     D = torch.sqrt(torch.sum((pos[None,:] - pos[:,None]) ** 2, dim = -1))
        #     dist_in_out = D[graph_edge_node[:,0], graph_edge_node[:,1]]
        # print('D shape: ', D.shape)
        dist_in_out = dist_in_out.reshape(-1,1)
         
        # TODO possibly put back
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        # graph_edge_attr = torch.ones((graph_edge_index.shape[1], 1), device=self.device)
        
        graph_batch = torch.arange((len(x_enc)), dtype=torch.long, device=self.device).repeat_interleave(num_nodes)

        # graph_x = torch.cat((pos_emb, numAgents_emb, x_enc), dim=1)
        # socialData = Data(x=graph_x.reshape(-1, self.in_dim), 
        #                   edge_index=graph_edge_index, 
        #                   edge_attr=graph_edge_attr, 
        #                   batch=graph_batch)

        # print('x_enc shape: ', x_enc.reshape(-1, self.in_dim).shape)
        # print('graph_edge_index shape: ', graph_edge_index.shape)
        # print('graph_edge_attr shape: ', graph_edge_attr.shape)
        # print('graph_batch shape: ', graph_batch.shape)
        socialData = Data(x=x_enc.reshape(-1, self.in_dim), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=graph_batch)

        
        graphEncoding = self.encoder(socialData) # (batch_size, emb_dim)
        
        # print('graphEncoding shape: ', graphEncoding.shape)
        sceneEncoding = self.encoder_fc(graphEncoding)

        return sceneEncoding
    

# class FutureSceneDecoder(nn.Module):
#     """Module for encoding past scene information."""

#     def __init__(self, socialGNNparams, enc_dim, pos_emb_dim, device=0):
#         super().__init__()
#         self.device = device
        
#         # Module for encoding social interactions
#         self.decoder = SocialInterDecGNN(num_layers=socialGNNparams['num_layers'],
#                                       emb_dim=socialGNNparams['emb_dim'],
#                                       in_dim=socialGNNparams['emb_dim'] + pos_emb_dim + 1,
#                                       edge_dim=socialGNNparams['edge_dim'])
        
#         ### Fully connected layer
#         self.decoder_fc = nn.Sequential(
#             nn.Linear(enc_dim, int(socialGNNparams['emb_dim']/2)),
#             nn.Tanh(),
#             nn.Linear(int(socialGNNparams['emb_dim']/2), socialGNNparams['emb_dim'])
#         )
        

#     # def forward(self, enc, num_agents, agentPos, target_length): 
#     def forward(self, enc, pos_emb, numAgents_emb, num_agents): 
#         enc_emb = self.decoder_fc(enc)
        
#         graph_x = enc_emb.repeat(num_agents, 1)

#         # print("Graph decoder")
#         # print('pos_emb', pos_emb.shape)
#         # print('numAgents_emb', numAgents_emb.shape)
#         # print('graph_x', graph_x.shape)
#         graph_x = torch.cat((pos_emb, numAgents_emb, graph_x), dim=1) # (n_agents, enc_dim + pos_emb_dim + 1)

#         num_nodes = num_agents
#         row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
#         col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

#         row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
#         col = col.repeat(num_nodes)
#         graph_edge_index = torch.stack([row, col], dim=0)

#         graph_edge_attr = torch.ones((num_nodes**2, 1), device=self.device)
#         graph_batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

#         encSocialData = Data(x=graph_x, edge_index=graph_edge_index, edge_attr=graph_edge_attr, batch=graph_batch)
        
#         graphDecoding = self.decoder(encSocialData) # (n_agents, emb_dim)

#         return graphDecoding
    
class FutureSceneDecoder(nn.Module):
    """Module for encoding past scene information."""

    def __init__(self, socialGNNparams, enc_dim, pos_emb_dim, T_all=None, device=0):
        super().__init__()
        self.device = device
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.in_dim = socialGNNparams['in_dim'] + pos_emb_dim + 1
        
        # Module for encoding social interactions
        self.decoder = SocialInterDecGNN(num_layers=socialGNNparams['num_layers'],
                                      emb_dim=socialGNNparams['emb_dim'],
                                      in_dim=socialGNNparams['emb_dim'] + pos_emb_dim + 1,
                                      edge_dim=socialGNNparams['edge_dim'])
        
        ### Fully connected layer
        self.decoder_fc = nn.Sequential(
            nn.Linear(enc_dim, int(socialGNNparams['emb_dim']/2)),
            nn.Tanh(),
            nn.Linear(int(socialGNNparams['emb_dim']/2), socialGNNparams['emb_dim'])
        )

    # def forward(self, enc, num_agents, agentPos, target_length): 
    def forward(self, pos, enc, pos_emb, numAgents_emb, num_agents, T): 

        """
        Args: 
            x: (batch_size, n_agents, seq_len, 2) - current position of all agents
            enc: (batch_size, n_agents, seq_len, fut_nout) - encoded future scene
            pos_emb: (batch_size, n_agents, pos_emb_dim) - positional encoding of all agents
            numAgents_emb: (batch_size, n_agents, 1) - encoding of number of agents in the scene 
            T: (batch_size, n_agents) - class labels for all agents
        """
        # print('Future Scene Decoder')
        # print('past position shape: ', pos.shape)
        # print('future scene encoding shape: ', enc.shape)
        # print('positional encoding shape: ', pos_emb.shape)
        # print('number of agents encoding shape: ', numAgents_emb.shape)
        # print('number of agents: ', num_agents)
        # print('T shape: ', T.shape)

        enc_emb = self.decoder_fc(enc)
        enc_emb = enc_emb.unsqueeze(1).repeat((1,num_agents,1))
        # print('enc_emb shape: ', enc_emb.shape)

        numAgents_emb = numAgents_emb.unsqueeze(1).repeat((1,num_agents,1))
        # print('numAgents_emb shape: ', numAgents_emb.shape)

        num_nodes = num_agents
        row = torch.arange(num_nodes, dtype=torch.long, device=self.device)
        col = torch.arange(num_nodes, dtype=torch.long, device=self.device)

        row = row.view(-1, 1).repeat(1, num_nodes).view(-1)
        col = col.repeat(num_nodes)
        graph_edge_node = torch.stack([row, col], dim=0).unsqueeze(0).repeat((len(enc),1,1))
        # Combine batch into one matrix
        sample_enc = torch.arange(len(enc), device = self.device).unsqueeze(1).repeat((1,num_nodes ** 2)) 
        graph_edge_index = graph_edge_node + sample_enc.unsqueeze(1) * num_nodes # get indices for separating out each sample
        # Combine
        graph_edge_index = graph_edge_index.permute(1,0,2).reshape(2,-1) # (2, n_agents ** 2 * batch_size)

        T_one_hot = (T.unsqueeze(-1) == self.t_unique.unsqueeze(0).unsqueeze(0)).float().reshape(-1, len(self.t_unique))

        class_in_out = T_one_hot[graph_edge_index, :].permute(1,0,2).reshape(-1, 2 * len(self.t_unique))
        
        
        # if len(pos.shape) == 4:
        D = torch.sqrt(torch.sum((pos[:,None,:] - pos[:,:,None]) ** 2, dim = -1))
        dist_in_out = D[sample_enc, graph_edge_node[:,0], graph_edge_node[:,1]]
        # else:
        #     D = torch.sqrt(torch.sum((pos[None,:] - pos[:,None]) ** 2, dim = -1))
        #     dist_in_out = D[graph_edge_node[:,0], graph_edge_node[:,1]]

        dist_in_out = dist_in_out.reshape(-1,1)
         
         # TODO possibly put back
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        # graph_edge_attr = torch.ones((graph_edge_index.shape[1], 1), device=self.device)
        
        graph_batch = torch.arange((len(enc)), dtype=torch.long, device=self.device).repeat_interleave(num_nodes)

        graph_x = torch.cat((pos_emb, numAgents_emb, enc_emb), dim=2)
        encSocialData = Data(x=graph_x.reshape(-1, self.in_dim), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=graph_batch)

        
        graphDecoding = self.decoder(encSocialData) # (n_agents, emb_dim)

        return graphDecoding

class FutureSceneAE(nn.Module):

    def __init__(self, futureRNNencParams, futureRNNdecParams, socialGNNparams, enc_dim, pos_emb_dim, T_all, device=0):
        super().__init__()
        self.device = device

        self.scene_encoder = FutureSceneEncoder(socialGNNparams, enc_dim, pos_emb_dim, T_all=T_all, device=device).to(device)
        self.scene_decoder = FutureSceneDecoder(socialGNNparams, enc_dim, pos_emb_dim, T_all=T_all, device=device).to(device)

        # self.scene_encoder = FutureSceneEncoder(socialGNNparams, enc_dim, pos_emb_dim).to(device)
        # self.scene_decoder = FutureSceneDecoder(socialGNNparams, enc_dim, pos_emb_dim).to(device)
        
        # self.traj_encoder = FutureTrajEncoder(futureRNNencParams, device=device).to(device)
        # self.traj_decoder = FutureTrajDecoder(futureRNNdecParams, device=device).to(device)

        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        for t in self.t_unique:
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            self.traj_encoder[t_key] = FutureTrajEncoder(futureRNNencParams, device=device).to(device)
            self.traj_decoder[t_key] = FutureTrajDecoder(futureRNNdecParams, device=device).to(device)

            self.traj_encoder[t_key].to(device)
            self.traj_decoder[t_key].to(device)
            
            
                                               
        self.pos_emb = nn.Linear(2, pos_emb_dim).to(device)
        self.numAgents_emb = nn.Linear(1, 1).to(device)

    def forward(self, agentTrajs, agentPos, T):
        num_agents = agentTrajs.size(dim=0)

        # print('agent position shape: ', agentPos.shape)
        pos_emb = F.tanh(self.pos_emb(agentPos)) # (n_agents, enc_dim)
        numAgents_emb = F.tanh(self.numAgents_emb(torch.tensor(num_agents).float().to(self.device).unsqueeze(0))) # (1, 1)
        numAgents_emb = numAgents_emb.repeat(num_agents, 1) # (n_agents, 1)


        # target_length = agentTrajs.size(dim=1)
        # num_agents = agentTrajs.size(dim=0)
        # batch_size = 1
        # Only valid when batch_size > 1
        target_length = agentTrajs.size(dim=2)
        num_agents = agentTrajs.size(dim=1)
        batch_size = agentTrajs.size(dim=0)

        # print('Future Scene AE')
        # print('target length: ', target_length) 
        # print('batch size: ', batch_size) 
        # print('num agents: ', num_agents)

        # print('agent trajectories shape: ', agentTrajs.shape)

        # Needed for batch training
        agentTrajs_flattened = agentTrajs.reshape(-1, target_length, 2) # (n_agents * batch_size, seq_len, 2)
        agentFutureTrajEnc_flattened = self.traj_encoder(agentTrajs_flattened) # (n_agents * batch_size, seq_len, hs)
        # print('agent future trajectory encoding shape: ', agentFutureTrajEnc_flattened.shape)
        agentFutureTrajEnc = agentFutureTrajEnc_flattened.reshape(batch_size, num_agents, -1) # (batch_size, n_agents, 1, hs)

        scene_enc = self.scene_encoder(agentPos, agentFutureTrajEnc, pos_emb, numAgents_emb, T) # (batch_size, enc_dim)
        graphDecoding = self.scene_decoder(agentPos, scene_enc, pos_emb, numAgents_emb, num_agents, T)

        agentFutureTrajDec = self.traj_decoder(graphDecoding, target_length=target_length, batch_size=batch_size*num_agents)
        # print('agent future trajectory decoding shape: ', agentFutureTrajDec.shape)
        # Needed for batch training
        agentFutureTrajDec = agentFutureTrajDec.reshape(batch_size, num_agents, target_length, 2)
        # print('agent future trajectory decoding shape: ', agentFutureTrajDec.shape)
        
        return agentFutureTrajDec, agentTrajs

        # When not using batches
        # agentFutureTrajEnc = self.traj_encoder(agentTrajs) 
        
        # scene_enc = self.scene_encoder(agentFutureTrajEnc, pos_emb, numAgents_emb) #(agentPos, agentFutureTrajEnc, pos_emb, numAgents_emb, T) # (batch_size, enc_dim)
        # graphDecoding = self.scene_decoder(scene_enc, pos_emb, numAgents_emb, num_agents) #(agentPos, scene_enc, pos_emb, numAgents_emb, num_agents, T)
        # # print('graph decoding shape: ', graphDecoding.shape)

        
        # return graphDecoding, agentTrajs
                                               
