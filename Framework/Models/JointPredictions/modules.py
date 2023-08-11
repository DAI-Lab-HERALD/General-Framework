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
        self.hs = hs
        self.nl = nl
        
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
            nn.Conv2d(nin, 8, kernel_size=ks[0], stride=strides[0], padding=paddings[0]),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=ks[1], stride=strides[1], padding=paddings[1]),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=ks[2], stride=strides[2], padding=paddings[2]),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        h_out_l1 = int((img_dim[0] + 2*paddings[0]-1*(ks[0]-1)-1)/strides[0] + 1)
        w_out_l1 = int((img_dim[1] + 2*paddings[0]-1*(ks[0]-1)-1)/strides[0] + 1)
        h_out_l2 = int((h_out_l1 + 2*paddings[1]-1*(ks[1]-1)-1)/strides[1] + 1)
        w_out_l2 = int((w_out_l1 + 2*paddings[1]-1*(ks[1]-1)-1)/strides[1] + 1)
        h_out_l3 = int((h_out_l2 + 2*paddings[2]-1*(ks[2]-1)-1)/strides[2] + 1)
        w_out_l3 = int((w_out_l2 + 2*paddings[2]-1*(ks[2]-1)-1)/strides[2] + 1)

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
        
        self.num_layers = num_layers
        self.emb_dim = emb_dim
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

    def __init__(self, socialGNNparams, envCNNparams, T_all=None, device=0):
        super().__init__()
        self.device = device
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.t_unique = self.t_unique[self.t_unique != 48]

        self.in_dim = socialGNNparams['in_dim']
        # Module for encoding social interactions
        self.encoder = SocialInterGNN(num_layers = socialGNNparams['num_layers'],
                                      emb_dim = socialGNNparams['emb_dim'],
                                      in_dim = socialGNNparams['in_dim'],
                                      edge_dim = socialGNNparams['edge_dim'])
        
        
        self.static_env_encoder = StaticEnvCNN(img_dim = envCNNparams['img_dim'],
                                               nin = envCNNparams['nin'], 
                                               enc_dim = envCNNparams['enc_dim'])

    def forward(self, pastTrajEnc, pos, T, staticEnv=None):
        """
        Args: 
            pos: (batch_size, max_num_agents, 2) - current position of all agents
            pastTrajEnc: (batch_size, max_num_agents, seq_len, past_nout) - encoded past trajectories of all agents
            T: (batch_size, max_num_agents) - class labels for all agents
        """
        if staticEnv is not None:
            staticEnv_flattened = staticEnv.view(-1, *staticEnv.shape[2:])
            sceneStaticEnc = self.static_env_encoder(staticEnv_flattened)
            sceneStaticEnc = sceneStaticEnc.view(*staticEnv.shape[:2], -1)
        
        pastTrajEnc     = pastTrajEnc[...,-1,:]
        max_num_agents  = pastTrajEnc.size(dim=1)
        
        # Find existing agents (T = 48 means that the agent does not exist)
        existing_agent = T != 48 # (batch_size, max_num_agents)

        # Find connection matrix for existing agents
        Edge_bool = existing_agent[:, None] & existing_agent[:, :, None] # (batch_size, max_num_agents, max_num_agents)

        exist_sample2, exist_row2 = torch.where(existing_agent)
        exist_sample3, exist_row3, exist_col3 = torch.where(Edge_bool)

        # Differentiate between agents of different samples
        node_adder = exist_sample3 * max_num_agents
        graph_edge_index = torch.stack([exist_row3, exist_col3]) + node_adder[None, :] # shape: 2 x num_edges

        # Eliminate the holes due to missing agents
        # i.e node ids go from (0,3,4,6,...) to (0,1,2,3,...)
        graph_edge_index = torch.unique(graph_edge_index, return_inverse = True)[1] 

        # Get type of existing agents
        T_exisiting_agents = T[exist_sample2, exist_row2] # shape: num_nodes
        # Get one hot encodeing of this agent type
        T_one_hot = (T_exisiting_agents.unsqueeze(-1) == self.t_unique.unsqueeze(0)).float() # shape: num_nodes x num_classes
        # Get one hot encoding from start and goal node of each edge
        T_one_hot_edge = T_one_hot[graph_edge_index, :] # shape: 2 x num_edges x num_classes
        # Concatenate in and output type for each edge
        class_in_out = T_one_hot_edge.permute(1,0,2).reshape(-1, 2 * len(self.t_unique)) # shape: num_edges x (2 num_classes)

        
        D = torch.sqrt(torch.sum((pos[:,None,:] - pos[:,:,None]) ** 2, dim = -1))
        dist_in_out = D[exist_sample3, exist_row3, exist_col3]
        dist_in_out = dist_in_out.reshape(-1,1)
         
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        
        pastTrajEnc_existing_agents = pastTrajEnc[exist_sample2, exist_row2]
        
        if staticEnv is not None:
            sceneStaticEncEnc_existing_agents = sceneStaticEnc[exist_sample2, exist_row2]
            sceneContext = torch.cat((sceneStaticEncEnc_existing_agents, pastTrajEnc_existing_agents), dim = 1)
        else:
            sceneContext = pastTrajEnc_existing_agents

        socialData = Data(x=sceneContext.reshape(-1, self.in_dim), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=exist_sample2)
        
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
        
    
class FutureSceneEncoder(nn.Module):
    """Module for encoding past scene information."""

    def __init__(self, socialGNNparams, enc_dim, T_all=None, device=0):
        super().__init__()
        self.device = device
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.t_unique = self.t_unique[self.t_unique != 48]
        self.in_dim = socialGNNparams['in_dim']
       
        
        # Module for encoding social interactions
        self.encoder = SocialInterGNN(num_layers=socialGNNparams['num_layers'],
                                      emb_dim=socialGNNparams['emb_dim'],
                                      in_dim=socialGNNparams['in_dim'],
                                      edge_dim=socialGNNparams['edge_dim'])
        
        
        ### Fully connected layer
        self.encoder_fc = nn.Sequential(
            nn.Linear(socialGNNparams['emb_dim'], int(socialGNNparams['emb_dim']/2)),
            nn.Tanh(),
            nn.Linear(int(socialGNNparams['emb_dim']/2), enc_dim)
        )

    def forward(self, pos, x_enc, T):

        """
        Args: 
            x: (batch_size, n_agents, 1, 2) - past trajectories of all agents
            x_enc: (batch_size, n_agents, fut_nout) - encoded future trajectories of all agents
            pos_emb: (batch_size, n_agents, pos_emb_dim) - positional encoding of all agents
            numAgents_emb: (batch_size, n_agents, 1) - encoding of number of agents in the scene 
            T: (batch_size, n_agents) - class labels for all agents
        """    
        max_num_agents  = x_enc.size(dim=1)
        
        # Find existing agents (T = 48 means that the agent does not exist)
        existing_agent = T != 48 # (batch_size, max_num_agents)

        # Find connection matrix for existing agents
        Edge_bool = existing_agent[:, None] & existing_agent[:, :, None] # (batch_size, max_num_agents, max_num_agents)

        exist_sample2, exist_row2 = torch.where(existing_agent)
        exist_sample3, exist_row3, exist_col3 = torch.where(Edge_bool)

        # Differentiate between agents of different samples
        node_adder = exist_sample3 * max_num_agents
        graph_edge_index = torch.stack([exist_row3, exist_col3]) + node_adder[None, :] # shape: 2 x num_edges

        # Eliminate the holes due to missing agents
        # i.e node ids go from (0,3,4,6,...) to (0,1,2,3,...)
        graph_edge_index = torch.unique(graph_edge_index, return_inverse = True)[1] 

        # Get type of existing agents
        T_exisiting_agents = T[exist_sample2, exist_row2] # shape: num_nodes
        # Get one hot encodeing of this agent type
        T_one_hot = (T_exisiting_agents.unsqueeze(-1) == self.t_unique.unsqueeze(0)).float() # shape: num_nodes x num_classes
        # Get one hot encoding from start and goal node of each edge
        T_one_hot_edge = T_one_hot[graph_edge_index, :] # shape: 2 x num_edges x num_classes
        # Concatenate in and output type for each edge
        class_in_out = T_one_hot_edge.permute(1,0,2).reshape(-1, 2 * len(self.t_unique)) # shape: num_edges x (2 num_classes)

        
        D = torch.sqrt(torch.sum((pos[:,None,:] - pos[:,:,None]) ** 2, dim = -1))
        dist_in_out = D[exist_sample3, exist_row3, exist_col3]
        dist_in_out = dist_in_out.reshape(-1,1)
         
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)
        
        x_enc_existing_agents = x_enc[exist_sample2, exist_row2]
        

        socialData = Data(x=x_enc_existing_agents, 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=exist_sample2)
        
        graphEncoding = self.encoder(socialData) # (batch_size, emb_dim)
        sceneEncoding = self.encoder_fc(graphEncoding)

        return sceneEncoding
    
    
class FutureSceneDecoder(nn.Module):
    """Module for encoding past scene information."""

    def __init__(self, socialGNNparams, enc_dim, pos_emb_dim, T_all=None, device=0):
        super().__init__()
        self.device = device
        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.t_unique = self.t_unique[self.t_unique != 48]
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
    def forward(self, pos, enc, pos_emb, numAgents_emb, max_num_agents, T): 

        """
        Args: 
            x: (batch_size, n_agents, seq_len, 2) - current position of all agents
            enc: (batch_size, n_agents, seq_len, fut_nout) - encoded future scene
            pos_emb: (batch_size, n_agents, pos_emb_dim) - positional encoding of all agents
            numAgents_emb: (batch_size, n_agents, 1) - encoding of number of agents in the scene 
            T: (batch_size, n_agents) - class labels for all agents
        """
        
        enc_emb = self.decoder_fc(enc)
        # repeat the graph scene encodings for the maximum number of possible agents
        enc_emb = enc_emb.unsqueeze(1).repeat((1,max_num_agents,1))

        # Find existing agents (T = 48 means that the agent does not exist)
        existing_agent = T != 48 # (batch_size, max_num_agents)

        numAgents_emb = numAgents_emb.unsqueeze(1).repeat((1,max_num_agents,1))
        # num_nodes = num_agents

        # Find connection matrix for existing agents
        Edge_bool = existing_agent[:, None] & existing_agent[:, :, None] # (batch_size, max_num_agents, max_num_agents)

        exist_sample2, exist_row2 = torch.where(existing_agent)
        exist_sample3, exist_row3, exist_col3 = torch.where(Edge_bool)

        # Differentiate between agents of different samples
        node_adder = exist_sample3 * max_num_agents
        graph_edge_index = torch.stack([exist_row3, exist_col3]) + node_adder[None, :] # shape: 2 x num_edges

        # Eliminate the holes due to missing agents
        # i.e node ids go from (0,3,4,6,...) to (0,1,2,3,...)
        graph_edge_index = torch.unique(graph_edge_index, return_inverse = True)[1] 

        # Get type of existing agents
        T_exisiting_agents = T[exist_sample2, exist_row2] # shape: num_nodes
        # Get one hot encodeing of this agent type
        T_one_hot = (T_exisiting_agents.unsqueeze(-1) == self.t_unique.unsqueeze(0)).float() # shape: num_nodes x num_classes
        # Get one hot encoding from start and goal node of each edge
        T_one_hot_edge = T_one_hot[graph_edge_index, :] # shape: 2 x num_edges x num_classes
        # Concatenate in and output type for each edge
        class_in_out = T_one_hot_edge.permute(1,0,2).reshape(-1, 2 * len(self.t_unique)) # shape: num_edges x (2 num_classes)


        D = torch.sqrt(torch.sum((pos[:,None,:] - pos[:,:,None]) ** 2, dim = -1))
        dist_in_out = D[exist_sample3, exist_row3, exist_col3]
        dist_in_out = dist_in_out.reshape(-1,1)
         
         # TODO possibly put back
        graph_edge_attr = torch.cat((class_in_out, dist_in_out), dim = 1)

        pos_emb_existing_agents = pos_emb[exist_sample2, exist_row2]
        numAgents_emb_existing_agents = numAgents_emb[exist_sample2, exist_row2]
        enc_emb_existing_agents = enc_emb[exist_sample2, exist_row2]


        graph_x = torch.cat((pos_emb_existing_agents, numAgents_emb_existing_agents, enc_emb_existing_agents), dim=1)
        decSocialData = Data(x=graph_x.reshape(-1, self.in_dim), 
                          edge_index=graph_edge_index, 
                          edge_attr=graph_edge_attr, 
                          batch=exist_sample2)

        
        graphDecoding = self.decoder(decSocialData) # (num_existing_agents, emb_dim)       


        return graphDecoding, existing_agent

class FutureSceneAE(nn.Module):

    def __init__(self, futureRNNencParams, futureRNNdecParams, socialGNNparams, enc_dim, pos_emb_dim, T_all, device=0):
        super().__init__()
        self.device = device

        self.scene_encoder = FutureSceneEncoder(socialGNNparams, enc_dim, T_all=T_all, device=device).to(device)
        self.scene_decoder = FutureSceneDecoder(socialGNNparams, enc_dim, pos_emb_dim, T_all=T_all, device=device).to(device)

        self.traj_encoder = nn.ModuleDict({})
        self.traj_decoder = nn.ModuleDict({})

        self.t_unique = torch.unique(torch.from_numpy(T_all).to(device))
        self.t_unique = self.t_unique[self.t_unique != 48]
        for t in self.t_unique:
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            self.traj_encoder[t_key] = FutureTrajEncoder(futureRNNencParams, device=device).to(device)
            self.traj_decoder[t_key] = FutureTrajDecoder(futureRNNdecParams, device=device).to(device)

            self.traj_encoder[t_key].to(device)
            self.traj_decoder[t_key].to(device)
            
         
        self.traj_encoder = nn.ModuleDict(self.traj_encoder)
        self.traj_decoder = nn.ModuleDict(self.traj_decoder)   
                                               
        self.pos_emb = nn.Linear(2, pos_emb_dim).to(device)
        self.numAgents_emb = nn.Linear(1, 1).to(device)

    def forward(self, agentTrajs, agentPos, T):
        target_length = agentTrajs.size(dim=2)
        num_agents = agentTrajs.size(dim=1)
        batch_size = agentTrajs.size(dim=0)

        existing_agent = T != 48 # (batch_size, max_num_agents)
        exist_sample2, exist_row2 = torch.where(existing_agent)

        agentPos_existing = agentPos[exist_sample2, exist_row2] # (num_existing_agents, 2)

        pos_emb = F.tanh(self.pos_emb(agentPos_existing)) # (n_agents, enc_dim)

        tmp = torch.zeros((batch_size, num_agents, 2), device = self.device)
        tmp[tmp == 0] = float('nan')
        tmp[exist_sample2, exist_row2] = pos_emb

        pos_emb = tmp

        numAgents_emb = F.tanh(self.numAgents_emb(torch.tensor(num_agents).float().to(self.device).unsqueeze(0))) # (1, 1)
        numAgents_emb = numAgents_emb.repeat(batch_size, 1) # (n_agents, 1)



        # Needed for batch training
        agentTrajs_flattened = agentTrajs.reshape(-1, target_length, 2) # (n_agents * batch_size, seq_len, 2)

        agentFutureTrajEnc_flattened = torch.zeros((agentTrajs_flattened.shape[0], 
                                                    self.traj_encoder[str(int(self.t_unique[0].detach().cpu().numpy().astype(int)))].rnn.hs),
                                                    device = self.device)
        
        T_flattened = T.reshape(-1)
        for t in self.t_unique:
            assert t in T_flattened
            t_in = T_flattened == t
            
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            agentFutureTrajEnc_flattened[t_in] = self.traj_encoder[t_key](agentTrajs_flattened[t_in])

        agentFutureTrajEnc = agentFutureTrajEnc_flattened.reshape(batch_size, num_agents, -1) # (batch_size, n_agents, 1, hs)

        scene_enc = self.scene_encoder(agentPos, agentFutureTrajEnc, T) # (batch_size, enc_dim)
        graphDecoding, existing_agents = self.scene_decoder(agentPos, scene_enc, pos_emb, numAgents_emb, num_agents, T)

        agentFutureTrajDec = torch.zeros((graphDecoding.shape[0], agentTrajs_flattened.shape[1], 2),
                                                    device = self.device)
        
        T_flattened = T_flattened[T_flattened != 48]
        for t in self.t_unique:
            assert t in T_flattened
            t_in = T_flattened == t
            
            t_key = str(int(t.detach().cpu().numpy().astype(int)))
            agentFutureTrajDec[t_in] = self.traj_decoder[t_key](graphDecoding[t_in], target_length=target_length, batch_size=len(graphDecoding[t_in]))

        # Needed for batch training
        tmp = torch.zeros((batch_size, num_agents, target_length, 2), device = self.device)
        tmp[tmp == 0] = float('nan')
        existing_sample, existing_row = torch.where(existing_agents)

        tmp[existing_sample, existing_row] = agentFutureTrajDec

        agentFutureTrajDec = tmp
        
        return agentFutureTrajDec, agentTrajs

                                               
