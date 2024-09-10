import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data

# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)

class TrajRNN(nn.Module):
    """ 
    GRU based recurrent neural network. 
    Used for encoding past trajectories.
    """

    def __init__(self, nin, nout, es, hs, nl, device=torch.device("cpu")):
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
        self.cuda(self.device)
        
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden=None):
        # x = F.relu(self.embedding(x))
        x = F.tanh(self.embedding(x))
        x = self.dropout(x)
        # x = self.embedding(x)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)
        # x = F.relu(self.output_layer(x))
        x = F.tanh(self.output_layer(x))
        return x, hidden
    
    
class MPNNLayer(MessagePassing):
    """ 
    Message Passing Neural Network Layer.
    Implementation taken from GDL lecture practical 3 (https://colab.research.google.com/drive/1p9vlVAUcQZXQjulA7z_FyPrB9UXFATrR)
    """

    def __init__(self, emb_dim, edge_dim, aggr='add', device=torch.device("cpu")):
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
        # self.cuda(self.device)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        
        try:
            from torch_scatter import scatter
            self.torch_scatter = scatter
            self.use_torch_scatter = True
        except:
            self.use_torch_scatter = False
            
        
        self.dropout = nn.Dropout(0.2)

        # MLP `\psi` for computing messages `m_ij`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: (2d + d_e) -> d
        self.mlp_msg = nn.Sequential(
            nn.Linear(2*emb_dim + edge_dim, emb_dim),
            # nn.BatchNorm1d(emb_dim), # original code used BatchNorm, however this generates problems if there is only one agent in a scene
            # nn.ReLU(),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim),
            # nn.BatchNorm1d(emb_dim), 
            # nn.ReLU()
            nn.Tanh()
          )
        
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim), 
            # nn.BatchNorm1d(emb_dim), 
            # nn.ReLU(), 
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim, emb_dim), 
            # nn.BatchNorm1d(emb_dim), 
            # nn.ReLU()
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
        if self.use_torch_scatter:
            return self.torch_scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        else:
            max_index = index.max().item() + 1
    
            # Create a tensor of zeros with the appropriate size
            output = torch.zeros((max_index, inputs.shape[1]), dtype=inputs.dtype, device=self.device)
            
            # Perform scatter-add by accumulating the values for each index
            if self.aggr == 'add':
                output.scatter_add_(0, torch.tile(index.unsqueeze(1), (1, inputs.shape[1])), inputs)
            else:
                raise AttributeError("Not implemented for this aggregation method.")
            return output
            
    
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
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, device = 0):
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
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add', device = device))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool
        
        # move model to specified device
        self.device = device
        self.to(device)
        
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
        # unique_indexes = torch.unique(data.batch)
        # Create a boolean mask where each row corresponds to whether each element of unique_indexes equals the elements of data.batch
        # mask = data.batch.unsqueeze(0) == unique_indexes.unsqueeze(1)
        # Use argmax along axis 1 to find the indices of the first occurrences
        # first_occurrence_indices = mask.int().argmax(dim=1)


        # h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)
        # h_graph = h[first_occurrence_indices]

        return h
 

