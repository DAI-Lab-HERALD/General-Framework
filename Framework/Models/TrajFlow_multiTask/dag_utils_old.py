import networkx as nx
import numpy as np
import torch

from collections import deque


def remove_terminal_nodes(edges, weights = None):
    """
    Remove terminal nodes from the graph.
    
    Args:
        edges (2 x E tensor): Edge list representation of the graph.
        weights (E tensor): Weights corresponding to each edge.
        
    Returns:
        filtered_edges (2 x E' tensor): The edge list without terminal nodes.
    """
    if edges.shape[1] == 0:
        filtered_mask = torch.tensor([], dtype=torch.bool, device=edges.device)
        removed = torch.tensor([], dtype=torch.bool, device=edges.device)
        return edges, filtered_mask, removed
    filtered_edges = edges.clone()
    if weights is not None:
        filter_weights = weights.clone()
    else:
        filter_weights = torch.ones(edges.shape[1], dtype=torch.float, device=edges.device)
    filtered_mask = torch.ones(edges.shape[1], dtype=torch.bool, device=edges.device)
    filtered_id = torch.arange(edges.shape[1], device=edges.device)
    removed = torch.zeros(edges.shape[1], dtype=torch.bool, device=edges.device)

    # Get connected nodes
    nodes = torch.unique(edges)

    # Initialize the graph
    G = nx.DiGraph(list(zip(edges[0].tolist(), edges[1].tolist())))

    # find terminal nodes
    removed_terminal_nodes = False
    while not removed_terminal_nodes:
        # Find nodes which do not appear at least ones in both rows of edges
        unique_in = torch.unique(filtered_edges[0])
        unique_out = torch.unique(filtered_edges[1])
        # Get intersection of the two sets
        idx = torch.searchsorted(unique_in[:-1], unique_out)
        non_terminal_nodes = unique_out[unique_in[idx] == unique_out]
        assert len(non_terminal_nodes) == (len(unique_in) + len(unique_out) - len(nodes)), "All nodes should be covered"

        if len(non_terminal_nodes) == len(nodes):
            removed_terminal_nodes = True

        else:
            # Adjust edges to only keep non-terminal nodes
            keep_edge = (filtered_edges.unsqueeze(2) == non_terminal_nodes.unsqueeze(0).unsqueeze(0)).any(-1).all(0)
            removed_edges = filtered_edges[:, ~keep_edge]
            filtered_edges = filtered_edges[:, keep_edge]

            # Update filtered mask and id
            filtered_mask[filtered_mask.clone()] = keep_edge
            filtered_id = filtered_id[keep_edge]

            # redefine non terminal nodes to be the remaining nodes with some edges
            non_terminal_nodes = torch.unique(filtered_edges)

            # Get terminal nodes
            terminal_nodes = nodes[~torch.isin(nodes, non_terminal_nodes)]

            # Remove edges and nodes from the graph
            G.remove_edges_from(list(zip(removed_edges[0].tolist(), removed_edges[1].tolist())))
            G.remove_nodes_from(terminal_nodes.tolist())

            # Overwrite nodes
            nodes = non_terminal_nodes

            if not keep_edge.any():
                removed_terminal_nodes = True

    return filtered_edges, filtered_mask, removed


def get_removable_edge_id(graph_edge_index, edge_probs, subgraphs, device='cuda'):
    # First identify cycles in graph
    cycle_edges = []
    cycle_id = []
    cycle_lengths = []
    print("        Find cycles", flush = True)
    for subgraph in subgraphs.unique():
        use_edges = subgraphs == subgraph
        subgraph_edge_index = graph_edge_index[:,use_edges]

        # get subgraph
        G_sub = nx.DiGraph(list(zip(subgraph_edge_index[0].tolist(), subgraph_edge_index[1].tolist()))) 
        cycles = nx.simple_cycles(G_sub) # This is a generator function, it is not evaluated yet

        # Get estimated number of cycles

        # Go throug cycles
        for i, cycle in enumerate(cycles):
            cycle_lengths.append(len(cycle))
            if i == 999:
                E = subgraph_edge_index.shape[1]
                N = len(list(G_sub.nodes()))
                print("        Subgraph has {} nodes and {} edges, after removing terminal nodes".format(N, E))
            if np.mod(i + 1, 1000) == 0:
                print("\r        Resolving cycle {}".format(i+1), end=' ', flush=True)
            out_cycle = torch.Tensor(cycle).long()
            in_cycle = torch.roll(out_cycle, 1)
            cycle_edges.append(torch.stack([in_cycle, out_cycle], dim=0).to(device))
            cycle_id.append(torch.full((cycle_lengths[-1],), i, dtype=torch.long, device=device))
        
        if i > 1000:
            print("\n")
            break
    

    if len(cycle_edges) > 0:
        print("        Remove cycles", flush = True)
        # Assume thata graph edge index is unique
        try:
            Cycle_edges = torch.cat(cycle_edges, dim=1) # 2 x E
            test, Eids_T = torch.where((graph_edge_index.unsqueeze(1) == Cycle_edges.unsqueeze(2)).all(0))
            assert (test.detach().cpu() == torch.arange(Cycle_edges.shape[1])).all(), "Cycle edge indexes are not unique"
            del Cycle_edges
        except:
            # clear memory
            torch.cuda.empty_cache()
    
            # Go iteratively through the edges
            Eids_T = []
            for cycle_edge in cycle_edges:
                test, eids_T = torch.where((graph_edge_index.unsqueeze(1) == cycle_edge.unsqueeze(2)).all(0))
                assert (test.detach().cpu() == torch.arange(cycle_edge.shape[1])).all(), "Cycle edge indexes are not unique"
                Eids_T.append(eids_T)
            
            Eids_T = torch.cat(Eids_T, dim=0)
        cycle_id = torch.cat(cycle_id, dim=0) # E
    
    
        # Given the uindirectional graph nature, eids will con
        to_remove = []
        ignore_i = torch.zeros(len(cycle_lengths), dtype=torch.bool, device=device)
        i_start = 0

        for i, length in enumerate(cycle_lengths):
            i_end = i_start + length
            eids_loc = Eids_T[i_start:i_end]
            i_start = i_end
            if ignore_i[i]:
                continue
    
            # Get edge to remove
            edge_probs_cycle = edge_probs[eids_loc]
            remove_eid = eids_loc[edge_probs_cycle.argmin()]
            to_remove.append(remove_eid)
    
            # Ignore all cycle that contain this edge
            checks = Eids_T == remove_eid
            ignore_new = cycle_id[checks].unique()
            ignore_i[ignore_new] = True
        
        to_remove = torch.tensor(to_remove, dtype=torch.long, device=device)
        num_cycles = len(cycle_lengths)
    else:
        print("        No cycles found", flush = True)
        to_remove = torch.tensor([], dtype=torch.long, device=device)
        num_cycles = 0
    
    return to_remove, num_cycles



def remove_cycles(G_in, graph_edge_index, edge_probs, subgraphs, max_num_agents, device='cuda'):
    G = G_in.copy()
    mask = torch.ones(graph_edge_index.shape[1], dtype=torch.bool, device=device)
    to_remove, num_cycles = get_removable_edge_id(graph_edge_index, edge_probs, subgraphs, device=device)
    
    if len(to_remove) > 0: 
        edge_list = list(zip(graph_edge_index.tolist()[0], graph_edge_index.tolist()[1]))
        edge_list_remove = [edge_list[i] for i in to_remove]
        G.remove_edges_from(edge_list_remove)

        mask[to_remove] = False

        if not nx.is_directed_acyclic_graph(G):
            remaining_cycles = list(nx.simple_cycles(G))
            print("        Initially found {} cycles, but {} are not removed:".format(num_cycles, len(remaining_cycles)))
            for cycle in remaining_cycles[:15]:
                cycle_array = np.array(cycle)

                # get corresponding batch id of the cycles
                batch_id = cycle_array // max_num_agents

                # Print unique batch ids
                print("            ", cycle, " - nodes come from batches ", batch_id)

            # Run the cycle removal on the remaining edges for combined batches
            to_remove_combined, _ = get_removable_edge_id(graph_edge_index[:,mask], edge_probs[mask], 
                                                          torch.zeros(mask.sum(), dtype = torch.long, device = device), device=device)
            # Transfert to original coordinates
            to_remove_combined = torch.where(mask)[0][to_remove_combined]

            mask[to_remove_combined] = False
            edge_list_remove = [edge_list[i] for i in to_remove_combined]
            G.remove_edges_from(edge_list_remove)
    
    assert nx.is_directed_acyclic_graph(G), "Graph should be acyclic"
    
    return G, mask 


def dagification(graph_edge_index, edge_probs, node_adder, device='cuda'):
    # Currently, all interactions are one directional (i.e., (edge_bool + edge_bool.T).max() == 1 outside main diagonal)

    # Check edge probs
    edge_bool = edge_probs!=0
    edge_probs_list = edge_probs[edge_bool]
    assert (edge_probs > 0).sum() == graph_edge_index.shape[1], "All edges are known"

    self_loops = graph_edge_index[0] == graph_edge_index[1]
    self_index = torch.where(self_loops)[0]
    non_self_index = torch.where(~self_loops)[0]

    # Divide by self and non self edges
    graph_edge_index_self     = graph_edge_index[:,self_index]
    graph_edge_index_non_self = graph_edge_index[:,non_self_index]

    # Break cycles of length 2
    # Find edges connected in both ways, which should not exist
    # remove main diagonal fron edge_bool
    edge_bool[:, torch.arange(edge_bool.shape[1]), torch.arange(edge_bool.shape[1])] = False
    assert (edge_bool & edge_bool.transpose(1,2)).sum() == 0, "Bidirectional edges are not allowed"

    # Remove terminal edges from the graph
    _, used_edges, _ = remove_terminal_nodes(graph_edge_index_non_self, edge_probs_list[non_self_index])
    non_self_index_cycle = non_self_index[used_edges]

    # Get probabilities for non self edges
    graph_edge_index_non_self_cycle = graph_edge_index[:,non_self_index_cycle]

    # Get edge list
    edge_list      = list(zip(graph_edge_index.tolist()[0], graph_edge_index.tolist()[1]))
    edge_list_self = list(zip(graph_edge_index_self.tolist()[0], graph_edge_index_self.tolist()[1]))
    G = nx.DiGraph(edge_list)
    G.remove_edges_from(edge_list_self)

    if nx.is_directed_acyclic_graph(G):
        G_with_self_loops = G.copy()
        G_with_self_loops.add_edges_from(edge_list_self)
        return graph_edge_index, node_adder, G, G_with_self_loops

    # Remove cycles
    G, mask_non_self_cycle = remove_cycles(G, graph_edge_index_non_self_cycle, 
                                           edge_probs_list[non_self_index_cycle],
                                           node_adder[non_self_index_cycle], 
                                           edge_bool.shape[1], device=device)
    
    mask = torch.ones(graph_edge_index.shape[1], dtype=torch.bool, device=device)
    mask[non_self_index_cycle] = mask_non_self_cycle
    
    # Add self loops
    G_with_self_loops = G.copy()
    G_with_self_loops.add_edges_from(edge_list_self)
    
    # Remove non self edges from graph index
    new_graph_edge_index = graph_edge_index[:,mask]
    new_node_addr = node_adder[mask]               
 
    return new_graph_edge_index, new_node_addr, G, G_with_self_loops


def kahn_toposort(G, num_nodes):
    """Given a directed graph, returns a list of nodes in topological order.
    >>> from networkx import DiGraph
    >>> get_topological_sorting(DiGraph({1: [], 2: [1], 3: [2]}))
    [3, 2, 1]
    >>> get_topological_sorting(DiGraph({1: [3], 2: [1], 3: [2]}))

    Parameters
    ----------
    digraph : DiGraph, a graph container instance

    Returns
    -------
    sorting : list of integers corresponding to node indices in the graph
        None, if there is no topological sorting (i.e., the graph is
        cyclic.

    Source: https://github.com/dani2819/Efficient-implementation-of-topological-sorting-of-graph
    Modified: anna-meszaros 09-04-2024
    """
    L_by_deg = {}
    deg = 0
    L = []
    S = []
    visited = set()
    L_pred_by_deg = {}
    L_pred = []

    graph = G.copy()

    assert num_nodes == len(graph.nodes()), "Number of nodes does not match"
    for node in graph.nodes(): 
        if node not in visited:
            l = list(graph.predecessors(node))
            if not l:
                S.append(node)
                
            visited.add(node)

    last_node_of_deg = S[-1]

    while (len(S)!=0):
        n = S.pop(0)
        L.append(n)
        if deg > 0:
            L_pred.append(list(G.predecessors(n)))
        edgeNodes = list(graph.successors(n))
        for m in edgeNodes:
            graph.remove_edge(n,m)
            l = list(graph.predecessors(m))
            if not l:
                S.append(m)

        if n == last_node_of_deg:
            L_by_deg[deg] = L
            if deg > 0:
                L_pred_by_deg[deg] = L_pred
            deg += 1
            L = []
            L_pred = []
            if S:
                last_node_of_deg = S[-1]

    l = list(graph.edges())
    if not l:
        return L_by_deg, L_pred_by_deg
    else:
        return None