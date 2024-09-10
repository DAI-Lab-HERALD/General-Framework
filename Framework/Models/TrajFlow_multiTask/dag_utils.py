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

    removed = torch.zeros(edges.shape[1], dtype=torch.bool, device=edges.device)
    # Check if there are cycles at all
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

    # Get connected nodes
    nodes = torch.unique(edges)

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
            # removed_terminal_nodes = True
            # There are only cycle left, start breaking them
            # Get scc components of graph
            print("        Start braking cycles in remaining graph ({} edges and {} nodes)".format(filtered_edges.shape[1], len(nodes)), flush = True)
            G = nx.DiGraph(list(zip(filtered_edges.tolist()[0], filtered_edges.tolist()[1])))
            SCC = nx.strongly_connected_components(G)
            remove_id = []
            for scc in SCC:
                # Ignore scc with only one node
                if len(scc) == 1:
                    continue
                # find edges inside the scc
                scc_nodes = torch.tensor(list(scc), device=edges.device)
                scc_edges_bool = (filtered_edges.unsqueeze(2) == scc_nodes.unsqueeze(0).unsqueeze(0)).any(-1).all(0)
                scc_edges_id = filtered_id[scc_edges_bool]
                scc_edges_weight = filter_weights[scc_edges_id]

                # Remove minimal weight edge
                min_edge_id = scc_edges_id[scc_edges_weight.argmin()]
                remove_id.append(min_edge_id)
            remove_id = torch.tensor(remove_id, device=edges.device)
            removed[remove_id] = True
            filtered_mask[remove_id] = False
            
            # Remove min_edge_id from filtered_id
            filtered_id = torch.where(filtered_mask)[0]
            filtered_edges = edges[:, filtered_id]

        else:
            # Adjust edges to only keep non-terminal nodes
            keep_edge = (filtered_edges.unsqueeze(2) == non_terminal_nodes.unsqueeze(0).unsqueeze(0)).any(-1).all(0)
            filtered_edges = filtered_edges[:, keep_edge]

            # Update filtered mask and id
            filtered_mask[filtered_mask.clone()] = keep_edge
            filtered_id = filtered_id[keep_edge]

            # redefine non terminal nodes to be the remaining nodes with some edgess
            nodes = torch.unique(filtered_edges)

            if not keep_edge.any():
                removed_terminal_nodes = True

    return removed


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

    # Get edge list
    edge_list      = list(zip(graph_edge_index.tolist()[0], graph_edge_index.tolist()[1]))
    edge_list_self = list(zip(graph_edge_index_self.tolist()[0], graph_edge_index_self.tolist()[1]))
    G = nx.DiGraph(edge_list)
    G.remove_edges_from(edge_list_self)

    # Prepare removal mask
    mask = torch.ones(graph_edge_index.shape[1], dtype=torch.bool, device=device)

    # Remove terminal edges from the graph
    if not nx.is_directed_acyclic_graph(G):
        removed_mask = remove_terminal_nodes(graph_edge_index_non_self, edge_probs_list[non_self_index])
        removed = non_self_index[torch.where(removed_mask)[0]]

        # Remove cycle edges
        edge_list_remove = [edge_list[i] for i in removed]
        G.remove_edges_from(edge_list_remove)

        # Adjust mask
        mask[non_self_index] = ~removed_mask

        assert nx.is_directed_acyclic_graph(G), "Graph should have been made acyclic"

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