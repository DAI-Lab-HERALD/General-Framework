import numpy, torch, dgl
import networkx as nx 
import matplotlib.pyplot as plt
from tqdm import tqdm

def build_dag_graph(graph, config):
    edge_type = torch.argmax(graph.edata["edge_probs"], dim=1)
    all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
    all_edges = torch.cat(all_edges, 1)

    # i --> j (i < j) edges in the graph
    src_edges_type_1 = all_edges[edge_type == 1][:, 0]
    dest_edges_type_1 = all_edges[edge_type == 1][:, 1]

    # j --> i (i < j) edges in the graph
    src_edges_type_2 = all_edges[edge_type == 2][:, 1]
    dest_edges_type_2 = all_edges[edge_type == 2][:, 0]

    dag_graph = dgl.graph((torch.cat([src_edges_type_1, src_edges_type_2], dim=0), torch.cat([dest_edges_type_1, dest_edges_type_2], dim=0)), num_nodes = graph.num_nodes())
    dag_edge_probs = torch.cat([graph.edata["edge_probs"][edge_type == 1][:, 1], graph.edata["edge_probs"][edge_type == 2][:, 2]], dim=0)
    dag_graph.edata["edge_probs"] = dag_edge_probs

    # Transfer features into "dagified" graph
    dag_graph.ndata["xt_enc"] = graph.ndata["xt_enc"] 
    dag_graph.ndata["ctrs"] = graph.ndata["ctrs"]
    dag_graph.ndata["rot"] = graph.ndata["rot"]
    dag_graph.ndata["orig"] = graph.ndata["orig"]
    dag_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()
    dag_graph.ndata["ground_truth_futures"] = graph.ndata["ground_truth_futures"].float()
    dag_graph.ndata["has_preds"] = graph.ndata["has_preds"].float()

    return dag_graph

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
        return removed
    filtered_edges = edges.clone()
    if weights is not None:
        filter_weights = weights.clone()
    else:
        filter_weights = torch.ones(edges.shape[1], dtype=torch.float, device=edges.device)
    filtered_mask = torch.ones(edges.shape[1], dtype=torch.bool, device=edges.device)
    filtered_id = torch.arange(edges.shape[1], device=edges.device)

    # Get connected nodes
    nodes = torch.unique(edges)
    print("        Start braking cycles in graph ({} edges and {} nodes)".format(filtered_edges.shape[1], len(nodes)), flush = True)

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

def prune_graph_johnson(dag_graph):
    """
    dag_graph: DGL graph with weighted edges
    graph contains edge property "edge_probs" which contains predicted probability of each edge type 

    Based on the predicted probabilities, prune graph until it is a DAG based on Johnson's algorithm

    Note that we can think of a batch of graphs as one big graph and apply the pruning procedure on the entire batch at once.
    """

    # G = dgl.to_networkx(dag_graph.cpu(), node_attrs=None, edge_attrs=None)
    # cycles = nx.simple_cycles(G)

    # # First identify cycles in graph
    # eids = []
    # for cycle in cycles:
    #     out_cycle = torch.Tensor(cycle).to(dag_graph.device).long()
    #     in_cycle = torch.roll(out_cycle, 1)

    #     eids.append(dag_graph.edge_ids(in_cycle, out_cycle))

    # to_remove = []
    # while len(eids) > 0:
    #     edge_probs_cycle = dag_graph.edata["edge_probs"][eids[0]]
    #     remove_eid = eids[0][torch.argmin(edge_probs_cycle)]
    #     to_remove.append(remove_eid)

    #     eids.pop(0)
    #     to_pop = []
    #     for j, eid_cycle in enumerate(eids):
    #         if remove_eid in eid_cycle:
    #             to_pop.append(j)
        
    #     eids = [v for i, v in enumerate(eids) if i not in to_pop]

    removed_mask = remove_terminal_nodes(edges = torch.stack(dag_graph.edges()), weights = dag_graph.edata["edge_probs"])
    to_remove = torch.where(removed_mask)[0]
    dag_graph.remove_edges(to_remove)

    return dag_graph

def build_dag_graph_test(graph, config):
    edge_type = torch.argmax(graph.edata["edge_probs"], dim=1)
    all_edges = [x.unsqueeze(1) for x in graph.edges('uv')]
    all_edges = torch.cat(all_edges, 1)

    # i --> j (i < j) edges in the graph
    src_edges_type_1 = all_edges[edge_type == 1][:, 0]
    dest_edges_type_1 = all_edges[edge_type == 1][:, 1]

    # j --> i (i < j) edges in the graph
    src_edges_type_2 = all_edges[edge_type == 2][:, 1]
    dest_edges_type_2 = all_edges[edge_type == 2][:, 0]

    dag_graph = dgl.graph((torch.cat([src_edges_type_1, src_edges_type_2], dim=0), torch.cat([dest_edges_type_1, dest_edges_type_2], dim=0)), num_nodes = graph.num_nodes())
    dag_edge_probs = torch.cat([graph.edata["edge_probs"][edge_type == 1][:, 1], graph.edata["edge_probs"][edge_type == 2][:, 2]], dim=0)
    dag_graph.edata["edge_probs"] = dag_edge_probs

    # Transfer features into "dagified" graph
    dag_graph.ndata["xt_enc"] = graph.ndata["xt_enc"] 
    dag_graph.ndata["ctrs"] = graph.ndata["ctrs"]
    dag_graph.ndata["rot"] = graph.ndata["rot"]
    dag_graph.ndata["orig"] = graph.ndata["orig"]
    dag_graph.ndata["agenttypes"] = graph.ndata["agenttypes"].float()

    return dag_graph
