import av2
import copy
import numpy as np
import os
import torch

from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet, _convert_tracks_to_tabular_format
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.interpolate import compute_midpoint_line

from pathlib import Path
from scipy import sparse

cross_dist = 6 
cross_angle = 0.5 * np.pi

# FROM FJMP
def read_argoverse2_data(file_path):
    # TODO: change first three lines to read directly from downloaded data
    # scene_directory = self.mapping[idx]
    parquet_file = os.path.join(file_path, "scenario_{}.parquet".format(os.path.basename(file_path)))
    scenario = load_argoverse_scenario_parquet(parquet_file)
    
    """observed, track_id, object_type, object_category, timestep, position_x, position_y, heading, velocity_x, velocity_y"""
    df = _convert_tracks_to_tabular_format(scenario.tracks)
    
    agt_ts = np.sort(np.unique(df['timestep'].values))
    timestamp_mapping = dict()
    for i, ts in enumerate(agt_ts):
        timestamp_mapping[ts] = i 

    trajs = np.concatenate((
        df.position_x.to_numpy().reshape(-1, 1),
        df.position_y.to_numpy().reshape(-1, 1)
    ), 1)

    vels = np.concatenate((
        df.velocity_x.to_numpy().reshape(-1, 1),
        df.velocity_y.to_numpy().reshape(-1, 1)
    ), 1)

    psirads = df.heading.to_numpy().reshape(-1, 1)

    track_ids = df.track_id.to_numpy().reshape(-1, 1)

    agentcategories = df.object_category.to_numpy().reshape(-1, 1)

    ### NOTE: We will only predict trajectories from classes 0-4
    object_type_dict = {
        'vehicle': 0,
        'pedestrian': 1,
        'motorcyclist': 2,
        'cyclist': 3,
        'bus': 4,
        'static': 5,
        'background': 6,
        'construction': 7,
        'riderless_bicycle': 8,
        'unknown': 9
    }

    agenttypes = []
    for x in df.object_type:
        agenttypes.append(object_type_dict[x])
    agenttypes = np.array(agenttypes).reshape(-1, 1)

    ### NOTE: no shape information in Argoverse 2.

    steps = [timestamp_mapping[x] for x in df['timestep'].values]
    steps = np.asarray(steps, np.int64)

    objs = df.groupby(['track_id']).groups 
    keys = list(objs.keys())
    ctx_trajs, ctx_steps, ctx_vels, ctx_psirads, ctx_agenttypes, ctx_agentcategories, ctx_track_ids = [], [], [], [], [], [], []
    for key in keys:
        idcs = objs[key]
        ctx_trajs.append(trajs[idcs])
        ctx_steps.append(steps[idcs])
        ctx_vels.append(vels[idcs])
        ctx_psirads.append(psirads[idcs])
        ctx_agenttypes.append(agenttypes[idcs])  
        ctx_agentcategories.append(agentcategories[idcs])
        ctx_track_ids.append(track_ids[idcs])

    data = dict()
    data['trajs'] = ctx_trajs
    data['steps'] = ctx_steps 
    data['vels'] = ctx_vels
    data['psirads'] = ctx_psirads
    data['agenttypes'] = ctx_agenttypes
    data['agentcategories'] = ctx_agentcategories
    data['track_ids'] = ctx_track_ids
    data['city_name'] = scenario.city_name
    data['focal_id'] = scenario.focal_track_id

    return data



# FROM FJMP
def get_lane_graph(file_path):
    # TODO: change first three lines to read directly from downloaded data
    # scene_directory = self.mapping[idx]
    static_map_path = os.path.join(file_path, "log_map_archive_{}.json".format(os.path.basename(file_path)))
    static_map = ArgoverseStaticMap.from_json(Path(static_map_path))

    lane_ids, ctrs, feats = [], [], []
    centerlines, left_boundaries, right_boundaries = [], [], []
    lane_type = []
    for lane_segment in static_map.vector_lane_segments.values():
        left_boundary = copy.deepcopy(lane_segment.left_lane_boundary.xyz[:, :2])
        right_boundary = copy.deepcopy(lane_segment.right_lane_boundary.xyz[:, :2])
        centerline, _ = compute_midpoint_line(left_boundary, right_boundary, min(10, max(left_boundary.shape[0], right_boundary.shape[0])))
        centerline = copy.deepcopy(centerline)       

        # Get the lane marker types
        lane_type.append((lane_segment.lane_type.value, lane_segment.is_intersection))
        
        # process lane centerline in same way as agent trajectories
        # centerline = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
        # left_boundary = np.matmul(data['rot'], (left_boundary - data['orig'].reshape(-1, 2)).T).T
        # right_boundary = np.matmul(data['rot'], (right_boundary - data['orig'].reshape(-1, 2)).T).T
    
        num_segs = len(centerline) - 1
        # locations between the centerline segments
        ctrs.append(np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32))
        # centerline segment offsets
        feats.append(np.asarray(centerline[1:] - centerline[:-1], np.float32))
        lane_ids.append(lane_segment.id)
        centerlines.append(centerline)
        left_boundaries.append(left_boundary)
        right_boundaries.append(right_boundary)

    # node indices (when nodes are concatenated into one array)
    node_idcs = []
    count = 0
    for i, ctr in enumerate(ctrs):
        node_idcs.append(range(count, count + len(ctr)))
        count += len(ctr)
    num_nodes = count

    # predecessors and successors of a lane
    pre, suc = dict(), dict()
    for key in ['u', 'v']:
        pre[key], suc[key] = [], []

    for i, lane_segment in enumerate(static_map.vector_lane_segments.values()):
        idcs = node_idcs[i]

        # points to the predecessor
        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]
        if lane_segment.predecessors is not None:
            for nbr_id in lane_segment.predecessors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre['u'].append(idcs[0])
                    pre['v'].append(node_idcs[j][-1])

        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]
        if lane_segment.successors is not None:
            for nbr_id in lane_segment.successors:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc['u'].append(idcs[-1])
                    suc['v'].append(node_idcs[j][0])
    
    # we now compute lane-level features
    # lane indices
    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane_segment in enumerate(static_map.vector_lane_segments.values()):
        lane = lane_segment 

        nbr_ids = lane.predecessors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    pre_pairs.append([i, j])

        nbr_ids = lane.successors
        if nbr_ids is not None:
            for nbr_id in nbr_ids:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    suc_pairs.append([i, j])

        nbr_id = lane.left_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                left_pairs.append([i, j])

        nbr_id = lane.right_neighbor_id
        if nbr_id is not None:
            if nbr_id in lane_ids:
                j = lane_ids.index(nbr_id)
                right_pairs.append([i, j])
    
    pre_pairs = np.asarray(pre_pairs, np.int64)
    suc_pairs = np.asarray(suc_pairs, np.int64)
    left_pairs = np.asarray(left_pairs, np.int64)
    right_pairs = np.asarray(right_pairs, np.int64)

    graph = dict()
    graph['ctrs'] = np.concatenate(ctrs, 0)
    graph['num_nodes'] = num_nodes
    graph['feats'] = np.concatenate(feats, 0)
    graph['centerlines'] = centerlines
    graph['left_boundaries'] = left_boundaries
    graph['right_boundaries'] = right_boundaries
    graph['pre'] = [pre]
    graph['suc'] = [suc]
    graph['lane_idcs'] = lane_idcs
    graph['pre_pairs'] = pre_pairs
    graph['suc_pairs'] = suc_pairs
    graph['left_pairs'] = left_pairs
    graph['right_pairs'] = right_pairs
    graph['lane_type'] = lane_type

    for k1 in ['pre', 'suc']:
        for k2 in ['u', 'v']:
            graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
    
    num_scales = 6
    # # longitudinal connections
    for key in ['pre', 'suc']:
        graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], num_scales)


    graph = preprocess(graph, cross_dist, cross_angle)

    # delete ctrs from graph
    del graph['ctrs']
    del graph['feats']

    return graph

### FROM LANE_GCN
def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs


# This function mines the left/right neighbouring nodes
def preprocess(graph, cross_dist, cross_angle=None):
    # like pre and sec, but for left and right nodes
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    # for each lane node lane_idcs returns the corresponding lane id
    num_nodes = len(lane_idcs)
    # indexing starts from 0, makes sense
    num_lanes = lane_idcs[-1].item() + 1

    # distances between all node centres
    dist = torch.tensor(graph['ctrs']).unsqueeze(1) - torch.tensor(graph['ctrs']).unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    
    
    # allows us to index through all pairs of lane nodes
    # if num_nodes == 3: [0, 0, 0, 1, 1, 1, 2, 2, 2]
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    # if num_nodes == 3: [0, 1, 2, 0, 1, 2, 0, 1, 2]
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    # if num_nodes == 3: [0, 1, 2]
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    # find possible left and right neighouring nodes
    if cross_angle is not None:
        # along lane
        f1 = torch.tensor(graph['feats'][hi])
        if len(f1.shape) == 1:
            f1 = f1.unsqueeze(0)
        # cross lane
        f2 = torch.tensor(graph['ctrs'][wi] - graph['ctrs'][hi])
        if len(f2.shape) == 1:
            f2 = f2.unsqueeze(0)
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < cross_angle)
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -cross_angle)
        right_mask = mask.logical_not()

    pre_suc_valid = False 
    if len(graph['pre_pairs'].shape) == 2 and len(graph['suc_pairs'].shape) == 2:
        pre_suc_valid = True
    # lanewise pre and suc connections
    if pre_suc_valid:
        pre = torch.tensor(graph['pre_pairs']).new().float().resize_(num_lanes, num_lanes).zero_()
        pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
        suc = torch.tensor(graph['suc_pairs']).new().float().resize_(num_lanes, num_lanes).zero_()
        suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    # find left lane nodes
    pairs = graph['left_pairs']
    if len(pairs) > 0 and pre_suc_valid:
        mat = torch.tensor(pairs).new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = torch.tensor(graph['feats'][ui])
        if len(f1.shape) == 1:
            f1 = f1.unsqueeze(0)
        f2 = torch.tensor(graph['feats'][vi])
        if len(f2.shape) == 1:
            f2 = f2.unsqueeze(0)
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    # find right lane nodes
    pairs = graph['right_pairs']
    if len(pairs) > 0 and pre_suc_valid:
        mat = torch.tensor(pairs).new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        if len(ui) == 1:
            f1 = torch.tensor(graph['feats'][[ui]])
        else:
            f1 = torch.tensor(graph['feats'][ui])
        if len(vi) == 1:
            f2 = torch.tensor(graph['feats'][[vi]])
        else:
            f2 = torch.tensor(graph['feats'][vi])
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    graph['left'] = [left]
    graph['right'] = [right]
    return graph
