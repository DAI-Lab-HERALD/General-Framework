import copy
import lanelet2
import numpy as np
import os
import pandas as pd
import re
import torch

from av2.geometry.interpolate import compute_midpoint_line
from lanelet2.projection import UtmProjector
from scipy import sparse


filename_pattern = re.compile(r'^(\w+)_(\w+).csv$')
projector = UtmProjector(lanelet2.io.Origin(0, 0))
traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                            lanelet2.traffic_rules.Participants.Vehicle)


cross_dist = 10
cross_angle =  1 * np.pi
# from FJMP

def read_interaction_data(file_path):
    # csv_file = self.mapping[idx]
    # city = self.filename_pattern.match(csv_file).group(1)
    # csv_path = os.path.join(self.tracks_reformatted, csv_file)
    city = filename_pattern.match(os.path.basename(file_path)).group(1)
    csv_path = file_path

    """TRACK_ID,FRAME_ID,TIMESTAMP_MS,AGENT_TYPE,X,Y,VX,VY,PSI_RAD,LENGTH,WIDTH"""
    df = pd.read_csv(csv_path)

    agt_ts = np.sort(np.unique(df['timestamp_ms'].values))
    timestamp_mapping = dict()
    for i, ts in enumerate(agt_ts):
        timestamp_mapping[ts] = i

    trajs = np.concatenate((
        df.x.to_numpy().reshape(-1, 1),
        df.y.to_numpy().reshape(-1, 1)
    ), 1)

    vels = np.concatenate((
        df.vx.to_numpy().reshape(-1, 1),
        df.vy.to_numpy().reshape(-1, 1)
    ), 1)

    psirads = df.psi_rad.to_numpy().reshape(-1, 1)

    agenttypes = df.agent_type
    agenttypes = np.array([1 if x == 'car' else 0 for x in agenttypes]).reshape(-1, 1)

    shapes = np.concatenate((
        df.length.to_numpy().reshape(-1, 1),
        df.width.to_numpy().reshape(-1, 1)
    ), 1)

    # the timestep indices the trajectory contains
    steps = [timestamp_mapping[x] for x in df['timestamp_ms'].values]
    steps = np.asarray(steps, np.int64)

    num_cases = len(df['case_id'].unique())
    num_agents = len(df['track_id'].unique())
    objs = df.groupby(['case_id', 'track_id']).groups 
    keys = list(objs.keys())
    # ctx_trajs, ctx_steps, ctx_vels, ctx_psirads, ctx_shapes, ctx_agenttypes = [], [], [], [], [], []
    ctx_trajs = np.empty((num_cases, num_agents, 40, trajs.shape[1]))
    ctx_trajs[:] = np.nan
    ctx_steps = np.empty((num_cases, num_agents, 40))
    ctx_steps[:] = np.nan
    ctx_vels = np.empty((num_cases, num_agents, 40, vels.shape[1]))
    ctx_vels[:] = np.nan
    ctx_psirads = np.empty((num_cases, num_agents, 40, 1))
    ctx_psirads[:] = np.nan
    ctx_shapes = np.empty((num_cases, num_agents, 40, shapes.shape[1]))
    ctx_shapes[:] = np.nan
    ctx_agenttypes = np.empty((num_cases, num_agents, 40, 1))
    ctx_agenttypes[:] = np.nan

    for key in keys:
        case_id, agent_id = key
        case_id = int(case_id) - 1
        agent_id = int(agent_id) - 1
        idcs = objs[key]
        ctx_trajs[case_id, agent_id, :len(idcs),:] = trajs[idcs]
        ctx_steps[case_id, agent_id, :len(idcs)] = steps[idcs]
        ctx_vels[case_id, agent_id, :len(idcs),:] = vels[idcs]
        ctx_psirads[case_id, agent_id, :len(idcs),:] = psirads[idcs]
        ctx_shapes[case_id, agent_id, :len(idcs),:] = shapes[idcs]
        ctx_agenttypes[case_id, agent_id, :len(idcs),:] = agenttypes[idcs]        

    data = dict()
    data['city'] = city 
    data['trajs'] = ctx_trajs
    data['steps'] = ctx_steps 
    data['vels'] = ctx_vels
    data['psirads'] = ctx_psirads
    data['shapes'] = ctx_shapes
    data['agenttypes'] = ctx_agenttypes

    return data


def get_lane_graph(file_path):
    # Note that we process the full lane graph -- we do not have a prediction range like LaneGCN
    # map_path = os.path.join(self.config['maps'], data['city'] + '.osm')
    # map = lanelet2.io.load(map_path, self.projector)
    # routing_graph = lanelet2.routing.RoutingGraph(map, self.traffic_rules)
    map_path = os.path.join(file_path)
    map = lanelet2.io.load(map_path, projector)
    routing_graph = lanelet2.routing.RoutingGraph(map, traffic_rules)

    is_intersection = 'Intersection' in os.path.basename(file_path)

    # build node features
    lane_ids, ctrs, feats = [], [], []
    centerlines, left_boundaries, right_boundaries = [], [], []
    lane_type = []
    for ll in map.laneletLayer:
        left_boundary = np.zeros((len(ll.leftBound), 2))
        right_boundary  = np.zeros((len(ll.rightBound), 2))

        for i in range(len(ll.leftBound)):
            left_boundary[i][0] = copy.deepcopy(ll.leftBound[i].x)
            left_boundary[i][1] = copy.deepcopy(ll.leftBound[i].y)

        for i in range(len(ll.rightBound)):
            right_boundary[i][0] = copy.deepcopy(ll.rightBound[i].x)
            right_boundary[i][1] = copy.deepcopy(ll.rightBound[i].y)
        
        # computes centerline with min(max(M,N), 10) data points per lanelet
        centerline, _ = compute_midpoint_line(left_boundary, right_boundary, min(10, max(left_boundary.shape[0], right_boundary.shape[0])))
        centerline = copy.deepcopy(centerline)       

        if ll.attributes['subtype'] == 'road' or ll.attributes['subtype'] == 'highway':
            lane_type_val = 'VEHICLE'
        else:
            raise ValueError('Unknown lanelet type: {}'.format(ll.attributes['subtype']))


        # Get the lane marker types
        lane_type.append((lane_type_val, is_intersection))

        # process lane centerline in same way as agent trajectories
        # centerline = np.matmul(data['rot'], (centerline - data['orig'].reshape(-1, 2)).T).T
        # left_boundary = np.matmul(data['rot'], (left_boundary - data['orig'].reshape(-1, 2)).T).T
        # right_boundary = np.matmul(data['rot'], (right_boundary - data['orig'].reshape(-1, 2)).T).T
        
        # num_segs = len(centerline) - 1
        # locations between the centerline segments
        ctrs.append(np.asarray((centerline[:-1] + centerline[1:]) / 2.0, np.float32))
        # distances between the centerline segments
        feats.append(np.asarray(centerline[1:] - centerline[:-1], np.float32))
        lane_ids.append(ll.id)
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
    
    for i, lane_id in enumerate(lane_ids):
        lane = map.laneletLayer[lane_id]
        idcs = node_idcs[i]

        # points to the predecessor
        pre['u'] += idcs[1:]
        pre['v'] += idcs[:-1]
        if len(routing_graph.previous(lane)) > 0:
            for prev_lane in routing_graph.previous(lane):
                if prev_lane.id in lane_ids:
                    j = lane_ids.index(prev_lane.id)
                    pre['u'].append(idcs[0])
                    pre['v'].append(node_idcs[j][-1])

        # points to the successor
        suc['u'] += idcs[:-1]
        suc['v'] += idcs[1:]
        if len(routing_graph.following(lane)) > 0:
            for foll_lane in routing_graph.following(lane):
                if foll_lane.id in lane_ids:
                    j = lane_ids.index(foll_lane.id)
                    suc['u'].append(idcs[-1])
                    suc['v'].append(node_idcs[j][0])

    # we now compute lane-level features
    # lane indices
    lane_idcs = []
    for i, idcs in enumerate(node_idcs):
        lane_idcs.append(i * np.ones(len(idcs), np.int64))
    lane_idcs = np.concatenate(lane_idcs, 0)

    pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
    for i, lane_id in enumerate(lane_ids):
        lane = map.laneletLayer[lane_id]

        # compute lane_id pairs of predecessor [u,v]
        if len(routing_graph.previous(lane)) > 0:
            for prev_lane in routing_graph.previous(lane):
                if prev_lane.id in lane_ids:
                    j = lane_ids.index(prev_lane.id)
                    pre_pairs.append([i, j])

        # compute lane_id pairs of successor [u,v]
        if len(routing_graph.following(lane)) > 0:
            for foll_lane in routing_graph.following(lane):
                if foll_lane.id in lane_ids:
                    j = lane_ids.index(foll_lane.id)
                    suc_pairs.append([i, j])

        # compute lane_id pairs of left [u,v]
        if routing_graph.left(lane) is not None:
            if routing_graph.left(lane).id in lane_ids:
                j = lane_ids.index(routing_graph.left(lane).id)
                left_pairs.append([i, j])

        # compute lane_id pairs of right [u,v]
        if routing_graph.right(lane) is not None:
            if routing_graph.right(lane).id in lane_ids:
                j = lane_ids.index(routing_graph.right(lane).id)
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
    # longitudinal connections
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