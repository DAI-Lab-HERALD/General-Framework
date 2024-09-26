import math
import numpy as np

from FJMP.fjmp_metrics import return_circle_list, return_collision_threshold
from FJMP.fjmp_utils import sign_func

avg_agent_length = {
            0: 4.0,
            1: 0.7,
            2: 2.0,
            3: 2.0,
            4: 12.0
        }

avg_agent_width = {
            0: 2.0,
            1: 0.7,
            2: 0.7,
            3: 0.7,
            4: 2.5
        }

avg_pedcyc_length = 0.7
avg_pedcyc_width = 0.7

def get_obj_feats(train, data, idx, dataset, num_timesteps_in, num_timesteps_out):
    if dataset == 'argoverse2':
        if train:
            orig_idx =  0
            while True:
                # Are the observed timesteps available for this agent?
                found = True
                for i in range(num_timesteps_in):
                    if i not in data['steps'][orig_idx]:
                        found = False
                        break
                if found:
                    break
                else:
                    orig_idx = (orig_idx + 1) % len(data['trajs'])
        else:
            found_AV = False
            for i in range(len(data['track_ids'])):
                if 'AV' in data['track_ids'][i]:
                    found_AV = True
                    break
            
            assert found_AV
            assert len(data['track_ids'][i]) == num_timesteps_in + num_timesteps_out

            orig_idx = i
            del data['track_ids']
        
        orig = data['trajs'][orig_idx][num_timesteps_in - 1].copy().astype(np.float32)
        pre = data['trajs'][orig_idx][num_timesteps_in - 2] - orig 
        # Since theta is pi - arctan(.), then the range of theta is
        # max: pi - (-pi) = 2pi
        # min: pi - (pi) = 0
        theta = np.pi - np.arctan2(pre[1], pre[0])

        # rotation matrix for rotating scene
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)

        feats, feat_locs, feat_vels, gt_preds, gt_vels, has_preds, has_obss = [], [], [], [], [], [], []
        feat_psirads, gt_psirads, ctrs, feat_agenttypes, feat_agentcategories = [], [], [], [], []
        is_valid_agent = []

        for traj, step, vel, psirad, agenttype, agentcategory in zip(data['trajs'], data['steps'], data['vels'], data['psirads'], data['agenttypes'], data['agentcategories']):
            if num_timesteps_in - 1 not in step:
                is_valid_agent.append(0)
                continue

            # if not a dynamic vehicle
            if agenttype[0, 0] >= 5:
                is_valid_agent.append(0)
                continue

            # ignore track fragments
            if agentcategory[0, 0] == 0:
                is_valid_agent.append(0)
                continue

            is_valid_agent.append(1)

            # ground-truth future positions
            gt_pred = np.zeros((num_timesteps_out, 2), np.float32)
            # ground truth future velocities
            gt_vel = np.zeros((num_timesteps_out, 2), np.float32)
            # ground truth yaw angles
            gt_psirad = np.zeros((num_timesteps_out, 1), np.float32)

            # has ground-truth future mask
            has_pred = np.zeros(num_timesteps_out, bool)
            has_obs = np.zeros(num_timesteps_in + num_timesteps_out, bool)

            future_mask = np.logical_and(step >= num_timesteps_in, step < num_timesteps_in + num_timesteps_out)
            post_step = step[future_mask] - num_timesteps_in
            post_traj = traj[future_mask]
            post_vel = vel[future_mask]
            post_agenttype = agenttype[future_mask]
            post_psirad = psirad[future_mask]
            gt_pred[post_step] = post_traj
            gt_vel[post_step] = post_vel
            gt_psirad[post_step] = post_psirad
            has_pred[post_step] = 1

            # observation + future horizon
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            vel = vel[idcs]
            agenttype = agenttype[idcs]
            psirad = psirad[idcs]
            agentcategory = agentcategory[idcs]
            has_obs[step] = 1

            # only observation horizon
            obs_step = step[step < num_timesteps_in]
            obs_idcs = obs_step.argsort()
            obs_step = obs_step[obs_idcs]

            # take contiguous past to be the past
            for i in range(len(obs_step)):
                if obs_step[i] == num_timesteps_in - len(obs_step) + i:
                    break
            step = step[i:]
            traj = traj[i:]
            vel = vel[i:]
            agenttype = agenttype[i:]
            psirad = psirad[i:]
            agentcategory = agentcategory[i:]

            feat = np.zeros((num_timesteps_in + num_timesteps_out, 2), np.float32)
            feat_vel = np.zeros((num_timesteps_in + num_timesteps_out, 2), np.float32)
            feat_agenttype = np.zeros((num_timesteps_in + num_timesteps_out, 1), np.float32)
            feat_psirad = np.zeros((num_timesteps_in + num_timesteps_out, 1), np.float32)
            feat_agentcategory = np.zeros((num_timesteps_in + num_timesteps_out, 2), np.float32)

            # center and rotate positions, rotate velocities
            feat[step] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat_vel[step] = np.matmul(rot, vel.T).T

            ### NOTE: max heading is pi, min_heading is -pi (same as INTERACTION)
            # Therefore, heading + theta has min: -pi + 0 = -pi and max: pi + 2pi = 3pi
            for j in range(len(psirad)):
                psirad[j, 0] = psirad[j, 0] + theta
                # angle now between -pi and 2pi
                if psirad[j, 0] >= (2 * math.pi):
                    psirad[j, 0] = psirad[j] % (2 * math.pi)
                # if between pi and 2pi
                if psirad[j, 0] > math.pi:
                    psirad[j, 0] = -1 * ((2 * math.pi) - psirad[j, 0])
            feat_psirad[step] = psirad

            feat_agentcategory[step] = agentcategory
            feat_agenttype[step] = agenttype

            # ctrs contains the centers at the present timestep
            ctrs.append(feat[num_timesteps_in - 1, :].copy())

            feat_loc = np.copy(feat)
            # feat contains trajectory offsets
            feat[1:, :] -= feat[:-1, :]
            feat[step[0], :] = 0 

            feats.append(feat)
            feat_locs.append(feat_loc)
            feat_vels.append(feat_vel)
            feat_agenttypes.append(feat_agenttype)
            feat_psirads.append(feat_psirad)
            feat_agentcategories.append(feat_agentcategory)
            gt_preds.append(gt_pred)
            gt_vels.append(gt_vel)
            gt_psirads.append(gt_psirad)
            has_preds.append(has_pred)
            has_obss.append(has_obs)

        ctrs = np.asarray(ctrs, np.float32)
        feats = np.asarray(feats, np.float32)
        feat_locs = np.asarray(feat_locs, np.float32)
        feat_vels = np.asarray(feat_vels, np.float32)
        feat_agenttypes = np.asarray(feat_agenttypes, np.float32)
        feat_psirads = np.asarray(feat_psirads, np.float32)
        feat_agentcategories = np.asarray(feat_agentcategories, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        gt_vels = np.asarray(gt_vels, np.float32)
        gt_psirads = np.asarray(gt_psirads, np.float32)
        has_preds = np.asarray(has_preds, np.float32)
        has_obss = np.asarray(has_obss, np.float32)
        is_valid_agent = np.asarray(is_valid_agent, bool)

        ig_labels_sparse = get_interaction_labels_fjmp(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, 25, num_timesteps_in, num_timesteps_out)
        ig_labels_sparse = np.asarray(ig_labels_sparse, np.float32)

        ig_labels_dense = get_interaction_labels_fjmp(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, 60, num_timesteps_in, num_timesteps_out)
        ig_labels_dense = np.asarray(ig_labels_dense, np.float32)

        ig_labels_m2i = get_interaction_labels_m2i(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, num_timesteps_in, num_timesteps_out)
        ig_labels_m2i = np.asarray(ig_labels_m2i, np.float32)

        # Check that there are no nans
        assert not np.any(np.isnan(feat_agentcategories))
        assert not np.any(np.isnan(gt_preds))
        assert not np.any(np.isnan(gt_vels))
        assert not np.any(np.isnan(has_preds))
        assert not np.any(np.isnan(has_obss))
        assert not np.any(np.isnan(is_valid_agent))
        assert not np.any(np.isnan(ig_labels_sparse))
        assert not np.any(np.isnan(ig_labels_dense))
        assert not np.any(np.isnan(ig_labels_m2i))

        data['feat_agentcategories'] = feat_agentcategories
        data['gt_preds'] = gt_preds 
        data['gt_vels'] = gt_vels
        data['gt_psirads'] = gt_psirads
        data['has_preds'] = has_preds
        data['has_obss'] = has_obss
        data['is_valid_agent'] = is_valid_agent
        data['ig_labels_sparse'] = ig_labels_sparse
        data['ig_labels_dense'] = ig_labels_dense
        data['ig_labels_m2i'] = ig_labels_m2i
        
    
    elif dataset=='interaction':
        # center on "random" agent
        # This ensures that the random agent chosen is same for loaded graph
        # processed on lanelet2-compatible machine and agent processed on other machine.
        if train:
            # orig_idx =  [0]*data['trajs'].shape[0]
            orig_idx = 0
            while True:
                # Is the present timestep available for this agent?
                if num_timesteps_in - 1 in data['steps'][orig_idx]:
                    break 
                else:
                    orig_idx = (orig_idx + 1) % len(data['trajs'])

        # center on the agent closest to the centroid
        else:
            scored_ctrs = []
            scored_indices = []
            for i in range(len(data['trajs'])):
                if num_timesteps_in - 2 in data['steps'][i] and num_timesteps_in - 1 in data['steps'][i] and num_timesteps_in + num_timesteps_out in data['steps'][i]:
                    present_index = list(data['steps'][i]).index(num_timesteps_in - 1)
                    scored_ctrs.append(np.expand_dims(data['trajs'][i][present_index], axis=0))
                    scored_indices.append(i)

            scored_ctrs = np.concatenate(scored_ctrs, axis=0)
            centroid = np.mean(scored_ctrs, axis=0)
            dist_to_centroid = np.linalg.norm(scored_ctrs - np.expand_dims(centroid, axis=0), ord=2, axis=-1)
            closest_centroid = np.argmin(dist_to_centroid, axis=0)
            orig_idx = scored_indices[closest_centroid]
        
        orig = data['trajs'][orig_idx][num_timesteps_in - 1].copy().astype(np.float32)
        pre = data['trajs'][orig_idx][num_timesteps_in - 2] - orig 
        theta = np.pi - np.arctan2(pre[1], pre[0])
        
        assert np.isfinite(theta), " theta is not finite"

        # rotation matrix for rotating scene
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)
        
        feats, feat_locs, feat_vels, gt_preds, gt_vels, has_preds, has_obss = [], [], [], [], [], [], []
        feat_psirads, feat_shapes, gt_psirads, ctrs, feat_agenttypes = [], [], [], [], []
        is_valid_agent = []
        
        for traj, step, vel, psirad, shape, agenttype in zip(data['trajs'], data['steps'], data['vels'], data['psirads'], data['shapes'], data['agenttypes']):
            if num_timesteps_in - 1 not in step:
                is_valid_agent.append(0)
                continue
            
            is_valid_agent.append(1)
            
            # ground-truth future positions
            gt_pred = np.zeros((num_timesteps_out, 2), np.float32)
            # ground truth future velocities
            gt_vel = np.zeros((num_timesteps_out, 2), np.float32)
            # ground truth yaw angles
            gt_psirad = np.zeros((num_timesteps_out, 1), np.float32)
            
            # has ground-truth future mask
            has_pred = np.zeros(num_timesteps_out, bool)
            has_obs = np.zeros(num_timesteps_in + num_timesteps_out, bool)

            future_mask = np.logical_and(step >= num_timesteps_in, step < num_timesteps_in + num_timesteps_out)
            post_step = step[future_mask] - num_timesteps_in
            post_traj = traj[future_mask]
            post_vel = vel[future_mask]
            post_agenttype = agenttype[future_mask]

#             object_type_dict = {
#     'V':0,
#     'P':1,
#     'M':2,
#     'B':3,
#     'other': -1,
#     '0': -1
# }

            # Added to adapt to the general framework intreface in which cars are being encoded as agenttype 0
            filter = np.zeros_like(post_agenttype)
            filter[post_agenttype == 0] = 1

            # post_psirad = np.nan_to_num(psirad[future_mask]) * post_agenttype # 0 out psirad if not a car
            post_psirad = np.nan_to_num(psirad[future_mask]) * filter # 0 out psirad if not a car
            gt_pred[post_step] = post_traj
            gt_vel[post_step] = post_vel
            gt_psirad[post_step] = post_psirad
            has_pred[post_step] = 1

            # observation + future horizon
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]
            vel = vel[idcs]
            agenttype = agenttype[idcs]
            psirad = psirad[idcs]
            shape = shape[idcs]
            has_obs[step] = 1

            # only observation horizon
            obs_step = step[step < num_timesteps_in]
            obs_idcs = obs_step.argsort()
            obs_step = obs_step[obs_idcs]

            # take contiguous past to be the past
            for i in range(len(obs_step)):
                if obs_step[i] == num_timesteps_in - len(obs_step) + i:
                    break
            step = step[i:]
            traj = traj[i:]
            vel = vel[i:]
            agenttype = agenttype[i:]
            psirad = psirad[i:]
            shape = shape[i:]

            feat = np.zeros((num_timesteps_in + num_timesteps_out, 2), np.float32)
            feat_vel = np.zeros((num_timesteps_in + num_timesteps_out, 2), np.float32)
            feat_agenttype = np.zeros((num_timesteps_in + num_timesteps_out, 1), np.float32)
            feat_psirad = np.zeros((num_timesteps_in + num_timesteps_out, 1), np.float32)
            feat_shape = np.zeros((num_timesteps_in + num_timesteps_out, 2), np.float32)
            
            # center and rotate positions, rotate velocities
            feat[step] = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            feat_vel[step] = np.matmul(rot, vel.T).T

            # recalculate yaw angles
            feat_agenttype[step] = agenttype
            # feat_shape[step] = np.nan_to_num(shape) * feat_agenttype[step] # 0 out if not a car
             
            # Added to adapt to the general framework intreface in which cars are being encoded as agenttype 0
            filter = np.zeros_like(feat_agenttype[step])
            filter[feat_agenttype[step] == 0] = 1

            feat_shape[step] = np.nan_to_num(shape) * filter # 0 out if not a car 
            
            # only vehicles have a yaw angle
            # apply rotation transformation to the yaw angle
            if feat_agenttype[num_timesteps_in - 1] != 0:
                for j in range(len(psirad)):
                    psirad[j, 0] = psirad[j, 0] + theta
                    # angle now between -pi and 2pi
                    if psirad[j, 0] >= (2 * math.pi):
                        psirad[j, 0] = psirad[j] % (2 * math.pi)
                    # if between pi and 2pi
                    if psirad[j, 0] > math.pi:
                        psirad[j, 0] = -1 * ((2 * math.pi) - psirad[j, 0])
            # pedestrian/bicycle does not have yaw angle; use velocity to infer yaw when available; otherwise set to 0
            # velocity is already in se(2) transformed coordinates
            else:
                vel_transformed = feat_vel[step]
                assert len(psirad) == len(vel_transformed)
                for j in range(len(psirad)):
                    speed_j = math.sqrt(vel_transformed[j, 0] ** 2 + vel_transformed[j, 1] ** 2)
                    if speed_j == 0:
                        psirad[j, 0] = 0.
                    else:
                        psirad[j, 0] = round(sign_func(vel_transformed[j, 1]) * math.acos(vel_transformed[j, 0] / speed_j), 3)

            assert not np.any(np.isnan(psirad))
            feat_psirad[step] = psirad
            
            # ctrs contains the centers at the present timestep
            ctrs.append(feat[num_timesteps_in - 1, :].copy())
            
            feat_loc = np.copy(feat)
            # feat contains trajectory offsets
            feat[1:, :] -= feat[:-1, :]
            feat[step[0], :] = 0 

            feats.append(feat)
            feat_locs.append(feat_loc)
            feat_vels.append(feat_vel)
            feat_agenttypes.append(feat_agenttype)
            feat_psirads.append(feat_psirad)
            feat_shapes.append(feat_shape)
            gt_preds.append(gt_pred)
            gt_vels.append(gt_vel)
            gt_psirads.append(gt_psirad)
            has_preds.append(has_pred)
            has_obss.append(has_obs)
        
        ctrs = np.asarray(ctrs, np.float32)
        feats = np.asarray(feats, np.float32)
        feat_locs = np.asarray(feat_locs, np.float32)
        feat_vels = np.asarray(feat_vels, np.float32)
        feat_agenttypes = np.asarray(feat_agenttypes, np.float32)
        feat_psirads = np.asarray(feat_psirads, np.float32)
        feat_shapes = np.asarray(feat_shapes, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        gt_vels = np.asarray(gt_vels, np.float32)
        gt_psirads = np.asarray(gt_psirads, np.float32)
        has_preds = np.asarray(has_preds, np.float32)
        has_obss = np.asarray(has_obss, np.float32)
        is_valid_agent = np.asarray(is_valid_agent, bool)

        ig_labels_dense = get_interaction_labels_dense(idx, ctrs, feat_locs, feat_vels, feat_shapes, has_obss, is_valid_agent, feat_agenttypes, num_timesteps_in, num_timesteps_out)
        ig_labels_dense = np.asarray(ig_labels_dense, np.float32)

        ig_labels_sparse = get_interaction_labels_sparse(idx, ctrs, feat_locs, feat_vels, feat_psirads, feat_shapes, has_obss, is_valid_agent, feat_agenttypes, num_timesteps_in, num_timesteps_out)
        ig_labels_sparse = np.asarray(ig_labels_sparse, np.float32)

        # Check that there are no nans
        assert not np.any(np.isnan(feat_shapes))
        assert not np.any(np.isnan(gt_preds))
        assert not np.any(np.isnan(gt_vels))
        assert not np.any(np.isnan(has_preds))
        assert not np.any(np.isnan(has_obss))
        assert not np.any(np.isnan(is_valid_agent))
        assert not np.any(np.isnan(ig_labels_dense))
        assert not np.any(np.isnan(ig_labels_sparse))

        data['feat_shapes'] = feat_shapes
        data['gt_preds'] = gt_preds 
        data['gt_vels'] = gt_vels
        data['gt_psirads'] = gt_psirads
        data['has_preds'] = has_preds
        data['has_obss'] = has_obss
        data['is_valid_agent'] = is_valid_agent
        data['ig_labels_dense'] = ig_labels_dense
        data['ig_labels_sparse'] = ig_labels_sparse


    theta = np.clip(0,theta,np.pi * 2)
    assert not np.any(np.isnan(ctrs))
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isnan(feat_locs))
    assert not np.any(np.isnan(feat_vels))
    assert not np.any(np.isnan(feat_agenttypes))
    assert not np.any(np.isnan(feat_psirads))
    
    data['feats'] = feats 
    data['ctrs'] = ctrs 
    data['feat_locs'] = feat_locs
    data['feat_vels'] = feat_vels 
    data['feat_agenttypes'] = feat_agenttypes
    data['feat_psirads'] = feat_psirads 
    data['orig'] = orig 
    data['theta'] = theta 
    data['rot'] = rot 

    return data



def get_interaction_labels_fjmp(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, eps_I, num_timesteps_in, num_timesteps_out):

    feat_locs = feat_locs[:, num_timesteps_in:]
    feat_vels = feat_vels[:, num_timesteps_in:]
    feat_psirads = feat_psirads[:, num_timesteps_in:]
    
    # only consider the future
    has_obss = has_obss[:, num_timesteps_in:]
    
    N = feat_locs.shape[0]
    labels = np.zeros((N, N))
    orig_trajs = feat_locs 

    circle_lists = []
    for i in range(N):
        length_i = avg_agent_length[feat_agenttypes[i, num_timesteps_in - 1, 0]]
        width_i = avg_agent_width[feat_agenttypes[i, num_timesteps_in - 1, 0]]
        traj_i = orig_trajs[i][has_obss[i] == 1]
        psirad_i = feat_psirads[i][has_obss[i] == 1]
        # shape is [60, c, 2], where c is the number of circles prescribed to vehicle i (depends on the size/shape of vehicle i)
        circle_lists.append(return_circle_list(traj_i[:, 0], traj_i[:, 1], length_i, width_i, psirad_i[:, 0]))
    
    for a in range(1, N):
        for b in range(a):
            width_a = avg_agent_width[feat_agenttypes[a, num_timesteps_in - 1, 0]]
            width_b = avg_agent_width[feat_agenttypes[b, num_timesteps_in - 1, 0]]
            # for each (unordered) pairs of vehicles, we check if they are interacting
            # by checking if there is a collision at any pair of future timesteps. 
            circle_list_a = circle_lists[a]
            circle_list_b = circle_lists[b]

            # threshold determined according to widths of vehicles
            thresh = return_collision_threshold(width_a, width_b)

            dist = np.expand_dims(np.expand_dims(circle_list_a, axis=1), axis=2) - np.expand_dims(np.expand_dims(circle_list_b, axis=0), axis=3)
            dist = np.linalg.norm(dist, axis=-1, ord=2)
            
            is_coll = dist < thresh
            is_coll_cumul = is_coll.sum(2).sum(2)

            
            # binary mask of shape [T_a, T_b], where T_a is the number of ground-truth future positions present in a's trajectory, and b defined similarly.
            is_coll_mask = is_coll_cumul > 0

            if is_coll_mask.sum() < 1:
                continue

            # fill in for indices (0) that do not have a ground-truth position
            for en, ind in enumerate(has_obss[a]):
                if ind == 0:
                    is_coll_mask = np.insert(is_coll_mask, en, 0, axis=0)

            for en, ind in enumerate(has_obss[b]):
                if ind == 0:
                    is_coll_mask = np.insert(is_coll_mask, en, 0, axis=1)  

            assert is_coll_mask.shape == (num_timesteps_out, num_timesteps_out)

            # [P, 2], first index is a, second is b; P is number of colliding pairs
            coll_ids = np.argwhere(is_coll_mask == 1)
            # only preserve the colliding pairs that are within eps_I (e.g. 6 seconds (= 60 timesteps)) of eachother
            valid_coll_mask = np.abs(coll_ids[:, 0] - coll_ids[:, 1]) <= eps_I

            if valid_coll_mask.sum() < 1:
                continue

            coll_ids = coll_ids[valid_coll_mask]
            
            # first order small_timestep, larger_timestep, index_of_larger_timestep
            coll_ids_sorted = np.sort(coll_ids, axis=-1)
            coll_ids_argsorted = np.argsort(coll_ids, axis=-1)

            conflict_time_influencer = coll_ids_sorted[:, 0].min()
            influencer_mask = coll_ids_sorted[:, 0] == conflict_time_influencer
            candidate_reactors = coll_ids_sorted[coll_ids_sorted[:, 0] == conflict_time_influencer][:, 1]
            conflict_time_reactor = candidate_reactors.min()
            conflict_time_reactor_id = np.argmin(candidate_reactors)

            a_is_influencer = coll_ids_argsorted[influencer_mask][conflict_time_reactor_id][0] == 0
            if a_is_influencer:
                min_a = conflict_time_influencer 
                min_b = conflict_time_reactor 
            else:
                min_a = conflict_time_reactor 
                min_b = conflict_time_influencer
            
            # a is the influencer
            if min_a < min_b:
                labels[a, b] = 1
            # b is the influencer
            elif min_b < min_a:
                labels[b, a] = 1
            else:                    
                # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                if np.linalg.norm(feat_vels[a][min_a], ord=2) > np.linalg.norm(feat_vels[b][min_b], ord=2):
                    labels[a, b] = 1
                elif np.linalg.norm(feat_vels[a][min_a], ord=2) < np.linalg.norm(feat_vels[b][min_b], ord=2):
                    labels[b, a] = 1
                else:
                    labels[a, b] = 0
                    labels[b, a] = 0
    
    # i --> j iff ig_labels_npy[i,j] = 1
    n_agents = labels.shape[0]

    assert n_agents == np.sum(is_valid_agent)

    # labels for interaction visualization
    valid_mask = is_valid_agent

    # add indices for the invalid agents (either not cars, or no gt position at timestep 9)
    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=1)

    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=0)

    # Here we now construct the interaction labels for SSL.
    # There is a label on each (undirected) edge in the fully connected interaction graph
    ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
    count = 0
    for i in range(len(is_valid_agent)):
        if is_valid_agent[i] == 0:
            assert labels[i].sum() == 0
            continue
        
        for j in range(len(is_valid_agent)):
            if is_valid_agent[j] == 0:
                assert labels[:,j].sum() == 0
                continue
            
            # we want only the indices where i < j
            if i >= j:
                continue 

            if labels[i, j] == 1:
                # i influences j
                ig_labels[count] = 1
                # j influences i
            elif labels[j, i] == 1:
                ig_labels[count] = 2
            
            count += 1   

    assert ig_labels.shape[0] == count

    return ig_labels

def get_interaction_labels_m2i(idx, ctrs, feat_locs, feat_vels, feat_psirads, has_obss, is_valid_agent, feat_agenttypes, num_timesteps_in, num_timesteps_out):
    """
    feat_locs: location features in transformed coordinates (not offsets but absolute positions) (past + future): [N, 40, 2]
    feat_vels: velocity features (past + future): [N, 40, 2]
    shapes: vehicle shape: [N, 40, 2] (length, width)
    has_obss: ground-truth mask (past + future): [N, 40]
    is_valid_agent: whether the agent is being considered during training (only cars considered): [N, ]
    """
    
    N = feat_locs.shape[0]
    # NOTE: labels[i, j] = 0 if no interaction exists, = 1 if i --> j, = 2 if j --> i
    labels = np.zeros((N, N))

    orig_trajs = feat_locs
    for a in range(1, N):
        for b in range(a):
            # sum of the length of these two vehicles.               
            len_a = avg_agent_length[feat_agenttypes[a, num_timesteps_in - 1, 0]]
            if np.isnan(len_a):
                print("This should not happen")
                len_a = 1
            len_b =  avg_agent_length[feat_agenttypes[b, num_timesteps_in - 1, 0]]
            if np.isnan(len_b):
                print("This should not happen")
                len_b = 1
            
            EPSILON_D = len_a + len_b
            
            # filter for the timesteps with a ground-truth position
            traj_a = orig_trajs[a][has_obss[a] == 1]
            traj_b = orig_trajs[b][has_obss[b] == 1]

            traj_a_expanded = traj_a.reshape(-1, 1, 2)
            traj_b_expanded = traj_b.reshape(1, -1, 2)

            # [A, B] array, where A = traj_a.shape[0], B = traj_a.shape[1]
            dist_ab = np.sqrt(np.sum((traj_a_expanded - traj_b_expanded)**2, axis=2))

            # fill in for indices that do not have a ground-truth position
            for en, ind in enumerate(has_obss[a]):
                if ind == 0:
                    dist_ab = np.insert(dist_ab, en, 10000, axis=0)

            for en, ind in enumerate(has_obss[b]):
                if ind == 0:
                    dist_ab = np.insert(dist_ab, en, 10000, axis=1)   

            # broadcast back into a length 110 tensor first.
            assert dist_ab.shape == (num_timesteps_in + num_timesteps_out, num_timesteps_in + num_timesteps_out) 

            # We only consider the future positions, as the past positions are already fed into the model.
            dist_ab = dist_ab[num_timesteps_in:, num_timesteps_in:]            

            # in [0, 59] (future timestep)
            min_a, min_b = np.unravel_index(dist_ab.argmin(), dist_ab.shape)
            
            if np.min(dist_ab) > EPSILON_D:
                continue 
            
            if min_a < min_b:
                labels[a, b] = 1
            elif min_b < min_a:
                labels[b, a] = 1
            else:                    
                # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                if np.linalg.norm(feat_vels[a][min_a + num_timesteps_in], ord=2) > np.linalg.norm(feat_vels[b][min_b + num_timesteps_in], ord=2):
                    labels[a, b] = 1
                elif np.linalg.norm(feat_vels[a][min_a + num_timesteps_in], ord=2) < np.linalg.norm(feat_vels[b][min_b + num_timesteps_in], ord=2):
                    labels[b, a] = 1
                else:
                    labels[a, b] = 0
                    labels[b, a] = 0

    # i --> j iff ig_labels_npy[i,j] = 1
    n_agents = labels.shape[0]

    assert n_agents == np.sum(is_valid_agent)

    # labels for interaction visualization
    valid_mask = is_valid_agent

    # add indices for the invalid agents (no gt position at timestep 49)
    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=1)

    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=0)

    # Here we now construct the interaction labels for SSL.
    # There is a label on each (undirected) edge in the fully connected interaction graph
    ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
    count = 0
    for i in range(len(is_valid_agent)):
        if is_valid_agent[i] == 0:
            assert labels[i].sum() == 0
            continue
        
        for j in range(len(is_valid_agent)):
            if is_valid_agent[j] == 0:
                assert labels[:,j].sum() == 0
                continue
            
            # we want only the indices where i < j
            if i >= j:
                continue 

            if labels[i, j] == 1:
                # i influences j
                ig_labels[count] = 1
                # j influences i
            elif labels[j, i] == 1:
                ig_labels[count] = 2
            
            count += 1   

    assert ig_labels.shape[0] == count

    return ig_labels


def get_interaction_labels_sparse(idx, ctrs, feat_locs, feat_vels, feat_psirads, shapes, has_obss, is_valid_agent, agenttypes, num_timesteps_in, num_timesteps_out):

    # only consider the future
    # we can use data in se(2) transformed coordinates (interaction labelling invariant to se(2)-transformations)
    feat_locs = feat_locs[:, num_timesteps_in:]
    feat_vels = feat_vels[:, num_timesteps_in:]
    feat_psirads = feat_psirads[:, num_timesteps_in:]
    has_obss = has_obss[:, num_timesteps_in:]
    
    N = feat_locs.shape[0]
    labels = np.zeros((N, N))
    orig_trajs = feat_locs 

    circle_lists = []
    for i in range(N):
        agenttype_i = agenttypes[i][num_timesteps_in - 1]
        if agenttype_i == 1:
            shape_i = shapes[i][num_timesteps_in - 1]
            length = shape_i[0]
            width = shape_i[1]
        else:
            length = avg_pedcyc_length
            width = avg_pedcyc_width

        traj_i = orig_trajs[i][has_obss[i] == 1]
        psirad_i = feat_psirads[i][has_obss[i] == 1]
        # shape is [30, c, 2], where c is the number of circles prescribed to vehicle i (depends on the size/shape of vehicle i)
        circle_lists.append(return_circle_list(traj_i[:, 0], traj_i[:, 1], length, width, psirad_i[:, 0]))
    
    for a in range(1, N):
        for b in range(a):
            agenttype_a = agenttypes[a][num_timesteps_in - 1]
            if agenttype_a == 1:
                shape_a = shapes[a][num_timesteps_in - 1]
                width_a = shape_a[1]
            else:
                width_a = avg_pedcyc_width

            agenttype_b = agenttypes[b][num_timesteps_in - 1]
            if agenttype_b == 1:
                shape_b = shapes[b][num_timesteps_in - 1]
                width_b = shape_b[1]
            else:
                width_b = avg_pedcyc_width
            
            # for each (unordered) pairs of vehicles, we check if they are interacting
            # by checking if there is a collision at any pair of future timesteps. 
            circle_list_a = circle_lists[a]
            circle_list_b = circle_lists[b]

            # threshold determined according to widths of vehicles
            thresh = return_collision_threshold(width_a, width_b)

            dist = np.expand_dims(np.expand_dims(circle_list_a, axis=1), axis=2) - np.expand_dims(np.expand_dims(circle_list_b, axis=0), axis=3)
            dist = np.linalg.norm(dist, axis=-1, ord=2)
            
            # [T_a, T_b, num_circles_a, num_circles_b], where T_a is the number of ground-truth future positions present in a's trajectory, and b defined similarly.
            is_coll = dist < thresh
            is_coll_cumul = is_coll.sum(2).sum(2)
            # binary mask of shape [T_a, T_b]
            is_coll_mask = is_coll_cumul > 0

            if is_coll_mask.sum() < 1:
                continue

            # fill in for indices (0) that do not have a ground-truth position
            for en, ind in enumerate(has_obss[a]):
                if ind == 0:
                    is_coll_mask = np.insert(is_coll_mask, en, 0, axis=0)

            for en, ind in enumerate(has_obss[b]):
                if ind == 0:
                    is_coll_mask = np.insert(is_coll_mask, en, 0, axis=1)  

            assert is_coll_mask.shape == (num_timesteps_out, num_timesteps_out)

            # [P, 2], first index is a, second is b; P is number of colliding pairs
            coll_ids = np.argwhere(is_coll_mask == 1)
            # only preserve the colliding pairs that are within 2.5 seconds (= 25 timesteps) of eachother
            valid_coll_mask = np.abs(coll_ids[:, 0] - coll_ids[:, 1]) <= 25

            if valid_coll_mask.sum() < 1:
                continue

            coll_ids = coll_ids[valid_coll_mask]
            
            # first order small_timestep, larger_timestep, index_of_larger_timestep
            coll_ids_sorted = np.sort(coll_ids, axis=-1)
            coll_ids_argsorted = np.argsort(coll_ids, axis=-1)

            conflict_time_influencer = coll_ids_sorted[:, 0].min()
            influencer_mask = coll_ids_sorted[:, 0] == conflict_time_influencer
            candidate_reactors = coll_ids_sorted[coll_ids_sorted[:, 0] == conflict_time_influencer][:, 1]
            conflict_time_reactor = candidate_reactors.min()
            conflict_time_reactor_id = np.argmin(candidate_reactors)

            a_is_influencer = coll_ids_argsorted[influencer_mask][conflict_time_reactor_id][0] == 0
            if a_is_influencer:
                min_a = conflict_time_influencer 
                min_b = conflict_time_reactor 
            else:
                min_a = conflict_time_reactor 
                min_b = conflict_time_influencer
            
            # a is the influencer
            if min_a < min_b:
                labels[a, b] = 1
            # b is the influencer
            elif min_b < min_a:
                labels[b, a] = 1
            else:                    
                # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                if np.linalg.norm(feat_vels[a][min_a], ord=2) > np.linalg.norm(feat_vels[b][min_b], ord=2):
                    labels[a, b] = 1
                elif np.linalg.norm(feat_vels[a][min_a], ord=2) < np.linalg.norm(feat_vels[b][min_b], ord=2):
                    labels[b, a] = 1
                else:
                    labels[a, b] = 0
                    labels[b, a] = 0
    
    n_agents = labels.shape[0]

    assert n_agents == np.sum(is_valid_agent)

    # labels for interaction visualization
    valid_mask = is_valid_agent

    # add indices for the invalid agents (no gt position at timestep 9)
    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=1)

    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=0)

    # There is a label on each (undirected) edge in the fully connected interaction graph
    ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
    count = 0
    for i in range(len(is_valid_agent)):
        if is_valid_agent[i] == 0:
            assert labels[i].sum() == 0
            continue
        
        for j in range(len(is_valid_agent)):
            if is_valid_agent[j] == 0:
                assert labels[:,j].sum() == 0
                continue
            
            # we want only the indices where i < j
            if i >= j:
                continue 

            # i influences j
            if labels[i, j] == 1:
                ig_labels[count] = 1
            # j influences i
            elif labels[j, i] == 1:
                ig_labels[count] = 2
            
            count += 1   

    assert ig_labels.shape[0] == count

    return ig_labels

def get_interaction_labels_dense(idx, ctrs, feat_locs, feat_vels, shapes, has_obss, is_valid_agent, agenttypes, num_timesteps_in, num_timesteps_out):
    
    N = feat_locs.shape[0]
    # labels[i, j] = 0 if no interaction exists, = 1 if i --> j, = 2 if j --> i
    labels = np.zeros((N, N))

    orig_trajs = feat_locs
    for a in range(1, N):
        for b in range(a):
            agenttype_a = agenttypes[a][num_timesteps_in - 1]
            if agenttype_a == 1:
                shape_a = shapes[a][num_timesteps_in - 1]
                len_a = shape_a[0]
            else:
                len_a = avg_pedcyc_length

            agenttype_b = agenttypes[b][num_timesteps_in - 1]
            if agenttype_b == 1:
                shape_b = shapes[b][num_timesteps_in - 1]
                len_b = shape_b[0]
            else:
                len_b = avg_pedcyc_length
            
            # sum of the lengths of the two agents
            EPSILON_D = len_a + len_b
            
            # filter for the timesteps with a ground-truth position
            traj_a = orig_trajs[a][has_obss[a] == 1]
            traj_b = orig_trajs[b][has_obss[b] == 1]

            traj_a_expanded = traj_a.reshape(-1, 1, 2)
            traj_b_expanded = traj_b.reshape(1, -1, 2)

            # [A, B] array, where A = traj_a.shape[0], B = traj_a.shape[1]
            dist_ab = np.sqrt(np.sum((traj_a_expanded - traj_b_expanded)**2, axis=2))

            # fill in for indices that do not have a ground-truth position
            for en, ind in enumerate(has_obss[a]):
                if ind == 0:
                    dist_ab = np.insert(dist_ab, en, 10000, axis=0)

            for en, ind in enumerate(has_obss[b]):
                if ind == 0:
                    dist_ab = np.insert(dist_ab, en, 10000, axis=1)   

            # broadcast back into a length 40 tensor first.
            assert dist_ab.shape == (num_timesteps_in + num_timesteps_out, num_timesteps_in + num_timesteps_out) 

            # We only consider the future positions, as the past positions are already fed into the model.
            dist_ab = dist_ab[num_timesteps_in:, num_timesteps_in:]            

            # in [0, 29] (future timestep)
            min_a, min_b = np.unravel_index(dist_ab.argmin(), dist_ab.shape)
            
            if np.min(dist_ab) > EPSILON_D:
                continue 
            
            if min_a < min_b:
                labels[a, b] = 1
            elif min_b < min_a:
                labels[b, a] = 1
            else:                    
                # if both reach the conflict point at the same timestep, the influencer is the vehicle with the higher velocity @ the conflict point.
                if np.linalg.norm(feat_vels[a][min_a + num_timesteps_in], ord=2) > np.linalg.norm(feat_vels[b][min_b + num_timesteps_in], ord=2):
                    labels[a, b] = 1
                elif np.linalg.norm(feat_vels[a][min_a + num_timesteps_in], ord=2) < np.linalg.norm(feat_vels[b][min_b + num_timesteps_in], ord=2):
                    labels[b, a] = 1
                else:
                    labels[a, b] = 0
                    labels[b, a] = 0

    # i --> j iff ig_labels_npy[i,j] = 1
    n_agents = labels.shape[0]

    assert n_agents == np.sum(is_valid_agent)

    # labels for interaction visualization
    valid_mask = is_valid_agent

    # add indices for the invalid agents (either not cars, or no gt position at timestep 9)
    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=1)

    for ind in range(valid_mask.shape[0]):
        if valid_mask[ind] == 0:
            labels = np.insert(labels, ind, 0, axis=0)

    # There is a label on each (undirected) edge in the fully connected interaction graph
    ig_labels = np.zeros(int(n_agents * (n_agents - 1) / 2))
    count = 0
    for i in range(len(is_valid_agent)):
        if is_valid_agent[i] == 0:
            assert labels[i].sum() == 0
            continue
        
        for j in range(len(is_valid_agent)):
            if is_valid_agent[j] == 0:
                assert labels[:,j].sum() == 0
                continue
            
            # we want only the indices where i < j
            if i >= j:
                continue 

            # i influences j
            if labels[i, j] == 1:
                ig_labels[count] = 1
            # j influences i
            elif labels[j, i] == 1:
                ig_labels[count] = 2
            
            count += 1   

    assert ig_labels.shape[0] == count

    return ig_labels
