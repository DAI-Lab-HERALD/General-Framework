import numpy as np
import pandas as pd

from scipy import interpolate

# String to number mapping for agent types
# np.fromstring(np.array(['V']), dtype=np.uint32) = 86
# np.fromstring(np.array(['P']), dtype=np.uint32) = 80
# np.fromstring(np.array(['B']), dtype=np.uint32) = 66
# np.fromstring(np.array(['M']), dtype=np.uint32) = 77
# np.fromstring(np.array(['0']), dtype=np.uint32) = 48

agent_width_dict = {
    86: 1.7,
    80: 0.6,
    66: 0.6,
    77: 0.8,
    48: 0.0

}

average_agent_speed_dict = {
    86: 6.6, #13.9,
    80: 1.3,
    66: 4.0, 
    77: 6.6, #13.9
    48: 0.0
} # speed in m/s, taken from statistics in NuScenes dataset which was recorded in an urban environment

 

def get_crossing_trajectories(Y, T):
    ''' 
    Function to get the crossing trajectories between sets of trajectories

    Parameters
    ----------
    Y : np.array
        Multidimensional array containing the trajectories to be compared.
        Shape: [batch_size, n_agents, n_timesteps, n_features]

    T : np.array
        Array containing the types of the agents.
        Shape: [batch_size, n_agents]

    Returns
    -------
    crossing_agent_ids: list of np_arrays
        List of length batch_size containing arrays of the indices of exisitng agent pairs.
    crossing_class: list of np.arrays
        List of length batch_size containing arrays on how and if the trajectories of existing agent pairs cross each other.
    '''

    n_batches = Y.shape[0]
    # Get maximum number of agents
    n_agents = Y.shape[1]

    # Get number of timesteps
    n_timesteps = Y.shape[2]

    T_agents = np.tile(T[:,:,None,None,None], (1, 1, n_agents, n_timesteps, n_timesteps))

    # Vectorize dictionary
    agent_width_dict_vect = np.vectorize(agent_width_dict.get)

    # Get thresholds depending on the average width of the agents
    thresholds = agent_width_dict_vect(T_agents)

    trajs_agent1 = np.tile(Y[:,None,:,None], (1,n_agents,1,n_timesteps,1,1))
    trajs_agent2 = np.tile(Y[:,:,None,:,None], (1,1,n_agents,1,n_timesteps,1))
    # Calculate the distance between the agents at each timestep
    dist = np.linalg.norm(trajs_agent1 - trajs_agent2, axis=-1)

    # Get where agents are close to each other depending on average class width
    d1 = dist < thresholds
    # only get the upper diagonal of the matrix w.r.t. the agents
    # agent pairs are stacked within each batch
    triu_indices = np.triu_indices(n_agents, k=1)
    d1_pairs_perBatch = d1[:,triu_indices[0],triu_indices[1],:,:]

    d1_pairs = d1_pairs_perBatch.reshape(-1, n_timesteps, n_timesteps)

    agent_ids_gen = np.concatenate([triu_indices[0][:,None], triu_indices[1][:,None]], axis=-1)

    # Get the crossing trajectories
    crossing_class_gen = np.zeros((n_batches, n_agents*(n_agents-1)))

    num_agent_pairs = triu_indices[0].shape[0]

    for i in range(len(d1_pairs)):
        if np.any(d1_pairs[i]):
            first_intersecting_timesteps = np.unravel_index(np.argmax(d1_pairs[i]), d1_pairs[i].shape)
            if first_intersecting_timesteps[0] < first_intersecting_timesteps[1]:
                crossing_class_gen[int(np.floor(i/num_agent_pairs)), i%num_agent_pairs] = 1

            elif first_intersecting_timesteps[0] > first_intersecting_timesteps[1]:
                crossing_class_gen[int(np.floor(i/num_agent_pairs)), i%num_agent_pairs] = 2
        else:
            crossing_class_gen[int(np.floor(i/num_agent_pairs)), i%num_agent_pairs] = 0


    # create duplicate array of the crossing_class array
    crossing_class_copy = crossing_class_gen[:,:num_agent_pairs].copy()
    # change the crossing classes around so that there are examples with the swapped agent order
    crossing_class_copy[crossing_class_gen[:,:num_agent_pairs] == 1] = 2
    crossing_class_copy[crossing_class_gen[:,:num_agent_pairs] == 2] = 1

    crossing_class_gen[:, num_agent_pairs:] = crossing_class_copy

    agent_ids_gen = np.concatenate([agent_ids_gen, np.concatenate([triu_indices[1][:,None], triu_indices[0][:,None]], axis=-1)])

    # remove nan entries
    existing_agents = ~np.isnan(Y).any(axis=(2,3))

    crossing_agent_ids = []
    crossing_class = []

    for i in range(n_batches):
        mask_0 = existing_agents[i, agent_ids_gen[:,0]]
        mask_1 = existing_agents[i, agent_ids_gen[:,1]]

        mask = mask_0 & mask_1

        crossing_agent_ids.append(agent_ids_gen[mask])
        crossing_class.append(crossing_class_gen[i, mask])
    

    return crossing_agent_ids, crossing_class

def propagate_last_seen_value(arr1, arr2, init_val):

    last_value = init_val
    for i in range(len(arr2)):
        if arr2[i] != 0:
            last_value = arr1[i]  # Update last seen non-zero value
        else:
            arr1[i] = last_value  # Overwrite with the last seen value

    return arr1


def get_hypothetical_path_crossing(X, Y, T, dt):

    ''' 
    Function to get the hypothetical crossing trajectories between sets of trajectories

    Parameters
    ----------
    X_t0: np.array
        Multidimensional array containing trajectory info at the last observed timestep.
        Shape: [batch_size, n_agents, n_features]

    Y : np.array
        Multidimensional array containing the future trajectories to be used as based for comparison.
        The velocity during the future trajectory is to be modified based on the observed velocity
        at the last observation in the past.
        Shape: [batch_size, n_agents, n_timesteps, n_features]

    T : np.array
        Array containing the types of the agents.
        Shape: [batch_size, n_agents]

    dt: float
        Time step of the recorded data.

    Returns
    -------
    crossing_agent_ids: list of np_arrays
        List of length batch_size containing arrays of the indices of exisitng agent pairs.
    crossing_class: list of np.arrays
        List of length batch_size containing arrays on how and if the trajectories of existing agent pairs cross each other.
    '''

    n_batches, n_agents, n_timesteps, _ = Y.shape

    # T_agents = np.tile(T[:,:,None,None,None], (1, 1, n_agents, n_timesteps, n_timesteps))

    # Vectorize dictionary
    # agent_width_dict_vect = np.vectorize(agent_width_dict.get)
    average_agent_speed_dict_vect = np.vectorize(average_agent_speed_dict.get)

    # Get thresholds depending on the average width of the agents
    # widht_thresholds = agent_width_dict_vect(T_agents)
    speed_thresholds = average_agent_speed_dict_vect(T)

    traj = np.concatenate([X[...,:2], Y[...,:2]], axis=2)

    X_t0 = X[..., -1, :2] - X[..., -2, :2]
    X_t0 = np.concatenate([X_t0, X[..., -1, -1:]], axis=-1)

    # Get the velocity of the agents at the last timestep
    X_t0_vel = X_t0[:,:,:2]/dt

    # Get the heading of the agents at the last timestep
    X_t0_heading = X_t0[:,:,2]

    # Check if agents are moving faster than the average speed
    speed_mask = np.linalg.norm(X_t0_vel, axis=-1) <= speed_thresholds

    # Calculate the cumulative displacements in the future
    Y_displacements = np.insert(np.diff(Y, axis=2),0,0, axis=2)
    Y_distances = np.linalg.norm(Y_displacements, axis=-1)
    # set small displacements to zero as this can otherwise lead to noise in the extrapolation
    Y_distances[Y_distances < 0.05] = 0 
    Y_displacements[Y_distances == 0] = 0

    # Calculate heading at each timestep
    Y_rel = traj[...,-n_timesteps:,:2] - traj[...,-n_timesteps-1:-1,:]
    Y_heading = np.arctan2(Y_rel[:,:,:,1], Y_rel[:,:,:,0])
    # At every timestep where displacement is zero, set the heading to the heading at the previous timestep
    Y_heading = np.array([propagate_last_seen_value(Y_heading[b,a], Y_distances[b,a], init_val=X_t0_heading[b,a]) 
                          for b in range(n_batches) for a in range(n_agents)]).reshape(n_batches, n_agents, n_timesteps)
    # At every timestep where displacement is zero, set the displacement to 1e-3 in direction of heading
    Y_displacements[Y_distances == 0] = np.stack([np.cos(Y_heading), np.sin(Y_heading)], axis=-1)[Y_distances == 0] * 1e-3
    Y_distances[Y_distances == 0] = 1e-3 

    Y_smoothed = Y[:,:,[0],:] + np.cumsum(Y_displacements, axis=2)

    # Y_distances[Y_distances == 0] = 1e-3 
    Y_cumulative_distances = np.cumsum(Y_distances, axis=-1)

    initial_speeds = np.zeros(T.shape)
    initial_speeds[speed_mask] = speed_thresholds[speed_mask]
    initial_speeds[~speed_mask] = np.linalg.norm(X_t0_vel[~speed_mask], axis=-1)

    # Determine the new distances based on constant velocity
    new_distances = Y_distances[:,:,1:].copy()
    speed_mask = new_distances < np.tile(initial_speeds[:,:,None], (1,1,n_timesteps-1)) * dt
    new_distances[speed_mask] = np.tile(initial_speeds[:,:,None], (1,1,n_timesteps-1))[speed_mask] * dt

    new_cumulative_distances = np.insert(np.cumsum(new_distances, axis=-1), 0, 0, axis=-1)

    # Interpolate positions to match the new distances
    Y_hypothetical = np.zeros_like(Y)

    f1 = np.stack([
            interpolate.interp1d(Y_cumulative_distances[b, a], Y_smoothed[b, a, :, 0], fill_value='extrapolate', assume_sorted=False) 
            for b in range(n_batches) 
            for a in range(n_agents)
        ]).reshape(n_batches, n_agents)
    
    f2 = np.stack([
            interpolate.interp1d(Y_cumulative_distances[b, a], Y_smoothed[b, a, :, 1], fill_value='extrapolate', assume_sorted=False) 
            for b in range(n_batches) 
            for a in range(n_agents)
        ]).reshape(n_batches, n_agents)
    
    Y_hypothetical[:,:,:,0] = np.stack([f1[b,a](new_cumulative_distances[b,a]) for b in range(n_batches) for a in range(n_agents)]).reshape(n_batches, n_agents, n_timesteps)

    Y_hypothetical[:,:,:,1] = np.stack([f2[b,a](new_cumulative_distances[b,a]) for b in range(n_batches) for a in range(n_agents)]).reshape(n_batches, n_agents, n_timesteps)
    
    crossing_agent_ids, crossing_class = get_crossing_trajectories(Y_hypothetical, T)

    return crossing_agent_ids, crossing_class


def get_closeness(Y):
    '''
    Function to get the development of the closeness of agents in the future.

    Parameters
    ----------
    Y : np.array
        Multidimensional array containing the trajectories to be compared.
        Shape: [batch_size, n_agents, n_timesteps, n_features]

    Returns
    -------
    closeness_agent_ids: list of np_arrays
        List of length batch_size containing arrays of the indices of exisitng agent pairs.
    closeness_class: list of np.arrays
        List of length batch_size containing arrays on how and if the trajectories of existing agent pairs get closer to each other.
    '''

    n_batches = Y.shape[0]

    # Get maximum number of agents
    n_agents = Y.shape[1]

    # Get number of timesteps
    n_timesteps = Y.shape[2]

    # Expand the dimensions of the trajectories
    trajs_agent1 = np.tile(Y[:,None,:,None], (1,n_agents,1,n_timesteps,1,1))
    trajs_agent2 = np.tile(Y[:,:,None,:,None], (1,1,n_agents,1,n_timesteps,1))

    # Calculate the distance between the agents at each timestep
    dist = np.linalg.norm(trajs_agent1 - trajs_agent2, axis=-1) # [batch_size, n_agents, n_agents, n_timesteps, n_timesteps]

    # Get cases where agent_1 gets closer or even reaches the position of agent_2 at timestep t0 at any point of the trajectory
    d1 = dist[:,:,:,:,0] < np.tile(dist[:,:,:,[0],0], (1,1,1,n_timesteps))
    d1 = d1.any(axis=-1)

    # Get cases where agent_2 gets closer or even reaches the position of agent_1 at timestep t0 at any point of the trajectory
    d2 = dist[:,:,:,0,:] < np.tile(dist[:,:,:,[0],0], (1,1,1,n_timesteps))
    d2 = d2.any(axis=-1)

    # Get cases where agent_1 and agent_2 are closer at the final timestep than at the initial timestep
    d3 = dist[:,:,:,-1,-1] < dist[:,:,:,0,0]

    # Remove the diagonal of the matrix and stack agent pairs
    triu_indices = np.triu_indices(n_agents, k=1)
    d1_pairs_perBatch_triu = d1[:,triu_indices[0],triu_indices[1]]
    d2_pairs_perBatch_triu = d2[:,triu_indices[0],triu_indices[1]]
    d3_pairs_perBatch_triu = d3[:,triu_indices[0],triu_indices[1]]

    tril_indices = np.tril_indices(n_agents, k=-1)
    d1_pairs_perBatch_tril = d1[:,tril_indices[0],tril_indices[1]]
    d2_pairs_perBatch_tril = d2[:,tril_indices[0],tril_indices[1]]
    d3_pairs_perBatch_tril = d3[:,tril_indices[0],tril_indices[1]]

    d1_pairs = np.concatenate((d1_pairs_perBatch_triu, d1_pairs_perBatch_tril), axis=-1)*1
    d2_pairs = np.concatenate((d2_pairs_perBatch_triu, d2_pairs_perBatch_tril), axis=-1)*1
    d3_pairs = np.concatenate((d3_pairs_perBatch_triu, d3_pairs_perBatch_tril), axis=-1)*1


    agent_ids_gen = np.concatenate([triu_indices[0][:,None], triu_indices[1][:,None]], axis=-1)
    agent_ids_gen = np.concatenate([agent_ids_gen, np.concatenate([tril_indices[0][:,None], tril_indices[1][:,None]], axis=-1)])

    closeness_class_gen = np.concatenate((d1_pairs[:,:,None], d2_pairs[:,:,None], d3_pairs[:,:,None]), axis=-1)

    # remove nan entries
    existing_agents = ~np.isnan(Y).any(axis=(2,3))

    closeness_agent_ids = []
    closeness_class = []

    for i in range(n_batches):
        mask_0 = existing_agents[i, agent_ids_gen[:,0]]
        mask_1 = existing_agents[i, agent_ids_gen[:,1]]

        mask = mask_0 & mask_1

        closeness_agent_ids.append(agent_ids_gen[mask])
        closeness_class.append(closeness_class_gen[i, mask])


    return closeness_agent_ids, closeness_class

    







    



