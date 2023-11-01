#%%
import numpy as np
import pandas as pd
import os
import scipy.stats
from scipy import interpolate as interp
import pickle

#%%
# Load raw data
Data = pd.read_pickle('./Distribution Datasets/Forking_Paths/Forking_Paths_complete/FP_comp_processed.pkl')

# analize raw dara 
num_tars = len(Data)
num_samples = 0 
Path = []
T = []
Domain_old = []

Path_init = []
T_init = []
Domain_init = []

# extract raw samples
max_number_other = 0
for i in range(num_tars):
    data_i = Data.iloc[i]
        
    scene, moment, tar_id, dest, subj, _ = data_i.scenario.split('_')

    domain_scene = "_".join([scene, moment, tar_id])

    if domain_scene != '0000_8_335':
        continue

    if int(data_i.Id) != int(tar_id):
        continue 
    # Get other agents
    other_agents_bool = ((Data.index != data_i.name) & 
                            (Data['First frame'] <= data_i['Last frame']) & 
                            (Data['Last frame'] >= data_i['First frame']) &
                            (Data.scenario == data_i.scenario))
    
    
    other_agents = Data.loc[other_agents_bool]
    
    Data.iloc[i].path['CN'] = np.empty((len(data_i.path), 0)).tolist()

    for j, frame in enumerate(data_i.path.index):
        useful = (other_agents['Last frame'] >= frame) & (other_agents['First frame'] <= frame)
        Data.iloc[i].path.CN.iloc[j] = list(other_agents.index[useful])
        max_number_other = max(max_number_other, useful.sum())
        
    
    # find crossing point
    path = pd.Series(np.zeros(0, np.ndarray), index = [])
    
    track_all = data_i.path.copy(deep = True)
    path['tar'] = np.stack([track_all.x.to_numpy(), track_all.y.to_numpy()], axis = -1)
    
    t = track_all.t.to_numpy()
    
    domain = pd.Series(np.zeros(3, object), index = ['scene', 'scene_full', 'neighbors'])
    domain.scene = domain_scene
    domain.scene_full = data_i.scenario
    track_all = track_all.set_index('t')
    domain.neighbors = track_all.CN
    
    Path_init.append(path)
    T_init.append(t)
    Domain_init.append(domain)

Path_init = pd.DataFrame(Path_init)
T_init = np.array(T_init+[()], np.ndarray)[:-1]
Domain_init = pd.DataFrame(Domain_init)


for i in range(len(Path_init)):
    path_init   = Path_init.iloc[i].tar
    t_init      = T_init[i]
    domain_init = Domain_init.iloc[i]

    other_samples_bool = (Domain_init.scene == domain_init.scene).to_numpy()
    
    if other_samples_bool.sum() < 2:
        continue
    
    num_T = path_init.shape[0]
    Paths_other = np.zeros((other_samples_bool.sum(), num_T, 2), np.float32)
    Paths_other_df = Path_init.iloc[other_samples_bool]
    for j in range(other_samples_bool.sum()):
        path_other = Paths_other_df.iloc[j]
        num_T_other = path_other.tar.shape[0]
        num_t = min(num_T, num_T_other)
        Paths_other[j, :num_t] = path_other.tar[:num_t]

    Dist = np.abs(Paths_other - path_init[np.newaxis])
    Dist = np.nanmax(Dist, axis = (0,2))
    ind_split = max(0, np.argmax(Dist > 1e-3) - 1)

    I_t = t_init[ind_split+1:]
    
    Neighbor = domain_init.neighbors.copy()
    N_U = (Neighbor.index >= I_t[0])
    N_ID = np.unique(np.concatenate(Neighbor.iloc[N_U].to_numpy())).astype(int)
    
    Pos = np.zeros((len(N_ID), len(I_t), 2))
    for j, nid in enumerate(N_ID):
        data_id = Data[(Data.index == nid) & (Data.scenario == domain_init.scene_full)].iloc[0]

        t = data_id.path.t.to_numpy()
        pos = np.stack([data_id.path.x.to_numpy(), data_id.path.y.to_numpy()], axis = -1)
            
        if len(t) > 1:
            for dim in range(2):
                Pos[j, :, dim] = interp.interp1d(np.array(t), pos[:,dim], 
                                                fill_value = 'extrapolate', assume_sorted = True)(I_t)
                
        else:
            Pos[j, :, :] = pos.repeat(len(I_t), axis = 0)[np.newaxis]
    
    # Prepare parameters for sampling
    s_std = 0.05
    
    s_std_ang = np.pi/144
    
    num_samples_test = 1000
    num_samples = 3334
    
    # Prepare samples trajectories
    # Traj_new = np.zeros((0, *path_init.shape), float)
    Traj_new = path_init[np.newaxis]

    P2s = Pos[np.newaxis, :, :-1]
    P2e = Pos[np.newaxis, :, 1:]

    P1s = Traj_new[:, np.newaxis, ind_split+1:-1] 
    P1e = Traj_new[:, np.newaxis, ind_split+2:] 
    
    # Get dp
    dP1 = P1e - P1s
    dP2 = P2e - P2s
    
    # Get the factors 
    A = P1s - P2s
    B = dP1 - dP2
    
    # The distance d(t) can be calculated in the form:
    # d(t) = ||A + t * B|| = a * t ^ 2 + b * t + c
    a = (B ** 2).sum(-1)
    b = 2 * (A * B).sum(-1)
    c = (A ** 2).sum(-1)
    
    # We know that a >= 0, so we can calculate t_min with:
    # d'(t_min) = 2 * a * t_min + b = 0
    t_min = - b / 2 * (a + 1e-6)
    
    # Constrict t_min to interval between 0 and 1,
    # As we only compare lines between the points
    t_min = np.clip(t_min, 0.0, 1.0)
    
    # Calculate d(t_min)
    D_min = a * t_min ** 2 + b * t_min + c
    
    # Get D_min over all other agents and timesteps
    d_min = D_min.min(axis = (1, 2))
    
    if d_min.min() < 0.5:
        d_collision = d_min.min() - 0.01
    else:
        d_collision = 0.5

    

    col_cnt = 0
    while len(Traj_new) < num_samples:
        # Create new trajectories
        Traj_test = np.tile(path_init[np.newaxis], (num_samples_test, 1, 1))
        
        # Sample random factors
        s_mean = 1.0        

        Factors = np.random.normal(s_mean, s_std, num_samples_test)   

        Angles = np.random.normal(0.0, s_std_ang, num_samples_test)

        # Prepare for vectorized operations
        Angles = Angles[:,np.newaxis]
        Factors = Factors[:,np.newaxis,np.newaxis]
        
        # Rotate and stretch trajectories
        c, s = np.cos(Angles), np.sin(Angles)
        Traj_centered = (Traj_test[:,ind_split + 1:] - Traj_test[:,[ind_split]])
        x_vals, y_vals = Traj_centered[...,0], Traj_centered[...,1]
        new_x_vals = c * x_vals - s * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        Traj_centered[...,0] = new_x_vals
        Traj_centered[...,1] = new_y_vals
        
        Traj_test[:, ind_split + 1:] = Traj_centered * Factors + Traj_test[:, [ind_split]]
        

        noise = np.random.normal(0, 0.03/np.sqrt(12), Traj_test[:, ind_split + 1:].shape)
        noise = np.cumsum(noise, axis = 1)

        Traj_test[:, ind_split + 1:] = Traj_test[:, ind_split + 1:] + noise
        # Get starting and end points
        P1s = Traj_test[:, np.newaxis, ind_split+1:-1] 
        P1e = Traj_test[:, np.newaxis, ind_split+2:] 
        
        # Get dp
        dP1 = P1e - P1s
        dP2 = P2e - P2s
        
        # Get the factors 
        A = P1s - P2s
        B = dP1 - dP2
        
        # The distance d(t) can be calculated in the form:
        # d(t) = ||A + t * B|| = a * t ^ 2 + b * t + c
        a = (B ** 2).sum(-1)
        b = 2 * (A * B).sum(-1)
        c = (A ** 2).sum(-1)
        
        # We know that a >= 0, so we can calculate t_min with:
        # d'(t_min) = 2 * a * t_min + b = 0
        t_min = - b / 2 * (a + 1e-6)
        
        # Constrict t_min to interval between 0 and 1,
        # As we only compare lines between the points
        t_min = np.clip(t_min, 0.0, 1.0)
        
        # Calculate d(t_min)
        D_min = a * t_min ** 2 + b * t_min + c
        
        # Get D_min over all other agents and timesteps
        d_min = D_min.min(axis = (1, 2))
        
        # Check if there are collisions
        no_collision = d_min > d_collision
        
        if ~no_collision.all():
            col_cnt += 1
            print(col_cnt)
        # Get collision free trajectories
        Traj_good = Traj_test[no_collision]
        
        # Add good collisions to collection
        Traj_new = np.concatenate((Traj_new, Traj_good), axis = 0)
    
    # Only get required number of samples
    Traj_new = Traj_new[:num_samples]
    
    # Save new trajectories
    for traj in Traj_new:
        path = pd.Series(np.zeros(0, np.ndarray), index = [])
        
        path['tar'] = traj
        path['tar'] = traj[ind_split+1:ind_split+145:12]


        if len(path.tar) < 12:
            continue
        
        Path.append(path)
        num_samples = num_samples + 1

Path = pd.DataFrame(Path)
    
    
# %%
trajectories = np.stack(Path.to_numpy().tolist()).squeeze()[:20000]

# %%
pickle.dump(trajectories, open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectories_20000samples', 'wb'))