#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats

#%%
past_trajectories = []
future_trajectories = []

aug_past_trajectories = []
aug_future_trajectories = []

past_traj_len = 10
future_traj_len = 14

#%%
trajectories = []
for i in range(2):
    traj = pickle.load(open('./Distribution Datasets/CyberZoo Data/cyberzooPedestrian_avoidObstacle_'+str(i+1), 'rb'))
    print(len(traj))
    trajectories.insert(len(trajectories), traj)

trajectories[0] = trajectories[0][0:700]
trajectories[1] = trajectories[1][108:808]
# trajectories[2] = trajectories[2][610:1310]
# trajectories[3] = trajectories[3][630:1330]
# trajectories[4] = trajectories[4][610:1310]
# trajectories[5] = trajectories[5][415:1115]
# trajectories[6] = np.concatenate((trajectories[6][400:], np.tile(trajectories[6][-1], (11,1))))
# trajectories[7] =  np.concatenate((trajectories[7][330:], np.tile(trajectories[7][-1], (89,1))))
# trajectories[8] = trajectories[8][570:1270]
# trajectories[9] = np.concatenate((trajectories[9][570:], np.tile(trajectories[9][-1], (21,1))))
# trajectories[10] = trajectories[10][19:]
# trajectories[11] = np.concatenate((trajectories[11][160:], np.tile(trajectories[11][-1], (131,1))))


#%%
trajectories = np.array(trajectories)


#%%
trajectories = trajectories[:,::25,0:2]

#%%
past_trajectories = np.concatenate((trajectories[:,:past_traj_len,0], trajectories[:,:past_traj_len,1]), axis=1)
future_trajectories = np.concatenate((trajectories[:,past_traj_len:past_traj_len+future_traj_len,0], trajectories[:,past_traj_len:past_traj_len+future_traj_len,1]), axis=1)

# %%
for i in range(0,2):
     # plt.figure()
     plt.title(str(i))
     plt.plot(past_trajectories[0,:past_traj_len], past_trajectories[0,past_traj_len:])
     plt.plot(future_trajectories[i,:future_traj_len], future_trajectories[i,future_traj_len:])

#%%
aug_past_trajectories = []
aug_future_trajectories = []

s_min=0.5
s_max = 1.5
sigma = 0.5#1

noise_scale = 0.01

num_samples_per_mode = 10000

for j in range(num_samples_per_mode):    
    for i in range(len(future_trajectories)):
            scaler = scipy.stats.truncnorm.rvs((s_min-1)/sigma,(s_max-1)/sigma, loc=1, scale=sigma)
            orig_past_pos_x = past_trajectories[0][:past_traj_len]
            orig_past_pos_y = past_trajectories[0][past_traj_len:]

            past_pos_x = orig_past_pos_x - orig_past_pos_x[-1]
            past_pos_y = orig_past_pos_y - orig_past_pos_y[-1]
            
            future_pos_x = future_trajectories[i][:future_traj_len]
            future_pos_y = future_trajectories[i][future_traj_len:]

            future_pos_x = future_pos_x - orig_past_pos_x[-1]
            future_pos_y = future_pos_y - orig_past_pos_y[-1]
            
             
            aug_past_trajectories.insert(len(aug_past_trajectories), 
                                         np.concatenate((past_pos_x[:,np.newaxis], past_pos_y[:,np.newaxis]), axis=1))
            
            
            mean_fut_pos_x = future_pos_x[0]
            mean_fut_pos_y = future_pos_y[0]
            
            shifted_fut_pos_x = future_pos_x - mean_fut_pos_x
            shifted_fut_pos_y = future_pos_y - mean_fut_pos_y
            
            scaled_fut_pos_x = shifted_fut_pos_x*scaler + mean_fut_pos_x
            scaled_fut_pos_y = shifted_fut_pos_y*scaler + mean_fut_pos_y

            scaled_fut_pos_x_noisy = scaled_fut_pos_x + np.random.normal(0, noise_scale, len(scaled_fut_pos_x))
            scaled_fut_pos_y_noisy = scaled_fut_pos_y + np.random.normal(0, noise_scale, len(scaled_fut_pos_y))
            
            aug_future_trajectories.insert(len(aug_future_trajectories), 
                                           np.concatenate((scaled_fut_pos_x_noisy[:,np.newaxis], scaled_fut_pos_y_noisy[:,np.newaxis]), axis=1))



#%%
past_trajectories = np.array(aug_past_trajectories)
future_trajectories = np.array(aug_future_trajectories)


# %%
# pickle.dump(past_trajectories, open('./Distribution Datasets/CyberZoo Data/Processed_Data/past_trajectories_' + str(num_samples_per_mode) + 'samples', 'wb'))
pickle.dump(future_trajectories, open('./Distribution Datasets/CyberZoo Data/Processed_Data/future_trajectories_' + str(num_samples_per_mode*2) + 'samples', 'wb'))

# %%
