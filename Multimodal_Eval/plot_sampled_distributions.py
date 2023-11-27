#%% Import libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Prob_function import OPTICS_GMM
import seaborn as sns


#%% Load the data

# 2D-Distributions
# Noisy Circles
np.random.seed(0)

# noisy_circles = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_circles_20000samples', 'rb'))

# Noisy Moons
noisy_moons = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_20000samples', 'rb'))

# Blobs
# blobs = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/blobs_20000samples', 'rb'))

# Varied
varied = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_20000samples', 'rb'))

# Anisotropic
aniso = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_20000samples', 'rb'))

# Datasets = {'Blobs': blobs, 'Aniso': aniso, 'Varied': varied, 'Two Moons': noisy_moons, 'Two Circles': noisy_circles}
Datasets = {'Aniso': aniso, 'Varied': varied, 'Two Moons': noisy_moons}

#%%
fitting_dict = {}
sampled_dict = {}


random_seeds = [
                # ['0','10'],
                # ['10','20'],
                # ['20','30'],
                # ['30','40'],
                # ['40','50'],
                ['50','60'],
                # ['60','70'],
                # ['70','80'],
                # # ['80','90'],
                # ['90','100']
                ]

# loop through all results files and save to corresponding dictionaries
for rndSeed in random_seeds:
    print(rndSeed)
    sampled_dict = {**sampled_dict,
                    **pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                    '_sampled_dict', 'rb'))}
    
    fitting_dict = {**fitting_dict,
                    **pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                    '_fitting_dict', 'rb'))}
   
config_keys = ['config_cluster_PCA_stdKDE',
               'config_cluster_PCAKDE',
               'config_PCA_stdKDE',
               'config_cluster_stdKDE',
               'config_clusterGMM',
               'KDevine',
               'MPS_Windows'
               ]

#%%

for i in range(100):
    print(i)
    varied_MPW_samples = []
    varied_VC_samples = []
    varied_ROME_samples = []
    trajectories_MPW_samples = []
    trajectories_VC_samples = []
    trajectories_ROME_samples = []
    trajectories_GMM_samples = []
    twoMoons_GMM_samples = []
    twoMoons_ROME_samples = []
    twoMoons_KDE_cluster_samples = []
    varied_KDE_PCA_std_samples = []
    aniso_ROME_samples = []
    aniso_KDE_cluster_std_samples = []
    aniso_KDE_cluster_samples = []
    aniso_KDE_PCA_std_samples = []
    aniso_MPW_samples = []
    varied_KDE_samples = []
    aniso_KDE_samples = []
    twoMoons_KDE_samples = []


    varied_fittigData = []
    trajectories_fittingData = []
    twoMoons_fittingData = []
    aniso_fittingData = []


    try:
        np.random.shuffle(sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_MPK_Windows'])
        data = sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_MPK_Windows']
        varied_MPW_samples.append(data)
    except:
        print('try 1')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
    try:
        # if not sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_KDevine'] == 'Failed':
        np.random.shuffle(sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_KDevine'])
        data = sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_KDevine']
        varied_VC_samples.append(data)
    except:
        print('try 2')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
    try:
        np.random.shuffle(sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE'])
        data = sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE']
        varied_ROME_samples.append(data)
    except:
        print('try 3')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
    try:
        np.random.shuffle(sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_MPK_Windows'])
        data = sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_MPK_Windows']
        trajectories_MPW_samples.append(data)
    except:
        print('try 4')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        # if not sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_KDevine'] == 'Failed':
        np.random.shuffle(sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_KDevine'])
        data = sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_KDevine']
        trajectories_VC_samples.append(data)
    except:
        print('try 5')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE'])
        data = sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE']
        trajectories_ROME_samples.append(data)
    except:
        print('try 6')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_config_clusterGMM'])
        data = sampled_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)+'_config_clusterGMM']
        trajectories_GMM_samples.append(data)
    except:
        print('try 8')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_clusterGMM'])
        data = sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_clusterGMM']
        twoMoons_GMM_samples.append(data)
    except:
        print('try 9')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE'])
        data = sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE']
        twoMoons_ROME_samples.append(data)
    except:
        print('try 10')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCAKDE'])
        data = sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCAKDE']
        twoMoons_KDE_cluster_samples.append(data)
    except:
        print('try 11')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_config_PCA_stdKDE'])
        data = sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_config_PCA_stdKDE']
        varied_KDE_PCA_std_samples.append(data)
    except:
        print('try 12')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE'])
        data = sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCA_stdKDE']
        aniso_ROME_samples.append(data)
    except:
        print('try 13')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_stdKDE'])
        data = sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_stdKDE']
        aniso_KDE_cluster_std_samples.append(data)
    except:
        print('try 14')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCAKDE'])
        data = sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_cluster_PCAKDE']
        aniso_KDE_cluster_samples.append(data)
    except:
        print('try 15')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_MPK_Windows'])
        data = sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_MPK_Windows']
        aniso_MPW_samples.append(data)
    except:
        print('try 16')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_config_PCAKDE'])
        data = sampled_dict['varied_n_samples_6000_rnd_seed_'+str(i)+'_config_PCAKDE']
        varied_KDE_samples.append(data)
    except:
        print('try 17')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_PCAKDE'])
        data = sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_PCAKDE']
        aniso_KDE_samples.append(data)
    except:
        print('try 18')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue
        
    try:
        np.random.shuffle(sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_PCAKDE'])
        data = sampled_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)+'_config_PCAKDE']
        twoMoons_KDE_samples.append(data)
    except:
        print('try 19')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue

    try:
        np.random.shuffle(sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_PCA_stdKDE'])
        data = sampled_dict['aniso_n_samples_6000_rnd_seed_'+str(i)+'_config_PCA_stdKDE']
        aniso_KDE_PCA_std_samples.append(data)
    except:
        print('try 16')
        print('error occured in retrieving samples for rnd_seed_'+str(i))
        continue

    varied_fittigData.append(fitting_dict['varied_n_samples_6000_rnd_seed_'+str(i)])
    trajectories_fittingData.append(fitting_dict['Trajectories_n_samples_6000_rnd_seed_'+str(i)])
    twoMoons_fittingData.append(fitting_dict['noisy_moons_n_samples_6000_rnd_seed_'+str(i)])
    aniso_fittingData.append(fitting_dict['aniso_n_samples_6000_rnd_seed_'+str(i)])
    
    break
        
#%%
varied_MPW_samples = varied_MPW_samples[0]
varied_VC_samples = varied_VC_samples[0]
varied_ROME_samples = varied_ROME_samples[0]
varied_KDE_PCA_std_samples = varied_KDE_PCA_std_samples[0]
varied_KDE_samples = varied_KDE_samples[0]

twoMoons_GMM_samples = twoMoons_GMM_samples[0]
twoMoons_ROME_samples = twoMoons_ROME_samples[0]
twoMoons_KDE_cluster_samples = twoMoons_KDE_cluster_samples[0]
twoMoons_KDE_samples = twoMoons_KDE_samples[0]

aniso_ROME_samples = aniso_ROME_samples[0]
aniso_KDE_cluster_std_samples = aniso_KDE_cluster_std_samples[0]
aniso_KDE_cluster_samples = aniso_KDE_cluster_samples[0]
aniso_KDE_PCA_std_samples = aniso_KDE_PCA_std_samples[0]
aniso_MPW_samples = aniso_MPW_samples[0]
aniso_KDE_samples = aniso_KDE_samples[0]

trajectories_MPW_samples = trajectories_MPW_samples[0]
trajectories_VC_samples = trajectories_VC_samples[0]
trajectories_ROME_samples = trajectories_ROME_samples[0]
trajectories_GMM_samples = trajectories_GMM_samples[0]

trajectories_MPW_samples = trajectories_MPW_samples.reshape(-1, 12, 2)
trajectories_VC_samples = trajectories_VC_samples.reshape(-1, 12, 2)
trajectories_ROME_samples = trajectories_ROME_samples.reshape(-1, 12, 2)
trajectories_GMM_samples = trajectories_GMM_samples.reshape(-1, 12, 2)

varied_fittigData = varied_fittigData[0]
trajectories_fittingData = trajectories_fittingData[0].reshape(-1, 12, 2)
twoMoons_fittingData = twoMoons_fittingData[0]
aniso_fittingData = aniso_fittingData[0]

#%%

# Plot varied_MPW_samples
name = 'Varied'
print('Extracting ' + name)
# Get data
data = varied_fittigData

# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# # Get colors
colors = sns.color_palette("husl", 3)

# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(varied_MPW_samples[:, 0], varied_MPW_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/varied_MPW_samples.svg', bbox_inches='tight')
plt.close()

# Plot varied_VC_samples

# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors

# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(varied_VC_samples[:, 0], varied_VC_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/varied_VC_samples.svg', bbox_inches='tight')
plt.close()

# Plot varied_ROME_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(varied_ROME_samples[:, 0], varied_ROME_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/varied_ROME_samples.svg', bbox_inches='tight')
plt.close()

# Plot varied_KDE_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(varied_KDE_samples[:, 0], varied_KDE_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/varied_KDE_samples.svg', bbox_inches='tight')
plt.close()


# Plot varied_KDE_PCA_std_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors
# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=1, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.9, c=colors[2])
plt.scatter(varied_KDE_PCA_std_samples[:, 0], varied_KDE_PCA_std_samples[:, 1], s=1, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/varied_KDE_PCA_std_samples.svg', bbox_inches='tight')


# Plot twoMoons_GMM_samples
data = twoMoons_fittingData

# Get clusters
name = 'Two Moons'
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(twoMoons_GMM_samples[:, 0], twoMoons_GMM_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/twoMoons_GMM_samples.svg', bbox_inches='tight')
plt.close()

# Plot twoMoons_ROME_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(twoMoons_ROME_samples[:, 0], twoMoons_ROME_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/twoMoons_ROME_samples.svg', bbox_inches='tight')
plt.close()

# Plot twoMoons_KDE_cluster_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(twoMoons_KDE_cluster_samples[:, 0], twoMoons_KDE_cluster_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/twoMoons_KDE_cluster_samples.svg', bbox_inches='tight')
plt.close()

# Plot twoMoons_KDE_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(twoMoons_KDE_samples[:, 0], twoMoons_KDE_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/twoMoons_KDE_samples.svg', bbox_inches='tight')
plt.close()

# Plot aniso_ROME_samples
data = aniso_fittingData

# Get clusters
name = 'Aniso'
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(aniso_ROME_samples[:, 0], aniso_ROME_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/aniso_ROME_samples.svg', bbox_inches='tight')
plt.close()

# Plot aniso_KDE_cluster_std_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(aniso_KDE_cluster_std_samples[:, 0], aniso_KDE_cluster_std_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/aniso_KDE_cluster_std_samples.svg', bbox_inches='tight')
plt.close()

# Plot aniso_KDE_cluster_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(aniso_KDE_cluster_samples[:, 0], aniso_KDE_cluster_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/aniso_KDE_cluster_samples.svg', bbox_inches='tight')
plt.close()

# Plot aniso_MPW_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors

# Plot aniso_KDE_PCA_std_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=1, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=1, alpha=0.9, c=colors[2])
plt.scatter(aniso_KDE_PCA_std_samples[:, 0], aniso_KDE_PCA_std_samples[:, 1], s=1, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/aniso_KDE_PCA_std_samples.svg', bbox_inches='tight')



# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(aniso_MPW_samples[:, 0], aniso_MPW_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/aniso_MPW_samples.svg', bbox_inches='tight')
plt.close()

# Plot aniso_KDE_samples
# Get clusters
print('Clustering ' + name)
Optics = OPTICS_GMM().fit(data)
cluster = Optics.cluster_labels 

# Get colors


# Plot
print('Plotting ' + name)
fig = plt.figure(i, figsize=(3, 3))
# plt.scatter(data[:, 0], data[:, 1], s=0.5, c=data_colors, alpha=0.9)
plt.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.9, c=colors[2])
plt.scatter(aniso_KDE_samples[:, 0], aniso_KDE_samples[:, 1], s=0.5, alpha=0.9, c=colors[0])
# plt.set_title(name)
plt.axis('equal')
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/aniso_KDE_samples.svg', bbox_inches='tight')
plt.close()

# %%
# Trajectories MPW

# Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
print('Extracting Trajectories')
Trajectories = trajectories_fittingData
n = len(Trajectories)
# Figure with 1 subplot

fig = plt.figure() 
for i in range(n):
    plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.2, c=colors[2])
for i in range(n):
    plt.plot(trajectories_MPW_samples[i,:,0], trajectories_MPW_samples[i,:, 1], alpha=0.05, c=colors[0])

# set axis equal
plt.axis('equal')

plt.xlim(3, 8)
plt.ylim(4, 8.5)
plt.gca().set_adjustable("box")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

plt.show()

# Remove all spines
fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories_MPW_samples.svg', bbox_inches='tight')
plt.close()

# Trajectories VC

fig = plt.figure() 
for i in range(n):
    plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.2, c=colors[2])
for i in range(n):
    plt.plot(trajectories_VC_samples[i,:,0], trajectories_VC_samples[i,:, 1], alpha=0.05, c=colors[0])
    
# set axis equal
plt.axis('equal')

plt.xlim(3, 8)
plt.ylim(4, 8.5)
plt.gca().set_adjustable("box")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

plt.show()

# Remove all spines
fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories_VC_samples.svg', bbox_inches='tight')
plt.close()

# Trajectories ROME

fig = plt.figure() 
for i in range(n):
    plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.2, c=colors[2])
for i in range(n):
    plt.plot(trajectories_ROME_samples[i,:,0], trajectories_ROME_samples[i,:, 1], alpha=0.05, c=colors[0])
    
# set axis equal
plt.axis('equal')

plt.xlim(3, 8)
plt.ylim(4, 8.5)
plt.gca().set_adjustable("box")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

plt.show()

# Remove all spines
fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories_ROME_samples.svg', bbox_inches='tight')
plt.close()

# Trajectories GMM

fig = plt.figure() 
for i in range(n):
    plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], alpha=0.2, c=colors[2])
for i in range(n):
    plt.plot(trajectories_GMM_samples[i,:,0], trajectories_GMM_samples[i,:, 1], alpha=0.05, c=colors[0])
    
# set axis equal
plt.axis('equal')

plt.xlim(3, 8)
plt.ylim(4, 8.5)
plt.gca().set_adjustable("box")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])

plt.show()

# Remove all spines
fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories_GMM_samples.svg', bbox_inches='tight')
plt.close()
# %%
