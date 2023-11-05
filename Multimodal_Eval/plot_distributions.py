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

noisy_circles = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_circles_20000samples', 'rb'))

# Noisy Moons
noisy_moons = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_20000samples', 'rb'))

# Blobs
blobs = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/blobs_20000samples', 'rb'))

# Varied
varied = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_20000samples', 'rb'))

# Anisotropic
aniso = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_20000samples', 'rb'))

Datasets = {'Blobs': blobs, 'Aniso': aniso, 'Varied': varied, 'Two Moons': noisy_moons, 'Two Circles': noisy_circles}
# Datasets = {'Varied': varied, 'Two Circles': noisy_circles, 'Two Moons': noisy_moons}
# Datasets = {'Two Circles': noisy_circles, 'Two Moons': noisy_moons}

#%% Plot the distributions
n = 3000    
# 2D-Distributions
# Figure with 5 subplots in one line without axes

fig, axs = plt.subplots(1, len(Datasets), figsize=(len(Datasets) * 3, 3))

for i, name in enumerate(Datasets):
    print('Extracting ' + name)
    # Get data
    data = Datasets[name]
    np.random.shuffle(data)
    data = data[:n]

    # Get clusters
    print('Clustering ' + name)
    Optics = OPTICS_GMM().fit(data)
    cluster = Optics.cluster_labels 

    # Get colors
    colors = sns.color_palette("husl", cluster.max() + 1)
    colors.append((0.0, 0.0, 0.0))
    data_colors = [colors[i] for i in cluster]

    # Plot
    print('Plotting ' + name)
    axs[i].scatter(data[:, 0], data[:, 1], s=1, c=data_colors, alpha=0.9)
    axs[i].set_title(name)
    axs[i].axis('equal')
    axs[i].set_xticks([])
    axs[i].set_yticks([])

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/2D-Distributions.pdf', bbox_inches='tight')

# %%
# Trajectories

# Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
print('Extracting Trajectories')
Trajectories = pickle.load(open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectories_20000samples', 'rb'))
# shuffle the trajectories
np.random.shuffle(Trajectories)
Trajectories = Trajectories[:n]
# Figure with 1 subplot

print('Clustering Trajectories')
Optics = OPTICS_GMM().fit(Trajectories.copy().reshape(len(Trajectories), -1))
cluster = Optics.cluster_labels 

# Get colors
colors = sns.color_palette("husl", cluster.max() + 1)
colors.append((0.0, 0.0, 0.0))
data_colors = [colors[i] for i in cluster]

plt.figure() #figsize=(5, 3))
for i in range(n):
    plt.plot(Trajectories[i,:,0], Trajectories[i,:, 1], c=data_colors[i], alpha=0.05)
plt.title('Multi-Modal Trajectories')

# set axis equal
plt.axis('equal')

# provide labels
plt.xlabel('$x\; [m]$')
plt.ylabel('$y\; [m]$')

plt.show()

# Remove all spines
fig.savefig('./Distribution Datasets/Forking_Paths/Plots/Trajectories.pdf', bbox_inches='tight')
