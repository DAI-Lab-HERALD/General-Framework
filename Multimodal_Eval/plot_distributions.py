#%% Import libraries
import pickle
import matplotlib.pyplot as plt


#%% Load the data

# 2D-Distributions
# Noisy Circles
noisy_circles = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_circles_20000samples', 'rb'))

# Noisy Moons
noisy_moons = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/noisy_moons_20000samples', 'rb'))

# Blobs
blobs = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/blobs_20000samples', 'rb'))

# Varied
varied = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/varied_20000samples', 'rb'))

# Anisotropic
aniso = pickle.load(open('./Distribution Datasets/2D-Distributions/Processed_Data/aniso_20000samples', 'rb'))


# Toy Problem: Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
ToyProblem_future = pickle.load(open('./Distribution Datasets/CyberZoo Data/Processed_Data/future_trajectories_20000samples', 'rb'))


#%% Plot the distributions

# 2D-Distributions
# Figure with 5 subplots in one line without axes
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

# plot order: blobs, aniso, varied, noisy_moons, noisy_circles
axs[0].scatter(blobs[:5000, 0], blobs[:5000, 1], s=1, alpha=0.05)
axs[1].scatter(aniso[:5000, 0], aniso[:5000, 1], s=1, alpha=0.05)
axs[2].scatter(varied[:5000, 0], varied[:5000, 1], s=1, alpha=0.05)
axs[3].scatter(noisy_moons[:5000, 0], noisy_moons[:5000, 1], s=1, alpha=0.05)
axs[4].scatter(noisy_circles[:5000, 0], noisy_circles[:5000, 1], s=1, alpha=0.05)


axs[0].set_title('Blobs')
axs[1].set_title('Aniso')
axs[2].set_title('Varied')
axs[3].set_title('Two Moons')
axs[4].set_title('Two Circles')

# set axis equal
for ax in axs:
    ax.axis('equal')

# remove the x and y ticks
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# Save figure as pdf
fig.savefig('./Distribution Datasets/2D-Distributions/Plots/2D-Distributions.pdf', bbox_inches='tight')

# %%
# Toy Problem
# Figure with 1 subplot

plt.figure()#figsize=(5, 3))
plt.plot(ToyProblem_future[:5000,:, 0].T, ToyProblem_future[:5000,:, 1].T, '#1f77b4', alpha=0.05)
plt.title('Bi-Modal Trajectories')


# set axis equal
plt.axis('equal')

# provide labels
plt.xlabel('$x\; [m]$')
plt.ylabel('$y\; [m]$')


# set y limits
plt.ylim(-2.5, 2.5)
plt.xlim(0, 6.8)


# Remove all spines

plt.savefig('./Distribution Datasets/CyberZoo Data/Plots/ToyProblem.pdf', bbox_inches='tight')

plt.show()

# Save plot as pdf


# %%
