#%% Import libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Prob_function import OPTICS_GMM
import seaborn as sns
from sklearn.decomposition import PCA


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
Datasets = {'Aniso': aniso}
# Datasets = {'Varied': varied, 'Two Circles': noisy_circles, 'Two Moons': noisy_moons}
# Datasets = {'Two Circles': noisy_circles, 'Two Moons': noisy_moons}

#%% Plot the distributions
n = 3000    
min_std = 0.01
# 2D-Distributions
# Figure with 5 subplots in one line without axes

# fig, axs = plt.subplots(1, len(Datasets), figsize=(len(Datasets) * 3, 3))

for i, name in enumerate(Datasets):
    print('Extracting ' + name)
    # Get data
    data = Datasets[name]
    np.random.shuffle(data)
    data = data[:n]

    # Get clusters
    print('Clustering ' + name)
    save_file = './Distribution Datasets/2D-Distributions/Plots/' + name + '_reachability'
    Optics = OPTICS_GMM().fit(data, plot_reach_file = save_file)
    cluster = Optics.cluster_labels 

    # Get colors
    colors = sns.color_palette("husl", cluster.max() + 1)
    colors.append((0.0, 0.0, 0.0))
    data_colors = [colors[i] for i in cluster]

    ## Plot reachability
    n = 200 

    print('Plotting ' + name)
    plt.clf()
    fig_help = plt.figure(i, figsize=(3*(cluster.max() + 1), 3))
    axs_help = []
    # Get function spaces:
    max_value = 0
    for label in range(cluster.max() + 1):
        cluster_data = data[cluster == label]
        axs_help.append(fig_help.add_subplot(1, cluster.max() + 1, label+1))
        
        mean_data = cluster_data - Optics.means[[label]]
        norm_data = mean_data @ Optics.T_mat[label]
        axs_help[label].scatter(norm_data[:, 0], norm_data[:, 1], s=1)
        max_value = max(max_value, np.max(np.abs(norm_data)))

    axs_id_largestLim = 0
    for i in range(len(axs_help)):
        if ((axs_help[i].get_xlim()[1] - axs_help[i].get_xlim()[0]) > 
            (axs_help[axs_id_largestLim].get_xlim()[1] - axs_help[axs_id_largestLim].get_xlim()[0])):

            axs_id_largestLim = i

    axs_lim = axs_help[axs_id_largestLim]

    fig = plt.figure(i, figsize=(3*(cluster.max() + 1), 3))
    axs = []

    for label in range(cluster.max() + 1):
        grid = np.linspace(-max_value, max_value, n)
        x, y = np.meshgrid(grid, grid)
        p = np.stack([x, y], axis=2).reshape(-1, 2)

        cluster_log_prob = Optics.Models[label].score_samples(p)
        cluster_prob = np.exp(cluster_log_prob.reshape(n, n))

        levels = np.linspace(0.005,0.2,5)
        Colors = []
        for level in levels:
            frac = level / cluster_prob.max()
            frac = 0.5 * frac + 0.5
            color = frac * np.array(colors[label]) + (1 - frac) * np.array([1, 1, 1])
            Colors.append(color)

        axs.append(fig.add_subplot(1, cluster.max() + 1, label+1))
        axs[label].contour(x, y, cluster_prob, levels, colors = Colors, linewidths = 2)
        axs[label].axis('equal')
        axs[label].set_xticks([])
        axs[label].set_yticks([])
        axs[label].spines['top'].set_visible(False)
        axs[label].spines['right'].set_visible(False)
        axs[label].spines['left'].set_visible(False)
        axs[label].spines['bottom'].set_visible(False)
        plt.setp(axs[label], xlim=axs_lim.get_xlim(), ylim=axs_lim.get_ylim())

    fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '_unimodal_contour.pdf', bbox_inches='tight')

    # Plot combined stuff
    plt.clf()
    fig_help2 = plt.figure(i, figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1], s=1, c=data_colors, alpha=0.9)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    # Get combined heightlines
    axs_help2 = fig_help2.gca()
    max_value = max(np.max(np.abs(axs_help2.get_xlim())), np.max(np.abs(axs_help2.get_ylim())))
    y_lim = axs_help2.get_ylim()
    x_lim = axs_help2.get_xlim()

    Grid = np.linspace(-max_value, max_value, n)
    X, Y = np.meshgrid(Grid, Grid)
    P = np.stack([X, Y], axis=2).reshape(-1, 2)

    Cluster_log_prob = Optics.score_samples(P)
    Cluster_prob = np.exp(Cluster_log_prob.reshape(n, n))

    # Normalisation
    # Plot
    plt.clf()
    Fig = plt.figure(i, figsize=(3, 3))
    Ax = Fig.add_subplot(1, 1, 1)

    levels = np.linspace(0.025 * Cluster_prob.max(),0.9 * Cluster_prob.max(),5)
    Colors = []
    for level in levels:
        frac = level / Cluster_prob.max()
        frac = 0.5 * frac + 0.5
        color = frac * np.array([31, 119, 180]) / 255 + (1 - frac) * np.array([1, 1, 1])
        Colors.append(color)

    Ax.contour(X, Y, Cluster_prob, levels, colors = Colors, linewidths = 1)
    Ax.axis('equal')
    Ax.set_xticks([])
    Ax.set_yticks([])
    Ax.spines['top'].set_visible(False)
    Ax.spines['right'].set_visible(False)
    Ax.spines['left'].set_visible(False)
    Ax.spines['bottom'].set_visible(False)
    plt.setp(Ax, xlim = x_lim, ylim = y_lim)

    Fig.savefig('./Distribution Datasets/2D-Distributions/Plots/' + name + '_mulitmodal_contour.pdf', bbox_inches='tight')

    plt.clf()
