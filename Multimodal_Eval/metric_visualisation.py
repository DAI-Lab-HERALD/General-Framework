#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re

from utils import *

#%% Load Results
JSD_testing = {}
Wasserstein_log_fitting_testing, Wasserstein_log_fitting_sampled, Wasserstein_log_testing_sampled = {}, {}, {}
Wasserstein_data_fitting_testing, Wasserstein_data_fitting_sampled, Wasserstein_data_testing_sampled = {}, {}, {}

fitting_pf_fitting_log_likelihood, testing_pf_testing_log_likelihood = {}, {}
fitting_pf_testing_log_likelihood, fitting_pf_sampled_log_likelihood = {}, {}

fitting_dict, testing_dict, sampled_dict = {}, {}, {}


random_seeds = [
                ['00','10'],
                ['10','20'],
                ['20','30'],
                ['30','40'],
                ['40','50'],
                ['50','60'],
                # ['60','70'],
                ['70','80'],
                ['80','90'],
                ['90','100']
                ]

# loop through all results files and save to corresponding dictionaries
for rndSeed in random_seeds:

    JSD_testing = {**JSD_testing, **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                     '_JSD_testing', 'rb'))}

    Wasserstein_data_fitting_testing = {**Wasserstein_data_fitting_testing,
                                        **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                           '_Wasserstein_data_fitting_testing', 'rb'))}
    Wasserstein_data_fitting_sampled = {**Wasserstein_data_fitting_sampled,
                                        **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                           '_Wasserstein_data_fitting_sampled', 'rb'))}
    Wasserstein_data_testing_sampled = {**Wasserstein_data_testing_sampled,
                                        **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                           '_Wasserstein_data_testing_sampled', 'rb'))}

    Wasserstein_log_fitting_testing = {**Wasserstein_log_fitting_testing,
                                       **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                        '_Wasserstein_log_fitting_testing', 'rb'))}
    Wasserstein_log_fitting_sampled = {**Wasserstein_log_fitting_sampled,
                                       **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                          '_Wasserstein_log_fitting_sampled', 'rb'))}
    Wasserstein_log_testing_sampled = {**Wasserstein_log_testing_sampled,
                                       **pickle.load(open('./Distribution Datasets/Results/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                          '_Wasserstein_log_testing_sampled', 'rb'))}

    fitting_pf_fitting_log_likelihood = {**fitting_pf_fitting_log_likelihood,
                                         **pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                            '_fitting_pf_fitting_log_likelihood', 'rb'))}
    testing_pf_testing_log_likelihood = {**testing_pf_testing_log_likelihood,
                                         **pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                            '_testing_pf_testing_log_likelihood', 'rb'))}
    fitting_pf_testing_log_likelihood = {**fitting_pf_testing_log_likelihood,
                                         **pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                            '_fitting_pf_testing_log_likelihood', 'rb'))}
    fitting_pf_sampled_log_likelihood = {**fitting_pf_sampled_log_likelihood,
                                         **pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                            '_fitting_pf_sampled_log_likelihood', 'rb'))}
    
    fitting_dict = {**fitting_dict,
                    **pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                    '_fitting_dict', 'rb'))}
    testing_dict = {**testing_dict,
                    **pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                    '_testing_dict', 'rb'))}
    sampled_dict = {**sampled_dict,
                    **pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                    '_sampled_dict', 'rb'))}

#%% Plotting
# Create an array of dimensions num_datasets x num_ablations x num_metrics x num_random_seeds
# Each element is a value of the metric for a given dataset, ablation and random seed
# Datasets: noisy_moons, noisy_circles, blobs, varied, aniso, Trajectories

Results = np.ones((6*5, 14, 30, 100)) * np.nan

# list of ablation keys
ablation_keys = ['config_cluster_PCA_stdKDE',
                 'config_cluster_PCAKDE',
                 'config_cluster_stdKDE',
                 'config_DBCV_PCA_stdKDE',
                 'config_DBCV_PCAKDE',
                 'config_DBCV_stdKDE',
                 'config_PCA_stdKDE',
                 'config_PCAKDE',
                 'config_stdKDE',
                 'config_clusterGMM',
                 'config_DBCVGMM',
                 'configGMM',
                 'config_cluster_PCA_stdKNN',
                 'config_cluster_PCAKNN',
                 'config_cluster_stdKNN',
                 'config_DBCV_PCA_stdKNN',
                 'config_DBCV_PCAKNN',
                 'config_DBCV_stdKNN',
                 'config_PCA_stdKNN',
                 'config_PCAKNN',
                 'config_stdKNN']

# list of dataset keys
dataset_keys = ['noisy_moons_n_samples_200',
                'noisy_circles_n_samples_200',
                'blobs_n_samples_200',
                'varied_n_samples_200',
                'aniso_n_samples_200',
                'Trajectories_n_samples_200',
                'noisy_moons_n_samples_600',
                'noisy_circles_n_samples_600',
                'blobs_n_samples_600',
                'varied_n_samples_600',
                'aniso_n_samples_600',
                'Trajectories_n_samples_600',
                'noisy_moons_n_samples_2000',
                'noisy_circles_n_samples_2000',
                'blobs_n_samples_2000',
                'varied_n_samples_2000',
                'aniso_n_samples_2000',
                'Trajectories_n_samples_2000',
                'noisy_moons_n_samples_6000',
                'noisy_circles_n_samples_6000',
                'blobs_n_samples_6000',
                'varied_n_samples_6000',
                'aniso_n_samples_6000',
                'Trajectories_n_samples_6000',
                'noisy_moons_n_samples_20000',
                'noisy_circles_n_samples_20000',
                'blobs_n_samples_20000',
                'varied_n_samples_20000',
                'aniso_n_samples_20000',
                'Trajectories_n_samples_20000']

# Fill the array with the values from the dictionaries
for _, (k, v) in enumerate(JSD_testing.items()):
    rndSeed = int(k[re.search(r"rnd_seed_\d{1,2}", k).start():re.search(r"rnd_seed_\d{1,2}", k).end()][9:])
    dataset_id = np.where(np.array(dataset_keys) == k[:re.search(r"n_samples_\d{1,5}", k).end()])[0][0]
    ablation_id = np.where(np.array(ablation_keys) == k[re.search(r"config\w{1,26}", k).start():])[0][0]
        
    base_data_key = k[:re.search(r"rnd_seed_\d{1,2}", k).end()]

    Results[dataset_id, ablation_id, 0, rndSeed] = JSD_testing[k]
    Results[dataset_id, ablation_id, 1, rndSeed] = Wasserstein_data_fitting_testing[base_data_key]
    try:
        Results[dataset_id, ablation_id, 2, rndSeed] = Wasserstein_data_fitting_sampled[k]
    except:
        Results[dataset_id, ablation_id, 2, rndSeed] = np.nan

    try:
        Results[dataset_id, ablation_id, 3, rndSeed] = Wasserstein_data_testing_sampled[k]
    except:
        Results[dataset_id, ablation_id, 3, rndSeed] = np.nan

    Results[dataset_id, ablation_id, 4, rndSeed] = Wasserstein_log_fitting_testing[k]

    try:
        Results[dataset_id, ablation_id, 5, rndSeed] = Wasserstein_log_fitting_sampled[k]
    except:
        Results[dataset_id, ablation_id, 5, rndSeed] = np.nan

    try:
        Results[dataset_id, ablation_id, 6, rndSeed] = Wasserstein_log_testing_sampled[k]
    except:
        Results[dataset_id, ablation_id, 6, rndSeed] = np.nan

    Results[dataset_id, ablation_id, 7, rndSeed] = np.mean(fitting_pf_fitting_log_likelihood[k])
    Results[dataset_id, ablation_id, 8, rndSeed] = np.mean(testing_pf_testing_log_likelihood[k])
    Results[dataset_id, ablation_id, 9, rndSeed] = np.mean(fitting_pf_testing_log_likelihood[k])

    try:
        Results[dataset_id, ablation_id, 10, rndSeed] = np.mean(fitting_pf_sampled_log_likelihood[k])
    except:
        Results[dataset_id, ablation_id, 10, rndSeed] = np.nan

    Results[dataset_id, ablation_id, 11, rndSeed] = np.std(fitting_pf_fitting_log_likelihood[k])
    Results[dataset_id, ablation_id, 12, rndSeed] = np.std(testing_pf_testing_log_likelihood[k])
    Results[dataset_id, ablation_id, 13, rndSeed] = np.std(fitting_pf_testing_log_likelihood[k])

    try:
        Results[dataset_id, ablation_id, 14, rndSeed] = np.std(fitting_pf_sampled_log_likelihood[k])
    except:
        Results[dataset_id, ablation_id, 14, rndSeed] = np.nan

    Results[dataset_id, ablation_id, 15, rndSeed] = np.quantile(fitting_pf_fitting_log_likelihood[k], [0.1])
    Results[dataset_id, ablation_id, 16, rndSeed] = np.quantile(fitting_pf_fitting_log_likelihood[k], [0.25])
    Results[dataset_id, ablation_id, 17, rndSeed] = np.quantile(fitting_pf_fitting_log_likelihood[k], [0.5])
    Results[dataset_id, ablation_id, 18, rndSeed] = np.quantile(fitting_pf_fitting_log_likelihood[k], [0.75])
    Results[dataset_id, ablation_id, 19, rndSeed] = np.quantile(fitting_pf_fitting_log_likelihood[k], [0.9])
    Results[dataset_id, ablation_id, 20, rndSeed] = np.quantile(fitting_pf_testing_log_likelihood[k], [0.1])
    Results[dataset_id, ablation_id, 21, rndSeed] = np.quantile(fitting_pf_testing_log_likelihood[k], [0.25])
    Results[dataset_id, ablation_id, 22, rndSeed] = np.quantile(fitting_pf_testing_log_likelihood[k], [0.5])
    Results[dataset_id, ablation_id, 23, rndSeed] = np.quantile(fitting_pf_testing_log_likelihood[k], [0.75])
    Results[dataset_id, ablation_id, 24, rndSeed] = np.quantile(fitting_pf_testing_log_likelihood[k], [0.9])

    try:
        Results[dataset_id, ablation_id, 25, rndSeed] = np.quantile(fitting_pf_sampled_log_likelihood[k], [0.1])
        Results[dataset_id, ablation_id, 26, rndSeed] = np.quantile(fitting_pf_sampled_log_likelihood[k], [0.25])
        Results[dataset_id, ablation_id, 27, rndSeed] = np.quantile(fitting_pf_sampled_log_likelihood[k], [0.5])
        Results[dataset_id, ablation_id, 28, rndSeed] = np.quantile(fitting_pf_sampled_log_likelihood[k], [0.75])
        Results[dataset_id, ablation_id, 29, rndSeed] = np.quantile(fitting_pf_sampled_log_likelihood[k], [0.9])
    except:
        Results[dataset_id, ablation_id, 25, rndSeed] = np.nan
        Results[dataset_id, ablation_id, 26, rndSeed] = np.nan
        Results[dataset_id, ablation_id, 27, rndSeed] = np.nan
        Results[dataset_id, ablation_id, 28, rndSeed] = np.nan
        Results[dataset_id, ablation_id, 29, rndSeed] = np.nan


#%% For each metric and each dataset plot the ablation results side by side
# with the mean and quantile values as boxplots
metric_keys = [
                'JSD',
                'WS(X_f, X_t)',
                'WS(X_f, X_s)',
                'WS(X_t, X_s)',
                'WS(log(X_f), log(X_t))',
                'WS(log(X_f), log(X_s))',
                'WS(log(X_t), log(X_s))',
                '\mu_log(p_f(X_f))',
                '\mu_log(p_t(X_t))',
                '\mu_log(p_f(X_t))',
                '\mu_log(p_f(X_s))',
                '\sigma_log(p_f(X_f))',
                '\sigma_log(p_t(X_t))',
                '\sigma_log(p_f(X_t))',
                '\sigma_log(p_f(X_s))',
                'quant0.1_log(p_f(X_f))',
                'quant0.25_log(p_f(X_f))',
                'quant0.5_log(p_f(X_f))',
                'quant0.75_log(p_f(X_f))',
                'quant0.9_log(p_f(X_f))',
                'quant0.1_log(p_t(X_t))',
                'quant0.25_log(p_t(X_t))',
                'quant0.5_log(p_t(X_t))',
                'quant0.75_log(p_t(X_t))',
                'quant0.9_log(p_t(X_t))',
                'quant0.1_log(p_f(X_s))',
                'quant0.25_log(p_f(X_s))',
                'quant0.5_log(p_f(X_s))',
                'quant0.75_log(p_f(X_s))',
                'quant0.9_log(p_f(X_s))',
]

all_indices_topKDE = []
all_indices_bestGMM = []

for i in range(30):
    for j in range(1,4):
        data = Results[i, :, j, :]
        data = list(data)

        for l in range(len(data)):
            data[l] = data[l][~np.isnan(data[l])]
            
        data = np.array(data)

        # Use Wasserstein to establish most promising candidates
        plt.figure()
        plt.title('Metric: '+metric_keys[j]+' Dataset: '+ dataset_keys[i])
        plt.boxplot(data.T, positions = np.arange(14), widths = 0.6)
        # plt.show()
        plt.savefig('./Distribution Datasets/Results/Plots/'+metric_keys[j]+'_'+dataset_keys[i]+'.pdf', bbox_inches='tight')
        plt.close()

        # get indices of top 2 ablations for KDE, best for GMM and top 2 for KNN
        top2kde = np.argsort(np.stack(data[0:6]).mean(axis=1))[:2]
        bestgmm = np.argsort(np.stack(data[6:8]).mean(axis=1))[0] + 6
        # top2knn = np.argsort(np.stack(data[8:14]).mean(axis=1))[:2] + 8

        all_indices_topKDE.append(top2kde)
        all_indices_bestGMM.append(bestgmm)
        # all_indices_topKNN.append(top2knn)

#%% count the number of times each ablation is in the top 2 for KDE, best for GMM and top 2 for KNN
cnt0 = np.concatenate(all_indices_topKDE).tolist().count(0)
cnt1 = np.concatenate(all_indices_topKDE).tolist().count(1)
cnt2 = np.concatenate(all_indices_topKDE).tolist().count(2)
cnt3 = np.concatenate(all_indices_topKDE).tolist().count(3)
cnt4 = np.concatenate(all_indices_topKDE).tolist().count(4)
cnt5 = np.concatenate(all_indices_topKDE).tolist().count(5)
cnt6 = all_indices_bestGMM.count(6)
cnt7 = all_indices_bestGMM.count(7)


top2kde_counts = [cnt0, cnt1, cnt2, cnt3, cnt4, cnt5]
bestgmm_counts = [cnt6, cnt7]

top2kde_indices = [np.argmax(top2kde_counts), 
                   int(np.where(np.array(top2kde_counts)==max([num for num in top2kde_counts if num < max(top2kde_counts)], default=None))[0])]

bestgmm_indices = [np.argmax(bestgmm_counts) + 6]

#%% Use JSD to get KNN candidates
all_indices_topKNN = []

candidate_ablations = np.concatenate([top2kde_indices, bestgmm_indices])
candidate_ablations = np.concatenate([candidate_ablations, np.arange(8,14)])

for i in range(30):
    j= 0 
    data = Results[i, candidate_ablations, j, :]
    data = list(data)

    for l in range(len(data)):
        data[l] = data[l][~np.isnan(data[l])]
        
    data = np.array(data)

    # Use Wasserstein to establish most promising candidates
    plt.figure()
    plt.title('Metric: '+metric_keys[j]+' Dataset: '+ dataset_keys[i])
    plt.boxplot(data.T, positions = np.arange(9), widths = 0.6)
    # plt.show()
    plt.savefig('./Distribution Datasets/Results/Plots/'+metric_keys[j]+'_'+dataset_keys[i]+'.pdf', bbox_inches='tight')
    plt.close()

    # get indices of top 2 ablations for KDE, best for GMM and top 2 for KNN
    top2knn = np.argsort(np.stack(data[3:9]).mean(axis=1))[:2] + 8
    all_indices_topKNN.append(top2knn)

#%%
cnt8 = np.concatenate(all_indices_topKNN).tolist().count(8)
cnt9 = np.concatenate(all_indices_topKNN).tolist().count(9)
cnt10 = np.concatenate(all_indices_topKNN).tolist().count(10)
cnt11 = np.concatenate(all_indices_topKNN).tolist().count(11)
cnt12 = np.concatenate(all_indices_topKNN).tolist().count(12)
cnt13 = np.concatenate(all_indices_topKNN).tolist().count(13)

top2knn_counts = [cnt8, cnt9, cnt10, cnt11, cnt12, cnt13]

top2knn_indices = [np.argmax(top2knn_counts) + 8,
                   int(np.where(np.array(top2knn_counts)==max([num for num in top2knn_counts if num < max(top2knn_counts)], default=None))[0]) + 8]
top2knn_indices = sorted(top2knn_indices)


#%% Plot the rest of the metrics only for the candidate ablations
# candidate_ablations = np.concatenate([top2kde_indices, bestgmm_indices, top2knn_indices])
candidate_ablations = np.concatenate([top2kde_indices, [6, 7], top2knn_indices]) # TODO put back for bestGMM
colors = ['r', 'g', 'b', 'c', 'm', 'y']
axes_list = []

from matplotlib.patches import Patch

#%%
for j in range(4, 30):
    axes_list = []
    plt.figure(0).clear()
    plt.figure(1).clear()
    plt.figure(2).clear()
    plt.figure(3).clear()
    plt.figure(4).clear()
    plt.figure(5).clear()

    legend_elements = []
    for i in range(30):
        data = Results[i, candidate_ablations, j, :]
        data = list(data)

        for l in range(len(data)):
            data[l] = data[l][~np.isnan(data[l])]
            
        data = np.array(data)

        offset = i//6 * 2 
    

        # Plot every 6th dataset in the same figure - number of samples increases from left to right
        plt.figure(i%6)
        # Create an empty axis for the first figure
        if len(axes_list) < 6:
            axes_list.append(plt.gca())
        
        ax = axes_list[i%6]
        plt.title('Metric: '+metric_keys[j] + ' Dataset: '+ dataset_keys[i%6][:-14])

        # Iterate through cases in the dataset
        for abl_id, abl in enumerate(candidate_ablations):
            # Calculate the x-position for the boxplot
            x = offset + 0.2*abl_id%6 #5 TODO put back for bestGMM
            
            # Create a boxplot for the current case with color and symbol
            boxplot = ax.boxplot(data[abl_id], positions=[x], patch_artist=True)
            box = boxplot['boxes'][0]
            box.set(facecolor=colors[abl_id])

            if len(legend_elements) < 6: #5:  TODO put back for bestGMM
                legend_elements.append(Patch(facecolor=colors[abl_id], edgecolor='k', label=f'Case {abl}'))

            
        # make one x tick for each group of ablations
        # ax.set_xticks(np.arange(0.4, 8.8, 2))
        ax.set_xticks(np.arange(0.5, 9, 2)) #TODO put back for bestGMM
        # label the x ticks with the ablation names
        ax.set_xticklabels(['200', '600', '2000', '6000', '20000'])

        # provide a legend for the ablations
        ax.legend(handles=legend_elements, loc='best')


        plt.savefig('./Distribution Datasets/Results/Plots/'+metric_keys[j]+'_'+dataset_keys[i%6][:-14]+'.pdf', bbox_inches='tight')
    plt.close()



# %% Gather results for data visualisation
# list of dataset keys
dataset_keys_twoD = ['noisy_moons_n_samples_200',
                    'noisy_circles_n_samples_200',
                    'blobs_n_samples_200',
                    'varied_n_samples_200',
                    'aniso_n_samples_200',
                    'noisy_moons_n_samples_600',
                    'noisy_circles_n_samples_600',
                    'blobs_n_samples_600',
                    'varied_n_samples_600',
                    'aniso_n_samples_600',
                    'noisy_moons_n_samples_2000',
                    'noisy_circles_n_samples_2000',
                    'blobs_n_samples_2000',
                    'varied_n_samples_2000',
                    'aniso_n_samples_2000',
                    'noisy_moons_n_samples_6000',
                    'noisy_circles_n_samples_6000',
                    'blobs_n_samples_6000',
                    'varied_n_samples_6000',
                    'aniso_n_samples_6000',
                    'noisy_moons_n_samples_20000',
                    'noisy_circles_n_samples_20000',
                    'blobs_n_samples_20000',
                    'varied_n_samples_20000',
                    'aniso_n_samples_20000']

dataset_keys_traj = [
                    'Trajectories_n_samples_200',
                    'Trajectories_n_samples_600',
                    'Trajectories_n_samples_2000',
                    'Trajectories_n_samples_6000',
                    'Trajectories_n_samples_20000'
                    ]

ablation_keys_candidates = np.array(ablation_keys)[candidate_ablations].tolist()

Results_twoD = np.ones((5*5, len(ablation_keys_candidates), 21, 2, 100)) * np.nan
Results_traj = np.ones((1*5, len(ablation_keys_candidates), 21, 24, 100)) * np.nan

# Fill the array with the values from the dictionaries
for _, (k, v) in enumerate(JSD_testing.items()):
    rndSeed = int(k[re.search(r"rnd_seed_\d{1,2}", k).start():re.search(r"rnd_seed_\d{1,2}", k).end()][9:])
    try:
        ablation_id_cand = np.where(np.array(ablation_keys_candidates) == k[re.search(r"config\w{1,26}", k).start():])[0][0]
    except:
        continue

    base_data_key = k[:re.search(r"rnd_seed_\d{1,2}", k).end()]

    if 'Trajectories' in k:
        dataset_id_traj = np.where(np.array(dataset_keys_traj) == k[:re.search(r"n_samples_\d{1,5}", k).end()])[0][0]

        Results_traj[dataset_id_traj, ablation_id_cand, 0, :, rndSeed] = np.mean(fitting_dict[base_data_key], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 1, :, rndSeed] = np.mean(testing_dict[base_data_key], axis=0)

        try:
            Results_traj[dataset_id_traj, ablation_id_cand, 2, :, rndSeed] = np.mean(sampled_dict[k], axis=0)
        except:
            Results_traj[dataset_id_traj, ablation_id_cand, 2, :, rndSeed] = np.nan

        Results_traj[dataset_id_traj, ablation_id_cand, 3, :, rndSeed] = np.std(fitting_dict[base_data_key], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 4, :, rndSeed] = np.std(testing_dict[base_data_key], axis=0)

        try:
            Results_traj[dataset_id_traj, ablation_id_cand, 5, :, rndSeed] = np.std(sampled_dict[k], axis=0)
        except:
            Results_traj[dataset_id_traj, ablation_id_cand, 5, :, rndSeed] = np.nan        

        Results_traj[dataset_id_traj, ablation_id_cand, 6, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.1], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 7, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.25], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 8, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.5], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 9, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.75], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 10, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.9], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 11, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.1], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 12, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.25], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 13, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.5], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 14, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.75], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 15, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.9], axis=0)

        try:
            Results_traj[dataset_id_traj, ablation_id_cand, 16, :, rndSeed] = np.quantile(sampled_dict[k], [0.1], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 17, :, rndSeed] = np.quantile(sampled_dict[k], [0.25], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 18, :, rndSeed] = np.quantile(sampled_dict[k], [0.5], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 19, :, rndSeed] = np.quantile(sampled_dict[k], [0.75], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 20, :, rndSeed] = np.quantile(sampled_dict[k], [0.9], axis=0)
        except:
            Results_traj[dataset_id_traj, ablation_id_cand, 16, :, rndSeed] = np.nan
            Results_traj[dataset_id_traj, ablation_id_cand, 17, :, rndSeed] = np.nan
            Results_traj[dataset_id_traj, ablation_id_cand, 18, :, rndSeed] = np.nan
            Results_traj[dataset_id_traj, ablation_id_cand, 19, :, rndSeed] = np.nan
            Results_traj[dataset_id_traj, ablation_id_cand, 20, :, rndSeed] = np.nan

    else:
        dataset_id_twoD = np.where(np.array(dataset_keys_twoD) == k[:re.search(r"n_samples_\d{1,5}", k).end()])[0][0]

        Results_twoD[dataset_id_twoD, ablation_id_cand, 0, :, rndSeed] = np.mean(fitting_dict[base_data_key], axis=0)
        Results_twoD[dataset_id_twoD, ablation_id_cand, 1, :, rndSeed] = np.mean(testing_dict[base_data_key], axis=0)

        try:
            Results_twoD[dataset_id_twoD, ablation_id_cand, 2, :, rndSeed] = np.mean(sampled_dict[k], axis=0)
        except:
            Results_twoD[dataset_id_twoD, ablation_id_cand, 2, :, rndSeed] = np.nan

        Results_twoD[dataset_id_twoD, ablation_id_cand, 3, :, rndSeed] = np.std(fitting_dict[base_data_key], axis=0)
        Results_twoD[dataset_id_twoD, ablation_id_cand, 4, :, rndSeed] = np.std(testing_dict[base_data_key], axis=0)

        try:
            Results_twoD[dataset_id_twoD, ablation_id_cand, 5, :, rndSeed] = np.std(sampled_dict[k], axis=0)
        except:
            Results_twoD[dataset_id_twoD, ablation_id_cand, 5, :, rndSeed] = np.nan

        Results_traj[dataset_id_traj, ablation_id_cand, 6, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.1], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 7, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.25], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 8, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.5], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 9, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.75], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 10, :, rndSeed] = np.quantile(fitting_dict[base_data_key], [0.9], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 11, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.1], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 12, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.25], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 13, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.5], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 14, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.75], axis=0)
        Results_traj[dataset_id_traj, ablation_id_cand, 15, :, rndSeed] = np.quantile(testing_dict[base_data_key], [0.9], axis=0)

        try:
            Results_traj[dataset_id_traj, ablation_id_cand, 16, :, rndSeed] = np.quantile(sampled_dict[k], [0.1], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 17, :, rndSeed] = np.quantile(sampled_dict[k], [0.25], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 18, :, rndSeed] = np.quantile(sampled_dict[k], [0.5], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 19, :, rndSeed] = np.quantile(sampled_dict[k], [0.75], axis=0)
            Results_traj[dataset_id_traj, ablation_id_cand, 20, :, rndSeed] = np.quantile(sampled_dict[k], [0.9], axis=0)
        except:
            Results_twoD[dataset_id_twoD, ablation_id_cand, 16, :, rndSeed] = np.nan
            Results_twoD[dataset_id_twoD, ablation_id_cand, 17, :, rndSeed] = np.nan
            Results_twoD[dataset_id_twoD, ablation_id_cand, 18, :, rndSeed] = np.nan
            Results_twoD[dataset_id_twoD, ablation_id_cand, 19, :, rndSeed] = np.nan
            Results_twoD[dataset_id_twoD, ablation_id_cand, 20, :, rndSeed] = np.nan



# %% Load the original data

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


# Multivariate Bi-Modal Distribution obtained by augmenting recorded pedestrian trajectories
Trajectories = pickle.load(open('./Distribution Datasets/Forking_Paths/Processed_Data/trajectories_20000samples', 'rb'))
Trajectories = Trajectories.reshape(len(Trajectories), Trajectories.shape[1]*Trajectories.shape[2])

#%% Create multiple datasets with different number of samples 
# and save to dictionaries with keys containing info on dataset_name, n_samples and rand_seed

num_samples = [200, 600, 2000, 6000, 20000]

random_seeds_range = range(0,10)

fitting_dict = {}
testing_dict = {}

for n_samples in num_samples:
    for rnd_seed in random_seeds_range:
        key = 'n_samples_' + str(n_samples) + '_rnd_seed_' + str(rnd_seed)

        fitting_dict['noisy_circles_' + key], testing_dict['noisy_circles_' + key] = create_random_data_splt(noisy_circles, rnd_seed, n_samples)
        fitting_dict['noisy_moons_' + key], testing_dict['noisy_moons_' + key] = create_random_data_splt(noisy_moons, rnd_seed, n_samples)
        fitting_dict['blobs_' + key], testing_dict['blobs_' + key] = create_random_data_splt(blobs, rnd_seed, n_samples)
        fitting_dict['varied_' + key], testing_dict['varied_' + key] = create_random_data_splt(varied, rnd_seed, n_samples)
        fitting_dict['aniso_' + key], testing_dict['aniso_' + key] = create_random_data_splt(aniso, rnd_seed, n_samples)
        fitting_dict['Trajectories_' + key], testing_dict['Trajectories_' + key] = create_random_data_splt(Trajectories, rnd_seed, n_samples)


# %% Create five subplots for each dataset with the number of samples growing left to right
# and plot the mean and quantiles of the original data and the samples data

# 2D-Distributions
n_smpls = ['200', '600', '2000', '6000', '20000']

axes_list = []
plt.figure(0).clear()
plt.figure(0).set_figwidth(25)
plt.figure(1).clear()
plt.figure(1).set_figwidth(25)
plt.figure(2).clear()
plt.figure(2).set_figwidth(25)
plt.figure(3).clear()
plt.figure(3).set_figwidth(25)
plt.figure(4).clear()

legend_elements = []
for i in range(25):
    data = Results_twoD[i, :, :, :, 80:]

    # Plot every 5th dataset in the same figure - number of samples increases from left to right
    fig = plt.figure(i%5)

    # Iterate through cases in the dataset
    for abl_id in range(5):
        # select subplots for each dataset version
        ax = plt.subplot(1, 5, i//5 + 1)

        # draw the original data mean with a black x
        plt.plot(np.mean(data[abl_id, 0, 0, :]), np.mean(data[abl_id, 0, 1, :]), 'kx')
        # draw sampled mean for each ablation with an x of the same color as the boxplot
        plt.plot(np.mean(data[abl_id, 2, 0, :]), np.mean(data[abl_id, 2, 1, :]), colors[abl_id]+'x')

        # draw the original data quantiles 
        plt.plot(np.mean(data[abl_id, 6, 0, :]), np.mean(data[abl_id, 6, 1, :]), 'ko')
        plt.plot(np.mean(data[abl_id, 7, 0, :]), np.mean(data[abl_id, 7, 1, :]), 'k|')
        # plt.plot(np.mean(data[abl_id, 8, 0, :]), np.mean(data[abl_id, 8, 1, :]), 'kv')
        plt.plot(np.mean(data[abl_id, 9, 0, :]), np.mean(data[abl_id, 9, 1, :]), 'k|')
        plt.plot(np.mean(data[abl_id, 10, 0, :]), np.mean(data[abl_id, 10, 1, :]), 'ko')

        # draw sampled quantiles for each ablation with a symbol of the same color as the boxplot
        plt.plot(np.mean(data[abl_id, 16, 0, :]), np.mean(data[abl_id, 16, 1, :]), colors[abl_id]+'o')
        plt.plot(np.mean(data[abl_id, 17, 0, :]), np.mean(data[abl_id, 17, 1, :]), colors[abl_id]+'|')
        # plt.plot(np.mean(data[abl_id, 18, 0, :]), np.mean(data[abl_id, 18, 1, :]), colors[abl_id]+'v')
        plt.plot(np.mean(data[abl_id, 19, 0, :]), np.mean(data[abl_id, 19, 1, :]), colors[abl_id]+'|')
        plt.plot(np.mean(data[abl_id, 20, 0, :]), np.mean(data[abl_id, 20, 1, :]), colors[abl_id]+'o')

        if len(legend_elements) < 5:
            legend_elements.append(Patch(facecolor=colors[abl_id], edgecolor='k', label=f'Case {candidate_ablations[abl_id]}'))

        # set subplot title
        ax.set_title(n_smpls[i//5])

    # set plot title
    fig.suptitle(dataset_keys_twoD[i%5][:-14])

    plt.savefig('./Distribution Datasets/Results/Plots/DataMetrics_'+dataset_keys[i%5][:-14]+'.pdf', bbox_inches='tight')
plt.close()


# %%
# Trajectories
n_smpls = ['200', '600', '2000', '6000', '20000']

axes_list = []
plt.figure(0).clear()
plt.figure(0).set_figwidth(25)

legend_elements = []
for i in range(5):
    data = Results_traj[i, :, :, :, 80:]

    # Plot every 5th dataset in the same figure - number of samples increases from left to right
    fig = plt.figure(0)

    # Iterate through cases in the dataset
    for abl_id in range(5):
        # select subplots for each dataset version
        ax = plt.subplot(1, 5, i + 1)

        data_abl = data[abl_id,:,:,:].reshape(1,data.shape[1],int(data.shape[2]/2),2,data.shape[-1])

        # draw the original data mean with a black x
        plt.scatter(np.mean(data_abl[0, 0, :, 0, :], axis=-1), np.mean(data_abl[0, 0, :, 1, :], axis=-1), c='k', marker= 'x')
        # draw sampled mean for each ablation with an x of the same color as the boxplot
        plt.scatter(np.mean(data_abl[0, 2, :, 0, :], axis=-1), np.mean(data_abl[0, 2, :, 1, :], axis=-1), c=colors[abl_id], marker= 'x')

        # draw the original data quantiles 
        plt.scatter(np.mean(data_abl[0, 6, :, 0, :], axis=-1), np.mean(data_abl[0, 6, :, 1, :], axis=-1), c='k', marker= 'o')
        plt.scatter(np.mean(data_abl[0, 7, :, 0, :], axis=-1), np.mean(data_abl[0, 7, :, 1, :], axis=-1), c='k', marker= '|')
        # plt.scatter(np.mean(data_abl[0, 9, :, 0, :], axis=-1), np.mean(data_abl[0, 8, :, 1, :], axis=-1), c='k', marker= 'v')
        plt.scatter(np.mean(data_abl[0, 9, :, 0, :], axis=-1), np.mean(data_abl[0, 9, :, 1, :], axis=-1), c='k', marker= '|')
        plt.scatter(np.mean(data_abl[0, 10, :, 0, :], axis=-1), np.mean(data_abl[0, 10, :, 1, :], axis=-1), c='k', marker= 'o')

        # draw sampled quantiles for each ablation with a symbol of the same color as the boxplot
        plt.scatter(np.mean(data_abl[0, 16, :, 0, :], axis=-1), np.mean(data_abl[0, 16, :, 1, :], axis=-1), c=colors[abl_id], marker= 'o')
        plt.scatter(np.mean(data_abl[0, 17, :, 0, :], axis=-1), np.mean(data_abl[0, 17, :, 1, :], axis=-1), c=colors[abl_id], marker= '|')
        # plt.scatter(np.mean(data_abl[0, 19, :, 0, :], axis=-1), np.mean(data_abl[0, 18, :, 1, :], axis=-1), c=colors[abl_id], marker= 'v')
        plt.scatter(np.mean(data_abl[0, 19, :, 0, :], axis=-1), np.mean(data_abl[0, 19, :, 1, :], axis=-1), c=colors[abl_id], marker= '|')
        plt.scatter(np.mean(data_abl[0, 20, :, 0, :], axis=-1), np.mean(data_abl[0, 20, :, 1, :], axis=-1), c=colors[abl_id], marker= 'o')

        if len(legend_elements) < 5:
            legend_elements.append(Patch(facecolor=colors[abl_id], edgecolor='k', label=f'Case {candidate_ablations[abl_id]}'))

        # set subplot title
        ax.set_title(n_smpls[i//5])

    # set plot title
    fig.suptitle('Trajectories')

    plt.savefig('./Distribution Datasets/Results/Plots/DataMetrics_Trajectories.pdf', bbox_inches='tight')
plt.close()

# %%
