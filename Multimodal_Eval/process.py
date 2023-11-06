#%%
import numpy as np
import os
import pickle
import re

import Prob_function as pf

from utils import *

def load_dir(path):
    if not os.path.exists(path):
        path = path[:path.rfind('/')]
        os.makedirs(path, exist_ok = True)
        return {}
    else:
        return pickle.load(open(path, 'rb'))


def overwrite_key(dict, key, overwrite_string):
    # Check if key already exists
    if not (key in dict.keys()):
        return True
    
    # Check if overwrite string is empty
    if len(overwrite_string) == 0:
        return False
    else:
        return key in overwrite_string


def main(random_seeds, overwrite_string = ''):
    #%% Load the datasets
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

    rand_str = '/rndSeed' + str(random_seeds.start) + str(random_seeds.stop)
    data_str = './Distribution Datasets/Fitted_Dists'+rand_str 

    fitting_dict_str = data_str+'_fitting_dict'
    testing_dict_str = data_str+'_testing_dict'

    fitting_dict = load_dir(fitting_dict_str)
    testing_dict = load_dir(testing_dict_str)

    if len(fitting_dict) == 0:
        for n_samples in num_samples:
            for rnd_seed in random_seeds:
                key = 'n_samples_' + str(n_samples) + '_rnd_seed_' + str(rnd_seed)

                fitting_dict['noisy_circles_' + key], testing_dict['noisy_circles_' + key] = create_random_data_splt(noisy_circles, rnd_seed, n_samples)
                fitting_dict['noisy_moons_' + key], testing_dict['noisy_moons_' + key] = create_random_data_splt(noisy_moons, rnd_seed, n_samples)
                fitting_dict['blobs_' + key], testing_dict['blobs_' + key] = create_random_data_splt(blobs, rnd_seed, n_samples)
                fitting_dict['varied_' + key], testing_dict['varied_' + key] = create_random_data_splt(varied, rnd_seed, n_samples)
                fitting_dict['aniso_' + key], testing_dict['aniso_' + key] = create_random_data_splt(aniso, rnd_seed, n_samples)
                fitting_dict['Trajectories_' + key], testing_dict['Trajectories_' + key] = create_random_data_splt(Trajectories, rnd_seed, n_samples)
        
        pickle.dump(fitting_dict, open(fitting_dict_str, 'wb'))
        pickle.dump(testing_dict, open(testing_dict_str, 'wb'))


    #%% Define test cases
    # 2D-Distributions
    twoD_min_std = 0.01
    # Trajectory Distributions
    traj_min_std = 0.1

    # configs format: use_cluster, use_PCA, use_std, use_KDE, min_std
    use_cluster = True
    use_PCA = True
    use_std = True

    testConfigs = [
                    [   'silhouette',     use_PCA,     use_std, 'KDE'],
                    [   'silhouette',     use_PCA, not use_std, 'KDE'],
                    [   'silhouette', not use_PCA,     use_std, 'KDE'],
                    [         'DBCV',     use_PCA,     use_std, 'KDE'],
                    [         'DBCV',     use_PCA, not use_std, 'KDE'],
                    [         'DBCV', not use_PCA,     use_std, 'KDE'],
                    [not use_cluster,     use_PCA,     use_std, 'KDE'],
                    [not use_cluster,     use_PCA, not use_std, 'KDE'],
                    [not use_cluster, not use_PCA,     use_std, 'KDE'],
                    [   'silhouette', not use_PCA, not use_std, 'GMM'],
                    [         'DBCV', not use_PCA, not use_std, 'GMM'],
                    [not use_cluster, not use_PCA, not use_std, 'GMM'],
                    [   'silhouette',     use_PCA,     use_std, 'KNN'],
                    [   'silhouette',     use_PCA, not use_std, 'KNN'],
                    [   'silhouette', not use_PCA,     use_std, 'KNN'],
                    [         'DBCV',     use_PCA,     use_std, 'KNN'],
                    [         'DBCV',     use_PCA, not use_std, 'KNN'],
                    [         'DBCV', not use_PCA,     use_std, 'KNN'],
                    [not use_cluster,     use_PCA,     use_std, 'KNN'],
                    [not use_cluster,     use_PCA, not use_std, 'KNN'],
                    [not use_cluster, not use_PCA,     use_std, 'KNN']
                    ] 

    #%% Loop over datasets and fit the probability functions


    fitting_pf_str = data_str+'_fitting_pf'
    testing_pf_str = data_str+'_testing_pf'

    fitting_pf = load_dir(fitting_pf_str)
    testing_pf = load_dir(testing_pf_str)

    if len(fitting_pf) == 0:
        for key, _ in fitting_dict.items():
            print('Dataset ' + key)
            # Get distribution independent key
            fitting_clusters_silh = None
            fitting_clusters_dbcv = None
            testing_clusters_silh = None
            testing_clusters_dbcv = None

            for config in testConfigs:
                pf_key = key + '_config'

                if config[0] == 'silhouette':
                    pf_key += '_cluster'
                elif config[0] == 'DBCV':
                    pf_key += '_DBCV'

                if config[1]:
                    pf_key += '_PCA'
                if config[2]:
                    pf_key += '_std'

                pf_key += config[3]

                if not overwrite_key(fitting_pf, pf_key, overwrite_string):
                    continue

                if not('Trajectories' in key):
                    min_std = twoD_min_std
                else:
                    min_std = traj_min_std

                distr_mdl = pf.OPTICS_GMM(use_cluster=config[0], use_PCA=config[1],
                                        use_std=config[2], estimator=config[3], 
                                        min_std=min_std)
                
                distr_mdl_test = pf.OPTICS_GMM(use_cluster=config[0], use_PCA=config[1],
                                            use_std=config[2], estimator=config[3], 
                                            min_std=min_std)

                if config[0] == 'silhouette':
                    distr_mdl.fit(fitting_dict[key], fitting_clusters_silh)
                    fitting_clusters_silh = distr_mdl.cluster_labels
                elif config[0] == 'DBCV':
                    distr_mdl.fit(fitting_dict[key], fitting_clusters_dbcv)
                    fitting_clusters_dbcv = distr_mdl.cluster_labels
                else:
                    distr_mdl.fit(fitting_dict[key])

                fitting_pf[pf_key] = distr_mdl

                if config[0] == 'silhouette':
                    distr_mdl_test.fit(testing_dict[key], testing_clusters_silh)
                    testing_clusters_silh = distr_mdl_test.cluster_labels
                elif config[0] == 'DBCV':
                    distr_mdl_test.fit(testing_dict[key], testing_clusters_dbcv)
                    testing_clusters_dbcv = distr_mdl_test.cluster_labels
                else:
                    distr_mdl_test.fit(testing_dict[key])
                    
                testing_pf[pf_key] = distr_mdl_test
        
        pickle.dump(fitting_pf, open(fitting_pf_str, 'wb'))
        pickle.dump(testing_pf, open(testing_pf_str, 'wb'))

    # Evaluate log likelihoos of samples
    sampled_dict_str = './Distribution Datasets/Fitted_Dists'+rand_str+'_sampled_dict'

    likelihood_str = './Distribution Datasets/Log_Likelihoods'+rand_str

    fitting_pf_fitting_log_likelihood_str = likelihood_str+'_fitting_pf_fitting_log_likelihood'
    fitting_pf_testing_log_likelihood_str = likelihood_str+'_fitting_pf_testing_log_likelihood'
    fitting_pf_sampled_log_likelihood_str = likelihood_str+'_fitting_pf_sampled_log_likelihood'
    testing_pf_fitting_log_likelihood_str = likelihood_str+'_testing_pf_fitting_log_likelihood'
    testing_pf_testing_log_likelihood_str = likelihood_str+'_testing_pf_testing_log_likelihood'

    sampled_dict = load_dir(sampled_dict_str)
    fitting_pf_fitting_log_likelihood = load_dir(fitting_pf_fitting_log_likelihood_str)
    fitting_pf_testing_log_likelihood = load_dir(fitting_pf_testing_log_likelihood_str)
    fitting_pf_sampled_log_likelihood = load_dir(fitting_pf_sampled_log_likelihood_str)
    testing_pf_fitting_log_likelihood = load_dir(testing_pf_fitting_log_likelihood_str)
    testing_pf_testing_log_likelihood = load_dir(testing_pf_testing_log_likelihood_str)

    if len(fitting_pf_fitting_log_likelihood) == 0:
        for key, _ in fitting_pf.items():
            if not overwrite_key(fitting_pf_fitting_log_likelihood, key, overwrite_string):
                continue

            base_data_key = key[:re.search(r"rnd_seed_\d{1,2}", key).end()]

            num_samples_X3 = re.findall(r"samples_\d{1,5}", key)[0][8:] # extract number of samples from key

            # Test if sampling is possible from estimated pdf
            try:
                sampled_data = fitting_pf[key].sample(int(num_samples_X3))
                
                sampled_dict[key] = sampled_data
                fitting_pf_sampled_log_likelihood[key] = fitting_pf[key].score_samples(sampled_dict[key])

            except:
                print('Sampling failed for ' + key)

            fitting_pf_fitting_log_likelihood[key] = fitting_pf[key].score_samples(fitting_dict[base_data_key])
            fitting_pf_testing_log_likelihood[key] = fitting_pf[key].score_samples(testing_dict[base_data_key])
            testing_pf_fitting_log_likelihood[key] = testing_pf[key].score_samples(fitting_dict[base_data_key])
            testing_pf_testing_log_likelihood[key] = testing_pf[key].score_samples(testing_dict[base_data_key])
        
        pickle.dump(fitting_pf_fitting_log_likelihood, open(fitting_pf_fitting_log_likelihood_str, 'wb'))
        pickle.dump(fitting_pf_testing_log_likelihood, open(fitting_pf_testing_log_likelihood_str, 'wb'))
        pickle.dump(fitting_pf_sampled_log_likelihood, open(fitting_pf_sampled_log_likelihood_str, 'wb'))
        pickle.dump(testing_pf_fitting_log_likelihood, open(testing_pf_fitting_log_likelihood_str, 'wb'))
        pickle.dump(testing_pf_testing_log_likelihood, open(testing_pf_testing_log_likelihood_str, 'wb'))
        pickle.dump(sampled_dict, open(sampled_dict_str, 'wb'))

    # %% Calculate Metrics 
    print('Calculate metrics')

    # Get JSD metric
    print('JSD')
    results_str = './Distribution Datasets/Results'+rand_str

    JSD_testing_str = results_str+'_JSD_testing'
    JSD_testing = load_dir(JSD_testing_str)
    if len(JSD_testing) == 0:
        for key, _ in fitting_pf_fitting_log_likelihood.items():
            if not overwrite_key(JSD_testing, key, overwrite_string):
                continue

            JSD_testing[key] = calculate_JSD(fitting_pf_fitting_log_likelihood[key], 
                                             fitting_pf_testing_log_likelihood[key], 
                                             testing_pf_fitting_log_likelihood[key], 
                                             testing_pf_testing_log_likelihood[key])
            
        pickle.dump(JSD_testing, open(JSD_testing_str, 'wb'))

    # Get log Wasserstein metric
    print('Log Wasserstein')
    Wasserstein_log_fitting_testing_str = results_str+'_Wasserstein_log_fitting_testing'
    Wasserstein_log_fitting_sampled_str = results_str+'_Wasserstein_log_fitting_sampled'
    Wasserstein_log_testing_sampled_str = results_str+'_Wasserstein_log_testing_sampled'

    Wasserstein_log_fitting_testing = load_dir(Wasserstein_log_fitting_testing_str)
    Wasserstein_log_fitting_sampled = load_dir(Wasserstein_log_fitting_sampled_str)
    Wasserstein_log_testing_sampled = load_dir(Wasserstein_log_testing_sampled_str)

    if len(Wasserstein_log_fitting_testing) == 0:
        for key, _ in fitting_pf_fitting_log_likelihood.items():
            if not overwrite_key(Wasserstein_log_fitting_testing, key, overwrite_string):
                continue

            Wasserstein_log_fitting_testing[key] = calculate_Wasserstein(fitting_pf_fitting_log_likelihood[key],
                                                                         fitting_pf_testing_log_likelihood[key])
            if key in sampled_dict.keys():
                Wasserstein_log_fitting_sampled[key] = calculate_Wasserstein(fitting_pf_fitting_log_likelihood[key],
                                                                             fitting_pf_sampled_log_likelihood[key])
                
                Wasserstein_log_testing_sampled[key] = calculate_Wasserstein(fitting_pf_testing_log_likelihood[key],
                                                                             fitting_pf_sampled_log_likelihood[key])
        
        pickle.dump(Wasserstein_log_fitting_testing, open(Wasserstein_log_fitting_testing_str, 'wb'))
        pickle.dump(Wasserstein_log_fitting_sampled, open(Wasserstein_log_fitting_sampled_str, 'wb'))
        pickle.dump(Wasserstein_log_testing_sampled, open(Wasserstein_log_testing_sampled_str, 'wb'))
                
    # Get data Wasserstein metric
    Wasserstein_data_fitting_testing_str = results_str+'_Wasserstein_data_fitting_testing'
    Wasserstein_data_fitting_sampled_str = results_str+'_Wasserstein_data_fitting_sampled'
    Wasserstein_data_testing_sampled_str = results_str+'_Wasserstein_data_testing_sampled'

    Wasserstein_data_fitting_testing = load_dir(Wasserstein_data_fitting_testing_str)
    Wasserstein_data_fitting_sampled = load_dir(Wasserstein_data_fitting_sampled_str)
    Wasserstein_data_testing_sampled = load_dir(Wasserstein_data_testing_sampled_str)

    if len(Wasserstein_data_fitting_testing) == 0:
        for key, _ in fitting_pf_fitting_log_likelihood.items():
            if not overwrite_key(Wasserstein_data_fitting_sampled, key, overwrite_string):
                continue

            base_data_key = key[:re.search(r"rnd_seed_\d{1,2}", key).end()]
            if not (base_data_key in Wasserstein_data_fitting_testing.keys()):
                Wasserstein_data_fitting_testing[base_data_key] = calculate_multivariate_Wasserstein(fitting_dict[base_data_key],
                                                                                                     testing_dict[base_data_key])
                
            if key in sampled_dict.keys():    
                Wasserstein_data_fitting_sampled[key] = calculate_multivariate_Wasserstein(fitting_dict[base_data_key], sampled_dict[key])
                Wasserstein_data_testing_sampled[key] = calculate_multivariate_Wasserstein(testing_dict[base_data_key], sampled_dict[key])
        
        pickle.dump(Wasserstein_data_fitting_testing, open(Wasserstein_data_fitting_testing_str, 'wb'))
        pickle.dump(Wasserstein_data_fitting_sampled, open(Wasserstein_data_fitting_sampled_str, 'wb'))
        pickle.dump(Wasserstein_data_testing_sampled, open(Wasserstein_data_testing_sampled_str, 'wb'))

