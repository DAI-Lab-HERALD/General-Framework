#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re

import Prob_function as pf
import Prob_function_knn as pf_knn

from utils import *

overwrite = False


def main(random_seeds):


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
    ToyProblem_future = ToyProblem_future.reshape(len(ToyProblem_future), ToyProblem_future.shape[1]*ToyProblem_future.shape[2])

    #%% Create multiple datasets with different number of samples 
    # and save to dictionaries with keys containing info on dataset_name, n_samples and rand_seed

    num_samples = [200, 400, 1000, 2000, 4000, 10000, 20000]

    fitting_dict = {}
    testing_dict = {}

    for n_samples in num_samples:
        for rnd_seed in random_seeds:
            key = 'n_samples_' + str(n_samples) + '_rnd_seed_' + str(rnd_seed)
            fitting_dict['noisy_circles_' + key], testing_dict['noisy_circles_' + key] = create_random_data_splt(noisy_circles, rnd_seed, n_samples)
            fitting_dict['noisy_moons_' + key], testing_dict['noisy_moons_' + key] = create_random_data_splt(noisy_moons, rnd_seed, n_samples)
            fitting_dict['blobs_' + key], testing_dict['blobs_' + key] = create_random_data_splt(blobs, rnd_seed, n_samples)
            fitting_dict['varied_' + key], testing_dict['varied_' + key] = create_random_data_splt(varied, rnd_seed, n_samples)
            fitting_dict['aniso_' + key], testing_dict['aniso_' + key] = create_random_data_splt(aniso, rnd_seed, n_samples)
            fitting_dict['ToyProblem_' + key], testing_dict['ToyProblem_' + key] = create_random_data_splt(ToyProblem_future, rnd_seed, n_samples)


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
                    [    use_cluster,     use_PCA,     use_std, 'KDE'],
                    [    use_cluster,     use_PCA, not use_std, 'KDE'],
                    [    use_cluster, not use_PCA,     use_std, 'KDE'],
                    [not use_cluster,     use_PCA,     use_std, 'KDE'],
                    [not use_cluster,     use_PCA, not use_std, 'KDE'],
                    [not use_cluster, not use_PCA,     use_std, 'KDE'],
                    [    use_cluster, not use_PCA, not use_std, 'GMM'],
                    [not use_cluster, not use_PCA, not use_std, 'GMM'],
                    [    use_cluster,     use_PCA,     use_std, 'KNN'],
                    [    use_cluster,     use_PCA, not use_std, 'KNN'],
                    [    use_cluster, not use_PCA,     use_std, 'KNN'],
                    [not use_cluster,     use_PCA,     use_std, 'KNN'],
                    [not use_cluster,     use_PCA, not use_std, 'KNN'],
                    [not use_cluster, not use_PCA,     use_std, 'KNN']
                    ] 

    #%% Loop over datasets and fit the probability functions
    if not os.path.exists('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf') or overwrite:

        fitting_pf = {}
        testing_pf = {}

        for key, _ in fitting_dict.items():
            print('Dataset ' + key)
            for config in testConfigs:
                pf_key = key + '_config'

                if config[0]:
                    pf_key += '_cluster'
                if config[1]:
                    pf_key += '_PCA'
                if config[2]:
                    pf_key += '_std'

                pf_key += config[3]


                if not('ToyProblem' in key):
                    distr_mdl = pf.OPTICS_GMM(use_cluster=config[0], use_PCA=config[1],
                                            use_std=config[2], use_KDE=config[3], 
                                            min_std=twoD_min_std)
                    
                    distr_mdl_test = pf.OPTICS_GMM(use_cluster=config[0], use_PCA=config[1],
                                            use_std=config[2], use_KDE=config[3], 
                                            min_std=twoD_min_std)
                    

                else:
                    distr_mdl = pf.OPTICS_GMM(use_cluster=config[0], use_PCA=config[1],
                                            use_std=config[2], use_KDE=config[3], 
                                            min_std=traj_min_std)
                    
                    distr_mdl_test = pf.OPTICS_GMM(use_cluster=config[0], use_PCA=config[1],
                                            use_std=config[2], use_KDE=config[3], 
                                            min_std=traj_min_std)

                distr_mdl.fit(fitting_dict[key])
                fitting_pf[pf_key] = distr_mdl

                distr_mdl_test.fit(testing_dict[key])
                testing_pf[pf_key] = distr_mdl_test

        pickle.dump(fitting_pf, open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf', 'wb'))
        pickle.dump(testing_pf, open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_testing_pf', 'wb'))

    else:
        fitting_pf = pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf', 'rb'))
        testing_pf = pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_testing_pf', 'rb'))

    # %% Loop over datasets and sample from the fitted probability functions and at the same time calculate the log-likelihood of the train and test data
    if not os.path.exists('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_sampled_dict') or overwrite:
        
        sampled_dict = {}
        fitting_pf_fitting_log_likelihood = {}
        fitting_pf_testing_log_likelihood = {}
        fitting_pf_sampled_log_likelihood = {}
        testing_pf_fitting_log_likelihood = {}
        testing_pf_testing_log_likelihood = {}

        for key, _ in fitting_pf.items():
            num_samples_X3 = re.findall(r"samples_\d{1,5}", key)[0][8:] # extract number of samples from key

            try:
                sampled_data = fitting_pf[key].sample(int(num_samples_X3))
                
                sampled_dict[key] = sampled_data
                fitting_pf_sampled_log_likelihood[key] = fitting_pf[key].score_samples(sampled_dict[key])

            except:
                print('Sampling failed for ' + key)


            base_data_key = key[:re.search(r"rnd_seed_\d{1,2}", key).end()]

            fitting_pf_fitting_log_likelihood[key] = fitting_pf[key].score_samples(fitting_dict[base_data_key])
            fitting_pf_testing_log_likelihood[key] = fitting_pf[key].score_samples(testing_dict[base_data_key])
            testing_pf_fitting_log_likelihood[key] = testing_pf[key].score_samples(fitting_dict[base_data_key])
            testing_pf_testing_log_likelihood[key] = testing_pf[key].score_samples(testing_dict[base_data_key])


        pickle.dump(sampled_dict, open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_sampled_dict', 'wb'))
        pickle.dump(fitting_pf_fitting_log_likelihood, open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf_fitting_log_likelihood', 'wb'))
        pickle.dump(fitting_pf_testing_log_likelihood, open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf_testing_log_likelihood', 'wb'))
        pickle.dump(fitting_pf_sampled_log_likelihood, open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf_sampled_log_likelihood', 'wb'))
        pickle.dump(testing_pf_fitting_log_likelihood, open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_testing_pf_fitting_log_likelihood', 'wb'))
        pickle.dump(testing_pf_testing_log_likelihood, open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_testing_pf_testing_log_likelihood', 'wb'))

    else:
        sampled_dict = pickle.load(open('./Distribution Datasets/Fitted_Dists/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_sampled_dict', 'rb'))
        fitting_pf_fitting_log_likelihood = pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf_fitting_log_likelihood', 'rb'))
        fitting_pf_testing_log_likelihood = pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf_testing_log_likelihood', 'rb'))
        fitting_pf_sampled_log_likelihood = pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_fitting_pf_sampled_log_likelihood', 'rb'))
        testing_pf_fitting_log_likelihood = pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_testing_pf_fitting_log_likelihood', 'rb'))
        testing_pf_testing_log_likelihood = pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_testing_pf_testing_log_likelihood', 'rb'))


    # %% Calculate Metrics 
    print('Calculate metrics')
    # 2D Distributions
    JSD_testing = {}
    Wasserstein_log_fitting_testing, Wasserstein_log_fitting_sampled, Wasserstein_log_testing_sampled = {}, {}, {}
    Wasserstein_data_fitting_testing, Wasserstein_data_fitting_sampled, Wasserstein_data_testing_sampled = {}, {}, {}

    logMean_fitting_fitting, logMean_testing_testing = {}, {}
    logMean_fitting_testing, logMean_fitting_sampled = {}, {}
    logStd_fitting_fitting, logStd_testing_testing = {}, {}
    logStd_fitting_testing, logStd_fitting_sampled = {}, {}
    
    dataMean_fitting, dataMean_testing, dataMean_sampled = {}, {}, {}
    dataStd_fitting, dataStd_testing, dataStd_sampled = {}, {}, {}

    logQuantile_fitting_fitting, logQuantile_testing_testing = {}, {}
    logQuantile_fitting_testing, logQuantile_fitting_sampled = {}, {}

    dataQuantile_fitting, dataQuantile_testing, dataQuantile_sampled = {}, {}, {}


    for key, _ in fitting_pf_fitting_log_likelihood.items():

        base_data_key = key[:re.search(r"rnd_seed_\d{1,2}", key).end()]


        JSD_testing[key] = calculate_JSD(fitting_pf_fitting_log_likelihood[key], 
                                            fitting_pf_testing_log_likelihood[key], 
                                            testing_pf_fitting_log_likelihood[key], 
                                            testing_pf_testing_log_likelihood[key])
                    
        Wasserstein_log_fitting_testing[key] = calculate_Wasserstein(fitting_pf_fitting_log_likelihood[key],
                                                                    fitting_pf_testing_log_likelihood[key])
        if fitting_pf_sampled_log_likelihood.has_key(key):
            Wasserstein_log_fitting_sampled[key] = calculate_Wasserstein(fitting_pf_fitting_log_likelihood[key],
                                                                        fitting_pf_sampled_log_likelihood[key])
            
            Wasserstein_log_testing_sampled[key] = calculate_Wasserstein(fitting_pf_testing_log_likelihood[key],
                                                                        fitting_pf_sampled_log_likelihood[key])
        
        if not Wasserstein_data_fitting_testing.has_key(base_data_key):
            Wasserstein_data_fitting_testing[base_data_key] = calculate_multivariate_Wasserstein(fitting_dict[base_data_key],
                                                                                                 testing_dict[base_data_key])
            
        if sampled_dict.has_key(key):    
            Wasserstein_data_fitting_sampled[key] = calculate_multivariate_Wasserstein(fitting_dict[base_data_key],
                                                                                                    sampled_dict[key])
            Wasserstein_data_testing_sampled[key] = calculate_multivariate_Wasserstein(testing_dict[base_data_key],
                                                                                                    sampled_dict[key])
        
        logMean_fitting_fitting[key] = np.mean(fitting_pf_fitting_log_likelihood[key])
        logMean_testing_testing[key] = np.mean(testing_pf_testing_log_likelihood[key])
        logMean_fitting_testing[key] = np.mean(fitting_pf_testing_log_likelihood[key])

        logStd_fitting_fitting[key] = np.std(fitting_pf_fitting_log_likelihood[key])
        logStd_testing_testing[key] = np.std(testing_pf_testing_log_likelihood[key])
        logStd_fitting_testing[key] = np.std(fitting_pf_testing_log_likelihood[key])

        if fitting_pf_sampled_log_likelihood.has_key(key):
            logMean_fitting_sampled[key] = np.mean(fitting_pf_sampled_log_likelihood[key])
            logStd_fitting_sampled[key] = np.std(fitting_pf_sampled_log_likelihood[key])

        if not dataMean_fitting.has_key(base_data_key):
            dataMean_fitting[base_data_key] = np.mean(fitting_dict[base_data_key], axis=0)
            dataStd_fitting[base_data_key] = np.std(fitting_dict[base_data_key], axis=0)
            dataQuantile_fitting[base_data_key] = np.quantile(fitting_dict[base_data_key], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)

        if not dataMean_testing.has_key(base_data_key):
            dataMean_testing[base_data_key] = np.mean(testing_dict[base_data_key], axis=0)
            dataStd_testing[base_data_key] = np.std(testing_dict[base_data_key], axis=0)
            dataQuantile_testing[base_data_key] = np.quantile(testing_dict[base_data_key], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)

        if sampled_dict.has_key(key):
            dataMean_sampled[key] = np.mean(sampled_dict[key], axis=0)
            dataStd_sampled[key] = np.std(sampled_dict[key], axis=0)
            dataQuantile_sampled[key] = np.quantile(sampled_dict[key], [0.1, 0.25, 0.5, 0.75, 0.9], axis=0)

        logQuantile_fitting_fitting[key] = np.quantile(fitting_pf_fitting_log_likelihood[key], [0.1, 0.25, 0.5, 0.75, 0.9])
        logQuantile_testing_testing[key] = np.quantile(testing_pf_testing_log_likelihood[key], [0.1, 0.25, 0.5, 0.75, 0.9])
        logQuantile_fitting_testing[key] = np.quantile(fitting_pf_testing_log_likelihood[key], [0.1, 0.25, 0.5, 0.75, 0.9])

        if fitting_pf_sampled_log_likelihood.has_key(key):
            logQuantile_fitting_sampled[key] = np.quantile(fitting_pf_sampled_log_likelihood[key], [0.1, 0.25, 0.5, 0.75, 0.9])

    # %% Save results
    pickle.dump(JSD_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_JSD_testing', 'wb'))
    pickle.dump(Wasserstein_log_fitting_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_Wasserstein_log_fitting_testing', 'wb'))
    pickle.dump(Wasserstein_log_fitting_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_Wasserstein_log_fitting_sampled', 'wb'))
    pickle.dump(Wasserstein_log_testing_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_Wasserstein_log_testing_sampled', 'wb'))

    pickle.dump(logMean_fitting_fitting, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logMean_fitting_fitting', 'wb'))
    pickle.dump(logMean_testing_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logMean_testing_testing', 'wb'))
    pickle.dump(logMean_fitting_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logMean_fitting_testing', 'wb'))
    pickle.dump(logMean_fitting_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logMean_fitting_sampled', 'wb'))

    pickle.dump(logStd_fitting_fitting, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logStd_fitting_fitting', 'wb'))
    pickle.dump(logStd_testing_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logStd_testing_testing', 'wb'))
    pickle.dump(logStd_fitting_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logStd_fitting_testing', 'wb'))
    pickle.dump(logStd_fitting_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logStd_fitting_sampled', 'wb'))

    pickle.dump(dataMean_fitting, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataMean_fitting', 'wb'))
    pickle.dump(dataMean_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataMean_testing', 'wb'))
    pickle.dump(dataMean_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataMean_sampled', 'wb'))

    pickle.dump(dataStd_fitting, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataStd_fitting', 'wb'))
    pickle.dump(dataStd_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataStd_testing', 'wb'))
    pickle.dump(dataStd_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataStd_sampled', 'wb'))

    pickle.dump(logQuantile_fitting_fitting, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logQuantile_fitting_fitting', 'wb'))
    pickle.dump(logQuantile_testing_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logQuantile_testing_testing', 'wb'))
    pickle.dump(logQuantile_fitting_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logQuantile_fitting_testing', 'wb'))
    pickle.dump(logQuantile_fitting_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_logQuantile_fitting_sampled', 'wb'))

    pickle.dump(dataQuantile_fitting, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataQuantile_fitting', 'wb'))
    pickle.dump(dataQuantile_testing, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataQuantile_testing', 'wb'))
    pickle.dump(dataQuantile_sampled, open('./Distribution Datasets/Results/rndSeed'+str(random_seeds.start)+str(random_seeds.stop)+'_dataQuantile_sampled', 'wb'))
