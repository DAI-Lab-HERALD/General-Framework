#%%
import os
import pickle
import re

import Prob_function_base as pf

from utils import calculate_JSD, calculate_multivariate_Wasserstein, create_random_data_splt

def load_dir(path):
    if not os.path.exists(path):
        path = path[:path.rfind('/')]
        os.makedirs(path, exist_ok = True)
        return {}
    else:
        return pickle.load(open(path, 'rb'))


def write_key(dict, key, overwrite_string):
    # Check if key already exists
    if not (key in dict.keys()):
        return True
    
    # Check if overwrite string is empty
    if len(overwrite_string) == 0:
        return False
    else:
        ret = True
        for string in overwrite_string:
            if string not in key:
                ret = False
        return ret


def main_base(random_seeds, overwrite_string = []):
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

    num_samples = [200, 600, 2000, 6000] # ,20000

    rand_str = '/rndSeed' + str(random_seeds.start) + str(random_seeds.stop)
    data_str = './Distribution Datasets/Fitted_Dists'+rand_str 

    fitting_dict_str = data_str+'_fitting_dict'
    testing_dict_str = data_str+'_testing_dict'

    fitting_dict = load_dir(fitting_dict_str)
    testing_dict = load_dir(testing_dict_str)

    print("", flush = True)
    print("Extracting datasets", flush = True)
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

    testConfigs = ['MP_Windows', 'MPS_Windows', 'MPK_Windows', 'KDevine'] 

    #%% Loop over datasets and fit the probability functions
    fitting_pf_str = data_str+'_fitting_pf'
    testing_pf_str = data_str+'_testing_pf'

    fitting_pf = load_dir(fitting_pf_str)
    testing_pf = load_dir(testing_pf_str)
    
    print("", flush = True)
    print("Fit distributions", flush = True)
    for key, _ in fitting_dict.items():
        for config in testConfigs:
            pf_key = key + '_' + config

            num_samples_X3 = re.findall(r"samples_\d{1,5}", pf_key)[0][8:] # extract number of samples from key

            # Short term expediant
            if int(num_samples_X3) > 8000:
                continue
            
            if not('Trajectories' in key):
                min_std = twoD_min_std
            else:
                min_std = traj_min_std
                if 2500 < int(num_samples_X3) < 8000:
                    if ((traj_min_std == 0.025) and
                        ('KDevine' not in pf_key)):
                            pf_key += '_0.025'
                        
            if not write_key(fitting_pf, pf_key, overwrite_string):
                continue
            
            print('Fit distribution for ' + pf_key, flush = True)   

            if config == 'MP_Windows':
                PF = pf.MP_Windows
            elif config == 'MPS_Windows':
                PF = pf.MPS_Windows
            elif config == 'MPK_Windows':
                PF = pf.MPK_Windows
            elif config == 'KDevine':
                PF = pf.KDevine

            # Get models
            distr_mdl      = PF(min_std=min_std).fit(fitting_dict[key])
            distr_mdl_test = PF(min_std=min_std).fit(testing_dict[key])

            # Save models
            fitting_pf[pf_key] = distr_mdl
            testing_pf[pf_key] = distr_mdl_test
        
            pickle.dump(fitting_pf, open(fitting_pf_str, 'wb'))
            pickle.dump(testing_pf, open(testing_pf_str, 'wb'))

    #%% Evaluate log likelihoos of samples
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

    print("", flush = True)
    print("Evaluate log likelihoods", flush = True)
    for key, _ in fitting_pf.items():
        if not write_key(fitting_pf_fitting_log_likelihood, key, overwrite_string):
            continue
        
        base_data_key = key[:re.search(r"rnd_seed_\d{1,2}", key).end()]

        num_samples_X3 = re.findall(r"samples_\d{1,5}", key)[0][8:] # extract number of samples from key

        # Short term expediant
        if int(num_samples_X3) > 8000:
            continue
        
        print("Evaluate log likelihood of samples for " + key, flush = True)

        # Test if sampling is possible from estimated pdf
        try:
            sampled_data = fitting_pf[key].sample(int(int(num_samples_X3) * 0.5))
            sampled_dict[key] = sampled_data
        except:
            print('Sampling failed for ' + key)
            
            sampled_dict[key] = 'Failed'
        
        if not isinstance(sampled_dict[key], str):
            try:
                fitting_pf_sampled_log_likelihood[key] = fitting_pf[key].score_samples(sampled_dict[key])
            except:
                fitting_pf_sampled_log_likelihood[key] = 'Failed'
                print('Scoring sampled samples failed for ' + key)
        
        try:
            fitting_pf_fitting_log_likelihood[key] = fitting_pf[key].score_samples(fitting_dict[base_data_key])
            fitting_pf_testing_log_likelihood[key] = fitting_pf[key].score_samples(testing_dict[base_data_key])
            testing_pf_fitting_log_likelihood[key] = testing_pf[key].score_samples(fitting_dict[base_data_key])
            testing_pf_testing_log_likelihood[key] = testing_pf[key].score_samples(testing_dict[base_data_key])
        except:
            print('Scoring old samples failed for ' + key)
            
            fitting_pf_fitting_log_likelihood[key] = 'Failed'
            fitting_pf_testing_log_likelihood[key] = 'Failed'
            testing_pf_fitting_log_likelihood[key] = 'Failed'
            testing_pf_testing_log_likelihood[key] = 'Failed'
    
        pickle.dump(fitting_pf_fitting_log_likelihood, open(fitting_pf_fitting_log_likelihood_str, 'wb'))
        pickle.dump(fitting_pf_testing_log_likelihood, open(fitting_pf_testing_log_likelihood_str, 'wb'))
        pickle.dump(fitting_pf_sampled_log_likelihood, open(fitting_pf_sampled_log_likelihood_str, 'wb'))
        pickle.dump(testing_pf_fitting_log_likelihood, open(testing_pf_fitting_log_likelihood_str, 'wb'))
        pickle.dump(testing_pf_testing_log_likelihood, open(testing_pf_testing_log_likelihood_str, 'wb'))
        pickle.dump(sampled_dict, open(sampled_dict_str, 'wb'))

    # %% Calculate Metrics 
    print("", flush = True)
    print('Calculate metrics', flush = True)

    results_str = './Distribution Datasets/Results'+rand_str
    
    # Get JSD metric
    JSD_testing_str = results_str+'_JSD_testing'
    JSD_testing = load_dir(JSD_testing_str)
    
    # Get data Wasserstein metric
    Wasserstein_data_fitting_testing_str = results_str+'_Wasserstein_data_fitting_testing'
    Wasserstein_data_fitting_sampled_str = results_str+'_Wasserstein_data_fitting_sampled'
    Wasserstein_data_testing_sampled_str = results_str+'_Wasserstein_data_testing_sampled'

    Wasserstein_data_fitting_testing = load_dir(Wasserstein_data_fitting_testing_str)
    Wasserstein_data_fitting_sampled = load_dir(Wasserstein_data_fitting_sampled_str)
    Wasserstein_data_testing_sampled = load_dir(Wasserstein_data_testing_sampled_str)
    
    for key, value in fitting_pf_fitting_log_likelihood.items():
        # Get estimator independent values
        base_data_key = key[:re.search(r"rnd_seed_\d{1,2}", key).end()]
        if not (base_data_key in Wasserstein_data_fitting_testing.keys()):
            Wasserstein_data_fitting_testing[base_data_key] = calculate_multivariate_Wasserstein(fitting_dict[base_data_key],
                                                                                                 testing_dict[base_data_key])
        
        # Check if metrics are possible
        if isinstance(value, str):
            JSD_testing[key]                      = 'Failed'
            Wasserstein_data_fitting_sampled[key] = 'Failed'
            Wasserstein_data_testing_sampled[key] = 'Failed'
            continue
        
        # Check if possible metrics allready exist
        if not write_key(JSD_testing, key, overwrite_string):
            continue
        
        # Calculate metrics not dependent on sampled data
        JSD_testing[key] = calculate_JSD(value, fitting_pf_testing_log_likelihood[key], 
                                         testing_pf_fitting_log_likelihood[key], 
                                         testing_pf_testing_log_likelihood[key])
            
        # Calculate metrics dependent on sampled data
        if not isinstance(sampled_dict[key], str):
            
            Wasserstein_data_fitting_sampled[key] = calculate_multivariate_Wasserstein(fitting_dict[base_data_key], sampled_dict[key])
            Wasserstein_data_testing_sampled[key] = calculate_multivariate_Wasserstein(testing_dict[base_data_key], sampled_dict[key])
        
        else:
            Wasserstein_data_fitting_sampled[key] = 'Failed'
            Wasserstein_data_testing_sampled[key] = 'Failed'
    
        # Save metrics
        pickle.dump(JSD_testing, open(JSD_testing_str, 'wb'))
        
        pickle.dump(Wasserstein_data_fitting_testing, open(Wasserstein_data_fitting_testing_str, 'wb'))
        pickle.dump(Wasserstein_data_fitting_sampled, open(Wasserstein_data_fitting_sampled_str, 'wb'))
        pickle.dump(Wasserstein_data_testing_sampled, open(Wasserstein_data_testing_sampled_str, 'wb'))

