#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import os
import scipy as sp

#%% Write tables
def write_table(data, filename, decimal_place = 2):

    assert len(data.shape) == 3, 'Data must have 3 dimensions'
    assert data.shape[0] == 4, 'Data must have 4 datasets'
    assert data.shape[1] == 9, 'Data must have 9 models/metric combinations'

    num_data_columns = data.shape[1]
        
    # Allow for split table
    Output_string = r'\begin{tabularx}{\textwidth}'
    
    Output_string += r'{X | Y Y Y | Y Y Y | Y Y Y}'
    Output_string += '\n'

    Output_string += r'\toprule[1pt] '
    Output_string += '\n'

    Metrics = [r'$D_{JS} \downarrow_{0}^{1}$', r'$\widehat{W} \rightarrow 0$', r'$\widehat{L} \uparrow$']
    Methods = [r'$f_{\text{ROME}}$', r'$f_{\text{MPW}}$', r'$f_{\text{VC}}$']
    
    for metric in Metrics:
        Output_string += r'& \multicolumn{' + str(len(Methods)) + r'}'
        if metric == Metrics[-1]:
            Output_string += r'{c}'
        else:
            Output_string += r'{c|}'
            
        Output_string += r'{' + metric + r'} '
    
    Output_string += r' \\'
    
    for metric in Metrics:
        for method in Methods:
            Output_string += r'& \multicolumn{1}'
            if (metric != Metrics[-1]) and method == Methods[-1]:
                Output_string += r'{c|}'
            else:
                Output_string += r'{c}'
                
            Output_string += r'{' + method + r'} '
    
    Output_string += r' \\ \midrule[1pt]'
    Output_string += '\n'

    data_mean = np.nanmean(data, axis = -1)
    data_std = np.nanstd(data, axis = -1)

    min_value = ('{:0.' + str(decimal_place) + 'f}').format(np.nanmin(data_mean))
    max_value = ('{:0.' + str(decimal_place) + 'f}').format(np.nanmax(data_mean))

    extra_str_length = max(len(min_value), len(max_value)) - (decimal_place + 1)
    
    Row_names = ['Aniso', 'Varied', 'Two Moons', 'Trajectories']
    
    for i, row_name in enumerate(Row_names): 
        Output_string += row_name + ' '
    
        T1 = r'& {{\scriptsize ${:0.' + str(decimal_place + 1) + r'f}^{{\pm {:0.' + str(decimal_place + 1) + r'f}}}$}} '
        T0 = r'& {{\scriptsize ${:0.' + str(decimal_place) + r'f}^{{\pm {:0.' + str(decimal_place) + r'f}}}$}} '
                
        template = T1 * len(Methods) + T0 * len(Methods) * (len(Metrics) - 1)
        
        Str = template.format(*np.array([data_mean[i], data_std[i]]).T.reshape(-1))

        # Replace 0.000 with hphantom if needed
        # Str = Str.replace("\\pm 0." + str(0) * decimal_place, r"\hphantom{\pm 0." + str(0) * decimal_place + r"}")
        
        # Adapt length to align decimal points
        Str_parts = Str.split('$} ')
        for idx, string in enumerate(Str_parts):
            if idx < len(Methods):
                dp = decimal_place + 1 
            else:
                dp = decimal_place
            
            if len(string) == 0:
                continue
            # previous_string = string.split('.')[0].split('$')[-1]
            # overwrite = False
            # if previous_string[0] == '-':
            #     overwrite = previous_string[1:].isnumeric()
            # else:
            #     overwrite = previous_string.isnumeric()
            # if overwrite:
            #     needed_buffer = extra_str_length - len(previous_string)  
            #     if needed_buffer > 0:
            #         Str_parts[idx] = string[:16] + r'\hphantom{' + '0' * needed_buffer + r'}' + string[16:]
            
            # Check for too long stds
            string_parts = Str_parts[idx].split('^')
            if len(string_parts) > 1 and 'hphantom' not in string_parts[1]:
                std_number = string_parts[1][5:7 + dp]
                if std_number[-1] == '.':
                    std_number = std_number[:-1] + r'\hphantom{0}'
                string_parts[1] = r'{\pm ' + std_number + r'}' 
        
            Str_parts[idx] = '^'.join(string_parts)
                
        Str = '$} '.join(Str_parts)

        Output_string += Str + r' \\' + ' \midrule \n'
    
    # replace last midrule with bottom rule  
    Output_string  = Output_string[:-10] + r'\bottomrule[1pt]'
    Output_string += '\n'
    Output_string += r'\end{tabularx}' + ' \n' 

    # split string into lines
    Output_lines = Output_string.split('\n')

    t = open(filename, 'w+')
    for line in Output_lines:
        t.write(line + '\n')
    t.close()


#%% Define results

# List of random seeds
random_seeds = [['0','10'],
                ['10','20'],
                ['20','30'],
                ['30','40'],
                ['40','50'],
                ['50','60'],
                ['60','70'],
                ['70','80'],
                ['80','90'],
                ['90','100']]

# list of ablation keys
ablation_keys = ['config_cluster_PCA_stdKDE',
                 'MP_Windows',
                 'MPS_Windows',
                 'MPK_Windows',
                 'KDevine']

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
                'Trajectories_n_samples_20000'
                ]


#%% Load Results
JSD_testing = {}
Wasserstein_data_fitting_testing, Wasserstein_data_fitting_sampled = {}, {}

fitting_pf_testing_log_likelihood = {}

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
    
    fitting_pf_testing_log_likelihood = {**fitting_pf_testing_log_likelihood,
                                         **pickle.load(open('./Distribution Datasets/Log_Likelihoods/rndSeed'+str(rndSeed[0])+str(rndSeed[1])+
                                                            '_fitting_pf_testing_log_likelihood', 'rb'))}

#%% Plotting
# Create an array of dimensions num_datasets x num_ablations x num_metrics x num_random_seeds
# Each element is a value of the metric for a given dataset, ablation and random seed
# Datasets: noisy_moons, noisy_circles, blobs, varied, aniso, Trajectories
Results = np.ones((len(dataset_keys), len(ablation_keys), 3, 100)) * np.nan

# Fill the array with the values from the dictionaries
for _, (k, v) in enumerate(JSD_testing.items()):
    results = np.ones(3) * np.nan
    # Get metrics from key
    if not isinstance(JSD_testing[k], str):
        results[0] = JSD_testing[k]
    
    if k in Wasserstein_data_fitting_sampled.keys():
        if not isinstance(Wasserstein_data_fitting_sampled[k], str):
            bk = k[:re.search(r"rnd_seed_\d{1,2}", k).end()]
            Wasserstein_hat = Wasserstein_data_fitting_sampled[k] - Wasserstein_data_fitting_testing[bk]
            results[1] = Wasserstein_hat / (Wasserstein_data_fitting_testing[bk] + 1e-4)
    
    if k in fitting_pf_testing_log_likelihood.keys():
        if not isinstance(fitting_pf_testing_log_likelihood[k], str):
            results[2] = np.mean(fitting_pf_testing_log_likelihood[k])
        
    # Place key in Results array
    rndSeed = int(k[re.search(r"rnd_seed_\d{1,2}", k).start():re.search(r"rnd_seed_\d{1,2}", k).end()][9:])
    dataset_id = np.where([ds in k for ds in dataset_keys])[0]
    if len(dataset_id) == 0:
        continue
    else:
        dataset_id = dataset_id[-1]

    ablation_id = np.where([abl in k for abl in ablation_keys])[0]
    if len(ablation_id) == 0:
        continue
    else:
        ablation_id = ablation_id[0]
     
    Results[dataset_id, ablation_id, :, rndSeed] = results

Results = Results.reshape((-1, 6, *Results.shape[1:]))

# Remove the unneeded datasets
datasets_used = [4, 3, 0, 5]
Results = Results[:, datasets_used]

#%% Write tables
# Use results from 3000 samples only
Data = Results[-2, :, :, :, :]

# Collapse models and metrics
Data = Data.transpose(0,2,1,3).reshape((len(datasets_used), 9, 100))
filename = './Tables/baseline_20000.tex'

write_table(Data, filename, 2)




