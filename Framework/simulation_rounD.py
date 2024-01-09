#%%
from experiment import Experiment

# Draw latex figure
Experiment_name = 'RounD'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# # Select the datasets
# Data_sets = [[{'scenario': 'RounD_round_about', 't0_type': 'start', 'conforming_t0_types': []},
#               {'scenario': 'CoR_left_turns',    't0_type': 'crit', 'conforming_t0_types': []}]]

Data_sets = [{'scenario': 'RounD_round_about', 'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.2, 'num_timesteps_in': (15, 15), 'num_timesteps_out': (25, 25)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Cross_split', 'repetition': [0,1,2,3,4], 'test_part': 0.2}]

# Select the models to be trained
Models = ['trajectron_salzmann_old', 
          'pecnet_mangalam',
          'mid_gu',
        {'model': 'flomo_schoeller',
          'kwargs': {
                     'obs_encoding_size': 16,
                     'beta_noise': 0.002,
                     'gamma_noise': 0.002,
                     'alpha': 3,
                     's_min': 0.8,
                     's_max': 1.2,
                     'sigma': 0.2}},
        {'model': 'trajflow_meszaros',
        'kwargs': {'fut_enc_sz': 4, 
                    'beta_noise': 0.0,
                    'gamma_noise': 0.0,
                    'alpha': 3,
                    's_min': 0.8,
                    's_max': 1.2,
                    'sigma': 0.2}},
        {'model': 'trajflow_meszaros',
        'kwargs': {'fut_enc_sz': 8, 
                    'beta_noise': 0.0,
                    'gamma_noise': 0.0,
                    'alpha': 3,
                    's_min': 0.8,
                    's_max': 1.2,
                    'sigma': 0.2}},
        {'model': 'trajflow_meszaros',
        'kwargs': {'fut_enc_sz': 12, 
                    'beta_noise': 0.0,
                    'gamma_noise': 0.0,
                    'alpha': 3,
                    's_min': 0.8,
                    's_max': 1.2,
                    'sigma': 0.2}}
          ]

# Select the metrics to be used
Metrics = ['minADE20_indep', 'minFDE20_indep', 'ADE_ML_indep', 'FDE_ML_indep',
'Oracle_indep', 'KDE_NLL_indep', 'ECE_traj_indep', 
'minADE20_joint', 'minFDE20_joint', 'ADE_ML_joint', 'FDE_ML_joint',
'Oracle_joint', 'KDE_NLL_joint', 'ECE_traj_joint', 'ECE_class', 'AUC_ROC']


new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 100

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_time = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = True

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = True

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = True

# Use all available agents for predictions
agents_to_predict = 'predefined'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = True

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_time, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, evaluate_on_train_set)


#%% Run experiment
# new_experiment.run() 

# Load results
Results, Train_results, Loss = new_experiment.load_results(plot_if_possible = False,
                                                           return_train_results = True,
                                                           return_train_loss = True)


# new_experiment.draw_figure(include_only_mean = False)
new_experiment.write_tables(dataset_row=False)

# new_experiment.plot_paths()
# %%
