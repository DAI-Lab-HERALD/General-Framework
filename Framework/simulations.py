from experiment import Experiment

# Draw latex figure
Experiment_name = 'Test'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# # Select the datasets
# Data_sets = [[{'scenario': 'RounD_round_about', 't0_type': 'start', 'conforming_t0_types': []},
#               {'scenario': 'CoR_left_turns',    't0_type': 'crit', 'conforming_t0_types': []}]]

Data_sets = [{'scenario': 'HighD_lane_change', 't0_type': 'start', 'conforming_t0_types': []}]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.2, 'num_timesteps_in': (4, 4), 'num_timesteps_out': (10, 10)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Random_split', 'repetition': 0, 'test_part': 0.2}]

# Select the models to be trained
Models = ['trajectron_salzmann']

# Select the metrics to be used
Metrics = ['ROC_curve']

new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)

#%% Other settings
# Set the number of different trajectories to be predicted by trajectory prediction models.
num_samples_path_pred = 100

# Deciding wether to enforce start times (or allow later predictions if not enough input data is available)
enforce_prediction_times = True

# determine if the upper bound for n_O should be enforced, or if prediction can be made without
# underlying output data (might cause training problems)
enforce_num_timesteps_out = False

# Determine if the useless prediction (i.e, prediction you cannot act anymore)
# should be exclude from the dataset
exclude_post_crit = True

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = False

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, overwrite_results)

#%% Run experiment
new_experiment.run()                  

# Load results
Results = new_experiment.load_results(plot_if_possible = True, return_train_results = False)

# new_experiment.draw_figure(include_only_mean = False)
# new_experiment.write_tables()
