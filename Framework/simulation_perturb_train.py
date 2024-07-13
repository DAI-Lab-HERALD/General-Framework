from experiment import Experiment

# Draw latex figure
Experiment_name = 'Train_nuscenes_cor_left_turns_trajectron++'
new_experiment = Experiment(Experiment_name)

#%% Select modules
# # Select the datasets
Data_sets = [[{'scenario': 'CoR_left_turns',  'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []},
              {'scenario': 'NuScenes_interactive', 'max_num_agents': None, 't0_type': 'all_10', 'conforming_t0_types': []}]]

# Select the params for the datasets to be considered
Data_params = [{'dt': 0.1, 'num_timesteps_in': (15, 15), 'num_timesteps_out': (20, 20)}] 

# Select the spitting methods to be considered
Splitters = [{'Type': 'Dataset_split', 'repetition': 'L-GAP (left turns)', 'train_on_test': True}]

# Select the models to be trained
Models = [{'model': 'trajectron_salzmann_old', 'kwargs': {'predict_ego': False}}]

# Select the metrics to be used
Metrics = ['ADE_indep','FDE_indep','Collision_rate_indep']

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

# Decide wether missing position in trajectory data can be extrapolated
allow_extrapolation = True

# Use all available agents for predictions
agents_to_predict = 'V'

# Determine if allready existing results shoul dbe overwritten, or if not, be used instead
overwrite_results = 'no'

# Determine if the model should be evaluated on the training set as well
evaluate_on_train_set = False

# Select method used for transformation function to path predictions
model_for_path_transform = 'trajectron_salzmann_old'

new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, allow_extrapolation, 
                              agents_to_predict, overwrite_results, evaluate_on_train_set)


#%% Run experiment
new_experiment.run()                  

# Load results
Results = new_experiment.load_results()

