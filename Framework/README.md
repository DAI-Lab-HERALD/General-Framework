# Running a new experiment
To set up an experiment with the modules in the framework, one has to create or change a *simulations.py* file. This Python script consists of a number of steps, which will be explained in more detail in the following sections.

## Set up an experiment
In a first step, we have to set up the experiment we want to run:
```
from experiment import Experiment

# Draw latex figure
Experiment_name = '<Experiment name>'
new_experiment = Experiment(Experiment_name)
```
The only required input here is **Experiment_name**, which will be included in the final saved files for created tables and figures. So if it is desired that those are not overwritten, it should be unique for each different experiment.

## Select Modules
In the second step, one then has to make the choices regarding the modules (datasets, models, etc.) one wants to include in the current experiment. The first selection regards the datasets:
```
Data_sets = [{'scenario': '<Dataset 1>', 'max_num_agents': 6, 't0_type': 'all', 'conforming_t0_types': []},
             [{'scenario': '<Dataset 2>', 'max_num_agents': 3, 't0_type': 'start', 'conforming_t0_types': []},
              {'scenario': '<Dataset 3>', 'max_num_agents': None, 't0_type': 'all', 'conforming_t0_types': []}],
             {'scenario': '<Dataset 4>', 'max_num_agents': None, 't0_type': 'crit', 'conforming_t0_types': [start]},
             {'scenario': '<Dataset 4>', 'max_num_agents': None, 't0_type': 'start', 'conforming_t0_types': [crit]}]
```
As can be seen above, each dataset is passed as a dictionary to the list of datasets. In such a dictionary, four entries are expected.
- 'scenario': This is the actual dataset. The string here given should correspond to the name of a respective **.py* file in the [Dataset folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Data_sets).
- 'max_num_agents': This is the maximum number of agents one wants to consider in a single scene (The farthest away agents are neglected). If None is passed, all possible agents are included in each sample. If multiple datasets are combined, then the smallest number is chosen for the combined dataset.
- 't0_type':This is a string that controls how the prediction timepoints are determined from a given trajectory, used to divide the trajectory into past and future observations. The following choices are available:
  - 'all': Every timestep where enough past and future observations are available (and agents aren't in wrong positions) is taken as on prediction time.
  - 'start': The first timestep at which all agents are in correct positions (see [*evaluate_scenario()*](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/README.md#extracting-classifiable-behavior)) is taken as the prediction time
  
  For datasets, where the classification of trajectories into certain behaviors is possible, three more options are available.
  - 'col_set': The prediction is made at the point where the predicted time until the [default classification](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-classifiable-behaviors) is fulfilled has the following size $\Delta t = \delta t \cdot n_I$ (see below at Data_params where $\delta t$ and $n_I$ are set.).
  - 'col_equal': Similar to 'col_set', only here $\Delta t$ is set so that over all possible behaviors the smallest number of samples that are included (i.e., at the prediction time the trajectory can not yet be classified) is maximized.
  - 'crit': The prediction is made at the last point in time where a prediction is still useful (for example, if one wants to predict in which direction a vehicle will turn at the intersection, this should be done before the vehicle enters the intersection). This can be defined via [*scenario.calculate_safe_action()*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-safe-actions).
- 'conforming_t0_types': If **t0_type** is not set to 'all', then one can set one of the other three choices for t0_types that is not 'all'. If this is done, then a sample is only included in the overall dataset, if it would have been included under those conforming t0_types as well. This allows one to compare the influence of different t0_types on model performance while guaranteeing that the datasets are otherwise identical. 

It is also possible to combine multiple datasets into one. In this case, one has to put those multiple datasets into another list inside the list **Data_sets**.

In the next step, one then has to set the parameters for the past and future trajectories given to the models. Like with **Data_sets** above, this will be a number of dictionaries:
```
Data_params = [{'dt': 0.2, 'num_timesteps_in': (8, 8), 'num_timesteps_out': (12, 12)},
               {'dt': 0.2, 'num_timesteps_in': (4, 8), 'num_timesteps_out': (12, 12)}] 
```
Here, the following keys have to be set:
- 'dt': This is the timestep size $\delta t$ given in seconds, i.e., the time that passes between observed trajectory positions.
- 'num_timesteps_in': These are the values for the number of input timesteps $n_I$. The first number in the tuple sets the number of actual timesteps given to the models. The second number meanwhile determines how many timesteps of data have to be available before the prediction time. This can be used to easier compare the influence of the number of input timesteps on the model performance, by keeping the dataset otherwise identical.
- 'num_timesteps_out': These are the values for the number of output timesteps $n_O$. Again, the second value can be used to tell the framework only to allow samples to be in the dataset if they would also be eligible for a dataset with a higher $n_O$.

After setting up the dataset and its parameters, one then has to select the splitting method.
```
Splitters = [{'Type': '<Split_method_1>', 'repetition': 0, 'test_part': 0.2},
             {'Type': '<Split_method_2>', 'repetition': [0,1,2], 'test_part': 0.2}]
```
Again, this is passed as a dictionary with three keys:
- 'Type': This is the name of the splitting method, which should be identical to one of the **.py* files in the [Splitting method Folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Splitting_methods).
- 'repetition': This is the repetition number of this split. It can either be an integer or a list of integers if the same method should be used repeatedly with shifted outputs (such as for cross-validation, or looping through locations).
- 'test_part': This is a value (between 0 and 1) that denotes the portion of the whole dataset that is used for the evaluation of the trained method.

Second to last, one has to select the models that are to be evaluated in this experiment.
```
Models = ['trajectron_salzmann_unicycle']
```
Different from the previous list, **Models** contains only the name of the available **.py* files from the [Model folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Models).

Lastly, one has to select the metrics by which the models are to be evaluated. 
```
Metrics = ['<Metric name 1>', '<Metric name 2>']
```
Like with the models, one includes metrics by giving the name of the respective **.py* file in the [Metrics Folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Evaluation_metrics).


Finally, one has to pass the selected modules to the experiment.
```
new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)
```

## Set the experiment hyperparameters
Besides selecting the modules, one must also set some hyperparameters for the overall framework.
```
num_samples_path_pred = 100
```
This sets the number $N_{preds}$ of different predictions that are expected from each trajectory prediction model to represent the inherent stochasticity of their predictions. While this number can be set to any liking, setting it to at least 20 is advisable to be comparable with most standard metrics, which are normally set on 20 such parameters.

```
enforce_prediction_times = True
```
When extracting the prediction time, it might be possible that there is not enough previously observed data available for a chosen prediction time to get positions for all required input time steps. In such cases, if **enforce_prediction_times** is set to *True*, then such a sample would be discarded, while otherwise the prediction time is moved back until sufficient observations are available.

```
enforce_num_timesteps_out = False
```
When extracting samples, it might be possible that not enough data is available to allow for a sufficient number of output time steps, the setting **enforce_num_timesteps_out** to *True* would result in dismissing the sample, while setting it to *False* would mean retaining it.

```
exclude_post_crit = True
```
If the dataset allows for classification, and a method for defining the [last useful prediction time](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-safe-actions) is available, the setting *exclude_post_crit** to *True* would result in all samples, where the prediction time is after this last useful time, to be discarded. If one chooses to set *False* instead, then only samples where the behavior can already be classified during past observations are discarded.

```
allow_extrapolation = True
```
In the framework, it is possible to extrapolate missing position data of certain agents during past and future observations, as some models might not be able to deal with such missing information. However, such extrapolated data might obfuscate the true human behavior. Therefore, this extrapolation can be prevented by setting **allow_extrapolation** to *False*, which might however lead to problems with some models.

```
dynamic_prediction_agents = False
```
In most situations, only a handful of agents in each scene are [set as prediction agents by the scenario](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-important-actors). This might not be enough to properly evaluate the model. In this case, it might be possible to include also any other agent in the scene, for which sufficient original past and future observations have been made, in the list of predicted and evaluated agents.

```
overwrite_results = False
```
It might be possible that one wants to retrain and reevaluate models, without having to delete the original files in their folder. In this case, one can set **overwrite_results** to *True*. It must however be noticed that this does not redo the extraction of the training and testing set.

```
model_for_path_transform = '<Trajecotry Prediction Model>'
```
One part of the framework allows one to transform the predictions made by classification models into trajectories. This is based on a number of conditional trajectory prediction models trained only in the samples where the respective class was observed, predicting a number of trajectories proportional to the predicted probability that this class will be observed. This trajectory prediction model has to be selected here.

Finally, one has to pass these parameters to the experiment.
```
new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, allow_extrapolation, 
                              dynamic_prediction_agents, overwrite_results)
```






