# Running up a new experiment
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
  
  For datasets, where the classification of trajectories into certain behaviors are possible, three more options are available.
  - 'col_set': The prediction is made at the point where the predicted time until the [default classification](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-classifiable-behaviors) is fulfilled has the following size $\Delta t$ = <dt> * <num_timesteps_in> (see below at Data_params).
  - 'col_equal': 




```
Data_params = [{'dt': 0.5, 'num_timesteps_in': (4, 4), 'num_timesteps_out': (12, 12)}] 
```
```
Splitters = [{'Type': '<Split_method_1>', 'repetition': [0,1], 'test_part': 0.2},
             {'Type': '<Split_method_2>', 'repetition': [0,1,2], 'test_part': 0.2}]
```
```
Models = ['trajectron_salzmann_unicycle']
```
```
Metrics = ['ADE_joint']
```


Finally, one has to pass the selected modules to the experiment.
```
new_experiment.set_modules(Data_sets, Data_params, Splitters, Models, Metrics)
```
