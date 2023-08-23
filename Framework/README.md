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
- 'max_num_agents': This is the maximum number of agents one wants to consider in a single scene (the farthest away agents are neglected). If None is passed, all possible agents are included in each sample. 
- 't0_type':This is a string that controls how the prediction timepoints are determined from a given trajectory, used to divide the trajectory into past and future observations. The following choices are available:
  - 'all': Every timestep where enough past and future observations are available (and agents are in correct positions) is taken as on prediction time.
  - 'start': The first timestep at which all agents are in correct positions (see [*evaluate_scenario()*](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/README.md#extracting-classifiable-behavior)) is taken as the prediction time
  
  For datasets, where the classification of trajectories into certain behaviors is possible, three more options are available.
  - 'col_set': At each point in time $t$, we can approximate based on constant velocities that the [default classification](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-classifiable-behaviors) will be fulfilled at point $\widehat{t}_D(t)$. We then define the prediction time $t_0$ to be the first point in time where the condition $\widehat{t}_D(t_0) = t_0 + \Delta t$ is met. Here, we set $\Delta t = \delta t \cdot n_O$ (see below at Data_params where $\delta t$ and $n_O$ are set.).
  - 'col_equal': We select the prediction times similar to 'col_set', except for a different value of $\Delta t$. Here, we select $\Delta t$ in such a way, that the number $N_{min} (\Delta t)$ is maximized. For a dataset with multiple possible classifiable behaviors, each behavior $b$ is represented by $N_b$ samples. Then, we set $N_{min} = \underset{b\in B}{\min} N_b$. These numbers vary with $\Delta t$, as there might not be enough input timesteps available before a selected prediction time $t_0$, or another classifiable behavior was already observed before $t_0$.
  - 'crit': The prediction is made at the last point in time where a prediction is still useful (for example, if one wants to predict in which direction a vehicle will turn at the intersection, this should be done before the vehicle enters the intersection). This can be defined via [*scenario.calculate_safe_action()*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-safe-actions).
- 'conforming_t0_types': If 't0_type' is not set to 'all', then it is possible to enforce additional constraints on the selection of samples for the final dataset (for 'all', one can still write stuff here, but it will be ignored). I.e., that a sample is only included in the final dataset if it also would have been included in the final dataset if a different choice for 't0_type' had been made. This allows one to compare the influence of the selection of 't0_type' on model performance while guaranteeing that the datasets still consist of the exact same scenes, with the only difference being the prediction time. Consequently, one can write from 0 up to and including 3 such different choices into the list 'conforming_t0_types' (3 possible choices: 5 overall possibilities without 'all' and the current choice for 't0_type'). For example, this was used to investigate the influence of choosing either 'crit' or 'start' for 't0_type' on *<Dataset 4>*.

It is also possible to combine multiple datasets into one. In this case, one has to put those multiple datasets into another list inside the list **Data_sets**, as was done with '<Dataset 2>' and '<Dataset 3>' in the example above. If multiple datasets are combined, then the 'max_num_agents' of the combined dataset will be the smallest number that is seen in all of the combined datasets (in this selection, 'None' would count as infinity). Meanwhile, 't0_type' and 'conforming_t0_types' are only applicable to one respective dataset, and are allowed to differ.

In the next step, one then has to set the parameters for the past and future trajectories given to the models. Like with **Data_sets** above, this will be a number of dictionaries:
```
Data_params = [{'dt': 0.2, 'num_timesteps_in': (8, 8), 'num_timesteps_out': (12, 12)},
               {'dt': 0.2, 'num_timesteps_in': (4, 8), 'num_timesteps_out': 12}] 
```
Here, the following keys have to be set:
- 'dt': This is the timestep size $\delta t$ given in seconds, i.e., the time that passes between observed trajectory positions.
- 'num_timesteps_in': These are the values for the number of input timesteps. The first number in the tuple sets the number of actual numbers of input timesteps $n_I$ given to the models. The second number meanwhile determines how many timesteps $n_{I, need}$ of data have to be available before the prediction time so that the sample is included in the dataset. This can be used to easier compare the influence of the number of input timesteps on the model performance, by keeping the dataset otherwise identical (see in the code example).
- 'num_timesteps_out': These are the values for the number of output timesteps. The first number is the maximum number of output timesteps $n_O$ that trajectory metrics are applied to. However, in datasets where behavior classifications are possible, more timesteps might be provided, as it would be ideal if the actual observed behavior were included here. The second number in the tuple meanwhile is the number of future observed timesteps $n_{O, need}$ that have to be available in the dataset so that the sample is included in the final dataset. It must be noted that this condition is only enforced if one sets the parameter 'enforce_num_timesteps_out' to *True*. Otherwise, it might happen that samples are included where the actual number of observed time steps is not only smaller than $n_{O, need}$ but also smaller than $n_{O}$.

It must be noted that both $n_{I, need}$ and $n_{O, need}$ are automatically set by the framework to be at least as large as $n_I$ and $n_O$ respectively, while values larger than 99 are set to 99.
It must be noted that if both values in the tuple are identical, setting one integer instead of a tuple with two integers is also permissible, as seen in the second line in the above code snippet, where setting *'num_timesteps_out': 12* is identical to setting *'num_timesteps_out': (12, 12)*.

After setting up the dataset and its parameters, one then has to select the splitting method.
```
Splitters = [{'Type': '<Split_method_1>', 'repetition': 0, 'test_part': 0.2},
             {'Type': '<Split_method_2>', 'repetition': [0,1,2], 'test_part': 0.2}]
```
Again, this is passed as a dictionary with three keys:
- 'Type': This is the name of the splitting method, which should be identical to one of the **.py* files in the [Splitting method Folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Splitting_methods).
- 'repetition': This is the repetition number of this split. It can either be an integer or a list of integers if the same method should be used repeatedly with shifted outputs (such as for cross-validation, or looping through locations).
- 'test_part': This is a value (between 0 and 1) that denotes the portion of the whole dataset that is used for the evaluation of the trained method.

Next, one has to select the models that are to be evaluated in this experiment.
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
This sets the number $N_{preds}$ of different predictions that are expected from each trajectory prediction model to represent the inherent stochasticity of their predictions. While this number can be set to any liking, setting it to at least 20 is advisable to be comparable with most standard metrics, which are normally evaluated on 20 predictions.

```
enforce_prediction_times = True
```
"When extracting samples, it might be possible that there is not enough data available to enforce the number of required input time steps $n_{I, need}$ at a set prediction time $t_0$ in a sample. If **enforce_prediction_times** is set to *True*, then such a sample would be discarded, while otherwise the prediction time $t_0$ is moved forward until sufficient past observations are available. It has to be noted that choosing the latter option will generally increase the size of the final dataset, but will also result in the actual definition of the prediction time $t_0$ to become diluted.

```
enforce_num_timesteps_out = False
```
When extracting samples, it might be possible that not enough data is available to allow for a sufficient number of output time steps. By setting **enforce_num_timesteps_out** to *True* such samples are dismissed, while setting it to *False* would retain them. This is also discussed [above](https://github.com/julianschumann/General-Framework/tree/main/Framework#select-modules).

```
exclude_post_crit = True
```
If the dataset allows for classification, and a method for defining the [last useful prediction time](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-safe-actions) is available, setting **exclude_post_crit** to *True* would result in all samples, where the prediction time is after this last useful time, to be discarded. If one chooses to set *False* instead, then only samples where the behavior can already be classified during past observations are discarded.

```
allow_extrapolation = True
```
In the framework, it is possible to extrapolate missing position data of certain agents during past and future observations, as some models might not be able to deal with such missing information. However, such extrapolated data might obfuscate the true human behavior. Therefore, this extrapolation can be prevented by setting **allow_extrapolation** to *False*, which might however lead to problems with some models.

```
dynamic_prediction_agents = False
```
In most situations, only a handful of agents in each scene are [set as prediction agents by the scenario](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#define-important-actors). This might not be enough to properly evaluate the model. In this case, it might be possible to include also any other agent in the scene, for which sufficient original past and future observations have been made, in the list of predicted and evaluated agents by setting **dynamic_prediction_agents** to *True*.

```
overwrite_results = False
```
It might be possible that one wants to retrain and reevaluate models, without having to delete the original files in their folder. In this case, one can set **overwrite_results** to *True*. It must however be noted that this does not redo the extraction of the training and testing set.

```
model_for_path_transform = '<Trajecotry Prediction Model>'
```
One part of the framework allows one to transform the predictions made by classification models into trajectories. This is based on a number of conditional trajectory prediction models trained only on the samples where the respective class was observed, predicting a number of trajectories proportional to the predicted probability that this class will be observed. This trajectory prediction model has to be selected here.

Finally, one has to pass these parameters to the experiment.
```
new_experiment.set_parameters(model_for_path_transform, num_samples_path_pred, 
                              enforce_num_timesteps_out, enforce_prediction_times, 
                              exclude_post_crit, allow_extrapolation, 
                              dynamic_prediction_agents, overwrite_results)
```

## Getting results
After preparing everything, the experiment can then be run with the following line:
```
new_experiment.run()     
```

After running the experiment, one can then get the results with the following command:
```
Results, Train_results, Loss = new_experiment.load_results(plot_if_possible = True,
                                                           return_train_results = True,
                                                           return_train_loss = True)
```
Here, **Results** are the results of the model on the testing set, while **Train_results** are similar, but with the results on the training set. Both are numpy arrays of the shape $\{len(Data\textunderscore sets), len(Data\textunderscore params), num\textunderscore splits, len(Models), len(Metrics)\}$. It must be noted that $len(Splitters)$ is not necessarily identical to num_splits, as by using the key 'repetition', each entry in Splitters can spawn multiple different training/testing splits.

Meanwhile, **Loss** is a similarly sized array, but instead of single float values, it contains arrays with the respective information collected during training, such as epoch loss. Due to the large variability in models, this has to be processed individually outside the framework.

The arguments *return_train_results* and *return_train_loss* respectively indicate if **Train_results** and **Loss** should be returned. Meanwhile, if *plot_is_possible = True*, then [plots](https://github.com/julianschumann/General-Framework/tree/main/Framework/Evaluation_metrics#metric-visualization) such as calibration curves for the ECE metrics are plotted as well. Those plots are saved at *../Framework/Results/<Dataset name>/Metric_figures/*

## Visualizing results
Besides getting numerical results and metric-specific plots, the framework also allows one to generate a number of other presentation contents.
For those to not throw an error upon running, it is paramount to run at least *load_results()* beforehand.
### Plotting metrics
Firstly, one can generate plots with the metrics using the following command:
```
new_experiment.draw_figure(include_only_mean = False, produce_single = False, plot_height = 2, plot_width = None, plot_x_labels = True)
```
TODO: Argument explanation

Those plots are then saved as *\*.tex* files in the folder *../Framework/Latex_files/*. While those could be compiled inside a larger document, it is advisable to compile them in a standalone format and import only the resulting *\*.pdf* document, as the compilation time might be quite long.

### Creating tables
One can also generate result tables, which present for each model, metric, dataset, and splitting method the mean value of all results of the repetitions of the splitting method.
```
write_tables(self, dataset_column = True, use_scriptsize = False, depict_std = True)
```
TODO: Argument explanation

Those tables are then also saved as *\*.tex* files in the folder *../Framework/Latex_files/*.

### Plotting trajectories
Lastly, one can also plot trajectories, true and predicted alike:
```
new_experiment.plot_paths(load_all = False)
```
Here, the first step will be to select for all the given modules one instance using console inputs (such as a dataset and model).

TODO: Argument explanation

The resulting *\*.pdf* images are then saved in *../Framework/Results/<Dataset name>/Metric_figures/*. 


