# Running up a new experiment
To set up an experiment with the modules in the framework, one has to create or change a *simulations.py* file. This files consists out of a number of steps, which will be explained in more detail in the following sections.

## Set up an experiment
In a first step, we have to set up the experiemnt we want to run:
```
from experiment import Experiment

# Draw latex figure
Experiment_name = '<Experiment name>'
new_experiment = Experiment(Experiment_name)
```
The only reuired input here is **Experiment_name**, which will be included in the final saved files for created tables and figures. So if it is desired that those are not overwritten, it should be unique for each different experiment
