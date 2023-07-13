# Adding a new scenario to the framework
One can easily add a new scenario type to the Framework, by implementing such a scenario as a new class. Such a scenario is mainly used to define certain 
types of classifiable behavior, such as accepting or rejecting a gap in a gap acceptance scenario.

## Setting up the class

This <scenario_class> class (defined in the <scenario_class>.py file) begins in the following way:
```
class <scenario_class>():
  def __init__():
    pass

  def get_name(self = None) -> str:
    return '<Scenario name>'

  ...
```
Here, the string returned by self.get_name() is mainly used for console outputs.

This class then needs to possess a number of prenamed methods via which it interacts with the rest of the framework.

## Define classifiable behaviors
Firstly, the class needs to define the potential behaviors that might be observed in this scenario:
```
  def give_classifications(self = None) -> dict:
    Class = {'<Behavior_0>': 0,
             '<Behavior_1>': 1,
             ...
            }
    return Class
```
Here, '<Behavior_i>' are the string keys, which describe certain classifiable behaviors.

In some cases, it might be the case that none of the criteria to classify a number of trajectories into a certain behavior (defined in a specific [dataset](https://github.com/julianschumann/General-Framework/tree/main/Framework/Data_sets)) are fulfilled, for which we also need to define a default behavior <Behavior_default> for such cases:
```
  def give_default_classification(self = None) -> str:
    return '<Behavior_default>'
```
It is important to note here that the returned string needs to be one of the keys of the dictionary produced by self.give_classifications().

## Define important actors
In a certain scenario, it is often the case that specific agents fulfill special roles needed for classifying certain behaviors. In the aforementioned example of gap acceptance, this would be 
the ego vehicle offering the gap and the target vehicle which has to either accept or reject the gap.
The names of those roles then have to be communicated to the Framework:
```
  def classifying_agents(self = None) -> list[str,]:
    return ['<v_1>', '<v_2>']

  def pov_agent(self = None) -> str:
    return '<v_pov>'
```
Here, the function self.classifying_agents() returns the list of those agents <v_i>. However, it needs to be noted that there might be an agent <v_pov> from whose point of view and for whose benefit a prediction is made, and for which extra information such as planned paths might be feasibly available.
In this case, such an agent <v_pov> is not included in the list produced by self.classifying_agents(), but instead returned as the output of self.pov_agent(). If, however, no such agent exists, one has to set self.pov_agent() to instead return None.

## Generalized inputs
Some scenarios require the provision of certain one-dimensional information, which might be used as additional model inputs, such as the size of the contested space in gap acceptance scenarios.
To this end, the following function is used:
```
  def can_provide_general_input(self = None) -> list[str,]:
    return ['<I_1>', '<I_2>']
```
Here, <I_i> are the string names of the information, that have to be calculated by the [dataset.calculate_additional_distances function](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). If no such information is required, one has to return None instead of an empty list.

## Define safe actions
Another important aspect of the framework is its ability to classify predictions as still useful, e.g., if an agent could still react upon those predictions and change their behavior in a safe manner. 

Going with the example of gap acceptance, for a prediction to be useful, it would have to be made at a point where the ego vehicle can still react in time to a predicted future where the target agent decides to accept the gap and move onto the contested space before the ego agent. Such a reaction of the ego agent could be to brake and come to a stop before the contested space, however, as this takes space and time, the prediction must be made at a suitably early point in time. 

Such usefulness of prediction can however be used in other contexts as well. For example, when we try to predict the turning behavior of vehicles at intersections, it would likely be useful to only consider prediction as useful and worthwhile in cases where the target vehicle, whose behavior is to be predicted, has not yet entered the intersection.

In the context of the framework, we then define the point in time, where the switch from useful to useless prediction happens as $t_{crit}$:

$$ \Delta t_{default}(t_{crit}) - \Delta t_{useful} (t_{crit}) = 0 $$

Here, $\Delta t_{default}(t)$ (included as an input of this function in the corresponding column of **t_D_class**) is the projected time until the criteria for the default behavior ([see above](#define-classifiable-behaviors)) is fulfilled, which is calculated by the [dataset.calculate_distances function](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). Meanwhile, $\Delta t_{useful}(t)$ is defined in the following scenario bound function, where it corresponds to the variable ***t_safe_action***:

```
  def calculate_safe_action(self, D_class, t_D_class, data_set, path, t, domain) -> np.ndarray:
    r'''
    Parameters
    ----------
    D_class : pandas.Series
      This is a series with :math:`N_{classes}` entries.
      For each column, it returns an array of length :math:`|t|` with the distance to the classification marker.
      The column names should correspond to the attribute self.Behaviors = list(self.scenario.give_classifications().keys()). 
      How those distances are defined depends on the scenario and behavior.
    t_D_class : pandas.Series
      A pandas series with :math:`N_{classes}` entries, whose columns, like **D_class**, correspond to the potential behavior
      in the scenario. Here, each entry is itself a numpy array of lenght :math:`|t|`, which contains the projected time until
      the trajectories of the agents can be classified as that behavior.
    data_set : Dataset class
      This is the instance of the <dataset_name> class that called this function.
    path : pandas.Series
      A pandas series with :math:`(N_{agents})` entries,
      where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
      The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
      and should include at least the relevant agents described in self.create_sample_paths.
    t : numpy.ndarray
      A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
      at which the positions in **path** were recorded.
    domain : pandas.Series
      A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
      sample. Its entries contain at least all the columns of **self.Domain_old**. 

    Returns
    -------
    t_safe_action : numpy.ndarray
      This is a :math:`|t|` dimensional boolean array.
  '''

  ...
  return t_safe_action
```

The output of this function is then the value of $\Delta t_{useful}(t)$ at each point $t$ included in the array t.






