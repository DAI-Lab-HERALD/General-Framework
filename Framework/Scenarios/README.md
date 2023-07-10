# Adding a new scenario to the framework
One can easily add a new scenario type to the Framework, by implementing such a scenario as a new class. Such a scenario is mainly used to define certain 
types of classifiable behavior, such as accepting or rejecting a gap in a gap acceptance scenario.

## Setting up the class

This scenario_name class (defined in the scenaro_name.py file) begins in the following way:
```
class scenario_name(data_set_template):
  def __init__():
    pass

  def get_name(self = None) -> str:
    return 'Scenario name'

  ...
```
Here, the string returned by self.get_name() is mainly used for console outputs.

This class then needs to possess a number of prenamed methods via which it interacts with the rest of the framework.

## Define classifiable behaviors
Firstly, the class needs to define the potential behaviors that might be observed in this scenario:
```
  def give_classifications(self = None) -> dict:
    Class = {'Behavior_0': 0,
             'Behavior_1': 1,
             ...
            }
    return Class
```
Here, 'Behavior_i' are the string keys, which describe certain classifiable behaviors.

In some cases, it might be the case that none of the criteria (defined in a specific [dataset](https://github.com/julianschumann/General-Framework/tree/main/Framework/Data_sets)) are fulfilled to classify a number of trajectories into a certain behavior,
we also need to define a default behavior for such cases:
```
  def give_default_classification(self = None) -> str:
    return 'behavior_default'
```
It is important to note here that the returned string needs to be one of the keys of the dictionary produced by self.give_classifications().

## Define important actors
In a certain scenario, it is often the case that specific agents fulfill special roles needed for classifying certain behaviors. In the aforementioned example of gap acceptance, this would be 
the ego vehicle offering the gap and the target vehicle which has to either accept or reject the gap.
The names of those roles then have to be communicated to the Framework:
```
  def classifying_agents(self = None) -> list[str,]:
    return ['tar']

  def pov_agent(self = None) -> str:
    return 'ego'
```
Here, the function self.classifying_agents() returns the list of those agents. However, it needs to be noted that there might be an agent from whose point of view and for whose benefit a prediction is made,
and for which extra information such as planned paths might be feasibly available.
In this case, such an agent is not included in the list produced by self.classifying_agents(), but instead returned as the output of self.pov_agent(). If, however, no such agent exists, self.pov_agent() will instead return None.



