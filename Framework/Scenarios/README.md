# Adding a new scenario to the framework
One can easily add a new scenario type to the Framework, by implementing such a scenario as a new class. Such a scenario is mainly used to define certain 
types of classifiable behavior, such as accepting or rejecting a gap in a gap acceptance scenario.

## Setting up the class

This scenario_name class (defined in the scenaro_name.py file) begins in the following way:
```
class scenario_name(data_set_template):
  def __init__():
    pass

  ...
```

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
It is important to note here that the returned string needs to be one of the keys of the dict returned by self.give_classifications().

