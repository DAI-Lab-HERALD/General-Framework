# Adding a new scenario to the framework
One can easily add a new scenario type to the Framework, by implementing such a scenario as a new class. 

## Setting up the class

This class begins in the following way:
```
class dataset_name(data_set_template):
  def __init__():
    pass

  ...
```

For this class, a number of other prenamed methods then need to be defined as well, via which the dataset interacts with the rest of the framework.

