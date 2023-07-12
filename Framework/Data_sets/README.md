# Adding a new dataset to the framework
One can easily add a new dataset to the Framework, by implementing this dataset as a new class. 

## Setting up the class

This class, and by it the dataset, interact with the other parts of the Framework using the [data_set_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). Therefore, the new dataset_name.py file with the class dataset_name (it is important that those two are the same strings so that the Framework can recognize the dataset) begins in the following way:
```
from data_set_template import data_set_template

class dataset_name(data_set_template):
  def get_name(self = None) -> dict:
    names = {'print': 'Dataset name',
             'file': 'dataset_nm',
             'latex': r'\emph{Dataset} name'}
    return names

  ...
```

Here, the get_name() function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the dataset in console outputs.
Meanwhile, the 'file' has to be a string with exactly 10 characters, that does not include any folder separators (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. Finally, the 'latex' key string is used in automatically generated tables and figures for latex, and can there include latex commands - such as using '$$' for math notation.  

For this class, a number of other prenamed methods then need to be defined as well, via which the dataset interacts with the rest of the framework.

## Define the data types
Firstly, one has to define what type of data types this class will be able to create:

```
  def future_input(self = None) -> bool:
    return False
```
This function defines whether for this dataset the recorded future trajectory of a designated ego agent can be used as a stand-in 
for the planned future trajectory of this agent at each point in time (return True) or not (return False).

While this information might be provided as an additional form of input for some models, it is only advised for datasets where this ego agent
is tightly controlled (for example, its motion is set by a very simple algorithm inside a simulation), as it otherwise might contain too many
clues about the actual future events about to happen.

```    
  def includes_images(self = None) -> bool:
    return True
```
The second datatype function meanwhile returns the information if this dataset can provide background images of the situations that it is covering (return True) or not (return False).


## Setting the scenario type
Next, the scenario that the dataset covers has to be set:
```
...
from scenario_class import scenario_class

class dataset_name(data_set_template):

  ...

  def set_scenario(self):
    self.scenario = scenario_class()
```

Here, the scenario_class has to be selected from those available in the [Scenario folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios), where its required properties are discussed in more detail. 

It has to be noted that this function is the only one that is called always at the initialization of the dataset class, so if your dataset requires for example any additional attributes, those should be set here.

## Creating initial paths
...

## Extracting classifiable behavior
... (This will be three functions)

## Filling empty paths
...

## Providing visulaization
...
