# Adding a new dataset to the framework
One can easily add a new Dataset to the Framework, by implementing this dataset as a new class. 

## Setting up the class

This class, and by it the dataset, interact with the other parts of the framework using the [data_set_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). Therefore, the new dataset_name.py file with the class dataset_name (it is important that those two are the same strings so that the framework can recognize the dataset) begins in the following way:
```
from data_set_template import data_set_template

class dataset_name(data_set_template):
  def get_name(self = None):
    names = {'print': 'Dataset name',
             'file': 'dataset_nm',
             'latex': r'\emph{Dataset} name'}
    return names

  ...
```

Here, the get_name() function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the dataset in console outputs.
Meanwhile, the 'file' has to be a string with exactly 10 characters, that does not include any folder separators (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. Finally, the 'latex' key string is used in automatically generated tables and figures for latex, and can there include latex notation - such as using '$$' for math notation.  

For this class, a number of other prenamed methods then need to be defined as well, via which the dataset interacts with the rest of the framework.

## Define the data output types
...

## Setting the scenario type
...

## Creating initial paths
...

## Extracting classifiable behavior
... (This will be three functions)

## Filling empty paths
...

## Providing visulaization
...
