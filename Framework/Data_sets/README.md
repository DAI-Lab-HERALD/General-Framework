# Adding a new dataset to the framework
One can easily add a new dataset to the Framework, by implementing this dataset as a new class. 

## Setting up the class

This class, and by it the dataset, interact with the other parts of the Framework using the [data_set_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). Therefore, the new <dataset_name>.py file with the class <dataset_name> (it is important that those two are the same strings so that the Framework can recognize the dataset) begins in the following way:
```
from data_set_template import data_set_template

class <dataset_name>(data_set_template):
 def get_name(self = None):
  r'''
  Provides a dictionary with the different names of the dataset
        
  Returns
  -------
  names : dict
    The first key of names ('print')  will be primarily used to refer to the dataset in console outputs. 
            
    The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
    (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. 
            
    The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
    latex commands - such as using '$$' for math notation.
        
  '''

  names = {'print': '<Dataset name>',
           'file': '<dataname>',
           'latex': r'<\emph{Dataset} name>'}

  return names
  ...
```

Here, the get_name() function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the dataset in console outputs.
Meanwhile, the 'file' has to be a string with exactly **10 characters**, that does not include any folder separators (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. Finally, the 'latex' key string is used in automatically generated tables and figures for latex, and can include latex commands - such as using '$$' for math notation.  

For this class, a number of other prenamed methods need to be defined as well, via which the dataset interacts with the rest of the framework.

## Set the scope of available data
Firstly, one has to define what data this class will be able to create:

```
  def future_input(self = None):
    r'''
    return True: The future data of the pov agent can be used as input.
    This is especially feasible if the ego agent was controlled by an algorithm in a simulation,
    making the recorded future data similar to the ego agent's planned path at each point in time.
        
    return False: This usage of future ego agent's trajectories as model input is prevented. This is especially advisable
    if the behavior of the vehicle might include too many clues for a prediction model to use.
        
    Returns
    -------
    future_input_decision : bool
        
    '''
    return future_input_decision
```
This function defines whether for this dataset the recorded future trajectory of a designated ego agent can be used as a stand-in 
for the planned future trajectory of this agent at each point in time (return True) or not (return False).

While this information might be provided as an additional form of input for some models, it is only advised for datasets where this ego agent
is tightly controlled (for example, its motion is set by a very simple algorithm inside a simulation), as it otherwise might contain too many
clues about the actual future events about to happen.

```    
  def includes_images(self = None):
    r'''
    If True, then image data can be returned (if true, .image_id has to be a column of 
    **self.Domain_old** to indicate which of the saved images is linked to which sample).
    If False, then no image data is provided, and models have to content without them.
        
    Returns
    -------
    image_decision : bool
        
    '''
    return image_decision
```
The second function meanwhile returns the information if this dataset can provide background images of the situations that it is covering (return True) or not (return False).


## Setting the scenario
Next, the scenario <scenario_class> that the dataset covers has to be set:
```
...
from <scenario_class> import <scenario_class>

class <dataset_name>(data_set_template):

  ...

  def set_scenario(self):
    r'''
    Sets the scenario to which this dataset belongs, using an imported class.
            
    Furthermore, if general information about the dataset is needed for later steps - 
    and not only the extraction of the data from its original recorded form - those 
    can be defined here. For example, certain distance measurements such as the radius 
    of a roundabout might be needed here.
    '''

    self.scenario = <scenario_class>()

    ...
```

Here, the <scenario_class> has to be selected from those available in the [Scenario folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios), where its required properties are discussed in more detail. 

It has to be noted that this function is the only one that is called always at the initialization of the dataset class, so if your dataset requires for example any additional attributes (for example such as the radius of roundabouts at each possible location), those should be set here.

## Creating initial paths
...

## Extracting classifiable behavior
... (This will be three functions)

## Filling empty paths
...

## Providing visulaization
...
