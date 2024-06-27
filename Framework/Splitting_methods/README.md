# existing splitting methods
In the framework, the following splitting methids are currently implemented:
...

# Adding a new splitting method to the framework

One can easily add a new splitting method to the Framework, by implementing this splitting method as a new class.

## Setting up the class

This class, and by it the splitting method, interacts with the other parts of the Framework using the [splitting_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Splitting_methods/splitting_template.py). Therefore, the new <splitting_name>.py file with the class <splitting_name> (it is essential that those two are the same strings so that the Framework can recognize the splitting method) begins in the following way:

```
class <splitting_name>(splitting_template):

def get_name(self):
  r'''
  Provides a dictionary with the different names of the splitting method
      
  Returns
  -------
  names : dict
    The first key of names ('print')  will be primarily used to refer to the splitting method in console outputs. 
          
    The 'file' key has to be a string with exactly **12 characters**, that does not include any folder separators 
    (for any operating system), as it is mostly used to indicate that certain result files belong to this splitting
    method. 
          
    The 'latex' key string is used in automatically generated tables and figures for latex and can include 
    latex commands - such as using '$$' for math notation.
      
  '''
  names = {'print': '<Splitting name (self.repetition)>',
           'file': '<splitting_name>',
           'latex': r'<Splitting name>'}
  return names
```
Especially for 'print' and 'latex', it might be possible that one wants different string outputs when using this method repeatedly. In such cases, it might be useful to include the repetition number **self.repetition** in the string.

## Using strings for splitting datasets
It might now be possible for certain splitting method, that repetitions can be passed as strings. The following function then is used to tell the framework if such an input can be processed or not.
```
  def can_process_str_repetition(self = None):
    r'''
    This returns the decision, whether the test set used can be indicated by
    strings or not.

    Returns
    -------
    processing_decision : int

    '''

    return processing_decision
```

If the directly above function returns *True*, then it is also necessary to define a following function *transform_str_to_number*, which allows one to map a string to the orignial integer number that would have resultet in the same test samples to be selected.
```
  def tranform_str_to_number(self, rep_str):
    '''
    This function tranforms a given string into the corresponding
    repetition number.

    Parameters
    ----------
    rep_str : str
      This is the given str to be tranformed.

    Returns
    -------
    rep_numbers : list
      This are the corresponding numbers that could be given as the
      repetition to the splitting method and result in the same
      outcome. The returned list should only return integers. If there
      is no corresponding number, then an empty list is returned instead.

    '''

    ...

    return rep_numbers
```

## Checking applicability
Given the settings, it might be possible that only a certain number of unique repetitions are possible. For example, when splitting by location, the number of locations might be limited. Similarly, when splitting for cross-validation, the number of repetitions is limited by the preset size of one split. For the framework to determine if the current repetition is admissible, it is important to know what the current number of maximum repetitions is:
```    
  def repetition_number(self = None):
    r'''
    This is number of how often this method can be applied without repeating results.
        
    Returns
    -------
    max_repetition_nunmber : int
        
    '''
    return max_repetition_nunmber
```

If a given repetition number would be too large, this repetition would be ignored. However, there might be further reasons to exclude a given repetition. For example, the splitting method might be limited to certain scenario types (see [*self.data_set.scenario_name*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#setting-up-the-class)), or a split by location would require at least two locations to be findable in the dataset (see [*self.Domain.location*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Splitting_methods#splitting-method-attributes)). Such requirements can then be set in the *check_splitability_method()*:

```    
  def check_splitability_method(self):
    r'''
    This function potentially returns reasons why the splitting method is not applicable to
    the chosen scenario.
        
    Returns
    -------
    reason : str
      This str gives the reason why the splitting method cannot be used in this instance. If the
      model is usable, return None instead.
        
    '''
    return reason
```

## Splitting the dataset
The main part of the splitting is then the function *split_data_method()*, where the actual splitting of the datasets is performed:
```
  def split_data_method(self):
    '''
    In this function, the actual splitting into training and testing sets
    is performed. While it is commonly the case that each sample is only once
    either filed into training or testing set, this is not hardcoded, and other
    options are permissible.

    Returns
    -------
    Train_index : np.ndarray
        This is a one-dimensional array with int values of a length that is 
        smaller than :math:`N_{samples}` <= len(**self.Domain**). It shows which
        of the original samples are part of the training set. 
    Test_index : np.ndarray
        This is a one-dimensional array with int values of a length that is 
        smaller than :math:`N_{samples}` <= len(**self.Domain**). It shows which
        of the original samples are part of the testing set, on which trained
        models are evaluated and compared.

    '''
    
    ...
    
    return Train_index, Test_index
```



## Splitting method attributes
The splitting template provides a number of attributes that might be useful in this part.
```
**self.repetition** : int
  This is the number of the current split, which can be used to differentiate between different splits of the
  same splitting method. As one can use multiple separate parts in the training set, this will be a list of
  numbers (if one single part is used, then the list will have a length of one). While it is possible to define
  repetitions as a string in the simulation.py file, this will only be numbers, using the function
  *tranform_str_to_number()* to get the specific number.
**self.test_split** : float
  This is a value between 0 and 1 which denotes the portion of the whole dataset that is used for the evaluation
  of the trained method.
**self.Domain** : pandas.DataFrame
  This is a pandas dataset, that is mainly used to include the metadata of each sample in the dataset, and has
  the shape :math:`\{N_{samples} {\times} (N_{info})\}`. The following keys can be used.
  - 'Scenario' : This is the name of the current possible datasets, such as *highD*, *rounD*, or *NuScenes*.
  - 'Path_id' : This is the id (from 0 to len(**Path_old**) - 1) of the original path data in
                **self.data_set.Datasets['Scenario'].Path_old** that this current sample is coming from.
  - 'Scenario_type' : This is the name of the Scenario, such as gap acceptance.
  Further keys that might be included, but do not necessarily have to exist:
  - 'location' : This is the location at which the current data was recorded.
  - 'perturbation': A boolean value, indicating if the corresponding trajectory comes from a perturbed dataset
                    or not.
  
```

