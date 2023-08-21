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

## Checking applicability
Given the settings, it might be possible that only a certain number of unique repetitions are possible. For example, when splitting by location, the number of locations might be limited. Similarly, when splitting for cross-validation, the number of repetitions is limited by the preset size of one split. For the framework to determine if the current repetition is admissible, it is important to know what the current number of maximum repetitions is:
```    
  def repetition_number(self = None):
    r'''
    This is number of how often this method can be applied without repeating results.
        
    Returns
    -------
    max_repetition_nunmber : bool
        
    '''
    return max_repetition_nunmber
```

However, besides the maximum number of allowable repetitions, other reasons might restrict the applicability of the current splitting method. For an alternative, the splitting method might be limited to certain scenario types (see *self.data_set.scenario.get_name()*), or a split by location would require at least two locations to be findable in the dataset (see *self.Domain.location*). Such requirements can then be set in the *check_splitability_method()*:

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
  This is the number of the current split, which can be used to differentiate between different splitting methods.
  It has to be noted that only up to ten repetitions per method are feasible, which means that this attribute
  can only assume values between 0 and 9.
**self.test_split** : float
  This is a value between 0 and 1 which denotes the portion of the whole dataset that is used for the evaluation
  of the trained method.
**self.Domain** : pandas.DataFrame
  This is a pandas dataset, that is mainly used to include the metadata of each sample in the dataset, and has
  the shape :math:`\{N_{samples} {\times} (N_{info})\}`. The following keys can be used here included.
  - 'Scenario' : This is the name of the current possible datasets, such as *highD*, *rounD*, or *NuScenes*.
  - 'Path_id' : This is the id (from 0 to len(**Path_old**) - 1) of the original path data in
                **self.data_set.Datasets['Scenario'].Path_old** this current sample is coming from.
  - 'Scenario_type' : This is the name of the Scenario, such as gap acceptance.
  Further key might be included, but do not necessarily exist:
  - 'location' : This is the location at which the current data was recorded.
  
```

