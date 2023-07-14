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
        names = {'print': '<Splitting name (repetition)>',
                 'file': '<splitting_name>',
                 'latex': r'<Splitting name>'}
        return names
```
Especially for 'print' and 'latex', it might be possible that one wants different string outputs when using this method repeatedly. In such cases, it might be useful to include the repetition number *self.repetition* in the string.

## Checking applicability
Given the settings, it might be possible that only a certain number of unique repetitions are possible. For example, when splitting by location, the number of locations might be limited. Similarly, when splitting for cross-validation, the number of repetitions is limited by the preset size of one split.

