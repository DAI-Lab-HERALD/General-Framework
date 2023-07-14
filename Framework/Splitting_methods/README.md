# Adding a new splitting method to the framework

One can easily add a new splitting method to the Framework, by implementing this splitting method as a new class.

## Setting up the class

This class, and by it the splitting method, interacts with the other parts of the Framework using the [splitting_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Splitting_methods/splitting_template.py). Therefore, the new <splitting_name>.py file with the class <splitting_name> (it is important that those two are the same strings so that the Framework can recognize the dataset) begins in the following way:

```
class <splitting_name>(splitting_template):

    def get_name(self):
        r'''
        Provides a dictionary with the different names of the dataset
            
        Returns
        -------
        names : dict
          The first key of names ('print')  will be primarily used to refer to the dataset in console outputs. 
                
          The 'file' key has to be a string with exactly **14 characters**, that does not include any folder separators 
          (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. 
                
          The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
          latex commands - such as using '$$' for math notation.
            
        '''
        names = {'print': '<Splitting name>'.format(self.repetition + 1),
                 'file': '<splitting_{}_name>'.format(self.repetition),
                 'latex': r'<Splitting name>'}
        return names
```
