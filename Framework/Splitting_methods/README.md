# Adding a new splitting method to the framework

One can easily add a new splitting method to the Framework, by implementing this splitting method as a new class.

## Setting up the class

This class, and by it the splitting method, interacts with the other parts of the Framework using the [splitting_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Splitting_methods/splitting_template.py). Therefore, the new <splitting_name>.py file with the class <splitting_name> (it is important that those two are the same strings so that the Framework can recognize the dataset) begins in the following way:

```
class Random_split(splitting_template):
    def get_name(self):
        names = {'print': 'Random splitting (random seed = {})'.format(self.repetition + 1),
                 'file': 'random_{}_split'.format(self.repetition),
                 'latex': r'Random split'}
        return names
```
