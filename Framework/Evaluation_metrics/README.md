# Adding a new evaluation metric to the framework

One can easily add a new evaluation metric to the Framework, by implementing this metric as a new class.

## Setting up the class

This class, and by it the evaluation metric, interacts with the other parts of the Framework using the [evaluation_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Evaluation_metrics/evaluation_template.py). Therefore, the new <eval_metric_name>.py file with the class <eval_metric_name> (it is important that those two are the same strings so that the Framework can recognize the evaluation metric) begins in the following way:

```
from evaluation_template import evaluation_template

class <eval_metric_name>(evaluation_template):
  def get_name(self = None):
    r'''
    Provides a dictionary with the different names of the evaluation metric.
        
    Returns
    -------
    names : dict
      The first key of names ('print')  will be primarily used to refer to the evaluation metric in console outputs. 
            
      The 'file' key has to be a string that does not include any folder separators 
      (for any operating system), as it is mostly used to indicate that certain result files belong to this evaluation metric. 
            
      The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
      latex commands - such as using '$$' for math notation.
        
    '''

    names = {'print': '<Eval metric name>',
             'file': '<Eval_metric_name>',
             'latex': r'<\emph{Eval metric} name>'}

    return names
```

Here, the get_name() function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the evaluation metric in console outputs. Meanwhile, the 'file' has to be a string that does not include any folder separators (for any operating system), as it is mostly used to indicate that certain result files belong to this evaluation metric. Finally, the 'latex' key string is used in automatically generated tables and figures for latex, and can include latex commands - such as using '$$' for math notation.

For this class, a number of other prenamed methods need to be defined as well, via which the evaluation metric interacts with the rest of the framework.

## Setting up the evaluation metric

Next, if the evaluation metric requires certain values to be calculated prior to being applied for evaluation this can be done with:

```
def setup_method(self):
    # Will do any preparation the method might require, like calculating
    # weights.
    # creates:
        # self.weights_saved -  The weights that were created for this metric,
        #                       will be in the form of a list
```
This is primarily intended for computationally heavy calculations of values which can be shared across the evaluation of different models. If no such calculations are required, one can simply write `pass` within the function definition. 
Connected to this, one also has to define whether this type of pre-processing is even required for the desired evaluation metric. This is done in:

```
def requires_preprocessing(self):
    # Returns a boolean output, True if preprocesing of true output
    # data for the calculation of weights is required, which might be 
    # avoided in repeated cases
```

## Defining the evaluation metric

The most important part of the evaluation module, is the definition of how the metric is calculated. This is done within:

```
def evaluate_prediction_method(self):
    # Takes true outputs and corresponding predictions to calculate some metric to evaluate a model.

    Returns
    -------
    results : list

    return results 
```

## Provide contextual (?) metric information
```
def get_output_type(self = None):
    # Should return 'class', 'class_and_time', 'path_tar', 'path_all'
    raise AttributeError('Has to be overridden in actual metric class')
```

In some cases there can be large differences in the metric results between different models. To improve the readability of graphs it can be helpful to represent the metric results on a log scale rather than a linear scale. In order to determine whether the log_scale is to be used, one has to define this in:
```
def is_log_scale(self = None):
    # Should return 'False' or 'True'
    raise AttributeError('Has to be overridden in actual metric class')
```

Every metric has an optimal value which indicates that a model's prediction is identical to the true output. In order to be able to indicate if a model performs better than other models on a particular evaluation metric it is important to know whether a lower or higher value of the metric's result is desired. This is defined in the following function:
```
def get_opt_goal(self = None):
    # Should return 'minimize' or 'maximize'
```
