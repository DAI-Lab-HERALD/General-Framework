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
      (for any operating system), as it is mostly used to indicate that certain result files belong to this
      evaluation metric. 
            
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
    # Will do any preparation the method might require, like calculating weights.
    # creates:
    # self.weights_saved -  The weights that were created for this metric,
    #                       will be in the form of a list
```
This is primarily intended for computationally heavy calculations of values which can be shared across the evaluation of different models. If no such calculations are required, one can simply write `pass` within the function definition. 
Connected to this, one must also define whether this pre-processing type is even required for the desired evaluation metric. This is done in:

```
  def requires_preprocessing(self):
    r''' 
    If True, then the model will use pytorch on the gpu.
        
    Returns
    -------
    preprocessing_decision : bool
        
    '''
    return preprocessing_decision
```

## Define metric type
In the next step, it is necessary to define the type of metric. Most important in the interactions with the rest of the framework is the type of output produced by the model that the metric evaluates:

```    
  def get_output_type(self = None):
    r'''
    This returns a string with the output type:
    The possibilities are:
    'path_all_wo_pov' : Predicted trajectories of all agents except the pov agent (defined
    in scenario), if this is for example assumed to be an AV.
    'path_all_wi_pov' : Predicted trajectories of all designated agents, including the pov agent.
    'class' : Predicted probability that some class of behavior will be observable in the future.
    'class_and_time' : Both the aforementioned probabilities, as well as the time at which the
    behavior will become observable.
        
    Returns
    -------
    output_type : str
        
    '''
    return output_type
```
Of the two trajectory prediction methods, *'path_all_wi_pov'* is generally to be preferred, as it does not rely on the existence of a distinguished pov agent, and even if such an agent exists, predicting its future behavior is most often no problem. However, in most cases, switching between those two possibilities should not require any more effort than changing the output of this specific function.


Furthermore, it must also be checked if the metric can be even applied to the selected dataset. For this, the method *check_applicability()* is needed.
If the model can be applied, it should return None, while otherwise, it should return a string that completes the sentence: "*This metric can not be applied, because...*".

```    
  def check_applicability(self):
    r'''
    This function potentially returns reasons why the metric is not applicable to the chosen scenario.
        
    Returns
    -------
    reason : str
      This str gives the reason why the model cannot be used in this instance. If the model is usable,
      return None instead.
        
    '''
    return reason
```
A potential reason why the metric might not be applicable could be the restriction to a certain scenario (see *self.data_set.scenario.get_name()*) or dataset (see *self.data_set.get_name()*).


Finally, when comparing metrics, it is important to know if a superior model would minimize or maximize them. For this, the method *get_opt_goal()* is used:
```    
  def get_opt_goal(self):
    r'''
    This function says if better metric values are smaller or larger.
        
    Returns
    -------
    metric_goal : str
      This str gives the goal of comparisions. If **metric_goal** = 'minimze', than a lower metric
      value indicates a better value, why 'maximize' is the other way around. As the framework
      only checks if the return here is 'minimize', any other string would be treated as if it
      was set to 'maximize'.
        
    '''
    return metric_goal
```


## Defining the evaluation metric

The most important part of the evaluation module is the definition of how the metric is calculated. This is done within the *evaluate_prediction_method()*.

```
  def evaluate_prediction_method(self):
    r'''
    # Takes true outputs and corresponding predictions to calculate some metric to evaluate a model.

    Returns
    -------
    results : list
      This is a list with at least one entry. The first entry must be a scalar, which allows the comparison
      of different models according to that metric. Afterwards, any information can be saved which might
      be useful for later visualization of the results, if this is so desired.
    '''

    ...
    
    return results 
```

Here, the [helper functions](#useful-helper-functions) can be used to load the true and predicted data, with the choice depending on the type of metric.


## Metric Visualization
The framework includes some tools for visualizing metrics. One is the plotting of the final metric values. As in some cases, there can be large differences in the metric results between different models, it can be helpful to represent the metric results on a log scale rather than a linear scale for easier readability. In order to determine whether the log_scale is to be used, one has to define this in:
```
  def is_log_scale(self = None):
    r''' 
    If True, then the metric values will be plotted on a logarithmic y-axis.
        
    Returns
    -------
    log_scale_decision : bool
        
    '''
    return log_scale_decision
```

Additionally, some metrics such as the various ECE metrics have additional information that can be plotted. In order to inform the framework if such information exists and can be plotted, one needs to define it through: 

```
  def allows_plot(self):
    r''' 
    If True, then the metric values will be plotted on a logarithmic y-axis.
        
    Returns
    -------
    plot_decision : bool
        
    '''
    return plot_decision
```

If this function *allow_plots()* returns *True*, then one also needs to define a function that actually executes this plotting.

```
  def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
    '''
    This function creates the final plot.
    
    This function is cycled over all included models, so they can be combined
    in one figure. However, it is also possible to save a figure for each model,
    if so desired. In that case, a new instanc of fig and ax should be created and
    filled instead of the ones passed as parameters of this functions, as they are
    shared between all models.
    
    If only one figure is created over all models, this function should end with:
        
    if save:
      ax.legend() # Depending on if this is desired or not
      fig.show()
      fig.savefig(test_file, bbox_inches='tight')  

    Parameters
    ----------
    results : list
      This is the list produced by self.evaluate_prediction_method().
    test_file : str
      This is the location at which the combined figure of all models can be
      saved (it ends with '.pdf'). If one saves a result for each separate model, 
      one should adjust the filename to indicate the actual model.
    fig : matplotlib.pyplot.Figure
      This is the overall figure that is shared between all models.
    ax : matplotlib.pyplot.Axes
      This is the overall axis that is shared between all models.
    save : bool, optional
      This is the trigger that indicates if one currently is plotting the last
      model, which should indicate that the figure should now be saved. The default is False.
    model_class : Framework_Model, optional
      The model for which the current results were calculated. The default is None.

    Returns
    -------
    None.

    '''
```

## Useful helper functions
For the evaluation, the following functions are provided by the evaluation template to allow for easier access 
to the true and predicted data.

```
def get_true_and_predicted_paths(self, num_preds = None, return_types = False):
  '''
  This returns the true and predicted trajectories.
  
  Parameters
  ----------
  num_preds : int, optional
    The number :math:`N_{preds}` of different predictions used. The default is None,
    in which case all available predictions are used.
  return_types : bool, optional
    Decides if agent types are returned as well. The default is False.
  
  Returns
  -------
  Path_true : np.ndarray
    This is the true observed trajectory of the agents, in the form of a
    :math:`\{N_{samples} \times 1 \times N_{agents} \times N_{O} \times 2\}` dimensional numpy 
    array with float values. If an agent is fully or some timesteps partially not observed, 
    then this can include np.nan values.
  Path_pred : np.ndarray
    This is the predicted furure trajectories of the agents, in the form of a
    :math:`\{N_{samples} \times N_{preds} \times N_{agents} \times N_{O} \times 2\}` dimensional 
    numpy array with float values. If an agent is fully or some timesteps partially not observed, 
    then this can include np.nan values.
  Pred_steps : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents} \times N_{O}\}` dimensional numpy array with 
    boolean values. It indicates for each agent and timestep if the prediction should influence
    the final metric result.
  T : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings 
    that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
    for available types). If an agent is not observed at all, the value will instead be '0'.
    It is only returned if **return_types** is *True*.
  
  '''
  
  ...
  
  if return_types:
      return Path_true, Path_pred, Pred_step, Types     
  else:
      return Path_true, Path_pred, Pred_step

```
```
def get_true_and_predicted_class_probabilities(self):
  '''
  This returns the true and predicted classification probabilities.

  Returns
  -------
  P_true : np.ndarray
    This is the true probabilities with which one will observe a class, in the form of a
    :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array with float values. 
    One value per row will be one, whil ethe others will be zero.
  P_pred : np.ndarray
    This is the predicted probabilities with which one will observe a class, in the form of 
    a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array with float values. 
    The sum in each row will be 1.
  Class_names : list
    The list with :math:`N_{classes}` entries contains the names of the aforementioned
    behaviors.

  '''
  
  ...
  
  return P_true, P_pred, Class_names
```
```
def get_true_and_predicted_class_times(self):
  '''
  This returns the true and predicted classification timepoints, at which a certain
  behavior can be first classified.

  Returns
  -------
  T_true : np.ndarray
    This is the true time points at which one will observe a class, in the form of a
    :math:`\{N_{samples} \times N_{classes} \times 1\}` dimensional numpy array with float 
    values. One value per row will be given (actual observed class), while the others 
    will be np.nan.
  T_pred : np.ndarray
    This is the predicted time points at which one will observe a class, in the form of a
    :math:`\{N_{samples} \times N_{classes} \times N_{quantiles}\}` dimensional numpy array 
    with float values. Along the last dimension, the time values correspond to the quantile 
    values of the predicted distribution of the time points. The quantile values can be found
    in **self.t_e_quantile**.
  Class_names : list
      The list with :math:`N_{classes}` entries contains the names of the aforementioned
      behaviors.

  '''

  ...
  
  return T_true, T_pred, Class_names
```
