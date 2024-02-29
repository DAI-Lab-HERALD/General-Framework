# Adding a new perturbation method to the framework
One can easily add more perturbation methods to this framework, by implementing this method as a new class inside this folder.

## Setting up the class
This class, and by it the perturbation method, interact with the other parts of the Framework using the [perturbation_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Perturbation_methods/perturbation_template.py). Therefore, the new <perturbation_method>.py file with the class <perturbation_method> (it is important that those two are the same strings so that the Framework can correctly load the class) begins in the following way:
```
from perturbation_template import perturbation_template

class <perturbation_method>(perturbation_template):
  ...
```

## Extract the given perturbation method parameters
As a first step, the potential parameters given in the [*perturbation dictionary* in the simulations.py file](https://github.com/julianschumann/General-Framework/tree/main/Framework#datasets) have to be extracted and checked for completeness. For this, one has to define the function *check_and_extract_kwargs*:
```
  def check_and_extract_kwargs(self, kwargs):
    '''
    This function checks if the input dictionary is complete and extracts the required values.

    Parameters
    ----------
    kwargs : dict
      A dictionary with the required keys and values.

    Returns
    -------
    None.

    '''

    ...

    # Define the name of the perturbation method
    self.name = ...
```
As can be seen here, one important aspect of this function is to define the value **self.name**, which has to be a string and is to be used to differentiate between two different versions of thee same perturbation method applied to the same dataset. Consequently, all the necessary aspectes in the *dictionary* **kwargs** should ideally be represented in this.

## Prepare perturbation
Before the actual perturbation can be performed, some possible constraints have to be put onto the extracted data. For this, one has to first define the function *set_batch_size*, which sets the maximum number of scenarios for which a perturbation can be performed at the same time, by setting the attribute **self.batch_size**, which has to be an integer:
```
  def set_batch_size(self):
    '''
    This function sets the batch size for the perturbation method.

    It must add a attribute self.batch_size to the class.

    Returns
    -------
    None.

    '''

    ...
```

Additionally, one can also constrain the perturbation to specific datasets, by setting a number of requirements in the function *requirements*:
```
  def requirerments(self):
    '''
    This function returns the requirements for the data to be perturbed.

    It returns a dictionary called **dict**, that may contain the following keys:

    n_I_max : int (optional)
      The number of maximum input timesteps in a given scenario.
    n_I_min : int (optional)
      The number of minimum input timesteps in a given scenario.

    n_O_max : int (optional)
      The number of maximum output timesteps in a given scenario.
    n_O_min : int (optional)
      The number of minimum output timesteps in a given scenario.

    dt : float (optional)
      The time step size of the data (a deviation by 0.001 s is still allowed).
    

    Returns
    -------
    dict
      A dictionary with the required keys and values.

    '''

    ...

    return dict
```

## Perturbing the dataset
Lastly, one has to define the actual perturbation performed, using the function *perturb_batch*:
```
  def perturb_batch(self, X, Y, Pred_agents, Perturb_agents):
    '''
    This function takes a batch of data and generates perturbations.

    Parameters
    ----------
    X : np.ndarray
      This is the past observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
      If an agent is fully or at some timesteps partially not observed, then this can include np.nan values.
    Y : np.ndarray, optional
      This is the future observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
      If an agent is fully or at some timesteps partially not observed, then this can include np.nan values. 
      This value is not returned for **mode** = *'pred'*.
    T : np.ndarray
        This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
        the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
        If an agent is not observed at all, the value will instead be '0'.
    Agent_names : np.ndarray
        This is a :math:`N_{agents}` long numpy array. It includes strings with the names of the agents.

    Returns
    -------
    X_pert : np.ndarray
        This is the past perturbed data of the agents, in the form of a
        :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
        If an agent is fully or at some timesteps partially not observed, then this can include np.nan values.
    Y_pert : np.ndarray, optional
        This is the future perturbed data of the agents, in the form of a
        :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
        If an agent is fully or at some timesteps partially not observed, then this can include np.nan values. 
    

    '''
    
    ...

    return X_pert, Y_pert
```
Here, $N_{samples}$ should be equal or lesser to **self.batch_size**.
