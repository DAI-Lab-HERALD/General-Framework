# Adding a new perturbation method to the framework
One can easily add more perturbation methods to this framework, by implementing this method as a new class inside this folder.

## Table of contents
- [Setting up the class](#setting-up-the-class)
- [Extract the given perturbation method parameters](#extract-the-given-perturbation-method-parameters)
- [Prepare perturbation](#prepare-perturbation)
- [Perturbing the dataset](#perturbing-the-dataset)



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
  def perturb_batch(self, X, Y, T, S, C, img, img_m_per_px, graph, Pred_agents, Agent_names, Domain):
    '''
    This function takes a batch of data and generates perturbations.

    Parameters
    ----------
    X : np.ndarray
      This is the past observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float 
      values. If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
    Y : np.ndarray
      This is the future observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
      If an agent is fully or for some timesteps partially not observed, then this can include np.nan values. 
    T : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
      the type of agent observed. If an agent is not observed at all, the value will instead be '0'.
    S : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It is the sizes of the agents,
      where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
      (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
      instead be np.nan.
    C : np.ndarray
      Optional return provided when return_categories = True. 
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
      category of agent observed, where the categories are dataset specific.
    img : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
      It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
      of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
      at its current position, and are rotated so that the agent is right now driving to the right. 
      If an agent is not observed at prediction time, 0 values are returned.
    img_m_per_px : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
      the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
    graph : np.ndarray
      This is a numpy array with length :math:`N_{samples}`, where the entries are pandas.Series with the following entries:
        num_nodes         - number of nodes in the scene graph.

        lane_idcs         - indices of the lane segments in the scene graph; array of length :math:`num_{nodes}`
                            with *lane_idcs.max()* :math:`= num_{lanes} - 1`.

        pre_pairs         - array with shape :math:`\{num_{lane pre} {\times} 2\}` lane_idcs pairs where the
                            first value of the pair is the source lane index and the second value is source's
                            predecessor lane index.

        suc_pairs         - array with shape :math:`\{num_{lane suc} {\times} 2\}` lane_idcs pairs where the
                            first value of the pair is the source lane index and the second value is source's
                            successor lane index.

        left_pairs        - array with shape :math:`\{num_{lane left} {\times} 2\}` lane_idcs pairs where the
                            first value of the pair is the source lane index and the second value is source's
                            left neighbor lane index.

        right_pairs       - array with shape :math:`\{num_{lane right} {\times} 2\}` lane_idcs pairs where the
                            first value of the pair is the source lane index and the second value is source's
                            right neighbor lane index.

        left_boundaries   - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                            :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                            of points needed to describe the left boundary in travel direction of the current lane.
                            Here, :math:`num_{nodes,l} = ` *(lane_idcs == l).sum()*. 
                                  
        right_boundaries  - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                            :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                            of points needed to describe the right boundary in travel direction of the current lane.

        centerlines       - array with length :math:`num_{lanes}`, whose elements are arrays with shape
                            :math:`\{num_{nodes,l} + 1 {\times} 2\}`, where :math:`num_{nodes,l} + 1` is the number
                            of points needed to describe the middle between the left and right boundary in travel
                            direction of the current lane.

        lane_type         - an array with length :math:`num_{lanes}`, whose elements are tuples with the length :math:`2`,
                            where the first element is a string that is either *'VEHILCE'*, '*BIKE*', or '*BUS*', and the 
                            second entry is a boolean, which is true if the lane segment is part of an intersection.

        pre               - predecessor nodes of each node in the scene graph;
                            list of dictionaries where the length of the list is equal to the number of scales for the neighbor
                            dilation as per the implementation in LaneGCN. 
                            Each dictionary contains the keys 'u' and 'v', where 'u' is the *node index* of the source node and
                            'v' is the index of the target node giving edges pointing from a given source node 'u' to its
                            predecessor.

        suc               - successor nodes of each node in the scene graph;
                            list of dictionaries where the length of the list is equal to the number of scales for the neighbor
                            dilation as per the implementation in LaneGCN. 
                            Each dictionary contains the keys 'u' and 'v', where 'u' is the *node index* of the source node and
                            'v' is the index of the target node giving edges pointing from a given source node 'u' to its
                            successor.

        left              - left neighbor nodes of each node in the scene graph;
                            list containing a dictionary with the keys 'u' and 'v', where 'u' is the *node index* of the source 
                            node and 'v' is the index of the target node giving edges pointing from a given source node 'u' to 
                            its left neighbor.

        right             - right neighbor nodes of each node in the scene graph;
                            list containing a dictionary with the keys 'u' and 'v', where 'u' is the *node index* of the source 
                            node and 'v' is the index of the target node giving edges pointing from a given source node 'u' to 
                            its right neighbor.
                              
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
