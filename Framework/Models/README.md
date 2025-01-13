# Existing Models
In the framework, the following models are currently implemented:
| Model | Input/Output | Description | Has kwargs |
| :------------ |:---------------| :----- | :----- |
| [ADAPT](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/agent_yuan.py) | Trajectories / Trajectories | An MLP with joint endpoint refinement. | Yes |
| [AgentFormer](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/agent_yuan.py) | Trajectories / Trajectories | A transformer with split agent and time attention. | No |
| [AutoBot-Ego](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/autobot_unitraj.py) | Trajectories / Trajectories | A transformer with split agent and time attention (Joint predictions). | Yes |
| [AutoBot-Joint](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/autobot_girgis.py) | Trajectories / Trajectories | A transformer based CVAE network. | Yes |
| [Commotions](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/commotions_markkula.py) | [Class distances](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Data_sets#extracting-classifiable-behavior) / Gap acceptance classifications | Combinations of optimal planning and evidence accumulation. | Yes |
| [Deep Belief Network](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/DBN.py) | Trajectories / Classifications | A simple deep belief network, i.e., a chain of random boltzmann machines. | No |
| [Deep Belief Netowrk - General](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/DBN_general.py) | [Class distances](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Data_sets#extracting-classifiable-behavior) / Classifications | See above. | No |
| [FJMP](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/fjmp_rowe.py) | Trajectories / Trajectories | Encoder-Decoder architecture with a Directed Acyclic Interaction Graph Predictor | Yes |
| [FloMo](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/flomo_schoeller.py) | Trajectories / Trajectories | Normalizing flow | Yes |
| [Logistic Regression](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/logit_theofilatos.py) | Trajectories / Classifications | Simple logistic regression | No |
| [Logistic Regression - General](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/logit_theofilatos_general.py) | [Class distances](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Data_sets#extracting-classifiable-behavior) / Classifications | See above. | No |
| [Motion Indeterminacy Diffusion](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/mid_gu.py) | Trajectories / Trajectories | Denoising Diffusion | Yes |
| [MTR](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/mtr_unitraj.py) | Trajectories / Trajectories | Transformer with global intention clustering | Yes |
| [PECNet](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/pecnet_mangalam.py) | Trajectories / Trajectories | Goal prediction followed by socially compliant trajectory inference. | Yes |
| [Trajectron++](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/trajectron_salzmann_old.py) | Trajectories / Trajectories | LSTM based CVAE network. | Yes |
| [TrajFlow](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/trajflow_meszaros.py) | Trajectories / Trajectories | Normalizing flow inside LSTM based autoencoder. | Yes |
| [WayFormer](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Models/watformer_unitraj.py) | Trajectories / Trajectories | Attention networks | Yes |


# Adding a new model to the framework

One can easily add a new model to the Framework by implementing this model as a new class.

## Table of contents
- [Setting up the class](#setting-up-the-class)
- [Define model type](#define-model-type)
- [Model Setup](#model-setup)
- [Training the model](#training-the-model)
- [Saving and loading the model](#saving-and-loading-the-model)
- [Making predictions](#making-predictions)
- [Predicting likelihoods](#predicting-likelihoods)
- [Making predictions for adversarial attacks](#making-predictions-for-adversarial-attacks)
- [Useful helper functions](#useful-helper-functions)
  - [Trajectory prediction models.](#trajectory-prediction-models)
  - [Classification models](#classification-models)
  - [Combined models](#combined-models)
- [Model attributes](#model-attributes)


## Setting up the class

This class, and by it the model, interacts with the other parts of the Framework using the [model_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Models/model_template.py). Therefore, the new <model_name>.py file with the class <model_name> (it is important that those two are the same strings so that the Framework can recognize the model) begins in the following way:
```
from model_template import model_template

class <model_name>(model_template):
  def get_name(self = None):
    r'''
    Provides a dictionary with the different names of the model
        
    Returns
    -------
    names : dict
      The first key of names ('print')  will be primarily used to refer to the model in console outputs. 
            
      The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
      (for any operating system), as it is mostly used to indicate that certain result files belong to this model. 
            
      The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
      latex commands - such as using '$$' for math notation.
        
    '''

    names = {'print': '<Model name>',
             'file': '<modelname>',
             'latex': r'<\emph{Model} name>'}

    return names

    ...
```
Here, the *get_name()* function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the model in console outputs.
Meanwhile, the 'file' has to be a string with exactly **10 characters**, that does not include any folder separators (for any operating system), as it is mostly used to indicate that certain result files belong to this model. Finally, the 'latex' key string is used in automatically generated tables and figures for latex, and can include latex commands - such as using '$$' for math notation.  

If you have different versions of your model in which certain hyperparameters have been varied it is advised to make use of the model attribute [**self.model_kwargs**](https://github.com/julianschumann/General-Framework/tree/main/Framework/Models#model-attributes) which can be passed during the simulation setup (see [Select Modules - Models](https://github.com/julianschumann/General-Framework/tree/main/Framework#models)). Most importantly you can use this to alter the 'file' string in the above dictionary as this determines under which name a model will be saved. Keep in mind that '<modelname>' needs to remain a string of exactly **10 characters**. Note: if you wish to define default hyperparameters for your model, take into consideration that *get_name()* might be called before the model's [*setup_method()*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Models#model-setup) function. So one cannot rely on the existence of model attributes defined in the latter function.

For this class, a number of other prenamed methods need to be defined as well, via which the model interacts with the rest of the framework.

## Define model type
In the first step, it is necessary to define the type of model. One aspect might be the use of the pytorch package, which is often the most efficient on the gpu. This is covered in the following function:

```    
  def requires_torch_gpu(self = None):
    r'''
    If True, then the model will use pytorch on the gpu.
        
    Returns
    -------
    pytorch_decision : bool
        
    '''
    return pytorch_decision
```
If this function returns true, then the corresponding model device will be named **self.device**.

But equally important in the interactions with the rest of the framework is the type of output the model produces:

```    
  def get_output_type(self = None):
    r'''
    This returns a string with the output type:
    The possibilities are:
    'path_all_wo_pov' : This returns the predicted trajectories of all agents except the pov agent (defined
    in scenario), if this is for example assumed to be an AV.
    'path_all_wi_pov' : This returns the predicted trajectories of all designated agents, including the
    pov agent.
    'class' : This returns the predicted probability that some class of behavior will be observable
    in the future.
    'class_and_time' : This predicted both the aforementioned probabilities, as well as the time at which
    the behavior will become observable.
        
    Returns
    -------
    output_type : str
        
    '''
    return output_type
```
Of the two trajectory prediction methods, *'path_all_wi_pov'* is generally to be preferred, as it does not rely on the existence of a distinguished pov agent, and even if such an agent exists, predicting its future behavior is most often no problem.


Furthermore, it must also be checked if the model can be even applied to the selected dataset. For this, the method *check_trainability_method()* is needed.
If the model can be applied, it should return None, while otherwise, it should return a string that completes the sentence: "*This model can not be trained, because...*".

```    
  def check_trainability_method(self):
    r'''
    This function potentially returns reasons why the model is not applicable to the chosen scenario.
        
    Returns
    -------
    reason : str
      This str gives the reason why the model cannot be used in this instance. If the model is usable,
      return None instead.
        
    '''
    return reason
```
Potential reasons why models might not be applicable include the availability of generalized position data (see [**self.general_input_available**](https://github.com/julianschumann/General-Framework/blob/main/Framework/Models/README.md#model-attributes)) or because it is restricted to a certain scenario (see [*self.data_set.scenario_name*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#setting-up-the-class)).



## Model Setup
The function *setup_method()* is called by the framework during the initialization of the model and is therefore run both before the training of a model as well as before making predictions with a loaded model. It is therefore advisable if the model structure (such as potential neural networks) is set up at this stage and hyperparameters are defined here. These hyperparameters can be provided through the simulation setup (see [Select Modules - Models](https://github.com/julianschumann/General-Framework/tree/main/Framework#models)) in which case the desired values can be accessed from [**self.model_kwargs**](https://github.com/julianschumann/General-Framework/tree/main/Framework/Models#model-attributes) and written into the corresponding hyperparameter variables. If one wants to use some additional [helper functions](#useful-helper-functions), this might also require the setting of some additional model attributes at this place. If one wants to set random seeds for the model, it is also advantageous to do this here and this can also be passed to the function using **self.model_kwargs**.

```
  def setup_method(self):
    ... 
```
This function does not have any returns, so every variable required in the downstream task has to be saved by setting a class attribute (**self.<attribute_name>**).

## Training the model
After setting up the model, the next step is to train certain parameters of the model. This is done in the function *train_method()*. This is a very varied process over different methods, but depending on the model type, the use of [helper functions](#useful-helper-functions) can ease this process, either for trajectory prediction models or for classification models. These functions are primarily designed to allow the quick extraction and saving of training data and predictions. 

```
  def train_method(self):
    ...
    self.weights_saved = []
```

While this function again does not have any returns, it must be noted that the framework expects one to set the attribute **self.weights_saved** (list). While this theoretically might be empty, its importance is nonetheless discussed in the following section.

If the data is somehow standardized based on data, those standardization parameters should also be saved similar to model weights, as the test dataset will not be identical to the training one.
  

## Saving and loading the model
An important part of the framework is the ability to save trained models, so repeated training is not necessarily needed. Saving a model can be done in two ways.
- First, model weights can be saved in the aforementioned list **self.weights_saved**. The model weights can be saved into the list in any form such as lists, numpy arrays, pandas DataFrames, etc. It has to be stressed that even if this method is not used, **self.weights_saved** has to be defined as an empty list anyway.
- Second, one can make use of [**self.model_file**](#model-attributes) to save the model in separate files (such as .h5 files for tensorflow models or .pkl ones for pytorch ones). If the model is especially large, so that a timeout during training on a high-performance cluster with a set computation time is likely, it might be advisable to use this method to save intermediate or partial versions of the model, so that training does not have to start from scratch.


While all model parameters in **self.weights_saved** are saved in a *.npy* file, this is a binary format and therefore not easily readable. Consequently, it is also possible to save these results in a *.csv* file. In order to do this, it is important to define the following function:

```    
  def save_params_in_csv(self = None):
    r'''
    If True, then the model's parameters will be saved to a .csv file for easier access.
        
    Returns
    -------
    csv_decision : bool
        
    '''
    return csv_decision
```

However, this is likely only useful for models with few trainable parameters that are also well-named. 

One can also use the framework to save training process data, such as epoch losses. In order to do this, the following function has to be defined:
```    
  def provides_epoch_loss(self = None):
    r'''
    If True, then the model's epoch loss will be saved.
        
    Returns
    -------
    loss_decision : bool
        
    '''
    return loss_decision
```
For later analysis, saving a model's loss during training might be valuable. To this end, this is facilitated by the framework, which will save training loss, as long as the following attribute is defined in  [*train_method()*](#training-the-model):
```
**self.train_loss** : np.ndarray
  This is a two-dimensional array. While one dimension is likely reserved for the different epochs,
  the second dimension allows the saving of loss data for multiple training steps (such as when training
  multiple model parts separately).
```

As one might not always train models and then make predictions in the same session, one has to also define the ability to load the model. This is done in the function *load_method()*.

```
  def load_method(self):
    ...
    [...] = self.saved_weights
```

In this function, one should be able to use **self.model_file** and **self.weights_saved** to set the model weights to the ones of the trained model. 
This function does not return any results, but as it is an alternative to [*train_method()*](#training-the-model), it should leave the class with the same attributes, so that prediction can be performed afterward.


  

## Making predictions
After training or loading a trained model, one then has to make predictions. Here, one first has to extract the input data using the [helper functions](#useful-helper-functions), before making predictions and then saving them with the use of the corresponding helper functions (this is necessary as the framework does not expect any return by the function).

```
  def predict_method(self):
    ...
```

This function can schematically be written as
```
  def predict_method(self):
    prediction_done = False
    while not prediction_done:
      # Get batch wise data
      batch_size = ...
      X, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, prediction_done = self.provide_batch_data('pred', batch_size)

      # Use model to make predictions Pred
      Pred = ...
      
      # save predictions
      self.save_predicted_batch_data(Pred, Sample_id, Agent_id)
```


## Predicting likelihoods
Some common metrics like do not only require the predictions by the model, but also their respective (log) likelihoods according to the underlying distribution. While the framework provides an [advanced KDE-based method for their estimation](https://github.com/anna-meszaros/ROME/tree/main/rome), some models are also capable to produce those values. For the framework to interact with them, it is then necessary to define the following two functions.


```
  def provides_likelihoods(self):
    r'''
    This function returns the information of wheter the model can provide likelihoods associated
    with predicted trajectories. 

    WARNING: If the underlying probability density is not normalized, the metrics based on these
    likelihoods will become meaningless! Please keep this in mind.
        
    Returns
    -------
    can_make_prob_prediction : bool
      The boolean value depicting the ability of the model to calculate log likelihoods.
    '''
    
    ...

    return can_make_prob_prediction
      
  def calculate_log_likelihoods(self, X, Y, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id): 
    r'''
    Given an batch of input, the model calculates the predicted probability density function. This is 
    then applied to the provided ground truth trajectories.

    WARNING: If the underlying probability density is not normalized, the metrics based on these
    likelihoods will become meaningless! Please keep this in mind.
    
    Parameters
    -------
    X : np.ndarray
      This is the past observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
      values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
      *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
      the position data (*self.input_data_type = ['x', 'y', ...]*). It must be noted that *self.input_data_type*
      will always correspond to the output of the *path_data_info()* of the data_set from which this batch data
      was loaded. If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
    Y : np.ndarray
      This is the future observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{O} \times N_{data}\}` dimensional numpy array with float values. 
      If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
      This value is not returned for **mode** = *'pred'*.
    T : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
      the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
      If an agent is not observed at all, the value will instead be '0'.
    S : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
      where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
      (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
      instead be np.nan.
    C : np.ndarray 
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
      category of agent observed, where the categories are dataset specific. If the dataset does not include such
      information, it will be set to None.
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

    Pred_agents : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
      if it expected by the framework that a prediction will be made for the specific agent.
      
      If only one agent has to be predicted per sample, for **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
      be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.
    num_steps : int
      This is the number of future timesteps provided in the case of traning in expected in the case of prediction. In the 
      former case, it has the value :math:`N_{O}`.
    Sample_id : np.ndarray
      This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
      in the dataset this sample was extracted.
    Agent_id : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
      original agent in the dataset this agent was extracted (for corresponding string names see self.data_set.Agents).
        
        
    Returns
    -------
    GT_log_probs : np.ndarray
      This is a :math:`\{N_{samples} \times M_{agents} \times N_{preds}\}` dimensional numpy array. it includes float values, with 
      the model assigned log likelihoods. Here, :math:`M_{agents} = N_{agents}` if **self.predict_single_agent** = *False* (i. e., 
      the model expects marginal likelihoods), while joint likelihoods are expected for the case of **self.predict_single_agent** = 
      *True*, (resulting in :math:`M_{agents} = 1`). In the former cases, this can include np.nan values for non predicted agents.
    
    '''

    ...

    return GT_log_probs          
```

It is important to note that those metrics often assume normalized probability density functions. If this cannot be guaranteed by the model, the resulting metric comparisons might be meaningless.


## Making predictions for adversarial attacks
If [adversarial attacks](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Perturbation_methods) are created for the prediction model, it is necessary to write a function that makes predictions on batches. However, if one wants to test the model on a perturbed dataset, this function is not needed.

```
  def predict_batch_tensor(self):
    Inputs
    -------
    X_batch : torch.Tensor
      This is the past observed data of the agents, in the form of a
      :math:`\{N_{batch size} \times N_{agents} \times N_{I} \times 2\}` dimensional tensor with float values.
    T_batch : np.ndarray
      This is a :math:`\{N_{batch size} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
      the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
      If an agent is not observed at all, the value will instead be '0'.
    domain : pandas.Series
      A pandas series of lenght :math:`\{N_{batch size} \times N_{info}`, that records the metadata for the considered
      sample.
    img_train : np.ndarray
      This is a :math:`\{N_{batch size} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
      It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values
      (:math:`C = 1`) of the map image with height :math:`H` and width :math:`W`. These images are centered around
      the agent at its current position and are rotated so that the agent is driving to the right. 
      If an agent is not observed at prediction time, 0 values are returned.
    img_m_per_px_train : np.ndarray
      This is a :math:`\{N_{batch size} \times N_{agents}\}` dimensional numpy array. It includes float values that
      indicate the resolution of the provided images in *m/Px*. If only black images are provided, this will be
      np.nan.
    num_steps: int
      This specifies the number of timesteps the prediction model needs to predict.
    num_samples: int
      This specifies how many predictions :math:`N_{predictions}` the prediction model needs to make per sample 
      in the batch

    Returns
    -------
    Y_pred: tensor
      These are prediction on past observed data of the agents, in the form of a
      :math:`\{N_{batch size} \times N_{predictions} \times N_{I} \times 2\}` dimensional tensor with float values.       
```

## Useful helper functions
During the training and prediction parts of the model, the following functions are provided by the model template to allow for easier access 
to the dataset on which the model is trained.

This is both possible for classification models as well as trajectory prediction models.

### Trajectory prediction models.
To use the following helper functions, the following attributes have to be set in [*setup_method()*](#model-setup):

- **self.min_t_O_train** (int): This is the number of future timesteps that have to be observed so that a sample can be used for training.
- **self.max_t_O_train** (int): This is the maximum number of future timesteps to be processed during training. While this will not lead to whole samples being discarded, for samples with more recorded future timesteps than **self.max_t_O_train**, these additional timesteps will not be used for training.
- **self.predict_single_agent** (bool): This is *True* if the model is unable to make joint predictions and is only able to predict the future trajectory of one agent at a time. For joint prediction models, each scene is then given as a single sample, with a potentially unlimited number of agents, of which [multiple have to be predicted](https://github.com/julianschumann/General-Framework/blob/main/Framework/README.md#set-the-experiment-hyperparameters) by the model. Meanwhile, for a model making single-agent predictions, each scene is split into multiple samples, wherein each sample has a different agent that has to be predicted, and its surrounding agents are given to the model. The number of surrounding agents is identical to the 'max_num_agents' - 1 given to the [Framework](https://github.com/julianschumann/General-Framework/blob/main/Framework/README.md#select-modules).
- **self.can_use_graph** (bool): This is true if the model is able to process graph based environment data. Only if this is the case, do the following attribute should be defined.
- **self.sceneGraph_radius** (float): This is the radius around the predicted agent in which scne graph points will be given to the model. If not set, a value of *100m* is assumed.
- **self.sceneGraph_wave_length** (float): This is the minimum size between consecutive points along a map line to which the given input will be interpolated to. Higher numbers results in a less dense map graph. If not set, a value of *1m* is assumed.
- **self.can_use_map** (bool): This is true if the model is able to process image data. Only if this is the case, do the following three attributes have to be defined.
- **self.target_width** (int): This is the width $W$ of the images to be extracted from the maps.
- **self.target_height** (int): This is the height $H$ of the images to be extracted from the maps.
- **self.grayscale** (bool): This is true if the images to be returned are grayscale (number of channels $C = 1$) or colored using RGB values instead ($C = 3$).

For a given sample, the extracted images are centered around the last observed position of an agent, which is driving to the right at this moment, with **self.target_width** $W_T$ and **self.target_height** $H_T$ (the positions $(x, y)$ of vehicles are, however, still provided in the agent-independent coordinate system). Here, $s$ is the scaling factor in $m/\text{Px}$, which is already set in the dataset.

<img src="https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/Coord_fully.svg" alt="Extraction of agent centered images." width="100%">

```
  def provide_all_training_trajectories(self, return_categories = False):
    r'''
    This function provides trajectroy data an associated metadata for the training of model
    during prediction and training. It returns the whole training set (including validation set)
    in one go

    Parameters
    ----------
    return_categories : bool, optional
      This indicates if the categories (**C_train**, see below) of the samples should be returned. 
      The default is *False*.
  
    Returns
    -------
    X_train : np.ndarray
      This is the past observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
      values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
      *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
      the position data (*self.input_data_type = ['x', 'y', ...]*). 
      If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
    Y_train : np.ndarray, optional
      This is the future observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{O} \times N_{data}\}` dimensional numpy array with float values. 
      If an agent is fully or for some timesteps partially not observed, then this can include np.nan values. 
      This value is not returned for **mode** = *'pred'*.
    T_train : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
      the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
      If an agent is not observed at all, the value will instead be '0'.
    S_train : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
      where the first column (S_train[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
      (S_train[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values 
      will instead be np.nan.
    C_train : np.ndarray
      Optional return provided when return_categories = True. 
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
      category of agent observed, where the categories are dataset specific.
    img_train : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
      It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values
      (:math:`C = 1`) of the map image with height :math:`H` and width :math:`W`. These images are centered around
      the agent at its current position and are rotated so that the agent is driving to the right. 
      If an agent is not observed at prediction time, 0 values are returned.
    img_m_per_px_train : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that
      indicate the resolution of the provided images in *m/Px*. If only black images are provided, this will be
      np.nan.
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

    Pred_agents_train : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean values and is
      true if it is expected by the framework that a prediction will be made for the specific agent.
      
      If only one agent has to be predicted per sample, for **img** and **img_m_per_px**,
      :math:`N_{agents} = 1` will be returned instead, and the agent to be predicted will be the one mentioned
      first in **X** and **T**.
    Sample_id_train : np.ndarray, optional
      This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original
      sample in the dataset this sample was extracted. This value is only returned for **mode** = *'pred'*.
    Agent_id_train : np.ndarray, optional
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate
      from which original position in the dataset this agent was extracted. This value is only returned for
      **mode** = *'pred'*.
  
    '''
    
    ...
    
    return [X_train, Y_train, T_train, img_train, img_m_per_px_train, 
            Pred_agents_train, Sample_id_train, Agent_id_train]
    if return_categories:
      return [X_train, Y_train, T_train, S_train, C_train, img_train, img_m_per_px_train, graph_train,
              Pred_agents_train, Sample_id_train, Agent_id_train]
    else:
      return [X_train, Y_train, T_train, S_train, img_train, img_m_per_px_train, graph_train,
              Pred_agents_train, Sample_id_train, Agent_id_train]
```
```
  def provide_all_included_agent_types(self):
    '''
    This function allows a quick extraction of all the available agent types. Right now, the following are implemented:
    - 'P':    Pedestrian
    - 'B':    Bicycle
    - 'M':    Motorcycle
    - 'V':    All other vehicles (cars, trucks, etc.)     

    Returns
    -------
    T_all : np.ndarray
      This is a one-dimensional numpy array that includes all agent types that can be found in the given dataset.

    '''

    ...

    return T_all
```
```
def get_batch_number(self, mode, batch_size, val_split_size = 0.0, ignore_map = False):
  r'''
  This function provides trajectroy data an associated metadata for the training of model
  during prediction and training.

  Parameters
  ----------
  mode : str
    This discribes the type of data needed. *'pred'* will indicate that this is for predictions,
    while during training, *'train'* and *'val'* indicate training and validation set respectively.
  batch_size : int
    The number of samples to be selected.
  val_split_size : float, optional
    The part of the overall training set that is set aside for model validation during the
    training process. The default is *0.0*.
  ignore_map : bool, optional
    This indicates if image data is not needed, even if available in the dataset 
    and processable by the model. The default is *False*.


  Returns
  -------
  num_batches : int
    The number of batches that would be generated under the given settings
  '''

  ...

  return num_batches
```
```
def provide_batch_data(self, mode, batch_size, val_split_size = 0.0, ignore_map = False, 
                       return_categories = False, return_classifications = False):
  r'''
  This function provides trajectory data and associated metadata for the training of the model
  during prediction and training.
  
  Parameters
  ----------
  mode : str
    This describes the type of data needed. *'pred'* will indicate that this is for predictions,
    while during training, *'train'* and *'val'* indicate training and validation set respectively.
  batch_size : int
    The number of samples to be selected.
  val_split_size : float, optional
    The part of the overall training set that is set aside for model validation during the
    training process. The default is *0.0*. At the beginning of each training epoch (mode == 'train'),
    the framework will check if this value has changed and - if so - redo the splitting of the 
    training set into training and validation part. 
  ignore_map : bool, optional
    This indicates if image data is not needed, even if available in the dataset 
    and processable by the model. The default is *False*.
  ignore_graph : bool, optional
    This indicates if scene graph data is not needed, even if available in the dataset
    and processable by the model. The default is *False*.
  return_categories : bool, optional
    This indicates if the categories (**C**, see below) of the samples should be returned. 
    The default is *False*.
  return_classifications : bool, optional
    This indicates if the behavior probabilities (**P**, see below) of the samples should be returned.
    If the underlying datasets do not include behavior classifications, None is returned instead. 
    Given that this encodes future behavior, if **mode** = *'pred'*, the framework will ignore this value.
    The default is *False*.
  
  Returns
  -------
  X : np.ndarray
    This is the past observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
    values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
    *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
    the position data (*self.input_data_type = ['x', 'y', ...]*). It must be noted that *self.input_data_type*
    will always correspond to the output of the *path_data_info()* of the data_set from which this batch data
    was loaded. If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
  Y : np.ndarray, optional
    This is the future observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{O} \times N_{data}\}` dimensional numpy array with float values. 
    If an agent is fully or for some timesteps partially not observed, then this can include np.nan values. 
    This value is not returned for **mode** = *'pred'*.
  T : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
    the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
    If an agent is not observed at all, the value will instead be '0'.
  S : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It is the sizes of the agents,
    where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
    (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
    instead be np.nan.
  C : np.ndarray, optional
    Optional return provided when return_categories = True. 
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
    category of agent observed, where the categories are dataset specific.
  P : np.ndarray, optional
    Optional return provided when return_classifications = True.
    This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array. It includes float values that indicate
    the probability of the agent to belong to a specific class. The classes are dataset specific. Given that this is the 
    ground truth, each row should be a one-hot encoded vector. If the dataset does not include behavior classifications,
    None is returned instead.
  class_names : list, optional
    Optional return provided when return_classifications = True.
    This is a list of length :math:`N_{classes}` of strings that indicate the names of the classes. If the dataset does 
    not include behavior classifications, None is returned instead.
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
                            
  Pred_agents : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean values and is
    true if it is expected by the framework that a prediction will be made for the specific agent.
    
    If only one agent has to be predicted per sample, for **img** and **img_m_per_px**,
    :math:`N_{agents} = 1` will be returned instead, and the agent to be predicted will be the one mentioned
    first in **X** and **T**.
  num_steps : int
    This is the number of future timesteps provided in the case of training and expected in the case of prediction.
    In the former case, it has the value :math:`N_{O}`.
  Sample_id : np.ndarray
    This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
    in the dataset this sample was extracted.
  Agent_id : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
    original agent in the dataset this agent was extracted (for corresponding string names see self.data_set.Agents).
  epoch_done : bool
    This indicates whether one has just sampled all batches from an epoch and has to go to the next one.
  
  '''
  
  ...
  if return_categories:
    if mode == 'pred':
      return X, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
    else:
      if return_classifications:
        return X, Y, T, S, C, P, class_names, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
      else:
        return X, Y, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
  else:
    if mode == 'pred':
      return X, T, S, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
    else:
      if return_classifications:
        return X, Y, T, S, P, class_names, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done
      else:
        return X, Y, T, S, img, img_m_per_px, graph, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done

```
```
def save_predicted_batch_data(self, Pred, Sample_id, Agent_id, Pred_agents = None, Log_probs = None):
  r'''
  This function allows the saving of predicted trajectories to be later used for model evaluation. It should
  only be used during the *predict_method()* part of a trajectory prediction model.

  Parameters
  ----------
  Pred : np.ndarray
    This is the predicted future observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{O} \times 2\}` dimensional numpy array
    with float values. If an agent is not to be predicted, then this can include np.nan values.
    The required value of :math:`N_{preds}` is given in **self.num_samples_path_pred**.
  Sample_id : np.ndarray
    This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which
    original sample in the dataset this sample was extracted.
  Agent_id : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those
    indicate from which original agent in the dataset this agent was extracted.
  Pred_agents : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean values and
    is true if it is expected by the framework that a prediction will be made for the specific agent.
    
    This input does not have to be provided if the model can only predict one single agent at the same time and
    is therefore incapable of joint predictions. In this case, None is assumed as the value.
  Log_probs : np.ndarray, optional
    This is a :math:`\{N_{samples} \times M_{agents} \times N_{preds}\}` dimensional numpy array. it includes float values, 
    with the model assigned log likelihoods. Here, :math:`M_{agents} = N_{agents}` if **self.predict_single_agent** = *False* 
    (i. e., the model expects marginal likelihoods), while joint likelihoods are expected for the case of 
    **self.predict_single_agent** = *True*, (resulting in :math:`M_{agents} = 1`). In the former cases, this can include 
    np.nan values for non predicted agents.
    
    This input does not have to be provided if the model does not predict likelihoods, but is expected otherwise, i. e., if the 
    model has the function *self.provides_likelihoods()* and it is defined to return *True*.

  Returns
  -------
  None.

  '''
```

### Classification models
```
def get_classification_data(self, train = True, return_categories = False):
  r'''
  This function retuns inputs and outputs for classification models.

  Parameters
  ----------
  train : bool, optional
    This describes whether one wants to generate training or testing data. The default is True.
  return_categories : bool, optional
    This indicates if the categories (**C**, see below) of the samples should be returned. 
    The default is *False*.

  Returns
  -------
  X : np.ndarray
    This is the past observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float 
    values. Here, :math:`N_{data}` are the number of information available. This information can be found in 
    *self.input_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
    the position data (*self.input_data_type = ['x', 'y', ...]*).  If an agent is fully or for some timesteps 
    partially not observed, then this can include np.nan values.
  T : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings 
    that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
    for available types). If an agent is not observed at all, the value will instead be '0'.
  S : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents} \times 2\}` dimensional numpy array. It the sizes of the agents,
    where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
    (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values will
    instead be np.nan.
  C : np.ndarray, optional
    Optional return provided when return_categories = True. 
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes ints that indicate the
    category of agent observed, where the categories are dataset specific.
  agent_names : list
    This is a list of length :math:`N_{agents}`, where each string contains the name of a possible 
    agent.
  D : np.ndarray
    This is the generalized past observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{dist} \times N_{I}\}` dimensional numpy array with float values. 
    It is dependent on the scenario and represenst characteristic attributes of a scene such as 
    distances between vehicles. If general_input_available is False, this variable will be set to None.
  dist_names : list
    This is a list of length :math:`N_{dist}`, where each string contains the name of a possible 
    characteristic distance.
  class_names : list
    This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
    class.
  P : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array, which for each 
    class contains the probability that it was observed in the sample. As these are observed values,
    there should be exactly one value per row that is equal to 1 and the rest should be zeroes.
    It is only retuned if **train** = *True*.
  DT : np.ndarray, optional
    This is a :math:`N_{samples}` dimensional numpy array, which for each 
    class contains the time period after the prediction time at which the fullfilment of the 
    classification crieria could be observed. It is only retuned if **train** = *True*.
  

  '''
  
  ...
  
  if return_categories:
    if train:
      return X, T, S, C, agent_names, D, dist_names, class_names, P, DT
    else:
      return X, T, S, C, agent_names, D, dist_names, class_names
  else:
    if train:
      return X, T, S, agent_names, D, dist_names, class_names, P, DT
    else:
      return X, T, S, agent_names, D, dist_names, class_names
```
```
def save_predicted_classifications(self, class_names, P, DT = None):
  r'''
  This function saves the predictions made by the classification model.

  Parameters
  ----------
  class_names : list
    This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
    class.
  P : np.ndarray
    This is a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array, which for each 
    class contains the predicted probability that it was observed in the sample. As these are 
    probability values, each row should sum up to 1. 
  DT : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{classes} \times N_{q-values}\}` dimensional numpy array, 
    which for each class contains the predicted time after the prediction time at which the 
    fullfilment of the classification crieria for each value could be observed. Each such prediction 
    consists out of the quantile values (**self.t_e_quantile**) of the predicted distribution.
    The default value is None. An entry is only expected for models which are designed to make 
    these predictions.

  Returns
  -------
  None.

  '''
```
### Combined models
```
  def classify_data(self, Pred, Sample_id, Agent_id):
    r'''
    This function classifies the predicted data into the categories of the dataset. It is only useful if the dataset
    includes categories. The function will return the categories of the predicted data.

    Parameters
    ----------
    Pred : np.ndarray
      This is the predicted future observed data of the agents, in the form of a
      :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{O} \times 2\}` dimensional numpy array with float values. 
      If an agent is fully or on some timesteps partially not observed, then this can include np.nan values. 
      The required value of :math:`N_{preds}` is given in **self.num_samples_path_pred**.
    Sample_id : np.ndarray
      This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
      in the dataset this sample was extracted.
    Agent_id : np.ndarray
      This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
      original agent in the dataset this agent was extracted.

    Returns
    -------
    P_hot : np.ndarray
      This is a :math:`\{N_{samples} \times N_{preds} \times N_{classes}\}` dimensional numpy array, which for each 
      class contains the probability that it was observed in the sample. As this are observed values, 
      per row, the sum of the values will be 1 (i. e., one-hot encoded). 

      If classification is not possible (not all required agents available, or no categories in the dataset),
      then all values in the row will be np.nan.
    
    class_names : list
      This is a list of length :math:`N_{classes}`, where each string contains the name of a possible 
      class.
    '''

    ...

    return P_hot, class_names
```
## Model attributes

Meanwhile, the following model attributes set by the framework are useful or give needed requirements:
```
**self.model_kwargs** : dict
  This is a dictionary with all relevant model parameters which you may want to easily vary for example
  for the sake of model tuning. It is beneficial to use this attribute not only for setting the model
  hyperparameters but also altering the model's filename to represent specific model characteristics.
  If the model requires it, it is recommended that one includes a function in the model class which
  fills in missing fields in the provided dictionary with some default values.

**self.data_set.path** : str
  This is the global path to the framework. It might for example be useful to load model
  hyper-parameters that are saved in a different directory.

**self.dt** : float
  This value gives the size of the time steps in the selected dataset.

**self.num_timesteps_in** : int
  The number of input timesteps that the model is given.

**self.num_timesteps_out** : int
  The number of output timesteps desired. It must be however noted, that depending on other
  framework settings, fewer or more timesteps can appear in the dataset.

**self.data_set.Agents** : list
  This is a list of length :math:`N_{agents}` with the names of all the agents.

**self.has_map** : bool
  True if the chosen dataset can provide map images, False if not.

**self.num_samples_path_pred** : int
  This gives the number of predictions the model should make.

**self.model_file** : str
  This is the location at which the model should be saved. If one wants to save for example parts of
  the model separately, this should still happen in the same folder and the corresponding file should
  include the name in **self.model_file** with potential extensions, but must not be the same.

**self.t_e_quantile** : np.ndarray
  This is a one-dimensional array that says which quantile values of the predicted distribution for the
  times at which classification criteria would be fulfilled are expected to be returned.

**self.general_input_available** : bool
  This is true if generalized distance values are available, if not, it is False.

**self.model_overwrite** : bool
  This if true if the framework demands that the model is retrained from scratch, so if one customarily
  saves parts of the model separately, they would need to be retrained as well.

```
