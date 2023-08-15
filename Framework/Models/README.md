# Adding a new model to the framework

One can easily add a new model to the Framework by implementing this model as a new class.

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

But equally important in the interactions with the rest of the framework is the type of input and output the model requires

*TODO Later: Decribe everythin here get_input_type(), get_output_type()*

## Training process data

One can also use the framework to save training process data, such as epoch losses or final model parameters. Here, two functions are important to define:

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
While all model parameters, if so chosen, are saved in a *.npy* file, this is a binary format and therefore not easily readable. Consequently, it is possible to also possible to save these results in a *.csv* file. However, this is likely only useful for models with few trainable parameters that are also well-named. 

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

## Model Setup


## Training the model


## Saving and loading the model


## Making predictions


## Useful helper functions
During the training and prediction parts of the model, the following functions are provided by the model template to allow for easier access 
to the dataset on which the model is trained:
```
  def provide_all_included_agent_types(self):
    '''
    This function allows a quick generation of all the available agent types. Right now, the following are implemented:
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
def provide_batch_data(self, mode, batch_size, val_split_size = 0.0, ignore_map = False):
  r'''
  This function provides trajectory data an associated metadata for the training of the model
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
    training process. The default is *0.0*.
  ignore_map : ignore_map, optional
    This indicates if image data is not needed, even if available in the dataset 
    and processable by the model. The default is *False*.
  
  Returns
  -------
  X : np.ndarray
    This is the past observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
    If an agent is fully or some timesteps partially not observed, then this can include np.nan values.
  Y : np.ndarray, optional
    This is the future observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
    If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
    This value is not returned for **mode** = *'pred'*.
  T : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
    the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
    If an agent is not observed at all, the value will instead be np.nan.
  img : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents} \times H \times W \times C\}` dimensional numpy array. 
    It includes uint8 integer values that indicate either the RGB (:math:`C = 3`) or grayscale values (:math:`C = 1`)
    of the map image with height :math:`H` and width :math:`W`. These images are centered around the agent 
    at its current position and are rotated so that the agent is right now driving to the right. 
    If an agent is not observed at prediction time, 0 values are returned.
  img_m_per_px : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes float values that indicate
    the resolution of the provided images in *m/Px*. If only black images are provided, this will be np.nan. 
    Both for **Y**, **img** and 
  Pred_agents : np.ndarray
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean values, and is true
    if it is expected by the framework that a prediction will be made for the specific agent.
    
    If only one agent has to be predicted per sample, for **Y**, **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
    be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.
  num_steps : int
    This is the number of future timesteps provided in the case of training and expected in the case of prediction. In the 
    former case, it has the value :math:`N_{O}`.
  Sample_id : np.ndarray, optional
    This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
    in the dataset this sample was extracted. This value is only returned for **mode** = *'pred'*.
  Agent_id : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
    original agent in the dataset this agent was extracted. This value is only returned for **mode** = *'pred'*.
  epoch_done : bool
    This indicates whether one has just sampled all batches from an epoch and has to go to the next one.
  
  '''
  
  ...
  
  if mode == 'pred':
    return X, T, img, img_m_per_px, Pred_agents, num_steps, Sample_id, Agent_id, epoch_done    
  else:
    return X, Y, T, img, img_m_per_px, Pred_agents, num_steps, epoch_done

```
```
def save_predicted_batch_data(self, Pred, Sample_id, Agent_id, Pred_agents = None):
  r'''

  Parameters
  ----------
  Pred : np.ndarray
    This is the predicted future observed data of the agents, in the form of a
    :math:`\{N_{samples} \times N_{agents} \times N_{preds} \times N_{I} \times 2\}` dimensional numpy array with float values. 
    If an agent is fully or on some timesteps partially not observed, then this can include np.nan values. 
    The required value of :math:`N_{preds}` is given in **self.num_samples_path_pred**.
  Sample_id : np.ndarray, optional
    This is a :math:`N_{samples}` dimensional numpy array with integer values. Those indicate from which original sample
    in the dataset this sample was extracted.
  Agent_id : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array with integer values. Those indicate from which 
    original agent in the dataset this agent was extracted.
  Pred_agents : np.ndarray, optional
    This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes boolean value, and is true
    if it expected by the framework that a prediction will be made for the specific agent.
    
    This input does not have to be provided if the model can only predict one single agent at the same time and is therefore
    incaable of joint predictions.

  Returns
  -------
  None.

  '''
```

Meanwhile, the following model attributes set by the framework are useful or give needed reuirements:
```
**self.dt** : float
  This value gives the current size of the time steps in the model.

**self.num_timesteps_in** : int
  The number of input timesteps that the model is given.

**self.num_timesteps_put** : int
  The number of output timesteps desired. It must be however noted, that depending on other framework settings, fewer or more timesteps can appear in the dataset.

**self.num_samples_path_pred** : int
  This gives the number of predictions the model must make to adequately simulate stochasticity.

**self.model_file** : str
  This is the location at which the model should be saved. If one wants to save for example parts of the model separately, this should still happen in the
  same folder and the corresponding file should include the name in **self.model_file** with potential extensions, but must not be the same.
```
