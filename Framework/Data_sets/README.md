# Adding a new dataset to the framework
One can easily add a new dataset to the Framework, by implementing this dataset as a new class. 

## Setting up the class

This class, and by it the dataset, interact with the other parts of the Framework using the [data_set_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). Therefore, the new <dataset_name>.py file with the class <dataset_name> (it is important that those two are the same strings so that the Framework can recognize the dataset) begins in the following way:
```
from data_set_template import data_set_template

class <dataset_name>(data_set_template):
  def get_name(self = None):
    r'''
    Provides a dictionary with the different names of the dataset
        
    Returns
    -------
    names : dict
      The first key of names ('print')  will be primarily used to refer to the dataset in console outputs. 
            
      The 'file' key has to be a string with exactly **10 characters**, that does not include any folder separators 
      (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. 
            
      The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
      latex commands - such as using '$$' for math notation.
        
    '''

    names = {'print': '<Dataset name>',
             'file': '<dataname>',
             'latex': r'<\emph{Dataset} name>'}

    return names

    ...
```

Here, the *get_name()* function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the dataset in console outputs.
Meanwhile, the 'file' has to be a string with exactly **10 characters**, that does not include any folder separators (for any operating system), as it is mostly used to indicate that certain result files belong to this dataset. Finally, the 'latex' key string is used in automatically generated tables and figures for latex, and can include latex commands - such as using '$$' for math notation.  

For this class, a number of other prenamed methods need to be defined as well, via which the dataset interacts with the rest of the framework.

## Set the scope of available data
Firstly, one has to define what data this class will be able to create:

```
  def future_input(self = None):
    r'''
    return True: The future data of the pov agent can be used as input.
    This is especially feasible if the ego agent was controlled by an algorithm in a simulation,
    making the recorded future data similar to the ego agent's planned path at each point in time.
        
    return False: This usage of future ego agent's trajectories as model input is prevented. This is
    especially advisable if the behavior of the vehicle might include too many clues for a prediction
    model to use.
        
    Returns
    -------
    future_input_decision : bool
        
    '''
    return future_input_decision
```
This function defines whether for this dataset the recorded future trajectory of a designated ego agent can be used as a stand-in 
for the planned future trajectory of this agent at each point in time (return True) or not (return False).

While this information might be provided as an additional form of input for some models, it is only advised for datasets where this ego agent
is tightly controlled (for example, its motion is set by a very simple algorithm inside a simulation), as it otherwise might contain too many
clues about the actual future events about to happen.

```    
  def includes_images(self = None):
    r'''
    If True, then image data can be returned (if true, .image_id has to be a column of 
    **self.Domain_old** to indicate which of the saved images is linked to which sample).
    If False, then no image data is provided.
        
    Returns
    -------
    image_decision : bool
        
    '''
    return image_decision
```
The second function meanwhile returns the information if this dataset can provide background images of the situations that it is covering (return True) or not (return False).


## Setting the scenario
Next, the scenario <scenario_class> that the dataset covers has to be set:
```
...
from <scenario_class> import <scenario_class>

class <dataset_name>(data_set_template):

  ...

  def set_scenario(self):
    r'''
    Sets the scenario to which this dataset belongs, using an imported class.
            
    Furthermore, if general information about the dataset is needed for later steps - 
    and not only the extraction of the data from its original recorded form - those 
    can be defined here. For example, certain distance measurements such as the radius 
    of a roundabout might be needed here.
    '''

    self.scenario = <scenario_class>()

    ...
```

Here, the <scenario_class> has to be selected from those available in the [Scenario folder](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios), where its required properties are discussed in more detail. 

It has to be noted that this function is the only one that is called always at the initialization of the dataset class, so if your dataset requires any additional attributes (for example the radius of roundabouts at each possible location), those should be set here.

## Importing the raw data
The most important part of the dataset module is to provide access to training and testing data for models in a unified format. Consequently, transforming the raw data into this unified format is of paramount importance, which is done by the function self.create_path_samples().

```
  def create_path_samples(self):
    r'''
    Loads the original trajectory data from wherever it is saved.
    Then, this function has to extract for each potential test case in the data set 
    some required information. This information has to be collected in the following attributes, 
    which do not have to be returned, but only defined in this function:

    **self.Path** : pandas.DataFrame          
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. 
      Here, each row :math:`i` represents one recorded sample, while each column includes the 
      trajectory of an agent (as a numpy array of shape :math:`\{\vert T_i \vert{\times} 2\}`. 
      It has to be noted that :math:`N_{agents}` is the maximum number of agents considered in one
      sample over all recorded samples. If the number of agents in a sample is lower than :math:`N_{agents}`
      the subsequent corresponding fields of the missing agents are filled with np.nan instead of the
      aforementioned numpy array. It is also possible that positional data for an agent is only available
      at parts of the required time points, in which cases, the missing positions should be filled up with
      (np.nan, np.nan).
                
      The name of each column corresponds to the name of the corresponding
      agent whose trajectory is covered. The name of such agents is relevant, as the selected scenario requires 
      some agents with a specific name to be present. The names of those relevant agents can be found in 
      self.scenario.pov_agent() and self.scenario.classifying_agents().
                
    **self.Type_old** : pandas.DataFrame  
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. Its column names are
      identical to the column names of **self.Path**. Each corresponding entry contains the type of the agent
      whose path is recorded at the same location in *self.Path**.
  
      Currently, four types of agents are implemented:
        - 'V': Vehicles like cars and trucks
        - 'M': Motorcycles
        - 'B': Bicycles
        - 'P': Pedestrians
            
    **self.T** : np.ndarray
      A numpy array (dtype = object) of length :math:`N_{samples}`. Each row :math:`i` contains the timepoints 
      of the data collected in **self.Path** in a tensor of length :math:`\vert T_i \vert`.
                
    **self.Domain_old** : pandas.DataFrame  
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} (N_{info})\}`.
      In this DataFrame, one can collect any ancillary metadata that might be needed
      in the future. An example might be the location at which a sample was recorded
      or the subject id involved, which might be needed later to construct the training
      and testing set. Another useful idea might be to record the place in the raw data the sample
      originated from, as might be used later to extract surrounding agents from this raw data.
                
    **self.num_samples** : int
      A scalar integer value, which gives the number of samples :math:`N_{samples}`. It should be noted 
      that :math:`self.num_Samples = len(self.Path) = len(self.T) = len(self.Domain_old) = N_{samples}`.
        
    It might be possible that the selected dataset can provide images. In this case, it is
    paramount that **self.Domain_old** contains a column named 'image_id', so that images can
    be assigned to each sample with only having to save one image for each location instead for
    each sample:

    **self.Images** : pandas.DataFrame  
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} 2\}`.
      In the first column, named 'Image', the images for each location are saved. It is paramount that the 
      indices of this DataFrame are equivalent to the unique values found in **self.Domain_old**['image_id']. 
      The entry for each cell of the column meanwhile should be a numpy array of dtype np.uint8 and shape
      :math:`\{H {\times} W \times 3\}`. It is assumed that a position :math:`(0,0)` that is recorded
      in the trajectories in **self.Path** corresponds to the upper left corner (that is self.Images.*.Image[0, 0])
      of the image, while the position :math:`(s \cdot W, - s \cdot H)` would be the lower right corner
      (that is self.Images.*.Image[H - 1, W - 1]).
                
      If this is not the case, due to some translation and subsequent rotation 
      of the recorded positions, the corresponding information has to be recorded in columns of 
      **self.Domain_old**, with the column names 'x_center' and 'y_center'. When we take a trajectory saved in
      self.Path_old, then rotate it counterclockwise by 'rot_angle', and then add 'x_center' and
      'y_center' to the rotated trajectory, the resulting trajectory would then be in the described coordinate
      system where (0,0) would be on the upper left corner of the corresponding image.

      Given a value :math:`\Delta x` for 'x_center' and :math:`\Delta y` for 'y_center',
      and :math:`\theta` for 'rot_angle', the relationship between a position :math:`(x,y)` in the trajectory
      included in **self.Path_old** and the same point :math:`(\hat{x}, \hat{y})` in the coordinate system aligned
      with the image would be the following.
      
      .. math::
          \begin{pmatrix} \hat{x} \\ \hat{y} \end{pmatrix} = \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix} +
          \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} 
          \begin{pmatrix} x \\ y \end{pmatrix}

      NOTE: if any one of the values 'x_center', 'y_center', or 'rot_angle' is set, then the other two values also 
      have to be set. Even if no translation or rotation is needed, these values should be set to zero. Otherwise,
      a missing attribute error will be thrown.

      The second column of the DataFrame, named 'Target_MeterPerPx', contains a scalar float value
      :math:`s` that gives us the scaling of the images in the unit :math:`m /` Px. 

    '''

    ...
```
If one uses a coordinate system unaligned with the image with height $H$ and width $W$ (in pixels), then these are the correlations between a position $(x,y)$ in a trajectory included in **self.Path_old** and the same point $(\hat{x}, \hat{y})$ in the coordinate system aligned with the image. Here, $\Delta x$ is a value in meters from **self.Domain_old.x_center**, $\Delta y$ from **self.Domain_old.y_center**, and $\theta$ (in radians) from **self.Domain_old.rot_angle**.

<img src="https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/Coord_small.svg" alt="Image alinged coordinate system" width="75%">

While the format of the original raw dataset might vary widely, the unified format required by the framework is clearly defined. Consequently, there are likely a wide array of possible solution to achieve this transformation function, which might involve some pre-processing tasks and saving of intermediate location as well.
It has to be noted that the framework will check if the provided attributes actually fulfill the format defined above, and will try to give feedback if this is not the case.

## Extracting classifiable behavior
While not necessary in all datasets, in some, being able to classify the interactions of certain vehicles might be an important aspect. To this end, the following three functions are needed, which are able to provide certain aspects of allowing a model to predict those classifications for a single sample. The first function is used to provide the approximated distance one or more agents need to travel to reach the criteria after which their trajectories are classified as a certain behavior. Once these distances turn negative, this is used by the framework to determine that a classification is now possible.
```
  def calculate_distance(self, path, t, domain):
    r'''
    If the chosen scenario contains a number of possible behaviors, as which recorded or
    predicted trajectories might be classified, this function calculates the abridged distance of the 
    relevant agents in a scenario toward fulfilling each of the possible classification criteria. 
    If the classification criterium is not yet fulfilled, those distances are positive, while them being negative 
    means that a certain behavior has occurred.
        
    This function extracts these distances for one specific sample.

    Parameters
    ----------
    path : pandas.Series
      A pandas series with :math:`(N_{agents})` entries,
      where each entry is itself a numpy array of shape :math:`\{N_{preds} \times |t| \times 2 \}`.
      The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
      and should include at least the relevant agents described in self.create_sample_paths.
    t : numpy.ndarray
      A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
      at which the positions in **path** were recorded.
    domain : pandas.Series
      A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
      sample. Its entries contain at least all the columns of **self.Domain_old**. 

    Returns
    -------
    Dist : pandas.Series
      This is a series with :math:`N_{classes}` entries.
      For each column, it returns an array of shape :math:`\{N_{preds} \times |t|\}` with the distance to
      the classification marker. The column names should correspond to the attribute
      self.Behaviors = list(self.scenario.give_classifications().keys()).
    '''

    ...

    return Dist
```
It has to be pointed out that in this case, the input **path** provides trajectories in a three-dimensional array, to account for the possibility that in a certain scenario,
multiple predictions ($N_{preds}$) are made at once. This will not be the case for the following two functions, as they are only applied to ground truth trajectories and not predictions, so one should be careful when indexing arrays.

The second function then is needed for cases, where some classifications are only possible under distinct conditions. For example, on a highway in a country with right-hand traffic, a target agent would only be in a position to cut in front of an ego agent trying to overtake them if they are one lane further to the right of them. The following function can then indicate for each time point in a given set of trajectories if those conditions are actually fulfilled.
```
  def evaluate_scenario(self, path, D_class, domain):
    r'''
    It might be that the given scenario requires all agents to be in certain positions so that
    it can be considered that the scenario is indeed there. This function makes that evaluation.

    This function tests this for one specific sample.

    Parameters
    ----------
    path : pandas.Series
      A pandas series with :math:`(N_{agents})` entries,
      where each entry is itself a numpy array of lenght :math:`\{|t| \times 2 \}`.
      The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
      and should include at least the relevant agents described in self.create_sample_paths.
    D_class : pandas.Series
      This is a series with :math:`N_{classes}` entries.
      For each column, it returns an array of length :math:`|t|` with the distance to
      the classification marker. The column names should correspond to the attribute
      self.Behaviors = list(self.scenario.give_classifications().keys()).
    domain : pandas.Series
      A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
      sample. Its entries contain at least all the columns of **self.Domain_old**. 

    Returns
    -------
    in_position : numpy.ndarray
      This is a :math:`|t|` dimensional boolean array, which is true if all agents are
      in a position where the scenario is valid.
    '''

    ...

    return in_position
```

Finally, one has to consider the possibilities that there are classification models that cannot handle the trajectories provided. While those models can of course use the distances to the classification criteria as a possible input, further information extracted from the trajectories might be useful. The required outputs are provided by in *self.scenario.can_provide_general_input()*, and if missing, will cause an error. In the aforementioned example of highway lane changing, such information could be the distance at which another vehicle is following the ego vehicle, as the size of the following gap might influence the decision of the target vehicle whether to merge now or later. 
```
  def calculate_additional_distances(self, path, t, domain):
    r'''
    Some models cannot deal with trajectory data and instead are constrained to quasi-one-dimensional
    data. While here the main data are the distances to the classification created in self.calculate_distance(),
    this might be incomplete to fully describe the current situation. Consequently, it might be necessary
    to extract further characteristic distances.

    This function extracts these distances for one specific sample.

    Parameters
    ----------
    path : pandas.Series
      A pandas series with :math:`(N_{agents})` entries,
      where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
      The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
      and should include at least the relevant agents described in self.create_sample_paths.
    t : numpy.ndarray
      A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
      at which the positions in **path** were recorded.
    domain : pandas.Series
      A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
      sample. Its entries contain at least all the columns of **self.Domain_old**. 

    Returns
    -------
    Dist_other : pandas.Series
      This is a :math:`N_{other dist}` dimensional Series.
      For each column, it returns an array of lenght :math:`|t|` with the distance to the classification marker.

      These columns should contain the minimum required distances set in self.scenario.can_provide_general_input().
      If self.can_provide_general_input() == False, one should return None instead.
    '''

    ...

    return Dist_other
```

If the specific dataset does not provide for classifications, then those functions can be set to *return None*. The only exception here is evaluate_scenario(), which can still be used to return a boolean array if one wants to exclude certain possible situations from the dataset.


## Filling empty paths
As mentioned in a previous [chapter](#importing-the-raw-data), it is possible that some of the trajectories provided to the models might contain np.nan positions, which for some models might be problematic. Consequently, we might need a function that for a given sample fill up those missing position with extrapolated data (although those might be deleted later if such a setting is chosen). However, it has to be noted that not doing this will not cause an error by the framework, but will possibly limit the number of models available or require the adjustment of such models.

Besides filling in missing positions in provided trajectories, it might also be possible to add further agents to the situation, which might not have been included in the scenario yet. However, it should be made sure that those agents are actually present in the scene during the timesteps used as model input. Generally, in a carefully thought-out dataset class, all agents should have been added during *self.create_path_samples()* already. It must also be noted that those agents added here will not be included in the set of agents used for evaluating predictions. Consequently, it should only be agents here that either are already included in another sample in such a role that their future can be predicted, or agent classes that are not really desired to be predicted.

```
  def fill_empty_path(self, path, t, domain, agent_types):
    r'''
    After extracting the trajectories of a sample at the given input and output timesteps, it might
    be possible that an agent's trajectory is only partially recorded over this timespan, resulting
    in the position values being np.nan at those missing time points. The main cause here is likely
    that the agent is outside the area over which its position could be recorded. 

    However, some models might be unable to deal with such missing data. Consequently, it is required
    to fill those missing positions with extrapolated data. 

    Additionally, it might be possible that **path** does not contain all the agents which were present
    during the *input* timesteps. As those might still be influencing the future behavior of the agents
    already included in  **path**, they can be added here. Consequntly,
    math:`N_{agents, full} \geq N_{agents}` will be the case.
        
    Parameters
    ----------
    path : pandas.Series
      A pandas series with :math:`(N_{agents})` entries,
      where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
      The columns should correspond to the columns in **self.Path** created in self.create_path_samples()
      and should include at least the relevant agents described in self.create_sample_paths.
    t : numpy.ndarray
      A one-dimensionl numpy array (len(t)  :math:`= |t|`). It contains the corresponding timesteps 
      at which the positions in **path** were recorded.
    domain : pandas.Series
      A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
      sample. Its entries contain at least all the columns of **self.Domain_old**. 
    agent_types : pandas.Series 
      A pandas series with :math:`(N_{agents})` entries, that records the type of the agents for
      the considered sample. The columns should correspond to the columns in **self.Type_old** created
      in self.create_path_samples() and should include at least the relevant agents described in
      self.create_sample_paths. Consequently, the column names are identical to those of **path**.

    Returns
    -------
    path_full : pandas.Series
      A pandas series with :math:`(N_{agents, full})` entries,
      where each entry is itself a numpy array of shape :math:`\{|t| \times 2 \}`.
      All columns of **path** should be included here. For those agents where trajectories are recorded, those
      trajectories should also no longer contain np.nan as a position value.
    agent_types_full : pandas.Series 
      A pandas series with :math:`(N_{agents, full})` entries, that records the type of the agents for the
      considered sample. The columns should correspond to the columns in **path_full** and include all columns
      of **agent_types**.
    '''

    ...

    return path_full, agent_types_full
```


## Providing visulaization
One important aspect of the framework is its ability to visualize ground truth and predicted trajectories. While it would be possible to just display these, putting them on a background might help with better understanding and easier analysis. While for datasets with images, those images can be taken as a background, it must be noted that those might not always be available. 

However, providing at least some orientation in forms such as lane markers might still be beneficial. Consequently, the following function allows one to add solid and dashed lines to such depiction, on top of which the trajectories are then plotted.
```
  def provide_map_drawing(self, domain):
    r'''
    For the visualization feature of the framework, a background picture is desirable. However, such an
    image might not be available, or it might be beneficial to highlight certain features. In that case,
    one can provide additional lines (either dashed or solid) to be drawn (if needed on top of images),
    that allow greater context for the depicted scenario.
        
    Parameters
    ----------
    domain : pandas.Series
      A pandas series of lenght :math:`N_{info}`, that records the metadata for the considered
      sample. Its entries contain at least all the columns of **self.Domain_old**. 

    Returns
    -------
    lines_solid : list
      This is a list of numpy arrays, where each numpy array represents on line to be drawn. 
      Each array is of the shape :math:`\{N_{points} \times 2 \}`, where the positions of the 
      points are given in the same coordinate frame as the positions in **self.Path**. The lines
      connecting those points will be solid.
            
    lines_dashed : list
      This is identical in its form to **lines_solid**, however, the depicted points will be 
      connected by dashed lines.
            
  '''

  ...

  return lines_solid, lines_dashed

```
