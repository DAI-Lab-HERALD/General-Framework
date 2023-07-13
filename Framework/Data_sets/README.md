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

Here, the get_name() function creates a dictionary with three keys, the value of each must be a string. The first key is 'print', which will be primarily used to refer to the dataset in console outputs.
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
        
    return False: This usage of future ego agent's trajectories as model input is prevented. This is especially advisable
    if the behavior of the vehicle might include too many clues for a prediction model to use.
        
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
    If False, then no image data is provided, and models have to content without them.
        
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

It has to be noted that this function is the only one that is called always at the initialization of the dataset class, so if your dataset requires for example any additional attributes (for example such as the radius of roundabouts at each possible location), those should be set here.

## Importing the raw data
The most important part of the dataset module is to provide access to training and testing data for models in a unified format. Consequently, transforming the raw data into this unified format is of paramount importance, which is done by the function self.create_path_samples().

```
  def create_path_samples(self):
    r'''
    Loads the original path data in its recorded form from wherever it is saved.
    Then, this function has to extract for each potential test case in the data set 
    some required information. This information has to be collected in the following attributes, 
    which do not have to be returned, but only defined in this function:

    **self.Path**          
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. 
      Here, each row :math:`i` represents one recorded sample, while each column includes the 
      trajectory of an agent (as a numpy array of shape :math:`\{\vert T_i \vert{\times} 2\}`. 
      It has to be noted that :math:`N_{agents}` is the maximum number of agents considered in one
      sample overall recorded samples. If the number of agents in a sample is lower than :math:`N_{agents}`
      the subsequent corresponding fields of the missing agents are filled with np.nan instead of the
      aforementioned numpy array. It is also possible that positional data for an agent is only available
      at parts of the required time points, in which cases, the missing positions should be filled up with
      (np.nan, np.nan).
                
      The name of each column corresponds to the name of the corresponding
      agent whose trajectory is covered. The name of such agents is relevant, as the selected scenario requires 
      some agents with a specific name to be present. The names of those relevant agents can be found in 
      self.scenario.pov_agent() and self.scenario.classifying_agents().
                
    **self.Type_old**
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} N_{agents}\}`. Its column names are
      identical to the column names of **self.Path**. Each corresponding entry contains the type of the agent
      whose path is recorded at the same location in *self.Path**. For example, a "V" stands for a vehicle,
      while a "P" stands for a pedestrian.
            
    **self.T**
      A numpy array (dtype = object) of length :math:`N_{samples}`. Each row :math:`i` contains the timepoints 
      of the data collected in **self.Path** in a tensor of length :math:`\vert T_i \vert`.
                
    **self.Domain_old**
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} (N_{info})\}`.
      In this DataFrame, one can collect any ancillary metadata that might be needed
      in the future. An example might be the location at which a sample was recorded
      or the subject id involved, which might be needed later to construct the training
      and testing set. Another useful idea might be to record the place in the raw data the sample
      originated from, as might be used later to extract surrounding agents from this raw data.
                
    **self.num_samples**
      A scalar integer value, which gives the number of samples :math:`N_{samples}`. It should be noted 
      that :math:`self.num_Samples = len(self.Path) = len(self.T) = len(self.Domain_old) = N_{samples}`.
        
    It might be possible that the selected dataset can provide images. In this case, it is
    paramount that **self.Domain_old** entails a column named 'image_id', so that images can
    be assigned to each sample with only having ot save on image for each location instead for
    each sample:

    **self.Images**
      A pandas DataFrame of dimensionality :math:`\{N_{samples} {\times} 2\}`.
      In the first column, named 'Image', the images for each location are saved. It is paramount that the 
      indices of this DataFrame are equivalent to the unique values found in **self.Domain_old**['image_id']. 
      The entry for each cell of the column meanwhile should be a numpy array of dtype np.uint8 and shape
      :math:`\{H {\times} W \times 3\}`. All images need to be of the same size. If this is not the case, zero
      padding to the right and bottom should be used to obtain the desired dimensions. It is assumed that a 
      position (0,0) that is recorded in the trajectories in **self.Path** corresponds to the upper left
      corner of the image. 
                
      If this is not the case, due to some translation and subsequent rotation 
      of the recoded positions, the corresponding information has to be recorded in columns of 
      **self.Domain_old**, where the columns 'x_center' and 'y_center' record the position in the 
      original coordinate system at which the current origin (0,0) now lies, and 'rot_angle' is 
      the angle by which the coordinate system was rotated afterward clockwise.

      The second column of the DataFrame, named 'Target_MeterPerPx', contains a scalar float value
      that gives us the scaling of the images in the unit :math:`m /` Px. 

    '''

    ...
```

While the format of the original raw dataset might vary widely, the unified format required by the framework is clearly defined.
It has to be noted that the framework will check if the provided attributes actually fulfill the format defined above, and will try to give feedback if this is not the case.

## Extracting classifiable behavior
... (This will be three functions)

## Filling empty paths
...

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

```
