## Adding a new dataset to the framework
One can easily add a new Datset to the Framework, by implementing this dataset as a new class. 
This class, and by it the dataset, interact with the other parts of the framework using the [data_set_template.py](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/data_set_template.py). Therefore, the new dataset_name.py file with the class dataset_name (it is important that those two are the same strings so that the framework can recognize the datset) begins in the following way:
```
from data_set_template import data_set_template

class dataset_name(data_set_template):
  ...
```

