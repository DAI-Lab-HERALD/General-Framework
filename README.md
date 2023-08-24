# A general framework for benchmarking human prediction methods in traffic
In the field of predicting human behavior in traffic situations, comprehensive and equitable comparisons between different methods are an important aspect. To this end, we present a generalized framework for benchmarking human behavior prediction models, which can be used to compare and test different models with a large amount of control over the actual testing environment.

## General usage description
If one wants to use the framework, one has to [set up a new simulations.py file](https://github.com/julianschumann/General-Framework/tree/main/Framework#running-a-new-experiment) with the desired experiment settings in the folder [../Framework/](https://github.com/julianschumann/General-Framework/tree/main/Framework). Once this is complete, one just has to run this file:
```
python ../Framework/simulations.py
```

It has to be noted that for the datasets already included in the model it is required to access the raw data first. The respective instructions can be found in the README.md files in the corresponding folders in [../Framework/Data_sets/](https://github.com/julianschumann/General-Framework/tree/main/Framework/Data_sets).

If one wants instead to add some new modules, one detailed explanation on how to integrate [datasets](https://github.com/julianschumann/General-Framework/blob/main/Framework/Data_sets/README.md) and [scenario types](https://github.com/julianschumann/General-Framework/blob/main/Framework/Scenarios/README.md), [splitting methods](https://github.com/julianschumann/General-Framework/blob/main/Framework/Splitting_methods/README.md), [models](https://github.com/julianschumann/General-Framework/blob/main/Framework/Models/README.md), as well as [metrics](https://github.com/julianschumann/General-Framework/blob/main/Framework/Evaluation_metrics/README.md) are available.

## Contact and collaboration
If you have questions regarding the usage of the framework or are interested in a collaboration, please contact J.F.Schumann@tudelft.nl.
