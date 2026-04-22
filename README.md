# STEP: Structured Training and Evaluation Platform for benchmarking human behavior models.
In the field of predicting human behavior in traffic situations, comprehensive and equitable comparisons between different methods are an important aspect. To this end, we present STEP, a generalized framework for benchmarking human behavior prediction models, which can be used to compare and test different models with a large amount of control over the actual testing environment.

## General usage description
If one wants to use the framework, one has to [set up a new simulations.py file](https://github.com/julianschumann/General-Framework/tree/main/Framework#running-a-new-experiment) with the desired experiment settings in the folder [../Framework/](https://github.com/julianschumann/General-Framework/tree/main/Framework). Once this is complete, one just has to run this file:
```
python ../Framework/simulations.py
```

It has to be noted that if one wants to use one of the datasets already included in the framework, it is required to access the respective raw data first. The instructions can be found in the README.md files in the corresponding folders in [../Framework/Data_sets/](https://github.com/julianschumann/General-Framework/tree/main/Framework/Data_sets).

If one wants to instead add new modules, detailed explanations on how to integrate [datasets](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Data_sets#adding-a-new-dataset-to-the-framework) and [scenario types](https://github.com/julianschumann/General-Framework/blob/main/Framework/Scenarios/README.md), [splitting methods](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Splitting_methods#adding-a-new-splitting-method-to-the-framework), [models](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Models#adding-a-new-model-to-the-framework), as well as [metrics](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework/Evaluation_metrics#adding-a-new-evaluation-metric-to-the-framework) are available.

## Contact and collaboration
If you have questions regarding the usage of the framework or are interested in a collaboration, please contact J.F.Schumann@tudelft.nl.

## Citation
When using this framework for your work, please cite the [paper](https://arxiv.org/abs/2509.14801). The simulation files and resulting paper results can be found here as well:
- The file [STEP_simulations.zip](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/STEP_simulations.zip) has to be unzipped in the [Framework](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework) folder.
- The file [STEP_results.zip](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/STEP_results.zip) has to be unzipped in the [Framework](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework) folder. Due to size constraints (Extracted data and predictions can approach over 100GB), we only share the resulting metrics reported in the paper.
- The file [STEP_analysis.zip](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/STEP_analysis.zip) has to be unzipped in the [Framework](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main/Framework) folder.

Thank you very much. 
