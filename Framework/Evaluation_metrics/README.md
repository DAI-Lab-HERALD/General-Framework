# Existing evaluation metrics
In the framework, the following metrics are currently implemented (notations below the table):
| Metric | Input | Formula |
| :------------ |:---------------| :----- |
| [ADE (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE_indep.py) | Trajectories | $`{1\over{\vert P \vert\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{p \in P} \sum\limits_{t \in T_{O,i}} \sum\limits_{j = 1}^{N_{A,i}} \sqrt{\left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [ADE (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE_joint.py) | Trajectories | $`{1\over{\vert P \vert N_{S}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{p \in P} \sum\limits_{t \in T_{O,i}} \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [ADE (marginal, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE20_indep.py) | Trajectories | $`{1\over{20\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{p \in P_{20}} \sum\limits_{t \in T_{O,i}} \sum\limits_{j = 1}^{N_{A,i}} \sqrt{\left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [ADE (joint, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE20_joint.py) | Trajectories | $`{1\over{20 N_{S}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{p \in P_{20}} \sum\limits_{t \in T_{O,i}} \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [ADE (marginal, 50)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE50_indep.py) | Trajectories | $`{1\over{50\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{p \in P_{50}} \sum\limits_{t \in T_{O,i}} \sum\limits_{j = 1}^{N_{A,i}} \sqrt{\left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [ADE (joint, 50)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE50_joint.py) | Trajectories | $`{1\over{50 N_{S}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{p \in P_{50}} \sum\limits_{t \in T_{O,i}} \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [min ADE (marginal, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/minADE20_indep.py) | Trajectories | $`{1\over{\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\sum\limits_{j = 1}^{N_{A,i}} \underset{p \in P_{20}}{\min} \sum\limits_{t \in T_{O,i}}  \sqrt{\left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [min ADE (joint, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/minADE20_joint.py) | Trajectories | $`{1\over{N_{S}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }}\underset{p \in P_{20}}{\min}  \sum\limits_{t \in T_{O,i}} \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(t) - \hat{x}_{i,p,j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p,j} (t) \right)^2}`$ |
| [most likely ADE (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE_ML_indep.py) | Trajectories | $`{1 \over{\sum\limits_{i = 1}^{N_{S}} N_{A, i}}} \sum\limits_{i = 1}^{N_{S}} {1\over{N_{O,i} }} \sum\limits_{j = 1}^{N_{A,i}}  \sum\limits_{t \in T_{O,i}} \sqrt{\left( x_{i,j}(t) - \hat{x}_{i,p^*_{i,j},j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p^*_{i,j},j} (t) \right)^2} `$, with $`p^*_{i,j} = \underset{p \in P}{arg \min} P_{KDE,i,j} \left(\{\{\hat{x}_{i,p,j} (t), \hat{y}_{i,p,j} (t) \} \, \vert \; \forall\, t \in T_O\} \right)`$ |
| [most likely ADE (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ADE_ML_joint.py) | Trajectories | $`{1\over{ N_{S}}} \sum\limits_{i = 1}^{N_S}{1\over{N_{O,i} }} \sum\limits_{t \in T_{O,i}} \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(t) - \hat{x}_{i,p^*_{i},j} (t) \right)^2 + \left( y_{i,j}(t) - \hat{y}_{i,p^*_{i},j} (t) \right)^2}`$, with $`p^*_{i} = \underset{p \in P}{\text{arg} \min} P_{KDE,i} \left(\{\{\{\hat{x}_{i,p,j} (t), \hat{y}_{i,p,j} (t) \} \, \vert \; \forall\, t \in T_O\} \, \vert \; \forall \, j \} \right)`$ |
| [FDE (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE_indep.py) | Trajectories | $`{1\over{\vert P \vert\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}\sum\limits_{p \in P}  \sum\limits_{j = 1}^{N_{A,i}} \sqrt{\left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [FDE (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE_joint.py) | Trajectories | $`{1\over{\vert P \vert N_{S}}} \sum\limits_{i = 1}^{N_S}\sum\limits_{p \in P}  \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [FDE (marginal, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE20_indep.py) | Trajectories | $`{1\over{20\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}\sum\limits_{p \in P_{20}}  \sum\limits_{j = 1}^{N_{A,i}} \sqrt{\left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [FDE (joint, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE20_joint.py) | Trajectories | $`{1\over{20 N_{S}}} \sum\limits_{i = 1}^{N_S}\sum\limits_{p \in P_{20}}  \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [FDE (marginal, 50)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE50_indep.py) | Trajectories | $`{1\over{50\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S}\sum\limits_{p \in P_{50}}  \sum\limits_{j = 1}^{N_{A,i}} \sqrt{\left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [FDE (joint, 50)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE50_joint.py) | Trajectories | $`{1\over{50 N_{S}}} \sum\limits_{i = 1}^{N_S}\sum\limits_{p \in P_{50}}  \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [min FDE (marginal, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/minFDE20_indep.py) | Trajectories | $`{1\over{\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S} \sum\limits_{j = 1}^{N_{A,i}} \underset{p \in P_{20}}{\min} \sqrt{\left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [min FDE (joint, 20)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/minFDE20_joint.py) | Trajectories | $`{1\over{N_{S}}} \sum\limits_{i = 1}^{N_S}\underset{p \in P_{20}}{\min}   \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p,j} (\max T_{O,i}) \right)^2}`$ |
| [most likely FDE (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE_ML_indep.py) | Trajectories | $`{1 \over{\sum\limits_{i = 1}^{N_{S}} N_{A, i}}} \sum\limits_{i = 1}^{N_{S}}  \sum\limits_{j = 1}^{N_{A,i}}   \sqrt{\left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p^*_{i,j},j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p^*_{i,j},j} (\max T_{O,i}) \right)^2} `$, with $`p^*_{i,j} = \underset{p \in P}{arg \min} P_{KDE,i,j} \left(\{\{\hat{x}_{i,p,j} (t), \hat{y}_{i,p,j} (t) \} \, \vert \; \forall\, t \in T_O\} \right)`$ |
| [most likely FDE (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/FDE_ML_joint.py) | Trajectories | $`{1\over{ N_{S}}} \sum\limits_{i = 1}^{N_S}  \sqrt{{1\over{N_{A,i}}} \sum\limits_{j = 1}^{N_{A,i}} \left( x_{i,j}(\max T_{O,i}) - \hat{x}_{i,p^*_{i},j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - \hat{y}_{i,p^*_{i},j} (\max T_{O,i}) \right)^2}`$, with $`p^*_{i} = \underset{p \in P}{\text{arg} \min} P_{KDE,i} \left(\{\{\{\hat{x}_{i,p,j} (t), \hat{y}_{i,p,j} (t) \} \, \vert \; \forall\, t \in T_O\} \, \vert \; \forall \, j \} \right)`$ |
| [NLL (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/KDE_NLL_indep.py) | Trajectories | $`- {1 \over{\sum\limits_{i = 1}^{N_{S}} N_{A, i}}}  \sum\limits_{i = 1}^{N_{S}}  \sum\limits_{j = 1}^{N_{A,i}} \ln \left( P_{KDE,i,j} \left(\{\{x_{i,j} (t), y_{i,j} (t) \} \, \vert \; \forall\, t \in T_{O,i}\} \right)\right) `$ |
| [NLL (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/KDE_NLL_joint.py) | Trajectories | $`- {1 \over{N_{S}}}  \sum\limits_{i = 1}^{N_{S}} \ln \left( P_{KDE,i} \left(\{\{\{x_{i,j} (t), y_{i,j} (t) \} \, \vert \; \forall\, t \in T_{O,i}\} \, \vert \; \forall \, j \} \right)\right) `$ |
| [ECE (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ECE_traj_indep.py) | Trajectories | $`{1\over{201}} \sum\limits_{k = 0}^{200} \left\vert \left({1\over{\sum\limits_{i = 1}^{N_{S}} N_{A, i}}} \lvert \left\{i,j \, \vert \, {1\over{\vert P\vert}} \lvert \left\{ p \in P \, \vert \, \hat{L}_{i,p,j} > L_{i,j}\right\} \rvert > {k\over{200}}  \right\} \rvert \right) + {k\over{200}} - 1 \right\vert `$, with $`L_{i,j} = P_{KDE,i,j} \left(\{\{x_{i,j} (t), y_{i,j} (t) \} \, \vert \; \forall\, t \in T_{O,i}\} \right) `$ and $` \hat{L}_{i,p,j} = P_{KDE,i,j} \left(\{\{\hat{x}_{i,p,j} (t), \hat{y}_{i,p,j} (t) \} \, \vert \; \forall\, t \in T_{O,i}\} \right) `$ |
| [ECE (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ECE_traj_joint.py) | Trajectories | $`{1\over{201}} \sum\limits_{k = 0}^{200} \left\vert \left({1\over{N_{S}}} \lvert \left\{i \, \vert \, {1\over{\vert P\vert}} \lvert \left\{ p \in P \, \vert \, \hat{L}_{i,p} > L_{i}\right\} \rvert > {k\over{200}}  \right\} \rvert \right) + {k\over{200}} - 1 \right\vert `$, with $`L_{i} = P_{KDE,i} \left(\{\{\{x_{i,j} (t), y_{i,j} (t) \} \, \vert \; \forall\, t \in T_O\} \, \vert \; \forall \, j \} \right)`$ and $` \hat{L}_{i,p} = P_{KDE,i} \left(\{\{\{\hat{x}_{i,p,j} (t), \hat{y}_{i,p,j} (t) \} \, \vert \; \forall\, t \in T_O\} \, \vert \; \forall \, j \} \right)`$ |
| [Miss rate (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/Miss_rate_indep.py) | Trajectories | $`1\over{\sum\limits_{i = 1}^{N_S} N_{A, i}}} \sum\limits_{i = 1}^{N_S} \sum\limits_{j = 1}^{N_{A,i}} \begin{cases} 1 & \sqrt{\left( x_{i,j}(\max T_{O,i}) - x_{pred,i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - y_{pred,i,p,j} (\max T_{O,i}) \right)^2} > 2 \forall p \in P \\ 0 & \text{otherwise} \end{cases}   `$ |
| [Miss rate (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/Miss_rate_joint.py) | Trajectories | $`{1\over{N_{S}}} \sum\limits_{i = 1}^{N_S} \underset{j \in \{1,..., N_{A,i} \}}{\max} \begin{cases} 1 & \sqrt{\left( x_{i,j}(\max T_{O,i}) - x_{pred,i,p,j} (\max T_{O,i}) \right)^2 + \left( y_{i,j}(\max T_{O,i}) - y_{pred,i,p,j} (\max T_{O,i}) \right)^2} > 2 \forall p \in P \\ 0 & \text{otherwise} \end{cases}   `$ |
| [Collision rate (marginal)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/Collision_rate_indep.py) | Trajectories | Needs to be rewritten, but should evaluate collision rate against GT of other agents |
| [Collision rate (joint)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/Collision_rate_joint.py) | Trajectories | Needs to be rewritten, but should evaluate collision rate against predicted futures of other agents if available, otherwise against GT |
| [Area under Curve (AUC)](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/AUC_ROC.py) | Classifications | $`{1 \over{N_{S}}} \sum\limits_{k} {\left(\sum\limits_{i = 1}^{N_{S}} r_{i,k} p_{i,k} \right) - {1\over{2}} N_k (N_k + 1)  \over {N_{S} - N_k }}`$, where likelihood rank $r_{i,k}$ fulfills $`r_{i_1,k} > r_{i_2,k} \Rightarrow {\hat{p}_{i_1,k} \over {\hat{p}_{i_1,k} + \underset{\widehat{k} \neq k}{\max} \hat{p}_{i_1,\widehat{k}}}} \geq {\hat{p}_{i_2,k} \over {\hat{p}_{i_2,k} + \underset{\widehat{k} \neq k}{\max} \hat{p}_{i_2,\widehat{k}}}}`$, with $`N_k = \sum\limits_{i = 1}^{N_{S}} p_{i,k}`$ |
| [ECE](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/ECE_class.py) | Classifications | Look at the corresponding python file for a definition. |
| [TNR-PR](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Evaluation_metrics/TNR_PR.py) | Gap acceptance classifications | $`{1 \over{\vert\{i \, \vert \, p_{i, k=accepted} = 0\}\vert}}  \sum\limits_{i} \begin{cases} 1  & p_{i, k=accepted} = 0  \land \hat{p}_{i, k=accepted} > \underset{\hat{i} \in \{\hat{i} \vert p_{\hat{i},k=accepted} = 1 \}  }{\min} \hat{p}_{\hat{i},k=accepted} \\ 0 & otherwise   \end{cases} `$ , only valid it *'t0_type': 'crit' |



Here, the following notation is used:
- $N_{S}$: Number of samples in the dataset.
- $i$: Index for those samples.
- $N_{A,i}$: The number of evaluated agents in each sample.
- $j$: Index for those agents.
- $P$: Number of predictions made for each agent in each sample.
- $P_{N_p}$: A random subset of $P$ with $|P_{N_p}| = N_p$.
- $p$: Index for those predictions.
- $T_{O,i}$: The output timesteps for trajectories, with length $N_{O,i}$.
- $t$: Value of those timesteps.
- $x_{i,j}(t)$, $y_{i,j}(t)$: Recorded positions of an agent.
- $`\hat{x}_{i,p,j}(t)`$, $`\hat{y}_{i,p,j}(t)`$: Predicted positions of an agent.
- $P_{KDE}$: A probability density function trained over $p \in P$. 
- $N_{B}$: Number of potentially classifiable behaviors.
- $k$: Index for those behaviors.
- $p_{i,k}$, $`\hat{p}_{i,k}`$ : The actual and predicted likelihood of behavior $k$ in smaple $i$.

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
    If True, then the function *setup_method* will be called and therefore has to be defined.
        
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
If the metric can be applied, it should return None, while otherwise, it should return a string that completes the sentence: "*This metric cannot be applied, because...*".

```    
  def check_applicability(self):
    r'''
    This function potentially returns reasons why the metric is not applicable to the chosen scenario.
        
    Returns
    -------
    reason : str
      This str gives the reason why the metric cannot be used in this instance. If the metric is usable,
      return None instead.
        
    '''
    return reason
```
A potential reason why the metric might not be applicable could be the restriction to a certain scenario (see [*self.data_set.scenario_name*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Scenarios#setting-up-the-class)) or dataset (see [*self.data_set.get_name()*](https://github.com/julianschumann/General-Framework/tree/main/Framework/Data_sets#setting-up-the-class)).


Furthermore, when comparing metrics, it is important to know if a superior model would minimize or maximize them. For this, the method *get_opt_goal()* is used:
```    
  def get_opt_goal(self):
    r'''
    This function says if better metric values are smaller or larger.
        
    Returns
    -------
    metric_goal : str
      This str gives the goal of comparisons. If **metric_goal** = 'minimize', then a lower metric
      value indicates a better value, while 'maximize' is the other way around. As the framework
      only checks if the return here is 'minimize', any other string would be treated as if it
      was set to 'maximize'.
        
    '''
    return metric_goal
```

Similarly, it is also the case that metric values can have hard limits on what form they can take, such as for example distance metrics, which should not be lower than 0. These limits are defined in the method *metric_boundaries()*:
```
  def metric_boundaries(self = None):
    r'''
    This function informs the framework about any boundaries that a metric value might take.

    Returns
    -------
    boundaries : list
      This is a list with two values, namely [min_value, max_value]. If one of those values
      does not exist, as the metric is unbounded, then one has to set either min_value or
      max_value (or both) equal to *None*.
      
    '''
    return boundaries
```

Lastly, when dealing with large datasets, it might not be possible to calculate a metric all in once. In those cases, the metric has to be calculated on subsets, and the combined into a final version in the end. For this combination, the framework need to know how this metric was calculated, with the different options visible in the text below
```
  def partial_calculation(self = None):
  r'''
  This function informs the framework, if and how a metric could be calculated on subsets of
  the complete test set.

  Returns
  -------
  caluculation_form : str
    This string returns the form of combining metrics calculated over subsets of the test set.
    There are three options, 'No' (The metric can only be calculated on the complete dataset,
    weighted average contains an error), 'Subgroup', (The metric is calculated by averaging 
    over subgroups of samples with identical inputs), 'Sample' (The metric is calcualted by 
    averaging over samples), 'Subgroup_pred_agents' (The metric is calculated by averaging
    over subgroups of samples with identical inputs, selecting each agent separately) 
    'Pred_agents' (The metric is calcualted by averaging over values calculated for the varying 
    number of predicted agents in each sample), 'Min' (The metric is calculated by taking the
    minimum over a certain set), and 'Max' (The metric is calculated by taking the maximum over
    a certain set)

  '''

  options = ['No', 'Subgroups', 'Sample', 'Subgroup_pred_agents', 'Pred_agents', 'Min', 'Max']
  i_option = ...

  return options[i_option] 
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

If the return list from **self.evaluate_prediction_method()** has more than one entry, then defining *partial_calculation* is not enough.
Instead, one also has to define the following function *combine_results*, which will take in multiple such lists of results, as well as the associated weights, and combine
them into a single list:
```
  def combine_results(self, result_lists, weights):
    r'''
    This function combines partial results.

    Parameters
    ----------
    result_lists : list
      A list of lists, which correpond to multiple outputs of *evaluate_prediction_method()*.

    weights : list
      A list of the same length as result_lists, which contains the weights based on the method in
      *self.partial_calculation()*, which might be useful.


    Returns
    -------
    results : list
      This is a list with more than one entry. The first entry must be a scalar, which allows the comparison
      of different models according to that metric. Afterwards, any information can be saved which might
      be useful for later visualization of the results, if this is so desired.
    

    '''

    ...
    
    return results
```
It must be noted that if based only on the partial metric result, one would have *self.partial_calculation() = 'No'*, saving intermediate results as additional ouputs of *self.evaluate_prediction_method()* and then using *combine_results* can allow for accurate calculation of metrics over large datasets without leading to extensive memory issues.


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
    If True, then plots such as calibration curves will be created.
        
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
    if so desired. In that case, a new instance of fig and ax should be created and
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
def get_true_and_predicted_paths(self, num_preds = None, return_types = False,
                                 exclude_late_timesteps = True):
  '''
  This returns the true and predicted trajectories.
  
  Parameters
  ----------
  num_preds : int, optional
    The number :math:`N_{preds}` of different predictions used. The default is None,
    in which case all available predictions are used.
  return_types : bool, optional
    Decides if agent types are returned as well. The default is False.
  exclude_late_timesteps : bool, optional
    Decides if predicted timesteps after the set prediction horizon should be excluded. 
    The default is True.
  
  Returns
  -------
  Path_true : np.ndarray
    This is the true observed trajectory of the agents, in the form of a
    :math:`\{N_S \times 1 \times N_{A} \times N_{O} \times 2\}` dimensional numpy 
    array with float values. If an agent is fully or some timesteps partially not observed, 
    then this can include np.nan values.
  Path_pred : np.ndarray
    This is the predicted furure trajectories of the agents, in the form of a
    :math:`\{N_S \times N_{preds} \times N_{A} \times N_{O} \times 2\}` dimensional 
    numpy array with float values. If an agent is fully or for some timesteps partially not predicted, 
    then this can include np.nan values.
  Pred_steps : np.ndarray
    This is a :math:`\{N_S \times N_{A} \times N_{O}\}` dimensional numpy array with 
    boolean values. It indicates for each agent and timestep if the prediction should influence
    the final metric result.
  Types : np.ndarray, optional
    This is a :math:`\{N_S \times N_{A}\}` dimensional numpy array. It includes strings 
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
def get_other_agents_paths(self):
  '''
  This returns the true observed trajectories of all agents that are not the
  predicted agents.
  
  Parameters
  ----------
  return_types : bool, optional
    Decides if agent types are returned as well. The default is False.

  Returns
  -------
  Path_other : np.ndarray
    This is the true observed trajectory of the agents, in the form of a
    :math:`\{N_S \times 1 \times N_{A_other} \times N_{O} \times 2\}` dimensional numpy 
    array with float values. If an agent is fully or or some timesteps partially not observed, 
    then this can include np.nan values.
  Types : np.ndarray, optional
    This is a :math:`\{N_S \times N_{A_other}\}` dimensional numpy array. It includes strings 
    that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
    for available types). If an agent is not observed at all, the value will instead be '0'.
    It is only returned if **return_types** is *True*.

  '''
  
  ...
  
  if return_types:
    return Path_other, Types     
  else:
    return Path_other
```
```
def get_KDE_probabilities(self, joint_agents = True):
  '''
  This return the probabilities asigned to trajectories according to 
  a Gaussian KDE method.

  Parameters
  ----------
  joint_agents : bool, optional
    This says if the probabilities for the predicted trajectories
    are to be calculated for all agents jointly. If this is the case,
    then the array dimension :math:`N_{A}` mentioned in the
    returns is 1. The default is True.

  Returns
  -------
  KDE_pred_log_prob_true : np.ndarray
    This is a :math:`\{N_S \times 1 \times N_{A}\}`
    array that includes the log probabilities for the true observations according
    to the KDE model trained on the predicted trajectories.
  KDE_pred_log_prob_pred : np.ndarray
    This is a :math:`\{N_S \times N_{preds} \times N_{A}\}`
    array that includes the log probabilities for the predicted trajectories 
    according to the KDE model trained on the predicted trajectories.

  '''
  
  ...
  
  return KDE_pred_log_prob_true, KDE_pred_log_prob_pred
```
```
def get_true_prediction_with_same_input(self):
  '''
  This returns the true trajectories from the current sample as well as all
  other samples that had the same past trajectories. It should be used only
  in conjunction with *get_true_and_predicted_paths()*.

  Returns
  -------
  Path_true_all : np.ndarray
    This is the true observed trajectory of the agents, in the form of a
    :math:`\{N_{subgroups} \times N_{same} \times N_{A} \times N_{O} \times 2\}` 
    dimensional numpy array with float values. If an agent is fully or on some 
    timesteps partially not observed, then this can include np.nan values. It
    must be noted that :math:`N_{same}` is the maximum number of similar samples,
    so for a smaller number, there will also be np.nan values.
  Subgroup_ind : np.ndarray
    This is a :math:`N_S` dimensional numpy array with int values. 
    All samples with the same value belong to a group with the same corresponding
    input. This can be used to avoid having to evaluate the same metric values
    for identical samples. It must however be noted, that due to randomness in 
    the model, the predictions made for these samples might differ.
    
    The value in this array will indicate which of the entries of **Path_true_all**
    should be chosen.

  '''
  
  ...
      
  return Path_true_all, Subgroup_ind
```
```
def get_true_and_predicted_class_probabilities(self):
  '''
  This returns the true and predicted classification probabilities.

  Returns
  -------
  P_true : np.ndarray
    This is the true probabilities with which one will observe a class, in the form of a
    :math:`\{N_S \times N_{classes}\}` dimensional numpy array with float values. 
    One value per row will be one, while the others will be zero.
  P_pred : np.ndarray
    This is the predicted probabilities with which one will observe a class, in the form of 
    a :math:`\{N_S \times N_{classes}\}` dimensional numpy array with float values. 
    The sum in each row will be 1.
  Class_names : list
    The list with :math:`N_{classes}` entries contains the names of the aforementioned
    behaviors.

  '''
  
  ...
  
  return P_true, P_pred, Class_names
```
```
def get_true_likelihood(self, joint_agents = True):
  '''
  This returns the probabilities assigned to ground truth trajectories 
  according to a Gaussian KDE method fitted to the ground truth samples
  with an identical inputs.
  

  Parameters
  ----------
  joint_agents : bool, optional
    This says if the probabilities for the predicted trajectories
    are to be calculated for all agents jointly. If this is the case,
    then, math:`N_{A}` in the output is 1. The default is True.

  Returns
  -------
  KDE_true_log_prob_true : np.ndarray
    This is a :math:`\{N_S \times 1 \times N_{A}\}`
    array that includes the probabilities for the true observations according 
    to the KDE model trained on the grouped true trajectories.

  KDE_true_log_prob_pred : np.ndarray
    This is a :math:`\{N_S \times N_{preds} \times N_{A}\}`
    array that includes the probabilities for the predicted trajectories
    according to the KDE model trained on the grouped true trajectories.

  '''
  
  ...
  
  return KDE_true_log_prob_true, KDE_true_log_prob_pred
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
    :math:`\{N_S \times N_{classes} \times 1\}` dimensional numpy array with float 
    values. One value per row will be given (actual observed class), while the others 
    will be np.nan.
  T_pred : np.ndarray
    This is the predicted time points at which one will observe a class, in the form of a
    :math:`\{N_S \times N_{classes} \times N_{quantiles}\}` dimensional numpy array 
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
