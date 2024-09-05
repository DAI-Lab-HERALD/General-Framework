# Adverarial attack generation

## General setting

### Adversarial attack
For the generation of adversarial attacks, four important settings need to be configured:
- Setting the number of predictions:
```
self.num_samples = 20 
```
- Setting the max number of iterations:
```
self.max_number_iterations = 50
```
- Setting an exponential decay learning rate ($\alpha = \gamma ^{iter} \cdot \alpha_{0}$):
```
self.gamma = 1
self.alpha = 0.01
```
- Setting the future data stored (essential for evaluation measures
self.store_GT = True # Ground truth future states are stored
self.store_GT = False # Perturbed future states are stored
self.store_pred_1 = True # average of num_samples predictions is stored on the initial unperturbed observed states



### Car size
To modify the size of the car ($m$) used in the animation, the length and width can be adjusted accordingly:
```
self.car_length = 4.1
self.car_width = 1.7
self.wheelbase = 2.7
```

### Clamping
For the adversarial attack strategy, 'Adversarial_Control_Action', the perturbed control action values are clamped absolute and relative.
```
self.epsilon_curv_absolute = 0.2
```
```
self.epsilon_acc_relative = 2
self.epsilon_curv_relative = 0.05
```

## Attack function
The attack function can be selected (See table):
```
self.loss_function_1 = 'ADE_Y_GT_Y_Pred_Max' # Mandotary
self.loss_function_2 = None # Option if not used set to None
```


| Type attack   | First input   | Second input  | Objective     | Formula       | Name framework (str)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- |   -------------       |
| ADE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $`-\frac{1}{T} {\sum}_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - {Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_GT_Y_Pred_Max' |
| ADE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $`\frac{1}{T} {\sum}_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - {Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_GT_Y_Pred_Min' |
| FDE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $`- \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - {Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_GT_Y_Pred_Max' |
| FDE           | $Y_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $` \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - {Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_GT_Y_Pred_Min' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $`-\frac{1}{T} {\sum}_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_Perturb_Y_Pred_Max' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $`\frac{1}{T} {\sum}_{t=1}^{T} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_Perturb_Y_Pred_Min' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Maximize distance | $`- \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_Perturb_Y_Pred_Max' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | $\hat{\tilde{Y}}_{\text{tar}}$ | Minimize distance | $` \left\| \hat{\tilde{Y}}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_Perturb_Y_Pred_Min' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Maximize distance | $`-\frac{1}{T} {\sum}_{t=1}^{T} \left\| {Y}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_Perturb_Y_GT_Max' |
| ADE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Minimize distance | $`\frac{1}{T} {\sum}_{t=1}^{T} \left\| {Y}_{\text{tar}}^{t} - \tilde{Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_Perturb_Y_GT_Min' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Maximize distance | $`- \left\| {Y}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_Perturb_Y_GT_Max' |
| FDE           | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{tar}}$ | Minimize distance | $` \left\| {Y}_{\text{tar}}^{T} - \tilde{Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_Perturb_Y_GT_Min' |
| ADE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Maximize distance | $`-\frac{1}{T} {\sum}_{t=1}^{T} \left\| \tilde{Y}_{\text{tar}}^{t} - \hat{Y}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_pred_iteration_1_and_Y_Perturb_Max' |
| ADE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Minimize distance | $`\frac{1}{T} {\sum}_{t=1}^{T} \left\| \tilde{Y}_{\text{tar}}^{t} - \hat{Y}_\text{tar}^{t} \right\|_2`$ | 'ADE_Y_pred_iteration_1_and_Y_Perturb_Min' |
| FDE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Maximize distance | $`-\left\| \tilde{Y}_{\text{tar}}^{T} - \hat{Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_pred_iteration_1_and_Y_Perturb_Max' |
| FDE           | $\hat{Y}_{\text{tar}}$  | $\tilde{Y}_{\text{tar}}$ | Minimize distance | $`\left\| \tilde{Y}_{\text{tar}}^{T} - \hat{Y}_{\text{tar}}^{T} \right\|_2`$ | 'FDE_Y_pred_iteration_1_and_Y_Perturb_Min' |
| ADE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Maximize distance | $`-\frac{1}{T} {\sum}_{t=1}^{T} \left\| \hat{Y}_{\text{tar}}^{t} - \hat{\tilde{Y}}_{\text{tar}}^{t} \right\|_2`$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Max' |
| ADE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Minimize distance | $`\frac{1}{T} {\sum}_{t=1}^{T} \left\| \hat{Y}_{\text{tar}}^{t} - \hat{\tilde{Y}}_\text{tar}^{t} \right\|_2`$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Min' |
| FDE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Maximize distance | $`- \left\| \hat{Y}_{\text{tar}}^{T} - \hat{\tilde{Y}}_{\text{tar}}^{T} \right\|_2`$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Max' |
| FDE           | $\hat{\tilde{Y}}_{\text{tar}}$  | $\hat{Y}_{\text{tar}}$ | Minimize distance | $` \left\| \hat{Y}_{\text{tar}}^{T} - \hat{\tilde{Y}}_\text{tar}^{T} \right\|_2`$ | 'ADE_Y_pred_and_Y_pred_iteration_1_Min' |
| Collision     | $\hat{\tilde{Y}}_{\text{tar}}$  | ${Y}_{\text{ego}}$ | Minimize smallest distance | $`\min_{t \in \{1, \ldots, T\}} \left\| \hat{\tilde{Y}}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2`$ | 'Collision_Y_pred_tar_Y_GT_ego' |
| Collision     | $\tilde{Y}_{\text{tar}}$  | ${Y}_{\text{ego}}$ | Minimize smallest distance | $`\min_{t \in \{1, \ldots, T\}} \left\| \tilde{Y}_{\text{tar}}^{t} - Y_{\text{ego}}^{t} \right\|_2`$ | 'Collision_Y_Perturb_tar_Y_GT_ego' |

## Barrier function
The barrier function can be selected (See table):
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None
 self.barrier_function_future = 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None
```

For the barrier function a distance threshold can be selected ($D_{\text{Adversarial threshold}}$):
```
 self.distance_threshold_past = 1
 self.distance_threshold_future = 1
```

For the barrier function the weight of the penalty can be changed ($\epsilon$):
```
self.log_value_past = 1.5
self.log_value_future = 1.5
```

| Type Regularization (Reference figure)  | Clean data   | Preturbed data | Edges clean data | First timestep |  Last timestep | Critical timestep | Deviation penalty | Formula       | Name framework (str)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- |   ------------- | ------------- | ------------- | ------------- | ------------- | 
| Time specific (A) | ${S}_{\text{tar}}$ | $\tilde{S}_{\text{tar}}$ |  |  ${t}_{0}$ | ${t}_{1}$  |   | $\epsilon$ | $`l_{\text{Time}}(S_{\text{tar}},\tilde{S}_{\text{tar}},{t}_{0},t_{1}) =-\frac{1}{t_{1}-{t}_{0}+1} \sum_{t={t}_{0}}^{{t}_{1}} \ln \left(D_{\text{max}} - \left\| \tilde{S}_{\text{tar}}^{t} -  S_{\text{tar}}^{t} \right\|_2\right)`$ | 'Time_specific'|
| Trajectory specific (B) | ${S}_{\text{tar}}$ | $\tilde{S}_{\text{tar}}$ | $Z_{\text{tar}}$ |  ${t}_{0}$ | ${t}_{1}$   |   | $\epsilon$ | $`l_{\text{Traj}}(S_{\text{tar}}, \tilde{S}_{\text{tar}}, t_0, t_1) = -\frac{1}{t_1 - t_0 + 1} \sum_{t=t_0}^{t_1} \ln \left( D_{\text{Max}} - \min_{z \in \{-H+1, \ldots, T-1\}} d(z, t) \right)`$ $`d(z, t) = \begin{cases} d_{\perp}(\tilde{S}_{\text{tar}}^t, Z_{\text{tar}}^z, Z_{\text{tar}}^{z+1}) \hspace{0.5cm} & \text{if } 0 < r(z, t) < 1, \\\min(\|\tilde{S}_{\text{tar}}^t - Z_{\text{tar}}^z\|_2, \\ \hspace{0.7cm}\|\tilde{S}_{\text{tar}}^t - Z_{\text{tar}}^{z+1}\|_2) & \text{otherwise}. \end{cases} `$  $` d_{\perp}(\tilde{S}_{\text{tar}}^t, Z_{\text{tar}}^z, Z_{\text{tar}}^{z+1}) = \| \frac{D^{1}_{x} D^{2}_{y} - D^{1}_{y} D^{2}_{x}}{\| D^{1}\|_2} \| `$ $`r(z, t) = \frac{D^{1}_{x} D^{2}_{x} + D^{1}_{y} D^{2}_{y}}{\|D^{1}\|_2^2} `$ $` \text{with} \quad D^{1} = Z_{\text{tar}}^{z+1} - Z_{\text{tar}}^z,D^{2} = \tilde{S}_{\text{tar}}^t - Z_{\text{tar}}^z.`$ | 'Trajectory_specific' |
| Time and Trajectory specific (C) | ${S}_{\text{tar}}$ | $\tilde{S}_{\text{tar}}$ | $Z_{\text{tar}}$ | ${t}_{0}$  |  ${t}_{1}$  |  ${t}_{critical}$ | $\epsilon$ | $`l_{\text{Time-traj}}(S_{\text{tar}},\tilde{S}_{\text{tar}},{t}_{0},t_{1},t_{crit}) = l_{\text{Time}}(S_{\text{tar}},\tilde{S}_{\text{tar}},{t}_{crit},t_{crit}) + l_{\text{Traj}}(S_{\text{tar}},\tilde{S}_{\text{tar}},{t}_{0},t_{1})`$ | 'Time_Trajectory_specific' |

**_NOTE:_**  Regularization for observed states $`{t}_{0} = -H + 1`$, $`{t}_{1} = 0`$, and $`{t}_{critical} = 0`$. Regularization for future states $`{t}_{0} = 1`$, $`{t}_{1} = T`$, and $`{t}_{critical} = T`$

![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Regularization_with_third_term-1.png)
## Gaussian smoothing

### Settings
- To select the total number of smoothed paths and predictions (1 per smoothed trajectory):
```
self.num_samples_used_smoothing = 15
```
- To select the sigmas to analyze, a list can be filled:
```
self.sigma = [0.05,0.1]
```
Specific for the attack type, 'Adversarial_Control_Action', sigmas specific for their control action can be selected:
```
self.sigma_acceleration = [0.05, 0.1]
self.sigma_curvature = [0.01, 0.05]
```

## Plot the data

### General
- Plot the loss over the iterations:
```
self.plot_loss = True
```
- Plot the image used for the neural network:
```
self.image_neural_network = True
```

### Dataset specific (left turns)
- Plot the input data:
```
self.plot_input = True
```
- Plot the adversarial scene (static):
```
self.static_adv_scene = True
```
- Plot the adversarial scene (animated):
```
self.animated_adv_scene = True
```
Set the smoothnes of the animation (higher is better):
```
self.total_spline_values = 100
```
For the attack type, 'Adversarial_Control_Action', control action can be animated:
```
self.control_action_graph = True
```


## Adversarial attack paper settings
### Nominal setting
![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Animation_basic.gif)

### ADE/FDE attack
![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Animation_ADE.gif)
To create this attack set:
```
self.loss_function_1 = 'ADE_Y_GT_Y_Pred_Max' or 'FDE_Y_GT_Y_Pred_Max'
self.loss_function_2 = None 
```
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific' or 'Time_Trajectory_specific' 
 self.barrier_function_future = None
```
### Collision attack
![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Animation_Collision.gif)
To create this attack set:
```
self.loss_function_1 = 'Collision_Y_pred_tar_Y_GT_ego'
self.loss_function_2 = None 
```
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific' or 'Time_Trajectory_specific' 
 self.barrier_function_future = None
```
### Max ADE/FDE attack
![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Animation_max_ADE.gif)
To create this attack set:
```
self.loss_function_1 = 'ADE_Y_Perturb_Y_Pred_Max' or 'FDE_Y_Perturb_Y_Pred_Max'
self.loss_function_2 = None 
```
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific' or'Time_Trajectory_specific' 
 self.barrier_function_future = 'Time_specific', 'Trajectory_specific' or 'Time_Trajectory_specific'
```
### Fake Collision attack
![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Animation_Fake_collision.gif)
To create this attack set:
```
self.loss_function_1 = 'Collision_Y_pred_tar_Y_GT_ego'
self.loss_function_2 = 'Y_Perturb' 
```
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific' or 'Time_Trajectory_specific' 
 self.barrier_function_future = 'Time_specific', 'Trajectory_specific' or 'Time_Trajectory_specific'
```
### Hide Collision attack
![image](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/animated_gif/Animation_Hide_collision.gif)
To create this attack set:
```
self.loss_function_1 = 'Collision_Y_Perturb_tar_Y_GT_ego'
self.loss_function_2 = 'ADE_Y_pred_and_Y_pred_iteration_1_Min'
```
```
 self.barrier_function_past = 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' 
 self.barrier_function_future = None
```

## How to add new attack or regularization function
[Modify here](https://github.com/DAI-Lab-HERALD/General-Framework/blob/main/Framework/Perturbation_methods/Adversarial_classes/loss.py)
### Attack
1.  In the Loss class, create a new function using the following structure. Use the inputs: X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, barrier_data, tar_agent, and ego_agent. Note that tar_agent and ego_agent are the indices of the target and ego agents, respectively.
```
def name_of_loss_function(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, barrier_data, tar_agent, ego_agent):
    return loss_function
```
2. Create a new class with the following structure. The sign depends on the specific objective of the attack.
```
class NameOfAttack(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return +/- name_of_loss_function(X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent)
```
3. Finally, define the name in the get_name function. If you want to use future perturbation, ensure the string includes 'Y_Perturb'.
```
elif loss_function == 'name_of_attack':
    return NameOfAttack()
```

### Regularization
1.  In the Loss class, create a new function using the following structure. Use the inputs: X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, barrier_data, tar_agent, and ego_agent. Note that tar_agent and ego_agent are the indices of the target and ego agents, respectively.
```
def name_of_regularization_function(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, barrier_data, tar_agent, ego_agent):
    return regularization_function
```
2. Create a new class with the following structure. The sign depends on the specific objective of the regularization.
```
class NameOfRegularization(LossFunction):
    def calculate_barrier(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return +/- name_of_regularization_function(X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent)
``` 
3. Finally, define the name in the barrier_function_name_past_states function or the barrier_function_name_future_states function, depending on which states the regularization is needed for.
```
elif barrier_function == 'name_of_regularization':
    return NameOfRegularization()
```
