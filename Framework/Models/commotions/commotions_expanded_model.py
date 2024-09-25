from enum import Enum
import numpy as np
import pandas as pd
import torch
import copy
import time
from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.registry import Models
from scipy import interpolate as interp


# Define an enumeration for the control type
class CtrlType(Enum):
    SPEED = 0
    ACCELERATION = 1


# Define an empty class for holding parameters
class Parameters:
    pass


# Define a class for simulation settings
class SimulationSettings:
    def __init__(self, start_time, end_time, time_step):
        # Store the simulation start, end, and time step values as attributes of the object
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        
        # Calculate the number of time steps based on the start, end, and time step values using torch
        self.n_time_steps = (torch.floor((end_time - start_time) / time_step)).to(dtype=torch.int64) + 1


# Define a class for the simulation state
class SimulationState:
    def __init__(self, simulation):
        # Store a reference to the simulation settings object as an attribute of the simulation state object
        self.setting = simulation.settings
        
        # Set the initial time step and time based on the simulation settings object
        self.i_time_step = 0
        self.time = self.setting.start_time

    # Define a method for setting the current time step and time
    def set_time_step(self, i_time_step):
        # Update the current time step and time based on the input time step value
        self.i_time_step = i_time_step
        self.time = self.setting.start_time + i_time_step * self.setting.time_step
   

class SCAgent:
    def __init__(self, device, name, ctrl_type, coll_dist, free_speeds, simulation, goal_pos, 
                 conflict_point, initial_pos, initial_long_speed, initial_long_accs, initial_yaw_angle, 
                 fixed_params, variable_params, give_priority = False, num_paths = 1,
                 const_acc = None, zero_acc_after_exit = False, 
                 plot_color = 'k', debug = False):
        '''
        Initializes a new agent that interact in a two agent game, coming up to
        an intersection, deciding who will go first.
        
        '''
        # Get GPU
        self.device = device
        self.zero_tensor = torch.tensor(0, dtype = torch.float32, device = self.device)
        
        # Check debugging mode
        self.debug = debug
        
        # run ancestor initialisation
        self.name = name
        self.can_reverse = False
        
        # set minimum speed
        if self.can_reverse:
            self.min_speed = torch.tensor(-torch.inf, device = self.device)
        else:
            self.min_speed = self.zero_tensor
        
        # assign agent to simulation
        self.simulation = simulation
        self.simulation.agents.append(self)
        
        # set control type
        self.ctrl_type = np.unique(ctrl_type)
        assert len(self.ctrl_type) == 1, "only one type of agent should be evaluated at the same time."
        self.ctrl_type = self.ctrl_type[0]
        
        # Get number of variable settings
        self.n_params = len(variable_params)
        self.n_samples = len(coll_dist)
        self.n_pred_paths = num_paths
        n_time_steps = self.simulation.settings.n_time_steps
        
        # parse and store the agent's goal position
        if goal_pos is None:
            self.goal = torch.zeros((2), dtype = torch.float32, device = self.device)
        else:
            self.goal = goal_pos
            
        # set conflict point
        self.simulation.conflict_point = conflict_point
            
        # allocate numpy arrays for trajectory states to be simulated
        self.traj_long_acc   = torch.zeros((self.n_params, self.n_samples, self.n_pred_paths, n_time_steps), 
                                           dtype = torch.float32, device = self.device)
        self.traj_yaw_rate   = torch.zeros((self.n_params, self.n_samples, self.n_pred_paths, n_time_steps), 
                                           dtype = torch.float32, device = self.device)
        self.traj_pos        = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 2, n_time_steps), 
                                          torch.nan, dtype = torch.float32, device = self.device)
        self.traj_long_speed = torch.full((self.n_params, self.n_samples, self.n_pred_paths, n_time_steps), 
                                          torch.nan, dtype = torch.float32, device = self.device)
        self.traj_yaw_angle  = torch.full((self.n_params, self.n_samples, self.n_pred_paths, n_time_steps), 
                                          torch.nan, dtype = torch.float32, device = self.device)
        
        # create initial kinematic values
        if initial_pos is None:
            initial_pos = torch.zeros((self.n_samples, 2), dtype = torch.float32, device = self.device)
            
        if initial_long_speed is None:
            initial_long_speed = torch.zeros((self.n_samples), dtype = torch.float32, device = self.device)
            
        if initial_long_accs is None:
            initial_long_accs = torch.zeros((self.n_samples), dtype = torch.float32, device = self.device)
            
        if initial_yaw_angle is None:
            if (initial_pos is not None) and (goal_pos is not None):
                agent_to_goal_vector = goal_pos[None] - initial_pos
                initial_yaw_angle = torch.atan2(agent_to_goal_vector[:, 1], agent_to_goal_vector[:, 0])
            else:
                initial_yaw_angle = torch.zeros((self.n_samples), dtype = torch.float32, device = self.device)
        
        # initial trajectory states (first update will be from 0 to 1)
        self.traj_pos[:, :, :, :, 0]     = initial_pos[None, :, None, :]
        self.traj_long_speed[:, :, :, 0] = torch.maximum(self.min_speed, initial_long_speed[None, :, None])
        self.traj_long_acc[:, :, :, 0]   = initial_long_accs[None, :, None]
        self.traj_yaw_angle[:, :, :, 0]  = initial_yaw_angle[None, :, None]
        
        # is this agent to just keep a constant acceleration?
        self.const_acc = const_acc
    
        # Set parameters 
        self.params = copy.copy(fixed_params)
        
        # Add variable parameters
        self.params.beta_V               = variable_params[:,0]
        self.params.DeltaV_th_rel        = variable_params[:,1]
        self.params.T                    = variable_params[:,2]
        self.params.T_delta              = variable_params[:,3]
        self.params.tau_theta            = variable_params[:,4]
        self.params.sigma_xdot           = variable_params[:,5]
        self.params.DeltaT               = variable_params[:,6]
        self.params.T_s                  = variable_params[:,7]
        self.params.D_s                  = variable_params[:,8]
        self.params.u_0_rel              = variable_params[:,9]
        self.params.k_da                 = variable_params[:,10]
        self.params.kalman_multi_pos     = variable_params[:,11]
        self.params.kalman_multi_speed   = variable_params[:,12]
        self.params.free_speed_multi_ego = variable_params[:,13]
        self.params.free_speed_multi_tar = variable_params[:,14]
        
        # Ensure positive, non zero values for tau theta
        self.params.tau_theta = torch.clip(self.params.tau_theta, min = 1e-6)
        
        # Use free_speeds to get kinematic parameters
        self.coll_dist_l = coll_dist
        
        if self.name == 'ego':
            self.v_free_l = free_speeds[None, :] * self.params.free_speed_multi_ego[:, None]
        elif self.name == 'tar':
            self.v_free_l = free_speeds[None, :] * self.params.free_speed_multi_tar[:, None]
        else:
            raise TypeError('Only ego and tar vehicles are to be modeled')
            
        self.params.k_g = 2 / self.v_free_l
        self.params.k_dv = 1 / self.v_free_l ** 2
        
        # get and store the number of actions, and the "non-action" action
        self.n_actions = len(self.params.ctrl_deltas)
        self.i_no_action = torch.where(self.params.ctrl_deltas == 0)[0][0]
        
        # Get number of possible behaviors
        self.n_beh = self.simulation.N_BEHAVIORS
        
        # Get derived parameters, vector with length self.n_params
        self.u_free = self.params.T_delta / torch.log(torch.tensor(2, dtype = torch.float32, device = self.device))
        self.params.u_0 = self.u_free * self.params.u_0_rel
        self.params.DeltaV_th = torch.tanh(1 / self.params.u_0_rel) * self.params.DeltaV_th_rel
        
        # Check consequences of priority
        if give_priority:
            self.params.u_ny = - 1.5 * self.u_free
        else:
            self.params.u_ny = torch.zeros(self.n_params, dtype = torch.float32, device = self.device)
        
        
        # allocate numpy arrays for perception variables
        self.perc_x_estimated = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 2, n_time_steps), 
                                           torch.nan, dtype = torch.float32, device = self.device)
        self.perc_cov_matrix  = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 2, 2, n_time_steps), 
                                           torch.nan, dtype = torch.float32, device = self.device)
        self.perc_x_perceived = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 2, n_time_steps), 
                                           torch.nan, dtype = torch.float32, device = self.device)
        
        
            
        # Set Kalman filtering matrices for state transition model 
        self.Fmatrix = torch.tensor([[[1, -self.simulation.settings.time_step], [0, 1]]], 
                                    dtype = torch.float32, device = self.device)
        self.Hmatrix = torch.tensor([[[1, 0],]], 
                                    dtype = torch.float32, device = self.device)
        
        self.Qmatrix = torch.zeros((self.n_params, 2, 2), dtype = torch.float32, device = self.device)
        self.Qmatrix[:,1,1] = self.params.sigma_xdot ** 2
        
        # store derived constants relating to the actions
        self.n_action_time_steps = (torch.ceil(self.params.DeltaT / 
                                               self.simulation.settings.time_step)).to(torch.int64)
        
        # prepare vectors for storing long acc, incl look_ahead
        n_actions_length = self.simulation.settings.n_time_steps + self.n_action_time_steps.max()
        self.action_long_accs = torch.zeros((self.n_params, self.n_samples, self.n_pred_paths, n_actions_length), 
                                            dtype = torch.float32, device = self.device)
        
        # set initial long acceleration
        for i, n_t in enumerate(self.n_action_time_steps):
            Ta = torch.linspace(1, 0, n_t + 1, dtype = torch.float32, device = self.device)
            self.action_long_accs[i,...,:n_t + 1] = initial_long_accs[:, None, None] * Ta[None,None,:]
        
        # A varibale that can be used to indicate which paths should still be updated further
        self.U_open_paths = torch.ones((self.n_params, self.n_samples, self.n_pred_paths), 
                                       dtype=torch.bool, device = self.device)
      

    def HELP_get_entry_exit_times(self, signed_CP_dist, coll_dist, long_speed):
        r'''
        This function allows one to calculate the time until one enters and then exits
        the contested space, which is used multiple times in this class.

        Parameters
        ----------
        signed_CP_dist : torch.tensor
            A :math:`N` dimensional tensor with the ditance of the agent to the conflict point.
        coll_dist : torch.tensor
            A :math:`N` dimensional tensor with the distance between conflict point and the 
            boundary of the contested space.
        long_speed : torch.tensor
            A :math:`N` dimensional tensor with the current speed of the agent towards the conflict point.

        Returns
        -------
        cs_entry_time : torch.tensor
            A :math:`N` dimensional tensor with the time needed to enter the contested space.
        cs_exit_time : torch.tensor
            A :math:`N` dimensional tensor with the time needed to exit the contested space.

        '''
        # Get distance to entering contested space and exiting it
        entry_distance = (signed_CP_dist - coll_dist)
        exit_distance  = (signed_CP_dist + coll_dist)
        
        # Determine the different cases of being before or after these stages
        cs_entry_update_1 = (entry_distance > 0) & (long_speed <= 0)
        cs_entry_update_2 = (entry_distance > 0) & (long_speed > 0)
        
        cs_exit_update_1 = (exit_distance > 0) & (long_speed <= 0)
        cs_exit_update_2 = (exit_distance > 0) & (long_speed > 0)
        
        # Allocate memory for storing the predicted entry and exit times
        cs_entry_time = torch.full(signed_CP_dist.shape, -torch.inf, dtype = torch.float32, device = self.device)
        cs_exit_time = torch.full(signed_CP_dist.shape, -torch.inf, dtype = torch.float32, device = self.device)
    
        # Assign the entry and exit time accordingly
        cs_entry_time[cs_entry_update_1] = torch.inf
        cs_entry_time[cs_entry_update_2] = entry_distance[cs_entry_update_2] / long_speed[cs_entry_update_2]
        
        cs_exit_time[cs_exit_update_1] = torch.inf
        cs_exit_time[cs_exit_update_2] = exit_distance[cs_exit_update_2] / long_speed[cs_exit_update_2]
        return cs_entry_time, cs_exit_time
    
     
    def HELP_get_accs_towards_goal(self, U_sum, U_param_index, own_ctrl_type,
                                   own_signed_CP_dist, own_coll_dist, own_long_speed, own_free_speed,
                                   oth_signed_CP_dist, oth_coll_dist, oth_long_speed, oth_time_to_entry, oth_time_to_exit):
        r'''
        This function is a vectorized form of DAU_get_implications(), in which it is called
        
        Parameters
        ----------
        U_sum : int
            The number :math:`N` of cases considered.
        U_param_index : torch.tensor
            This is a integer :math:`N` dimensional tensor, which maps the considered cases
            to their set of parameters.
        own_ctrl_type : CtrlType
            This is the control type (speed or acceleration) of the agent for which 
            the values are calculated.
        own_signed_CP_dist : torch.tensor
            This is a :math:`N` dimensional tensor, which describes at what signed distance 
            the own agent is towards the conflict point.
        own_coll_dist : torch.tensor
            This is a :math:`N` dimensional tensor, which describes at what signed distance 
            the own agent enters and leaves the contested space.
        own_long_speed : torch.tensor
            This is a :math:`N` dimensional tensor, which describes with which
            veloctiy the own agent moves.
        own_free_speed : torch.tensor
            This is a :math:`N` dimensional tensor, which describes with which
            veloctiy the own agent wants to move.
        oth_signed_CP_dist : torch.tensor
            This is a :math:`N` dimensional tensor, which describes at what signed distance 
            the other agent is towards the conflict point.
        oth_coll_dist : torch.tensor
            This is a :math:`N` dimensional tensor, which describes at what signed distance 
            the other agent enters and leaves the contested space.
        oth_long_speed : torch.tensor
            This is a :math:`N` dimensional tensor, which describes with which
            veloctiy the other agent moves.
        oth_time_to_entry : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the time until the other
            agent enters the constested space.
        oth_time_to_exit : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the time until the other
            agent leaves the constested space.

        Returns
        -------
        Accs_first : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the acceleration needed 
            for the own agent to reach the contested space first.
        T_accs_first : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the time the acceleration 
            above needs to be applied for the own agent to reach the contested space first.
        T_dws_first : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the time the agent needs to 
            wait before the contested space after brakeing to enter the contested space after 
            the other agent. This should be always either zero or nan.
        Accs_second : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the acceleration needed 
            for the own agent to reach the contested space second.
        T_accs_second : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the time the acceleration 
            above needs to be applied for the own agent to reach the contested space second.
        T_dws_second : torch.tensor
            This is a :math:`N` dimensional tensor, which describes the time the agent needs to 
            wait before the contested space after brakeing to enter the contested space after 
            the other agent.

        '''
        # Get the cases where there is currently a collision at the contested space
        collision = (torch.abs(own_signed_CP_dist) < own_coll_dist) & (torch.abs(oth_signed_CP_dist) < oth_coll_dist)
        
        # Get distance that needs to be traveld to leave contested space
        own_dist_to_exit  = (own_signed_CP_dist + own_coll_dist + self.params.D_s[U_param_index])
        
        # Get distance that needs to be traveld to leave contested space
        own_dist_to_entry = (own_signed_CP_dist - own_coll_dist - self.params.D_s[U_param_index])
        
        # Get cases where the own agent already exited the contested space
        own_agent_exited     = own_dist_to_exit <= 0
        
        # Get cases where the other agent has exited the contested space
        oth_agent_exited     = oth_signed_CP_dist <= - oth_coll_dist
        
        # Get the time until the own agent consideres the other to have entered
        oth_time_to_entry_safe = oth_time_to_entry - self.params.T_s[U_param_index]
        oth_time_to_exit_safe  = oth_time_to_exit  + self.params.T_s[U_param_index]
        
        # Get current deviation from free speed
        own_difference_to_free_speed = own_long_speed - own_free_speed
        
        # Get time needed to get to free speed
        if own_ctrl_type is CtrlType.SPEED:
            own_time_to_free_speed = self.params.DeltaT[U_param_index]
        else:
            own_time_to_free_speed = torch.full((U_sum,), self.params.T_acc_regain_spd, 
                                                dtype = torch.float32, device = self.device)
        
        # Get corresponding acceleration needed
        own_acc_to_free_speed = - own_difference_to_free_speed / own_time_to_free_speed
        
        # Find cases where no acceleration would be needed and set the time there to zero
        own_no_acceleration = own_difference_to_free_speed == 0
        own_time_to_free_speed[own_no_acceleration] = 0
        
        # Allocate memory for first output
        Accs_first    = torch.full((U_sum,), torch.nan, dtype = torch.float32, device = self.device)
        T_accs_first  = torch.full((U_sum,), torch.nan, dtype = torch.float32, device = self.device)
        T_dws_first   = torch.full((U_sum,), torch.nan, dtype = torch.float32, device = self.device)
        
        Accs_second   = torch.full((U_sum,), torch.nan, dtype = torch.float32, device = self.device)
        T_accs_second = torch.full((U_sum,), torch.nan, dtype = torch.float32, device = self.device)
        T_dws_second  = torch.full((U_sum,), torch.nan, dtype = torch.float32, device = self.device)
        
        # Answer cases where the own agent already is outside the contested space
        own_agent_exited_safe = (~collision) & own_agent_exited
        Accs_first[own_agent_exited_safe]    = own_acc_to_free_speed[own_agent_exited_safe]
        T_accs_first[own_agent_exited_safe]  = own_time_to_free_speed[own_agent_exited_safe]
        T_dws_first[own_agent_exited_safe]   = 0
        
        # Answer cases where the own agent already is outside the contested space
        one_agent_exited_safe = (~collision) & (own_agent_exited | oth_agent_exited)
        Accs_second[one_agent_exited_safe]   = own_acc_to_free_speed[one_agent_exited_safe]
        T_accs_second[one_agent_exited_safe] = own_time_to_free_speed[one_agent_exited_safe]
        T_dws_second[one_agent_exited_safe]  = 0
        
        # Answer the case where both agents have not yet enterd the contested space
        own_can_still_be_first = ((~collision) & (~own_agent_exited) & (oth_time_to_entry_safe > 0))
        OSF_index = torch.where(own_can_still_be_first)[0]
        
        # Decide if one can reach free speed before the other vehicle enters the contested space
        own_free_speed_later = oth_time_to_entry_safe[OSF_index] < own_time_to_free_speed[OSF_index]
        own_free_speed_first = ~(own_free_speed_later)
        
        # Get the distance travelled while accelerating to free speed until the other agent enters the contested space
        own_free_acc_dist = torch.zeros(len(OSF_index), dtype = torch.float32, device = self.device)
        own_free_acc_dist[own_free_speed_later] = (own_long_speed[OSF_index][own_free_speed_later] * 
                                                   oth_time_to_entry_safe[OSF_index][own_free_speed_later] +
                                                   0.5 * own_acc_to_free_speed[OSF_index][own_free_speed_later] * 
                                                   oth_time_to_entry_safe[OSF_index][own_free_speed_later] ** 2)
        
        own_free_acc_dist[own_free_speed_first] = (own_long_speed[OSF_index][own_free_speed_first] * 
                                                   own_time_to_free_speed[OSF_index][own_free_speed_first] + 
                                                   0.5 * own_acc_to_free_speed[OSF_index][own_free_speed_first] * 
                                                   own_time_to_free_speed[OSF_index][own_free_speed_first] ** 2 + 
                                                   own_free_speed[OSF_index][own_free_speed_first] * 
                                                   (oth_time_to_entry_safe[OSF_index][own_free_speed_first] - 
                                                    own_time_to_free_speed[OSF_index][own_free_speed_first]))
        
        
        
        # Decide if accelerating to free speed is enough to leave contested space first
        own_free_speed_enough = own_free_acc_dist > own_dist_to_exit[OSF_index]
        own_free_speed_slower = ~(own_free_speed_enough)
        
        # Get relative indices 
        FSE_index = OSF_index[own_free_speed_enough]
        FSS_index = OSF_index[own_free_speed_slower]
        
        # Get the higher acceleration necessary to be first to the contested space
        target_time = oth_time_to_entry_safe[FSS_index] # This is larger than 0 but not infinite
        target_acc = 2 * (own_dist_to_exit[FSS_index] - 
                          own_long_speed[FSS_index] * target_time) / (target_time ** 2 + torch.finfo(torch.float32).eps)
        
        # Define minimal permissible acceleration that gets you to the free speed
        target_acc_min = torch.minimum((own_free_speed[FSS_index] - own_long_speed[FSS_index]) / 
                                       (target_time + torch.finfo(torch.float32).eps), self.zero_tensor)
        
        # Ensure minimum acceleration is kept
        target_acc = torch.maximum(target_acc, target_acc_min)
        
        # Assign the final values
        Accs_first[FSE_index]   = own_acc_to_free_speed[FSE_index]
        T_accs_first[FSE_index] = own_time_to_free_speed[FSE_index]
        Accs_first[FSS_index]   = target_acc
        T_accs_first[FSS_index] = target_time
        T_dws_first[OSF_index]  = 0
        
        # Answer the case where one can still be second
        own_can_still_be_second = (~collision) & (own_dist_to_entry > 0) & (~oth_agent_exited)
        OSS_index = torch.where(own_can_still_be_second)[0]
        
        # Decide if one can reach free speed before the other vehicle enters the contested space
        own_free_speed_later = oth_time_to_exit_safe[OSS_index] < own_time_to_free_speed[OSS_index]
        own_free_speed_first = ~(own_free_speed_later)
        
        # Get the distance travelled while accelerating to free speed until the other agent enters the contested space
        own_free_acc_dist = torch.zeros(len(OSS_index), dtype = torch.float32, device = self.device)
        own_free_acc_dist[own_free_speed_later] = (own_long_speed[OSS_index][own_free_speed_later] * 
                                                   oth_time_to_exit_safe[OSS_index][own_free_speed_later] +
                                                   0.5 * own_acc_to_free_speed[OSS_index][own_free_speed_later] * 
                                                   oth_time_to_exit_safe[OSS_index][own_free_speed_later] ** 2)
        
        own_free_acc_dist[own_free_speed_first] = (own_long_speed[OSS_index][own_free_speed_first] * 
                                                   own_time_to_free_speed[OSS_index][own_free_speed_first] + 
                                                   0.5 * own_acc_to_free_speed[OSS_index][own_free_speed_first] * 
                                                   own_time_to_free_speed[OSS_index][own_free_speed_first] ** 2 + 
                                                   own_free_speed[OSS_index][own_free_speed_first] * 
                                                   (oth_time_to_exit_safe[OSS_index][own_free_speed_first] - 
                                                    own_time_to_free_speed[OSS_index][own_free_speed_first]))
        
        # Decide if accelerating to free speed is enough to not enter the conteseted space first
        own_free_speed_second = own_free_acc_dist < own_dist_to_entry[OSS_index]
        own_free_speed_faster = ~(own_free_speed_second)
        
        # Get relative indices 
        FSD_index = OSS_index[own_free_speed_second]
        FSF_index = OSS_index[own_free_speed_faster]
        
        
        # Get the lower acceleration necessary to be first to the contested space
        target_time = oth_time_to_exit_safe[FSF_index] # This is larger than 0 but not infinite
        target_acc  = torch.zeros(len(FSF_index), dtype = torch.float32, device = self.device)
        target_dws  = torch.zeros(len(FSF_index), dtype = torch.float32, device = self.device)
        
        # Get remaining parts
        finite_time = target_time != torch.inf
        
        # Get infinite traget time were braking is not necessary
        infinte_time_no_brake = (~finite_time) & (own_long_speed[FSF_index] <= 0)
        
        # Get infinite target time where braking is necessary
        infinte_time_brake = (~finite_time) & (own_long_speed[FSF_index] > 0)
        
        # Do nothing if braking is not necessary            
        target_acc[infinte_time_no_brake]  = 0
        target_time[infinte_time_no_brake] = 0
        target_dws[infinte_time_no_brake]  = torch.inf
        
        # Get to a total stop if one has to wait an infinite time
        target_acc[infinte_time_brake]  = - (own_long_speed[FSF_index][infinte_time_brake] ** 2 / 
                                             (2 * own_dist_to_entry[FSF_index][infinte_time_brake] + torch.finfo(torch.float32).eps))
        target_time[infinte_time_brake] = - (own_long_speed[FSF_index][infinte_time_brake] / 
                                             (target_acc[infinte_time_brake] - torch.finfo(torch.float32).eps))
        target_dws[infinte_time_brake]  = torch.inf
        
        
        target_acc[finite_time] = 2 * ((own_dist_to_entry[FSF_index][finite_time] - own_long_speed[FSF_index][finite_time] * 
                                        target_time[finite_time]) / (target_time[finite_time] ** 2 + torch.finfo(torch.float32).eps))
        
        # Check if one can brake slow enough to not come to a stop befroe target time whil not 
        # crossing in to the contested space.
        finite_time_replace = ((2 * own_dist_to_entry[FSF_index]) < target_time * own_long_speed[FSF_index]) & finite_time
        
        # If this is not possible, brake harder so that one come to a stop at the entrance to the contested space
        # and prepare for waiting on the other car to pass
        target_acc[finite_time_replace]  = - (own_long_speed[FSF_index][finite_time_replace] ** 2 / 
                                              (2 * own_dist_to_entry[FSF_index][finite_time_replace] + torch.finfo(torch.float32).eps))
        target_time[finite_time_replace] = - (own_long_speed[FSF_index][finite_time_replace] / 
                                              (target_acc[finite_time_replace] - torch.finfo(torch.float32).eps))
        target_dws[finite_time]          = torch.maximum(oth_time_to_exit_safe[FSF_index][finite_time] - target_time[finite_time],
                                                         self.zero_tensor)
        
        # Set the calcualted values
        Accs_second[FSD_index]                 = own_acc_to_free_speed[FSD_index]
        T_accs_second[FSD_index]               = own_time_to_free_speed[FSD_index]
        T_dws_second[FSD_index]                = 0
        Accs_second[FSF_index]                 = target_acc
        T_accs_second[FSF_index]               = target_time
        T_dws_second[FSF_index]                = target_dws
        
        return Accs_first, T_accs_first, T_dws_first, Accs_second, T_accs_second, T_dws_second
    
       
    def prepare_for_simulation(self):
        '''
        Sets up the empty tensors for the values to be simulated.
        Prepares the stochastic perception of the other agent.

        '''
        n_time_steps = self.simulation.settings.n_time_steps
        
        # store a reference to the other agent
        assert(len(self.simulation.agents) == 2)
        for agent in self.simulation.agents:
            if agent is not self:
                self.other_agent = agent

        # allocate vectors for storing internal states
        self.est_action_vals           = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 
                                                     self.n_actions, n_time_steps), 
                                                    torch.nan, dtype = torch.float32, device = self.device)
        self.beh_value_given_actions   = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 
                                                     self.n_actions, self.n_beh, n_time_steps), 
                                                    torch.nan, dtype = torch.float32, device = self.device) 
        self.beh_long_accs             = torch.full((self.n_params, self.n_samples, self.n_pred_paths, 
                                                     self.n_beh, n_time_steps), 
                                                    torch.nan, dtype = torch.float32, device = self.device)
        
        
        # set inital state for perceived position
        cp_dist_mean = torch.amax(torch.abs(self.other_agent.traj_pos[:, :, :, :, 0]), axis = -1)
        speed_mean = self.other_agent.v_free_l[:, :, None]
        self.perc_x_estimated[:, :, :, 0, -1] = cp_dist_mean
        self.perc_x_estimated[:, :, :, 1, -1] = speed_mean
        
        # Get initial uncertainty values for the perceived state
        cp_dist_var = (self.params.kalman_multi_pos[:, None, None] * cp_dist_mean) ** 2
        speed_var = (self.params.kalman_multi_speed[:, None, None] * speed_mean) ** 2
        
        # set initial state for preception kovariance matrix
        self.perc_cov_matrix[:, :, :, :, :, -1] = 0.0
        self.perc_cov_matrix[:, :, :, 0, 0, -1] = cp_dist_var 
        self.perc_cov_matrix[:, :, :, 1, 1, -1] = speed_var  
        
        
        # initial value perception states (first update will be from -1 to 0)
        self.est_action_vals[:, :, :, :, -1] = 0
        self.beh_value_given_actions[:, :, :, :, :, -1] = 0
        
    
    def do_kinematics_update(self):
        '''
        This function calculates the next position and velocity based on
        old position, velocity and acceleration.

        '''
        # Get current timestep
        i_time_step = self.simulation.state.i_time_step
        time_step = self.simulation.settings.time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # extract old trajectory data
        pos_old = self.traj_pos[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step - 1]
        long_speed_old = self.traj_long_speed[Up_param_index, Up_sample_index, Up_path_index, i_time_step - 1] 
        long_acc_old = self.traj_long_acc[Up_param_index, Up_sample_index, Up_path_index, i_time_step - 1] 
        
        yaw_angle_old = self.traj_yaw_angle[Up_param_index, Up_sample_index, Up_path_index, i_time_step - 1]
        yaw_rate_old = self.traj_yaw_rate[Up_param_index, Up_sample_index, Up_path_index, i_time_step - 1]
        
        # update trajectory data
        pos = (torch.maximum(self.min_speed, time_step * long_speed_old + long_acc_old * (time_step ** 2) / 2)[:,None] * 
               torch.concat((torch.cos(yaw_angle_old)[:,None], torch.sin(yaw_angle_old)[:,None]), axis = 1) + pos_old)
        long_speed = torch.maximum(self.min_speed, long_speed_old + long_acc_old * time_step)
        yaw_angle  = yaw_angle_old + yaw_rate_old * time_step
        
        # save updated trajectory data
        self.traj_pos[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]     = pos
        self.traj_long_speed[Up_param_index, Up_sample_index, Up_path_index, i_time_step] = long_speed
        self.traj_yaw_angle[Up_param_index, Up_sample_index, Up_path_index, i_time_step]  = yaw_angle
        
    
    def prepare_for_action_update(self):
        '''
        For simulation, acceleration has to be calculated. This function prepares that
        calculation by exchanging updated information between the agents.

        '''
        # Get current timestep
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Get useful own values
        self.coll_dist = self.coll_dist_l[Up_sample_index]
        self.v_free = self.v_free_l[Up_param_index, Up_sample_index]
        
        # Get useful other values of the other agent
        self.oth_coll_dist = self.other_agent.coll_dist_l[Up_sample_index]
        self.oth_v_free = self.other_agent.v_free_l[Up_param_index, Up_sample_index]
        
        # Extrac current trajectory data
        pos = self.traj_pos[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]
        long_speed = self.traj_long_speed[Up_param_index, Up_sample_index, Up_path_index, i_time_step]
        long_acc = self.traj_long_acc[Up_param_index, Up_sample_index, Up_path_index, i_time_step-1]
        yaw_angle =  self.traj_yaw_angle[Up_param_index, Up_sample_index, Up_path_index, i_time_step]
        
        # Get difference to conflict point
        vect_to_conflict_point = self.simulation.conflict_point[None,:] - pos
        
        # Get current heading
        heading_vect = torch.concat((torch.cos(yaw_angle)[:,None], 
                                     torch.sin(yaw_angle)[:,None]), dim = 1)
        
        # Get signed distance to conflict point
        signed_CP_dist = torch.sum(heading_vect * vect_to_conflict_point, 1)
        
        # Get time needed to enter and exit the contested space
        cs_entry_time, cs_exit_time = self.HELP_get_entry_exit_times(signed_CP_dist, self.coll_dist, long_speed)
        
        # Set current state
        self.Curr_state = torch.concat([long_speed[None], 
                                        long_acc[None], 
                                        signed_CP_dist[None], 
                                        cs_entry_time[None], 
                                        cs_exit_time[None]], dim = 0)
    
    
    def DAU_update_perception(self):
        r'''
        This function updates the state of the other agent perceived by the current agent.
        It is callled inside do_action_update().
        

        Returns
        -------
        mean_estimated : torch.tensor
            This is a :math:`\{n_{paths} \times 2\}` dimensional tensor, that give the 
            estimated state of the other agent, using Kalman filtering to extract
            it from the previously perceived states.
        mean_perceived : torch.tensor
            This is a :math:`\{n_{paths} \times 2\}` dimensional tensor with the 
            currently perceived state of the other agent.
        cov_matrix_new : torch.tensor
            This is a :math:`\{n_{paths} \times 2 \times 2\}` dimensional tensor with 
            the current setimation of the covariance matrix.

        '''
        # get current time step
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine the number of those paths
        Up_sum = len(Up_sample_index)
        
        # Calculate the distance between the two agents and relative distance factor
        D = torch.sqrt(self.Curr_state[2] ** 2 + self.other_agent.Curr_state[2] ** 2)
        D_rel = self.params.H_e / D
        
        # Calculate the factor to adjust the observation noise standard deviation
        D_rel_fac = 1 - D_rel / torch.tan(torch.arctan(D_rel) + self.params.tau_theta[Up_param_index])
        
        # Calculate the observation noise standard deviation for the current time step
        curr_obs_noise_stddev = torch.abs(self.other_agent.Curr_state[2]) * D_rel_fac
        
        # Ensure the observation noise standard deviation is non-negative
        curr_obs_noise_stddev = torch.maximum(self.zero_tensor, curr_obs_noise_stddev)
        
        # Generate random Gaussian noise for the observation model
        eps_1 = torch.randn((Up_sum, ), dtype = torch.float32, device = self.device)
        eps_2 = torch.randn((Up_sum,2), dtype = torch.float32, device = self.device)
        
        # Predict the next state estimate using the motion model
        x_pred = torch.matmul(self.Fmatrix, self.perc_x_estimated[Up_param_index, Up_sample_index, 
                                                                  Up_path_index, :, i_time_step-1, None])
        
        # Get the old covariance matrix for the state estimate
        cov_matrix_old = self.perc_cov_matrix[Up_param_index, Up_sample_index, Up_path_index, :, :, i_time_step-1]
        
        # Calculate the predicted covariance matrix for the next time step
        pred_cov_matrix = self.Qmatrix[Up_param_index] + torch.matmul(torch.matmul(self.Fmatrix, cov_matrix_old), 
                                                                      self.Fmatrix.permute(0,2,1))
        
        # Calculate the innovation (difference between the actual observation and the predicted observation)
        ytilde = (self.other_agent.Curr_state[2] + 
                  eps_1 * curr_obs_noise_stddev)[:,None,None] - torch.matmul(self.Hmatrix, x_pred)
        
        # Calculate the Kalman gain
        Kmatrix = torch.matmul(pred_cov_matrix, self.Hmatrix.permute(0,2,1)) 
        
        # Calculate the innovation covariance matrix
        innov_cov = torch.matmul(self.Hmatrix, Kmatrix) + curr_obs_noise_stddev[:, None, None] ** 2
        
        # Calculate the updated state mean using the Kalman gain
        mean_estimated = (x_pred + Kmatrix * (ytilde / innov_cov))[:,:,0]
        
        # Calculate the updated state covariance matrix using the Kalman gain
        # (Here, dividing by innov_cov is needed before matmul as otherwise there 
        #  are numerical issues (x ** 2 / x == x is not guaranteed, while (x/x)*x == x can be))
        cov_matrix_new = pred_cov_matrix - torch.matmul(Kmatrix / innov_cov, Kmatrix.permute(0,2,1)) 
        
        # Calculate the helper matrix to update the covariance matrix
        helper = cov_matrix_new.clone()
        helper[:,0,1] = 0.0
        helper[:,1,1] = torch.sqrt(torch.maximum(self.zero_tensor, torch.det(cov_matrix_new)))
        
        # Calcualte the perception error based on Gaußian noise created accoring to the new covariance matrix
        mean_perc_error  = torch.matmul(helper, eps_2.unsqueeze(-1)).squeeze(-1)  
        mean_perc_error_fac = torch.sqrt(torch.maximum(self.zero_tensor,cov_matrix_new[:,0,0])) + torch.finfo(torch.float32).eps * 5
        
        # Calculate the perceived mean by adding random Gaussian noise to the updated mean
        mean_perceived = mean_estimated + mean_perc_error / mean_perc_error_fac.unsqueeze(-1)
        
        # Ensure the minimum speed limit is enforced
        mean_perceived[:,1] = torch.maximum(self.min_speed, mean_perceived[:,1])
        
        return mean_estimated, mean_perceived, cov_matrix_new


    def DAU_beh_long_acc_update(self):
        r'''
        This function calcualtes the accelerations necessary for the other agent
        to go either first or second through the contested space.
        It is callled inside do_action_update().
        

        Returns
        -------
        accs_egofirst : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor, that give the 
            estimated state of the other agent, using Kalman filtering to extract
            it from the previously perceived states.
        accs_egosecond : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor with the 
            currently perceived state of the other agent.

        '''
        
        # get current time step
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Get the current perceived distance and speed
        Oth_perceived_signed_CP_dist = self.perc_x_perceived[Up_param_index, Up_sample_index, Up_path_index, 0, i_time_step]
        Oth_perceived_long_speed     = self.perc_x_perceived[Up_param_index, Up_sample_index, Up_path_index, 1, i_time_step]

        [accs_egofirst, _, _, 
         accs_egosecond, _, _] = self.HELP_get_accs_towards_goal(len(Up_param_index), Up_param_index, self.other_agent.ctrl_type, 
                                                                 Oth_perceived_signed_CP_dist, self.oth_coll_dist,
                                                                 Oth_perceived_long_speed, self.oth_v_free,
                                                                 self.Curr_state[2], self.coll_dist, self.Curr_state[0],
                                                                 self.Curr_state[3], self.Curr_state[4]) 

        return accs_egofirst, accs_egosecond
    
    
    def DAU_predict_oth(self):
        r"""
        This function predicts the short term prediction of kinematics of the other agent
        by the current agent, depending on the previously assumed accelerations in self.beh_long_accs. 
        It is callled inside do_action_update().
        
        Returns
        -------
        Pred_oth_states : torch.tensor
            This is a :math:`\{5 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It includes along its first dimension the perceived speed, acceleration, signed distance to crossing point,
            the predicted time until entry into the conflicted space, and the predicted time until leaving the contested space.
            This is calculated for all possible behaviors and every given path.

        """
        # get current time step
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine the number of those paths
        Up_sum = len(Up_sample_index)
        
        # Determine the behaviors that can be assessed
        U = ~(torch.isnan(self.beh_long_accs[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]))
        
        # Get the corresponding path and behavior index and their number
        U_path_index, U_beh_index = torch.where(U)
        
        # Get relevant perceived state
        perc_oth_state = self.perc_x_perceived[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]
        
        # Get the current perceived distance and speed
        Oth_perceived_signed_CP_dist = perc_oth_state[U_path_index, 0]
        Oth_perceived_long_speed     = perc_oth_state[U_path_index, 1]
        
        
        # Extract the corresponding values as a vector Oth_pred_long_acc
        Oth_pred_long_acc = self.beh_long_accs[Up_param_index[U_path_index], Up_sample_index[U_path_index], 
                                               Up_path_index[U_path_index], U_beh_index, i_time_step]
        
        # Get the prediction time space
        Dist_pred_time = self.params.DeltaT[Up_param_index[U_path_index]] 
        
        # Predict the speed at the time
        Oth_pred_long_speed = Oth_perceived_long_speed + Oth_pred_long_acc * Dist_pred_time
        
        # Calculate predicted speeds that are negative 
        To_slow = Oth_pred_long_speed < self.min_speed
        
        # Assert that this is because the acceleration here was to low 
        # (i.e., the speed previously was high enough)
        assert (Oth_pred_long_acc[To_slow] < 0).all()
        
        # Enforce minimum speed boundary
        Oth_pred_long_speed[To_slow] = self.min_speed
        
        # Reduce the prediction time accordingly, so that at prediction time min speed is reached
        Dist_pred_time[To_slow] = (self.min_speed - Oth_perceived_long_speed[To_slow]) / Oth_pred_long_acc[To_slow]
        
        # Calculate predicted sitance to conflicted space 
        Oth_pred_signed_CP_dist = (Oth_perceived_signed_CP_dist - Oth_perceived_long_speed * Dist_pred_time - 
                                   Oth_pred_long_acc * Dist_pred_time ** 2 / 2)
        
        # Get time needed to enter and exit the contested space
        Oth_pred_entry_time, Oth_pred_exit_time = self.HELP_get_entry_exit_times(Oth_pred_signed_CP_dist, 
                                                                                 self.oth_coll_dist[U_path_index], 
                                                                                 Oth_pred_long_speed)
        
        # Allocate memory for the final output Pred_oth_states and set the results
        Pred_oth_states = torch.zeros((5, Up_sum, self.n_beh), dtype = torch.float32, device = self.device)
        Pred_oth_states[0, U_path_index, U_beh_index] = Oth_pred_long_speed
        Pred_oth_states[1, U_path_index, U_beh_index] = Oth_pred_long_acc
        Pred_oth_states[2, U_path_index, U_beh_index] = Oth_pred_signed_CP_dist
        Pred_oth_states[3, U_path_index, U_beh_index] = Oth_pred_entry_time
        Pred_oth_states[4, U_path_index, U_beh_index] = Oth_pred_exit_time
        
        # Copy results over all possible actions
        Pred_oth_states = torch.tile(Pred_oth_states[:,:,None,:], (1, 1, self.n_actions, 1))
        
        return Pred_oth_states
    
    
    def DAU_action_long_acc_update(self):
        '''
        This function updates the planned accelerations of the current agent.
        It is callled inside do_action_update().
        
        Returns
        -------
        Action_long_accs_corrected : torch.tensor
            This returns the corrected planned accelerations.

        '''
        # get current time step
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Get the future planned accelerations
        Action_long_accs_corrected = self.action_long_accs[Up_param_index, Up_sample_index, Up_path_index, i_time_step:]
        
        # Check which paths are currently at minimum velocity
        to_check = self.Curr_state[0] <= self.min_speed
        
        # Prevent negative planned accelerations in those cases
        Action_long_accs_corrected[to_check] = torch.maximum(self.zero_tensor, Action_long_accs_corrected[to_check])
        return Action_long_accs_corrected
        
    
    def DAU_predict_own(self):
        r"""
        This function predicts the short term prediction of kinematics of the current agent
        by the current agent, depending on the previously assumed accelerations in self.beh_long_accs. 
        It is callled inside do_action_update().

        Returns
        -------
        Pred_own_states : torch.tensor
            This is a :math:`\{5 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It includes along its first dimension the speed, acceleration, signed distance to crossing point,
            the predicted time until entry into the conflicted space, and the predicted time until leaving the contested space.
            This is calculated for all possible actions and every given path.

        """
        # get current time step and size of time steps
        i_time_step = self.simulation.state.i_time_step
        time_step = self.simulation.settings.time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine the number of those paths
        Up_sum = len(Up_sample_index)
        
        # Get curently planned acceleration prowfile
        Own_planned_long_acc = self.action_long_accs[Up_param_index, Up_sample_index, Up_path_index, i_time_step:]
        Own_planned_long_acc = torch.tile(Own_planned_long_acc.unsqueeze(-1), (1, 1, self.n_actions))
        
        # Get number of time steps in prediction period
        n_action_time_steps = self.n_action_time_steps[Up_param_index]
        
        # Find actual time steps over which constant control input is to be applied
        U_AS = (n_action_time_steps.unsqueeze(-1) > torch.arange(Own_planned_long_acc.shape[1], device = self.device).unsqueeze(0))
        U_AS_path_index, U_AS_time_index   = torch.where(U_AS)
        Un_AS_path_index, Un_AS_time_index = torch.where(~U_AS)
        
        # Constant acceleration for speed control
        if self.ctrl_type is CtrlType.SPEED:
            Acc_value = self.params.ctrl_deltas[None,:] / self.params.DeltaT[Up_param_index, None]
            Own_planned_long_acc[U_AS_path_index, U_AS_time_index, :] += Acc_value[U_AS_path_index, :]
        # Increasing acceleration for acceleration control
        else:
            # Get step values during increase
            Step_value = (U_AS_time_index.to(dtype = torch.float32) / (n_action_time_steps[U_AS_path_index] - 1)).unsqueeze(-1)
            # Add additionally planned inputs
            Own_planned_long_acc[U_AS_path_index, U_AS_time_index, :]   += self.params.ctrl_deltas.unsqueeze(0) * Step_value
            Own_planned_long_acc[Un_AS_path_index, Un_AS_time_index, :] += self.params.ctrl_deltas.unsqueeze(0)
        
        # Get current position
        Own_planned_pos = self.traj_pos[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]
        Own_planned_pos = torch.tile(Own_planned_pos.unsqueeze(-1), (1, 1, self.n_actions))
        
        # Get current speed
        Own_planned_long_speed = self.traj_long_speed[Up_param_index, Up_sample_index, Up_path_index, i_time_step]
        Own_planned_long_speed = torch.tile(Own_planned_long_speed.unsqueeze(-1), (1, self.n_actions))
        
        # Get current yaw angle
        Own_yaw_angle = self.traj_yaw_angle[Up_param_index, Up_sample_index, Up_path_index, i_time_step]
        Own_yaw_angle = torch.tile(Own_yaw_angle.unsqueeze(-1).unsqueeze(-1), (1, 1, self.n_actions))
        
        # Calculate current heading vector 
        Own_heading_vect = torch.concat((torch.cos(Own_yaw_angle), torch.sin(Own_yaw_angle)), axis = 1)
        
        # Go through all needed prediction timesteps
        imin = 1
        imax = 1 + n_action_time_steps.max()
        for i in range(imin, imax):
            # Check which paths still need to be updated
            update = n_action_time_steps >= i 
            
            # Calculate the moved distance
            Movement_distance = time_step * Own_planned_long_speed[update] + Own_planned_long_acc[update,i-1, :] * (time_step ** 2) / 2
            
            # Add the moved distance to the position
            Own_planned_pos[update] += torch.maximum(Movement_distance, time_step * self.min_speed)[:,None,:] * Own_heading_vect[update]
            
            # Update the speed, while keeping the speed limit intact
            Own_planned_long_speed[update] += Own_planned_long_acc[update,i-1] * time_step
            Own_planned_long_speed[update]  = torch.maximum(self.min_speed, Own_planned_long_speed[update])    
        
        # Convert position into signed distance to conflict point
        Own_planned_signed_CP_dist = torch.sum(Own_heading_vect * (self.simulation.conflict_point[None, :, None] - Own_planned_pos), 1)
        
        # Get time needed to enter and exit the contested space
        Own_planned_entry_time, Own_planned_exit_time = self.HELP_get_entry_exit_times(Own_planned_signed_CP_dist, 
                                                                                       self.coll_dist[:,None], 
                                                                                       Own_planned_long_speed)
        
        # Allocate memory for the final output Pred_own_states and set the results
        Pred_own_states = torch.zeros((5, Up_sum, self.n_actions), dtype = torch.float32, device = self.device)
        Pred_own_states[0] = Own_planned_long_speed 
        Pred_own_states[1] = Own_planned_long_acc[torch.arange(Up_sum, dtype = torch.int64, device = self.device), n_action_time_steps, :]
        Pred_own_states[2] = Own_planned_signed_CP_dist
        Pred_own_states[3] = Own_planned_entry_time
        Pred_own_states[4] = Own_planned_exit_time
        
        # Copy results over all possible bheaviors
        Pred_own_states = torch.tile(Pred_own_states[:,:,:,None], (1, 1, 1, self.n_beh))
        return Pred_own_states
        
    
    def DAU_get_implications(self, Up_sum, Up_param_index, own_ctrl_type, Check_useful,  
                             Own_pred_state, Own_coll_dist, Own_free_speed, 
                             Oth_pred_state, Oth_coll_dist):
        r'''
        This function calculates the acceleration needed and the time 
        needed to get from a input state to a goal where one either enters the 
        contested space first or second.
        It is called multiple times inside do_action_update(). 

        Parameters
        ----------
        Up_sum : torch.tensor
            The number of paths :math:`n_{path}`.
        Up_param_index : torch.tensor
            This is a integer :math:`n_{path}` dimensional tensor, which 
            links the path to corresponding parameter setting.
        own_ctrl_type : CtrlType
            This is the control type (speed or acceleration) of the agent for which 
            the values are calculated.
        Check_useful : torch.tensor
            This is a boolean :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It points out the combinations of path, action and behavior that has to be considered.
        Own_pred_state : torch.tensor
            This is a :math:`\{5 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It includes along its first dimension the speed, acceleration, signed distance to crossing point,
            the predicted time until entry into the conflicted space, and the predicted time until leaving the contested space.
            This is given for the own agent wanting to achieve a goal.
        Own_coll_dist : torch.tensor
            This is a :math:`n_{path}` dimensional tensor, which describes at what signed distance 
            the own agent enters and leaves the contested space.
        Own_free_speed : torch.tensor
            This is a :math:`n_{path}` dimensional tensor, which describesthe desired free speed of
            the agent.
        Oth_pred_state : torch.tensor
            This is a :math:`\{5 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It includes along its first dimension the speed, acceleration, signed distance to crossing point,
            the predicted time until entry into the conflicted space, and the predicted time until leaving the contested space.
            This is given for the other agent on whom the goal of the own agent is to be imposed.
        Oth_coll_dist : torch.tensor
            This is a :math:`n_{path}` dimensional tensor, which describes at what signed distance 
            the other agent enters and leaves the contested space.

        Returns
        -------
        Accs : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It gives the acceleration needed for the own agent to reach its goal (pass first or second).
        T_accs : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It gives the acceleration needed for the own agent to reach its goal (pass first or second).
        T_dws : torch.tensor
            DESCRIPTION.

        '''
        
        # Get useful indices
        U_path_index, U_act_index, U_beh_index = torch.where(Check_useful)
        
        # Get number of useful paths 
        U_sum = len(U_path_index)
        
        # Get current speed and position data
        own_long_speed     = Own_pred_state[0, U_path_index, U_act_index, U_beh_index]
        own_signed_CP_dist = Own_pred_state[2, U_path_index, U_act_index, U_beh_index]
        
        oth_long_speed     = Oth_pred_state[0, U_path_index, U_act_index, U_beh_index]
        oth_signed_CP_dist = Oth_pred_state[2, U_path_index, U_act_index, U_beh_index]
        
        # Get collision distance
        own_coll_dist      = Own_coll_dist[U_path_index]
        oth_coll_dist      = Oth_coll_dist[U_path_index]
        
        # Get further information
        own_free_speed     = Own_free_speed[U_path_index]
        oth_time_to_entry  = Oth_pred_state[3, U_path_index, U_act_index, U_beh_index]
        oth_time_to_exit   = Oth_pred_state[4, U_path_index, U_act_index, U_beh_index]
        
        
        (Accs_first, T_accs_first, T_dws_first, 
         Accs_second, T_accs_second, T_dws_second) = self.HELP_get_accs_towards_goal(U_sum, Up_param_index[U_path_index], 
                                                                                     own_ctrl_type, own_signed_CP_dist, 
                                                                                     own_coll_dist, own_long_speed, 
                                                                                     own_free_speed, oth_signed_CP_dist, 
                                                                                     oth_coll_dist, oth_long_speed, 
                                                                                     oth_time_to_entry, oth_time_to_exit)
        
        # Allocate memory for final output
        Accs   = torch.full((2, Up_sum, self.n_actions, self.n_beh), 
                             torch.nan, dtype = torch.float32, device = self.device)
        T_accs = torch.full((2, Up_sum, self.n_actions, self.n_beh), 
                             torch.nan, dtype = torch.float32, device = self.device)
        T_dws  = torch.full((2, Up_sum, self.n_actions, self.n_beh), 
                             torch.nan, dtype = torch.float32, device = self.device)
        
        # Set the final results
        Accs[0, Check_useful] = Accs_first
        Accs[1, Check_useful] = Accs_second
        
        T_accs[0, Check_useful] = T_accs_first
        T_accs[1, Check_useful] = T_accs_second
        
        T_dws[0, Check_useful] = T_dws_first
        T_dws[1, Check_useful] = T_dws_second
        return Accs, T_accs, T_dws
    
    
    def DAU_GAV_get_Action_acc(self, ctrl_type, Pred_state, curr_speed, curr_acc, 
                               Up_param_index, U1_path_index, U1_action_index, U1_beh_index, Up_sum):
        r'''
        This function determines the initial acceleration and jerk in the prediction phase. 
        It is callled inside DAU_get_action_values().

        Parameters
        ----------
        ctrl_type : CtrlType
            This is the control type (speed or acceleration) of the agent for which the values are
            calculated.
        Pred_state : torch.tensor
            This is a :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It entailes the predicted state of the agent. 
        curr_speed : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor, with the current speed of the agent.
        curr_acc : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor, with the current acceleration of the agent.
        Up_param_index : torch.tensor
            This is a integer :math:`n_{path}` dimensional tensor, which 
            links the path to corresponding parameter setting
        U1_path_index :  torch.tensor
            This is a integer :math:`n_{motion-plan}` dimensional tensor, which 
            links the motion plan to corresponding path
        U1_action_index :  torch.tensor
            This is a integer :math:`n_{motion-plan}` dimensional tensor, which 
            links the motion plan to corresponding action
        U1_beh_index :  torch.tensor
            This is a integer :math:`n_{motion-plan}` dimensional tensor, which 
            links the motion plan to corresponding behavior
        Up_sum : int
            The number of paths :math:`n_{path}`.

        Returns
        -------
        Action_acc0_v : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which gives the current acceleration 
            of the agent.
        Action_jerk_v : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which gives the increase in
            acceleration (i.e., the jerk) during the original prediction time.

        '''
        # Constant acceleration for speed control
        if ctrl_type is CtrlType.SPEED:
            # Assume Acceleration is constant during prediction period, interpolate
            Action_acc0 = ((Pred_state[0] - curr_speed[:, None, None]) / 
                           self.params.DeltaT[Up_param_index, None, None])
            # Assumption of constant acceleration leaves no jerk
            Action_jerk = torch.zeros((Up_sum, self.n_actions, self.n_beh), 
                                      dtype = torch.float32, device = self.device)
        # Increasing acceleration for acceleration control
        else:
            # Recover initial acceleration value
            Action_acc0 = torch.tile(curr_acc[:, None,None], (1, self.n_actions, self.n_beh))
            # Get the jerk by linear interpolation of acceleration
            Action_jerk = ((Pred_state[1] - curr_acc[:, None, None]) / 
                           self.params.DeltaT[Up_param_index, None, None])
        
        # Only cosider those values for actually useful motion plans
        Action_acc0_v = Action_acc0[U1_path_index, U1_action_index, U1_beh_index]
        Action_jerk_v = Action_jerk[U1_path_index, U1_action_index, U1_beh_index]
        return Action_acc0_v, Action_jerk_v
    
    
    def DAU_GAV_integrated_discounted_value(self, s, k_g, k_dv, k_da, T_start, Speed_start = None, Acc_start = None, Jerk_start = None, DT = None):
        r'''
        This is a function which allows tha analytical calculation of the discounted value
        over a given phase, where 
        
        .. math::
            D = \int\limits_{t_s}^{t_s + \Delta t} s^{\tau} \left( k_g v(\tau) - k_{dv} v(\tau)^2 - k_a a(\tau)^2  \right) d\tau
            
        This is done based on the assumption of constant jerk, leading to:
            
        .. math::
            & a(t) = j_s (t- t_s) + a_s \\
            & v(t) = {1\over{2}} j_s (t - t_s) ^2 + a_s(t-t_s) + v_s
        
        Putting this in, one then has to solve the following:
            
        .. math::
            D = & \int\limits_{t_s}^{t_s + \Delta t} s^{\tau} \left( \alpha_0 + \alpha_1 (t-t_s) 
                                                                    + \alpha_2 (t-t_s)^2  + \alpha_3 (t-t_s)^3 
                                                                    + \alpha_4 (t-t_s)^4  \right) d\tau \\
                & \alpha_0 = v_s \left(k_g - k_{dv} v_s\right) - k_a a_s^2 \\
                & \alpha_1 = a_s \left(k_g - 2 \left(k_a j_s + k_{dv} v_s \right)   \right) \\
                & \alpha_2 = {1\over{2}} j_s \left( k_g - 2 \left(k_a j_s + k_{dv} v_s\right) \right) - k_{dv}a_s^2 \\
                & \alpha_3 = - k_{dv} a_s j_s \\
                & \alpha_4 = - {1\over{4}} k_{dv} j_s^2 \\
            
        
        Analytical solutions are available:
            
        .. math::
            D = & + {s^{t_s} \over {\log (s)  }} \left(s^{\Delta t} \left(\alpha_4{\Delta t}^4 + \alpha_3{\Delta t}^3 + \alpha_2{\Delta t}^2 + \alpha_1{\Delta t}  + \alpha_0 \right) - \alpha_0 \right) \\
                & - {s^{t_s} \over {\log (s)^2}} \left(s^{\Delta t} \left(4 \alpha_4 {\Delta t}^3 + 3 \alpha_3 {\Delta t}^2 + 2 \alpha_2 {\Delta t} + \alpha_1 \right) -\alpha_1 \right) \\ 
                & + {s^{t_s} \over {\log (s)^3}} \left(s^{\Delta t} \left(12 \alpha_4 {\Delta t}^2 + 6 \alpha_3 {\Delta t} + 2 \alpha_2 \right) - 2 \alpha_2 \right) \\
                & - {s^{t_s} \over {\log (s)^4}} \left(s^{\Delta t} \left(24 \alpha_4 {\Delta t} + 6 \alpha_3 \right) - 6 \alpha_3 \right) \\
                & + {s^{t_s} \over {\log (s)^5}} \left(s^{\Delta t} \left(24 \alpha_4\right)  - 24 \alpha_4 \right) \\ 
        
        This function is called inside DAU_GAV_get_discounted_value_alternative()

        Parameters
        ----------
        s : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the time discount factor.
        k_g : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the scale factor 
            for rewarding high speed.
        k_dv : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the scale factor
            for punishing high speeds.
        k_da : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the scale factor
            for punishing high accelerations.
        T_start : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, that determines the 
            starting time of the phase to be evaluated.
        Speed_start : torch.tensor, optional
            This is a :math:`n_{motion-plan}` dimensional tensor, which gives the 
            speed at the start of the phase. The default is None.
        Acc_start : torch.tensor, optional
            This is a :math:`n_{motion-plan}` dimensional tensor, which gives the
            acceleration at the start of the phase. The default is None.
        Jerk_start : torch.tensor, optional
            This is a :math:`n_{motion-plan}` dimensional tensor, which gives the increase in
            acceleration (i.e., the jerk) at the start of the phase. The default is None.
        DT : torch.tensor, optional
            This is a :math:`n_{motion-plan}` dimensional tensor, that determines the
            duration of the phase to be evaluated. The default is None.

        Returns
        -------
        Discounted_value_phase : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which entails
            the integrated value of the discounted value for this phase.

        '''
        # Assume constant jerk
        if Jerk_start != None and Acc_start != None and Speed_start != None:
            # Calculate the polynomialy factors
            H = k_g - 2 * (k_da * Jerk_start + k_dv * Speed_start)
            alpha_0 = Speed_start * (k_g - k_dv * Speed_start) - k_da * Acc_start ** 2
            alpha_1 = Acc_start * H
            alpha_2 = Jerk_start * H * 0.5 - k_dv * Acc_start ** 2
            alpha_3 = - k_dv * Acc_start * Jerk_start
            alpha_4 = - 0.25 * k_dv * Jerk_start ** 2
            
            # Calculate inverse log
            ls1 = 1 / torch.log(s)
            # Calculate the respective powers of this
            ls2 = ls1 * ls1
            ls3 = ls2 * ls1
            ls4 = ls3 * ls1
            ls5 = ls4 * ls1
            
            # Get start exponential
            sts = s ** T_start
            
            if DT != None:
                # Precalculate polynomials
                sDt = s ** DT
                DT2 = DT * DT
                DT3 = DT2 * DT
                DT4 = DT3 * DT
                
                # Calculate discounted value
                Discounted_value_phase  = ls1 * sts * (sDt * (alpha_4 * DT4 + alpha_3 * DT3 + alpha_2 * DT2 
                                                             + alpha_1 * DT + alpha_0) - alpha_0)
                Discounted_value_phase -= ls2 * sts * (sDt * (4 * alpha_4 * DT3 + 3 * alpha_3 * DT2 
                                                              + 2 * alpha_2 * DT + alpha_1) - alpha_1)
                Discounted_value_phase += ls3 * sts * (sDt * (12 * alpha_4 * DT2 + 6 * alpha_3 * DT + 2 * alpha_2) - 2 * alpha_2)
                Discounted_value_phase -= ls4 * sts * (sDt * (24 * alpha_4 * DT + 6 * alpha_3) - 6 * alpha_3)
                Discounted_value_phase += ls5 * sts * ((sDt - 1) * 24 * alpha_4)
            else:
                # If Dt is infinite, than sDt = 0 can be assumed
                Discounted_value_phase  = ls1 * sts * (- alpha_0)
                Discounted_value_phase -= ls2 * sts * (- alpha_1)
                Discounted_value_phase += ls3 * sts * (- 2 * alpha_2)
                Discounted_value_phase -= ls4 * sts * (- 6 * alpha_3)
                Discounted_value_phase += ls5 * sts * (- 24 * alpha_4)
        
        # assume constant acceleration
        elif Jerk_start == None and Acc_start != None and Speed_start != None:
            # Calculate the polynomialy factors
            H = k_g - 2 * k_dv * Speed_start
            alpha_0 = Speed_start * (k_g - k_dv * Speed_start) - k_da * Acc_start ** 2
            alpha_1 = Acc_start * H
            alpha_2 = - k_dv * Acc_start ** 2
            
            # Calculate inverse log
            ls1 = 1 / torch.log(s)
            # Calculate the respective powers of this
            ls2 = ls1 * ls1
            ls3 = ls2 * ls1
            
            # Get start exponential
            sts = s ** T_start
            
            if DT != None:
                # Precalculate polynomials
                sDt = s ** DT
                DT2 = DT * DT
                
                # Calculate discounted value
                Discounted_value_phase  = ls1 * sts * (sDt * (alpha_2 * DT2 + alpha_1 * DT + alpha_0) - alpha_0)
                Discounted_value_phase -= ls2 * sts * (sDt * (2 * alpha_2 * DT + alpha_1) - alpha_1)
                Discounted_value_phase += ls3 * sts * ((sDt - 1) * 2 * alpha_2)
            else:
                # If Dt is infinite, than sDt = 0 can be assumed
                Discounted_value_phase  = ls1 * sts * (- alpha_0)
                Discounted_value_phase -= ls2 * sts * (- alpha_1)
                Discounted_value_phase += ls3 * sts * (- 2 * alpha_2)
        
        # assume constant velocity
        elif Jerk_start == None and Acc_start == None and Speed_start != None:
            # Calculate the polynomialy factors
            alpha_0 = Speed_start * (k_g - k_dv * Speed_start)
            
            # Calculate inverse log
            ls1 = 1 / torch.log(s)
            
            # Get start exponential
            sts = s ** T_start
            if DT != None:
                # Precalculate polynomials
                sDt = s ** DT
                
                # Calculate discounted value
                Discounted_value_phase  = ls1 * sts * (sDt - 1) * (alpha_0)
            else:
                # If Dt is infinite, than sDt = 0 can be assumed
                Discounted_value_phase  = - ls1 * sts * alpha_0     
        else:
            raise TypeError('This should not have happened')
        return Discounted_value_phase
    
    
    def DAU_GAV_get_discounted_value(self, U1_sum, U1_param_index, k_g, k_dv, k_da, DT2, DT3, 
                                     Speed_T0, Acc_T0, Jerk_T0, Acc_T1, Speed_T4):
        r'''
        This function determines the time discounted kinematic value of a combination of action and behavior. 
        It is callled inside DAU_get_action_values().

        Parameters
        ----------
        U1_sum : int
            The number of motion plans :math:`n_{motion-plan}` considered, based on different paths,
            actions, behaviors, and goals of the agent (passing first or second).
        U1_param_index : torch.tensor
            This is a integer :math:`n_{motion-plan}` dimensional tensor, which 
            links the motion plan to corresponding parameter setting
        k_g : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the scale factor 
            for rewarding high speed.
        k_dv : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the scale factor
            for punishing high speeds.
        k_da : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, the scale factor
            for punishing high accelerations.
        DT2 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which describes the 
            time the acceleration towards the goal state (Acc_T1) has to be applied 
            to go either first or second.
        DT3 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, that describes how 
            long the agent stays at the achieved speed after the acceleration, before
            accelerating towards the desired speed.
        Speed_T0 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which 
            gives the speed at time :math:`t_0`.
        Acc_T0 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which 
            gives the acceleration at time :math:`t_0`.
        Jerk_T0 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which gives the increase in
            acceleration (i.e., the jerk) at time :math:`t_0`.
        Acc_T1 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which 
            gives the acceleration at time :math:`t_1`.
        Speed_T4 : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, which 
            gives the desired free speed of the agent, the speed reached at time at time :math:`t_4`.

        Returns
        -------
        Discounted_value : torch.tensor
            This is a :math:`n_{motion-plan}` dimensional tensor, providing the
            discounted kinematc value of all proposed motion plans. This also includes 
            the post simulation discounted value.

        '''
        # Assume T0 = 0
        T0 = torch.zeros(U1_sum, dtype = torch.float32, device = self.device)
        
        # First phase has a duration that is predescribed
        DT1 = self.params.DeltaT[U1_param_index]
        T1 = T0 + DT1
        
        # Second phase has a precalculated duration
        T2 = T1 + DT2
        
        # The third phase has also a precalculated duration
        T3 = T2 + DT3
        
        # The third phase's duration is also predesribed, depending on control type
        if self.ctrl_type is CtrlType.SPEED:
            DT4 = DT1
            T4 = T3 + DT1
        else:
            DT4 = self.params.T_acc_regain_spd
            T4 = T3 + self.params.T_acc_regain_spd
        
        # Note: T5 is assumed to be infinity
        
        # Get remaining kinematics values, if not mentioned they are assumed to be zero
        # Phase I has constant jerk
        Speed_T1 = Speed_T0 + Acc_T0 * DT1 + 0.5 * Jerk_T0 * DT1 ** 2
        
        # Phase II has constant acceleration (constant speed in Phase III)
        Speed_T2_T3 = Speed_T1 + Acc_T1 * DT2
        
        # Phase IV has constant acceleration
        Acc_T3 = (Speed_T4 - Speed_T2_T3) / DT4
        
        # Get the time discount factor
        S = 2 ** (- 1 / self.params.T_delta[U1_param_index])
        
        # Get the discounted value
        # Phase I has constant jerk
        Discounted_value  = self.DAU_GAV_integrated_discounted_value(S, k_g, k_dv, k_da, T0, Speed_T0, Acc_T0, Jerk_T0, DT = DT1)
        # Phase II has constant acceleration
        Discounted_value += self.DAU_GAV_integrated_discounted_value(S, k_g, k_dv, k_da, T1, Speed_T1, Acc_T1, DT = DT2)
        # Phase III has constant speed
        Discounted_value += self.DAU_GAV_integrated_discounted_value(S, k_g, k_dv, k_da, T2, Speed_T2_T3, DT = DT3)
        # Phase IV has constant acceleration
        Discounted_value += self.DAU_GAV_integrated_discounted_value(S, k_g, k_dv, k_da, T3, Speed_T2_T3, Acc_T3, DT = DT4)
        # Phase V has constant speed
        Discounted_value += self.DAU_GAV_integrated_discounted_value(S, k_g, k_dv, k_da, T4, Speed_T4)
        return Discounted_value
        
      
    def DAU_get_action_values(self, Not_needed, Accs, T_accs, T_dws,
                              ctrl_type, v_free, k_g, k_dv, k_da, u_ny,
                              Pred_state, curr_speed, curr_acc):
        r'''
        This function determines the value that is assigned by an agent to a certain 
        combination of actions and behaviors, depending on if they want to go first or second. 
        It is callled multiple times inside do_action_update().
        
        DAU_GAV = DAU_get_action_values abbreviation

        Parameters
        ----------
        Not_needed : torch.tensor
            This is a boolean :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It points out unneeded action-beahvior-goal combinations.
        Accs : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes the acceleration necessary to go either first or second, depending action and behavior combination.
        T_accs : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes the time the acceleration above has to be applied 
            to go either first or second, depending action and behavior combination.
        T_dws : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes how long the agent stays at the achieved speed after the acceleration, before
            accelerating towards the desired speed.
        ctrl_type : CtrlType
            This is the control type (speed or acceleration) of the agent for which the values are
            calculated.
        v_free : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor, which holds the desired free speed
            of the agent.
        k_g : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples}\}` dimensional tensor, the scale factor 
            for rewarding high speed.
        k_dv : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples}\}` dimensional tensor, the scale factor
            for punishing high speeds.
        k_da : torch.tensor
            This is a :math:`n_{params}` dimensional tensor, the scale factor
            for punishing high accelerations.
        u_ny : torch.tensor
            This is a :math:`n_{params}` dimensional tensor, the added value for going first.
        Pred_state : torch.tensor
            This is a :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It entailes the predicted state of the agent. 
        curr_speed : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor, with the current speed of the agent.
        curr_acc : torch.tensor
            This is a :math:`n_{paths}` dimensional tensor, with the current acceleration of the agent.

        Returns
        -------
        Access_order_values : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It provides the value the agent gives a combination of action and behavior, where it
            either decides to go first or second (first dimension).

        '''
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine the number of those paths
        Up_sum = len(Up_sample_index)
        
        # Allocate memory for the values
        Access_order_values = torch.full((2, Up_sum, self.n_actions, self.n_beh), 
                                         torch.nan, dtype = torch.float32, device = self.device)
        
        # Set useful values to - infinity as a preperation
        Access_order_values[:, self.U_L] = -torch.inf
        
        # Find those useful paths were new values can be assigned
        U1 = self.U_L.unsqueeze(0) & ~(torch.isnan(Accs) | Not_needed)
        
        # Determine the number of paths that are to be overridden
        U1_sum = U1.sum()
        
        # Check if there are any such paths to consider
        if U1_sum > 0:
            # Determine the inides of those paths
            U1_access_index, U1_path_index, U1_action_index, U1_beh_index = torch.where(U1)
            
            # Find the corresponding higher level indeces
            U1_param_index  = Up_param_index[U1_path_index]
            U1_sample_index = Up_sample_index[U1_path_index]
            
            # Determine initially planned accelerations
            Action_acc0_v, Action_jerk_v = self.DAU_GAV_get_Action_acc(ctrl_type, Pred_state, curr_speed, curr_acc, Up_param_index,
                                                                       U1_path_index, U1_action_index, U1_beh_index, Up_sum)
            
            # Get the kinematic values assigned for simulated behavior (with time discount)
            Discounted_value = self.DAU_GAV_get_discounted_value(U1_sum, U1_param_index, 
                                                                 k_g[U1_param_index, U1_sample_index],
                                                                 k_dv[U1_param_index, U1_sample_index],
                                                                 k_da[U1_param_index],
                                                                 T_accs[U1], T_dws[U1], curr_speed[U1_path_index],
                                                                 Action_acc0_v, Action_jerk_v, Accs[U1],
                                                                 v_free[U1_path_index])
            
            # Get negative value for going first (zero if priority does not have to be given)
            Inh_access_value = torch.zeros((self.n_params, 2), dtype = torch.float32, device = self.device)
            Inh_access_value[:, 0] = u_ny
            
            # Add up all values
            Total_value = Inh_access_value[U1_param_index, U1_access_index] + Discounted_value
            
            # Scale this value with value of free driving
            Access_order_values[U1] = Total_value / self.params.u_0[U1_param_index]
        
        # Use tanh for scaling
        Access_order_values[:, self.U_L] = torch.tanh(Access_order_values[:, self.U_L])
        return Access_order_values
        
        
    def DAU_beh_probs_given_actions(self, Access_order_values_oth):
        r'''
        This function determines the likelihood of the other agent choosing a certain behavior
        under a given action by the current agent. 
        It is callled inside do_action_update().
        
        Parameters
        ----------
        Access_order_values_oth : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It provides the value the other agent gives a combination of action and behavior, where it
            either decides to go first or second (first dimension).

        Returns
        -------
        Beh_value_given_actions_new : torch.tensor
            This is a :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes how likely a certain it is for a behavior to get choosen by the other agent given 
            an action by the current agent.
        Beh_probs_given_actions : torch.tensor
            This is a :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes how likely a certain it is for a behavior to get choosen by the other agent given 
            an action by the current agent.

        '''
        # get current time step
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine the number of those paths
        Up_sum = len(Up_sample_index)
        
        # Determine the paths were investigation is actually needed
        U_L_path_index,  U_L_act_index,  U_L_beh_index  = torch.where(self.U_L)
        U_L_param_index = Up_param_index[U_L_path_index]
        
        # Get the decay factor for the accumulation model
        f_T = self.simulation.settings.time_step / self.params.T[U_L_param_index]
        ##########################################################################################
        
        # Choose between going first and going second, based on higher value
        Beh_values_given_actions_current = torch.fmax(Access_order_values_oth[0, self.U_L], Access_order_values_oth[1, self.U_L])
         
        # Get the previous values assigned to action and behavior combinations
        Beh_value_given_actions_old = self.beh_value_given_actions[Up_param_index, Up_sample_index, Up_path_index, :, :, i_time_step - 1]
        
        # Get the new value assigned to action and behavior combinations, by accumulating values
        Beh_value_given_actions_new = torch.zeros(Beh_value_given_actions_old.shape, dtype = torch.float32, device = self.device)
        Beh_value_given_actions_new[self.U_L] = ((1 - f_T) * Beh_value_given_actions_old[self.U_L] + f_T * Beh_values_given_actions_current)
                                                             
        # Multiply the values with the softmax parameter (the higher, the more 
        # likely is the function with the highest value)                                                
        Beh_activations_given_actions = torch.full((Up_sum, self.n_actions, self.n_beh), 
                                                   - torch.inf, dtype = torch.float32, device = self.device)
        Beh_activations_given_actions[self.U_L] = (self.params.beta_V[U_L_param_index] * Beh_value_given_actions_new[self.U_L])
        
        # Determine the probability of a behavior being chosen
        Beh_probs_given_actions = torch.nn.Softmax(dim = -1)(Beh_activations_given_actions)
        return Beh_value_given_actions_new, Beh_probs_given_actions
    
    
    def DAU_choose_action(self, Access_order_values_own, Beh_probs_given_actions, Mom_action_useful):
        r'''
        This function chooses the action to take out of the set of available action,
        by trying to maximize the accumaleted expected value. 
        It is callled inside do_action_update().
        
        Parameters
        ----------
        Access_order_values_own : torch.tensor
            This is a :math:`\{2 \times n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It provides the value the current agent gives a combination of action and behavior, where it
            either decides to go first or second (first dimension).
        Beh_probs_given_actions : torch.tensor
            This is a :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes how likely a certain it is for a behavior to get choosen by the other agent given 
            an action by the current agent.
        Mom_action_useful : torch.tensor
            This is a boolean :math:`\{n_{paths} \times n_{actions} \times n_{behavior}\}` dimensional tensor.
            It describes which combinations of actions and behaviors are actually woth a consideration.

        Returns
        -------
        Est_action_vals_new : torch.tensor
            This is a :math:`\{n_{paths} \times n_{actions}\}` dimensional tensor.
            It provides the current accumulated value assigned by the current agents to the current behavior.
        Action_long_accs_new : torch.tensor
            This is a :math:`\{n_{paths} \times n_{timesteps}\}` dimensional tensor.
            It shows the updated planned acceleration based on the curently choosen action.
        action_changed : torch.tensor
            This is a boolean :math:`n_{paths}` dimensional tensor.    
            It describes for which of the feasible paths the current agent decided
            to pursue a different action. 

        '''
        # get current time step
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine the number of those paths
        Up_sum = len(Up_sample_index)
        
        # Get the decay factor for the accumulation model
        f_T = (self.simulation.settings.time_step / self.params.T[Up_param_index]).unsqueeze(-1) 
        
        # Decide between going first or second for any combination of own actiona and other's behavior
        Action_vals_given_behs = torch.fmax(Access_order_values_own[0,Mom_action_useful], Access_order_values_own[1,Mom_action_useful])
        
        # Get the expected value of an action over all probabilities
        Expected_action_vals = torch.zeros((Up_sum, self.n_actions, self.n_beh), 
                                      dtype = torch.float32, device = self.device)
        Expected_action_vals[Mom_action_useful] = Beh_probs_given_actions[Mom_action_useful] * Action_vals_given_behs
        Expected_action_vals = torch.sum(Expected_action_vals, axis = - 1)
        
        # Get the previously accumualted value of an action
        Est_action_vals_old = self.est_action_vals[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step-1]
        
        # Get the new accumualted value of an action
        Est_action_vals_new = (1 - f_T) * Est_action_vals_old + f_T * Expected_action_vals
        
        # Get the improvement of possible actions over current action
        Est_action_vals_improvement = Est_action_vals_new - Est_action_vals_new[:, self.i_no_action].unsqueeze(-1)
        
        # Find the best action and its corresponding improvement
        vals_best_action, i_best_action = torch.max(Est_action_vals_improvement, axis = - 1)
        
        # Determine if the improvement is sufficient for changing the action
        action_changed = vals_best_action > self.params.DeltaV_th[Up_param_index]
        
        # Get number of time steps in prediction period
        n_action_time_steps = self.n_action_time_steps[Up_param_index]
        
        Action_long_accs_new = self.action_long_accs[Up_param_index, Up_sample_index, Up_path_index, i_time_step:]
        
        # Find actual time steps over which constant control input is to be applied
        U_AS = (n_action_time_steps.unsqueeze(-1) > torch.arange(Action_long_accs_new.shape[1], device = self.device).unsqueeze(0))
        U_AS_path_index, U_AS_time_index   = torch.where(U_AS    & action_changed.unsqueeze(-1))
        Un_AS_path_index, Un_AS_time_index = torch.where((~U_AS) & action_changed.unsqueeze(-1))
        
        # Constant acceleration for speed control
        if self.ctrl_type is CtrlType.SPEED:
            acc_value = (self.params.ctrl_deltas[i_best_action] / self.params.DeltaT[Up_param_index])
            Action_long_accs_new[U_AS_path_index, U_AS_time_index] += acc_value[U_AS_path_index]
        # Increasing acceleration for acceleration control  
        else:
            # Get step values during increase
            acc_delta  = self.params.ctrl_deltas[i_best_action]
            acc_linear = U_AS_time_index.to(dtype = torch.float32) / (n_action_time_steps[U_AS_path_index] - 1)
            
            # Add additionally planned inputs
            Action_long_accs_new[U_AS_path_index, U_AS_time_index]   += acc_linear * acc_delta[U_AS_path_index]
            Action_long_accs_new[Un_AS_path_index, Un_AS_time_index] += acc_delta[Un_AS_path_index]
        return Est_action_vals_new, Action_long_accs_new, action_changed
    
    
    def do_action_update(self):
        '''
        The acceleration control value is determined.
        
        DAU = do_action_update abbreviation

        '''
        # Get current timestep
        i_time_step = self.simulation.state.i_time_step
        
        # Determine still interesting paths
        Up_param_index, Up_sample_index, Up_path_index = torch.where(self.U_open_paths) 
        
        # Determine number of still interesting paths
        Up_sum = len(Up_sample_index)
        
        # Check if constant acceleration is given
        if self.const_acc != None:
            # If yes, assign the same constant acceleration to all paths
            self.traj_long_acc[Up_param_index, Up_sample_index, Up_path_index, i_time_step] = self.const_acc
        else: 
            # If not, update the perception and long-term behavior
            if self.debug:
                print('')
                print('    At time step {:2.0f} for agent'.format(i_time_step) + self.name)
                print('    Current allocated memory: {:11.2f} MB'.format(torch.cuda.memory_allocated() / 2 ** 20))
        
            # Update the estimated state, perceived state, and covariance matrix
            mean_estimated, mean_perceived, cov_matrix_new = self.DAU_update_perception()  
            self.perc_x_estimated[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]   = mean_estimated
            self.perc_x_perceived[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]   = mean_perceived  
            self.perc_cov_matrix[Up_param_index, Up_sample_index, Up_path_index, :, :, i_time_step] = cov_matrix_new
        
            # Update the short-term acceleration for other agent, depending on behavior
            Accs_oth_first, Accs_oth_second = self.DAU_beh_long_acc_update()
            self.beh_long_accs[Up_param_index, Up_sample_index, Up_path_index, self.simulation.i_CONSTANT, i_time_step] = 0 
            self.beh_long_accs[Up_param_index, Up_sample_index, Up_path_index, self.simulation.i_PASS1ST,  i_time_step] = Accs_oth_first
            self.beh_long_accs[Up_param_index, Up_sample_index, Up_path_index, self.simulation.i_PASS2ND,  i_time_step] = Accs_oth_second
        
            # Predict the states of other agents
            Pred_oth_states = self.DAU_predict_oth()

            # Update the short-term acceleration for current agent
            Action_long_accs_corrected = self.DAU_action_long_acc_update()
            self.action_long_accs[Up_param_index, Up_sample_index, Up_path_index, i_time_step:] = Action_long_accs_corrected
            
            # Predict the state of the current agent
            Pred_own_states = self.DAU_predict_own()
        
            # Determine the paths which can be examined further
            U = ~torch.isnan(self.beh_long_accs[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step])
            
            # Exclude constant behavior if other behavior is permissible (Why?)
            # and set beh accs to zero accordingly
            Useless_constant = U[:, self.simulation.i_PASS1ST] | U[:, self.simulation.i_PASS2ND]
            U[Useless_constant, self.simulation.i_CONSTANT] = False
            self.beh_long_accs[Up_param_index[Useless_constant], 
                               Up_sample_index[Useless_constant], 
                               Up_path_index[Useless_constant], 
                               self.simulation.i_CONSTANT, i_time_step] = torch.nan
        
            # Mirror the further examinable paths over all possible actions
            self.U_L  = torch.tile(U[:,None,:], (1, self.n_actions, 1))
            
            # Get the future target accelerations of other aggent for all possible behaviors and actions,
            # depending on if the want to pass first or second
            Accs_oth, T_accs_oth, T_dws_oth = self.DAU_get_implications(Up_sum, Up_param_index, 
                                                                        self.other_agent.ctrl_type, 
                                                                        self.U_L, 
                                                                        Pred_oth_states, 
                                                                        self.oth_coll_dist, 
                                                                        self.oth_v_free, 
                                                                        Pred_own_states, 
                                                                        self.coll_dist)
            
            # Exclude from further consideration the cases where the order of the 
            # other agent going first or last does not match up in 0-th and 3-rd dimension 
            Oth_not_needed = torch.zeros(Accs_oth.shape, dtype = torch.bool, device = self.device)
            Oth_not_needed[1,:, :, self.simulation.i_PASS1ST] = True
            Oth_not_needed[0,:, :, self.simulation.i_PASS2ND] = True
            
            if self.debug:
                print('    Evaluate other:')
            
            # Get the value the other agents assignes to each combination of action and behavior
            Access_order_values_oth = self.DAU_get_action_values(Oth_not_needed, Accs_oth, 
                                                                 T_accs_oth, T_dws_oth, 
                                                                 self.other_agent.ctrl_type, 
                                                                 self.oth_v_free, 
                                                                 self.other_agent.params.k_g, 
                                                                 self.other_agent.params.k_dv, 
                                                                 self.other_agent.params.k_da,
                                                                 self.other_agent.params.u_ny,
                                                                 Pred_oth_states, 
                                                                 self.perc_x_perceived[Up_param_index, 
                                                                                       Up_sample_index, 
                                                                                       Up_path_index, 
                                                                                       1, i_time_step], 
                                                                 self.other_agent.Curr_state[1])
            # Empty GPU cache
            torch.cuda.empty_cache()
            
            # Get values and probabilities assigned to certain behavior by other agent given actions of current agents
            Beh_value_given_actions_new, Beh_probs_given_actions = self.DAU_beh_probs_given_actions(Access_order_values_oth)
            
            # Determine which combinations of actions and behaviors are useful 
            # and have a probability greater than the minimum threshold
            
            # Get maximum behavior threshold
            Beh_probs_threshold = min(self.params.min_beh_prob, 1 / self.n_beh)
            Mom_action_useful = self.U_L & (Beh_probs_given_actions > Beh_probs_threshold)
            
            # Get the future target accelerations of current aggent for all possible behaviors and actions
            # depending on if the want to pass first or second
            Accs_own, T_accs_own, T_dws_own = self.DAU_get_implications(Up_sum, Up_param_index,
                                                                        self.ctrl_type, 
                                                                        Mom_action_useful, 
                                                                        Pred_own_states, 
                                                                        self.coll_dist, 
                                                                        self.v_free, 
                                                                        Pred_oth_states, 
                                                                        self.oth_coll_dist)
            
            # Exclude from further consideration the cases which are impropable 
            Own_not_needed = torch.tile(~(Mom_action_useful), (2, 1, 1, 1))
            
            if self.debug:
                print('    Evaluate self:')
            
            # Get the value the current agents assignes to each combination of action and behavior
            Access_order_values_own = self.DAU_get_action_values(Own_not_needed, Accs_own, 
                                                                 T_accs_own, T_dws_own, 
                                                                 self.ctrl_type, 
                                                                 self.v_free, 
                                                                 self.params.k_g, 
                                                                 self.params.k_dv, 
                                                                 self.params.k_da,
                                                                 self.params.u_ny,
                                                                 Pred_own_states, 
                                                                 self.Curr_state[0], 
                                                                 self.Curr_state[1])
            # Empty GPU cache
            torch.cuda.empty_cache()
            
            # Determine the acceleration the current agents chooses based on the evaluation of own actions for different behavior of 
            # the behavior of other agents and the likelihood of those behaviors
            [Est_action_vals_new, 
             Action_long_accs_new, 
             action_changed] = self.DAU_choose_action(Access_order_values_own, Beh_probs_given_actions, Mom_action_useful)
            
            # Reset Values of paths where the action was changed
            Est_action_vals_new[action_changed] = 0
            Beh_value_given_actions_new[action_changed] = 0
            
            # Set the accumulated values and the new chosen acceleration
            self.beh_value_given_actions[Up_param_index, Up_sample_index, Up_path_index, :, :, i_time_step] = Beh_value_given_actions_new
            self.est_action_vals[Up_param_index, Up_sample_index, Up_path_index, :, i_time_step]            = Est_action_vals_new
            self.action_long_accs[Up_param_index, Up_sample_index, Up_path_index, i_time_step:]             = Action_long_accs_new
            self.traj_long_acc[Up_param_index, Up_sample_index, Up_path_index, i_time_step]                 = Action_long_accs_new[:, 0]
            

class SCSimulation():
    '''
    This class regulates the interaction between the two agents
    
    '''
    
    def __init__(self, device, n_agents, agent_names, ctrl_types,
                 initial_positions, initial_speeds, initial_accs, coll_dist, free_speeds,
                 goal_positions, conflict_point, give_priority, 
                 const_accs, stop_criteria, fixed_params, variable_params,
                 num_paths = 1, start_time = 0, end_times = [10,], time_step = 0.1,
                 noise_seed = 0, zero_acc_after_exit = False, plot_colors = ('c', 'm'), debug = False):
        
        # set the current torch device on which this is model is done
        self.device = device
        
        # Set random generator noise seed for the simulation
        torch.manual_seed(noise_seed)
        
        # set the stopping criteria
        self.stop_criteria = stop_criteria
        
        # Set the possible behavaiors and count the possibilities
        self.BEHAVIORS = ('Const.', 'Pass 1st', 'Pass 2nd')
        self.N_BEHAVIORS = len(self.BEHAVIORS)
        
        # Set the corresponding indices of these behaviors
        self.i_CONSTANT = 0
        self.i_PASS1ST = 1
        self.i_PASS2ND = 2
        
        # set the number of different phases during simulations
        self.N_ANTICIPATION_PHASES = 5
        # set the indices of these different pahases
        self.i_ACTION = 0
        self.i_ACH_ACCESS = 1
        self.i_WAIT = 2
        self.i_REGAIN_SPD = 3
        
        # set the time after which a certain simulation ends (sample dependent)
        self.end_times = end_times
        
        # Create the time creation class
        self.settings = SimulationSettings(torch.tensor(start_time, dtype = torch.float32, device = self.device), 
                                           torch.max(self.end_times), time_step)
        self.state = SimulationState(self)
        
        # Create the agents participating in this simulation
        self.agents = []
        for i_agent in range(n_agents):
            # Initialize each agent
            SCAgent(device              = self.device, 
                    name                = agent_names[i_agent], 
                    ctrl_type           = ctrl_types[i_agent], 
                    coll_dist           = coll_dist[i_agent],
                    free_speeds         = free_speeds[i_agent],
                    simulation          = self, 
                    goal_pos            = goal_positions[i_agent, :], 
                    conflict_point      = conflict_point,
                    initial_pos         = initial_positions[i_agent,:], 
                    initial_long_speed  = initial_speeds[i_agent],  
                    initial_long_accs   = initial_accs[i_agent], 
                    initial_yaw_angle   = None, 
                    fixed_params        = fixed_params,
                    variable_params     = variable_params,
                    give_priority       = give_priority[i_agent],
                    num_paths           = num_paths,
                    const_acc           = const_accs[i_agent],
                    zero_acc_after_exit = zero_acc_after_exit,
                    plot_color          = plot_colors[i_agent],
                    debug               = debug)
        

    def run(self):
        '''
        Execute the simulation for the required cases.

        Returns
        -------
        None.

        '''
        # Extract agent names
        agent_names = [agent.name for agent in self.agents]
        # Find index of the target agent
        i_agent_tar = np.where([name[-3:] == 'tar' for name in agent_names])[0][0]
        
        # Prepare agents for simulation
        for agent in self.agents:
            agent.prepare_for_simulation()
    
        # Initialize stop_now flag to False
        self.stop_now = False
        
        # Loop through time steps
        for i_time_step in range(self.settings.n_time_steps):
            # Set time step for state
            self.state.set_time_step(i_time_step)
            
            # Update kinematics of agents after first time step
            if i_time_step > 0:
                for agent in self.agents:
                    agent.do_kinematics_update()
                    
            # Prepare agents for action update
            for agent in self.agents:
                agent.prepare_for_action_update()
            
            # Update actions of agents
            for agent in self.agents:
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                agent.do_action_update()
            
            # Get path currently still simulated
            still_needed_old = self.agents[0].U_open_paths
            
            # Check which paths still need to be updated based on end times
            time_not_up = self.end_times >= (i_time_step + 1) * self.settings.time_step
            Time_not_up = time_not_up[torch.where(still_needed_old)[1]]
            
            # Check if stop criteria has been met
            for stop_crit in self.stop_criteria:
                # Assert that this stop criterum can actually be considered
                assert list(stop_crit.keys())[0] == 'Accepted', f'Unexpected simulation stop criterion: {stop_crit}'
                
                # Get target agent's current position
                pos = self.agents[i_agent_tar].traj_pos[:, :, :, 0, self.state.i_time_step]
                
                # Get acceptance position for the target agent
                pos_acc = torch.tile(list(stop_crit.values())[0][:, None], (pos.shape[0], 1, pos.shape[2]))
                
                # Find paths where the gap is not yet accepted and end time not yyet reached
                update = (pos[still_needed_old] >= pos_acc[still_needed_old]) & Time_not_up
                
                # Updated list of paths still needed
                still_needed = torch.zeros(still_needed_old.shape, dtype=torch.bool, device=self.device)
                still_needed[still_needed_old] = update
                
                # Send the updated list of still initeresting paths to the agents
                for agent in self.agents:
                    agent.U_open_paths = still_needed
                
                # Check if all paths have been closed
                if not torch.any(still_needed):
                    self.stop_now = True
                
            # Stop simulation if stop criteria has been met
            if self.stop_now:
                break
        
        # Save actual end time where all paths have been closed
        self.actual_end_time = self.state.time
        
    
    def empty(self):
        # Remove all memory occupied on gpu during previous simulation
        for agent in self.agents:
            attributes = list(agent.__dict__.keys())
            for attr in attributes:
                delattr(agent, attr)
            del agent
        
        
class Commotions_nn(torch.nn.Module):
    '''
    This class takes a certain scenario and allows the repeated running of different variations of this scenario.
    
    '''
    def __init__(self, device, num_samples_path_pred, params, p_quantile):
        super(Commotions_nn, self).__init__()
        
        # set the current torch device on which this is model is done
        self.device = device
        # Set the number of agents involved in the whole scenario
        self.N_AGENTS = 2 
        # Set the number of differnt path used for each initial condition and parameter set
        # to represent the stochastic nature of the humans behavior
        self.num_samples_path_pred = num_samples_path_pred
        # Get the current nonvariable parameter that influence the simulation
        self.fixed_params = params
        # Define the quantile values of the probability distribution of the time of accepting
        # the offered gap which should be used to represent this distribution
        self.p_quantile = torch.from_numpy(p_quantile).to(dtype = torch.float32, device = self.device)
        
    
    def _get_zero_point(self, D, T_pred):
        complete = torch.min(torch.nan_to_num(D, D[...,0].max() * 2), -1)[0] < 0
        
        Ind = torch.argmax(((torch.isfinite(D) & (D < 0)) |
                            torch.isnan(D)).to(torch.float32), dim=-1).to(torch.int64)
        Ind[~complete & (Ind > 0)] -= 1
        
        assert torch.isfinite(D[Ind == 0][:,-1]).all() 
        Ind[Ind == 0] = len(T_pred) - 1
        assert Ind.min() > 0
        
        I1 = torch.tile(torch.arange(D.shape[0], device = self.device)[:,None,None], (1, D.shape[1], D.shape[2]))
        I2 = torch.tile(torch.arange(D.shape[1], device = self.device)[None,:,None], (D.shape[0], 1, D.shape[2]))
        I3 = torch.tile(torch.arange(D.shape[2], device = self.device)[None,None,:], (D.shape[0], D.shape[1], 1))
        
        D0 = D[I1, I2, I3, Ind - 1] + 1e-3
        D1 = D[I1, I2, I3, Ind]
        
        assert (D1 < D0).all()
        
        T0 = T_pred[Ind - 1]
        T1 = T_pred[Ind]
        
        T_gap = T0 - D0 * (T1 - T0) / (D1 - D0)
        return T_gap

    def forward(self, names, ctrl_types, params, initial_positions, speeds, accs, 
                coll_dist, free_speeds, T_out, dt, const_accs):
        r'''
        Setting up the simulation of the humans behavior

        Parameters
        ----------
        names : list
            A list of strings naming the agents involved in teh scenario.
        ctrl_types : list
            A list of comtrol types the agents in the scenario follow 
            (speed control => pedestrian, acceleration control => vehicle).
        params : torch.tensor
            This is a :math:`\{n_{params} \times 9\}` dimensional tensor with 
            the different parameter settings considered in all scenarios.
        initial_positions : torch.tensor
            This is a :math:`\{2 \times n_{samples} \times 2\}` dimensional tensor with
            the different initial position of each agent in each scenario.
        speeds : torch.tensor
            This is a :math:`\{2 \times n_{samples}\}` dimensional tensor with
            the different initial speed of each agent in each scenario.
        accs : torch.tensor
            This is a :math:`\{2 \times n_{samples}\}` dimensional tensor with
            the different initial accelerations of each agent in each scenario.
        coll_dist : torch.tensor
            This is a :math:`\{2 \times n_{samples}\}` dimensional tensor with
            the distance to the conflicted space of each agent in each scenario.
        free_speeds : torch.tensor
            This is a :math:`\{2 \times n_{samples}\}` dimensional tensor with
            the assumed desired speed of each agent in each scenario.
        T_out : torch.tensor
            This is a :math:`n_{samples}` dimensional tensor with
            the end time at which the gap is presumed closed in each scenario.
        dt : float
            The time step size for simulations, after which the current behavior 
            of agents is reconsidered.
        const_accs : list
            A list that shows if any of the agents has a prescribed acceleration.

        Returns
        -------
        A : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the binary decision (accept gap/ reject gap) for each set of parameters, each initial scenario, and
            each probabalistic path.
        T_A : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the predicted time of moving onto the contested space for each set of parameters, each initial scenario, and
            each probabalistic path.

        '''
        
        # Define whether an agent has to give priority to the other agent or not
        has_to_give_priority = [False, True]
        
        # Define stop criterias for the simulation
        Stop_criterias = ({'Accepted': coll_dist[1]},)
        
        # Define goals and conflict point for all the scenario
        goals = torch.tensor([[0,500], [-500, 0]], dtype=torch.float32, device=self.device)
        conflict_point = torch.tensor([0, 0], dtype=torch.float32, device=self.device)
        
        # Initialize simulation object
        sc_simulation = SCSimulation(self.device, self.N_AGENTS, names, ctrl_types, initial_positions, speeds, accs, coll_dist, 
                                     free_speeds, goals, conflict_point, has_to_give_priority, const_accs, Stop_criterias, 
                                     fixed_params = self.fixed_params, variable_params = params, num_paths=self.num_samples_path_pred,
                                     end_times=T_out, time_step=dt, zero_acc_after_exit=False)
        
        # Run simulation
        sc_simulation.run()
        
        # Extract the distance to conflicted space of the target vehicle form simulation results
        Dc_pred = (-sc_simulation.agents[0].traj_pos[:, :, :, 1] - coll_dist[0][None, :, None, None])
        Da_pred = (sc_simulation.agents[1].traj_pos[:, :, :, 0] - coll_dist[1][None, :, None, None])
        # Clear and delete simulation object
        sc_simulation.empty()
        del sc_simulation
        
        # Assume that the gap is rejected as a baseline fo reach scenario (a = 0)
        A = torch.zeros(Da_pred.shape[:-1], dtype=torch.bool, device=self.device)
        
        # Get the time value at every simulated time step
        T_pred = torch.arange(0, torch.max(T_out) + 0.5 * dt, dt, dtype=torch.float32, device=self.device)
        
        # Allocate empty tensor to save the time of the ego vehicle entering the contested space
        T_C = self._get_zero_point(Dc_pred, T_pred)
        T_A = self._get_zero_point(Da_pred, T_pred)
        
        A = T_A < T_C
        
        return A, T_A, T_C


    def predict(self, A, T_A, T_C):
        r'''
        This forms the prediction agreeable to the framework for benchmarking
        gap acceptance models.

        Parameters
        ----------
        A : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the binary decision (accept gap/ reject gap) for each set of parameters, each initial scenario, and
            each probabalistic path.
        T_A : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the predicted time of moving onto the contested space for each set of parameters, each initial scenario, and
            each probabalistic path.
        T_C : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the predicted time of the ego vehicle moving onto the contested space for each set of parameters, each initial scenario, and
            each probabalistic path.

        Returns
        -------
        A_pred : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples}\}` dimensional tensor with
            the likelihodd of accepting the gap for each set of parameters and each initial scenario
        T_A_pred : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times 9\}` dimensional tensor with
            the probabilitic time of accepting the gap (using decile values) for each set of parameters and 
            each initial scenario.
        T_C_pred : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times 9\}` dimensional tensor with
            the probabilitic time of closing the gap (using decile values) for each set of parameters and 
            each initial scenario.

        '''
        
        # Get mean probability of accepting gap over all probabalistic paths
        A_pred = torch.mean(A.to(dtype = torch.float32), dim = -1)
        
        # Get qunatile over the time of accepting the gap based on all probabalistic paths
        # Rejected gaps are ignored at this point.
        T_A_help = torch.full(T_A.shape, torch.nan, dtype = torch.float32, device = self.device)
        T_A_help[A] = T_A[A]
        T_A_pred = torch.nanquantile(T_A_help, self.p_quantile, dim = -1).permute(1,2,0)
        
        T_C_help = torch.full(T_C.shape, torch.nan, dtype = torch.float32, device = self.device)
        T_C_help[~A] = T_C[~A]
        T_C_pred = torch.nanquantile(T_C_help, self.p_quantile, dim = -1).permute(1,2,0)
        return A_pred, T_A_pred, T_C_pred
            
        
    def loss(self, A_pred, T_A_pred, T_C_pred, A_true, t_E_true, loss_type = 'Time_MSE'):
        r'''
        This calculates the training loss of each set of parameters

        Parameters
        ----------
        A_pred : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the binary decision (accept gap/ reject gap) for each set of parameters, each initial scenario, and
            each probabalistic path.
        T_A_pred : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the predicted time of moving onto the contested space for each set of parameters, each initial scenario, and
            each probabalistic path.
        T_C_pred : torch.tensor
            This is a :math:`\{n_{params} \times n_{samples} \times n_{preds}\}` dimensional tensor with
            the predicted time of the ego vehicle moving onto the contested space for each set of parameters, each initial scenario, and
            each probabalistic path.
        A_true : torch.tensor
            The true acceptance behavior for each initial scenario.
        t_E_true : torch.tensor
            The true time of accepting a gap (if accepted) for each initial scenario.
        loss_type: string
            The type of loss to be calculated. The default is 'Time_MSE'.

        Returns
        -------
        Loss : torch.tensor
            This is a :math:`\{n_{params} \times n_{preds}\}` dimensional tensor with
            the training loss assigned to each set of parameters and each probabalistic path.

        '''
        # Check the loss type
        if loss_type == 'Time_MSE':
            # Allocate empty array for the time loss of each param-set, scenario and probablistic path
            Loss_T_A = torch.zeros(A_pred.shape, dtype = torch.float32, device = self.device)
            Loss_T_C = torch.zeros(A_pred.shape, dtype = torch.float32, device = self.device)
            
            Delta_T_A = t_E_true[None, :, None] - T_A_pred
            Delta_T_C = t_E_true[None, :, None] - T_C_pred
            
            # For accepted gaps, loss is simply squared error to true time
            Loss_T_A[:, A_true, :] = Delta_T_A[:, A_true] ** 2
            # For rejected gaps, loss is squared error (assuming that T_A_pred < T_out)
            Loss_T_A[:, ~A_true, :] = (torch.maximum(torch.tensor(0.0), Delta_T_A[:, ~A_true])) ** 2
            
            # For accepted gaps, loss is squared error (assuming that T_A_pred < T_out)
            Loss_T_C[:, A_true, :] = (torch.maximum(torch.tensor(0.0), Delta_T_C[:, A_true])) ** 2
            # For rejected gaps, loss is simply squared error to true time
            Loss_T_C[:, ~A_true, :] = Delta_T_C[:, ~A_true] ** 2
            
            
            # The binary loss is the multiplied binary cross entropy (4 for false prediction, 0 for correct one)
            A_diff = A_pred.to(torch.float32) - A_true[None, :, None].to(torch.float32)
            Loss_A = 6 * torch.abs(A_diff)
            
            # Get the combined Loss
            Loss = Loss_A + Loss_T_A + Loss_T_C
            
            # Sum up the loss over all scenarios, so the final output is only dependent on parameters and probabalistic paths
            Loss = Loss.sum(1)
            
        elif loss_type == 'ADE':
            # Allocate empty array for the time loss of each param-set, scenario and probablistic path
            Loss_T_A = torch.zeros(A_pred.shape, dtype = torch.float32, device = self.device)
            Loss_T_C = torch.zeros(A_pred.shape, dtype = torch.float32, device = self.device)
            
            Delta_T_A = t_E_true[None, :, None] - T_A_pred
            Delta_T_C = t_E_true[None, :, None] - T_C_pred
            
            # For accepted gaps, loss is simply squared error to true time
            Loss_T_A[:, A_true, :] = Delta_T_A[:, A_true] ** 2
            # For rejected gaps, loss is squared error (assuming that T_A_pred < T_out)
            Loss_T_A[:, ~A_true, :] = (torch.maximum(torch.tensor(0.0), Delta_T_A[:, ~A_true])) ** 2
            
            # For accepted gaps, loss is squared error (assuming that T_A_pred < T_out)
            Loss_T_C[:, A_true, :] = (torch.maximum(torch.tensor(0.0), Delta_T_C[:, A_true])) ** 2
            # For rejected gaps, loss is simply squared error to true time
            Loss_T_C[:, ~A_true, :] = Delta_T_C[:, ~A_true] ** 2
            
            
            # The binary loss is the multiplied binary cross entropy (4 for false prediction, 0 for correct one)
            A_diff = A_pred.to(torch.float32) - A_true[None, :, None].to(torch.float32)
            Loss_A = 6 * torch.abs(A_diff)
            
            # Punish identical predictions (identical for each prediction)
            T_A_pred_std = T_A_pred.std(dim = -1, keepdims = True) / t_E_true[None, :, None]
            
            std_0 = torch.tensor(0.1)
            
            T_A_pred_std = torch.minimum(T_A_pred_std, std_0)
            c_a = 1 / (std_0 ** 2)
            c_b = - 2 / std_0
            c_c = 1
            
            # Loss
            Loss_D = c_a * T_A_pred_std ** 2 + c_b * T_A_pred_std + c_c
            
            # Get the combined Loss
            Loss = Loss_A + Loss_T_A + Loss_T_C + 1.5 * Loss_D
            
            # Sum up the loss over all scenarios, so the final output is only dependent on parameters and probabalistic paths
            Loss = Loss.sum(1)
        else:
            raise AttributeError("This loss type is not implemented")
        
        return Loss
        
#%%
class commotions_template():
    '''
    This class is a template for different implementations of the commotions model and their interaction
    with the framework for benchmarking gap accceptance models.
    
    It provides four basic types of functions:
        - Analysing the search space of the variable parameters
        - Optimization methods for the commations mdoel to find the best values for the variable parameters
        - A way of analyzing the given initial scenarios and preparing simulations.
        - Methods for setting up training and evaluating of a model
        
    '''
    
    # Analyzing the search space of the variable parameters
    def Param_boundaries(self):
        '''
        This function extracts and analyzes the given ranges for the variable parameters of the commotions model

        Returns
        -------
        None.

        '''
        # Combine variable parameters used in the model in a list
        Params = [self.Beta_V, self.DDeltaV_th_rel, self.TT, self.TT_delta, self.Tau_theta,
                  self.Sigma_xdot, self.DDeltaT, self.TT_s, self.DD_s, self.VV_0_rel,
                  self.K_da, self.Kalman_multi_pos, self.Kalman_multi_speed, 
                  self.Free_speed_multi_ego, self.Free_speed_multi_tar]
        
        # Store the number of variable parameters in the model
        self.num_variable_params = len(Params)
        
        # Store the number of elements in each parameter as a tensor
        self.Variable_params_num = [len(param) for param in Params]
        self.Variable_params_num = torch.tensor(self.Variable_params_num, dtype=torch.int64, device=self.device)
        
        # Determine if each parameter should be represented in log space or not
        self.Variable_params_log = [bool((param[1:] / param[:-1]).std() < (param[1:] / param[:-1]).mean() / 1e4) 
                                    for param in Params]
        self.Variable_params_log = torch.tensor(self.Variable_params_log, dtype=torch.bool, device=self.device)
        
        # Store the minimum value of each parameter as a tensor
        self.Variable_params_min = [param.min() for param in Params]
        self.Variable_params_min = torch.tensor(self.Variable_params_min, dtype=torch.float32, device=self.device)
        
        # If a parameter is represented in log space, convert its minimum value to log space
        self.Variable_params_min[self.Variable_params_log] = torch.log10(self.Variable_params_min[self.Variable_params_log])
        self.Variable_params_min = self.Variable_params_min[None, :]
        
        # Store the maximum value of each parameter as a tensor
        self.Variable_params_max = [param.max() for param in Params]
        self.Variable_params_max = torch.tensor(self.Variable_params_max, dtype=torch.float32, device=self.device)
        
        # If a parameter is represented in log space, convert its maximum value to log space
        self.Variable_params_max[self.Variable_params_log] = torch.log10(self.Variable_params_max[self.Variable_params_log])
        self.Variable_params_max = self.Variable_params_max[None, :]
        
        # Compute the range of each variable parameter as a tensor
        self.Variable_params_int = self.Variable_params_max - self.Variable_params_min

        
    def Overwrite_boundaries(self, param):
        '''
        This functions allows one to focus the search space of variable params
        arround one set of given values for these parameters

        Parameters
        ----------
        param : numpy.array
            For each varaible param, one value is given to focus the search arround.

        Returns
        -------
        None.

        '''
        # Get old parameter values
        Param_min_old = self.Variable_params_min[0].cpu().detach().numpy()
        Param_max_old = self.Variable_params_max[0].cpu().detach().numpy()
        Param_int_old = self.Variable_params_int[0].cpu().detach().numpy()
        Param_log_old = self.Variable_params_log.cpu().detach().numpy()
        
        # Calculate small interval for parameter search
        Int_small = Param_int_old / (2 * (self.Variable_params_num.cpu().detach().numpy() - 1))   
        
        # Get current parameter values
        Params = [self.Beta_V, self.DDeltaV_th_rel, self.TT, self.TT_delta, self.Tau_theta,
                  self.Sigma_xdot, self.DDeltaT, self.TT_s, self.DD_s, self.VV_0_rel,
                  self.K_da, self.Kalman_multi_pos, self.Kalman_multi_speed, 
                  self.Free_speed_multi_ego, self.Free_speed_multi_tar]
        
        
        # Log-transform parameters as necessary
        param[Param_log_old] = np.log10(param[Param_log_old])
        for i in range(len(Params)):
            if Param_log_old[i]:
                Params[i][:] = np.log10(Params[i][:])
        
        # Update parameter bounds based on new values
        Param_min_new = np.maximum(Param_min_old, param - Int_small)
        Param_max_new = np.minimum(Param_max_old, param + Int_small)
        Param_int_new = Param_max_new - Param_min_new
        
        # Calculate scaling factor and offset for each parameter
        Factor = Param_int_new / Param_int_old
        Offset = Param_min_new - Factor * Param_min_old
        
        # Apply scaling and offset to parameters
        for i in range(len(Params)):
            # Use modifications to avoid overwriting references
            Params[i] *= Factor[i]
            Params[i] += Offset[i]
            
        # Reverse the Log-transform of parameters as necessary
        for i in range(len(Params)):
            if Param_log_old[i]:
                Params[i][:] = 10 ** Params[i][:]
              
    
    def Param_decoder(self, Params):
        '''
        This function allows the transformation of values inside a unit hypercube 
        to the actual search space of variable parameters
    
        Parameters
        ----------
        Params : torch.tensor
            The different cases of parameters for each set of parameters, given in the interval [0,1].
    
        Returns
        -------
        Params_dec : torch.tensor
            The decoded parameter values for each set of parameters.
    
        '''
        
        # Check if Params is a 2D tensor (i.e., batch mode)
        if Params.dim() == 2:
            # Compute the decoded parameter values
            Params_dec = self.Variable_params_min + self.Variable_params_int * Params
            
            # Reverse the log transformation of the appropriate parameters
            Params_dec[:, self.Variable_params_log] = 10 ** Params_dec[:, self.Variable_params_log]
        else: 
            # If Params is not 2D, we need to handle the extra dimensions.
            # Create copies of the Variable_params_min and Variable_params_int tensors
            V_min = self.Variable_params_min.clone()
            V_int = self.Variable_params_int.clone()
            
            # Add extra dimensions to V_min and V_int to match the dimensions of Params
            for i in range(Params.dim() - 2):
                V_min = V_min[None, :]
                V_int = V_int[None, :]
            
            # Compute the decoded parameter values using the expanded V_min and V_int tensors
            Params_dec = V_min  + V_int * Params
            
            # Reorder dimensions so that the dimension including the different parameters becomes the first dimension
            Params_dec = Params_dec.permute(tuple(np.roll(np.arange(Params_dec.dim()), 1)))
            
            # Reverse the log transformation of the appropriate parameters
            Params_dec[self.Variable_params_log] = 10 ** Params_dec[self.Variable_params_log]
            
            # Reorder dimensions back to their original order
            Params_dec = Params_dec.permute(tuple(np.roll(np.arange(Params_dec.dim()), -1)))
        
        return Params_dec
    
    
    #%% Optimization methods for the commotions model to find the best values for the variable parameters
    def grid_search(self):
        '''
        Find the best value for the given parameters by using grid search.

        Returns
        -------
        numpy.array
            The parameters explored during this optimization method.
        numpy.array
            The corresponding loss value.

        '''
        # Decide on resolution, and adjust the number of probabalistic paths (not all are needed during training)
        self.commotions_model.num_samples_path_pred = int(self.num_samples_path_pred / 20)
        
        # Preprocesspath data of initial scenarios
        commotion_args = self.extract_train_data()
        
        # Get parameter boundaries
        self.Param_boundaries()
        
        # Generate the variable params in the unit hypercube
        Inputs = [torch.linspace(0, 1, self.Variable_params_num[i], dtype = torch.float32, device = self.device)
                  for i in range(self.num_variable_params)]
        Params = torch.meshgrid(Inputs, indexing = 'ij')
        Params = torch.stack(Params, dim = self.num_variable_params).reshape(-1, self.num_variable_params)
        
        # Define method name
        method = 'Grid search'
        
        # Decode parameters to parameter seach space
        Params_dec = self.Param_decoder(Params)
        
        # Evaluate the different sets of parameters
        Loss = self.evaluate_samples_params(Params_dec, method, *commotion_args)
        
        # Print final loss value
        print(method + ' - min(Loss): {:7.2f}'.format(Loss.min()), flush=True)       
        
        return Params_dec.cpu().detach().numpy(), Loss.cpu().detach().numpy()
    
    
    def DE(self, n_params_per_generation = 20, generations = 250):
        '''
        Find the best value for the given parameters by using differential evolution.
        
        Parameters
        ----------
        n_params_per_generation : int, optional
            The numbers of parameter sets explored at each generation. The default is 20.
        generations : int, optional
            The number of generations used. The default is 250.

        Returns
        -------
        numpy.array
            The parameters explored during this optimization method.
        numpy.array
            The corresponding loss value.

        '''
        # Decide on resolution, and adjust the number of probabalistic paths (not all are needed during training)
        self.commotions_model.num_samples_path_pred = int(self.num_samples_path_pred / 20)
        
        # Preprocesspath data of initial scenarios
        commotion_args = self.extract_train_data()
        
        # Get parameter boundaries
        self.Param_boundaries()
        
        # Prepare empty tensors for all explored sets of parameters
        Loss   = torch.zeros((generations + 1, n_params_per_generation), 
                             dtype = torch.float32, device = self.device)
        Params = torch.zeros((generations + 1, n_params_per_generation, self.num_variable_params), 
                             dtype = torch.float32, device = self.device)
        
        # Start the first generation Differentioal Evolution
        g = 0
        print('Differential Evolution', flush=True)
        TT = time.time() 
        
        # Randomly initialize the first sets of parameters
        Params[g, :, :] = torch.rand((n_params_per_generation, Params.shape[-1]), 
                                     dtype = torch.float32, device = self.device)
        
        # Decode the intial values
        Params_g = self.Param_decoder(Params[g, :, :])
        
        # Define method name
        method = 'Differential Evolution - Generation ' + str(g).rjust(len(str(generations))) + '/{}'.format(generations)
        
        # Calculate the corresponding loss
        Loss[g,:,:] = self.evaluate_samples_params(Params_g, method, *commotion_args)
        
        # print the current best loss value
        print(method + ' - min(Loss): {:7.2f}'.format(Loss[g].min()), flush=True)        
        print('', flush=True)
        
        # Go through generation after generation
        for g in range(1, generations + 1): 
            # Get set of available individuals for reproduction
            samples = torch.ones((n_params_per_generation, n_params_per_generation), 
                                 dtype = torch.float32, device = self.device)
            samples[torch.eye(n_params_per_generation, dtype = torch.bool, device = self.device)] = 0.0
            
            # Sample three non repeteive individual from the rpevious generation for each case 
            test_case_rank = torch.multinomial(samples, 3, replacement = False)
            Params_a = Params[g-1, test_case_rank[:,0],:]
            Params_b = Params[g-1, test_case_rank[:,1],:]
            Params_c = Params[g-1, test_case_rank[:,2],:]
                
            # Employ dithering for better convergence (i.e., F in [0.5, 1])
            F = torch.rand((1,1), dtype = torch.float32, device = self.device) * 0.5 + 0.5
            
            # Perform reproduction and create children
            Params_com = Params_a + F * (Params_b - Params_c)
            
            # Determine the elements on which crossover betwenn parent and child will be performed
            chi_0 = 0.6
            Params_crossover = torch.rand((n_params_per_generation, Params.shape[-1]),
                                          dtype = torch.float32, device = self.device) > chi_0
            # apply the crossover between parent and child
            Params_com[Params_crossover] = Params[g-1, Params_crossover]
            
            # Enforce the boundaries of the unit hypercube search space
            torch.clip(Params_com, min = 0.0, max = 1.0, out = Params_com)
            
            # Decode the children
            Params_com_g = self.Param_decoder(Params_com)
            
            # Evaluate the loss of the children
            method = 'Differential Evolution - Generation ' + str(g).rjust(len(str(generations))) + '/{}'.format(generations)
            Loss_com_g = self.evaluate_samples_params(Params_com_g, method, *commotion_args)
            
            # find the children that provide improved loss values compared to parents
            Replace = Loss_com_g <= Loss[g-1,:]
            
            # keep best samples and replace unfit parents
            Loss[g, ~Replace]     = Loss[g-1, ~Replace]
            Params[g, ~Replace,:] = Params[g-1, ~Replace,:]
            
            Loss[g, Replace]      = Loss_com_g[Replace]
            Params[g, Replace,:]  = Params_com[Replace,:]
            
            # print the current best loss value
            print(method + ' - min(Loss): {:7.2f}'.format(Loss[g].min()), flush=True)        
            print('', flush=True)
        
        # Decode sets of parameters before exiting 
        Params_out = self.Param_decoder(Params)
        # Print final loss value and running time 
        TT = time.time() - TT
        print('Differential Evolution: {} min {:0.2f} s'.format(int(np.floor(TT / 60)), np.mod(TT, 60)), flush=True)        
        print('', flush=True)
        
        
        return Params_out.cpu().detach().numpy(), Loss.cpu().detach().numpy()
    
    
    def BO_EI(self, iterations = 200):
        '''
        Find the best value for the given parameters by using bayesian optimization,
        using the expected improvement as the acquisition function.
        
        Parameters
        ----------
        iterations : int, optional
            The numbers of iterations performed. The default is 200.

        Returns
        -------
        numpy.array
            The parameters explored during this optimization method.
        numpy.array
            The corresponding loss value.

        '''
        
        method_all = 'Bayesian Optimization (EI)'
        
        # Confirm that method indeed was completed (Error in model fitting are possible)
        completed = False
        seed = 0
        while not completed:
            torch.manual_seed(seed)
            # Choose expected improvement as acquisition function
            results = self.BO(method_all, iterations, 1, Models.GPEI)
            completed = True # TODO: Reset
            # try:
            #     # Set initial seed for the generation of the inital BO samples
            #     torch.manual_seed(seed)
            #     # Choose expected improvement as acquisition function
            #     results = self.BO(method_all, iterations, 1, Models.GPEI)
            #     completed = True
            # except:
            #     print('')
            #     print(method_all + ': Failed due to numerical issues!')
            #     print('')
            #     completed = False
            #     seed += 1
        return results
    
    
    def BO_KG(self, iterations = 50, parallel_samples = 4):
        '''
        Find the best value for the given parameters by using bayesian optimization,
        using the knowledge gradient as the acquisition function.
        
        Parameters
        ----------
        iterations : int, optional
            The numbers of iterations performed. The default is 40.
        parallel_samples : int, optional
            The number new samples generated and added to the surrogate model in each iteration. The default is 5.

        Returns
        -------
        numpy.array
            The parameters explored during this optimization method.
        numpy.array
            The corresponding loss value.

        '''
        
        method_all = 'Bayesian Optimization (KG)'
        
        # Confirm that method indeed was completed (Error in model fitting are possible)
        completed = False
        seed = 0
        while not completed:
            try:
                # Set initial seed for the generation of the inital BO samples
                torch.manual_seed(seed)
                # Choose knowledge gradient as acquisition function
                results = self.BO(method_all, iterations, parallel_samples, Models.GPKG)
                completed = True
            except:
                print('')
                print(method_all + ': Failed due to numerical issues!')
                print('')
                completed = False
                seed += 1
        return results
    
    
    def BO(self, method_all, iterations = 100, parallel_samples = 1, optimizer_model = Models.GPEI):
        '''
        Find the best value for the given parameters by using a general bayesian optimization method.
        This is implement using the ax module.

        Parameters
        ----------
        method_all : string
            The name of the bayesian optimization algorithm used.
        iterations : int, optional
            The numbers of iterations performed. The default is 100.
        parallel_samples : int, optional
            The number new samples generated and added to the surrogate model in each iteration. The default is 1.
        optimizer_model : ax.modelbridge.registry.model, optional
            The acquisition function used to generat new samples. The default is Models.GPEI.

        Returns
        -------
        numpy.array
            The parameters explored during this optimization method.
        numpy.array
            The corresponding loss value.

        '''
        
        # print the method name
        print(method_all, flush=True)
        TT = time.time() 
        
        # Decide on resolution, and adjust the number of probabalistic paths (not all are needed during training)
        self.commotions_model.num_samples_path_pred = int(self.num_samples_path_pred / 10)
        
        # Preprocesspath data of initial scenarios
        commotion_args = self.extract_train_data()
        
        # Get parameter boundaries
        self.Param_boundaries()
        
        # Initialize the BO experiment
        BO_steps = []
        # Get the generator model of the initiasl random points 
        # Note: Sobol series better than latin hyperqube
        BO_steps.append(GenerationStep(model = Models.SOBOL, num_trials = 15 * self.num_variable_params,
                                       max_parallelism = 15 * self.num_variable_params))
        # Get the generator model used in later iterations
        BO_steps.append(GenerationStep(model = optimizer_model, num_trials = -1,
                                       max_parallelism = parallel_samples,
                                       model_kwargs = {'torch_device': self.device}))
        
        # Implement the overall generation strategy
        ax_client = AxClient(generation_strategy = GenerationStrategy(steps = BO_steps), verbose_logging = False)
        
        # Set the continuous optimization parameters and their ranges
        Parameters = [{'name': 'x{}'.format(i + 1), 'type': 'range',
                       'bounds': [0.0, 1.0], 'value_type': 'float'}
                      for i in range(self.num_variable_params)]
        
        # Setup the plan for the whole optimization process
        ax_client.create_experiment(name="commotion BO", parameters = Parameters,
                                    objective_name = "commotion loss", minimize = True)
        
        # Setup memory location for final sets of parameters and corresponding loss values
        Params_dec = torch.zeros((0, self.num_variable_params), dtype = torch.float32, device = self.device)
        Loss       = torch.zeros((0,), dtype = torch.float32, device = self.device)
        
        Step_loss = []
        
        # Go through all the iterations of BO
        for iteration in range(iterations + 1):
            # Check if initial samples need to be created
            if iteration == 0:
                method = (method_all + ' - Initialization')
            else:
                method = (method_all + ' - Iteration ' + 
                          str(iteration).rjust(len(str(iterations))) + 
                          '/{}'.format(iterations))
            
            # Determine how many samples need to be generated per iteration
            parallel_steps, _ = ax_client.get_current_trial_generation_limit()
            
            # Set up empty list to hold these parameters
            new_parameters = []
            trial_indices = []
            
            T = time.time()
            steps_digits = len(str(parallel_steps))
            
            # Generate the initial sets of parameters
            for i in range(parallel_steps):
                t = time.time()
                torch.cuda.empty_cache()
                params, trial_index = ax_client.get_next_trial()
                torch.cuda.empty_cache()
                new_parameters.append(params)
                trial_indices.append(trial_index)
                t = time.time() - t
                if iteration > 0 and parallel_steps > 1:
                    print(method + ' - Generate candidate ' + str(i + 1).rjust(steps_digits) + 
                          '/{}: {:0.3f} s'.format(parallel_steps, t), flush=True)
                    
            T = time.time() - T
            print(method + ' - Generate candidates: {:0.3f} s'.format(T), flush=True)
            
            # Transform generated parameters into torch tensor 
            Params = torch.from_numpy(np.array([[new_parameters[i_sample].get("x{}".format(i_param + 1)) 
                                                 for i_param in range(self.num_variable_params)] 
                                                for i_sample in range(len(new_parameters))])
                                      ).to(dtype = torch.float32, device = self.device)
            
            # Set name of current step of method
            method_eval = method + ' - Evaluate candidates'
            
            # Decode the generated sets of parameters
            Params_dec_new = self.Param_decoder(Params)
            
            # Evaluate the loss of the generated children and the estimation od the standard variation of this loss
            Loss_new, Loss_std = self.evaluate_samples_params(Params_dec_new, method_eval, *commotion_args, get_std = True)
            
            # Transform the loss data into numpy arrays
            Loss_new_np = Loss_new.cpu().detach().numpy()
            Loss_std_np = Loss_std.cpu().detach().numpy()
            
            # Append the new loss data of generated sets of parameters to previous samples
            Params_dec = torch.concat((Params_dec, Params_dec_new), dim = 0)
            Loss       = torch.concat((Loss, Loss_new), dim = 0)
            
            # Update the generation model, by feeding it the evaluation results and refitting the surrogate model.
            T = time.time()
            for i in range(parallel_steps):
                ax_client.complete_trial(trial_index = trial_indices[i], raw_data = (Loss_new_np[i], Loss_std_np[i]))
            ax_client.fit_model()
            T = time.time() - T
            print(method + ' - Update model: {:0.3f} s'.format(T), flush=True)
            
            print(method + ' - min(Loss): {:7.2f}'.format(Loss.min()), flush=True)        
            print('', flush=True)
            
            Step_loss.append(Loss.min())
        
        self.train_loss = np.array(Step_loss)[np.newaxis]
        
        # Print final loss value and running time    
        TT = time.time() - TT
        print(method_all + ': {} min {:0.2f} s'.format(int(np.floor(TT / 60)), np.mod(TT, 60)), flush=True)        
        print('', flush=True)
        
        return Params_dec.cpu().detach().numpy(), Loss.cpu().detach().numpy() 
    

    #%% A way of runnging simulations.
    def evaluate_samples_params(self, Params, method, names, ctrl_types, 
                                initial_positions, speeds, accs, coll_dist, free_speeds, 
                                T_out, dt, A_true, t_E_true, get_std = False):
        '''
        This allows the evalution of the loss of the model given certain initial scenarios and 
        certain sets of variable parameters.

        Parameters
        ----------
        Params : torch.tensor
            the sets of parameters for which teh model is to be evaluated.
        method : string
            The name of the training method in which this evaluation is performed.
        names : list
            A list of string containing the names of the agents involved in the scenario.
        ctrl_types : list
            A list of comtrol types the agents in the scenario follow 
            (speed control => pedestrian, acceleration control => vehicle).
        initial_positions : torch.tensor
            The different initial position of each agent in each scenario.
        speeds : torch.tensor
            The different initial speed of each agent in each scenario.
        accs : torch.tensor
            The different initial acceleration of each agent in each scenario.
        coll_dist : torch.tensor
            The distance to the conflicted space of each agent in each scenario.
        free_speeds : torch.tensor
            The assumed desired speed of each agent in each scenario.
        T_out : torch.tensor
            The end time at which the gap is presumed closed in each scenario.
        dt : float
            The time steps for simulations, after which the current behavior of agents is reconsidered.
        A_true : torch.tensor
            The true acceptance behavior for each initial scenario.
        t_E_true : torch.tensor
            The true time of accepting a gap (if accepted) for each initial scenario.
        get_std : bool, optional
            If true, also returns Loss_std. The default is False.

        Returns
        -------
        Loss_mean : torch.tensor
            The expected loss for each set of parameters.
        Loss_std : torch.tensor, optional
            The standard deviation of the expected loss for each set of parameters. 

        '''
        
        # Assert the sets of parameters are aligned along a single dimension
        assert Params.dim() == 2
        # Get the numebr of sets of parameters to be evaluated.
        num_params = len(Params)
        
        # Allocate an empty error for the loss values
        Loss = torch.zeros((num_params, self.commotions_model.num_samples_path_pred), 
                           dtype = torch.float32, device = self.device)
        
        # Determine how many samples can be performed in parallel
        runs_per_batch = int(np.floor(self.calc_max / self.commotions_model.num_samples_path_pred))
        
        
        
        Agent_type_combinations = np.unique(ctrl_types, axis = 1).T
        
        T = time.time() 
        print(method, flush = True)
        batch = 0
        for agent_type_combination in Agent_type_combinations:
            combination_samples = np.where((ctrl_types == agent_type_combination[:,np.newaxis]).all(0))[0]
            
            combination_num_samples = len(combination_samples)
            
            # Based on this, determine the number of batches per set of parameters and number of batches of sets of parameters
            # Also determine, how many sets of parameters and how many initial scenarios are evaluated simultainiously.
            if runs_per_batch > combination_num_samples:
                n_samples_per_batch = combination_num_samples
                n_sample_batches = 1
                n_params_per_batch = int(np.floor(runs_per_batch / combination_num_samples))
                n_param_batches  = int(np.ceil(num_params / n_params_per_batch))
            else:
                n_samples_per_batch = runs_per_batch
                n_sample_batches = int(np.ceil(combination_num_samples / runs_per_batch))
                n_params_per_batch = 1
                n_param_batches  = num_params
            
            # Go through the sets of parameters
            for param_batch in range(n_param_batches):
                param_batch_index = np.arange(n_params_per_batch * param_batch, 
                                              n_params_per_batch * (param_batch + 1))
                param_batch_index = param_batch_index[param_batch_index < num_params]
                
                # Go through the initial scearios for each set of parameters
                for sample_batch in range(n_sample_batches):
                    sample_batch_index_inter = np.arange(n_samples_per_batch * sample_batch, 
                                                         n_samples_per_batch * (sample_batch + 1))
                    sample_batch_index_inter = sample_batch_index_inter[sample_batch_index_inter < combination_num_samples]
                    sample_batch_index = combination_samples[sample_batch_index_inter]
                    
                    t = time.time()
                    
                    # Get the binary and time prediction for each set of parameters and each initial 
                    # scenario in this batch, with one value for each probebalistic path
                    A, T_A, T_C = self.commotions_model(names             = names,
                                                        ctrl_types        = ctrl_types[:, sample_batch_index],
                                                        params            = Params[param_batch_index, :],
                                                        initial_positions = initial_positions[:, sample_batch_index], 
                                                        speeds            = speeds[:, sample_batch_index], 
                                                        accs              = accs[:, sample_batch_index], 
                                                        coll_dist         = coll_dist[:, sample_batch_index], 
                                                        free_speeds       = free_speeds[:, sample_batch_index],
                                                        T_out             = T_out[sample_batch_index],
                                                        dt                = dt,
                                                        const_accs        = self.const_accs)
                    
                    # Go over the initial scenarios and calculate the loss for each set of parameters and
                    # each probebalistic path
                    loss = self.commotions_model.loss(A_pred    = A, 
                                                      T_A_pred  = T_A, 
                                                      T_C_pred  = T_C,
                                                      A_true    = A_true[sample_batch_index], 
                                                      t_E_true  = t_E_true[sample_batch_index], 
                                                      loss_type = self.train_loss_type)
                    
                    # Add to overall loss (if not all intial sampels coud be covered in one batch)
                    Loss[param_batch_index, :] += loss
                    
                    t = time.time() - t
                    
                    batch += 1
                    print(method + ' - Batch {}: {:0.3f} s'.format(batch, t), flush=True)
                
        T = time.time() - T
        print(method + ': {} min {:0.3f} s'.format(int(np.floor(T / 60)), np.mod(T, 60)), flush=True)
        
        # calculate the mean loss over all probebalistic paths
        Loss_mean = Loss.mean(-1)
        
        # check if the standard deviation output flag is enabled
        if get_std:
            # if so, calculate the standard deviation of the loss over all probebalistic paths
            Loss_std = Loss.std(-1)
            # We do not want stadard deviation, but standard error (i.e., standard deviation of the mean),
            # so this is calculated next
            Loss_std_err = Loss_std / float(np.sqrt(self.commotions_model.num_samples_path_pred))
            return Loss_mean, Loss_std_err
        else:
            return Loss_mean
    
    
    # Methods for setting up training and evaluating of a model
    def prepare_gpu(self):
        '''
        This sets up the GPU on which the training and evaluation of the commotions model are run

        Returns
        -------
        None.

        '''
        total_memory_GB = torch.cuda.get_device_properties(0).total_memory / 2 ** 30

        self.calc_max = 10000 * total_memory_GB
    
    def extract_data(self, purpose = 'train'):
        '''
        This extract the initial scenario data from the given input data

        Parameters
        ----------
        purpose : string, optional
            This sets the mode, e.g, extracting training or testing data. The default is 'train'.

        Returns
        -------
        names : list
            A list of string containing the names of the agents involved in the scenario.
        ctrl_types : list
            A list of comtrol types the agents in the scenario follow 
            (speed control => pedestrian, acceleration control => vehicle).
        initial_positions : torch.tensor
            The different initial position of each agent in each scenario.
        speeds : torch.tensor
            The different initial speed of each agent in each scenario.
        accs : torch.tensor
            The different initial acceleration of each agent in each scenario.
        coll_dist : torch.tensor
            The distance to the conflicted space of each agent in each scenario.
        free_speeds : torch.tensor
            The assumed desired speed of each agent in each scenario.
        T_out : torch.tensor
            The end time at which the gap is presumed closed in each scenario.
        dt : float
            The time steps for simulations, after which the current behavior of agents is reconsidered.

        '''
        
        
        
        # Get the purpose specific names (train or test)
        if purpose == 'train':
            _, T, S, agent_names, D, dist_names, class_names, P, DT = self.get_classification_data(train = True)
            Output_T_pred = self.data_set.Output_T_pred[self.Index_train]
        elif purpose == 'test':
            _, T, S, agent_names, D, dist_names, class_names = self.get_classification_data(train = False)
            Output_T_pred = self.data_set.Output_T_pred
        else:
            raise KeyError('The purpose "' + purpose + '" is not an available split of the data')
        
        num_samples = len(D)
        
        # Get the timestep size (similar for all scenarios)
        dt = torch.tensor(self.dt, dtype = torch.float32, device = self.device)
        
        # Allocate empty tensors for scenario specific initial information
        T_in = torch.arange(-2, 1, dtype = torch.float32, device = self.device) * dt
        T_in = torch.tile(T_in[None], (num_samples, 1))
        
        Dc = torch.zeros((num_samples, 3), dtype = torch.float32, device = self.device)
        Da = torch.zeros((num_samples, 3), dtype = torch.float32, device = self.device)
        Le = torch.zeros(num_samples, dtype = torch.float32, device = self.device)
        Lt = torch.zeros(num_samples, dtype = torch.float32, device = self.device)
        
        T_out = torch.zeros(num_samples, dtype = torch.float32, device = self.device)
        
        
        i_accepted = np.where(np.array(dist_names) == 'accepted')[0][0]
        i_rejected = np.where(np.array(dist_names) == 'rejected')[0][0]
        
        i_Le = np.where(np.array(dist_names) == 'L_e')[0][0]
        i_Lt = np.where(np.array(dist_names) == 'L_t')[0][0]
        
        t = np.arange()
        
        t = np.arange(- self.num_timesteps_in, 1) * dt
        t_in = np.arange(-2, 1) * dt
        # Go through provided input data
        for ind in range(num_samples):
            
            # Get the gap sizes
            Le[ind] = torch.tensor(D[ind, i_Le, -1]).to(device = self.device, dtype = torch.float32)
            Lt[ind] = torch.tensor(D[ind, i_Lt, -1]).to(device = self.device, dtype = torch.float32)
            
            if self.num_timesteps_in >= 3:
                # transform from numpy to torch the initial position data
                Dc[ind] = torch.from_numpy(D[ind, i_rejected, -3:]).to(device = self.device)
                Da[ind] = torch.from_numpy(D[ind, i_accepted, -3:]).to(device = self.device)
            else:
                # Extrapolate with zero acceleration
                Dc_in = interp.interp1d(t, D[ind, i_rejected], fill_value = 'extrapolate', assume_sorted = True)(t_in)
                Da_in = interp.interp1d(t, D[ind, i_accepted], fill_value = 'extrapolate', assume_sorted = True)(t_in) 
                
                Dc[ind] = torch.from_numpy(Dc_in).to(device = self.device)
                Da[ind] = torch.from_numpy(Da_in).to(device = self.device)
            
            # Get the scenario specific end time
            T_out[ind] = Output_T_pred[ind][-1] 
        
        
        
        names = []
        for i in range(self.commotions_model.N_AGENTS):
            if i == 0:
                names.append('ego')
            else:
                names.append('tar')
               
        # Determine ctrl type of agent 
        i_ego = np.where(np.array(agent_names) == 'ego')[0][0]
        i_tar = np.where(np.array(agent_names) == 'ego')[0][0]
        
        Types = T[:, [i_ego, i_tar]].T
        Ped_agents = torch.from_numpy(Types == 'P').to(dtype = torch.bool, device = self.device)
        
        ctrl_types = np.empty((self.commotions_model.N_AGENTS, num_samples), dtype = CtrlType)
        ctrl_types[:,:] = CtrlType.SPEED
        if self.vehicle_acc_ctrl:
            ctrl_types[Types == 'V'] = CtrlType.ACCELERATION
        
        # Allocate empty tensors for further information
        lengths           = torch.from_numpy(S[...,0].T).to(device = self.device, dtype = torch.float32)
        widths            = torch.zeros((self.commotions_model.N_AGENTS, num_samples), 
                                        dtype = torch.float32, device = self.device)
        free_speeds       = torch.zeros((self.commotions_model.N_AGENTS, num_samples), 
                                        dtype = torch.float32, device = self.device)
        initial_positions = torch.zeros((self.commotions_model.N_AGENTS, num_samples, 2), 
                                        dtype = torch.float32, device = self.device)
        speeds            = torch.zeros((self.commotions_model.N_AGENTS, num_samples), 
                                        dtype = torch.float32, device = self.device)
        accs              = torch.zeros((self.commotions_model.N_AGENTS, num_samples), 
                                        dtype = torch.float32, device = self.device)
        coll_dist         = torch.zeros((self.commotions_model.N_AGENTS, num_samples), 
                                        dtype = torch.float32, device = self.device)
        
        # Get the free speed of the agents
        free_speeds[Ped_agents]  = self.fixed_params.FREE_SPEED_PED
        free_speeds[~Ped_agents] = self.fixed_params.FREE_SPEED_VEH
        
        # Get the width of th agents, based on set gap sizes
        widths[0] = Lt
        widths[1] = Le
       
        # Get the size of contested space (vehicle position assumes censter of mass)
        coll_dist[0] = 0.5 * (widths[1] + lengths[0])
        coll_dist[1] = 0.5 * (widths[0] + lengths[1])
        
        # Get the initial position of center of mass of agents
        initial_positions[0,:,1] = -coll_dist[0] - Dc[:,-1]
        initial_positions[1,:,0] = coll_dist[1] + Da[:,-1]
        
        # Calculate the inital velocities of agents based on linear interpolation
        speeds[0] = (Dc[:,-2] - Dc[:,-1]) / dt
        speeds[1] = (Da[:,-2] - Da[:,-1]) / dt
        
        # Calculate the inital velocities of agents based on linear interpolation
        accs[0] = (2 * Dc[:,-2] - (Dc[:,-1] + Dc[:,-3])) / dt ** 2
        accs[1] = (2 * Da[:,-2] - (Da[:,-1] + Da[:,-3])) / dt ** 2
        
        # increase free speed if free speed is higher than assumed free speed
        # and this is so desired
        if self.adjust_free_speeds:
            free_speeds = torch.maximum(free_speeds, speeds)
        
        return names, ctrl_types, initial_positions, speeds, accs, coll_dist, free_speeds, T_out, dt
    
    
    def extract_train_data(self):
        '''
        This extracts all the information necessary to calculate the model loss

        Returns
        -------
        names : list
            A list of string containing the names of the agents involved in the scenario.
        ctrl_types : list
            A list of comtrol types the agents in the scenario follow 
            (speed control => pedestrian, acceleration control => vehicle).
        initial_positions : torch.tensor
            The different initial position of each agent in each scenario.
        speeds : torch.tensor
            The different initial speed of each agent in each scenario.
        accs : torch.tensor
            The different initial acceleration of each agent in each scenario.
        coll_dist : torch.tensor
            The distance to the conflicted space of each agent in each scenario.
        free_speeds : torch.tensor
            The assumed desired speed of each agent in each scenario.
        T_out : torch.tensor
            The end time at which the gap is presumed closed in each scenario.
        dt : float
            The time steps for simulations, after which the current behavior of agents is reconsidered.
        A_true : torch.tensor
            The true acceptance behavior for each initial scenario.
        t_E_true : torch.tensor
            The true time of accepting a gap (if accepted) for each initial scenario or rejecting.

        '''
        # Extract scenario specifc data from input training data
        names, ctrl_types, initial_positions, speeds, accs, coll_dist, free_speeds, T_out, dt = self.extract_data('train')
        
        # Allocate tensors for desired output data
        _, _, _, _, _, _, class_names, P, DT = self.get_classification_data(train = True)
        
        i_accepted = np.where(np.array(class_names) == 'accepted')[0][0]
        
        A_true = torch.from_numpy(P[:,i_accepted].astype(bool)).to(device = self.device)
        t_E_true = torch.from_numpy(DT).to(device = self.device)
        
        return [names, ctrl_types, initial_positions, speeds, accs, 
                coll_dist, free_speeds, T_out, dt, A_true, t_E_true]
        
        
    def extract_predictions(self):
        '''
        This uses the trained model to make prediction about the testing scenarios

        Returns
        -------
        Output_A_pred : numpy.array
            The probability for each scenario of accepting the gap.
        Output_T_E_pred : numpy.array
            The proabability distribution of the time of accepting the gap, under the assumption 
            that the gap will be accepted.

        '''
        # Extract scenario specifc data from input test data
        names, ctrl_types, initial_positions, speeds, coll_dist, accs, free_speeds, T_out, dt = self.extract_data('test')
        
        # Get the variable parameters of the trained model
        variable_params = torch.from_numpy(self.param_best[np.newaxis, :]).to(dtype = torch.float32, device = self.device)
        
        # Set the number of probebalistic paths that are to be predicted
        self.commotions_model.num_samples_path_pred = self.num_samples_path_pred
        
        # Determine how many samples can be performed in parallel (i.e., batch size)
        batch_size = int(np.floor(self.calc_max / self.commotions_model.num_samples_path_pred))
        # Determine the number of batches needed for all testing scenarios
        
        # Allocate memory for the prediction
        A_pred   = torch.zeros((1, self.num_samples_test), dtype = torch.float32, device = self.device)
        T_A_pred = torch.zeros((1, self.num_samples_test, len(self.data_set.p_quantile)), 
                               dtype = torch.float32, device = self.device)
        T_C_pred = torch.zeros((1, self.num_samples_test, len(self.data_set.p_quantile)), 
                               dtype = torch.float32, device = self.device)
        
        Agent_type_combinations = np.unique(ctrl_types, axis = 1).T
         
        batch_all = 0
        for agent_type_combination in Agent_type_combinations:
            combination_samples = np.where((ctrl_types == agent_type_combination[:,np.newaxis]).all(0))[0]
            
            combination_num_samples = len(combination_samples)
        
            n_batches = int(np.ceil(combination_num_samples / batch_size))
            # Assume no training of the model needed, backwards graphs not calculated
            with torch.no_grad():
                Ti = time.time()
                # Go through all the initial scenarios
                for batch in range(n_batches):
                    b_ind = np.arange(batch_size * batch, batch_size * (batch + 1))
                    b_ind = b_ind[b_ind < combination_num_samples]
                    
                    sample_ind = combination_samples[b_ind]
                    
                    ti = time.time()
                    # Get the binary and time prediction for the set of parameters and each initial 
                    # scenario in this batch, with one value for each probebalistic path
                    A, T_A, T_C = self.commotions_model(names             = names,
                                                        ctrl_types        = ctrl_types[:, sample_ind],
                                                        params            = variable_params,
                                                        initial_positions = initial_positions[:, sample_ind], 
                                                        speeds            = speeds[:, sample_ind], 
                                                        accs              = accs[:, sample_ind],
                                                        coll_dist         = coll_dist[:, sample_ind], 
                                                        free_speeds       = free_speeds[:, sample_ind],
                                                        T_out             = T_out[sample_ind],
                                                        dt                = dt,
                                                        const_accs        = self.const_accs)
                    
                    # Get the predictioned likelihood of accepteding the gap and the probability 
                    # distribution of the time of acceptance depicted by its decile values
                    [A_pred[:,sample_ind], 
                     T_A_pred[:,sample_ind], 
                     T_C_pred[:,sample_ind]] = self.commotions_model.predict(A, T_A, T_C)
                    to = time.time()
                    batch_all += 1
                    if n_batches > 1:
                        print('Prediction - Batch {}: {:0.3f} s'.format(batch_all + 1, to - ti),flush=True)
                    
                print('Prediction - all batches: {:0.3f} s'.format(to - Ti),flush=True)
                print('',flush=True)
        
        A_pred   = A_pred.squeeze(0)
        T_A_pred = T_A_pred.squeeze(0)
        T_C_pred = T_C_pred.squeeze(0)
        
        # Transofrm the prediction to numpy arrays
        Output_A_pred = A_pred.cpu().detach().numpy().astype('float64')
        Output_A_pred = np.stack((Output_A_pred, 1 - Output_A_pred), axis = 1)
        Output_T_E_pred = np.stack((T_A_pred.cpu().detach().numpy().astype('float32'),
                                    T_C_pred.cpu().detach().numpy().astype('float32')), axis = 1)
        
        return [Output_A_pred, Output_T_E_pred]