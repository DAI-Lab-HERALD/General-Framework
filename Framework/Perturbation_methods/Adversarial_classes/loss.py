from abc import ABC, abstractmethod
import torch

# Abstract base class for loss functions


class LossFunction(ABC):
    @abstractmethod
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        pass

# Abstract base class for barrier functions past states


class BarrierFunctionPast(ABC):
    @abstractmethod
    def calculate_barrier(self, X_new, X, tar_agent):
        pass

# Abstract base class for barrier functions future states


class BarrierFunctionFuture(ABC):
    @abstractmethod
    def calculate_barrier(self, Y_new, Y, tar_agent):
        pass

# Context for calculating loss with optional barrier


class LossContext:
    def __init__(self, loss_strategy_1: LossFunction, loss_strategy_2: LossFunction, barrier_strategy_past: BarrierFunctionPast = None, barrier_strategy_future: BarrierFunctionFuture = None):
        self.loss_strategy_1 = loss_strategy_1
        self.loss_strategy_2 = loss_strategy_2
        self.barrier_strategy_past = barrier_strategy_past
        self.barrier_strategy_future = barrier_strategy_future

    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent, iteration):
        loss = self.loss_strategy_1.calculate_loss(
            X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent)

        if self.loss_strategy_2:
            loss += self.loss_strategy_2.calculate_loss(
                X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent)
            
        if self.barrier_strategy_past:
            barrier_output = self.barrier_strategy_past.calculate_barrier(
                X_new, X, tar_agent)
            loss += barrier_output

        if self.barrier_strategy_future:
            barrier_output = self.barrier_strategy_future.calculate_barrier(
                Y_new, Y, tar_agent)
            loss += barrier_output
        return loss

# Static class containing various loss functions and barrier functions


class Loss:
    @staticmethod
    def calculate_loss(adversarial, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, barrier_data, iteration):
        """
        Calculates the loss based on the specified loss and barrier functions.

        Args:
            X (torch.Tensor): The ground truth postition tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y))
            X_new (torch.Tensor): The perturbed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y))
            Y (torch.Tensor): The ground truth future position tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y))
            Y_new (torch.Tensor): The perturbed future position tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y))
            Y_Pred (torch.Tensor): The predicted future position tensor with array shape (batch size, number predictions (K), number time steps future, coordinates (x,y))
            Y_Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration with array shape (batch size, number predictions (K), number time steps future, coordinates (x,y))
            distance_threshold (float): The distance threshold for the barrier function.
            log_value (float): The logarithm base value for the barrier function.
            barrier_data (torch.Tensor): The data (concatenation of X and Y) for the barrier function with array shape (batch size, number agents, number time steps observed and future, coordinates (x,y))
            loss_function (str): The name of the loss function.
            barrier_function (str): The name of the barrier function.
            tar_agent (int): The index of the target agent.
            ego_agent (int): The index of the ego agent.

        Returns:
            torch.Tensor: The calculated loss.
        """
        loss_function_1 = get_name.loss_function_name(
            adversarial.loss_function_1)
        loss_function_2 = get_name.loss_function_name(
            adversarial.loss_function_2) if adversarial.loss_function_2 else None
        barrier_function_past = get_name.barrier_function_name_past_states(
            adversarial.barrier_function_past, adversarial.distance_threshold_past, adversarial.log_value_past, barrier_data) if adversarial.barrier_function_past else None
        barrier_function_future = get_name.barrier_function_name_future_states(
            adversarial.barrier_function_future, adversarial.distance_threshold_future, adversarial.log_value_future, barrier_data) if adversarial.barrier_function_future else None
        loss_context = LossContext(
            loss_function_1, loss_function_2, barrier_function_past, barrier_function_future)
        return loss_context.calculate_loss(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, adversarial.tar_agent_index, adversarial.ego_agent_index, iteration)

    @staticmethod
    def ADE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent):
        """
        Calculates the Average Displacement Error between tar agent's loss between ground truth future positions and predicted positions.

        Args:
            Y (torch.Tensor): The ground truth future position tensor.
            Pred_t (torch.Tensor): The predicted future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y[:, tar_agent, :, :].unsqueeze(1) - Pred_t, dim=-1, ord=2), dim=-1), dim=-1)

    @staticmethod
    def FDE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent):
        """
        Calculates the Final Displacement Error between tar agent's loss between ground truth future positions and predicted positions.

        Args:
            Y (torch.Tensor): The ground truth future position tensor.
            Pred_t (torch.Tensor): The predicted future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The FDE loss.
        """
        # norm is over the positions -> torch.mean is over the number of prediciton
        return torch.mean(torch.linalg.norm(Y[:, tar_agent, -1, :].unsqueeze(1) - Pred_t[:, :, -1, :], dim=-1, ord=2), dim=-1)

    @staticmethod
    def ADE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent):
        """
        Calculates the ADE loss between tar agent's perturbed future positions and predicted positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Pred_t (torch.Tensor): The predicted future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:, tar_agent, :, :].unsqueeze(1) - Pred_t, dim=-1, ord=2), dim=-1), dim=-1)

    @staticmethod
    def FDE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent):
        """
        Calculates the FDE loss between tar agent's perturbed future positions and predicted positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Pred_t (torch.Tensor): The predicted future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The FDE loss.
        """
        # norm is over the positions -> torch.mean is over the number of prediciton
        return torch.mean(torch.linalg.norm(Y_new[:, tar_agent, -1, :].unsqueeze(1) - Pred_t[:, :, -1, :], dim=-1, ord=2), dim=-1)

    @staticmethod
    def ADE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent):
        """
        Calculates the ADE loss between tar agent's first iteration predicted positions and perturbed future positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Y_new[:, tar_agent, :, :].unsqueeze(1) - Pred_iter_1, dim=-1, ord=2), dim=-1), dim=-1)

    @staticmethod
    def FDE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent):
        """
        Calculates the FDE loss between tar agent's first iteration predicted positions and perturbed future positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The FDE loss.
        """
        # norm is over the positions -> torch.mean is over the number of prediciton
        return torch.mean(torch.linalg.norm(Y_new[:, tar_agent, -1, :].unsqueeze(1) - Pred_iter_1[:, :, -1, :], dim=-1, ord=2), dim=-1)

    @staticmethod
    def ADE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1):
        """
        Calculates the ADE loss between tar agent's current predicted positions and the first iteration predicted positions.

        Args:
            Pred_t (torch.Tensor): The current predicted future position tensor.
            Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps -> outer torch.mean is over the number of prediciton
        return torch.mean(torch.mean(torch.linalg.norm(Pred_t - Pred_iter_1, dim=-1, ord=2), dim=-1), dim=-1)

    @staticmethod
    def FDE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1):
        """
        Calculates the FDE loss between tar agent's current predicted positions and the first iteration predicted positions.

        Args:
            Pred_t (torch.Tensor): The current predicted future position tensor.
            Pred_iter_1 (torch.Tensor): The predicted future position tensor from the first iteration.

        Returns:
            torch.Tensor: The FDE loss.
        """
        # norm is over the positions -> torch.mean is over the number of prediciton
        return torch.mean(torch.linalg.norm(Pred_t[:, :, -1, :] - Pred_iter_1[:, :, -1, :], dim=-1, ord=2), dim=-1)

    @staticmethod
    def ADE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent):
        """
        Calculates the ADE loss between tar agent's perturbed future positions and ground truth future positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Y (torch.Tensor): The ground truth future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The ADE loss.
        """
        # norm is over the positions -> inner torch.mean is over the time steps 
        return torch.mean(torch.linalg.norm(Y_new[:, tar_agent, :, :] - Y[:, tar_agent, :, :], dim=-1, ord=2), dim=-1)

    @staticmethod
    def FDE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent):
        """
        Calculates the FDE loss between tar agent's perturbed future positions and ground truth future positions.

        Args:
            Y_new (torch.Tensor): The perturbed future position tensor.
            Y (torch.Tensor): The ground truth future position tensor.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The FDE loss.
        """
        # norm is over the positions
        return torch.linalg.norm(Y_new[:, tar_agent, -1, :] - Y[:, tar_agent, -1, :], dim=-1, ord=2)

    @staticmethod
    def collision_loss_Y_ego_GT_and_Y_pred_tar(Y, Pred_t, ego_agent):
        """
        Calculates the collision loss between ego agent's ground truth positions and target agent's predicted positions.

        Args:
            Y (torch.Tensor): The ground truth position tensor.
            Pred_t (torch.Tensor): The predicted position tensor.
            ego_agent (int): The index of the ego agent.

        Returns:
            torch.Tensor: The collision loss.
        """
        # norm is over the positions -> torch.mean is over the number of prediciton -> torch.min is over the time steps
        return torch.mean(torch.linalg.norm(Y[:, ego_agent, :, :].unsqueeze(1) - Pred_t, dim=-1, ord=2), dim=-2).min(dim=-1).values

    @staticmethod
    def collision_loss_Y_ego_GT_and_Y_perturb_tar(Y_new, Y, tar_agent, ego_agent):
        """
        Calculates the collision loss between ego agent's ground truth positions and target agent's perturbed positions.

        Args:
            Y_new (torch.Tensor): The perturbed position tensor.
            Y (torch.Tensor): The ground truth position tensor.
            tar_agent (int): The index of the target agent.
            ego_agent (int): The index of the ego agent.

        Returns:
            torch.Tensor: The collision loss.
        """
        # norm is over the positions -> torch.min is over the time steps
        return torch.linalg.norm(Y[:, ego_agent, :, :] - Y_new[:, tar_agent, :, :], dim=-1, ord=2).min(dim=-1).values

    @staticmethod
    def barrier_log_function_Time(distance_threshold, input_data, barrier_data, log_value, tar_agent):
        """
        Calculates the barrier log function based on the distance between tar agent's adversarial and original positions.

        Args:
            distance_threshold (float): The distance threshold for the barrier function.
            input_data (torch.Tensor): The position tensor used for regularization.
            barrier_data (torch.Tensor): The barrier data tensor that aligns the input data.
            log_value (float): The logarithm base value for the barrier function.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The barrier log function value.
        """

        barrier_norm = torch.norm(
            input_data[:, tar_agent, :, :] - barrier_data[:, tar_agent, :, :], dim=-1)
        
        distance_threshold = torch.ones_like(barrier_norm) * distance_threshold
        barrier_log = torch.log(distance_threshold - barrier_norm)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        return torch.mean(barrier_log_new, dim=-1)
    
    def barrier_log_function_Time_V2(distance_threshold, input_data, barrier_data, log_value, tar_agent):
        """
        Calculates the barrier log function based on the distance between tar agent's adversarial and original positions.

        Args:
            distance_threshold (float): The distance threshold for the barrier function.
            input_data (torch.Tensor): The position tensor used for regularization.
            barrier_data (torch.Tensor): The barrier data tensor that aligns the input data.
            log_value (float): The logarithm base value for the barrier function.
            tar_agent (int): The index of the target agent.

        Returns:
            torch.Tensor: The barrier log function value.
        """

        barrier_norm = torch.norm(
            input_data[:, tar_agent, :, :] - barrier_data[:, tar_agent, :, :], dim=-1)
        
        distance_threshold = torch.ones_like(barrier_norm) * distance_threshold
        barrier_log = torch.log(distance_threshold**2 - barrier_norm**2)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        return torch.mean(barrier_log_new, dim=-1)

    @staticmethod
    def barrier_log_function_Trajectory(distance_threshold, input_data, barrier_data, log_value, tar_agent, remove_final):
        """
        Calculates the barrier log function based on the distance between adversarial positions and barrier data.

        Args:
            distance_threshold (float): The distance threshold for the barrier function.
            input_data (torch.Tensor): The position tensor used for regularization.
            barrier_data (torch.Tensor): The barrier data tensor that aligns the input data.
            log_value (float): The logarithm base value for the barrier function.
            tar_agent (int): The index of the target agent.
            remove_final (bool): Whether to remove the final time step from the barrier function.

        Returns:
            torch.Tensor: The barrier log function value.
        """
        # Calculate the distance between the adversarial observed states and barrier data
        X_tar = input_data[:, tar_agent, :, :].unsqueeze(1).unsqueeze(1) # batch x 1 x 1 x nT_X x 2

        # Define lines out of the barrier data
        barrier_lines = torch.stack([barrier_data[:, tar_agent, 1:], barrier_data[:, tar_agent, :-1]], dim=-3).unsqueeze(-2) # batch x 2 x nT_B x 1 x 2

        # Get distance to line points
        distance_line_points = (X_tar - barrier_lines).norm(dim=-1) # batch x 2 x nT_B x nT_X
        distance_line_lower_bound = torch.min(distance_line_points, dim=1).values # batch x nT_B x nT_X

        # Get distance to unbounded barrier lines segments
        D1 = barrier_lines[:, 1] - barrier_lines[:, 0] # batch x nT_B x 1 x 2
        D2 = X_tar[:, 0] - barrier_lines[:, 0] # batch x nT_B x nT_X x 2
        D_cross = D1[..., 0] * D2[..., 1] - D1[..., 1] * D2[..., 0]
        D_dot = D1[..., 0] * D2[..., 0] + D1[..., 1] * D2[..., 1]
        distance_line = torch.abs(D_cross / (D1.norm(dim=-1) + 1e-6))
        rel_spacing = D_dot / (D1.norm(dim=-1) + 1e-6) ** 2
        
        # if 0 < rel_spacing < 1, we use distance_line, otherwise distance_line_lower_bound
        distance = torch.where((rel_spacing > 0) & (rel_spacing < 1), distance_line, distance_line_lower_bound) # batch x nT_B x nT_X
        
        # Get minimum distance over all barrier lines
        distance = distance.min(dim=1).values # batch x nT_X

        if remove_final:
            distance = distance[:, :-1]

        # calculate the barrier function
        distance_threshold = torch.ones_like(distance) * distance_threshold
        barrier_log = torch.log(distance_threshold - distance)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        return torch.mean(barrier_log_new, dim=-1)
    
    @staticmethod
    def barrier_log_function_Trajectory_V2(distance_threshold, input_data, barrier_data, log_value, tar_agent, remove_final):
        """
        Calculates the barrier log function based on the distance between adversarial positions and barrier data.

        Args:
            distance_threshold (float): The distance threshold for the barrier function.
            input_data (torch.Tensor): The position tensor used for regularization.
            barrier_data (torch.Tensor): The barrier data tensor that aligns the input data.
            log_value (float): The logarithm base value for the barrier function.
            tar_agent (int): The index of the target agent.
            remove_final (bool): Whether to remove the final time step from the barrier function.

        Returns:
            torch.Tensor: The barrier log function value.
        """
        # Calculate the distance between the adversarial observed states and barrier data
        X_tar = input_data[:, tar_agent, :, :].unsqueeze(1).unsqueeze(1) # batch x 1 x 1 x nT_X x 2

        # Define lines out of the barrier data
        barrier_lines = torch.stack([barrier_data[:, tar_agent, 1:], barrier_data[:, tar_agent, :-1]], dim=-3).unsqueeze(-2) # batch x 2 x nT_B x 1 x 2

        # Get distance to line points
        distance_line_points = (X_tar - barrier_lines).norm(dim=-1) # batch x 2 x nT_B x nT_X
        distance_line_lower_bound = torch.min(distance_line_points, dim=1).values # batch x nT_B x nT_X

        # Get distance to unbounded barrier lines segments
        D1 = barrier_lines[:, 1] - barrier_lines[:, 0] # batch x nT_B x 1 x 2
        D2 = X_tar[:, 0] - barrier_lines[:, 0] # batch x nT_B x nT_X x 2
        D_cross = D1[..., 0] * D2[..., 1] - D1[..., 1] * D2[..., 0]
        D_dot = D1[..., 0] * D2[..., 0] + D1[..., 1] * D2[..., 1]
        distance_line = torch.abs(D_cross / (D1.norm(dim=-1) + 1e-6))
        rel_spacing = D_dot / (D1.norm(dim=-1) + 1e-6) ** 2
        
        # if 0 < rel_spacing < 1, we use distance_line, otherwise distance_line_lower_bound
        distance = torch.where((rel_spacing > 0) & (rel_spacing < 1), distance_line, distance_line_lower_bound) # batch x nT_B x nT_X
        
        # Get minimum distance over all barrier lines
        distance = distance.min(dim=1).values # batch x nT_X

        if remove_final:
            distance = distance[:, :-1]

        # calculate the barrier function
        distance_threshold = torch.ones_like(distance) * distance_threshold
        barrier_log = torch.log(distance_threshold**2 - distance**2)
        barrier_log_new = barrier_log / torch.log(torch.tensor(log_value))
        return torch.mean(barrier_log_new, dim=-1)

# Helper class for retrieving loss and barrier function instances by name


class get_name:
    @staticmethod
    def loss_function_name(loss_function):
        if loss_function == 'ADE_Y_GT_Y_Pred_Max':
            return ADE_Y_GT_Y_pred_Max_Loss()
        elif loss_function == 'ADE_Y_GT_Y_Pred_Min':
            return ADE_Y_GT_Y_pred_Min_Loss()
        elif loss_function == 'FDE_Y_GT_Y_Pred_Max':
            return FDE_Y_GT_Y_pred_Max_loss()
        elif loss_function == 'FDE_Y_GT_Y_Pred_Min':
            return FDE_Y_GT_Y_pred_Min_loss()
        elif loss_function == 'ADE_Y_Perturb_Y_Pred_Max':
            return ADE_Y_Perturb_Y_pred_Max_Loss()
        elif loss_function == 'ADE_Y_Perturb_Y_Pred_Min':
            return ADE_Y_Perturb_Y_pred_Min_Loss()
        elif loss_function == 'FDE_Y_Perturb_Y_Pred_Max':
            return FDE_Y_Perturb_Y_pred_Max_Loss()
        elif loss_function == 'FDE_Y_Perturb_Y_Pred_Min':
            return FDE_Y_Perturb_Y_pred_Min_Loss()
        elif loss_function == 'ADE_Y_Perturb_Y_GT_Max':
            return ADE_Y_Perturb_Y_GT_Max_Loss()
        elif loss_function == 'ADE_Y_Perturb_Y_GT_Min':
            return ADE_Y_Perturb_Y_GT_Min_Loss()
        elif loss_function == 'FDE_Y_Perturb_Y_GT_Max':
            return FDE_Y_Perturb_Y_GT_Max_Loss()
        elif loss_function == 'FDE_Y_Perturb_Y_GT_Min':
            return FDE_Y_Perturb_Y_GT_Min_Loss()
        elif loss_function == 'ADE_Y_pred_iteration_1_and_Y_Perturb_Max':
            return ADE_Y_pred_iteration_1_and_Y_Perturb_Max_Loss()
        elif loss_function == 'ADE_Y_pred_iteration_1_and_Y_Perturb_Min':
            return ADE_Y_pred_iteration_1_and_Y_Perturb_Min_Loss()
        elif loss_function == 'FDE_Y_pred_iteration_1_and_Y_Perturb_Max':
            return FDE_Y_pred_iteration_1_and_Y_Perturb_Max_Loss()
        elif loss_function == 'FDE_Y_pred_iteration_1_and_Y_Perturb_Min':
            return FDE_Y_pred_iteration_1_and_Y_Perturb_Min_Loss()
        elif loss_function == 'ADE_Y_pred_and_Y_pred_iteration_1_Max':
            return ADE_Y_pred_and_Y_pred_iteration_1_Max_Loss()
        elif loss_function == 'ADE_Y_pred_and_Y_pred_iteration_1_Min':
            return ADE_Y_pred_and_Y_pred_iteration_1_Min_Loss()
        elif loss_function == 'FDE_Y_pred_and_Y_pred_iteration_1_Max':
            return FDE_Y_pred_and_Y_pred_iteration_1_Max_Loss()
        elif loss_function == 'FDE_Y_pred_and_Y_pred_iteration_1_Min':
            return FDE_Y_pred_and_Y_pred_iteration_1_Min_Loss()
        elif loss_function == 'Collision_Y_pred_tar_Y_GT_ego':
            return Collision_Y_pred_tar_Y_GT_ego_Loss()
        elif loss_function == 'Collision_Y_Perturb_tar_Y_GT_ego':
            return Collision_Y_Perturb_tar_Y_GT_ego_Loss()
        elif loss_function == 'Y_Perturb':
            return Y_Perturb_Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

    @staticmethod
    def barrier_function_name_past_states(barrier_function, distance_threshold, log_value, barrier_data):
        if barrier_function == 'Time_specific':
            return TimeBarrierPast(distance_threshold, log_value)
        elif barrier_function == 'Trajectory_specific':
            return TrajectoryBarrierPast(distance_threshold, barrier_data, log_value)
        elif barrier_function == 'Time_Trajectory_specific':
            return TimeTrajectoryBarrierPast(distance_threshold, barrier_data, log_value)
        if barrier_function == 'Time_specific_V2':
            return TimeBarrierPastV2(distance_threshold, log_value)
        elif barrier_function == 'Trajectory_specific_V2':
            return TrajectoryBarrierPastV2(distance_threshold, barrier_data, log_value)
        elif barrier_function == 'Time_Trajectory_specific_V2':
            return TimeTrajectoryBarrierPastV2(distance_threshold, barrier_data, log_value)
        elif barrier_function == 'None':
            return barrier_None_Past()
        else:
            raise ValueError(
                f"Unknown barrier function past: {barrier_function}")

    def barrier_function_name_future_states(barrier_function, distance_threshold, log_value, barrier_data):
        if barrier_function == 'Time_specific':
            return TimeBarrierFuture(distance_threshold, log_value)
        elif barrier_function == 'Trajectory_specific':
            return TrajectoryBarrierFuture(distance_threshold, barrier_data, log_value)
        elif barrier_function == 'Time_Trajectory_specific':
            return TimeTrajectoryBarrierFuture(distance_threshold, barrier_data, log_value)
        if barrier_function == 'Time_specific_V2':
            return TimeBarrierFutureV2(distance_threshold, log_value)
        elif barrier_function == 'Trajectory_specific_V2':
            return TrajectoryBarrierFutureV2(distance_threshold, barrier_data, log_value)
        elif barrier_function == 'Time_Trajectory_specific_V2':
            return TimeTrajectoryBarrierFutureV2(distance_threshold, barrier_data, log_value)
        elif barrier_function == 'None':
            return barrier_None_Future()
        else:
            raise ValueError(
                f"Unknown barrier function future: {barrier_function}")

# Specific loss function implementations


class ADE_Y_GT_Y_pred_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent)


class ADE_Y_GT_Y_pred_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.ADE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent)


class FDE_Y_GT_Y_pred_Max_loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.FDE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent)


class FDE_Y_GT_Y_pred_Min_loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.FDE_loss_Y_GT_and_Y_pred(Y, Pred_t, tar_agent)


class ADE_Y_Perturb_Y_pred_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent)


class ADE_Y_Perturb_Y_pred_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.ADE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent)


class FDE_Y_Perturb_Y_pred_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.FDE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent)


class FDE_Y_Perturb_Y_pred_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.FDE_loss_Y_pred_and_Y_perturb(Y_new, Pred_t, tar_agent)


class ADE_Y_Perturb_Y_GT_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent)


class ADE_Y_Perturb_Y_GT_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.ADE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent)


class FDE_Y_Perturb_Y_GT_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.FDE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent)


class FDE_Y_Perturb_Y_GT_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.FDE_loss_Y_perturb_and_Y_GT(Y_new, Y, tar_agent)


class ADE_Y_pred_iteration_1_and_Y_Perturb_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent)


class ADE_Y_pred_iteration_1_and_Y_Perturb_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.ADE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent)


class FDE_Y_pred_iteration_1_and_Y_Perturb_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.FDE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent)


class FDE_Y_pred_iteration_1_and_Y_Perturb_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.FDE_loss_Y_pred_iteration_1_and_Y_perturb(Y_new, Pred_iter_1, tar_agent)


class ADE_Y_pred_and_Y_pred_iteration_1_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.ADE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1)


class ADE_Y_pred_and_Y_pred_iteration_1_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.ADE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1)


class FDE_Y_pred_and_Y_pred_iteration_1_Max_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return -Loss.FDE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1)


class FDE_Y_pred_and_Y_pred_iteration_1_Min_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.FDE_loss_Y_pred_and_Y_pred_iteration_1(Pred_t, Pred_iter_1)


class Collision_Y_pred_tar_Y_GT_ego_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_pred_tar(Y, Pred_t, ego_agent)


class Collision_Y_Perturb_tar_Y_GT_ego_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return Loss.collision_loss_Y_ego_GT_and_Y_perturb_tar(Y_new, Y, tar_agent, ego_agent)


class Y_Perturb_Loss(LossFunction):
    def calculate_loss(self, X, X_new, Y, Y_new, Pred_t, Pred_iter_1, tar_agent, ego_agent):
        return 0

# Specific barrier function implementations


class TimeBarrierPast(BarrierFunctionPast):
    def __init__(self, distance_threshold, log_value):
        self.distance_threshold = distance_threshold
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return -Loss.barrier_log_function_Time(self.distance_threshold, X_new, X, self.log_value, tar_agent)
    
class TimeBarrierPastV2(BarrierFunctionPast):
    def __init__(self, distance_threshold, log_value):
        self.distance_threshold = distance_threshold
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return -Loss.barrier_log_function_Time_V2(self.distance_threshold, X_new, X, self.log_value, tar_agent)


class TimeBarrierFuture(BarrierFunctionFuture):
    def __init__(self, distance_threshold, log_value):
        self.distance_threshold = distance_threshold
        self.log_value = log_value

    def calculate_barrier(self, Y_new, Y, tar_agent):
        return -Loss.barrier_log_function_Time(self.distance_threshold, Y_new, Y, self.log_value, tar_agent)
    

class TimeBarrierFutureV2(BarrierFunctionFuture):
    def __init__(self, distance_threshold, log_value):
        self.distance_threshold = distance_threshold
        self.log_value = log_value

    def calculate_barrier(self, Y_new, Y, tar_agent):
        return -Loss.barrier_log_function_Time_V2(self.distance_threshold, Y_new, Y, self.log_value, tar_agent)


class TrajectoryBarrierPast(BarrierFunctionPast):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return -Loss.barrier_log_function_Trajectory(self.distance_threshold, X_new, self.barrier_data, self.log_value, tar_agent, remove_final=False)


class TrajectoryBarrierPastV2(BarrierFunctionPast):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return -Loss.barrier_log_function_Trajectory_V2(self.distance_threshold, X_new, self.barrier_data, self.log_value, tar_agent, remove_final=False)
    

class TrajectoryBarrierFuture(BarrierFunctionFuture):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, Y_new, Y, tar_agent):
        return -Loss.barrier_log_function_Trajectory(self.distance_threshold, Y_new, self.barrier_data, self.log_value, tar_agent, remove_final=False)


class TrajectoryBarrierFutureV2(BarrierFunctionFuture):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, Y_new, Y, tar_agent):
        return -Loss.barrier_log_function_Trajectory_V2(self.distance_threshold, Y_new, self.barrier_data, self.log_value, tar_agent, remove_final=False)


class TimeTrajectoryBarrierPast(BarrierFunctionPast):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return -Loss.barrier_log_function_Trajectory(self.distance_threshold, X_new, self.barrier_data, self.log_value, tar_agent, remove_final=True) - Loss.barrier_log_function_Time(self.distance_threshold, X_new[:, :, -1, :].unsqueeze(2), X[:, :, -1, :].unsqueeze(2), self.log_value, tar_agent)


class TimeTrajectoryBarrierPastV2(BarrierFunctionPast):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, X_new, X, tar_agent):
        return -Loss.barrier_log_function_Trajectory_V2(self.distance_threshold, X_new, self.barrier_data, self.log_value, tar_agent, remove_final=True) - Loss.barrier_log_function_Time_V2(self.distance_threshold, X_new[:, :, -1, :].unsqueeze(2), X[:, :, -1, :].unsqueeze(2), self.log_value, tar_agent)
    

class TimeTrajectoryBarrierFuture(BarrierFunctionFuture):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, Y_new, Y, tar_agent):
        return -Loss.barrier_log_function_Trajectory(self.distance_threshold, Y_new, self.barrier_data, self.log_value, tar_agent, remove_final=True) - Loss.barrier_log_function_Time(self.distance_threshold, Y_new[:, :, -1, :].unsqueeze(2), Y[:, :, -1, :].unsqueeze(2), self.log_value, tar_agent)


class TimeTrajectoryBarrierFutureV2(BarrierFunctionFuture):
    def __init__(self, distance_threshold, barrier_data, log_value):
        self.distance_threshold = distance_threshold
        self.barrier_data = barrier_data
        self.log_value = log_value

    def calculate_barrier(self, Y_new, Y, tar_agent):
        return -Loss.barrier_log_function_Trajectory_V2(self.distance_threshold, Y_new, self.barrier_data, self.log_value, tar_agent, remove_final=True) - Loss.barrier_log_function_Time_V2(self.distance_threshold, Y_new[:, :, -1, :].unsqueeze(2), Y[:, :, -1, :].unsqueeze(2), self.log_value, tar_agent)
    

class barrier_None_Past(BarrierFunctionPast):
    def calculate_barrier(self, X_new, X, tar_agent):
        return 0


class barrier_None_Future(BarrierFunctionFuture):
    def calculate_barrier(self, Y_new, Y, tar_agent):
        return 0
