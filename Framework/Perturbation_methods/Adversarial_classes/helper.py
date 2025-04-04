import numpy as np
import torch

from Adversarial_classes.control_action import Control_action


class Helper:
    @staticmethod
    def check_conversion(Data_1, Data_2):
        """
        Checks if two tensors are approximately equal within a specified tolerance.

        Args:
            Data_1 (torch.Tensor): The first tensor to compare.
            Data_2 (torch.Tensor): The second tensor to compare.

        Raises:
            ValueError: If the tensors are not approximately equal within the specified tolerance.
        """
        equal_tensors = torch.allclose(Data_1, Data_2, atol=1e-2)

        if not equal_tensors:
            raise ValueError("The conversion is not correct.")

    @staticmethod
    def create_data_to_perturb(X, Y, loss_function_1, loss_function_2):
        """
        Creates data to perturb based on the specified loss function.

        Args:
            X (torch.Tensor): A tensor containing the initial data with shape (batch size, number agents, number time steps past, coordinates (x,y)).
            Y (torch.Tensor): A tensor containing the future data with shape (batch size, number agents, number time steps future, coordinates (x,y)).
            loss_function_1 (str): A string specifying the first loss function to use for adversarial perturbations.
            loss_function_2 (str): A string specifying the second loss function to use for adversarial perturbations.

        Returns:
            tuple: A tuple containing:
                   - positions_perturb (torch.Tensor): A tensor containing the data to be perturbed.
                   - future_action_included (bool): A boolean indicating whether future states are included in the perturbed data.
        """
        if ('Y_Perturb' in loss_function_1):
            future_action_included = True
            positions_perturb = torch.cat((X, Y), dim=2)
        elif loss_function_2 is None:
            future_action_included = False
            positions_perturb = X
        elif 'Y_Perturb' in loss_function_2:
            future_action_included = True
            positions_perturb = torch.cat((X, Y), dim=2)
        else:
            future_action_included = False
            positions_perturb = X

        return positions_perturb, future_action_included

    @staticmethod
    def validate_adversarial_loss(loss_function):
        """
        Validates the adversarial loss function based on the barrier function for future states.

        Args:
            loss_function (str): A string specifying the loss function to use for adversarial perturbations.

        Raises:
            ValueError: If the loss function is None.
        """
        if loss_function is None:
            raise ValueError("The loss function cannot be None.")

    @staticmethod
    def remove_nan_values(data):
        """
        Removes NaN values from the data by trimming the array to the maximum length where all values are non-NaN
        in the path length channel across all samples.

        Args:
            data (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time_steps, coordinates (x,y)).

        Returns:
            np.ndarray: A trimmed array containing no NaN values in the path length channel.
        """
        # Calculate the maximum length where all values are non-NaN in the path lenght channel across all samples
        max_length = np.min(np.sum(~np.isnan(data[:, :, :, 0]), axis=2)[:, 0])

        # Trim the array to the maximum length without NaN values in the path lenght channel
        Y_trimmed = data[:, :, :max_length, :]

        return Y_trimmed

    @staticmethod
    def return_to_old_shape(Y_new_pert, Y_shape):
        """
        Restores the perturbed tensor to its original shape by appending NaN values.

        Args:
            Y_new_pert (np.ndarray): The perturbed tensor with reduced shape.
            Y_shape (tuple): The original shape of the tensor.

        Returns:
            np.ndarray: The tensor restored to its original shape with NaN values appended.
        """
        nan_array = np.full(
            (Y_shape[0], Y_shape[1], Y_shape[2]-Y_new_pert.shape[2], Y_shape[3]), np.nan)
        return np.concatenate((Y_new_pert, nan_array), axis=2)
    
    @staticmethod
    def return_to_old_shape_pred_1(Y_new_pert, Y_old, Y_shape, ego_agent_index):
        """
        Restores the perturbed tensor to its original shape by appending NaN values.

        Args:
            Y_new_pert (np.ndarray): The perturbed tensor with reduced shape.
            Y_old (np.ndarray): The original tensor.
            Y_shape (tuple): The original shape of the tensor.
            ego_agent_index (int): The index of the ego agent.

        Returns:
            np.ndarray: The tensor restored to its original shape with NaN values appended.
        """
        mean_Y_new_pert = np.expand_dims(np.mean(Y_new_pert, axis=1), axis=1)
        Y_final = np.concatenate((mean_Y_new_pert, np.expand_dims(Y_old[:, ego_agent_index, :, :], axis=1)), axis=1)

        nan_array = np.full(
            (Y_shape[0], Y_shape[1], Y_shape[2]-Y_final.shape[2], Y_shape[3]), np.nan)
        return np.concatenate((Y_final, nan_array), axis=2)

    @staticmethod
    def validate_settings_order(First, Second):
        """
        Validates the order of two boolean settings.

        Args:
            First (bool): The first boolean setting.
            Second (bool): The second boolean setting.

        Raises:
            ValueError: If the second element is True while the first element is False.
        """
        if Second and not First:
            raise ValueError(
                "Second element can only be True if First element is also True.")

    @staticmethod
    def assert_only_one_true(*args):
        # Check that exactly one argument is True
        assert sum(args) == 1, "Assertion Error: Exactly one can be set on True."

    @staticmethod
    def assert_only_zero_or_one_true(*args):
        # Check that exactly one argument is True
        assert sum(
            args) <= 1, "Assertion Error: Only one value can be set on True or none."

    @staticmethod
    def assert_value_is_to_large(*args):
        # Check that the first two values are larger than the third value
        assert args[0] * args[1] >= args[2], "The third value is to large."

    @staticmethod
    def flip_dimensions_2(X_new_pert, Y_new_pert, Y_pred_1, agent_order):
        """
        Reorders the dimensions of the perturbed tensors based on the original agent order if flipping is required.

        Args:
            X_new_pert (np.ndarray): The perturbed X tensor.
            Y_new_pert (np.ndarray): The perturbed Y tensor.
            agent_order (np.ndarray): The original order of agents.

        Returns:
            tuple: A tuple containing:
                - X_new_pert (np.ndarray): The reordered X tensor.
                - Y_new_pert (np.ndarray): The reordered Y tensor.
        """

        agent_order_inverse = np.argsort(agent_order)
        X_new_pert = X_new_pert[:, agent_order_inverse, :, :]
        Y_new_pert = Y_new_pert[:, agent_order_inverse, :, :]
        Y_pred_1 = Y_pred_1[:, agent_order_inverse, :, :]

        return X_new_pert, Y_new_pert, Y_pred_1

    @staticmethod
    def compute_mask_values_standing_still(array):
        """
        Compute the mask where the difference between the first and last elements in the third dimension is less than 0.2.

        Parameters:
        array (numpy.ndarray): The input array to be checked.

        Returns:
        numpy.ndarray: The mask array with True where the condition is met and broadcasted to match the input shape.
        """
        # Compute the mask where the difference between the first and last elements in the third dimension is less than 0.2
        mask = np.abs(array[:, :, 0, :] - array[:, :, -1, :]) < 0.2

        # Broadcast the mask to match the shape of the input tensor
        mask = np.expand_dims(mask, axis=2)
        mask = np.broadcast_to(mask, array.shape)

        # Create a mask of the same shape with all False
        mask_clone = mask.copy()
        mask_clone[:, :, :, 0] = True

        return mask_clone

    @staticmethod
    def compute_mask_values_tensor(tensor):
        """
        Compute the mask where the difference between the first and last elements in the third dimension is less than 0.2
        for a PyTorch tensor.

        Parameters:
        tensor (torch.Tensor): The input tensor to be checked.

        Returns:
        torch.Tensor: The mask tensor with the condition applied and broadcasted to match the input shape.
        """
        # Compute the mask where the difference between the first and last elements in the third dimension is less than 5
        mask = torch.abs(tensor[:, :, 0, :] - tensor[:, :, -1, :]) < 5
        # Broadcast the mask to match the shape of the input tensor
        mask = mask[:, :, None, :].expand_as(tensor)

        # # Identify columns that are all true or all false in the last dimension
        all_true = mask.all(dim=-2)

        # Identify where one element is True and the other is False in the last dimension
        condition = (all_true.sum(dim=-1) == 1)

        mask_clone = mask.clone()

        mask_clone[condition, :] = False

        return mask_clone

    @staticmethod
    def flip_dimensions(X, Y, agent):
        """
        Flips the dimensions of the input arrays based on the specified agent and reorders the agent dimensions.

        Args:
            X (np.ndarray): A 4-dimensional array of shape (batch size, number agents, number time steps observed, coordinates (x,y)).
            Y (np.ndarray): A 4-dimensional array of shape (batch size, number agents, number time steps future, coordinate (x,y)).
            agent (np.ndarray): A 1-dimensional array indicating the type of each agent.

        Returns:
            tuple: A tuple containing:
                   - agent_order (np.ndarray or None): The new order of agents, or None if no flipping is required.
                   - tar_index (int): The index of the target agent.
                   - ego_index (int): The index of the ego agent.
        """
        # Early exit if no dimension flipping is required

        # Determine the indices for the target and ego agents
        i_agent_perturbed = np.where(agent == 'tar')[0][0]
        i_agent_collision = np.where(agent == 'ego')[0][0]

        # Create an array of indices for other agents, excluding the target and ego agents
        other_agents = np.arange(Y.shape[1])
        other_agents = np.delete(other_agents, [i_agent_perturbed, i_agent_collision])

        # Construct a new order for agents: target, ego, followed by the rest
        agent_order = np.array([i_agent_perturbed, i_agent_collision, *other_agents])

        # Rearrange the X and Y arrays according to the new agent order
        X = X[:, agent_order, :, :]
        Y = Y[:, agent_order, :, :]

        # Return the index of tar and ego agent
        tar_index = 0
        ego_index = 1

        return X, Y, agent_order, tar_index, ego_index
    
    def flip_dimensions_index(agent):
        """
        Flips the dimensions of the input arrays based on the specified agent and reorders the agent dimensions.

        Args:
            agent (np.ndarray): A 1-dimensional array indicating the type of each agent.

        Returns:
            tuple: A tuple containing:
                   - X (np.ndarray): The reordered X array.
                   - Y (np.ndarray): The reordered Y array.
                   - agent_order (np.ndarray or None): The new order of agents, or None if no flipping is required.
                   - tar_index (int): The index of the target agent.
                   - ego_index (int): The index of the ego agent.
        """
        # Early exit if no dimension flipping is required

        # Determine the indices for the target and ego agents
        i_agent_perturbed = np.where(agent == 'tar')[0][0]
        i_agent_collision = np.where(agent == 'ego')[0][0]

        # Create an array of indices for other agents, excluding the target and ego agents
        other_agents = np.arange(len(agent))
        other_agents = np.delete(other_agents, [i_agent_perturbed, i_agent_collision])

        # Construct a new order for agents: target, ego, followed by the rest
        agent_order = np.array([i_agent_perturbed, i_agent_collision, *other_agents])

        # Return the index of tar and ego agent
        tar_index = 0
        ego_index = 1

        return agent_order, tar_index, ego_index

    def get_dimensions_physical_bounds(constraints, agent_order):
        """
        Reorders the constraints dictionary based on the agent order.

        Args:
            constraints (dict): A dictionary where each value is a 1-dimensional array with the same length as the number of agents.
            agent_order (np.ndarray): The new order of agents.

        Returns:
            dict: A dictionary with reordered constraints.
        """
        reordered_constraints = {}
        for key, value in constraints.items():
            reordered_constraints[key] = value[agent_order]
        return reordered_constraints

    @staticmethod
    def convert_to_tensor(device, *args):
        """
        Converts multiple inputs to tensors and moves them to the specified device.

        Args:
            device (torch.device): The device to move the tensors to (e.g., 'cpu' or 'cuda').
            *args: Variable length argument list of inputs to be converted to tensors.

        Returns:
            list: A list of converted tensors on the specified device.
        """
        converted_tensors = []
        for arg in args:
            converted_tensors.append(Helper.to_cuda_tensor(arg, device))
        return converted_tensors

    @staticmethod
    def determine_min_max_values_coordinates(data_observed, data_future):
        """
        Determines the minimum and maximum coordinate values of the data.

        Args:
            data_observed (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time steps observed, coordinates (x,y)).
            data_future (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time steps future, coordinates (x,y)).

        Returns:
            tuple: A tuple containing:
                   - min_value_x (np.ndarray): The minimum value of the data.
                   - max_value_x (np.ndarray): The maximum value of the data.
                   - min_value_y (np.ndarray): The minimum value of the data.
                   - max_value_y (np.ndarray): The maximum value of the data.

        """
        min_value_x = np.inf
        max_value_x = -np.inf
        max_value_y = -np.inf
        min_value_y = np.inf

        for i in range(data_observed.shape[0]):
            for j in range(data_future.shape[1]):
                min_value_x = min(min_value_x, np.min(data_future[i, j, :, 0]))
                max_value_x = max(max_value_x, np.max(data_future[i, j, :, 0]))
                min_value_y = min(min_value_y, np.min(data_future[i, j, :, 1]))
                max_value_y = max(max_value_y, np.max(data_future[i, j, :, 1]))

        return min_value_x, max_value_x, min_value_y, max_value_y

    @staticmethod
    def determine_min_max_values_control_actions_acceleration(data_observed, data_future, dt):
        """
        Determines the minimum and maximum control action values of the data.

        Args:
            data_observed (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time steps observed, coordinates (x,y)).
            data_future (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time steps future, coordinates (x,y)).
            dt (float): The time difference between consecutive time steps.

        Returns:
            tuple: A tuple containing:
                   - min_value_acceleration (np.ndarray): The minimum value of the data.
                   - max_value_acceleration (np.ndarray): The maximum value of the data.

        """
        min_value_acceleration = []
        max_value_acceleration = []

        data_observed = Helper.to_cuda_tensor(data_observed, "cpu")
        data_future = Helper.to_cuda_tensor(data_future, "cpu")

        data = torch.cat((data_observed, data_future), dim=-2)

        mask_data = Helper.compute_mask_values_tensor(data)

        control_action, _, _ = Control_action.Inverse_Dynamical_Model(
            data, mask_data, dt, "cpu")

        control_action = Helper.detach_tensor(control_action)[0]

        for i in range(control_action.shape[0]):
            for j in range(control_action.shape[1]):
                min_value_acceleration.append(
                    np.min(control_action[i, j, :, 0]))
                max_value_acceleration.append(
                    np.max(control_action[i, j, :, 0]))

        min_value_acceleration_sorted = np.sort(min_value_acceleration)
        max_value_acceleration_sorted = np.sort(max_value_acceleration)

        min_value_acceleration_sorted_without_nan = Helper.remove_nan_values_from_list(
            min_value_acceleration_sorted)
        max_value_acceleration_sorted_without_nan = Helper.remove_nan_values_from_list(
            max_value_acceleration_sorted)

        min_value_acceleration_selected = min_value_acceleration_sorted_without_nan[int(
            len(min_value_acceleration_sorted_without_nan) * 0.05)]
        max_value_acceleration_selected = max_value_acceleration_sorted_without_nan[int(
            len(max_value_acceleration_sorted_without_nan) * 0.95)]

        return np.max([np.abs(min_value_acceleration_selected), np.abs(max_value_acceleration_selected)])

    def remove_nan_values_from_list(list):
        """
        Removes NaN values from the list.

        Args:
            list (list): The input list to be checked.

        Returns:
            list: The list with NaN values removed.
        """
        return [x for x in list if not np.isnan(x)]

    @staticmethod
    def convert_to_numpy_array(*args):
        """
        Converts multiple inputs to numpy arrays.

        Args:
            *args: Variable length argument list of inputs to be converted to numpy arrays.

        Returns:
            list: A list of converted numpy arrays.
        """
        numpy_array = [np.array(arg) for arg in args]
        return numpy_array

    @staticmethod
    def set_device(device, *args):
        """
        Moves multiple tensors to the specified device.

        Args:
            device (torch.device): The device to move the tensors to (e.g., 'cpu' or 'cuda').
            *args: Variable length argument list of tensors to be moved to the specified device.

        Returns:
            list: A list of tensors moved to the specified device.
        """
        tensor = [arg.to(device=device) for arg in args]
        return tensor

    @staticmethod
    def to_cuda_tensor(data, device):
        """
        Converts the input data to a tensor and moves it to the specified device.

        Args:
            data: The input data to be converted.
            device (torch.device): The device to move the tensor to (e.g., 'cpu' or 'cuda').

        Returns:
            torch.Tensor: The converted tensor on the specified device.
        """
        return torch.from_numpy(data).to(device=device).float()

    @staticmethod
    def detach_tensor(*args):
        """
        Detaches multiple tensors from the computation graph and moves them to the CPU as numpy arrays.

        Args:
            *args: Variable length argument list of tensors to be detached and converted to numpy arrays.

        Returns:
            list: A list of detached tensors converted to numpy arrays.
        """
        detached_tensor = [arg.detach().cpu().numpy() for arg in args]
        return detached_tensor

    @staticmethod
    def relative_clamping(control_action, epsilon_acc_relative, epsilon_curv_relative):
        # Clamp the control actions relative to ground truth (Not finished yet)
        tensor_addition = torch.zeros_like(control_action)
        tensor_addition[:, :, :, 0] = epsilon_acc_relative
        tensor_addition[:, :, :, 1] = epsilon_curv_relative

        # JULIAN: Those function need to be done for the relative clamping
        control_actions_clamp_low = control_action - tensor_addition
        control_actions_clamp_high = control_action + tensor_addition

        return control_actions_clamp_low, control_actions_clamp_high

    @staticmethod
    def convert_data(data, index_batch, index_agent, prediction):
        if prediction:
            data = np.mean(data, axis=1)
            return np.expand_dims(np.expand_dims(data[index_batch, :, :], axis=0), axis=0)
        else:
            return np.expand_dims(np.expand_dims(data[index_batch, index_agent, :, :], axis=0), axis=0)

    @staticmethod
    def check_size_list(list_1, list_2):
        """
        Checks if two lists have the same size.

        Args:
            list_1 (list): The first list to compare.
            list_2 (list): The second list to compare.

        Raises:
            AssertionError: If the two lists do not have the same size.
        """
        assert len(list_1) == len(
            list_2), "The two lists must have the same size."

    @staticmethod
    def is_monotonic(data):
        """
        Checks if the data is monotonic (either increasing or decreasing) along the last dimension.

        Args:
            data (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time_steps, coordinates (x,y)).

        Returns:
            np.ndarray: A boolean array indicating if the data is monotonic for each sub-array.
        """
        # Check for monotonic increasing
        is_increasing = Helper.is_increasing(data)
        # Check for monotonic decreasing
        is_decreasing = Helper.is_decreasing(data)

        # Combine the results to get the final monotonicity status for each sub-array in dim 1
        return np.logical_or(is_increasing, is_decreasing)

    @staticmethod
    def is_increasing(data):
        """
        Checks if the data is monotonically increasing along the last dimension.

        Args:
            data (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time_steps, coordinates (x,y)).

        Returns:
            np.ndarray: A boolean array indicating if the data is increasing for each sub-array.
        """
        return np.all(data[:, :, :-1, 0] <= data[:, :, 1:, 0], axis=-1)

    @staticmethod
    def is_decreasing(data):
        """
        Checks if the data is monotonically decreasing along the last dimension.

        Args:
            data (np.ndarray): A 4-dimensional numpy array of shape (batch size, number agents, number time_steps, coordinates (x,y)).

        Returns:
            np.ndarray: A boolean array indicating if the data is decreasing for each sub-array.
        """
        return np.all(data[:, :, :-1, 0] >= data[:, :, 1:, 0], axis=-1)

    @staticmethod
    def return_data(adv_position, X, Y, future_action):
        """
        Splits or assigns the adversarial position data based on whether future action is included.

        Args:
            adv_position (torch.Tensor): A tensor containing the adversarial positions with shape (batch size, number agents, number time_steps, coordinates (x,y)).
            X (torch.Tensor): A tensor containing the initial data with shape (batch size, number agents, number time steps observed, coordinates (x,y)).
            Y (torch.Tensor): A tensor containing the future data with shape (batch size, number agents, number future steps future, coordinates (x,y)).
            future_action (bool): A boolean indicating whether future action is included.

        Returns:
            tuple: A tuple containing:
                   - X_new (torch.Tensor): The updated adversarial X tensor.
                   - Y_new (torch.Tensor): The updated adversarial Y tensor.
        """
        if future_action:
            X_new, Y_new = torch.split(
                adv_position, [X.shape[2], Y.shape[2]], dim=-2)
        else:
            X_new = adv_position
            Y_new = Y

        return X_new, Y_new

    @staticmethod
    def remove_indices(data, mask):
        """
        Removes indices from the data where the mask is fully true.

        Args:
            data (torch.Tensor): The data tensor to filter.
            mask (torch.Tensor): The mask tensor indicating which data to remove.

        Returns:
            tuple: A tuple containing:
                   - indices_to_remove (list): The list of indices that were removed.
                   - data_filtered (torch.Tensor or None): The filtered data tensor or None if all data is removed.
        """
        indices_to_remove = []
        for i in range(data.shape[0]):
            if torch.all(mask[i, 0, :, :] == True):
                indices_to_remove.append(i)

        if len(indices_to_remove) == data.shape[0]:
            data_filtered = None
        elif len(indices_to_remove) == data.shape[0]-1:
            data_filtered = data[indices_to_remove[0]:indices_to_remove[0]+1]
        else:
            data_filtered = torch.cat(
                [data[i:i+1] for i in range(data.shape[0]) if i not in indices_to_remove], dim=0)

        return indices_to_remove, data_filtered

    @staticmethod
    def add_back_indices(data_old, data_new, indices_to_remove):
        """
        Adds back the removed indices to the new data tensor.

        Args:
            data_old (torch.Tensor): The original data tensor before removal.
            data_new (torch.Tensor): The new data tensor after some indices were removed.
            indices_to_remove (list): The list of indices that were removed.

        Returns:
            torch.Tensor: The data tensor with removed indices added back.
        """
        new_data = torch.zeros_like(data_old)
        j = 0
        for i in range(data_old.shape[0]):
            if i in indices_to_remove:
                new_data[i] = data_old[i:i+1]
            else:
                new_data[i] = data_new[j:j+1]
                j += 1
        return new_data
