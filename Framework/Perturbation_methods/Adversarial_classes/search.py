import copy
import numpy as np
import torch


class Search:
    @staticmethod
    def hard_constraint(positions_perturb, perturbation_tensor, ego_agent_index, tar_agent_index, hard_bound, physical_bounds, dt, device):
        '''
        Apply hard constraints to the perturbation tensor.

        This method applies hard constraints to the perturbation tensor by setting the perturbation values to zero if they exceed the specified hard bound.
        The constraints are applied to the perturbation tensor for the target agent, while the perturbation values for the ego agent are set to zero.

        Parameters:
        positions_perturb (torch.Tensor): Array containing the position data with shape (batch size, number agents, number time steps, coordinates (x,y)).
        perturbation_tensor (torch.Tensor): Tensor containing the perturbation data with shape (batch size, number agents, number time steps, coordinates (x,y)).
        ego_agent_index (int): Index of the ego agent.
        tar_agent_index (int): Index of the target agent.
        hard_bound (float): Hard bound value for the perturbation.
        physical_bounds (dict): Dictionary containing the physical bounds for the agents.
        dt (float): Time step value used for metric calculations.

        Returns:
        torch.Tensor: Tensor containing the perturbation data with the hard constraints applied.
        '''
        # Convert the perturbation tensor and positions_perturb to numpy arrays if they are not already
        if not isinstance(perturbation_tensor, np.ndarray):
            perturbation_array = perturbation_tensor.cpu().detach().numpy()

        if not isinstance(positions_perturb, np.ndarray):
            positions_perturb = positions_perturb.cpu().detach().numpy()

        # Set the perturbation values for the ego agent to zero
        perturbation_array[:, ego_agent_index, :, :] = 0

        # Initialize the theta
        step = 0.01
        theta = 1 + step

        # initialize the check_pass array
        check_pass = [False] * perturbation_array.shape[0]

        # initialize the theta_storage array
        theta_storage = np.zeros_like(perturbation_array)

        # Apply hard constraints to the perturbation tensor for the target agent
        while not all(check_pass):
            # Update the theta value
            theta -= 0.01
            if theta <= 0.01:
                break

            # Merge the perturbation tensor with the positions_perturb
            merged_trace_array = copy.deepcopy(positions_perturb)
            merged_trace_array += theta * perturbation_array

            # Calculate the physical metrics
            scalar_v, linear_a, rotate_a, linear_aa, rotate_aa = Search.get_metrics(
                merged_trace_array, dt)
            # deviation = Search.get_deviation(theta * perturbation_array)

            # Check if the perturbation values exceed the hard bound
            for i in range(scalar_v.shape[0]):
                # Skip if the check has already passed
                if check_pass[i]:
                    continue

                # checks = [
                #     np.all(scalar_v[i, tar_agent_index] <= physical_bounds["scalar_v"][tar_agent_index]),
                #     np.all(linear_a[i, tar_agent_index] <= physical_bounds["linear_a"][tar_agent_index]),
                #     np.all(rotate_a[i, tar_agent_index] <= physical_bounds["rotate_a"][tar_agent_index]),
                #     np.all(linear_aa[i, tar_agent_index] <= physical_bounds["linear_aa"][tar_agent_index]),
                #     np.all(rotate_aa[i, tar_agent_index] <= physical_bounds["rotate_aa"][tar_agent_index]),
                #     np.all(alignment[i, tar_agent_index] >= 0)  
                # ]

                checks = [
                    np.all(scalar_v[i, tar_agent_index] <= physical_bounds["scalar_v"][tar_agent_index]),
                    np.all(linear_a[i, tar_agent_index] <= physical_bounds["linear_a"][tar_agent_index]),
                    np.all(rotate_a[i, tar_agent_index] <= physical_bounds["rotate_a"][tar_agent_index]),
                    np.all(linear_aa[i, tar_agent_index] <= physical_bounds["linear_aa"][tar_agent_index]),
                    np.all(rotate_aa[i, tar_agent_index] <= physical_bounds["rotate_aa"][tar_agent_index])  
                ]

                if all(checks):
                    check_pass[i] = True
                    theta_storage[i, tar_agent_index, :, :] = theta

        return perturbation_tensor * torch.tensor(theta_storage).to(device)
    

    @staticmethod
    def get_deviation(perturbation_array):
        '''
        Calculate the deviation of the perturbation array.

        This method calculates the deviation of the perturbation array by taking the square root of the sum of the squares of the perturbation array.

        Parameters:
        perturbation_array (numpy.ndarray): Array containing the perturbation data with shape (batch size, number agents, number time steps, coordinates (x,y)).

        Returns:
        numpy.ndarray: Array containing the deviation values with shape (batch size, number agents, number time steps).
        '''
        return np.sum(perturbation_array ** 2, axis=-1) ** 0.5

    @staticmethod
    def get_unit_vector(vectors):
        """
        Calculate and return the unit vectors for the given vectors.

        This method computes the unit vectors for a given array of vectors by normalizing each vector. 
        The input vectors are assumed to have the shape (batch size, number agents, number vectors, coordinates (x,y)).

        Parameters:
        vectors (numpy.ndarray): Array containing vectors with shape (batch size, number agents, number vectors, coordinates (x,y)).

        Returns:
        numpy.ndarray: Array containing the unit vectors with the same shape as the input.
        """
        scale = np.sum(vectors ** 2, axis=-1, keepdims=True) ** 0.5 + 0.001
        result = np.zeros_like(vectors)
        result[..., 0] = vectors[..., 0] / scale[..., 0]
        result[..., 1] = vectors[..., 1] / scale[..., 0]
        return result

    @staticmethod
    def get_metrics(trace_array, dt):
        """
        Calculate various physical metrics from the trace array.

        Parameters:
        trace_array (numpy.ndarray): Array containing the trace data with shape (batch size, number agents, number time steps, coordinates (x,y)).
        dt (float): Time step value used for metric calculations.

        Returns:
        dict: A dictionary containing the computed metrics:
            - scalar_v: Scalar velocity.
            - linear_a: Linear acceleration.
            - rotate_a: Rotational acceleration.
            - linear_aa: Linear angular acceleration.
            - rotate_aa: Rotational angular acceleration.
        """
        # Calculate the velocity, acceleration, and angular acceleration
        v = trace_array[..., 1:, :] - trace_array[..., :-1, :]
        a = v[..., 1:, :] - v[..., :-1, :]
        aa = a[..., 1:, :] - a[..., :-1, :]
        
        direction = Search.get_unit_vector(v)

        # Calculate the perpendicular direction vector 
        direction_r = np.zeros_like(direction)
        direction_r[..., 0] = direction[..., 1]
        direction_r[..., 1] = -direction[..., 0]
        
        scalar_v = np.sum(v ** 2, axis=-1) ** 0.5
        linear_a = np.abs(np.sum(direction[..., :-1, :] * a, axis=-1))
        rotate_a = np.abs(np.sum(direction_r[..., :-1, :] * a, axis=-1))
        linear_aa = np.abs(np.sum(direction[..., :-2, :] * aa, axis=-1))
        rotate_aa = np.abs(np.sum(direction_r[..., :-2, :] * aa, axis=-1))
        
        return scalar_v, linear_a, rotate_a, linear_aa, rotate_aa

    @staticmethod
    def filtered_max_data(data, data_type):
        """
        Calculate and return the maximum value from the filtered data.

        This method takes an array of data, flattens it, removes NaN values, 
        and filters the data to exclude the extreme values (below the 1st percentile 
        and above the 99th percentile). It then computes the maximum value from the filtered data.

        Parameters:
        data (array-like): Input array containing the data to be filtered and analyzed.

        Returns:
        float: The maximum value from the filtered data.
        """
        
        if data_type == "rotate_a" or data_type == "rotate_aa":
            percentile_min = 0.1
            percentile_max = 99.9
        else:
            percentile_min = 0
            percentile_max = 100

        # Remove NaN values
        data_array = np.nan_to_num(data)

        # Reshape the data to (batch_size * time_steps, number_agents)
        data_reshaped = np.swapaxes(
            data_array, 0, 1).reshape(data_array.shape[1], -1)

        max_values = []

        for agent_data in data_reshaped:  # Transpose to iterate over agents
            percentile_001 = np.percentile(agent_data, percentile_min)
            percentile_999 = np.percentile(agent_data, percentile_max)

            # Filter the data based on the 1st and 99th percentile values
            filtered_data = agent_data[(agent_data >= percentile_001) & (
                agent_data <= percentile_999)]

            # Calculate the maximum value of the filtered data
            max_value = np.max(filtered_data)

            max_values.append(max_value)

        return np.array(max_values)

    @staticmethod
    def get_physical_constraints(X, Y, dt):
        """
        Calculate and return the maximum physical constraints from the given data.

        This method processes the input arrays X (historical data) and Y (future data) to compute various physical 
        metrics, and then calculates the filtered maximum values for these metrics.

        Parameters:
        X (numpy.ndarray): Input array X with shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        Y (numpy.ndarray): Input array Y with shape (batch size, number agents, number time steps future, coordinates (x,y)).
        dt (float): Time step value used for metric calculations.

        Returns:
        dict: A dictionary containing the following keys and their corresponding maximum values:
            - 'max_scalar_v': Maximum scalar velocity.
            - 'max_linear_a': Maximum linear acceleration.
            - 'max_rotate_a': Maximum rotational acceleration.
            - 'max_linear_aa': Maximum linear angular acceleration.
            - 'max_rotate_aa': Maximum rotational angular acceleration.
        """

        # concatenate the historical and future data along the time axis
        data_concatenate = np.concatenate((X, Y), axis=-2)

        scalar_v, linear_a, rotate_a, linear_aa, rotate_aa = Search.get_metrics(
            data_concatenate, dt)

        # calculate the maximum values for the metrics and filter outliers
        constraints = {
            'scalar_v': Search.filtered_max_data(scalar_v, "scalar_v"),
            'linear_a': Search.filtered_max_data(linear_a, "linear_a"),
            'rotate_a': Search.filtered_max_data(rotate_a, "rotate_a"),
            'linear_aa': Search.filtered_max_data(linear_aa, "linear_aa"),
            'rotate_aa': Search.filtered_max_data(rotate_aa, "rotate_aa")
        }

        return constraints
