import torch


class Control_action:
    @staticmethod
    def inverse_Dynamical_Model(positions_perturb, mask_data, dt, device):
        """
        Computes the control actions, heading, and velocity of agents in a perturbed positions dataset.

        Args:
            positions_perturb (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps, coordinates (x,y)).
            data_concatenate (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps X and Y, coordinates (x,y)).
            dt (float): The time difference between consecutive time steps.
            device (torch.device): The device on which to perform the computations (e.g., 'cpu' or 'cuda').

        Returns:
            tuple: A tuple containing:
                   - control_action (torch.Tensor): A tensor of shape (batch size, number agents, number time_steps - 1, 2)
                     containing the control actions.
                   - heading (torch.Tensor): A tensor of shape (batch size, number agents, number time_steps) containing the headings.
                   - velocity (torch.Tensor): A tensor of shape (batch size, number agents, number time_steps) containing the velocities.
        """
        # Initialize control action
        control_action = torch.zeros(
            (positions_perturb.shape[0], positions_perturb.shape[1], positions_perturb.shape[2]-1, positions_perturb.shape[3])).to(device)

        # Compute the mask values for agents standing still
        mask_length = positions_perturb.shape[2] - 1
        mask = mask_data[:, :, :mask_length, :].to(device)

        # Initialize heading and velocity
        heading = torch.zeros(positions_perturb.shape[:3]).to(device)
        velocity = torch.zeros(positions_perturb.shape[:3]).to(device)

        # update initial velocity and heading
        velocity[:, :, 0] = Control_action.compute_velocity(
            positions_perturb, dt, 0)
        heading[:, :, 0] = Control_action.compute_heading(positions_perturb, 0)

        # Create a time step tensor
        time_steps = torch.arange(positions_perturb.shape[2] - 1)

        # Calculate velocities for all time steps
        all_velocities = Control_action.compute_velocity(
            positions_perturb, dt, time_steps)
        velocity[:, :, 1:] = all_velocities

        # Calculate headings for all time steps
        all_headings = Control_action.compute_heading(
            positions_perturb, time_steps)
        heading[:, :, 1:] = all_headings

        # acceleration control actions
        acceleration_control = (velocity[:, :, 1:] - velocity[:, :, :-1]) / dt

        # Calculate the change of heading (yaw rate)
        yaw_rate = (heading[:, :, 1:] - heading[:, :, :-1]) / dt

        # curvature control actions
        curvature_control = yaw_rate / velocity[:, :, :-1]

        # Update the control actions
        control_action[:, :, :, 0] = torch.where(
            ~mask[:, :, :, 0], acceleration_control, control_action[:, :, :, 0])
        control_action[:, :, :, 1] = torch.where(
            ~mask[:, :, :, 1], curvature_control, control_action[:, :, :, 1])

        return control_action, heading, velocity

    @staticmethod
    def compute_heading(data, time_step):
        """
        Computes the heading angle in a dataset between two consecutive time steps.

        Args:
            data (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps, coordinates (x,y)).
            time_step (int): The index of the time step to compute the heading from.

        Returns:
            torch.Tensor: A tensor containing the heading angles (in radians) for each point, with shape (batch_size, number agents).
        """
        # Calculate dx and dy
        dx = data[:, :, time_step + 1, 0] - data[:, :, time_step, 0]
        dy = data[:, :, time_step + 1, 1] - data[:, :, time_step, 1]

        # Compute heading using atan2
        heading = torch.atan2(dy, dx)

        return heading

    @staticmethod
    def compute_velocity(data, dt, time_step):
        """
        Computes the velocity of agents in a dataset between two consecutive time steps.

        Args:
            data (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps, coordinates (x,y)).
            dt (float): The time difference between consecutive time steps.
            time_step (int): The index of the time step to compute the velocity from.

        Returns:
            torch.Tensor: A tensor containing the velocities for each agent, with shape (batch size, number agents).
        """
        # Compute the displacement between consecutive time steps
        displacement = data[:, :, time_step + 1, :] - data[:, :, time_step, :]

        # Compute the Euclidean norm of the displacement (velocity magnitude)
        velocity = torch.linalg.norm(displacement, dim=-1, ord=2) / dt

        return velocity

    @staticmethod
    def dynamical_model(control_action, positions_perturb, heading, velocity, dt, device):
        """
        Computes the updated positions of agents based on the dynamical model using control actions, initial postion, velocity, and heading.

        Args:
            control_action (torch.Tensor): A tensor of shape (batch size, number agents, number time_steps - 1, 2) containing the control actions.
            positions_perturb (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps, coordinates (x,y)).
            velocity (torch.Tensor): A tensor of shape (batch size, number agents, number time_steps) containing the velocities.
            heading (torch.Tensor): A tensor of shape (batch size, number agents, number time_steps) containing the headings.
            dt (float): The time difference between consecutive time steps.

        Returns:
            torch.Tensor: A tensor containing the updated positions of agents, with shape (batch size, number agents, number time_steps, coordinates (x,y)).
        """
        # Initial velocity and headin
        velocity_init = velocity[:, :, 0].to(device)
        heading_init = heading[:, :, 0].to(device)

        # Adversarial position storage
        adv_position = positions_perturb.clone().detach().to(device)

        # Update adversarial position based on dynamical model
        acc = control_action[:, :, :, 0]
        cur = control_action[:, :, :, 1]

        # Calculate the velocity for all time steps
        Velocity_set = torch.cumsum(acc, dim=-1) * \
            dt + velocity_init.unsqueeze(-1)
        Velocity = torch.cat(
            (velocity_init.unsqueeze(-1), Velocity_set), dim=-1)

        # Calculte the change of heading for all time steps
        D_yaw_rate = Velocity[:, :, :-1] * cur

        # Calculate Heading for all time steps
        Heading = torch.cumsum(D_yaw_rate, dim=-1) * \
            dt + heading_init.unsqueeze(-1)
        Heading = torch.cat((heading_init.unsqueeze(-1), Heading), dim=-1)

        # Calculate the new position for all time steps
        adv_position[:, :, 1:, 0] = torch.cumsum(Velocity[:, :, 1:] * torch.cos(
            Heading[:, :, 1:]), dim=-1) * dt + adv_position[:, :, 0, 0].unsqueeze(-1)
        adv_position[:, :, 1:, 1] = torch.cumsum(Velocity[:, :, 1:] * torch.sin(
            Heading[:, :, 1:]), dim=-1) * dt + adv_position[:, :, 0, 1].unsqueeze(-1)

        return adv_position
