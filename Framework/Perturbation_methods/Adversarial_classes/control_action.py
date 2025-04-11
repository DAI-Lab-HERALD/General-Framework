import torch
import torch.nn.functional as F


class Control_action:
    @staticmethod
    def Inverse_Dynamical_Model(positions_perturb, mask_data, dt, device):
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

        # Initialize heading and velocity
        heading = torch.zeros(positions_perturb.shape[:3]).to(device)
        velocity = torch.zeros(positions_perturb.shape[:3]).to(device)

        # Remove standing vehicle probles
        # Get cummulative distance between the timepoints
        diff = positions_perturb[:, :, 1:, :] - positions_perturb[:, :, :-1, :]
        dist = torch.norm(diff, p=2, dim=-1)
        dist = torch.nan_to_num(dist, nan=2.0)

        # Distances under 0.1 are considered neglible. Correspondingly, all points where the distance is less than 
        dist_cum = torch.cumsum(dist, dim=-1)
        dist_cum = torch.cat([torch.zeros(*dist_cum.shape[:-1], 1).to(device), dist_cum], dim=-1)
        dist_cum_ind = torch.floor(dist_cum / 0.1)

        # Assume the positions at which the dist_cum_ind is the same as the previous step, remove the recorded position.
        # instead, use linear interpolation
        replace = dist_cum_ind[..., 1:] == dist_cum_ind[..., :-1]
        # Add a flase in the front
        replace = torch.cat([torch.zeros(*replace.shape[:-1], 1).to(device).bool(), replace], dim=-1)
        # Last timestep will never be replaced
        replace[..., -1] = False

        # Linearly interpolate the positions (assume that the timepoints correspond to the dist_cum)
        # For each replace = True, find the previous time index that was not replaced, as well as the following one
        replace_sample, replace_agent, replace_time = torch.where(replace)
        prev_ind = replace_time - 1
        while replace[replace_sample, replace_agent, prev_ind].any():
            redo = replace[replace_sample, replace_agent, prev_ind]
            prev_ind[redo] = prev_ind[redo] - 1
        
        assert prev_ind.min() >= 0, "Previous index is negative, which should not happen"
        next_ind = replace_time + 1
        while replace[replace_sample, replace_agent, next_ind].any():
            redo = replace[replace_sample, replace_agent, next_ind]
            next_ind[redo] = next_ind[redo] + 1
        assert next_ind.max() < positions_perturb.shape[2], "Next index is out of bounds, which should not happen"

        Dist_cum_prev = dist_cum[replace_sample, replace_agent, prev_ind]
        Dist_cum_next = dist_cum[replace_sample, replace_agent, next_ind]
        Dist_cum_inter = dist_cum[replace_sample, replace_agent, replace_time]
        fac = (Dist_cum_inter - Dist_cum_prev) / (Dist_cum_next - Dist_cum_prev)
        fac = fac.unsqueeze(-1)

        # Interpolate the positions
        pos_prev = positions_perturb[replace_sample, replace_agent, prev_ind]
        pos_next = positions_perturb[replace_sample, replace_agent, next_ind]

        positions_perturb_init = positions_perturb.clone()

        debug = False
        if debug:
            x0 = positions_perturb[23,0].detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 7.5))
            plt.plot(x0[:,0], x0[:,1], 'k')
            plt.scatter(x0[:,0], x0[:,1], c='k')
            plt.xlim([16, 17])
            plt.ylim([8, 8.5])
        positions_perturb[replace_sample, replace_agent, replace_time] = fac * pos_next + (1 - fac) * pos_prev
        useful_agent = positions_perturb.isfinite().all(-1).sum(-1) > 2
        if debug:
            x1 = positions_perturb[23,0].detach().cpu().numpy()
            plt.plot(x1[:,0], x1[:,1], 'b')
            plt.scatter(x1[:,0], x1[:,1], c='b')

            use_sample, use_agent = torch.where(useful_agent)
            i_use = torch.where((use_sample == 23) & (use_agent == 0))[0][0]
            move_plot = plt.plot(x1[:,0], x1[:,1], 'r')


        ## Smooth along time dimension
        # Determine if we shoudl smooth from the beginning or the end
        pos_use = positions_perturb[useful_agent] # Shape: (n, num_time_steps, 2)

        i_first = pos_use.isfinite().all(-1).float().argmax(-1)
        pos_first = pos_use.gather(-2, i_first.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0]
        pos_post_first = pos_use.gather(-2, (i_first + 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0]
        assert pos_first.isfinite().all(), "First position is not finite, which should not happen"
        assert pos_post_first.isfinite().all(), "First position is not finite, which should not happen"
        # Get distance between first and second position
        dist_0 = (pos_first - pos_post_first).norm(p=2, dim=-1) # Shape: (n) 

        # Get first np.nan index
        viable_last = torch.arange(pos_use.shape[1]).unsqueeze(0).to(pos_use.device) >= i_first.unsqueeze(-1)
        i_last = (pos_use.isnan().any(-1) & viable_last).float().argmax(-1) - 1
        i_last[i_last == -1] = positions_perturb.shape[2] - 1
        assert i_last.min() > 0
        pos_last = pos_use.gather(-2, i_last.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0]
        pos_pre_last = pos_use.gather(-2, (i_last - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0]
        assert pos_last.isfinite().all(), "Last position is not finite, which should not happen"
        assert pos_pre_last.isfinite().all(), "Last position is not finite, which should not happen"
        # Get distance between last and second last position
        dist_1 = (pos_last - pos_pre_last).norm(p=2, dim=-1)

        Start_from_behind = dist_1 > dist_0

        # Go from last timestep to first
        for i in range(pos_use.shape[1] - 2):
            i_A = torch.ones_like(i_first) * (i+2)
            i_B = i_A - 1
            i_C = i_A - 2

            if Start_from_behind.any():
                i_A[Start_from_behind] = pos_use.shape[1] - i - 3
                i_B[Start_from_behind] = i_A[Start_from_behind] + 1
                i_C[Start_from_behind] = i_A[Start_from_behind] + 2

            A = pos_use.gather(-2, i_A.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0] # Shape: (n, 2)
            B = pos_use.gather(-2, i_B.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0] # Shape: (n, 2)
            C = pos_use.gather(-2, i_C.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2))[:,0] # Shape: (n, 2)
            # Ignore cases with nan
            valid = A.isfinite().all(dim=-1) & B.isfinite().all(dim=-1) & C.isfinite().all(dim=-1) # Shape: (batch_size, num_agents)

            # Get steps
            AB = B - A # Shape: (batch_size, num_agents, 2)
            BC = C - B # Shape: (batch_size, num_agents, 2)

            # get distances and angles 
            dist_AB = torch.norm(AB, p=2, dim=-1) # Shape: (batch_size, num_agents)
            dist_BC = torch.norm(BC, p=2, dim=-1) # Shape: (batch_size, num_agents)
            angle_AB = torch.atan2(AB[:, 1], AB[:, 0]) # Shape: (batch_size, num_agents)
            angle_BC = torch.atan2(BC[:, 1], BC[:, 0]) # Shape: (batch_size, num_agents)

            dist = torch.where(Start_from_behind, dist_AB, dist_BC)
            # Calculate the angle difference
            angle_AB_diff = angle_AB - angle_BC
            # Unwrap the angle difference
            angle_AB_diff = (angle_AB_diff + 0.5 * torch.pi) % (torch.pi) - torch.pi * 0.5
            # Calculate the curvature
            curvature = torch.where(dist != 0, angle_AB_diff / dist, torch.zeros_like(angle_AB_diff)) # Shape: (batch_size, num_agents)

            # Get needed replacement
            violation = (curvature.abs() > 0.2) & valid

            if violation.any():
                i_violation = torch.where(violation)[0]
                Behind_violated = Start_from_behind[i_violation]
                dist_BC_viol = dist_BC[i_violation]
                # Project A onto the line spanned by B and C (call that point Ap) 
                Ap = B + BC * (torch.sum(-AB * BC, dim=-1) / dist_BC ** 2).unsqueeze(-1) # Shape: (batch_size, num_agents, 2)
                ApA = A[i_violation] - Ap[i_violation] # Shape: (n, 2)
                BAp = B[i_violation] - Ap[i_violation] # Shape: (n, 2)
                dist_ApA = torch.norm(ApA, p=2, dim=-1) # Shape: (n)

                # Find the point and the line in between A and Ap for which the resulting curvature is < 0.2
                # For t in [0,1], we have to solve the following equation:
                # f(t) = arcsin (t * ||ApA|| / ||BAp + t * ApA||) / ||BAp + t * ApA|| <= 0.2
                
                # Use bisection, as gradient is too difficult to calculate for newton rhapson
                t_max = torch.ones_like(dist_ApA)
                t_min = torch.zeros_like(dist_ApA)

                num_steps = 15
                for _ in range(num_steps):
                    t = (t_max + t_min) / 2
                    # Calculate the new curvature
                    dist_BAs = torch.norm(BAp + t.unsqueeze(-1) * ApA, p=2, dim=-1) # Shape: (batch_size, num_agents)

                    dist_vel = torch.where(Behind_violated, dist_BAs, dist_BC_viol)
                    # Calculate the new curvature
                    Ft = torch.arcsin(t * dist_ApA / dist_BAs) / dist_vel

                    success = Ft <= 0.2

                    # Smaller t should lead to smaller Ft
                    t_min[success] = t[success]
                    t_max[~success] = t[~success]

                # get the new point by using t_min (guaranteed success)
                As = Ap[i_violation] + t_min.unsqueeze(-1) * ApA

                # overwrite the positions
                pos_use[i_violation,i_A[i_violation], :] = As

            # Overwrite valid positions
            if debug:
                x2 = pos_use[i_use].detach().cpu().numpy()
                # Overwrite move_plot positions
                move_plot[0].set_xdata(x2[:,0])
                move_plot[0].set_ydata(x2[:,1])

                plt.pause(0.1)
                plt.draw()

        if debug:
            plt.scatter(x2[:,0], x2[:,1], c='r')

        positions_perturb[useful_agent] = pos_use

        displacement = positions_perturb - positions_perturb_init
        # Calculate the displacement
        displacement = torch.norm(displacement, p=2, dim=-1).nan_to_num(0.0)

        # update initial velocity and heading
        # heading[:, :, 0] = Control_action.compute_heading(positions_perturb, 0)
        heading[:, :, 0], first_non_zero = Control_action.compute_initial_heading(positions_perturb)
        velocity[:, :, 0] = Control_action.compute_velocity(positions_perturb, dt, 0)
        
        # Create a time step tensor
        time_steps = torch.arange(positions_perturb.shape[2] - 1)

        velocity[:, :, 1:], heading[:, :, 1:] = Control_action.compute_velocity_heading_vect(positions_perturb, time_steps, first_non_zero, heading, dt)

        # acceleration control actions
        acceleration_control = (velocity[:, :, 1:] - velocity[:, :, :-1]) / dt

        # Calculate the change of heading (yaw rate)
        heading_diff = heading[:, :, 1:] - heading[:, :, :-1]
        # Unwrap heading diffs
        heading_diff = (heading_diff + torch.pi) % (2 * torch.pi) - torch.pi
        # Calculate the yaw rate
        yaw_rate = heading_diff / dt
        yaw_rate[yaw_rate.abs() < 0.005] = yaw_rate[yaw_rate.abs() < 0.005] * 0.5

        # curvature control actions
        # curvature_control = yaw_rate / velocity[:, :, :-1]
        curvature_control = torch.where(velocity[:, :, :-1].abs() > 1e-2, yaw_rate / velocity[:, :, :-1], torch.zeros_like(yaw_rate))

        # Update the control actions
        control_action[:, :, :, 0] = acceleration_control
        control_action[:, :, :, 1] = curvature_control

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
    def compute_initial_heading(data):
        """
        Computes the initial heading angle in a dataset by finding the first change in position
        compared to the initial point, using vectorized operations. Differences are checked
        up to 3 decimal places.

        Args:
            data (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps, coordinates (x,y)).

        Returns:
            torch.Tensor: A tensor containing the initial heading angles (in radians) for each agent, with shape (batch_size, number agents).
        """
        # Compute the differences between the initial point and all subsequent points
        diff = data[:, :, 1:] - data[:, :, 0:1]
        
        # Create a mask for non-zero differences (after rounding)
        non_zero_mask = (diff[:, :, :, 0] != 0) | (diff[:, :, :, 1] != 0)
        
        # Find the index of the first non-zero difference
        first_non_zero = non_zero_mask.to(torch.float).argmax(dim=2)
        
        # Check if there are any non-zero differences
        has_non_zero = non_zero_mask.any(dim=2)
        
        # For cases with no non-zero differences, set index to the last timestep
        first_non_zero = torch.where(has_non_zero, first_non_zero, non_zero_mask.shape[2] - 1)
        
        # Gather the appropriate differences based on the first non-zero index
        batch_indices = torch.arange(data.shape[0]).unsqueeze(1).expand(-1, data.shape[1])
        agent_indices = torch.arange(data.shape[1]).unsqueeze(0).expand(data.shape[0], -1)
        
        selected_diff = diff[batch_indices, agent_indices, first_non_zero]
        
        # For cases with all zero differences, use the difference between the first two timesteps
        first_diff = data[:, :, 1] - data[:, :, 0]
        
        # Use first_diff where all differences were zero
        final_diff = torch.where(has_non_zero.unsqueeze(-1), selected_diff, first_diff)
        
        # Compute the heading using atan2
        heading = torch.atan2(final_diff[:, :, 1], final_diff[:, :, 0])
        
        return heading, first_non_zero

    @staticmethod
    def compute_velocity_heading_vect(data, time_steps, first_non_zero, heading_initial, dt):
        """
        Computes the heading angle and velocity of agents in a dataset for multiple time steps.
        Args:
            data (torch.Tensor): A 4-dimensional tensor of shape (batch size, number agents, number time_steps, coordinates (x,y)).
            time_steps (torch.Tensor): A 1-dimensional tensor containing the time step indices to compute the heading from.
            dt (float): The time difference between consecutive time steps.
        Returns:
            tuple: A tuple containing:
                    - velocity (torch.Tensor): A tensor containing the velocities for each agent and time step, with shape (batch size, number agents, number of time steps).
                    - heading (torch.Tensor): A tensor containing the heading angles (in radians) for each point and time step, with shape (batch_size, number agents, number of time steps).
        """
        # Calculate dx and dy for all time steps
        dx = data[:, :, time_steps + 1, 0] - data[:, :, time_steps, 0]
        dy = data[:, :, time_steps + 1, 1] - data[:, :, time_steps, 1]

        # Compute initial heading using atan2
        theta = torch.atan2(dy, dx)

        for i in range(first_non_zero.shape[0]):
            for j in range(first_non_zero.shape[1]):
                for k in range(first_non_zero[i,j]):
                    theta[i, j, k] = heading_initial[i, j, 0]

        # Initialize sign tensor
        sign = torch.ones_like(theta)

        # Calculate angle differences
        angle_diff = torch.abs(theta[:, :, 1:] - theta[:, :, :-1])
        
        # Create a mask for 90-degree (π/2) changes
        half_pi_change_mask = torch.isclose(angle_diff, torch.tensor(torch.pi/2, device=theta.device), atol=0.01)
        three_half_pi_change_mask = torch.isclose(angle_diff, torch.tensor(3*torch.pi/2, device=theta.device), atol=0.01)

        # Combine the masks
        combined_change_mask = half_pi_change_mask | three_half_pi_change_mask

        # Remove the second True value when there are two consecutive True values
        remove_second_true = combined_change_mask & torch.roll(combined_change_mask, shifts=1, dims=-1)
        combined_change_mask = combined_change_mask & ~remove_second_true
        
        # Create a mask for the first occurrence in consecutive 90-degree changes
        first_change_mask = torch.zeros_like(theta, dtype=torch.bool)
        first_change_mask[:, :, 1:] = combined_change_mask

        # Add π/2 only to the first value where a 90-degree change is detected
        theta_filtered = theta.clone()
        theta_filtered = torch.where(first_change_mask, theta_filtered + torch.pi/2, theta_filtered)

        # Compute heading changes
        heading_change = theta_filtered[:, :, 1:] - theta_filtered[:, :, :-1]

        # Normalize heading change to [-pi, pi]
        heading_change = (heading_change + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Detect significant direction changes (more than 90 degrees)
        significant_change = torch.abs(heading_change) > (torch.pi / 2)
        
        # Update sign based on significant changes
        sign[:, :, 1:] = torch.where(significant_change, -sign[:, :, :-1], sign[:, :, :-1])
        
        # Compute cumulative sign changes
        cum_sign = torch.cumprod(sign, dim=-1)
        
        # Calculate velocity magnitude
        velocity_magnitude = torch.sqrt(dx**2 + dy**2) / dt
        
        # Apply cumulative sign to velocity
        velocity = cum_sign * velocity_magnitude
        
        # Recalculate heading using the cumulative sign
        heading = torch.atan2(dy * cum_sign, dx * cum_sign)
        
        # Add π/2 only to the first value where a 90-degree change is detected
        heading_filtered = heading.clone()
        heading_filtered = torch.where(first_change_mask, heading_filtered + torch.pi/2, heading_filtered)
        
        return velocity, heading_filtered
    
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
    def Dynamical_Model(control_action, positions_perturb, heading, velocity, dt, device):
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
        adv_position[:, :, 1:, 0] = torch.cumsum(Velocity[:, :, 1:] * torch.cos(Heading[:, :, 1:]), dim=-1) * dt + adv_position[:, :, 0, 0].unsqueeze(-1)
        adv_position[:, :, 1:, 1] = torch.cumsum(Velocity[:, :, 1:] * torch.sin(Heading[:, :, 1:]), dim=-1) * dt + adv_position[:, :, 0, 1].unsqueeze(-1)

        return adv_position
