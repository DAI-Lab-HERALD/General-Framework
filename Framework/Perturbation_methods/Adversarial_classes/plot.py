import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import gridspec
import torch
from pathlib import Path

from Adversarial_classes.helper import Helper
from Adversarial_classes.spline import Spline
from Adversarial_classes.control_action import Control_action


class Plot:
    def __init__(self, adversarial):
        # control_action_settings
        self.future_action = adversarial.future_action_included
        self.dt = adversarial.dt

        # plotting setting
        try:
            self.control_action_graph = adversarial.control_action_graph
        except:
            self.control_action_graph = False

        # device
        self.device = adversarial.pert_model.device

        # agents index
        self.tar_agent = adversarial.tar_agent_index
        self.ego_agent = adversarial.ego_agent_index

        # Clamping values
        try:
            self.epsilon_acc_absolute = adversarial.epsilon_acc_absolute
            self.epsilon_curv_absolute = adversarial.epsilon_curv_absolute
            self.epsilon_acc_relative = adversarial.epsilon_acc_relative
            self.epsilon_curv_relative = adversarial.epsilon_curv_relative
        except:
            pass

        # smoothing sigmas
        try:
            self.sigma_acceleration = adversarial.sigma_acceleration
            self.sigma_curvature = adversarial.sigma_curvature
            self.sigma_count = len(self.sigma_acceleration)
            self.smoothing_strategy = 'Control_Action'   
        except:
            self.sigma = adversarial.sigma
            self.sigma_count = len(self.sigma)
            self.smoothing_strategy = 'Position'

        # animation interpolation
        self.interpolation = 4
        self.dt_new = self.dt / self.interpolation

        # car dimensions
        self.wheelbase = adversarial.wheelbase
        self.car_length = adversarial.car_length
        self.car_width = adversarial.car_width

        # loss function
        self.Name_attack = adversarial.loss_function_1
        self.Name_attack_2 = adversarial.loss_function_2

    def plot_static_data(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier, plot_input):
        # Iterate over each example in the data
        for index_batch in range(X.shape[0]):
            fig, ax = self.subplot_setup(
                index_batch, 'Static scene', scene='static')

            for index_agent in range(X.shape[1]):
                if plot_input:
                    # Plot barrier data
                    ax.plot(data_barrier[index_batch, index_agent, :, 0], data_barrier[index_batch, index_agent,
                                                                                     :, 1], marker='o', color='m', label='Spline data', markersize=6, alpha=0.2)

                    # Plot ego and tar data
                    self.plot_ego_and_tar_agent(X=X, X_new=None, Y=Y, Y_new=None, Y_Pred=None, Y_Pred_iter_1=None, figure_input=ax,
                                                index_batch=index_batch, index_agent=index_agent, future_action=False, style='nominal')
                else:
                    # Plot the adversarial data
                    self.plot_ego_and_tar_agent(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1,
                                                figure_input=ax, index_batch=index_batch, index_agent=index_agent, future_action=self.future_action, style=None)

            # Setup the plot
            if plot_input:
                self.plot_settings(figure_input=ax, min_value_x=-120, max_value_x=20, min_value_y=-30,
                                   max_value_y=20, title=f'Example {index_batch} of batch - Input data plot', legend=True)
            else:
                self.plot_settings(figure_input=ax, min_value_x=-120, max_value_x=20, min_value_y=-30,
                                   max_value_y=20, title=f'Example {index_batch} of batch - Adversarial data plot', legend=True)

            plt.show()

    def plot_loss_over_iterations(self, loss_store):
        # Plot the loss over the iterations
        loss_store = np.array(loss_store)
        plt.figure(0)
        for i in range(loss_store.shape[1]):
            plt.plot(loss_store[:, i], marker='o',
                     linestyle='-', label=f'Sample {i}')
        plt.title('Loss for samples')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    def plot_animated_adv_scene(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, control_action = None, perturbed_control_action = None):
        for index_batch in range(X.shape[0]):
            # Set the length of the interpolation
            self.number_interpolation = (
                X.shape[2] + Y.shape[2]-1) * self.interpolation

            # Interpolate the data to smooth the animation
            self.create_interpolated_data_animation(
                X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, index_batch)
            
            # Create the number count same lenght as interpolation
            self.num_count = np.arange(
                0, self.number_interpolation, 1)

            # Setup the subplots
            if self.control_action_graph:
                # Create subplots
                fig, plot_acc, plot_curv, animation_ax, zoom, static = self.subplot_setup(
                    index_batch=index_batch, title=f'Example {index_batch} of batch - Adversarial scene plot animated', scene='animation')

                # Add control action details
                self.add_clamping_limits(
                    plot_acc, plot_curv, control_action, perturbed_control_action, index_batch, X, Y)

            else:
                fig, animation_ax, zoom, static = self.subplot_setup(
                    index_batch=index_batch, title=f'Example {index_batch} of batch - Adversarial scene plot animated', scene='animation')

            # Initialize the cars
            self.cars_initialization(animation_ax)

            # setup the animation
            if self.control_action_graph:
                self.plot_settings(figure_input=animation_ax, min_value_x=-100, max_value_x=10,
                                   min_value_y=-20, max_value_y=5, title='Animation of the adversarial scene', legend=True)
            else:
                self.plot_settings(figure_input=animation_ax, min_value_x=-100, max_value_x=10, min_value_y=-
                                   30, max_value_y=10, title='Animation of the adversarial scene', legend=True)

            # Create the animation
            if self.control_action_graph:
                ani = animation.FuncAnimation(fig, self.update, self.number_interpolation-1, fargs=[self.num_count, index_batch, self.tar_agent_control_actions, self.adv_agent_control_actions],
                                          interval=self.dt_new*1000, blit=False)
            else:
                ani = animation.FuncAnimation(fig, self.update, self.number_interpolation-1, fargs=[self.num_count, index_batch],
                                          interval=self.dt_new*1000, blit=False)

            # setup the zoom figure
            for index_agent in range(X.shape[1]):
                self.plot_ego_and_tar_agent(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1,
                                            figure_input=zoom, index_batch=index_batch, index_agent=index_agent, future_action=self.future_action, style=None)

            self.plot_settings(figure_input=zoom, min_value_x=2, max_value_x=10, min_value_y=-2,
                               max_value_y=4, title='Zoomed adversarial scene plot', legend=False)

            # setup the static figure
            for index_agent in range(X.shape[1]):
                self.plot_ego_and_tar_agent(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1,
                                            figure_input=static, index_batch=index_batch, index_agent=index_agent, future_action=self.future_action, style=None)

            if self.control_action_graph:
                self.plot_settings(figure_input=static, min_value_x=-110, max_value_x=10,
                                   min_value_y=-12.5, max_value_y=5, title='Adversarial scene static', legend=True)
            else:
                self.plot_settings(figure_input=static, min_value_x=-80, max_value_x=10,
                                   min_value_y=-15, max_value_y=5, title='Adversarial scene static', legend=True)

            # Plot the rectangle for zoom
            static.add_patch(patches.Rectangle(
                (2, -2), 8, 6, edgecolor='black', facecolor='none', linestyle='dashed', linewidth=1))

            # Adding an arrow to point from figure to figure
            self.add_arrow_animation(fig)

            # ani.save(f'{Path(__file__).parent}/Animations/Animation_scene_{Name_attack}-{np.random.rand(1)}.mp4')
            if self.Name_attack_2:
                ani.save(
                    f'Animation_scene_{self.Name_attack}-{self.Name_attack_2}-{np.random.rand(1)}.mp4')
            else:
                ani.save(
                    f'Animation_scene_{self.Name_attack}-{np.random.rand(1)}.mp4')

            plt.show()

    def plot_smoothing(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv):
        # Plot the randomized smoothing
        for index_batch in range(X.shape[0]):
            # loop over the sigmas
            for index_sigma in range(self.sigma_count):
                # Setup smoohting plot
                if self.smoothing_strategy == 'Control_Action':
                    fig, ax, ax1, ax2 = self.subplot_setup(
                        index_batch=index_batch, title=f'Example {index_batch} of batch, sigma acceleration: {self.sigma_acceleration[index_sigma]}, sigma curvature: {self.sigma_curvature[index_sigma]} - Randomized smoothing plot', scene='smoothing')
                else:
                    fig, ax, ax1, ax2 = self.subplot_setup(
                        index_batch=index_batch, title=f'Example {index_batch} of batch, sigma: {self.sigma[index_sigma]} - Randomized smoothing plot', scene='smoothing')
                    
                # Plot 1: Plot the smoothed nominal scene
                for index_agent in range(X.shape[1]):
                    self.plot_ego_and_tar_agent(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1, figure_input=ax,
                                                index_batch=index_batch, index_agent=index_agent, future_action=self.future_action, style='unperturbed')

                self.plot_smoothed_data(X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv,
                                        index_sigma=index_sigma, index_batch=index_batch, style='unperturbed', figure_input=ax)

                # setting for plot nominal scene
                self.plot_settings(figure_input=ax, min_value_x=-20, max_value_x=15, min_value_y=-10,
                                   max_value_y=5, title='Smoothing in nominal setting', legend=False)

                # Plot 2: Plot the smoothed adversarial scene
                for index_agent in range(X.shape[1]):
                    self.plot_ego_and_tar_agent(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1, figure_input=ax1,
                                                index_batch=index_batch, index_agent=index_agent, future_action=self.future_action, style='perturbed')

                self.plot_smoothed_data(X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv,
                                        index_sigma=index_sigma, index_batch=index_batch, style='perturbed', figure_input=ax1)
                # setting for plot nominal scene
                self.plot_settings(figure_input=ax1, min_value_x=-20, max_value_x=15, min_value_y=-10,
                                   max_value_y=5, title='Smoothing in adversarial setting', legend=False)

                # Plot 3: Plot the smoothed nominal and adversarial scene
                for index_agent in range(X.shape[1]):
                    self.plot_ego_and_tar_agent(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1, figure_input=ax2,
                                                index_batch=index_batch, index_agent=index_agent, future_action=self.future_action, style='adv_smoothing')

                self.plot_smoothed_data(X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv,
                                        index_sigma=index_sigma, index_batch=index_batch, style='adv_smoothing', figure_input=ax2)

                # setting for plot nominal scene
                self.plot_settings(figure_input=ax2, min_value_x=-50, max_value_x=15, min_value_y=-10,
                                   max_value_y=5, title='Adversarial scene plot with smoothed prediction', legend=True)

                plt.show()

    def subplot_setup(self, index_batch, title, scene):
        # Create subplots
        fig = plt.figure(figsize=(18, 12), dpi=1920/16)
        fig.suptitle(title)

        # Create subplots
        if scene == 'static':
            ax = fig.add_subplot(1, 1, 1)
            return fig, ax

        if scene == 'animation':
            if self.control_action_graph:
                gs = gridspec.GridSpec(3, 4, figure=fig)
                plot_acc = fig.add_subplot(gs[0, :2])
                plot_curv = fig.add_subplot(gs[0, 2:])
                animation_ax = fig.add_subplot(gs[1, :3])
                zoom = fig.add_subplot(gs[1, 3])
                static = fig.add_subplot(gs[2, :])
                return fig, plot_acc, plot_curv, animation_ax, zoom, static
            else:
                gs = gridspec.GridSpec(2, 4, figure=fig)
                animation_ax = fig.add_subplot(gs[0, :3])
                zoom = fig.add_subplot(gs[0, 3])
                static = fig.add_subplot(gs[1, :])
                return fig, animation_ax, zoom, static

        if scene == 'smoothing':
            ax = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 2)
            ax2 = fig.add_subplot(2, 1, 2)
            return fig, ax, ax1, ax2

    def plot_settings(self, figure_input, min_value_x, max_value_x, min_value_y, max_value_y, title, legend):
        # Plot the road lines
        self.plot_road_lines(min_value_x, max_value_x,
                             min_value_y, max_value_y, figure_input)

        # Set the plot limits
        figure_input.set_xlim(min_value_x, max_value_x)
        figure_input.set_ylim(min_value_y, max_value_y)
        figure_input.set_aspect('equal')
        figure_input.set_title(title)
        if legend:
            figure_input.legend(loc='lower left')

    def control_action_animation(self, control_action, perturbed_control_action, plot_acc, plot_curv, index_batch, Y):
        # Create control actions with max length
        if not self.future_action:
            addition_control_action = torch.zeros_like(
                torch.tensor(Y)).to(device=self.device)
            control_action = torch.cat(
                (control_action, addition_control_action), axis=2)
            perturbed_control_action = torch.cat(
                (perturbed_control_action, addition_control_action), axis=2)

        # Find the relative clamping
        control_actions_relative_low, control_actions_relative_high = Helper.relative_clamping(
            control_action, self.epsilon_acc_relative, self.epsilon_curv_relative)

        # Detach the tensors to numpy
        control_actions_relative_low, control_actions_relative_high, control_action, perturbed_control_action = Helper.detach_tensor(
            control_actions_relative_low, control_actions_relative_high, control_action, perturbed_control_action)

        # Create control action with same lenght as interpolation
        self.tar_agent_control_actions = np.repeat(
            control_action, self.number_interpolation/control_action.shape[2], axis=2)
        self.adv_agent_control_actions = np.repeat(
            perturbed_control_action, self.number_interpolation/perturbed_control_action.shape[2], axis=2)
        self.control_actions_relative_high = np.repeat(
            control_actions_relative_high, self.number_interpolation/control_actions_relative_high.shape[-2], axis=-2)
        self.control_actions_relative_low = np.repeat(
            control_actions_relative_low, self.number_interpolation/control_actions_relative_low.shape[-2], axis=-2)

        # Create the number count same lenght as interpolation
        self.num_count = np.arange(
            0, self.tar_agent_control_actions.shape[2], 1)

        # initialize the control action in animation
        self.animation_acc_tar = plot_acc.plot(
            self.num_count, self.tar_agent_control_actions[index_batch, self.tar_agent, :, 0], color='yellow')
        self.animation_acc_adv = plot_acc.plot(
            self.num_count, self.adv_agent_control_actions[index_batch, self.tar_agent, :, 0], color='red')

        self.animation_curv_tar = plot_curv.plot(
            self.num_count, self.tar_agent_control_actions[index_batch, self.tar_agent, :, 1], color='yellow')
        self.animation_curv_adv = plot_curv.plot(
            self.num_count, self.adv_agent_control_actions[index_batch, self.tar_agent, :, 1], color='red')

    def create_interpolated_data_animation(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, index_batch):
        # Interpolate the data to smooth the animation
        for index_agent in range(X.shape[1]):
            if index_agent == 0:
                # Convert the data for the agent to be plotted
                X_1, X_new_1, Y_1, Y_new_1, Y_Pred_1, Y_Pred_iter_1_1 = [Helper.convert_data(data, index_batch, index_agent, pred) for data, pred in zip(
                    [X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1], [False, False, False, False, True, True])]

                # interpolate data tar agent
                self.interpolated_data_tar_agent = Spline.spline_data(
                    X_1, Y_1, self.number_interpolation)

                # interpolate data tar agent prediction iteration 1
                self.interpolated_data_pred_iter_1 = Spline.spline_data(
                    X_1, Y_Pred_iter_1_1, self.number_interpolation)

                # interpolate data tar agent prediction
                self.interpolated_data_adv_Pred = Spline.spline_data(
                    X_new_1, Y_Pred_1, self.number_interpolation)

                # interpolate data tar agent adversarial
                if self.future_action:
                    self.interpolated_data_adv_agent = Spline.spline_data(
                        X_new_1, Y_new_1, self.number_interpolation)

            else:
                # Convert the data for the agent to be plotted
                X_2, Y_2 = [Helper.convert_data(
                    data, index_batch, index_agent, pred) for data, pred in zip([X, Y], [False, False])]

                # interpolate data ego agent
                self.interpolated_data_ego_agent = Spline.spline_data(
                    X_2, Y_2, self.number_interpolation)

    def add_clamping_limits(self, plot_acc, plot_curv, control_actions, perturbed_control_actions, index_batch, X, Y):
        # Create control action with same lenght as interpolation
        self.control_action_animation(
            control_actions, perturbed_control_actions, plot_acc, plot_curv, index_batch, Y)

        # set relative limits
        plot_acc.plot(
            self.num_count, self.control_actions_relative_high[index_batch, self.tar_agent, :, 0], color='black', linestyle='dashed')
        plot_acc.plot(
            self.num_count, self.control_actions_relative_low[index_batch, self.tar_agent, :, 0], color='black', linestyle='dashed')

        plot_curv.plot(
            self.num_count, self.control_actions_relative_high[index_batch, self.tar_agent, :, 1], color='black', linestyle='dashed')
        plot_curv.plot(
            self.num_count, self.control_actions_relative_low[index_batch, self.tar_agent, :, 1], color='black', linestyle='dashed')

        # set absolute limits
        plot_acc.plot(self.epsilon_acc_absolute *
                      np.ones(len(self.num_count)), color='black', linestyle='solid')
        plot_acc.plot(-self.epsilon_acc_absolute *
                      np.ones(len(self.num_count)), color='black', linestyle='solid')

        plot_curv.plot(self.epsilon_curv_absolute *
                       np.ones(len(self.num_count)), color='black', linestyle='solid')
        plot_curv.plot(-self.epsilon_curv_absolute *
                       np.ones(len(self.num_count)), color='black', linestyle='solid')

        # plot settings for acceleration control actions
        plot_acc.set_xlim(self.interpolation,
                          (X.shape[2]-2)*(self.interpolation))
        plot_acc.set_ylim(-self.epsilon_acc_absolute-2,
                          self.epsilon_acc_absolute+2)
        plot_acc.set_title(r'Control action $X^{t}_{tar}$: Acceleration')
        plot_acc.set(xticks=np.arange(self.interpolation, (X.shape[2]-2)*(
            self.interpolation), self.interpolation), xticklabels=np.arange(2, (X.shape[2]-1)))

        # plot settings for curvature control actions
        plot_curv.set_xlim(self.interpolation,
                           (X.shape[2]-2)*(self.interpolation))
        plot_curv.set(xticks=np.arange(self.interpolation, (X.shape[2]-2)*(
            self.interpolation), self.interpolation), xticklabels=np.arange(2, (X.shape[2]-1)))
        plot_curv.set_ylim(-self.epsilon_curv_absolute-0.1,
                           self.epsilon_curv_absolute+0.1)
        plot_curv.set_title(r'Control action $X^{t}_{tar}$: Curvature')

    def cars_initialization(self, animation_ax):
        # initialize the cars in animation
        self.rectangles_tar_pred = self.add_rectangles(animation_ax, [
                                                       self.interpolated_data_pred_iter_1], 'm', r'Targent-agent ($\hat{Y}_{ego}$)', self.car_length, self.car_width, alpha=0.5)
        self.rectangles_tar = self.add_rectangles(animation_ax, [
                                                  self.interpolated_data_tar_agent], 'yellow', r'Target-agent ($X_{tar}$ and $Y_{tar}$)', self.car_length, self.car_width, alpha=1)
        self.rectangles_ego = self.add_rectangles(animation_ax, [
                                                  self.interpolated_data_ego_agent], 'blue', r'Ego-agent ($X_{ego}$ and $Y_{ego}$)', self.car_length, self.car_width, alpha=1)

        if self.future_action:
            self.rectangles_tar_adv_future = self.add_rectangles(animation_ax, [
                                                                 self.interpolated_data_adv_agent], 'red', r'Adversarial target agent ($\tilde{X}_{tar}$ and $\tilde{Y}_{tar}$)', self.car_length, self.car_width, alpha=1)
            self.rectangles_tar_adv = self.add_rectangles(animation_ax, [
                                                          self.interpolated_data_adv_Pred], 'red', r'Adversarial prediction ($\hat{\tilde{Y}}_{tar}$)', self.car_length, self.car_width, alpha=0.3)
        else:
            self.rectangles_tar_adv = self.add_rectangles(animation_ax, [
                                                          self.interpolated_data_adv_Pred], 'red', r'Adversarial prediction ($\tilde{X}_{tar}$ and $\hat{\tilde{Y}}_{tar}$)', self.car_length, self.car_width, alpha=1)

    def update(self, num, num_count, index_batch, tar_agent_control_actions = None, adv_agent_control_actions = None):
        # Update the location of the car
        self.update_box_position([self.interpolated_data_pred_iter_1[0, self.tar_agent, :, :]],
                                 self.rectangles_tar_pred, self.car_length, self.car_width, num)
        self.update_box_position([self.interpolated_data_tar_agent[0, self.tar_agent, :, :]],
                                 self.rectangles_tar, self.car_length, self.car_width, num)
        self.update_box_position([self.interpolated_data_adv_Pred[0, self.tar_agent, :, :]],
                                 self.rectangles_tar_adv, self.car_length, self.car_width, num)
        self.update_box_position([self.interpolated_data_ego_agent[0, self.tar_agent, :, :]],
                                 self.rectangles_ego, self.car_length, self.car_width, num)

        if self.future_action:
            self.update_box_position([self.interpolated_data_adv_agent[0, self.tar_agent, :, :]],
                                     self.rectangles_tar_adv_future, self.car_length, self.car_width, num)

        if self.control_action_graph:
            # Update the control action in animation acceleration
            self.animation_acc_tar[0].set_data(
                num_count[:num], tar_agent_control_actions[index_batch, self.tar_agent, :num, 0])
            self.animation_curv_tar[0].set_data(
                num_count[:num], tar_agent_control_actions[index_batch, self.tar_agent, :num, 1])

            # Update the control action in animation curvature
            self.animation_acc_adv[0].set_data(
                num_count[:num], adv_agent_control_actions[index_batch, self.tar_agent, :num, 0])
            self.animation_curv_adv[0].set_data(
                num_count[:num], adv_agent_control_actions[index_batch, self.tar_agent, :num, 1])

        return

    def add_arrow_animation(self, fig):
        # Adding an arrow to point from figure to figure
        if self.control_action_graph:
            arrow = FancyArrowPatch((0.87, 0.30), (0.83, 0.40),
                                    transform=fig.transFigure,
                                    mutation_scale=20,
                                    lw=1,
                                    arrowstyle="-|>",
                                    color='black')
        else:
            arrow = FancyArrowPatch((0.85, 0.40), (0.81, 0.60),
                                    transform=fig.transFigure,
                                    mutation_scale=20,
                                    lw=1,
                                    arrowstyle="-|>",
                                    color='black')

        fig.patches.extend([arrow])

    def plot_ego_and_tar_agent(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, figure_input, index_batch, index_agent, future_action, style):
        if index_agent == 0:
            # Plot target agent
            self.draw_arrow(X[index_batch, index_agent, :], Y[index_batch, index_agent, :], figure_input, 'y',
                            3, '-', 'dashed', r'Observed target agent ($X_{tar}$)', r'Future target agent ($Y_{tar}$)', 1, 1)

            # Plot the prediction on unperturbed target agent
            if style != 'perturbed' and style != 'nominal':
                Y_Pred_iter_1 = np.mean(Y_Pred_iter_1, axis=1)
                self.draw_arrow(X[index_batch, index_agent, :], Y_Pred_iter_1[index_batch, :], figure_input,
                                'm', 3, '-', '-', None, r'Prediction target agent ($\hat{Y}_{tar}$)', 0, 0.5)

            # Plot pertubed history target agent and prediction
            if style != 'unperturbed' and style != 'nominal':
                Y_Pred = np.mean(Y_Pred, axis=1)
                self.draw_arrow(X_new[index_batch, index_agent, :], Y_Pred[index_batch, :], figure_input, 'r', 3, '-', '-',
                                r'Adversarial target agent ($\tilde{X}_{tar}$)', r'Adversarial prediction ($\hat{\tilde{Y}}_{tar}$)', 1, 0.5)

            # Plot future perturbed target agent
            if future_action and style != 'unperturbed' and style != 'perturbed' and style != 'nominal':
                self.draw_arrow(X_new[index_batch, index_agent, :], Y_new[index_batch, index_agent, :], figure_input,
                                'r', 3, '-', 'dashed', None, r'Adversarial target agent ($\tilde{Y}_{tar}$)', 0, 1)

        else:
            self.draw_arrow(X[index_batch, index_agent, :], Y[index_batch, index_agent, :], figure_input, 'b',
                            3, '-', 'dashed', r'Observed ego agent ($X_{ego}$)', r'Future ego agent ($Y_{ego}$)', 1, 1)

    def plot_smoothed_data(self, X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv, index_sigma, index_batch, style, figure_input):
        if style == 'unperturbed' or style == 'adv_smoothing':
            if style == 'unperturbed':
                for k in range(X_smoothed.shape[1]):
                    self.draw_arrow(X_smoothed[index_sigma, k, index_batch, 0, :], Y_pred_smoothed[index_sigma,
                                                                                                   k, index_batch, :], figure_input, 'c', 3, '-.', '-', None, None, 0.4, 0.4)

            # calculate the expectation of the unperturbed data
            X_smoothed_mean = np.mean(X_smoothed, axis=1)

            # calculate the expectation of all predictions
            Y_pred_smoothed_mean = np.mean(Y_pred_smoothed, axis=1)
            if style == 'unperturbed':
                self.draw_arrow(X_smoothed_mean[index_sigma, index_batch, 0, :], Y_pred_smoothed_mean[index_sigma,
                                                                                                      index_batch, :], figure_input, 'c', 3, '-.', '-', None, None, 1, 1)
            else:
                self.draw_arrow(X_smoothed_mean[index_sigma, index_batch, 0, :], Y_pred_smoothed_mean[index_sigma, index_batch, :], figure_input,
                                'c', 3, '-.', '-', r'Smoothed target agent ($\bar{X}_{tar}$)', r'Smoothed prediction target agent ($\hat{\bar{Y}}_{tar}$)', 1, 0.5)

        if style == 'perturbed' or style == 'adv_smoothing':
            if style == 'perturbed':
                for k in range(X_smoothed_adv.shape[1]):
                    self.draw_arrow(X_smoothed_adv[index_sigma, k, index_batch, 0, :], Y_pred_smoothed_adv[index_sigma,
                                                                                                           k, index_batch, :], figure_input, 'g', 3, '-.', '-', None, None, 0.4, 0.4)

            # calculate the expectation of the unperturbed data
            X_smoothed_adv_mean = np.mean(X_smoothed_adv, axis=1)

            # calculate the expectation of all predictions
            Y_pred_smoothed_adv_mean = np.mean(Y_pred_smoothed_adv, axis=1)
            if style == 'perturbed':
                self.draw_arrow(X_smoothed_adv_mean[index_sigma, index_batch, 0, :], Y_pred_smoothed_adv_mean[index_sigma,
                                                                                                              index_batch, :], figure_input, 'g', 3, '-.', '-', None, None, 1, 1)
            else:
                self.draw_arrow(X_smoothed_adv_mean[index_sigma, index_batch, 0, :], Y_pred_smoothed_adv_mean[index_sigma, index_batch, :], figure_input, 'g', 3, '-.', '-',
                                r'Smoothed adversarial target agent ($\bar{\tilde{X}}_{tar}$)', r'Smoothed adversarial prediction target agent ($\hat{\bar{\tilde{Y}}}_{tar}$)', 1, 0.5)

    def draw_arrow(self, data_X, data_Y, figure_input, color, linewidth, line_style_input, line_style_output, label_input, label_output, alpha_input, alpha_output):
        # Draw the arrow path
        figure_input.plot(data_X[:, 0], data_X[:, 1], linestyle=line_style_input,
                          linewidth=linewidth, color=color, label=label_input, alpha=alpha_input)
        figure_input.plot((data_X[-1, 0], data_Y[0, 0]), (data_X[-1, 1], data_Y[0, 1]),
                          linestyle=line_style_output, linewidth=linewidth, color=color, alpha=alpha_output)
        figure_input.plot(data_Y[:-1, 0], data_Y[:-1, 1], linestyle=line_style_output,
                          linewidth=linewidth, color=color, alpha=alpha_output, label=label_output)
        figure_input.annotate('', xy=(data_Y[-1, 0], data_Y[-1, 1]), xytext=(data_Y[-2, 0], data_Y[-2, 1]),
                              size=20, arrowprops=dict(arrowstyle='-|>', linestyle=None, color=color, lw=linewidth, alpha=alpha_output))

    def add_rectangles(self, figure_input, data_list, color, label, car_length, car_width, alpha=1):
        rectangles = []
        # Add rectangles to the plot
        for _ in range(len(data_list)):
            rect = patches.Rectangle(
                (0, 0), car_length, car_width, edgecolor='none', facecolor=color, label=label, alpha=alpha)
            figure_input.add_patch(rect)
            rectangles.append(rect)
        # To only add one label per type in the legend
        handles, labels = figure_input.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure_input.legend(by_label.values(), by_label.keys())

        return rectangles

    def update_box_position(self, data, rectangle_data, car_length, car_width, num):
        # Compensate that the rectangle is drawn from the bottom left corner
        for i in range(len(rectangle_data)):
            x, y = data[i][:, 0], data[i][:, 1]
            dx = x[num + 1] - x[num]
            dy = y[num + 1] - y[num]
            angle_rad = np.arctan2(dy, dx)
            shift_x = (car_width / 2) * np.sin(angle_rad) - \
                (car_length / 2) * np.cos(angle_rad)
            shift_y = -(car_width / 2) * np.cos(angle_rad) - \
                (car_length / 2) * np.sin(angle_rad)
            rectangle_data[i].set_xy([x[num-1] + shift_x, y[num-1] + shift_y])
            angle = np.arctan2(dy, dx) * (180 / np.pi)
            rectangle_data[i].set_angle(angle)

    def plot_road_lines(self, min_value_x, max_value_x, min_value_y, max_value_y, figure_input):
        # Plot the dashed road lines
        y_dash = [0, 0]
        x_min_dash = [min_value_x, 4.5]
        x_max_dash = [-4.5, max_value_x]

        x_dash = [0, 0]
        y_min_dash = [min_value_y, 4.5]
        y_max_dash = [-4.5, max_value_y]

        figure_input.hlines(y_dash, x_min_dash, x_max_dash,
                            linestyle='dashed', colors='k', linewidth=0.75)
        figure_input.vlines(x_dash, y_min_dash, y_max_dash,
                            linestyle='dashed', colors='k', linewidth=0.75)

        y_solid = [-3.5, -3.5, 3.5, 3.5]
        x_min_solid = [min_value_x, 3.5, min_value_x, 3.5]
        x_max_solid = [-3.5, max_value_x, -3.5, max_value_x]

        x_solid = [-3.5, 3.5, 3.5, -3.5]
        y_min_solid = [min_value_y, min_value_y, 3.5, 3.5]
        y_max_solid = [-3.5, -3.5, max_value_y, max_value_y]

        figure_input.hlines(y_solid, x_min_solid, x_max_solid,
                            linestyle="solid", colors='k')
        figure_input.vlines(x_solid, y_min_solid, y_max_solid,
                            linestyle="solid", colors='k')
