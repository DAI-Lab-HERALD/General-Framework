from perturbation_template import perturbation_template
import pandas as pd
import os
import numpy as np
import importlib
from Data_sets.data_interface import data_interface
import torch

from Adversarial_classes.helper import Helper
from Adversarial_classes.loss import Loss
from Adversarial_classes.plot import Plot
from Adversarial_classes.smoothing import Smoothing
from Adversarial_classes.search import Search

from PIL import Image


class Adversarial_Search(perturbation_template):
    def check_and_extract_kwargs(self, kwargs):
        '''
        This function checks if the input dictionary is complete and extracts the required values.

        Parameters
        ----------
        kwargs : dict
            A dictionary with the required keys and values.

        Returns
        -------
        None.

        '''
        assert 'data_set_dict' in kwargs.keys(
        ), "Adverserial model dataset is missing (required key: 'data_set_dict')."
        assert 'data_param' in kwargs.keys(
        ), "Adverserial model data param is missing (required key: 'data_param')."
        assert 'splitter_dict' in kwargs.keys(
        ), "Adverserial model splitter is missing (required key: 'splitter_dict')."
        assert 'model_dict' in kwargs.keys(
        ), "Adverserial model is missing (required key: 'model_dict')."
        assert 'exp_parameters' in kwargs.keys(
        ), "Adverserial model experiment parameters are missing (required key: 'exp_parameters')."

        assert kwargs['exp_parameters'][6] == 'predefined', "Perturbed datasets can only be used if the agents' roles are predefined."

        self.kwargs = kwargs
        self.initialize_settings()

        # Check that in splitter dict the length of repetition is only 1 (i.e., only one splitting method)
        if isinstance(kwargs['splitter_dict']['repetition'], list):

            if len(kwargs['splitter_dict']['repetition']) > 1:
                raise ValueError("The splitting dictionary neccessary to define the trained model used " +
                                 "for the adversarial attack can only contain one singel repetition " +
                                 "(i.e, the value assigned to the key 'repetition' CANNOT be a list with a lenght larger than one).")

            kwargs['splitter_dict']['repetition'] = kwargs['splitter_dict']['repetition'][0]

        # Load the perturbation model
        pert_data_set = data_interface(
            kwargs['data_set_dict'], kwargs['exp_parameters'])
        pert_data_set.reset()

        # Select or load repective datasets
        pert_data_set.get_data(**kwargs['data_param'])

        # Exctract splitting method parameters
        pert_splitter_name = kwargs['splitter_dict']['Type']
        if 'repetition' in kwargs['splitter_dict'].keys():
            pert_splitter_rep = [kwargs['splitter_dict']['repetition']]
        else:
            pert_splitter_rep = [(0,)]
        if 'test_part' in kwargs['splitter_dict'].keys():
            pert_splitter_tp = kwargs['splitter_dict']['test_part']
        else:
            pert_splitter_tp = 0.2

        if 'train_pert' in kwargs['splitter_dict'].keys():
            pert_splitter_train_pert = kwargs['splitter_dict']['train_pert']
        else:
            pert_splitter_train_pert = False
        if 'test_pert' in kwargs['splitter_dict'].keys():
            pert_splitter_test_pert = kwargs['splitter_dict']['test_pert']
        else:
            pert_splitter_test_pert = False

        pert_splitter_module = importlib.import_module(pert_splitter_name)
        pert_splitter_class = getattr(pert_splitter_module, pert_splitter_name)

        # Initialize and apply Splitting method
        pert_splitter = pert_splitter_class(
            pert_data_set, pert_splitter_tp, pert_splitter_rep, pert_splitter_train_pert, pert_splitter_test_pert)
        pert_splitter.split_data()

        # Extract per model dict
        if isinstance(kwargs['model_dict'], str):
            pert_model_name = kwargs['model_dict']
            pert_model_kwargs = {}
        elif isinstance(kwargs['model_dict'], dict):
            assert 'model' in kwargs['model_dict'].keys(
            ), "No model name is provided."
            assert isinstance(kwargs['model_dict']['model'],
                              str), "A model is set as a string."
            pert_model_name = kwargs['model_dict']['model']
            if not 'kwargs' in kwargs['model_dict'].keys():
                pert_model_kwargs = {}
            else:
                assert isinstance(
                    kwargs['model_dict']['kwargs'], dict), "The kwargs value must be a dictionary."
                pert_model_kwargs = kwargs['model_dict']['kwargs']
        else:
            raise TypeError("The provided model must be string or dictionary")

        # Get model class
        pert_model_module = importlib.import_module(pert_model_name)
        pert_model_class = getattr(pert_model_module, pert_model_name)

        # Initialize the model
        self.pert_model = pert_model_class(
            pert_model_kwargs, pert_data_set, pert_splitter, True)

        # TODO: Check if self.pert_model can call the function that is needed later in perturb_batch (i.e., self.pert_model.adv_generation())

        # Train the model on the given training set
        self.pert_model.train()

        # Define the name of the perturbation method
        self.name = self.pert_model.model_file.split(os.sep)[-1][:-4]

    def initialize_settings(self):
        # Initialize parameters
        self.num_samples = 20  
        self.max_number_iterations = 30

        # Learning decay
        self.gamma = 1
        self.alpha = 0.2

        # Car size
        self.car_length = 4.1
        self.car_width = 1.7
        self.wheelbase = 2.7

        # ADE attack select (Maximize distance): 'ADE_Y_GT_Y_Pred_Max', 'ADE_Y_Perturb_Y_Pred_Max', 'ADE_Y_Perturb_Y_GT_Max', 'ADE_Y_pred_iteration_1_and_Y_Perturb_Max', 'ADE_Y_pred_and_Y_pred_iteration_1_Max'
        # ADE attack select (Minimize distance): 'ADE_Y_GT_Y_Pred_Min', 'ADE_Y_Perturb_Y_Pred_Min', 'ADE_Y_Perturb_Y_GT_Min', 'ADE_Y_pred_iteration_1_and_Y_Perturb_Min', 'ADE_Y_pred_and_Y_pred_iteration_1_Min'
        # FDE attack select (Maximize distance): 'FDE_Y_GT_Y_Pred_Max', 'FDE_Y_Perturb_Y_Pred_Max', 'FDE_Y_Perturb_Y_GT_Max', 'FDE_Y_pred_iteration_1_and_Y_Perturb_Max', 'FDE_Y_pred_and_Y_pred_iteration_1_Max'
        # FDE attack select (Minimize distance): 'FDE_Y_GT_Y_Pred_Min', 'FDE_Y_Perturb_Y_Pred_Min', 'FDE_Y_Perturb_Y_GT_Min', 'FDE_Y_pred_iteration_1_and_Y_Perturb_Min', 'FDE_Y_pred_and_Y_pred_iteration_1_Min'
        # Collision attack select: 'Collision_Y_pred_tar_Y_GT_ego', 'Collision_Y_Perturb_tar_Y_GT_ego'
        self.loss_function_1 = 'ADE_Y_pred_and_Y_pred_iteration_1_Min'
        self.loss_function_2 = 'Collision_Y_Perturb_tar_Y_GT_ego' # If not used set to None

        # For barrier function past select: 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None
        self.barrier_function_past = 'Trajectory_specific'
        self.barrier_function_future = None

        # Barrier function parameters
        self.distance_threshold_past = 1
        self.distance_threshold_future = 1
        self.log_value_past = 2.5
        self.log_value_future = 2.5

        # Randomized smoothing
        self.smoothing = False
        self.num_samples_used_smoothing = 15  
        self.sigma = [0.05, 0.1]
        self.plot_smoothing = False

        # Plot the loss over the iterations
        self.plot_loss = True

        # Image neural network
        self.image_neural_network = False

        # Left turn settings!!!
        # Plot input data 
        self.plot_input = False

        # Plot the adversarial scene
        self.static_adv_scene = True
        self.animated_adv_scene = False

        # Spline settings adversarial scene
        self.total_spline_values = 100

        # Time step
        self.dt = self.kwargs['data_param']['dt']

        # Do a assertion check on settings
        self._assertion_check()

    def perturb_batch(self, X, Y, T, agent, Domain):
        '''
        This function takes a batch of data and generates perturbations.


        Parameters
        ----------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values.
        Y : np.ndarray, optional
            This is the future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        Agent_names : np.ndarray
            This is a :math:`N_{agents}` long numpy array. It includes strings with the names of the agents.

        Returns
        -------
        X_pert : np.ndarray
            This is the past perturbed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values.
        Y_pert : np.ndarray, optional
            This is the future perturbed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or at some timesteps partially not observed, then this can include np.nan values. 
        '''

        # Prepare the data (ordering/spline/edge_cases)
        X, Y = self._prepare_data(X, Y, T, agent, Domain)

        # Prepare data for adversarial attack (tensor/image prediction model)
        X, Y, positions_perturb, Y_Pred_iter_1, data_barrier = self._prepare_data_attack(
            X, Y)

        # Create a tensor for the perturbation
        perturbation = torch.zeros_like(
            positions_perturb).to(self.pert_model.device)
        perturbation.requires_grad = True

        # Store the loss for plot
        loss_store = []

        # Start the optimization of the adversarial attack
        for i in range(self.max_number_iterations):
            # Reset gradients
            perturbation.grad = None

            # Process the perturbations
            perturbation_new = Search.hard_constraint(positions_perturb=positions_perturb, perturbation_tensor=perturbation, ego_agent_index=self.ego_agent_index,
                                                      tar_agent_index=self.tar_agent_index, hard_bound=self.distance_threshold_past, physical_bounds=self.physical_bounds, dt=self.dt, device=self.pert_model.device)

            # Split the adversarial position back to X and Y
            X_new, Y_new = Helper.return_data(
                positions_perturb + perturbation_new, X, Y, self.future_action_included)

            # Forward pass through the model
            Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, Domain=Domain, img=self.img, img_m_per_px=self.img_m_per_px,
                                                          num_steps=self.num_steps_predict, num_samples=self.num_samples)

            if i == 0:
                # check conversion
                # Helper.check_conversion(adv_position, positions_perturb)

                # Store the first prediction
                Y_Pred_iter_1 = Y_Pred.detach()

            losses = self._loss_module(
                X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier)

            # Store the loss for plot
            loss_store.append(losses.detach().cpu().numpy())
            print(losses)

            # Calulate gradients
            losses.sum().backward()
            grad = perturbation.grad

            # Update Control inputs
            with torch.no_grad():
                perturbation.subtract_(grad, alpha=self.alpha)

                # set perturbations of ego agent to zero
                perturbation[:, 1:] = 0.0

            # Update the step size
            self.alpha *= self.gamma

        # Split the adversarial position back to X and Y
        X_new, Y_new = Helper.return_data(
            positions_perturb + perturbation_new, X, Y, self.future_action_included)

        # Forward pass through the model
        Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, Domain=Domain, img=self.img, img_m_per_px=self.img_m_per_px,
                                                      num_steps=self.num_steps_predict, num_samples=self.num_samples)

        # Gaussian smoothing module
        self.X_smoothed, self.X_smoothed_adv, self.Y_pred_smoothed, self.Y_pred_smoothed_adv = self._smoothing_module(
            X, Y, positions_perturb + perturbation)

        # Detach the tensor and convert to numpy
        X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier = Helper.detach_tensor(
            X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier)

        # Plot the data
        self._ploting_module(X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1,
                             data_barrier, loss_store)

        # Return Y to old shape
        Y_new = Helper.return_to_old_shape(Y_new, self.Y_shape)

        # Flip dimensions back
        X_new_pert, Y_new_pert = Helper.flip_dimensions_2(
            X_new, Y_new, self.agent_order)

        return X_new_pert, Y_new_pert

    def _ploting_module(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier, loss_store):
        """
        Handles the plotting for the left-turns dataset.

        Parameters:
        X (array-like): The ground truth observed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        X_new (array-like): The modified observed position tensor after applying perturbations.
        Y (array-like): The ground truth future position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        Y_new (array-like): The modified future position tensor after applying perturbations.
        Y_Pred (array-like): The predicted future position tensor.
        Y_Pred_iter_1 (array-like): The initial prediction of future positions.
        data_barrier (array-like): Concatenated tensor of observed and future positions for barrier function.
        loss_store (array-like): Storage of loss values over iterations.

        Returns:
        None
        """
        # Initialize the plot class
        plot = Plot(self)

        # Plot the input/barrier data if required
        if self.plot_input:
            plot.plot_static_data(X=X, X_new=None, Y=Y, Y_new=None, Y_Pred=None,
                                  Y_Pred_iter_1=None, data_barrier=data_barrier, plot_input=self.plot_input)

        # Plot the loss over the iterations
        if self.plot_loss:
            plot.plot_loss_over_iterations(loss_store)

        # Plot the static adversarial scene
        if self.static_adv_scene:
            plot.plot_static_data(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred,
                                  Y_Pred_iter_1=Y_Pred_iter_1, data_barrier=data_barrier, plot_input=False)

        # Plot the animated adversarial scene
        if self.animated_adv_scene:
            plot.plot_animated_adv_scene(
                X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1)

        # Plot the randomized smoothing
        if self.plot_smoothing:
            plot.plot_smoothing(X=X, X_new=X_new, Y=Y, Y_new=Y_new, Y_Pred=Y_Pred, Y_Pred_iter_1=Y_Pred_iter_1,
                                X_smoothed=self.X_smoothed, X_smoothed_adv=self.X_smoothed_adv, Y_pred_smoothed=self.Y_pred_smoothed, Y_pred_smoothed_adv=self.Y_pred_smoothed_adv)

    def _loss_module(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier):
        """
        Calculates the loss for the given input data, predictions, and barrier data.

        Parameters:
        X (array-like): The ground truth observed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        X_new (array-like): The modified observed position tensor after applying perturbations.
        Y (array-like): The ground truth future position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        Y_new (array-like): The modified future position tensor after applying perturbations.
        Y_Pred (array-like): The predicted future position tensor.
        Y_Pred_iter_1 (array-like): The initial prediction of future positions.
        data_barrier (array-like): Concatenated tensor of observed and future positions for barrier function.

        Returns:
        losses (array-like): Calculated loss values based on the input data and predictions.
        """
        # calculate the loss
        losses = Loss.calculate_loss(self,
                                     X=X,
                                     X_new=X_new,
                                     Y=Y,
                                     Y_new=Y_new,
                                     Y_Pred=Y_Pred,
                                     Y_Pred_iter_1=Y_Pred_iter_1,
                                     barrier_data=data_barrier
                                     )

        return losses

    def _smoothing_module(self, X, Y, adv_position):
        """
        Applies a smoothing module to the input data to perform randomized smoothing on control actions.

        Parameters:
        X (array-like): The ground truth observed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        Y (array-like): The ground truth future position tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y)).
        adv_position (array-like): The adversarial positions for the agents.

        Returns:
        X_smoothed (array-like): Smoothed observed position tensor.
        X_smoothed_adv (array-like): Smoothed adversarial observed position tensor.
        Y_pred_smoothed (array-like): Smoothed future position predictions.
        Y_pred_smoothed_adv (array-like): Smoothed adversarial future position predictions.
        """
        # initialize smoothing
        smoothing = Smoothing(self,
                              adv_position=adv_position,
                              X=X,
                              Y=Y
                              )

        # Randomized smoothing
        X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv = smoothing.randomized_smoothing()

        return X_smoothed, X_smoothed_adv, Y_pred_smoothed, Y_pred_smoothed_adv

    def _assertion_check(self):
        """
        Performs assertion checks to validate the consistency of certain attributes.

        This method checks:
        - If the settings for `smoothing` and `plot_smoothing` are valid and ordered correctly.

        Returns:
        None
        """
        # check if the size of both sigmas are the same
        Helper.validate_settings_order(self.smoothing, self.plot_smoothing)

        Helper.validate_adversarial_loss(self.loss_function_1)

    def _load_images(self, X, Domain):
        """
        Loads images required for neural netwrok on the given observed positions and domain information.

        Parameters:
        X (array-like): The ground truth observed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        Domain (DataFrame): A DataFrame containing domain-specific information related to the agents.

        Returns:
        img (array-like): Loaded images in the required format and dimensions.
        img_m_per_px (array-like): Meter-per-pixel values for the images.
        """
        Img_needed = np.zeros(X.shape[:2], bool)
        Img_needed[:, 0] = True

        if self.data.includes_images():
            if self.pert_model.grayscale:
                channels = 1
            else:
                channels = 3
            img = np.zeros((*Img_needed.shape, self.pert_model.target_height,
                            self.pert_model.target_width, channels), np.uint8)
            img_m_per_px = np.ones(Img_needed.shape, np.float32) * np.nan

            centre = X[Img_needed, -1, :]
            x_rel = centre - X[Img_needed, -2, :]
            rot = np.angle(x_rel[:, 0] + 1j * x_rel[:, 1])
            domain_needed = Domain.iloc[np.where(Img_needed)[0]]

            img[Img_needed] = self.data.return_batch_images(domain_needed, centre, rot,
                                                            target_height=self.pert_model.target_height,
                                                            target_width=self.pert_model.target_width,
                                                            grayscale=self.pert_model.grayscale,
                                                            Imgs_rot=img[Img_needed],
                                                            Imgs_index=np.arange(Img_needed.sum()))

            img_m_per_px[Img_needed] = self.data.Images.Target_MeterPerPx.loc[domain_needed.image_id]
        else:
            img = None
            img_m_per_px = None

        return img, img_m_per_px

    def _prepare_data(self, X, Y, T, agent, Domain):
        """
        Prepares data for further processing by removing NaN values,
        flipping dimensions of the agent data, and storing relevant
        attributes.

        Parameters:
        X (array-like): The ground truth observed postition tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y))
        Y (array-like): The ground truth future postition tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y))
        T (int): Type of agent observed.
        agent (object): It includes strings with the names of the agents.
        Domain (object): A domain object specifying the context of the agents

        Returns:
        X (array-like): Processed observed feature matrix.
        Y (array-like): Processed future feature matrix.
        """

        # Remove nan from input and remember old shape
        self.Y_shape = Y.shape
        Y = Helper.remove_nan_values(data=Y)

        # Flip dimensions agents
        X, Y, self.agent_order, self.tar_agent_index, self.ego_agent_index = Helper.flip_dimensions(
            X=X, Y=Y, agent=agent)

        self.physical_bounds = Helper.get_dimensions_physical_bounds(
            constraints=self.contstraints, agent_order=self.agent_order)

        self.T = T
        self.Domain = Domain

        return X, Y

    def _prepare_data_attack(self, X, Y):
        """
        Prepares data for an adversarial attack by converting inputs to tensors,
        creating data to perturb, and initializing necessary attributes for
        further processing.

        Parameters:
        X (array-like): The ground truth observed position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).
        Y (array-like): The ground truth future position tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y)).

        Returns:
        X (tensor): Converted observed feature tensor.
        Y (tensor): Converted future feature tensor.
        positions_perturb (tensor): Tensor containing positions to perturb.
        Y_Pred_iter_1 (tensor): Storage for the adversarial prediction on nominal setting.
        data_barrier (tensor): Concatenated tensor of observed and future positions for barrier function.
        """
        # Convert to tensor
        X, Y = Helper.convert_to_tensor(self.pert_model.device, X, Y)

        # Check if future action is required
        positions_perturb, self.future_action_included = Helper.create_data_to_perturb(
            X=X, Y=Y, loss_function_1=self.loss_function_1, loss_function_2=self.loss_function_2)

        # data for barrier function
        data_barrier = torch.cat((X, Y), dim=2)

        self.mask_data = Helper.compute_mask_values_tensor(positions_perturb)

        # Load images for adversarial attack (change when using image)
        # img, img_m_per_px = self._load_images(X,Domain)
        self.img, self.img_m_per_px = None, None

        # Show image
        if self.image_neural_network:
            plot_img = Image.fromarray(self.img[0, 0, :], 'RGB')
            plot_img.show()

        # Create storage for the adversarial prediction on nominal setting
        Y_Pred_iter_1 = torch.zeros(
            (Y.shape[0], self.num_samples, Y.shape[2], Y.shape[3]))

        # number of steps to predict
        self.num_steps_predict = Y.shape[2]

        return X, Y, positions_perturb, Y_Pred_iter_1, data_barrier

    def set_batch_size(self):
        '''
        This function sets the batch size for the perturbation method.

        It must add a attribute self.batch_size to the class.

        Returns
        -------
        None.

        '''

        self.batch_size = 1
    
    def get_constraints(self):
        '''
        This function returns the constraints for the data to be perturbed.

        Returns
        -------
        def
            A function used to calculate constraints.

        '''
        return Search.get_physical_constraints

    def requirerments(self):
        '''
        This function returns the requirements for the data to be perturbed.

        It returns a dictionary, that may contain the following keys:

        n_I_max : int (optional)
            The number of maximum input timesteps.
        n_I_min : int (optional)
            The number of minimum input timesteps.

        n_O_max : int (optional)
            The number of maximum output timesteps.
        n_O_min : int (optional)
            The number of minimum output timesteps.

        dt : float (optional)
            The time step of the data.


        Returns
        -------
        dict
            A dictionary with the required keys and values.

        '''

        # TODO: Implement this function, use self.pert_model to get the requirements of the model.

        return {}
