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

from PIL import Image


class Adversarial_Position(perturbation_template):
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

        self.kwargs = kwargs
        self.initialize_settings()

        # Load the perturbation model
        pert_data_set = data_interface(
            kwargs['data_set_dict'], kwargs['exp_parameters'])
        pert_data_set.reset()

        # Select or load repective datasets
        pert_data_set.get_data(**kwargs['data_param'])

        # Exctract splitting method parameters
        pert_splitter_name = kwargs['splitter_dict']['Type']
        if 'repetition' in kwargs['splitter_dict'].keys():
            # Check that in splitter dict the length of repetition is only 1 (i.e., only one splitting method)
            if isinstance(kwargs['splitter_dict']['repetition'], list):

                if len(kwargs['splitter_dict']['repetition']) > 1:
                    raise ValueError("The splitting dictionary neccessary to define the trained model used " +
                                    "for the adversarial attack can only contain one singel repetition " +
                                    "(i.e, the value assigned to the key 'repetition' CANNOT be a list with a lenght larger than one).")

                kwargs['splitter_dict']['repetition'] = kwargs['splitter_dict']['repetition'][0]

            pert_splitter_rep = kwargs['splitter_dict']['repetition']

            # Check the value of the repetition key
            assert (isinstance(pert_splitter_rep, int) or
                    isinstance(pert_splitter_rep, str) or
                    isinstance(pert_splitter_rep, tuple)), "Split repetition has a wrong format."
            if isinstance(pert_splitter_rep, tuple):
                assert len(pert_splitter_rep) > 0, "Some repetition information must be given."
                for rep_part in pert_splitter_rep:
                    assert (isinstance(rep_part, int) or
                            isinstance(rep_part, str)), "Split repetition has a wrong format."
            else:
                pert_splitter_rep = (pert_splitter_rep,)
        else:
            pert_splitter_rep = (0,)
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
        self.pert_model = pert_model_class(pert_model_kwargs, pert_data_set, pert_splitter, True)
        assert hasattr(self.pert_model, 'predict_batch_tensor'), "The model does not have the method 'predict_batch_tensor'."

        # Train the model on the given training set
        self.pert_model.train()

        # After training, set model up to only take position input
        self.pert_model.input_data_type = ['x', 'y']
        self.pert_model.num_samples_path_pred = self.num_samples

        # Define the name of the perturbation method
        self.name = self.pert_model.model_file.split(os.sep)[-1][:-4]
        self.name += '---' + kwargs['attack']
        self.name += '---' + str(kwargs['gamma'])
        self.name += '---' + str(kwargs['alpha'])
        self.name += '---' + str(kwargs['num_samples_perturb'])
        self.name += '---' + str(kwargs['max_number_iterations'])
        self.name += '---' + str(kwargs['loss_function_1'])
        self.name += '---' + str(self.kwargs['GT_data'])
        if 'loss_function_2' in kwargs.keys() is not None:
            self.name += '---' + str(kwargs['loss_function_2'])
        if 'barrier_function_past' in kwargs.keys() is not None:
            self.name += '---' + str(kwargs['barrier_function_past'])
            self.name += '---' + str(kwargs['distance_threshold_past'])
            self.name += '---' + str(kwargs['log_value_past'])
        if 'barrier_function_future' in kwargs.keys() is not None:
            self.name += '---' + str(kwargs['barrier_function_future'])
            self.name += '---' + str(kwargs['distance_threshold_future'])
            self.name += '---' + str(kwargs['log_value_future'])

    def initialize_settings(self):
        # Initialize parameters
        if 'num_samples_perturb' in self.kwargs.keys():
            self.num_samples = self.kwargs['num_samples_perturb']
        else:
            self.num_samples = 20
            self.kwargs['num_samples_perturb'] = self.num_samples

        if 'max_number_iterations' in self.kwargs.keys():
            self.max_number_iterations = self.kwargs['max_number_iterations']
        else:
            self.max_number_iterations = 50
            self.kwargs['max_number_iterations'] = self.max_number_iterations
        
        # Learning decay
        if 'gamma' in self.kwargs.keys():
            self.gamma = self.kwargs['gamma']
        else:
            self.gamma = 1.0
            self.kwargs['gamma'] = self.gamma

        if 'alpha' in self.kwargs.keys():
            self.alpha = self.kwargs['alpha']
        else:
            self.alpha = 0.01
            self.kwargs['alpha'] = self.alpha



        # Car size
        self.car_length = 4.1
        self.car_width = 1.7
        self.wheelbase = 2.7

        # ADE attack select (Maximize distance): 'ADE_Y_GT_Y_Pred_Max', 'ADE_Y_Perturb_Y_Pred_Max', 'ADE_Y_Perturb_Y_GT_Max', 'ADE_Y_pred_iteration_1_and_Y_Perturb_Max', 'ADE_Y_pred_and_Y_pred_iteration_1_Max'
        # ADE attack select (Minimize distance): 'ADE_Y_GT_Y_Pred_Min', 'ADE_Y_Perturb_Y_Pred_Min', 'ADE_Y_Perturb_Y_GT_Min', 'ADE_Y_pred_iteration_1_and_Y_Perturb_Min', 'ADE_Y_pred_and_Y_pred_iteration_1_Min'
        # FDE attack select (Maximize distance): 'FDE_Y_GT_Y_Pred_Max', 'FDE_Y_Perturb_Y_Pred_Max', 'FDE_Y_Perturb_Y_GT_Max', 'FDE_Y_pred_iteration_1_and_Y_Perturb_Max', 'FDE_Y_pred_and_Y_pred_iteration_1_Max'
        # FDE attack select (Minimize distance): 'FDE_Y_GT_Y_Pred_Min', 'FDE_Y_Perturb_Y_Pred_Min', 'FDE_Y_Perturb_Y_GT_Min', 'FDE_Y_pred_iteration_1_and_Y_Perturb_Min', 'FDE_Y_pred_and_Y_pred_iteration_1_Min'
        # Collision attack select: 'Collision_Y_pred_tar_Y_GT_ego', 'Collision_Y_Perturb_tar_Y_GT_ego'
        # Other: 'Y_perturb', None
        if 'loss_function_1' in self.kwargs.keys():
            self.loss_function_1 = self.kwargs['loss_function_1']
        else:
            self.loss_function_1 = 'ADE_Y_GT_Y_Pred_Max'
            self.kwargs['loss_function_1'] = self.loss_function_1
        
        if 'loss_function_2' in self.kwargs.keys():
            self.loss_function_2 = self.kwargs['loss_function_2'] 
        else:
            self.loss_function_2 = None
            self.kwargs['loss_function_2'] = self.loss_function_2

        # For barrier function past select: 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None
        if 'barrier_function_past' in self.kwargs.keys():
            self.barrier_function_past = self.kwargs['barrier_function_past']
        else:
            self.barrier_function_past = None
            self.kwargs['barrier_function_past'] = self.barrier_function_past

        if self.barrier_function_past is not None:
            barrier_test = self.barrier_function_past.split('_V')[0]
            assert barrier_test in ['Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific'], "The barrier function can only be 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None."
            if 'distance_threshold_past' in self.kwargs.keys():
                self.distance_threshold_past = self.kwargs['distance_threshold_past']
            else:
                self.distance_threshold_past = 1
                self.kwargs['distance_threshold_past'] = self.distance_threshold_past
            
            if 'log_value_past' in self.kwargs.keys():
                self.log_value_past = self.kwargs['log_value_past']
            else:
                self.log_value_past = 1.5
                self.kwargs['log_value_past'] = self.log_value_past
        else:
            self.distance_threshold_past = 1
            self.log_value_past = 1.5

        if 'barrier_function_future' in self.kwargs.keys():
            self.barrier_function_future = self.kwargs['barrier_function_future']
        else:
            self.barrier_function_future = None
            self.kwargs['barrier_function_future'] = self.barrier_function_future
        
        if self.barrier_function_future is not None:
            barrier_test = self.barrier_function_future.split('_V')[0]
            assert barrier_test in ['Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific'], "The barrier function can only be 'Time_specific', 'Trajectory_specific', 'Time_Trajectory_specific' or None."
            if 'distance_threshold_future' in self.kwargs.keys():
                self.distance_threshold_future = self.kwargs['distance_threshold_future']
            else:
                self.distance_threshold_future = 1
                self.kwargs['distance_threshold_future'] = self.distance_threshold_future
            
            if 'log_value_future' in self.kwargs.keys():
                self.log_value_future = self.kwargs['log_value_future']
            else:
                self.log_value_future = 1.5
                self.kwargs['log_value_future'] = self.log_value_future
        else:
            self.distance_threshold_future = 1
            self.log_value_future = 1.5

        # store which data
        if 'GT_data' in self.kwargs.keys():
            self.GT_data = self.kwargs['GT_data']
        else:
            self.GT_data = 'no'
            self.kwargs['GT_data'] = self.GT_data

        assert self.GT_data in ['no', 'one', 'full'], "The GT data can only be 'no' (use unperturbed GT), 'one' (use first iteration prediction) or 'full' (use last perturbed iteration)."
        

        # Randomized smoothing
        self.smoothing = False
        self.num_samples_used_smoothing = 15  
        self.sigma = [0.05, 0.1]
        self.plot_smoothing = False

        # Plot the loss over the iterations
        self.plot_loss = True

        # Left turns settings!!!
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

    def perturb_batch(self, X, Y, T, S, C, img, img_m_per_px, graph, Agent_names): 
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
        X, Y, T, S, C, img, img_m_per_px = self._prepare_data(X, Y, T, S, C, img, img_m_per_px, Agent_names)
        Pred_agents = np.zeros((X.shape[0], X.shape[1]), dtype=bool)
        Pred_agents[:, 0] = True

        # Prepare data for adversarial attack (tensor/image prediction model)
        X, Y, positions_perturb, Y_Pred_iter_1, data_barrier = self._prepare_data_attack(X, Y)
        if img is not None:
            img = torch.from_numpy(img).float().to(self.pert_model.device)
            img_m_per_px = torch.from_numpy(img_m_per_px).float().to(self.pert_model.device)

        # Create a tensor for the perturbation
        perturbation_storage = torch.zeros_like(positions_perturb)

        # Store the loss for plot
        loss_store = []

        # learning rate
        alpha = self.alpha * torch.ones_like(positions_perturb)

        # Start the optimization of the adversarial attack
        for i in range(self.max_number_iterations):
            # Create a tensor for the perturbation
            perturbation = perturbation_storage.detach().clone()
            perturbation.requires_grad = True

            # Reset gradients
            perturbation.grad = None

            # Split the adversarial position back to X and Y
            X_new, Y_new = Helper.return_data(
                positions_perturb + perturbation, X, Y, self.future_action_included)

            # Forward pass through the model
            Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, S=S, C=C, 
                                                          img=img, img_m_per_px=img_m_per_px, graph = graph,
                                                          Pred_agents = Pred_agents, num_steps = self.num_steps_predict)
            # Only use actually predicted target agent
            Y_Pred = Y_Pred[:,0]

            if i == 0:
                # check conversion
                # Helper.check_conversion(adv_position, positions_perturb)

                # Store the first prediction
                Y_Pred_iter_1 = Y_Pred.detach()

            losses = self._loss_module(
                X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier, i)

            # # Store the loss for plot
            print(losses)

            # Calulate gradients
            losses.sum().backward()
            grad = perturbation.grad

            inner_loop_count = 0

            # copy learning rates
            # alpha_iter = alpha.clone()

            # Update Control inputs
            while True:
                inner_loop_count += 1

                # Update Control inputs
                with torch.no_grad():
                    perturbation_new = perturbation.clone()
                    perturbation_new.subtract_(grad * alpha)

                    # set perturbations of ego agent to zero
                    perturbation_new[:, 1:] = 0.0

                # Split the adversarial position back to X and Y
                X_new, Y_new = Helper.return_data(
                    positions_perturb + perturbation_new, X, Y, self.future_action_included)
                
                # Forward pass through the model
                Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, S=S, C=C, 
                                                              img=img, img_m_per_px=img_m_per_px, graph = graph,
                                                              Pred_agents = Pred_agents, num_steps = self.num_steps_predict)
                # Only use actually predicted target agent
                Y_Pred = Y_Pred[:,0]

                losses = self._loss_module(
                    X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier, i)
                
                print(losses)

                # Check for NaN values in losses
                invalid_mask = torch.isnan(losses) | torch.isinf(losses)
                if invalid_mask.any():
                    # check if agent crashes replace tensor with zero tensor
                    if inner_loop_count >= 20:
                        # perturbation_new[invalid_mask] = torch.zeros_like(perturbation_new[invalid_mask])
                        perturbation_new[invalid_mask] = perturbation_storage[invalid_mask].clone()
                        perturbation_storage = perturbation_new.detach().clone()
                        break
                    # Half the learning rate only for samples with NaN losses
                    alpha[invalid_mask] *= 0.5
                    continue  # Skip this iteration and try again with reduced learning rate for NaN samples
                else:
                    perturbation_storage = perturbation_new.detach().clone()
                    break

            # Store the loss for plot
            loss_store.append(losses.detach().cpu().numpy())

            # Update the step size
            alpha *= self.gamma

        # Split the adversarial position back to X and Y
        X_new, Y_new = Helper.return_data(
            positions_perturb + perturbation, X, Y, self.future_action_included)

        # Forward pass through the model
        Y_Pred = self.pert_model.predict_batch_tensor(X=X_new, T=T, S=S, C=C,
                                                      img=img, img_m_per_px=img_m_per_px, graph = graph,
                                                      Pred_agents = Pred_agents, num_steps = self.num_steps_predict)
        # Only use actually predicted target agent
        Y_Pred = Y_Pred[:,0]

        # Detach the tensor and convert to numpy
        X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier = Helper.detach_tensor(
            X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier)

        # Return Y to old shape
        Y_new = Helper.return_to_old_shape(Y_new, self.Y_shape)
        self.copy_Y = Helper.return_to_old_shape(self.copy_Y, self.Y_shape)
        Y_Pred_iter_1_new = Helper.return_to_old_shape_pred_1(Y_Pred_iter_1, Y, self.Y_shape, self.ego_agent_index)

        # Flip dimensions back
        X_new_pert, Y_new_pert, Y_Pred_iter_1_new = Helper.flip_dimensions_2(
            X_new, Y_new, Y_Pred_iter_1_new, self.agent_order)
        
        if self.GT_data == 'one':
            # Use the first perturbed iteration
            return X_new_pert, Y_Pred_iter_1_new
        elif self.GT_data == 'no':
            # Use the unperturbed GT
            return X_new_pert, self.copy_Y
        else:
            # Use the last perturbed iteration
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

    def _loss_module(self, X, X_new, Y, Y_new, Y_Pred, Y_Pred_iter_1, data_barrier, iteration):
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
        iteration (int): The current iteration of the adversarial attack.

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
                                     barrier_data=data_barrier,
                                     iteration=iteration
                                     )

        return losses

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


    def _prepare_data(self, X, Y, T, S, C, I1, I2, agent):
        """
        Prepares data for further processing by removing NaN values,
        flipping dimensions of the agent data, and storing relevant
        attributes.

        Parameters:
        X (array-like): The ground truth observed postition tensor with array shape (batch size, number agents, number time steps observed, coordinates (x,y))
        Y (array-like): The ground truth future postition tensor with array shape (batch size, number agents, number time steps future, coordinates (x,y))
        T (int): Type of agent observed.
        agent (object): It includes strings with the names of the agents.

        Returns:
        X (array-like): Processed observed feature matrix.
        Y (array-like): Processed future feature matrix.
        """

        # Remove nan from input and remember old shape
        self.Y_shape = Y.shape
        Y = Helper.remove_nan_values(data=Y)

        # Copy the original data
        self.copy_Y = Y.copy()

        # Flip dimensions agents
        self.agent_order, self.tar_agent_index, self.ego_agent_index = Helper.flip_dimensions_index(agent)

        X  = X[:, self.agent_order]
        Y  = Y[:, self.agent_order]
        T  = T[:, self.agent_order]
        S  = S[:, self.agent_order]
        if C is not None:
            C  = C[:, self.agent_order]
        if I1 is not None:
            I1 = I1[:, self.agent_order]
            I2 = I2[:, self.agent_order]

        return X, Y, T, S, C, I1, I2

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

        self.batch_size = 5

    def get_constraints(self):
        '''
        This function returns the constraints for the data to be perturbed.

        Returns
        -------
        None
        '''
        return None


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
