import pandas as pd
import numpy as np
import importlib
from Data_sets.data_interface import data_interface
import random

from Adversarial_classes.search import Search
from Adversarial_classes.helper import Helper

class perturbation_template():
    def __init__(self, kwargs):
        self.check_and_extract_kwargs(kwargs)

        self.kwargs = kwargs   

        # Get the attack type
        self.attack = self.__class__.__name__

        # Ensure that the name identifying the perturbation method is defined
        assert hasattr(self, 'name'), "The name identifying the perturbation method is not defined."



    def perturb(self, data):
        self.data = data

        Input_path  = data.Input_path
        Input_T     = data.Input_T

        Output_path = data.Output_path
        Output_T    = data.Output_T

        Type        = data.Type
        Domain      = data.Domain

        # Get input data type
        self.input_path_data_type = self.data.path_data_info()

        # Check the requirements for the data, create error if not fulfilled
        requirements = self.requirerments()

        # Transform agent types to numpy array
        T = Type.to_numpy().astype(str)
        T[T == 'nan'] = '0'

        # Get length of future data
        N_O = np.zeros(len(Output_T), int)
        for i_sample in range(Output_T.shape[0]):
            if 'dt' in requirements.keys():
                assert np.all(np.abs(np.diff(Input_T[i_sample]) - requirements['dt']) < 1e-3), "The input time steps are not constant."
                assert np.all(np.abs(np.diff(Output_T[i_sample]) - requirements['dt']) < 1e-3), "The output time steps are not constant."

            N_O[i_sample] = len(Output_T[i_sample])

        # Transform the data into numpy arrays
        Agents = np.array(Input_path.columns)

        X = np.ones(list(Input_path.shape) + [data.num_timesteps_in_real, 2], dtype = np.float32) * np.nan
        Y = np.ones(list(Output_path.shape) + [N_O.max(), 2], dtype = np.float32) * np.nan
        
        # Extract data from original number a samples
        for i_sample, i_index in enumerate(Input_path.index):
            for i_agent, agent in enumerate(Agents):
                # Check if the named agent actually exists
                if not isinstance(Input_path.loc[i_index, agent], float):    
                    n_time = N_O[i_sample]

                    if 'n_I_max' in requirements.keys():
                        assert len(Input_path.loc[i_index, agent]) <= requirements['n_I_max'], "The number of input timesteps is too large."
                    if 'n_I_min' in requirements.keys():
                        assert len(Input_path.loc[i_index, agent]) >= requirements['n_I_min'], "The number of input timesteps is too small."
                    if 'n_O_max' in requirements.keys():
                        assert n_time <= requirements['n_O_max'], "The number of output timesteps is too large."
                    if 'n_O_min' in requirements.keys():
                        assert n_time >= requirements['n_O_min'], "The number of output timesteps is too small."

                    X[i_sample, i_agent]          = Input_path.loc[i_index, agent].astype(np.float32)
                    Y[i_sample, i_agent, :n_time] = Output_path.loc[i_index, agent][:n_time].astype(np.float32)

        # Get the batch size
        self.set_batch_size()
        assert hasattr(self, 'batch_size'), "The batch size is not defined."
        assert isinstance(self.batch_size, int), "The given batch size must be an integer."

        sorted_indices = np.argsort(-N_O)

        # Reorder both X_pert and Y_pert arrays based on sorted indices
        X_sort = X[sorted_indices]
        Y_sort = Y[sorted_indices]
        T_sort = T[sorted_indices]
        Domain_sort = Domain.iloc[sorted_indices]

        # Run perturbation
        X_pert_sort = np.copy(X_sort)
        Y_pert_sort = np.copy(Y_sort)

        dt = self.data.dt

        # Get constraints of datasets
        contstraints = self.get_constraints()

        if contstraints is not None:
            self.contstraints = contstraints(X_pert_sort, Y_pert_sort, dt)

        # Go through the data 
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        for i_batch in range(num_batches):
            i_start = i_batch * self.batch_size
            i_end = min((i_batch + 1) * self.batch_size, X.shape[0])

            samples = np.arange(i_start, i_end)
            
            X_pert_sort[samples], Y_pert_sort[samples] = self.perturb_batch(X_sort[samples], Y_sort[samples], T_sort[samples], Agents, Domain_sort.iloc[samples])


        sort_indices_inverse = np.argsort(sorted_indices)
        X_pert = X_pert_sort[sort_indices_inverse]
        Y_pert = Y_pert_sort[sort_indices_inverse]

        # Add unperturberd input and output columns to Domain
        Domain['Unperturbed_input'] = None
        Domain['Unperturbed_output'] = None

        # Write the unperturbed data into new columns in domain and overwrite Input_path and Output_path with the perturbed data
        for i_sample, i_index in enumerate(Input_path.index):
            # Save the unperturbed data
            input_i = Input_path.loc[i_index].copy()
            output_i = Output_path.loc[i_index].copy()

            Domain.Unperturbed_input.loc[i_index] = [input_i]
            Domain.Unperturbed_output.loc[i_index] = [output_i]

            # Overwrite data with
            for i_agent, agent in enumerate(Agents):
                if not isinstance(Input_path.loc[i_index, agent], float):
                    Input_path.loc[i_index, agent]  = X_pert[i_sample, i_agent]
                    Output_path.loc[i_index, agent] = Y_pert[i_sample, i_agent]
        
        # Save changes to data object
        data.Input_path = Input_path
        data.Output_path = Output_path
        data.Domain = Domain
        return data
        
    
    def perturb_batch(self, X, Y, T, Agent_names, Domain):
        '''
        This function takes a batch of data and generates perturbations.


        Parameters
        ----------
        X : np.ndarray
            This is the past observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float values. 
            Here, :math:`N_{data}` are the number of information available. This information can be found in 
            *self.input_path_data_type*, which is a list of strings with the length of *N_{data}*. It will always contain
            the position data (*self.input_path_data_type = ['x', 'y', ...]*).
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values.
        Y : np.ndarray, optional
            This is the future observed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        T : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings that indicate
            the type of agent observed (see definition of **provide_all_included_agent_types()** for available types).
            If an agent is not observed at all, the value will instead be '0'.
        Agent_names : np.ndarray
            This is a :math:`N_{agents}` long numpy array. It includes strings with the names of the agents.
            
            If only one agent has to be predicted per sample, for **img** and **img_m_per_px**, :math:`N_{agents} = 1` will
            be returned instead, and the agent to predicted will be the one mentioned first in **X** and **T**.

        Returns
        -------
        X_pert : np.ndarray
            This is the past perturbed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{I} \times N_{data}\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values.
        Y_pert : np.ndarray, optional
            This is the future perturbed data of the agents, in the form of a
            :math:`\{N_{samples} \times N_{agents} \times N_{O} \times 2\}` dimensional numpy array with float values. 
            If an agent is fully or or some timesteps partially not observed, then this can include np.nan values. 
            This value is not returned for **mode** = *'pred'*.
        

        '''
        raise AttributeError('This function has to be implemented in the actual perturbation method.')

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
        raise AttributeError('This function has to be implemented in the actual perturbation method.')
    

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
        raise AttributeError('This function has to be implemented in the actual perturbation method.')
    
    def get_constraints(self):
        '''
        This function returns the constraints for the data to be perturbed.

        Returns
        -------
        def
            A function used to calculate constraints.

        '''
        return AttributeError('This function has to be implemented in the actual perturbation method.')
    

    def set_batch_size(self):
        '''
        This function sets the batch size for the perturbation method.

        It must add a attribute self.batch_size to the class.

        Returns
        -------
        None.

        '''
        raise AttributeError('This function has to be implemented in the actual perturbation method.')
