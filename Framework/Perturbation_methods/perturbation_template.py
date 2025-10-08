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

    def set_default_size(self, Type):
        Size = pd.DataFrame(columns=Type.columns, index=Type.index, dtype = object)

        default_dict = {
            'V': np.array([5.0, 2.0]),
            'M': np.array([2.0, 0.5]),
            'B': np.array([2.0, 0.5]),
            'P': np.array([0.5, 0.5]),
        }
        T_array = Type.to_numpy()

        for typ, size in default_dict.items():
            useful = T_array == typ
            size_array = np.array([size] * useful.sum() + ['helper'], dtype = object)[:-1]
            Size.values[useful] = size_array

        return Size

    def perturb(self, data):
        self.data = data

        Input_path  = data.Input_path
        Input_T     = data.Input_T

        Output_path = data.Output_path
        Output_T    = data.Output_T

        Type        = data.Type
        Domain      = data.Domain

        if data.size_given:
            Size = data.Size
        else:
            Size = self.set_default_size(Type)

        # Transform the data into numpy arrays
        Agents = np.array(Input_path.columns)
        pred_agents = np.array([agent in data.needed_agents for agent in Agents])

        # Get input data type
        self.input_path_data_type = self.data.path_data_info()

        # Check the requirements for the data, create error if not fulfilled
        requirements = self.requirerments()

        # Transform agent types to numpy array
        T = Type.to_numpy().astype(str)
        T[T == 'nan'] = '0'

        # Transform size to array
        overwrite = Size.isna().to_numpy()
        S = Size.to_numpy()
        
        # overwrtite nan stuff
        overwrite_array = np.array([np.full(2, np.nan)] * overwrite.sum() + ['test'], dtype = object)[:-1]
        S[overwrite] = overwrite_array
        S = np.stack(S.tolist())

        # Get length of future data
        N_O = np.zeros(len(Output_T), int)
        for i_sample in range(Output_T.shape[0]):
            if 'dt' in requirements.keys():
                assert np.all(np.abs(np.diff(Input_T[i_sample]) - requirements['dt']) < 1e-3), "The input time steps are not constant."
                assert np.all(np.abs(np.diff(Output_T[i_sample]) - requirements['dt']) < 1e-3), "The output time steps are not constant."

            N_O[i_sample] = len(Output_T[i_sample])


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

                    X[i_sample, i_agent]          = Input_path.loc[i_index, agent].astype(np.float32)[...,:2]
                    Y[i_sample, i_agent, :n_time] = Output_path.loc[i_index, agent][:n_time].astype(np.float32)[...,:2]

        # Get the batch size
        self.set_batch_size()
        assert hasattr(self, 'batch_size'), "The batch size is not defined."
        assert isinstance(self.batch_size, int), "The given batch size must be an integer."

        sorted_indices = np.argsort(-N_O)

        # Reorder both X_pert and Y_pert arrays based on sorted indices
        X_sort = X[sorted_indices]
        Y_sort = Y[sorted_indices]
        T_sort = T[sorted_indices]
        S_sort = S[sorted_indices]
        Domain_sort = Domain.iloc[sorted_indices]

        # Run perturbation
        X_pert_sort = np.copy(X_sort)
        Y_pert_sort = np.copy(Y_sort)

        dt = self.data.dt

        # Get constraints of datasets
        contstraints = self.get_constraints()

        if contstraints is not None:
            self.contstraints = contstraints(X_pert_sort, Y_pert_sort, dt)
            
        # Prepare graphs
        if 'graph_id' in Domain_sort.columns:
            graph_ids = Domain_sort.graph_id.to_numpy()
            Path_ids = Domain_sort[['Path_ID', 'path_addition']].to_numpy().astype(str)

            # Get unique Path_ids, with index
            index_unique_path = np.unique(Path_ids, axis = 0, return_index = True)[1]
            graph_ids_old = graph_ids[index_unique_path]

            # For each unique graph_id, check how often they are repeated
            unqiue_graph_id, counts = np.unique(graph_ids_old, return_counts = True)

            # Transfer to dictionary
            data.graph_count = dict(zip(unqiue_graph_id, counts))

            if np.max(counts) == 1:
                data.graph_count_always_one = True
            else:
                data.graph_count_always_one = False

        # Go through the data 
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        for i_batch in range(num_batches):
            print(f'Perturbing batch {i_batch + 1}/{num_batches}', flush = True)
            # Get the samples for this batch
            i_start = i_batch * self.batch_size
            i_end = min((i_batch + 1) * self.batch_size, X.shape[0])

            samples = np.arange(i_start, i_end)

            # Predagents are the predefined agents
            Pred_agents = np.tile(pred_agents[np.newaxis], (len(samples), 1))

            # Get images and graphs
            if data.includes_images() and hasattr(self, 'pert_model') and self.pert_model.can_use_map:
                Img_needed = T_sort[samples] != '0'
                domain_needed = Domain_sort.iloc[samples[np.where(Img_needed)[0]]]

                # Only use the input positions
                X_needed = X_sort[samples][Img_needed]
                centre = X_needed[:, -1,:].copy()
                x_rel = centre - X_needed[:, -2,:]
                rot = np.angle(x_rel[:,0] + 1j * x_rel[:,1]) 

                if hasattr(self, 'use_batch_extraction') and self.use_batch_extraction:
                    print_progress = False
                else:
                    print_progress = True
                
                if self.pert_model.grayscale:
                    img_needed = np.zeros((X_needed.shape[0], self.pert_model.target_height, self.pert_model.target_width, 1), dtype = np.uint8)
                else:
                    img_needed = np.zeros((X_needed.shape[0], self.pert_model.target_height, self.pert_model.target_width, 3), dtype = np.uint8)
                img_needed = data.return_batch_images(domain_needed, centre, rot,
                                                      target_height = self.pert_model.target_height, 
                                                      target_width = self.pert_model.target_width,
                                                      grayscale = self.pert_model.grayscale,
                                                      Imgs_rot = img_needed,
                                                      Imgs_index = np.arange(X_needed.shape[0]),
                                                      print_progress = False)
                img_m_per_px_needed = data.Images.Target_MeterPerPx.loc[domain_needed.image_id]

                img = np.zeros((len(samples), X_sort.shape[1], *img_needed.shape[1:]), dtype = np.uint8)
                img_m_per_px = np.zeros((len(samples), X_sort.shape[1]), dtype = np.float32)
                img[Img_needed] = img_needed
                img_m_per_px[Img_needed] = img_m_per_px_needed
            else:
                img = None
                img_m_per_px = None
            
            if data.includes_sceneGraphs() and hasattr(self, 'pert_model') and self.pert_model.can_use_graph:
                X_last_all = X_sort[samples][...,-1,:2].copy() # num_samples x num_agents x 2
                X_last_all[~Pred_agents] = np.nan
                if hasattr(self.pert_model, 'sceneGraph_radius'):
                    radius = self.pert_model.sceneGraph_radius
                else:
                    radius = 100
                
                if hasattr(self.pert_model, 'sceneGraph_wave_length'):
                    wave_length = self.pert_model.sceneGraph_wave_length
                else:
                    wave_length = 1.0
                    
                graph = np.full(len(samples), np.nan, dtype = object)
                graph = data.return_batch_sceneGraphs(Domain_sort.iloc[samples], X_last_all, radius, wave_length, graph, np.arange(len(samples)), print_progress = False)
                pass # TODO
            else:
                graph = None

            # get categories
            if 'category' in Domain.columns:
                C = Domain_sort.iloc[samples].category
                C = pd.DataFrame(C.to_list())

                # Adjust columns to match self.data_set.Agents
                C = C.reindex(columns = Agents, fill_value = np.nan)

                # Replace missing agents
                C = C.fillna(4)

                # Get to numpy and apply indices
                C = C.to_numpy().astype(int)
            else:
                C = None

            # self.perturb_batch has to provide: X, Y, T, S, C, img, img_m_per_px, graph, Pred_agents, num_steps
            X_pert_sort[samples], Y_pert_sort[samples] = self.perturb_batch(X_sort[samples], Y_sort[samples], T_sort[samples], S_sort[samples], C,
                                                                            img, img_m_per_px, graph, Agents)


        sort_indices_inverse = np.argsort(sorted_indices)
        X_pert = X_pert_sort[sort_indices_inverse]
        Y_pert = Y_pert_sort[sort_indices_inverse]

        # Get the additional required information.
        data_type = data.path_data_info()
        X_pert, Y_pert = self.extend_postion_data(X_pert, Y_pert, dt, data_type)
        
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
    
    
    def get_nan_gradient(self, F, axis = -1):
        '''
        This function calculates the gradient of a numpy array, while ignoring NaN values.

        Parameters
        ----------
        F : np.ndarray
            The input array. Shape is [..., M].
        axis : int, optional
            The axis along which to calculate the gradient. The default is -1.

        Returns
        -------
        np.ndarray
            The gradient of the input array.

        '''
        missing_timesteps_start = np.isfinite(F).argmax(-1) # [...]
        missing_timesteps_end = missing_timesteps_start + np.isfinite(F).sum(-1) # [...]

        missing_timesteps = np.stack([missing_timesteps_start, missing_timesteps_end], -1) # [..., 2]
        missing_timesteps = np.unique(missing_timesteps.reshape(-1,2), axis = 0) # n, 2 
        
        dF = np.zeros_like(F) # [..., M]
        for (missing_timestep_start, missing_timestep_end) in missing_timesteps:
            mask = (missing_timestep_start == missing_timesteps_start) & (missing_timestep_end == missing_timesteps_end) # [...]
            
            if np.abs(missing_timestep_end - missing_timestep_start) > 1:
                f_mask = F[mask] # [N, M]
                f_mask_time = f_mask[:,missing_timestep_start:missing_timestep_end] # [N,m]
                assert np.isfinite(f_mask_time).all(), "There are still NaN values in the original data."
                df_mask_time = np.gradient(f_mask_time, axis = -1) # [N,m]
                assert np.isfinite(df_mask_time).all(), "There are still NaN values in the derivatives."
                df_mask = np.zeros_like(f_mask) # [N, M]
                df_mask[:,missing_timestep_start:missing_timestep_end] = df_mask_time
                dF[mask] = df_mask
                
        dF[~np.isfinite(F)] = np.nan
        return dF

    def extend_postion_data(self, P_in, P_out, dt, data_type):
        if len(data_type) > 2:
            print('Position derivatives need to be calculated.', flush = True)
            # Combine trajectories
            n_in = P_in.shape[2]
            n_out = P_out.shape[2]

            P_full = np.ones((P_in.shape[0], P_in.shape[1], n_in + n_out, len(data_type)), dtype = np.float32) * np.nan
            P_full[..., :2] = np.concatenate((P_in, P_out), axis = -2) 
            
            
            # Collapse sample/agent dimensions
            P_full = P_full.reshape(-1, n_in + n_out, len(data_type))

            # Do marginal velocities first
            if 'v_x' in data_type:
                i_vx = data_type.index('v_x')
                
                P_v_x = self.get_nan_gradient(P_full[..., 0], axis = -1) / dt
                P_full[..., i_vx] = P_v_x

            if 'v_y' in data_type:
                i_vy = data_type.index('v_y')
                P_v_y = self.get_nan_gradient(P_full[..., 1], axis = -1) / dt
                P_full[..., i_vy] = P_v_y
            
            # Do marginal accelerations, based on previous velocities
            if 'a_x' in data_type:
                i_ax = data_type.index('a_x')
                # If velocities are required, they will have been calculated already
                if 'v_x' in data_type:
                    i_vx = data_type.index('v_x')
                    P_vx = P_full[..., i_vx]
                else:
                    P_vx = self.get_nan_gradient(P_full[..., 0], axis = -1) / dt
                P_ax = self.get_nan_gradient(P_vx, axis = -1) / dt
                P_full[..., i_ax] = P_ax
            
            if 'a_y' in data_type:
                i_ay = data_type.index('a_y')
                # If velocities are required, they will have been calculated already
                if 'v_y' in data_type:
                    i_vy = data_type.index('v_y')
                    P_vy = P_full[..., i_vy]
                else:
                    P_vy = self.get_nan_gradient(P_full[..., 1], axis = -1) / dt
                P_ay = self.get_nan_gradient(P_vy, axis = -1) / dt
                P_full[..., i_ay] = P_ay

            # Do total velocities, based on previous velocities
            if 'v' in data_type:
                i_v = data_type.index('v')
                if 'v_x' in data_type:
                    i_vx = data_type.index('v_x')
                    P_vx = P_full[..., i_vx]
                else:
                    P_vx = self.get_nan_gradient(P_full[..., 0], axis = -1) / dt
                if 'v_y' in data_type:
                    i_vy = data_type.index('v_y')
                    P_vy = P_full[..., i_vy]
                else:
                    P_vy = self.get_nan_gradient(P_full[..., 1], axis = -1) / dt
                P_v = np.sqrt(P_vx**2 + P_vy**2)
                P_full[..., i_v] = P_v
            
            # Do headings, based on previous velocities
            if 'theta' in data_type:
                i_theta = data_type.index('theta')
                if 'v_x' in data_type:
                    i_vx = data_type.index('v_x')
                    P_vx = P_full[..., i_vx]
                else:
                    P_vx = self.get_nan_gradient(P_full[..., 0], axis = -1) / dt
                if 'v_y' in data_type:
                    i_vy = data_type.index('v_y')
                    P_vy = P_full[..., i_vy]
                else:
                    P_vy = self.get_nan_gradient(P_full[..., 1], axis = -1) / dt
                P_theta = np.arctan2(P_vy, P_vx)
                P_full[..., i_theta] = P_theta

            # Do total accelerations, based on previous velocities
            if 'a' in data_type:
                i_a = data_type.index('a')
                if 'v' in data_type:
                    i_v = data_type.index('v')
                    P_v = P_full[..., i_v]
                else:
                    i_v = data_type.index('v')
                    if 'v_x' in data_type:
                        i_vx = data_type.index('v_x')
                        P_vx = P_full[..., i_vx]
                    else:
                        P_vx = self.get_nan_gradient(P_full[..., 0], axis = -1) / dt
                    if 'v_y' in data_type:
                        i_vy = data_type.index('v_y')
                        P_vy = P_full[..., i_vy]
                    else:
                        P_vy = self.get_nan_gradient(P_full[..., 1], axis = -1) / dt
                    P_v = np.sqrt(P_vx**2 + P_vy**2)

                P_a = self.get_nan_gradient(P_v, axis = -1) / dt
                P_full[..., i_a] = P_a

            # Do change in heading
            if 'd_theta' in data_type:
                i_d_theta = data_type.index('d_theta')
                if 'theta' in data_type:
                    i_theta = data_type.index('theta')
                    P_theta = P_full[..., i_theta]
                else:                
                    if 'v_x' in data_type:
                        i_vx = data_type.index('v_x')
                        P_vx = P_full[..., i_vx]
                    else:
                        P_vx = self.get_nan_gradient(P_full[..., 0], axis = -1) / dt
                    if 'v_y' in data_type:
                        i_vy = data_type.index('v_y')
                        P_vy = P_full[..., i_vy]
                    else:
                        P_vy = self.get_nan_gradient(P_full[..., 1], axis = -1) / dt
                    P_theta = np.arctan2(P_vy, P_vx)

                P_theta = np.unwrap(P_theta, axis = -1)
                P_d_theta = self.get_nan_gradient(P_theta, axis = -1) / dt
                P_full[..., i_d_theta] = P_d_theta

            # Reverse flattening of P_full
            P_full = P_full.reshape(P_in.shape[0], P_in.shape[1], n_in + n_out, len(data_type))
            
            P_in_full  = P_full[..., :n_in, :]
            P_out_full = P_full[..., n_in:, :]
            print('Past.shape: {} - Future.shape: {}'.format(P_in_full.shape, P_out_full.shape), flush = True)
            return P_in_full, P_out_full
        else:   
            print('No derivatives needed.', flush = True)
            print('Past.shape: {} - Future.shape: {}'.format(P_in.shape, P_out.shape), flush = True)
            return P_in, P_out



        
    
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
