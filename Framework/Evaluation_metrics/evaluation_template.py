import pandas as pd
import numpy as np
import os


class evaluation_template():
    def __init__(self, metric_kwargs, data_set, splitter, model):
        self.metric_kwargs = metric_kwargs
        if data_set is not None:
            self.data_set = data_set
            self.splitter = splitter
            self.model = model
            
            self.depict_results = False
            
            self.Scenario_full   = self.data_set.Domain.Scenario_type
            
            self.t_e_quantile = self.data_set.p_quantile
            
            self.metric_override = self.data_set.overwrite_results in ['model', 'prediction', 'metric']
            
            self.setup_method() # output needs to be a list of components
            
            if self.requires_preprocessing():
                test_file = self.data_set.change_result_directory(splitter.split_file,
                                                                  'Metrics', self.get_name()['file'] + '_weights')
                if os.path.isfile(test_file):
                    self.weights_saved = list(np.load(test_file, allow_pickle = True)[:-1])  
                else:
                    save_data = np.array(self.weights_saved + [0], object) # 0 is there to avoid some numpy load and save errros
                    
                    os.makedirs(os.path.dirname(test_file), exist_ok=True)
                    np.save(test_file, save_data)
        else:
            self.depict_results = True


    def _evaluate_on_subset(self, Output_pred, evaluation_index):
        self.Index_curr = evaluation_index

        # Get the correspoding file index
        if self.data_set.data_in_one_piece:
            self.file_index = 0
        else:
            file_indices = self.data_set.Domain.file_index.iloc[self.Index_curr].to_numpy()
            assert len(np.unique(file_indices)) == 1, 'This metric does not support multiple files in one evaluation.'
            self.file_index = file_indices[0]
        
        if len(self.Index_curr) == 0:
            return None
        
        available = self._set_current_data(Output_pred)
        if available:
            results = self.evaluate_prediction_method()
        else:
            results = None
        
        return results
    

    #%% Actual evaluation functions
    def _set_current_data(self, Output_pred):
        assert np.array_equal(self.Index_curr, Output_pred[0]) # Index of evaluation samples does not overlapwith predicted samples

        if self.data_set.data_in_one_piece:
            Output_A_full   = self.data_set.Output_A.iloc[self.Index_curr]
            Output_T_E_full = self.data_set.Output_T_E[self.Index_curr]
        
        else:
            file_indeces_curr = self.data_set.Domain.file_index.iloc[self.Index_curr]

            assert len(np.unique(file_indeces_curr)) == 1, 'This metric does not support multiple files in one evaluation.'
            file_index = file_indeces_curr.iloc[0]

            # Load Output_A and Output_T_E
            data_file = self.data_set.Files[file_index] + '_data.npy'
            [_, _, _, _, _, _, Output_A_local, Output_T_E_local, _] = np.load(data_file, allow_pickle = True)

            ind_used = self.data_set.Domain.Index_saved.iloc[self.Index_curr]
            Output_A_full   = Output_A_local.reindex(columns = self.data_set.Behaviors).loc[ind_used] # Shape len(Index_curr) x num_behaviors
            Output_T_E_full = Output_T_E_local[ind_used]    # Shape len(Index_curr)
        
        if self.get_output_type()[:5] == 'class':
            # Get label predictions
            Output_A_pred = Output_pred[1]
            columns = Output_A_pred.columns
            self.Output_A_pred = Output_A_pred[columns]
            self.Output_A      = Output_A_full[columns]
            self.Scenario      = np.unique(self.Scenario_full.iloc[self.Index_curr])
            
            if self.get_output_type() == 'class_and_time':
                Output_T_E_pred = Output_pred[2]
                
                # reorder columns if needed
                self.Output_T_E      = Output_T_E_full
                self.Output_T_E_pred = Output_T_E_pred[columns]
        
        else:
            Agents = np.array(self.data_set.Agents)
            self.Output_path_pred = Output_pred[1][Agents]
            self.Output_path_pred_probs = Output_pred[2][Agents]
        
        return True
    


    ##############################################################################################
    #                Helper functions for the evaluation of the metric                           #
    ##############################################################################################

    def _check_collisions(self, Path_A, Path_B, Size_A, Size_B):
        r'''
        This function checks if two agents collide with each other.

        Parameters
        ----------
        Path_A : np.ndarray
            The path of the first agent, in the form of a :math:`\{... \times N_{O} \times 2\}` dimensional 
            numpy array. Here, :math:`N_{O}` is the number of observed timesteps.
        Path_B : np.ndarray
            The path of the second agent, in the form of a :math:`\{... \times N_{O} \times 2\}` dimensional
            numpy array. Here, :math:`N_{O}` is the number of observed timesteps.
        Size_A : np.ndarray
            The type of the first agent, in the form of a :math:`\{... \times 2\}` dimensional numpy array, 
            with string the length x width values of the agents
        Size_B : np.ndarray
            The type of the second agent, in the form of a :math:`\{... \times 2\}` dimensional numpy array, 
            with string the length x width values of the agents
        
        Returns
        -------
        Collided : np.ndarray
            This is a :math:`\{...\}` dimensional numpy array with boolean values. It indicates if a 
            collision was detected for the corresponding pair of agents.
        '''
        # Get the corresponding angles
        Theta_A = np.arctan2(Path_A[...,1:,1] - Path_A[...,:-1,1], Path_A[...,1:,0] - Path_A[...,:-1,0])
        Theta_B = np.arctan2(Path_B[...,1:,1] - Path_B[...,:-1,1], Path_B[...,1:,0] - Path_B[...,:-1,0])
        
        # Elongate the theta values to original length, assuming that the
        # Calculated angels correspond to the recorded angles at the middle of each timestep
        Theta_A_start = Theta_A[...,0] - 0.5 * (Theta_A[...,1] - Theta_A[...,0])
        Theta_B_start = Theta_B[...,0] - 0.5 * (Theta_B[...,1] - Theta_B[...,0])

        Theta_A_end = Theta_A[...,-1] + 0.5 * (Theta_A[...,-1] - Theta_A[...,-2])
        Theta_B_end = Theta_B[...,-1] + 0.5 * (Theta_B[...,-1] - Theta_B[...,-2])
        
        Theta_A = np.nanmean(np.stack((Theta_A[...,1:], Theta_A[...,:-1]), -1), -1)
        Theta_B = np.nanmean(np.stack((Theta_B[...,1:], Theta_B[...,:-1]), -1), -1)

        Theta_A = np.concatenate((Theta_A_start[...,np.newaxis], Theta_A, Theta_A_end[...,np.newaxis]), axis=-1)
        Theta_B = np.concatenate((Theta_B_start[...,np.newaxis], Theta_B, Theta_B_end[...,np.newaxis]), axis=-1)

        # Get the relative positions
        Path_B_adj = Path_B - Path_A

        # Turn the relative positions so that the ego vehicle is aligned with the x-axis
        Path_B_adj = np.stack((Path_B_adj[...,0] * np.cos(-Theta_A) - Path_B_adj[...,1] * np.sin(-Theta_A),
                               Path_B_adj[...,0] * np.sin(-Theta_A) + Path_B_adj[...,1] * np.cos(-Theta_A)), axis=-1)

        # Get the relative angles
        Theta_B -= Theta_A # Shape (..., N_O)
        del Theta_A

        # Get initial corner positions
        Corner_A = np.stack([np.stack([-Size_A[...,0]/2, -Size_A[...,1]/2], -1),
                             np.stack([ Size_A[...,0]/2, -Size_A[...,1]/2], -1),
                             np.stack([ Size_A[...,0]/2,  Size_A[...,1]/2], -1),
                             np.stack([-Size_A[...,0]/2,  Size_A[...,1]/2], -1)], -2) # Shape (..., 4, 2)
        
        Corner_B = np.stack([np.stack([-Size_B[...,0]/2, -Size_B[...,1]/2], -1),
                             np.stack([ Size_B[...,0]/2, -Size_B[...,1]/2], -1),
                             np.stack([ Size_B[...,0]/2,  Size_B[...,1]/2], -1),
                             np.stack([-Size_B[...,0]/2,  Size_B[...,1]/2], -1)], -2) # Shape (..., 4, 2)
        
        # Rotate the Corner B with the relative angles
        Corner_A = Corner_A[...,np.newaxis, :, :] # Shape (..., 1, 4, 2)
        Corner_B = Corner_B[...,np.newaxis, :, :] # Shape (..., 1, 4, 2)
        Theta_B  = Theta_B[...,np.newaxis] # Shape (..., N_O, 1)

        Corner_B = np.stack((Corner_B[...,0] * np.cos(Theta_B) - Corner_B[...,1] * np.sin(Theta_B),
                             Corner_B[...,0] * np.sin(Theta_B) + Corner_B[...,1] * np.cos(Theta_B)), axis=-1) # Shape (..., N_O, 4, 2)
        
        # Translate the corners to the center
        Corner_B += Path_B_adj[...,np.newaxis, :] # Shape (..., N_O, 4, 2)

        # Use the separating axis theorem to check for collisions
        # Get the 2 normals of rectangle A
        Norm_A_1 = np.array([1, 0])
        Norm_A_2 = np.array([0, 1])

        # Get the 2 normals of rectangle B
        Norm_B_1 = np.concatenate([np.cos(Theta_B), np.sin(Theta_B)], -1) # Shape (..., N_O, 2)
        Norm_B_2 = np.concatenate([-np.sin(Theta_B), np.cos(Theta_B)], -1) # Shape (..., N_O, 2)

        # Combine the normals
        for size in Norm_B_1.shape[:-1]:
            Norm_A_1 = np.repeat(Norm_A_1[...,np.newaxis, :], size, axis = -2)
            Norm_A_2 = np.repeat(Norm_A_2[...,np.newaxis, :], size, axis = -2)

        Normals = np.stack((Norm_A_1, Norm_A_2, Norm_B_1, Norm_B_2), -2) # Shape (..., N_O, 4, 2)
        
        # Switch last two dimensions for the corners A and B
        Corner_A = np.moveaxis(Corner_A, -1, -2) # Shape (..., 1, 2, 4)
        Corner_B = np.moveaxis(Corner_B, -1, -2) # Shape (..., N_O, 2, 4)

        # Project the corners of rectangle A onto Normals
        Projections_A = np.matmul(Normals, Corner_A) # Shape (..., N_O, 4(normals), 4(corners A))
        Projections_B = np.matmul(Normals, Corner_B) # Shape (..., N_O, 4(normals), 4(corners B))
        
        del Corner_A, Corner_B, Normals
        
        # Get the diffrences between the points
        Differences = Projections_A[...,np.newaxis,:] - Projections_B[...,np.newaxis] # Shape (..., N_O, 4(normals), 4(corners A), 4(corners B))
        Differences = Differences.reshape((*Differences.shape[:-2], -1)) # Shape (..., N_O, 4(normals), 16(corners A - corners B))
        del Projections_A, Projections_B

        Differences_0 = Differences[...,:-1,:,:] # Shape (..., N_O - 1, 4(normals), 16(corners A - corners B))
        Differences_1 = Differences[...,1:,:,:] # Shape (..., N_O - 1, 4(normals), 16(corners A - corners B))
        
        # Find seperated normals
        Sign_0 = np.sign(Differences_0) # Shape (..., N_O - 1, 4(normals), 16(corners A - corners B))
        Sign_1 = np.sign(Differences_1) # Shape (..., N_O - 1, 4(normals), 16(corners A - corners B))
        
        # Check if the signs are different
        Sign_0_equal = (Sign_0[...,[0]] == Sign_0).all(-1) # Shape (..., N_O - 1, 4(normals))
        Sign_1_equal = (Sign_1[...,[0]] == Sign_1).all(-1) # Shape (..., N_O - 1, 4(normals))
        
        # There was definetly no overlap if both starting and endpoint had no overlap, and had the same sign
        No_overlap = Sign_0_equal & Sign_1_equal & (Sign_0[...,0] == Sign_1[...,0]) # Shape (..., N_O - 1, 4(normals))
        
        # If there is any normal with no overlap, then there is no collision
        No_collision = No_overlap.any(-1) # Shape (..., N_O - 1)
        
        # If either at the start or the end all normal have mixed signs, then there is a collision
        Definite_collision = (~np.any(Sign_0_equal, axis = -1)) | (~np.any(Sign_1_equal, axis = -1)) # Shape (..., N_O - 1)
        
        # If neither No_collision nor Definite_collision is observed, then further investigation is needed
        Investigate_further = ~(No_collision | Definite_collision) # Shape (..., N_O - 1)
        
        # Investigate further
        if Investigate_further.any():
            # N = Num furhter investigations
            Diff_0_further = Differences_0[Investigate_further] # Shape (N, 4, 16(corners A - corners B))
            Diff_1_further = Differences_1[Investigate_further] # Shape (N, 4, 16(corners A - corners B))
            
            Sign_0_further = Sign_0[Investigate_further] # Shape (N, 4(normals), 16(corners A - corners B))
            Sign_1_further = Sign_1[Investigate_further] # Shape (N, 4(normals), 16(corners A - corners B))
            
            Sign_0_equal_further = Sign_0_equal[Investigate_further] # Shape (N, 4(normals))
            Sign_1_equal_further = Sign_1_equal[Investigate_further] # Shape (N, 4(normals))
            
            # Get the factor at which the Diff gets zero
            Diff_diff = Diff_0_further - Diff_1_further # Shape (N, 4, 16(corners A - corners B))
            Diff_diff[(0 <= Diff_diff) & (Diff_diff < 1e-6)] = 1e-6
            Diff_diff[(0 >= Diff_diff) & (Diff_diff > -1e-6)] = -1e-6
            Factor = Diff_0_further / Diff_diff # Shape (N, 4, 16(corners A - corners B))
            
            # Check for each normal, during which interval there is an overlap
            # There four possibilities:
            # Case 1: Start with no overlap, end with no overlap => There is an interval [a,b] with a,b in [0,1] where there is an overlap
            # Case 2: Start with no overlap, end with overlap => There is an interval [a,1] with a in [0,1] where there is an overlap
            # Case 3: Start with overlap, end with no overlap => There is an interval [0,b] with b in [0,1] where there is an overlap
            # Case 4: Start with overlap, end with overlap on different sides => There is a possibility of an interval [a,b] with a,b in [0,1] where there is no overlap
            
            # Get the different cases
            case_1 = Sign_0_equal_further & Sign_1_equal_further # Shape (N, 4(normals))
            # For this case, the sign should be different
            assert (Sign_0_further[case_1] != Sign_1_further[case_1]).all(), 'There is a bug in the code'
            
            case_2 = Sign_0_equal_further & ~Sign_1_equal_further # Shape (N, 4(normals))
            
            case_3 = ~Sign_0_equal_further & Sign_1_equal_further # Shape (N, 4(normals))
            
            case_4 = ~Sign_0_equal_further & ~Sign_1_equal_further # Shape (N, 4(normals))
            
            Overlap_start = np.zeros_like(Factor[...,0], float) # Shape (N, 4)
            Overlap_end   = np.ones_like(Factor[...,0], float) # Shape (N, 4)
            
            Overlap_start_2 = np.ones_like(Factor[...,0], float) # Shape (N, 4)
            Overlap_end_2   = np.ones_like(Factor[...,0], float) # Shape (N, 4)
            # Go through the different cases
            if case_1.any():
                # Get the factors where there is an overlap
                Factor_case_1 = Factor[case_1] # Shape (N_case_1, 16(corners A - corners B))
                
                # All the factors should be between 0 and 1
                Fac_min = Factor_case_1.min(-1) # Shape (N_case_1)
                Fac_max = Factor_case_1.max(-1) # Shape (N_case_1)
                assert (Fac_min >= 0).all() and (Fac_max <= 1).all(), 'There is a bug in the code'
                
                Overlap_start[case_1] = Fac_min
                Overlap_end[case_1] = Fac_max
                
            if case_2.any():
                # Get the factors where there is an overlap
                Factor_case_2 = Factor[case_2] # Shape (N_case_2, 16(corners A - corners B))
                
                # Get the corresponding signs
                Sign_0_case_2 = Sign_0_further[case_2] # Shape (N_case_2, 16(corners A - corners B))
                Sign_1_case_2 = Sign_1_further[case_2] # Shape (N_case_2, 16(corners A - corners B))
                
                # Get the factors where the sign changes
                Sign_change = Sign_0_case_2 != Sign_1_case_2 # Shape (N_case_2, 16(corners A - corners B))
                
                # Factors with sign change should be between 0 and 1
                assert (Factor_case_2[Sign_change] >= 0).all() and (Factor_case_2[Sign_change] <= 1).all(), 'There is a bug in the code'
                
                # Get the minimum factor where the sign changes
                Fac_min = np.maximum(Factor_case_2, (~Sign_change).astype(float)).min(-1) # Shape (N_case_2)
                
                # Get interval start
                Overlap_start[case_2] = Fac_min
            
            if case_3.any():
                # Get the factors where there is an overlap
                Factor_case_3 = Factor[case_3] # Shape (N_case_3, 16(corners A - corners B))	
                
                # Get the corresponding signs
                Sign_0_case_3 = Sign_0_further[case_3] # Shape (N_case_3, 16(corners A - corners B))
                Sign_1_case_3 = Sign_1_further[case_3] # Shape (N_case_3, 16(corners A - corners B))
                
                # Get the factors where the sign changes
                Sign_change = Sign_0_case_3 != Sign_1_case_3 # Shape (N_case_3, 16(corners A - corners B))
                
                # Factors with sign change should be between 0 and 1
                assert (Factor_case_3[Sign_change] >= 0).all() and (Factor_case_3[Sign_change] <= 1).all(), 'There is a bug in the code'
                
                # Get the maximum factor where the sign changes
                Fac_max = np.minimum(Factor_case_3, Sign_change.astype(float)).max(-1) # Shape (N_case_3)
                
                # Get interval end
                Overlap_end[case_3] = Fac_max
            
            if case_4.any():
                # Get the factors where there is an overlap
                Factor_case_4 = Factor[case_4] # Shape (N_case_4, 16(corners A - corners B))
                Diff_0_case_4 = Diff_0_further[case_4] # Shape (N_case_4, 16(corners A - corners B))
                Diff_1_case_4 = Diff_1_further[case_4] # Shape (N_case_4, 16(corners A - corners B))
                
                # Clip Factors to be between 0 and 1
                Factor_case_4 = np.clip(Factor_case_4, 0, 1) # Shape (N_case_4, 16(corners A - corners B))
                
                # Each of the lines shoudl have an interval where the sign is positive and one interval where the sign is negative
                Positive_start = np.zeros_like(Factor_case_4, float) # Shape (N_case_4, 16(corners A - corners B))
                Positive_end   = np.ones_like(Factor_case_4, float) # Shape (N_case_4, 16(corners A - corners B))
                
                Negative_start = np.zeros_like(Factor_case_4, float) # Shape (N_case_4, 16(corners A - corners B))
                Negative_end   = np.ones_like(Factor_case_4, float) # Shape (N_case_4, 16(corners A - corners B))
                
                # For decresing lines, it swithes from positive to negative
                Decresing = Diff_0_case_4 > Diff_1_case_4 # Shape (N_case_4, 16(corners A - corners B))
                
                Positive_end[Decresing] = Factor_case_4[Decresing]
                Negative_start[Decresing] = Factor_case_4[Decresing]
                
                # For increasing lines, it swithes from negative to positive
                Increasing = ~Decresing # Shape (N_case_4, 16(corners A - corners B))
                
                Positive_start[Increasing] = Factor_case_4[Increasing]
                Negative_end[Increasing] = Factor_case_4[Increasing]
                
                # Get the maximum start and minimum end values
                Positive_start = Positive_start.max(-1) # Shape (N_case_4)
                Positive_end = Positive_end.max(-1) # Shape (N_case_4)
                Negative_start = Negative_start.min(-1) # Shape (N_case_4)
                Negative_end = Negative_end.min(-1) # Shape (N_case_4)
                
                # The no overlap is if positive_start < positive_end or negative_start < negative_end
                Positive_no_overlap = Positive_start < Positive_end # Shape (N_case_4)
                Negative_no_overlap = Negative_start < Negative_end # Shape (N_case_4)
                
                # Both conditions should be impossible
                assert (Positive_no_overlap & Negative_no_overlap).any(), 'There is a bug in the code'
                
                # Get the interval where there is no overlap
                No_overlap_start = np.ones_like(Positive_start, float) # Shape (N_case_4)
                No_overlap_end = np.ones_like(Positive_start, float) # Shape (N_case_4)
                
                No_overlap_start[Positive_no_overlap] = Positive_start[Positive_no_overlap]
                No_overlap_end[Positive_no_overlap] = Positive_end[Positive_no_overlap]
                
                No_overlap_start[Negative_no_overlap] = Negative_start[Negative_no_overlap]
                No_overlap_end[Negative_no_overlap] = Negative_end[Negative_no_overlap]
                
                # Define the left overlap interval
                Overlap_end[case_4] = No_overlap_start
                
                # Define the right overlap interval
                Overlap_start_2[case_4] = No_overlap_end 
            
            # Check if second intervals exist
            Two_intervals = (Overlap_start_2 < 1.0).any(-1) # Shape (N)
            One_interval = ~Two_intervals # Shape (N)
            
            Collision_found = np.zeros_like(Two_intervals, bool) # Shape (N)
            
            if Two_intervals.any():
                Overlap_start_two = np.stack((Overlap_start[Two_intervals], Overlap_start_2[Two_intervals]), -1) # Shape (M, 4, 2)
                Overlap_end_two = np.stack((Overlap_end[Two_intervals], Overlap_end_2[Two_intervals]), -1) # Shape (M, 4, 2)
                # Check if there is any time where there is an overlap on all normals
                # Given that there are 4 x 2 intervals, ther are 4 ^ 2 possible interval combinations an overlap could happen in.
                # I.e, transform the N x 4 x 2 intervals to a N x 4 x 16 interval matrix
                
                I_normal = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                                     [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                                     [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],])
                
                I_interv = np.array([[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
                                     [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                                     [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                                     [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])
                
                Overlap_start_two = Overlap_start_two[...,I_normal, I_interv] # Shape (M, 4, 16)
                Overlap_end_two = Overlap_end_two[...,I_normal, I_interv] # Shape (M, 4, 16)
                
                # Check if there is any overlap on all normals
                Overlap_start_two = Overlap_start_two.max(-2) # Shape (M, 16)
                Overlap_end_two = Overlap_end_two.min(-2) # Shape (M, 16)
                
                Overlap_two = Overlap_start_two < Overlap_end_two # Shape (M, 16)
                Overlap_two = Overlap_two.any(-1) # Shape (M)
                
                Collision_found[Two_intervals] = Overlap_two
            
            if One_interval.any():
                # Get combined start interval
                Overlap_start_one = Overlap_start[One_interval].max(-1) # Shape (M)
                Overlap_end_one = Overlap_end[One_interval].min(-1) # Shape (M)
                
                Overlap_one = Overlap_start_one < Overlap_end_one # Shape (M)
                
                Collision_found[One_interval] = Overlap_one
                
            # Expand No_collision to the original shape
            No_collision[Investigate_further] = ~Collision_found

        # Check if any of the Paths had nan values here
        Path_nan = (np.isnan(Path_A) | np.isnan(Path_B)).any(-1) # Shape (..., N_O)

        # For a collision to be possible, both ends need to be observed
        Missing_agent = Path_nan[...,1:] | Path_nan[...,:-1] # Shape (..., N_O - 1)
        No_collision |= Missing_agent

        # For no collision to be observed, this must be the case for all original timesteps
        No_collision = No_collision.all(-1) # Shape (...,)

        Collided = ~No_collision
        return Collided
    


    def get_true_and_predicted_class_probabilities(self):
        '''
        This returns the true and predicted classification probabilities.

        Returns
        -------
        P_true : np.ndarray
            This is the true probabilities with which one will observe a class, in the form of a
            :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array with float values. 
            One value per row will be one, whil ethe others will be zero.
        P_pred : np.ndarray
            This is the predicted probabilities with which one will observe a class, in the form of 
            a :math:`\{N_{samples} \times N_{classes}\}` dimensional numpy array with float values. 
            The sum in each row will be 1.
        Class_names : list
            The list with :math:`N_{classes}` entries contains the names of the aforementioned
            behaviors.

        '''
        assert self.get_output_type()[:5] == 'class', 'This is not a classification metric.'
        
        # Get true and predicted probabilities
        P_true = self.Output_A.to_numpy().astype(float)
        P_pred = self.Output_A_pred.fillna(0.0).to_numpy().astype(float)
        Class_names = self.Output_A.columns
        
        # Remove cases where there was only on possible case to predict
        use_column = np.zeros(P_true.shape[1], bool)
        for i, behs in enumerate(self.data_set.scenario_behaviors):
            if len(behs) > 1:
                if self.data_set.unique_scenarios[i] in self.Scenario:
                    use_column |= np.in1d(Class_names, behs)
        
        # Remove useless columns
        P_true = P_true[:,use_column]
        P_pred = P_pred[:,use_column]
        
        Class_names = Class_names[use_column]
        
        # Remove useles rows
        use_row = P_true.sum(1) == 1
        
        P_true = P_true[use_row]
        P_pred = P_pred[use_row]
        
        return P_true, P_pred, Class_names
    
    
    def get_true_and_predicted_class_times(self):
        '''
        This returns the true and predicted classification timepoints, at which a certain
        behavior can be first classified.

        Returns
        -------
        T_true : np.ndarray
            This is the true time points at which one will observe a class, in the form of a
            :math:`\{N_{samples} \times N_{classes} \times 1\}` dimensional numpy array with float 
            values. One value per row will be given (actual observed class), while the others 
            will be np.nan.
        T_pred : np.ndarray
            This is the predicted time points at which one will observe a class, in the form of a
            :math:`\{N_{samples} \times N_{classes} \times N_{quantiles}\}` dimensional numpy array 
            with float values. Along the last dimesnion, the time values corresponf to the quantile 
            values of the predicted distribution of the time points. The quantile values can be found
            in **self.t_e_quantile**.
        Class_names : list
            The list with :math:`N_{classes}` entries contains the names of the aforementioned
            behaviors.

        '''
        assert self.get_output_type() == 'class_and_time', 'This is not a classification metric.'
        Class_names = self.Output_A.columns
        T_true = np.ones((*self.Output_A.shape, 1)) * np.nan
        T_pred = np.ones((*self.Output_A.shape, self.t_e_quantile)) * np.nan
        
        T_true[np.arange(len(T_true)), np.argmax(self.Output_A.to_numpy(), 1), 0] = self.Output_T_E.to_numpy()
        
        for i in range(T_pred.shape[0]):
            for j in range(T_pred.shape[1]):
                t_pred = self.Output_T_E_pred.iloc[i,j]
                if isinstance(t_pred, np.ndarray):
                    T_pred[i,j] = t_pred
        
        # Get use column and use row
        P_true = self.Output_A.to_numpy().astype(float)
        use_column = np.zeros(P_true.shape[1], bool)
        for i, behs in enumerate(self.data_set.scenario_behaviors):
            if len(behs) > 1:
                if self.data_set.unique_scenarios[i] in self.Scenario:
                    use_column |= np.in1d(Class_names, behs)
        
        P_true = P_true[:,use_column]
        use_row = P_true.sum(1) == 1
        
        # remove useles rows and columns
        Class_names = Class_names[use_column]
        T_true = T_true[use_row][:,use_column]
        T_pred = T_pred[use_row][:,use_column]
        
        return T_true, T_pred, Class_names
    
    
    def get_true_and_predicted_paths(self, num_preds = None, return_types = False, exclude_late_timesteps = True):
        '''
        This returns the true and predicted trajectories.

        Parameters
        ----------
        num_preds : int, optional
            The number :math:`N_{preds}` of different predictions used. The default is None,
            in which case all available predictions are used.
        return_types : bool, optional
            Decides if agent types are returned as well. The default is False.
        exclude_late_timesteps : bool, optional
            Decides if predicted timesteps after the set prediction horizon should be excluded. 
            The default is True.

        Returns
        -------
        Path_true : np.ndarray
            This is the true observed trajectory of the agents, in the form of a
            :math:`\{N_{samples} \times 1 \times N_{agents} \times N_{O} \times 2\}` dimensional numpy 
            array with float values. If an agent is fully or or some timesteps partially not observed, 
            then this can include np.nan values.
        Path_pred : np.ndarray
            This is the predicted furure trajectories of the agents, in the form of a
            :math:`\{N_{samples} \times N_{preds} \times N_{agents} \times N_{O} \times 2\}` dimensional 
            numpy array with float values. If an agent is fully or or some timesteps partially not observed, 
            then this can include np.nan values.
        Pred_steps : np.ndarray
            This is a :math:`\{N_{samples} \times N_{agents} \times N_{O}\}` dimensional numpy array with 
            boolean values. It indicates for each agent and timestep if the prediction should influence
            the final metric result.
        Types : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents}\}` dimensional numpy array. It includes strings 
            that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
            for available types). If an agent is not observed at all, the value will instead be '0'.
            It is only returned if **return_types** is *True*.
        Sizes : np.ndarray, optional
            This is a :math:`\{N_S \times N_{A_other} \times 2\}` dimensional numpy array. It is the sizes of the agents,
            where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
            (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values 
            will instead be np.nan.
            It is only returned if **return_types** is *True*.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        if not hasattr(self, 'pred_idx'):
            # Get the stochastic prediction indices
            if num_preds is None:
                self.pred_idx = np.arange(self.data_set.num_samples_path_pred)
            else:
                if num_preds <= self.data_set.num_samples_path_pred:
                    self.pred_idx = np.random.permutation(self.data_set.num_samples_path_pred)[:num_preds] #
                else:
                    self.pred_idx = np.random.randint(0, self.data_set.num_samples_path_pred, num_preds)
        else:
            if num_preds is None:
                N = self.data_set.num_samples_path_pred
            else:
                N = num_preds
            assert N == len(self.pred_idx), 'The number of predictions does not match the number of predictions in the model.'
        
        self.model._transform_predictions_to_numpy(self.Index_curr, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov',
                                                   exclude_late_timesteps)
        
        Path_true = self.model.Path_true
        Path_pred = self.model.Path_pred[:, self.pred_idx]
        Pred_step = self.model.Pred_step

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        Path_true = Path_true[Use_samples]
        Path_pred = Path_pred[Use_samples]
        Pred_step = Pred_step[Use_samples]

        if return_types:
            Types = self.model.T_pred
            Types = Types[Use_samples]

            Sizes = self.model.S_pred
            Sizes = Sizes[Use_samples]
            return Path_true, Path_pred, Pred_step, Types, Sizes   
        else:
            return Path_true, Path_pred, Pred_step
        
    
    def get_other_agents_paths(self, return_types = False):
        '''
        This returns the true observed trajectories of all agents that are not the
        predicted agents.

        Parameters
        ----------
        return_types : bool, optional
            Decides if agent types are returned as well. The default is False.

        Returns
        -------
        Path_other : np.ndarray
            This is the true observed trajectory of the agents, in the form of a
            :math:`\{N_{samples} \times 1 \times N_{agents_other} \times N_{O} \times 2\}` dimensional numpy 
            array with float values. If an agent is fully or or some timesteps partially not observed, 
            then this can include np.nan values.
        Types : np.ndarray, optional
            This is a :math:`\{N_{samples} \times N_{agents_other}\}` dimensional numpy array. It includes strings 
            that indicate the type of agent observed (see definition of **provide_all_included_agent_types()** 
            for available types). If an agent is not observed at all, the value will instead be '0'.
            It is only returned if **return_types** is *True*.
        Sizes : np.ndarray, optional
            This is a :math:`\{N_S \times N_{A_other} \times 2\}` dimensional numpy array. It is the sizes of the agents,
            where the first column (S[:,:,0]) includes the lengths of the agents (longitudinal size) and the second column
            (S[:,:,1]) includes the widths of the agents (lateral size). If an agent is not observed at all, the values 
            will instead be np.nan.
            It is only returned if **return_types** is *True*.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        
        self.data_set._determine_pred_agents(eval_pov = self.get_output_type() != 'path_all_wo_pov')

        Pred_agents = self.data_set.Pred_agents_eval[self.Index_curr]

        Index_curr_data = self.Index_curr
        if not self.data_set.data_in_one_piece:
            used_index = np.where(self.data_set.Domain.file_index == self.file_index)[0]
            Index_curr_data = self.data_set.get_indices_1D(Index_curr_data, used_index)

        self.data_set._extract_original_trajectories(self.file_index)

        # Get information on available data
        _, data_index_mask = self.model.get_orig_data_index(Index_curr_data)
        
        # Get other interesting agents
        Other_agents = (~Pred_agents) & data_index_mask # at least one position must be fully known

        if Other_agents.any():

            # Order to reduce unnecessary data loading and memory usage
            num_samples = len(self.Index_curr)
            max_num_other_agents = Other_agents.sum(1).max()

            i_agent_sort = np.argsort(-Other_agents.astype(float))
            i_agent_sort = i_agent_sort[:,:max_num_other_agents]
            i_sampl_sort = np.tile(np.arange(num_samples)[:,np.newaxis], (1, max_num_other_agents))

            # Load the required data
            Path_other = np.full((*i_sampl_sort.shape, self.data_set.Y_orig.shape[-2], 2), np.nan, dtype = np.float32)
            data_index, data_index_mask = self.model.get_orig_data_index(Index_curr_data[i_sampl_sort], i_agent_sort)
            Path_other[data_index_mask] = self.data_set.Y_orig[data_index, ..., :2]
            
            # Add the num_samples_path_pred dimension
            Path_other = Path_other[:,np.newaxis]

            if return_types:
                Types = self.model.Type[self.Index_curr[i_sampl_sort], i_agent_sort]
                Sizes = self.model.Size[self.Index_curr[i_sampl_sort], i_agent_sort]

                # Find positions where Path_other is nan
                nan_pos = np.isnan(Path_other).all(-1).all(-1).squeeze(1)
                
                # Test if all nonexisting paths have type zero                 
                assert np.array_equal(Types == '0', np.isnan(Sizes).any(-1)), 'Mismatch between types and sizes.'
                assert not (Types[~nan_pos] == '0').any(), 'There are no types for existing paths.'
                if not (Types[nan_pos] == '0').all():
                    # Find agents with wrong types
                    wrong_agents = nan_pos & (Types != '0')
                    sample_idx, agent_idx = np.where(wrong_agents)

                    Path_other_past = np.full((*i_sampl_sort.shape, self.data_set.X_orig.shape[-2], 2), np.nan, dtype = np.float32)
                    Path_other_past[data_index_mask] = self.data_set.X_orig[data_index, ..., :2]
                    wrong_past_data = Path_other_past[sample_idx, agent_idx] # (num_wrong, num_timesteps, 2)

                    # Check if the past data is missing
                    wrong_wrong = np.isnan(wrong_past_data).all(-1).all(-1)
                    assert not wrong_wrong.any(), 'There are types for nonexisting paths.'
        
        else:
            Path_other = np.full((len(Other_agents), 1, 0, self.data_set.Y_orig.shape[-2], 2), np.nan, dtype = np.float32)
            if return_types:
                Types = np.full((len(Other_agents), 0), '0', dtype = str)
                Sizes = np.full((len(Other_agents), 0, 2), np.nan, dtype = float)

        # Adjust to used samples
        Use_samples = Pred_agents.any(-1)

        # Apply the same adjustemnt as to predicted agents
        Path_other = Path_other[Use_samples]

        if return_types:
            # Apply the same adjustemnt as to predicted agents
            Types = Types[Use_samples]
            Sizes = Sizes[Use_samples]
            return Path_other, Types, Sizes
        
        else:
            return Path_other
    
    
    def get_true_prediction_with_same_input(self):
        '''
        This returns the true trajectories from the current sample as well as all
        other samples which had the same past trajectories. It should be used only
        in conjunction with *get_true_and_predicted_paths()*.

        Returns
        -------
        Path_true_all : np.ndarray
            This is the true observed trajectory of the agents, in the form of a
            :math:`\{N_{subgroups} \times N_{same} \times N_{agents} \times N_{O} \times 2\}` 
            dimensional numpy array with float values. If an agent is fully or on some 
            timesteps partially not observed, then this can include np.nan values. It
            must be noted that :math:`N_{same}` is the maximum number of similar samples,
            so for a smaller number, there will also be np.nan values.
        Subgroup_ind : np.ndarray
            This is a :math:`N_{samples}` dimensional numpy array with int values. 
            All samples with the same value belong to a group with the same corresponding
            input. This can be used to avoid having to evaluate the same metric values
            for identical samples. It must however be noted, that due to randomness in 
            the model, the predictions made for these samples might differ.
            
            The value in this array will indicate which of the entries of **Path_true_all**
            should be chosen.

        '''
        # Get the indentical input data
        self.data_set._extract_identical_inputs(eval_pov = self.get_output_type() == 'path_all_wi_pov', file_index = self.file_index)
        available_subgroup = self.data_set.unique_subgroups

        # Get useful samples
        self.model._transform_predictions_to_numpy(self.Index_curr, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov')
        Pred_step = self.model.Pred_step

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        # Get the used subgroups
        Use_subgroups = self.data_set.Subgroups[self.Index_curr[Use_samples]]
        Subgroup_unique, Subgroup = np.unique(Use_subgroups, return_inverse = True)
        
        # Get the index of Subgroup_unique in available_subgroup
        subgroup_ind = self.data_set.get_indices_1D(Subgroup_unique, available_subgroup)

        Path_true_all = self.data_set.Path_true_all[subgroup_ind]
        
        return Path_true_all, Subgroup
    
    
    def get_true_likelihood(self, joint_agents = True):
        '''
        This return the probabilities asigned to ground truth trajectories 
        according to a Gaussian KDE method fitted to the ground truth samples
        with an identical inputs.
        

        Parameters
        ----------
        joint_agents : bool, optional
            This says if the probabilities for the predicted trajectories
            are to be calcualted for all agents jointly. If this is the case,
            then, :math:`N_{agents}` in the output is 1. The default is True.

        Returns
        -------
        KDE_true_log_prob_true : np.ndarray
            This is a :math:`\{N_{samples} \times 1 \times N_{agents}\}`
            array that includes the probabilites for the true observations according 
            to the KDE model trained on the grouped true trajectories.
            
        KDE_true_log_prob_pred : np.ndarray
            This is a :math:`\{N_{samples} \times N_{preds} \times N_{agents}\}`
            array that includes the probabilities for the predicted trajectories
            according to the KDE model trained on the grouped true trajectories.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'

        if joint_agents:
            self.data_set._get_joint_KDE_probabilities(self.get_output_type() == 'path_all_wo_pov', self.file_index)
            self.model._get_joint_KDE_true_probabilities(self.Index_curr, self.Output_path_pred, 
                                                         self.get_output_type() == 'path_all_wo_pov')
            
            KDE_true_log_prob_true = self.data_set.Log_prob_true_joint[:,np.newaxis,np.newaxis]
            KDE_true_log_prob_pred = self.model.Log_prob_true_joint_pred[:,:,np.newaxis]
            
        else:
            self.data_set._get_indep_KDE_probabilities(self.get_output_type() == 'path_all_wo_pov', self.file_index)
            self.model._get_indep_KDE_true_probabilities(self.Index_curr, self.Output_path_pred, 
                                                         self.get_output_type() == 'path_all_wo_pov')
            
            KDE_true_log_prob_true = self.data_set.Log_prob_true_indep[:,np.newaxis,:]
            KDE_true_log_prob_pred = self.model.Log_prob_true_indep_pred

        # Get the actual data based on the evaluated file
        Index_curr_data = self.Index_curr
        if not self.data_set.data_in_one_piece:
            used_index = np.where(self.data_set.Domain.file_index == self.file_index)[0]
            Index_curr_data = self.data_set.get_indices_1D(Index_curr_data, used_index)

        num_agents_pred = KDE_true_log_prob_pred.shape[-1]
        

        # KDE_true_log_prob_true is calculated for all samples is file
        # It therefore must be reduced to Pred_index, and the number of agents
        # might need to be reduced to the one in KDE_true_log_prob_pred
        KDE_true_log_prob_true = KDE_true_log_prob_true[Index_curr_data, :, :num_agents_pred]
        KDE_true_log_prob_pred = KDE_true_log_prob_pred
        
        # Get useful samples
        self.model._transform_predictions_to_numpy(self.Index_curr, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov')
        Pred_step = self.model.Pred_step

        # Get samples where a prediction is actually useful
        Use_samples = Pred_step.any(-1).any(-1)

        KDE_true_log_prob_true = KDE_true_log_prob_true[Use_samples]
        KDE_true_log_prob_pred = KDE_true_log_prob_pred[Use_samples]

        return KDE_true_log_prob_true, KDE_true_log_prob_pred
        
        
    def get_KDE_probabilities(self, joint_agents = True):
        '''
        This return the probabilities asigned to trajectories according to 
        a Gaussian KDE method.

        Parameters
        ----------
        joint_agents : bool, optional
            This says if the probabilities for the predicted trajectories
            are to be calcualted for all agents jointly. If this is the case,
            then, :math:`N_{agents}` in the output is 1. The default is True.

        Returns
        -------
        KDE_pred_log_prob_true : np.ndarray
            This is a :math:`\{N_{samples} \times 1 \times N_{agents}\}`
            array that includes the probabilites for the true observations according to
            the KDE model trained on the predicted trajectories.
        KDE_pred_log_prob_pred : np.ndarray
            This is a :math:`\{N_{samples} \times N_{preds} \times N_{agents}\}`
            array that includes the probabilites for the predicted trajectories 
            according to the KDE model trained on the predicted trajectories.

        '''
        assert self.get_output_type()[:4] == 'path', 'This is not a path prediction metric.'
        
        # Get useful samples
        self.model._transform_predictions_to_numpy(self.Index_curr, self.Output_path_pred, 
                                                   self.get_output_type() == 'path_all_wo_pov')
        Pred_step = self.model.Pred_step # num_samples x num_agents x num_O
        Pred_agent = Pred_step.any(-1) # num_samples x num_agents


        # Check if the model can predict likelihoods
        model_predicts_probs = self.model.predict_path_probs
        
        # Check if the model actually predicted useful values
        if model_predicts_probs:
            # Test the provided probabilities
            Output_path_probs = self.Output_path_pred_probs[self.data_set.Agents].to_numpy() # [num_samples x num_agents_all] x [(num_preds + 1)]
            
            # Adjust sorting
            agents_id = self.model.Pred_agent_id # num_samples x num_agents_all
            sample_id = np.tile(np.arange(len(agents_id))[:,np.newaxis], (1, agents_id.shape[1])) # num_samples x num_agents_all
            Output_path_probs = Output_path_probs[sample_id, agents_id]  # [num_samples x num_agents] x [(num_preds + 1)]
            
            Output_path_probs_useful = np.stack(list(Output_path_probs[Pred_agent]), axis = 0).astype(np.float32) # num_pred_agents x num_preds
            
            # Only use predicted values if they are useful
            model_predicts_probs = np.isfinite(Output_path_probs_useful).all() # Check if the given values are useful
        
        # Use model predicted values.
        if model_predicts_probs:
            Output_path_probs_array = np.full((*agents_id.shape, Output_path_probs_useful.shape[-1]), np.nan, np.float32) # num_samples x num_agents x num_preds
            Output_path_probs_array[Pred_agent] = Output_path_probs_useful
            
            # Check if values are identical across agents, to see if it was a joint probability
            Diff = np.abs(Output_path_probs_array[:,[0]] - Output_path_probs_array)
            joint_samples_given = np.nanmax(Diff) < 1e-3
            
            # Reorder the array to nmathc required output 
            Output_path_probs_array = Output_path_probs_array.transpose(0,2,1) # num_Samples x num_preds x num_agents
            
            # Fit the output to desired joint/marginal value
            # If model joint, output joint       => take the first agent (contains allready joint prob)
            # If model marginal, output joint    => sum all agents (this are actually log prob values)
            # If model joint, output marginal    => divide by number of actual agents (this are actually log prob values)
            # If model marginal, output marginal => take the values as they are (this are actually log prob values)
            if joint_agents:
                if joint_samples_given:
                    # First agent shoudl always be defined
                    KDE_pred_log_prob = Output_path_probs_array[...,[0]] # num_samples x num_preds x 1
                else:
                    # For marginal predictin models, assume independence, so we can add stuff together
                    KDE_pred_log_prob = np.nansum(Output_path_probs_array, axis = -1, keepdims=True) # num_samples x num_preds x 1
            
            else:
                if joint_samples_given:
                    # Assume equal contribution from each sample
                    KDE_pred_log_prob = Output_path_probs_array / Pred_agent.sum(-1)[:,np.newaxis, np.newaxis] # num_samples x num_preds x num_agents
                else:
                    KDE_pred_log_prob = Output_path_probs_array # num_samples x num_preds x num_agents
            
            # Divide into log probs of predictions and ground truths
            KDE_pred_log_prob_pred = KDE_pred_log_prob[:,:-1]
            KDE_pred_log_prob_true = KDE_pred_log_prob[:,[-1]]
            
        else:
            if joint_agents:
                self.model._get_joint_KDE_pred_probabilities(self.Index_curr, self.Output_path_pred, 
                                                                self.get_output_type() == 'path_all_wo_pov')
                    
                KDE_pred_log_prob_true = self.model.Log_prob_joint_true[:,:,np.newaxis] # num_samples x 1 x 1
                KDE_pred_log_prob_pred = self.model.Log_prob_joint_pred[:,:,np.newaxis] # num_Samples x num_preds x 1
                
            else:
                self.model._get_indep_KDE_pred_probabilities(self.Index_curr, self.Output_path_pred, 
                                                            self.get_output_type() == 'path_all_wo_pov')
                
                KDE_pred_log_prob_true = self.model.Log_prob_indep_true # num_samples x 1 x num_agents
                KDE_pred_log_prob_pred = self.model.Log_prob_indep_pred # num_Samples x num_preds x num_agents

        # Get the KDE probabilities corresponding to the selected trajectory samples        
        if not hasattr(self, 'pred_idx'):
            self.pred_index = np.arange(self.data_set.num_samples_path_pred)

        KDE_pred_log_prob_pred = KDE_pred_log_prob_pred[:, self.pred_idx]

        # Get samples where a prediction is actually useful
        Use_samples = Pred_agent.any(-1)

        KDE_pred_log_prob_true = KDE_pred_log_prob_true[Use_samples]
        KDE_pred_log_prob_pred = KDE_pred_log_prob_pred[Use_samples]

        return KDE_pred_log_prob_true, KDE_pred_log_prob_pred

        
    #########################################################################################
    #########################################################################################
    ###                                                                                   ###
    ###                    Evaluation metric dependend functions                          ###
    ###                                                                                   ###
    #########################################################################################
    #########################################################################################
    

    def get_name(self = None):
        r'''
        Provides a dictionary with the different names of the evaluation metric.
            
        Returns
        -------
        names : dict
          The first key of names ('print')  will be primarily used to refer to the evaluation metric in console outputs. 
                
          The 'file' key has to be a string that does not include any folder separators 
          (for any operating system), as it is mostly used to indicate that certain result files belong to this evaluation metric. 
                
          The 'latex' key string is used in automatically generated tables and figures for latex, and can there include 
          latex commands - such as using '$$' for math notation.
            
        '''
        raise AttributeError('Has to be overridden in actual metric class')
        
    def setup_method(self):
        # Will do any preparation the method might require, like calculating
        # weights.
        # creates:
            # self.weights_saved -  The weights that were created for this metric,
            #                       will be in the form of a list
        raise AttributeError('Has to be overridden in actual metric class.')
        
    def requires_preprocessing(self):
        # Returns a boolean output, True if preprocesing of true output
        # data for the calculation of weights is required, which might be 
        # avoided in repeated cases
        raise AttributeError('Has to be overridden in actual metric class.')
        
    
    def get_output_type(self = None):
        # Should return 'class', 'class_and_time', 'path_tar', 'path_all'
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def check_applicability(self):
        # Provides feedback on if a metric can be used, as it might be
        # related to only specifc datasets/scenarios/models/etc.
        # Returns None if metric is unrestricedly applicable.
        raise AttributeError('Has to be overridden in actual metric class.')
        
    
    def get_opt_goal(self = None):
        # Should return 'minimize' or 'maximize'
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def metric_boundaries(self = None):
        # Should return a list with two entries. These are the minimum and 
        # maximum possible values. If no such boundary on potential metric values 
        # exists, then those values should be none instead
        raise AttributeError('Has to be overridden in actual metric class')
    

    def partial_calculation(self = None):
        # This function returns the way that the metric work over subparts of the dataset.
        options = ['No', 'Subgroups', 'Sample', 'Subgroup_pred_agents', 'Pred_agents']
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def evaluate_prediction_method(self):
        # Takes true outputs and corresponding predictions to calculate some
        # metric to evaluate a model
        raise AttributeError('Has to be overridden in actual metric class.')
        # return results # results is a list

    def combine_results(self, result_lists, weights):
        r'''
        This function combines partial results.

        Parameters
        ----------
        result_lists : list
            A list of lists, which correpond to multiple outputs of *evaluate_prediction_method()*.

        weights : list
            A list of the same length as result_lists, which contains the weights based on the method in
            *self.partial_calculation()*, which might be useful.


        Returns
        -------
        results : list
            This is a list with more than one entry. The first entry must be a scalar, which allows the comparison
            of different models according to that metric. Afterwards, any information can be saved which might
            be useful for later visualization of the results, if this is so desired.
        

        '''

        raise AttributeError('Has to be overridden in actual metric class.')
        #  return results
    
    
    def is_log_scale(self = None):
        # Should return 'False' or 'True'
        raise AttributeError('Has to be overridden in actual metric class')
        
        
    def allows_plot(self):
        # Returns a boolean output, True if a plot can be created, False if not.
        raise AttributeError('Has to be overridden in actual metric class.')
        
        
    def create_plot(self, results, test_file, fig, ax, save = False, model_class = None):
        '''
        This function creates the final plot.
        
        This function is cycled over all included models, so they can be combined
        in one figure. However, it is also possible to save a figure for each model,
        if so desired. In that case, a new instanc of fig and ax should be created and
        filled instead of the ones passed as parameters of this functions, as they are
        shared between all models.
        
        If only one figure is created over all models, this function should end with:
            
        if save:
            ax.legend() # Depending on if this is desired or not
            fig.show()
            fig.savefig(test_file, bbox_inches='tight')  

        Parameters
        ----------
        results : list
            This is the list produced by self.evaluate_prediction_method().
        test_file : str
            This is the location at which the combined figure of all models can be
            saved (it ends with '.pdf'). If one saves a result for each separate model, 
            one should adjust the filename to indicate the actual model.
        fig : matplotlib.pyplot.Figure
            This is the overall figure that is shared between all models.
        ax : matplotlib.pyplot.Axes
            This is the overall axes that is shared between all models.
        save : bool, optional
            This is the trigger that indicates if one currently is plotting the last
            model, which should indicate that the figure should now be saved. The default is False.
        model_class : Framework_Model, optional
            The model for which the current results were calculated. The default is None.

        Returns
        -------
        None.

        '''
        # Function that visualizes result if possible
        if self.allows_plot():
            raise AttributeError('Has to be overridden in actual metric class.')
        else:
            pass
