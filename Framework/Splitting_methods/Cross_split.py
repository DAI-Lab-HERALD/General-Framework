import pandas as pd
import numpy as np
import os
from splitting_template import splitting_template


class Cross_split(splitting_template):
    '''
    This splitting method implements one of the standard methods of evaluating
    prediction model, the method of crossvalidation. Here, the dataset is split
    into a number of roughly similar sized partitions, which take turns as the
    testing set, while the respective other partitions are used as the 
    training set. 
    
    The number of possible repetitions depends on the chosen size of each partitition,
    with smaller test sets allowing for more repetitions.
    '''
    def split_data_method(self):
        num_splits = int(np.ceil(1 / self.test_part))
        
        # Get identical input cases
        self.data_set._group_indentical_inputs()
        Subgroups = self.data_set.Subgroups - 1
        
        # Get Behaviors, with non appearing ones being neglected
        if self.data_set.classification_possible:
            if self.data_set.data_in_one_piece:
                Behaviors = self.data_set.Output_A.to_numpy().argmax(1)
            else:
                Behaviors = np.zeros(len(self.Domain), int)
                for file_index in range(len(self.data_set.Files)):
                    file = self.data_set.Files[file_index] + '.npy'
                    used = self.Domain.file_index == file_index
                    used_index = np.where(used)[0]

                    [_, _, _, _, _, _, Output_A, _, _] = np.load(file, allow_pickle=True)
                    
                    # Get the actual indices of Output_A
                    Output_A = Output_A.loc[self.Domain[used].Index_saved]
                    
                    # Map behaviors from Output_A column to self.data_set.Behaviors
                    index_map = np.array(Output_A.columns)[:,np.newaxis] == self.data_set.Behaviors[np.newaxis]
                    assert (index_map.sum(1) == 1).all()
                    index_map = index_map.argmax(1)
                    
                    # Get the actual behaviors
                    beh_ind = Output_A.to_numpy().argmax(1)
                    Behaviors[used_index] = index_map[beh_ind]
                    
            Behaviors = np.unique(Behaviors, return_inverse = True)[1]
        else:
            Behaviors = np.zeros(len(self.Domain))
        
        # Get unique subgroups
        uni_subgroups = np.unique(Subgroups)
        assert len(uni_subgroups) > num_splits, "Not enough unique input conditions for the desired number of splits."
        
        # Get number of behaviors for each subgroup
        uni_subgroups_beh = np.zeros((len(uni_subgroups), Behaviors.max() + 1))
        for ind, subgroup in enumerate(uni_subgroups):
            subgroup_beh = Behaviors[Subgroups == subgroup]
            beh_included, beh_num = np.unique(subgroup_beh, return_counts = True)
            
            uni_subgroups_beh[ind, beh_included] = beh_num
        
        desired_beh = uni_subgroups_beh.sum(0, keepdims = True) / num_splits
        
        # Sort by overall number of samples
        sort_ind = np.argsort(-uni_subgroups_beh.sum(1))
        
        # Prepare subgroup
        sort_subgroups_beh = np.zeros((num_splits, uni_subgroups_beh.shape[1]))
        
        splitcase = np.ones(len(uni_subgroups_beh)) * -1
        Splitcase = np.ones(len(Subgroups))
        
        for ind in sort_ind:
            subgroup_beh_pot = sort_subgroups_beh + uni_subgroups_beh[ind]
            case_loss_with = ((desired_beh - subgroup_beh_pot) ** 2).sum(1)
            case_loss_without = ((desired_beh - sort_subgroups_beh) ** 2).sum(1)
            
            loss_decrease = case_loss_without - case_loss_with
            
            best_case = np.argmax(loss_decrease)
            
            # update current collection
            sort_subgroups_beh[best_case] += uni_subgroups_beh[ind]
            
            
            splitcase[ind] = best_case
            Splitcase[Subgroups == uni_subgroups[ind]] = best_case
        
        assert Splitcase.min() >= 0
        
        Situations_test = (Splitcase[:,np.newaxis] == np.array(self.repetition)[np.newaxis]).any(1)
        
        Index = np.arange(len(Subgroups))
        Train_index = Index[~Situations_test]
        Test_index  = Index[Situations_test]
        
        return Train_index, Test_index
    
    def get_name(self):
        num_splits = int(np.ceil(1 / self.test_part))
        rep_str = str(self.repetition)[1:-1]
        names = {'print': '{} fold Cross validation (Splits '.format(num_splits) + rep_str + ')',
                 'file': 'crossv_split',
                 'latex': r'CV - Fold ' + rep_str}
        return names
        
    def check_splitability_method(self):
        return None
    
    
    def repetition_number(self):
        num_splits = int(np.ceil(1 / self.test_part))
        return num_splits
    
    
    def can_process_str_repetition(self = None):
        return False
        
        
        
    
        



