import os
import shutil

# Get the current path
path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)) + os.sep + 'Results' + os.sep 


# Define replacements to look for
old = 'Pertubation_000'
new = 'Pertubation_013'

# Add certain parts (to prevent accidental doubbleings)
prefix = '--'
suffix = '--agents_0--identi_split_0_pert=0--t_pp_old_0_0'

old_term = prefix + old + suffix
new_term = prefix + new + suffix
 
data_set_folders = os.listdir(path)
for data_set_folder in data_set_folders: 
    folder_path = path + data_set_folder + os.sep + 'Models' + os.sep
    model_files = os.listdir(folder_path)
    for model_file in model_files:
        model_file_split = model_file.split(old_term)
        if len(model_file_split) > 1:
            model_file_new = new_term.join(model_file_split)
            if model_file_new in model_files:
                continue
            
            old_file = folder_path + model_file
            new_file = folder_path + model_file_new
            if os.path.isfile(new_file):
                print('Model already exists.')
            else:
                shutil.copyfile(old_file, new_file)