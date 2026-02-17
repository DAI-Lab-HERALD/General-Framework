import os
import shutil

# Get the current path
path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)) + os.sep + 'Results' + os.sep 


# Define replacement strings to look for
old = '...'
new = '...'

# Add certain parts (to prevent accidental doubbleings)
prefix = '...'
suffix = '...'

old_term = prefix + old + suffix
new_term = prefix + new + suffix
 
data_set_folders = os.listdir(path)
for data_set_folder in data_set_folders: 
    folder_path = path + data_set_folder + os.sep + 'Models' + os.sep
    if not os.path.exists(folder_path):
        continue
    model_files = os.listdir(folder_path)
    for model_file in model_files:
        if old_term not in model_file:
            continue
        model_file_split = model_file.split(old_term)
        if len(model_file_split) > 1:
            model_file_new = new_term.join(model_file_split)
            old_file = folder_path + model_file
            new_file = folder_path + model_file_new
            if os.path.exists(new_file):
                print('Model already exists.')
            else:
                if os.path.isfile(old_file):
                    shutil.copyfile(old_file, new_file)
                else:
                    # Copy the whole folder
                    shutil.copytree(old_file, new_file)
