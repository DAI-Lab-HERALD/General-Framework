#%%
import glob
import json
import numpy as np
import os
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))
dataset_paths = os.listdir(path + os.sep + 'data' + os.sep)
dataset_paths.sort()
# %%

## TODO check
framerate = 2.5

Final_data = pd.DataFrame(np.zeros((1,6), object), columns = ['scenario', 'Id', 'type', 'First frame', 'Last frame', 'path'])
overall_id = 0
# Load each dataset separately; these will be concatenated later into one dataset
for dataset_path in dataset_paths:
    if not dataset_path[-5:] == '.json':
        continue
    print("Loading dataset {}".format(dataset_path))
    data_path_expanded = path + os.sep + 'data' + os.sep + dataset_path
    detection_file = open(data_path_expanded, 'r')
    detection = json.load(detection_file)

    if dataset_path[0:3]=='zar' or dataset_path[0:3]=='hot' or dataset_path[0:3]=='eth':
        fps = 25.0
    else:
        fps = 30.0

    for i in range(len(detection)):
        frame = detection[i]['frame_id'] 
        # check if agent is already know
        Index = Final_data.index[(Final_data.scenario == dataset_path[:-5]) & (Final_data.Id == detection[i]['track_id'])]

        assert len(Index) < 2
        if len(Index) == 0:
            track = pd.DataFrame(np.zeros((1,4), object), columns = ['frame', 't', 'x', 'y'])
            track.frame = frame
            track.t = frame / fps #(frame - 1) / framerate
            track.x = detection[i]['bbox'][0] + detection[i]['bbox'][2] / 2
            track.y = detection[i]['bbox'][1] + detection[i]['bbox'][3] / 2
            track = track.set_index('frame')

            data = pd.Series(np.zeros((6), object), index = ['scenario', 'Id', 'type', 'First frame', 'Last frame', 'path'])
            data.Id             = detection[i]['track_id']
            data.type           = detection[i]['class_name'][0]
            data['First frame'] = frame
            data['Last frame']  = frame
            data.scenario       = dataset_path[:-5]
            data.path           = track

            Final_data.loc[overall_id] = data
            overall_id += 1

        else:
            index = Index[0]
            data = Final_data.loc[index]

            data['First frame'] = min(frame, data['First frame'])
            data['Last frame']  = max(frame, data['Last frame'])
            
            data.path.loc[frame] = pd.Series(np.zeros(3), index = ['t', 'x', 'y'])
            data.path.loc[frame].t = frame / fps #(frame - 1) / framerate
            data.path.loc[frame].x = detection[i]['bbox'][0] + detection[i]['bbox'][2] / 2
            data.path.loc[frame].y = detection[i]['bbox'][1] + detection[i]['bbox'][3] / 2

            Final_data.loc[index] = data 
    
    # do tests
    for index, data in Final_data[Final_data.scenario == dataset_path].iterrows():
        data.path = data.path.sort_index()
        Final_data.loc[index] = data
        assert len(data.path.index) == (1 + data['Last frame'] - data['First frame']), "Gaps in data"
        
#%%
Final_data.to_pickle(path + os.sep + "FP_comp_processed.pkl")   