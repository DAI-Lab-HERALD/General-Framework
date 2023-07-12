import pandas as pd
import numpy as np
import os
import glob
import json


#%%
path = os.path.dirname(os.path.realpath(__file__))
dataset_paths = os.listdir(path + os.sep + 'data' + os.sep)

framerate = 2.5

Final_data = pd.DataFrame(np.zeros((1,5), object), columns = ['scenario', 'Id', 'First frame', 'Last frame', 'path'])
overall_id = 0
# Load each dataset separately; these will be concatenated later into one dataset
for dataset_path in dataset_paths:
    print("Loading dataset {}".format(dataset_path))
    data_path_expanded = path + os.sep + 'data' + os.sep + dataset_path + os.sep + '*.json'
    detection_paths = glob.glob(data_path_expanded)
    
    ordered_detections = []
    for i, detection_path in enumerate(detection_paths):
        if np.mod(i,10) == 0:
            print("Dataset {}: Stamp {:4.0f}/{}".format(dataset_path, i + 1, len(detection_paths)))
        detection_file = open(detection_path, 'r')
        detection = json.load(detection_file)
        detection_file.close()
        frame = detection['timestamp']
        for obj in detection['object_list']:
            # check if agent is already know
            Index = Final_data.index[(Final_data.scenario == dataset_path) & (Final_data.Id == obj['id'])]
            assert len(Index) < 2
            if len(Index) == 0:
                track = pd.DataFrame(np.zeros((1,4), object), columns = ['frame', 't', 'x', 'y'])
                track.frame = frame
                track.t = (frame - 1) / framerate
                track.x = obj['position'][0]
                track.y = obj['position'][1]
                track = track.set_index('frame')
                
                data = pd.Series(np.zeros((5), object), index = ['scenario', 'Id', 'First frame', 'Last frame', 'path'])
                data.Id             = obj['id']
                data['First frame'] = frame
                data['Last frame']  = frame
                data.scenario       = dataset_path
                data.path           = track
                
                Final_data.loc[overall_id] = data
                overall_id += 1
            else:
                index = Index[0]
                data = Final_data.loc[index]

                data['First frame'] = min(frame, data['First frame'])
                data['Last frame']  = max(frame, data['Last frame'])
                 
                data.path.loc[frame] = pd.Series(np.zeros(3), index = ['t', 'x', 'y'])
                data.path.loc[frame].t = (frame - 1) / framerate
                data.path.loc[frame].x = obj['position'][0]
                data.path.loc[frame].y = obj['position'][1]

                Final_data.loc[index] = data
    
# do tests
    for index, data in Final_data[Final_data.scenario == dataset_path].iterrows():
        data.path = data.path.sort_index()
        Final_data.loc[index] = data
        assert len(data.path.index) == (1 + data['Last frame'] - data['First frame']), "Gaps in data"
        
Final_data.to_pickle(path + os.sep + "FP_processed.pkl")   
    