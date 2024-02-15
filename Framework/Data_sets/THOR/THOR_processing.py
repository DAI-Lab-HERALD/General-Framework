import csv
import pandas as pd
import numpy as np
import os
import glob
import json

path = os.path.dirname(os.path.realpath(__file__))

#%%
dataset_paths = ["Experiment_1",
                 "Experiment_2",
                 "Experiment_3"]

# dataset_paths = ["Experiment_1"]

framerate = 100

frame_reduction = 10

Final_data = pd.DataFrame(np.zeros((1,5), object), columns = ['scenario', 'Id', 'First frame', 'Last frame', 'path'])
overall_id = 0
# Load each dataset separately; these will be concatenated later into one dataset
for dataset_path in dataset_paths:
    print("Loading dataset {}".format(dataset_path))
    data_path_expanded = path + os.sep + 'data' + os.sep + dataset_path + os.sep + '3D' + os.sep + '*.tsv'
    detection_paths = glob.glob(data_path_expanded)


    with open(path + os.sep + 'data' + os.sep + dataset_path + '.csv', 'w') as f:
        writer = csv.writer(f)

        MeterPerPx = 0.01
        x_center = -11000*MeterPerPx
        y_center = -10000*MeterPerPx

        writer.writerow(['MeterPerPx', 'x_center', 'y_center', 'rot_angle'])
        writer.writerow([MeterPerPx, x_center, y_center, 0])
    
    ordered_detections = []
    for i, detection_path in enumerate(detection_paths):
        print("Dataset {}: Stamp {:4.0f}/{}".format(dataset_path, i + 1, len(detection_paths)))
        
        measurements = []
        columns = ['Frame', 'Time']
        dtype = {'Frame': int, 'Time': float}
        num_markers = [4,4,5,5,5,5,5,4,4,4,5]
        if dataset_path == 'Experiment_2':
            num_agents = 11
        else:
            num_agents = 10

        for i in range(1, num_agents+1):
            for j in range(1, num_markers[i-1]+1):
                columns.extend([f'Agent {i} X Marker {j}', f'Agent {i} Y Marker {j}', f'Agent {i} Z Marker {j}'])

                dtype.update({f'Agent {i} X Marker {j}': float})
                dtype.update({f'Agent {i} Y Marker {j}': float})
                dtype.update({f'Agent {i} Z Marker {j}': float})


        with open(detection_path) as f:
            for i, line in enumerate(f):
                if i >= 11:
                    measurement = line.split('\t')
                    if len(measurement) > 137 and dataset_path != 'Experiment_2':
                        print(dataset_path)
                        print("Getting relevant measurements")
                        measurement = measurement[:137]
                    measurements.append(measurement)


        detection = pd.DataFrame(measurements, columns=columns).astype(dtype)

        # Obtain Agent positions by averaging the marker positions
        
        for i in range(1, num_agents+1):
            detection[f'Agent {i} X'] = detection[[f'Agent {i} X Marker {j}' for j in range(1, num_markers[i-1]+1)]].mean(axis=1)
            detection[f'Agent {i} Y'] = detection[[f'Agent {i} Y Marker {j}' for j in range(1, num_markers[i-1]+1)]].mean(axis=1)
            detection[f'Agent {i} Z'] = detection[[f'Agent {i} Z Marker {j}' for j in range(1, num_markers[i-1]+1)]].mean(axis=1)

        
        
        for agent in range(1, num_agents+1):
            observed_detections = detection[detection[f'Agent {agent} X'] != 0]
                
            # Identify jumps in indices
            jumps = observed_detections.index.to_series().diff().fillna(1) != 1

            # Group consecutive indices into segments
            segments = jumps.cumsum()

            # Analyze each segment of continuous data
            for segment_id, segment_data in observed_detections.groupby(segments):
                print(f"Segment {segment_id}")

                frame = segment_data['Frame'].reset_index(drop=True).iloc[::frame_reduction].reset_index(drop=True)
                frame = frame.to_frame()
                frame['Frame'] = frame.index
                frame = frame.melt()['value']
                frame.name = 'Frame'

                track = pd.DataFrame(np.zeros((len(frame),4), object), columns = ['frame', 't', 'x', 'y'])

                track.frame = frame
                track.t = segment_data['Time'].reset_index(drop=True).iloc[::frame_reduction].reset_index(drop=True)
                track.x = segment_data[f'Agent {agent} X'].reset_index(drop=True).iloc[::frame_reduction].reset_index(drop=True)*MeterPerPx
                track.y = segment_data[f'Agent {agent} Y'].reset_index(drop=True).iloc[::frame_reduction].reset_index(drop=True)*MeterPerPx
                track = track.set_index('frame')
                
                data = pd.Series(np.zeros((5), object), index = ['scenario', 'Id', 'First frame', 'Last frame', 'path'])
                if agent == 10:
                    data.Id             = 'Velodyne'
                elif agent == 11:
                    data.Id             = 'Robot'
                else:
                    data.Id             = agent + segment_id*10

                data['First frame'] = frame[0]
                data['Last frame']  = frame[len(frame)-1]
                data.scenario       = dataset_path
                data.path           = track
                
                Final_data.loc[overall_id] = data
                overall_id += 1
            
    
# do tests
    for index, data in Final_data[Final_data.scenario == dataset_path].iterrows():
        data.path = data.path.sort_index()
        Final_data.loc[index] = data
        assert len(data.path.index) == (1 + data['Last frame'] - data['First frame']), "Gaps in data"


        
Final_data.to_pickle(path + os.sep + "THOR_processed.pkl")   
    