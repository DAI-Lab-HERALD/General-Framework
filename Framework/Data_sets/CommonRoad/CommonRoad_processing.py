#%%
import numpy as np
import os
import pandas as pd

from commonroad.common.file_reader import CommonRoadFileReader

path = os.path.dirname(os.path.realpath(__file__))
dataset_paths = os.listdir(path + os.sep + 'data' + os.sep)

Final_data = pd.DataFrame(np.zeros((1,6), object), columns = ['scenario', 'Id', 'type', 'First frame', 'Last frame', 'path'])
overall_id = 0

agent_type_mapping = {
    'car': 'V',
    'pedestrian': 'P',
    'bicycle': 'B',
    'motorcycle': 'M',
    'truck': 'V',       # TODO give this and the following types their own marker when expanding agent types
    'bus': 'V',
    'priorityVechicle': 'V',
    'parkedVehicle': 'V',
    'train': 'V'
}

for dataset_path in dataset_paths:
    print("Loading dataset {}".format(dataset_path))
    data_path_expanded = path + os.sep + 'data' + os.sep + dataset_path
    scenario, planning_problem_set = CommonRoadFileReader(data_path_expanded).open()

    dt = scenario.dt

    scenario_name = scenario.benchmark_id #scenario.scenario_id.country_id + '_' + scenario.scenario_id.map_name
    detections = scenario.dynamic_obstacles
    for detection in detections:
    
        
        for i in range(detection.initial_state.time_step, detection.prediction.final_time_step + 1):
            Index = Final_data.index[(Final_data.scenario == scenario_name) & (Final_data.Id == detection.obstacle_id)]

            assert len(Index) < 2
            if len(Index) == 0:
                track = pd.DataFrame(np.zeros((1,4), object), columns = ['frame', 't', 'x', 'y'])
                track.frame = i
                track.t = i * dt
                track.x = detection.state_at_time(i).position[0]
                track.y = detection.state_at_time(i).position[1]
                track = track.set_index('frame')

                data = pd.Series(np.zeros((6), object), index = ['scenario', 'Id', 'type', 'First frame', 'Last frame', 'path'])
                data.Id             = detection.obstacle_id
                data.type           = agent_type_mapping.get(detection.obstacle_type.value, "Unknown")
                data['First frame'] = detection.initial_state.time_step
                data['Last frame']  = detection.prediction.final_time_step
                data.scenario       = scenario_name
                data.path           = track

                Final_data.loc[overall_id] = data
                overall_id += 1

            else:
                index = Index[0]
                data = Final_data.loc[index]

                data.path.loc[i] = pd.Series(np.zeros(3), index = ['t', 'x', 'y'])
                data.path.loc[i].t = i * dt
                data.path.loc[i].x = detection.state_at_time(i).position[0]
                data.path.loc[i].y = detection.state_at_time(i).position[1]

                Final_data.loc[index] = data

    # do tests
    for index, data in Final_data[Final_data.scenario == dataset_path].iterrows():
        data.path = data.path.sort_index()
        Final_data.loc[index] = data
        assert len(data.path.index) == (1 + data['Last frame'] - data['First frame']), "Gaps in data"
        

# %%
Final_data.to_pickle(path + os.sep + "CommonRoad_processed.pkl")