#%%
import copy
import csv
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object

import os

px = 256 * 1.28
# Draw scene
def generate_scimg(
    lanelet_network, now_point, theta, time_step, watch_radius=64, draw_shape=True
):
    """Generate image input for neural network

    Arguments:
        scenario {[Commonroad scenario]} -- [Scenario object from CommonRoad]
        now_point {[list]} -- [[x,y] coordinates of vehicle right now that will be predicted]
        theta {[float]} -- [orientation of the vehicle that will be predicted]
        time_step {[float]} -- [Global time step of scenario]

    Keyword Arguments:
        draw_shape {bool} -- [Draw shapes of dynamic obstacles in image] (default: {True})

    Returns:
        img_gray [np.array] -- [Black and white image with 256 x 256 pixels of the scene]
    """
    my_dpi = 300
    draw_fig = plt.figure(figsize=(px / my_dpi, px / my_dpi), dpi=my_dpi)
    

    if theta > 2 * np.pi:
        theta -= 2 * np.pi
    elif theta < -(2 * np.pi):
        theta += 2 * np.pi

    lanelet_network.translate_rotate(np.array(-now_point), -theta)

    draw_params = {
        "time_begin": time_step,
        "lanelet_network": {"traffic_light": {"draw_traffic_lights": False}},
    }

    draw_object(
        lanelet_network,
        draw_params=draw_params,
    )

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().set_aspect("equal")
    plt.xlim(-watch_radius, watch_radius)
    plt.ylim(-watch_radius, watch_radius)

    draw_fig.canvas.draw()
    plt.close(draw_fig)

    # convert canvas to image
    img = np.fromstring(draw_fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(draw_fig.canvas.get_width_height()[::-1] + (3,))

    img_gray = ~cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img_gray




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
    if not dataset_path.endswith('.xml'): continue
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

    draw_network = copy.deepcopy(scenario.lanelet_network)
    translation = np.array([0,0])
    theta = 0
    watch_radius = 0
    for i in range(len(draw_network.lanelets)):
        a=draw_network.lanelets[i]
        if watch_radius < np.max(np.abs(a.center_vertices)):
            watch_radius = np.max(np.abs(a.center_vertices))

    watch_radius = 256 #math.ceil(watch_radius / 10.0) * 10

    timestep = 0
    gray_img = generate_scimg(draw_network,
                            translation,
                            theta,
                            timestep,
                            watch_radius)
    
    
    cv2.imwrite(path + os.sep + 'data' + os.sep + data.scenario + '.png', gray_img)

    with open(path + os.sep + 'data' + os.sep + data.scenario + '.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerow(['MeterToPx', 'x_center', 'y_center', 'rot_angle'])
        writer.writerow([(watch_radius)/px, (px/2), -(px/2), 0])

        

# Save data
Final_data.to_pickle(path + os.sep + "CommonRoad_processed.pkl")

# scaling = px/(2*watch_radius)
# plt.plot((np.array(Final_data.loc[3].path[0:20].x))*(scaling)+(px/2), (-np.array(Final_data.loc[3].path[0:20].y))*(scaling)+(px/2))
