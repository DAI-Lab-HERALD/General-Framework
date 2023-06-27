#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:39:00 2022

@author: anna
"""
import argparse
import numpy as np
import os
import pickle
import sys

from loguru import logger
from tracks_import import read_from_csv
#%%
sceneID = []
trajectories = []
trajectories_Vis = []
background_image_paths = []

sceneID_smaller = []
trajectories_smaller = []
trajectories_Vis_smaller = []
background_image_paths_smaller = []

#%%
for d_id in range(33):
    def_dataset_dir = '../inD-dataset-v1.0/data'
    def_dataset = 'ind'
    def_recording = str(d_id)
    
    def create_args():
        cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")
        # --- Input ---
        cs.add_argument('--dataset_dir', default=def_dataset_dir,
                        help="Path to directory that contains the dataset csv files.", type=str)
        cs.add_argument('--dataset', default=def_dataset,
                        help="Name of the dataset. Needed to apply dataset specific visualization adjustments.",
                        type=str)
        cs.add_argument('--recording', default=def_recording,
                        help="Name of the recording given by a number with a leading zero.", type=str)
        cs.add_argument('--visualizer_params_dir', default="../data/visualizer_params/",
                        help="Name of the recording given by a number with a leading zero.", type=str)
    
        # --- Visualization settings ---
        cs.add_argument('--playback_speed', default=4,
                        help="During playback, only consider every nth frame. This option also applies to the outer"
                             "backward/forward jump buttons.",
                        type=int)
        cs.add_argument('--suppress_track_window', default=False,
                        help="Do not show the track window when clicking on a track. Only surrounding vehicle colors are"
                             " displayed.",
                        type=str2bool)
        cs.add_argument('--show_bounding_box', default=True,
                        help="Plot the rotated bounding boxes of all vehicles. Please note, that for vulnerable road users,"
                             " no bounding box is given.",
                        type=str2bool)
        cs.add_argument('--show_orientation', default=False,
                        help="Indicate the orientation of all vehicles by triangles.",
                        type=str2bool)
        cs.add_argument('--show_trajectory', default=False,
                        help="Show the trajectory up to the current frame for every track.",
                        type=str2bool)
        cs.add_argument('--show_future_trajectory', default=False,
                        help="Show the remaining trajectory for every track.",
                        type=str2bool)
        cs.add_argument('--annotate_track_id', default=True,
                        help="Annotate every track by its id.",
                        type=str2bool)
        cs.add_argument('--annotate_class', default=False,
                        help="Annotate every track by its class label.",
                        type=str2bool)
        cs.add_argument('--annotate_speed', default=False,
                        help="Annotate every track by its current speed.",
                        type=str2bool)
        cs.add_argument('--annotate_orientation', default=False,
                        help="Annotate every track by its current orientation.",
                        type=str2bool)
        cs.add_argument('--annotate_age', default=False,
                        help="Annotate every track by its current age.",
                        type=str2bool)
        cs.add_argument('--show_maximized', default=False,
                        help="Show the track Visualizer maximized. Might affect performance.",
                        type=str2bool)
    
        return vars(cs.parse_args())
    
    
    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    
    config = create_args()
    
    dataset_dir = config["dataset_dir"] + "/"
    recording = config["recording"]
    
    if recording is None:
        logger.error("Please specify a recording!")
        sys.exit(1)
    
    recording = "{:02d}".format(int(recording))
    
    logger.info("Loading recording {} from dataset {}", recording, config["dataset"])
    
    # Create paths to csv files
    tracks_file = dataset_dir + recording + "_tracks.csv"
    tracks_meta_file = dataset_dir + recording + "_tracksMeta.csv"
    recording_meta_file = dataset_dir + recording + "_recordingMeta.csv"
    
    # Load csv files
    logger.info("Loading csv files {}, {} and {}", tracks_file, tracks_meta_file, recording_meta_file)
    tracks, static_info, meta_info = read_from_csv(tracks_file, tracks_meta_file, recording_meta_file,
                                                   include_px_coordinates=True)
    
    # Load background image for visualization
    background_image_path = dataset_dir + recording + "_background.png"
    if not os.path.exists(background_image_path):
        logger.warning("Background image {} missing. Fallback to using a black background.", background_image_path)
        background_image_path = None
    config["background_image_path"] = background_image_path
    
    # print(meta_info['locationId'])
    # print(meta_info['orthoPxToMeter'])
    
    #%% Code for generating the large dataset (sliding along the entire trajectory observed for an agent as long as a trajectory of 200 timesteps is obtainable)
    # always_observed = all((l < m) and (m-l==1) for l, m in zip(tracks[i]['trackLifetime'], tracks[i]['trackLifetime'][1:]))
    # for i in range(len(static_info)):
        
    #     if static_info[i]['numFrames']>=200 and static_info[i]['class']=='car':
    #         traj = np.stack((tracks[i]['xCenter'], tracks[i]['yCenter'])).transpose()
    #         trajVis = np.stack((tracks[i]['xCenterVis'], tracks[i]['yCenterVis'])).transpose()
    #         track_life = tracks[i]['trackLifetime']
    #         for j in range(len(traj)-200):
    #             always_observed = all((l < m) and (m-l==1) for l, m in zip(track_life[j:200+j], track_life[j+1:200+j+1]))
                
    #             if always_observed:
    #                 eightSec_traj = traj[j:200+j]
    #                 eightSec_trajVis = trajVis[j:200+j]
                    
    #                 trajectories.insert(len(trajectories), eightSec_traj)
    #                 trajectories_Vis.insert(len(trajectories_Vis), eightSec_trajVis)
    #                 background_image_paths.insert(len(background_image_paths), background_image_path)
    #                 sceneID.insert(len(sceneID), meta_info['locationId'])
            
    # %% Code for generating the smaller dataset
    for i in range(len(static_info)):
        print(static_info[i]['class'])
        # For extracting only vehicles
        if static_info[i]['numFrames']>=200 and (static_info[i]['class']=='car' or static_info[i]['class']=='truck_bus'):
        # The commented out line is the condition for extracting all the agents
        # if static_info[i]['numFrames']>=200:
            traj = np.stack((tracks[i]['xCenter'], tracks[i]['yCenter'])).transpose()
            trajVis = np.stack((tracks[i]['xCenterVis'], tracks[i]['yCenterVis'])).transpose()
            track_life = tracks[i]['trackLifetime']
            
            always_observed = all((l < m) and (m-l==1) for l, m in zip(tracks[i]['trackLifetime'], tracks[i]['trackLifetime'][1:]))
                
            if always_observed:
                eightSec_traj = traj[:200]
                eightSec_trajVis = trajVis[:200]
                
                trajectories_smaller.insert(len(trajectories_smaller), eightSec_traj)
                trajectories_Vis_smaller.insert(len(trajectories_Vis_smaller), eightSec_trajVis)
                background_image_paths_smaller.insert(len(background_image_paths_smaller), background_image_path)
                sceneID_smaller.insert(len(sceneID_smaller), meta_info['locationId'])
            
#%% Save data
pickle.dump(sceneID_smaller, open('inD_sceneID_smaller_vehicles_8s', 'wb'))
pickle.dump(trajectories_smaller, open('inD_trajectories_smaller_vehicles_8s', 'wb'))
pickle.dump(trajectories_Vis_smaller, open('inD_trajectories_Vis_smaller_vehicles_8s', 'wb'))
pickle.dump(background_image_paths_smaller, open('inD_backgrnd_img_paths_smaller_vehicles_8s', 'wb'))