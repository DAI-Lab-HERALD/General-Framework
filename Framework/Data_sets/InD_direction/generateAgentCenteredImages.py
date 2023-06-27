#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:16:27 2022

@author: anna
"""

import cv2
import math
import matplotlib.image
import numpy as np
import pickle
import torch

from scipy import ndimage


def _rotate(x, x_t, angles_rad):
    c, s = torch.cos(angles_rad), torch.sin(angles_rad)
    c, s = c.unsqueeze(1), s.unsqueeze(1)
    x_center = x - x_t # translate
    x_vals, y_vals = x_center[:,:,0], x_center[:,:,1]
    new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
    new_y_vals = s * x_vals + c * y_vals # _rotate y
    x_center[:,:,0] = new_x_vals
    x_center[:,:,1] = new_y_vals
    return x_center + x_t # translate back

background_image_paths = pickle.load(open('inD_backgrnd_img_paths_smaller_vehicles_8s', 'rb'))
scene_ids = pickle.load(open('inD_sceneID_smaller_vehicles_8s', 'rb'))
trajectories = pickle.load(open('inD_trajectories_smaller_vehicles_8s', 'rb'))


trajectories = np.array(trajectories)


trajectories = trajectories[:,:200:5,:]

past_traj = np.concatenate((trajectories[:,:15,0],\
                                    trajectories[:,:15,1]), axis=1)
    
past_traj = torch.tensor(past_traj)
dist_around = 25 #10 #25
crop_img = False

for i in range(len(background_image_paths)):
    curr_past_traj = torch.concat((past_traj[i][0:15].unsqueeze(1), past_traj[i][15:].unsqueeze(1)), dim=1)/5
    
    # img = cv2.imread('./inD-dataset-v1.0/data_original/'+background_image_paths[i][25:], 0)
    img = cv2.imread('./inD-dataset-v1.0/data/'+background_image_paths[i][25:], 0)
    # img = np.array(img)#torch.tensor(img, dtype=torch.float)
    img = torch.tensor(img, dtype=torch.float)
   
    scale_down_factor = 12
    
    scene_id = scene_ids[i]
    
    flip_y = -1
    if scene_id == 1:
        orthoPxToMeter = 0.0081463609172491
    elif scene_id == 2:
        orthoPxToMeter = 0.008146360917245
    elif scene_id == 3:
        orthoPxToMeter = 0.0081463537957561
    elif scene_id == 4:
        orthoPxToMeter = 0.0126999352667008
        
    x_t = curr_past_traj[-1:,:].unsqueeze(0)
    
    x_t_rel = curr_past_traj[-1] - curr_past_traj[-2]
    rot_angles_rad = -1 * torch.atan2(x_t_rel[1], x_t_rel[0])
    rot_angles_deg = math.degrees(rot_angles_rad)
    
    # print(rot_angles_deg)
    curr_past_rot = _rotate(curr_past_traj.unsqueeze(0), x_t, rot_angles_rad.unsqueeze(0))
    curr_past_rot = curr_past_rot.squeeze(0)
    
    shift_amt = 0#100
    
    x_mid = max(0,int(np.floor(((curr_past_rot[-1][0])/orthoPxToMeter)/scale_down_factor)))
    y_mid = max(0,int(np.floor((flip_y*(curr_past_rot[-1][1])/orthoPxToMeter)/scale_down_factor)))
    
    img_to_rot = img#_to_rot[0]
    padX = [img_to_rot.shape[1] - x_mid, x_mid]
    padY = [img_to_rot.shape[0] - y_mid, y_mid]
    imgP = np.pad(img_to_rot, [padY, padX], 'constant')
    
    img_rot = ndimage.rotate(imgP, rot_angles_deg, reshape=False, axes=(1,0))
    
    img_rot = ndimage.shift(img_rot, shift=[shift_amt , 0])
    
    if padY[1]==0 and padX[1]>0:
        img_rot = img_rot[padY[0] : , padX[0] : -padX[1]]
    elif padY[1]>0 and padX[1]==0:
        img_rot = img_rot[padY[0] : -padY[1], padX[0] : ]
    elif padY[1]==0 and padX[1]==0:
        img_rot = img_rot[padY[0] : , padX[0] : ]
    else:
        img_rot = img_rot[padY[0] : -padY[1], padX[0] : -padX[1]]
    print(img_rot.shape)
    if crop_img:
        x_left = int(np.floor(((curr_past_rot[-1][0]-5)/orthoPxToMeter)/scale_down_factor))
        x_right = int(np.floor(((curr_past_rot[-1][0]+dist_around)/orthoPxToMeter)/scale_down_factor))
        
        y_down = int(np.floor(shift_amt +(flip_y*(curr_past_rot[-1][1]-dist_around)/orthoPxToMeter)/scale_down_factor))
        y_up = int(np.floor(shift_amt +(flip_y*(curr_past_rot[-1][1]+dist_around)/orthoPxToMeter)/scale_down_factor))
        
        img_rot = img_rot[y_up:y_down+1, x_left:x_right+1]
        padY = [int(np.floor((550-img_rot.shape[0])/2)), int(np.ceil((550-img_rot.shape[0])/2))]
        padX = [int(np.floor((350-img_rot.shape[1])/2)), int(np.ceil((350-img_rot.shape[1])/2))]
        img_rot = np.pad(img_rot, [padY, padX], 'constant')
    
    print(i)
    matplotlib.image.imsave('./inD_agentCenteredImgs_vehiclesfullScene/agentScene'+str(i)+'.png', img_rot)