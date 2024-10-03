import pandas as pd
import numpy as np
import os
from os import path

pd.options.mode.chained_assignment=None
n=1

### Prepare to get situation data, adjust values and position reference

local_path = path.dirname(path.realpath(__file__))

print('Preprocessing Data')
while path.exists(local_path + os.sep + 'data' + os.sep + '{}_recordingMeta.csv'.format(str(n).zfill(2))):
    Meta_data=pd.read_csv(local_path + os.sep + 'data' + os.sep + '{}_recordingMeta.csv'.format(str(n).zfill(2)))
    local_data=Meta_data[['id', 'locationId', 'numVehicles', 'speedLimit',
                          'upperLaneMarkings', 'lowerLaneMarkings']].copy(deep=True)
    upper_Lane=np.array(local_data['upperLaneMarkings'][0].split(';')).astype('float')    
    
    if len(upper_Lane)>4:
        new_upper_Lane=upper_Lane-upper_Lane[1]
    else:
        new_upper_Lane=upper_Lane-upper_Lane[0]
    
    lower_Lane = np.array(local_data['lowerLaneMarkings'][0].split(';')).astype('float')
    if len(lower_Lane)>4:
        new_lower_Lane = lower_Lane[-2]-lower_Lane
    else:
        new_lower_Lane = lower_Lane[-1]-lower_Lane
        
    
    local_data['upperLaneMarkings'][0] = upper_Lane  
    
    local_data['lowerLaneMarkings'][0] = lower_Lane
    
    if n==1:
        Scenario = local_data
    else:
        Scenario = pd.concat((Scenario, local_data),ignore_index=True)
    n=n+1



    
frame_addition = 0
id_addition = 0
Final_out = []
for n in range(1, 1 + len(Scenario)):
    print('Processing Scenario {}/{}'.format(str(n).zfill(2),len(Scenario)))
    # Load recoding data
    Track_data = pd.read_csv(local_path + os.sep + 'data' + os.sep + '{}_tracks.csv'.format(str(n).zfill(2)))
    Track_data = Track_data[['frame', 'id', 'x', 'y', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration',
                             'precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId',
                             'rightPrecedingId', 'rightAlongsideId','rightFollowingId', 'laneId']]
    
    Track_meta_data = pd.read_csv(local_path + os.sep + 'data' + os.sep + '{}_tracksMeta.csv'.format(str(n).zfill(2)))
    scenario = Scenario.iloc[n - 1]
    
    # Set up final data container for each vehicle
    Final = Track_meta_data[['id','width','height','class', 'numLaneChanges', 'drivingDirection']].copy(deep=True)
    Final['numMerges']       = '0'
    Final['locationId']      = scenario['locationId']
    Final['recordingId']     = scenario['id']
    Final['laneMarkings']    = '0'
    Final['track']           = '0'
    Final['frame_min']       = '0'
    Final['frame_max']       = '0'
    
    # get the maximum x position in the track
    max_x = np.max(Track_data['x'])
    
    num_lane_markers = len(scenario['upperLaneMarkings'])
    # get lane markers in upper lanes (where vehicles go from right to left)
    upper_Lane_Markers = scenario['upperLaneMarkings']
    # get lane markers in lower lanes (where vehicles go from left to right)
    lower_Lane_Markers = scenario['lowerLaneMarkings']

    # Switsch y coordinate system
    # Transform into standard coordinate system
    Track_data['y']             = -Track_data['y'] 
    Track_data['yVelocity']     = -Track_data['yVelocity']
    Track_data['yAcceleration'] = -Track_data['yAcceleration']
    upper_Lane_Markers          = -upper_Lane_Markers
    lower_Lane_Markers          = -lower_Lane_Markers
    
    # Prepare lane markings and track for final output
    max_frame = 0
    for loc_id in range(len(Final)):
        final = Final.iloc[loc_id]
        local_track = Track_data[Track_data['id'] == final['id']].copy(deep=True)
        # Adjust pos to be the middle of the vehicle
        local_track['x'] = local_track['x'] + 0.5 * final['width']
        local_track['y'] = local_track['y'] + 0.5 * final['height']
        

        
        max_frame = max(max_frame, local_track['frame'].max())
        final['frame_min'] = local_track['frame'].min()
        final['frame_max'] = local_track['frame'].max()
        
        
        # driving to the left in the upper part
        if Track_meta_data['drivingDirection'].iloc[loc_id] == 1: 
            if num_lane_markers>4:
                local_track['laneId'] = local_track['laneId']-2
            else:
                local_track['laneId'] = local_track['laneId']-1
                
            final['laneMarkings'] = upper_Lane_Markers
        
        # driving to the right in the lower
        else: 
            if num_lane_markers>4:
                local_track['laneId'] = 2 * num_lane_markers - local_track['laneId']
            else:
                local_track['laneId'] = 2 * num_lane_markers + 1 - local_track['laneId']
            
            final['laneMarkings'] = np.flip(lower_Lane_Markers)
        
        local_track.drop(columns = ['id'], inplace=True)
        local_track[['precedingId', 'followingId', 
                     'leftPrecedingId', 'leftAlongsideId','leftFollowingId',
                     'rightPrecedingId', 'rightAlongsideId','rightFollowingId']] += id_addition
        
        #set non existent vehicles to zero again.
        local_track[local_track[['precedingId', 'followingId', 'leftPrecedingId', 'leftAlongsideId','leftFollowingId',
                                 'rightPrecedingId', 'rightAlongsideId','rightFollowingId']] == id_addition] = 0
        
        local_track['frame'] = local_track['frame'] + frame_addition
        
        # reclassify merging from standard lane changing
        if local_track['laneId'].iloc[0] < 1:
            final['numMerges'] = np.int64(1)
            final['numLaneChanges'] = final['numLaneChanges'] - 1
        else:
            final['numMerges'] = np.int64(0)
            
        final['track'] = local_track
        final['id']    = final['id'] + id_addition
    
        Final.iloc[loc_id] = final
    
    Final['frame_min'] = Final['frame_min'] + frame_addition
    Final['frame_max'] = Final['frame_max'] + frame_addition
    
    frame_addition = frame_addition + max_frame
    id_addition    = id_addition + scenario['numVehicles']
    
    Final_out.append(Final.copy(deep=True))
   
print('Save Data')
Final_out = pd.concat(Final_out)
Final_out.to_pickle(local_path + os.sep + "highD_processed.pkl")