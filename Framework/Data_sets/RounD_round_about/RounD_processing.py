import pandas as pd
import numpy as np
from os import path

pd.options.mode.chained_assignment=None
n = 0

### Prepare to get situation data, adjust values and position reference

print('Preprocessing Data')
while path.exists('data/{}_recordingMeta.csv'.format(str(n).zfill(2))):
    Meta_data=pd.read_csv('data/{}_recordingMeta.csv'.format(str(n).zfill(2)))
    local_data=Meta_data[['locationId','speedLimit','orthoPxToMeter']].copy(deep=True)   
    
    if n==0:
        Scenario=local_data
    else:
        Scenario=Scenario.append(local_data,ignore_index=True)
    n=n+1



    

id_addition=0
Final_out=[]
for n in range(0, len(Scenario)):
    print('Processing Scenario {}/{}'.format(str(n).zfill(2),len(Scenario)))
    Meta_data=pd.read_csv('data/{}_recordingMeta.csv'.format(str(n).zfill(2)))
    Track_data=pd.read_csv('data/{}_tracks.csv'.format(str(n).zfill(2)))
    Track_data=Track_data[['frame', 'trackId', 'xCenter', 'yCenter', 'heading', 'lonVelocity', 'latVelocity', 'lonAcceleration', 'latAcceleration']]
    Track_Meta_data=pd.read_csv('data/{}_tracksMeta.csv'.format(str(n).zfill(2)))
    Final=Track_Meta_data[['trackId','width','length','class']].copy(deep=True)
    
    Track_data_keys = Track_data.set_index(['frame'])['trackId']
    
    Final['locationId']=Meta_data['locationId'][0]
    Final['recordingId']=Meta_data['recordingId'][0]
    
    Final['otherVehicles']='0'
    Final['track']='0'
    
    for loc_id in Final['trackId']:
        local_track = Track_data[Track_data['trackId'] == loc_id].copy(deep=True)
        local_track.drop(columns=['trackId'], inplace=True)
        
        Final['track'][loc_id] = local_track
        Final['trackId'][loc_id] = Final['trackId'][loc_id] + id_addition
        
        local_frames = local_track['frame'].to_numpy()
        possible_id = Track_data_keys[local_frames].to_numpy()
    
        allVehicles = np.unique(possible_id) + id_addition
        Final['otherVehicles'][loc_id] = np.delete(allVehicles, np.where(allVehicles == Final['trackId'][loc_id])[0])
        
    
    
    
    id_addition = id_addition + Meta_data['numTracks'][0] 
    Final_out.append(Final.copy(deep=True))
   
print('Save Data')
Final_out= pd.concat(Final_out)
Final_out.to_pickle("RounD_processed.pkl")