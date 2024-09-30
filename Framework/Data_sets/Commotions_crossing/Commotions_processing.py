import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def correct_time(T):
    t_min = T.t.iloc[0]
    t_max = T.t.iloc[-1]
    
    td_min = np.ceil(t_min * 30) / 30 
    
    td = np.arange(td_min, t_max, 1/30)
    
    T_new = pd.DataFrame(np.zeros((len(td), len(T.columns)), int), columns = T.columns)
    T_new.t = td
    T_new.x         = np.interp(T_new.t, T.t, T.x, left = T.x.iloc[0], right = T.x.iloc[-1])
    T_new.y         = np.interp(T_new.t, T.t, T.y, left = T.y.iloc[0], right = T.y.iloc[-1])
    T_new.offset    = np.interp(T_new.t, T.t, T.offset, left = T.offset.iloc[0], right = T.offset.iloc[-1])
    
    try:
        T_new.iloc[:,3:] = T.iloc[0,3:].astype(int)
    except:
        pass
    
    T_new['Frame'] = (round(T_new.t * 30)).astype(int)
    T_new = T_new.set_index('Frame')
    return T_new

pd.options.mode.chained_assignment = None

participants = [p for p in os.listdir('Data') if p[:3] == 'sub']

columns = ['participant', 'participant_track']
Final_data = pd.DataFrame(np.zeros((len(participants) * 2, len(columns)), object), columns = columns)

for i, participant in enumerate(participants):
    print('Participant ' + str(i))

    
    participant_logs = os.listdir('Data' + os.sep + participant)
    participant_log = [pd.read_csv('Data' + os.sep + participant + os.sep + p, sep = '\t') 
                       for p in participant_logs if p[:7] == 'subject' and p[-5:] == '2.txt'][0]
    participant_track = participant_log[['simTime', 'x', 'y', 'offset']].rename(columns={"simTime": "t"})

    participant_track = correct_time(participant_track)
 
    # First scenario
    i_sample = 2 * i
    crossing_logs = [p for p in participant_logs if p[:6] == 'drone3']
    
    frame_min = participant_track.index[-1]
    frame_max = 0
    
    for j, name in enumerate(crossing_logs):
        # Look for laterals distance 
        track = pd.read_csv('Data' + os.sep + participant + os.sep + name, sep = '\t')[['%simTime', 'x_r', 'y_r', 'offset_p', 
                                                                                        'leaderID_r', 'followerID_r']]

        track = track.rename(columns={"%simTime": "t", "x_r": "x", "y_r": "y", 'offset_p': 'offset',
                                      "leaderID_r": "leaderID", "followerID_r": "followerID"}) 

        
        track_id = name[6:8] 
        
        track[['leaderID', 'followerID']] = np.maximum(track[['leaderID', 'followerID']] - 300, -1)
        
        track['trackID'] = int(track_id)             
        
        track = correct_time(track)
        
        frame_min = min(frame_min, track.index[0])
        
        frame_max = max(frame_max, track.index[-1])
        
        try:
            Final_data['drone_track_' + track_id].iloc[i_sample] = track
        except:
            Final_data['drone_track_' + track_id] = '0'
            Final_data['drone_track_' + track_id].iloc[i_sample] = track
     
    
    Final_data.participant.iloc[i_sample] = participant    
    Final_data.participant_track.iloc[i_sample] = participant_track.loc[frame_min:frame_max]  
    
    
    # Second scenario
    i_sample = 2 * i + 1
    crossing_logs = [p for p in participant_logs if p[:7] == 'drone10']
    
    frame_min = participant_track.index[-1]
    frame_max = 0
    
    for j, name in enumerate(crossing_logs):
        track = pd.read_csv('Data' + os.sep + participant + os.sep + name, sep = '\t')[['%simTime', 'x_r', 'y_r', 'offset_p', 
                                                                                        'leaderID_r', 'followerID_r']]
        
        track = track.rename(columns={"%simTime": "t", "x_r": "x", "y_r": "y", 'offset_p': 'offset',
                                      "leaderID_r": "leaderID", "followerID_r": "followerID"}) 

        
        track_id = name[7:9] 
        
        track[['leaderID', 'followerID']] = np.maximum(track[['leaderID', 'followerID']] - 1000, -1)
        
        track['trackID'] = int(track_id)             

        track = correct_time(track)   
        
        frame_min = min(frame_min, track.index[0])
        
        frame_max = max(frame_max, track.index[-1])
        
        try:
            Final_data['drone_track_' + track_id].iloc[i_sample] = track
        except:
            Final_data['drone_track_' + track_id] = '0'
            Final_data['drone_track_' + track_id].iloc[i_sample] = track
     
    
    Final_data.participant.iloc[i_sample] = participant    
    Final_data.participant_track.iloc[i_sample] = participant_track.loc[frame_min:frame_max]  


Final_data.to_pickle("Commotions_processed.pkl")