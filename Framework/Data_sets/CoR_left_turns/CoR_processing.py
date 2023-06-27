import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment=None


def plot_tracks(Final,i):
    n=5
    plt.close('all')
    ego=Final.ego_track[i]
    bot=Final.bot_track[i]
    fig, ax = plt.subplots(1, 1,figsize=(15,8))
    ax.set_aspect('equal')
    ax.set_xlim(-150, 50)
    ax.set_ylim(-50, 10)   
    
    ax.plot([-3.5,-3.5],[-50,10],'k')
    ax.plot([3.5,3.5],[-50,10],'k')
    ax.plot([0,0],[-50,10],'--k')
    
    ax.plot([-150,50],[-3.5,-3.5],'k')
    ax.plot([-150,50],[3.5,3.5],'k')
    ax.plot([-150,50],[0,0],'--k')
    points_ego = ax.scatter(ego.x.loc[0:1].tolist(),ego.y.loc[0:1].tolist(),c='r',marker='o',s=10)
    points_bot = ax.scatter(bot.x.loc[0:1].tolist(),bot.y.loc[0:1].tolist(),c='b',marker='o',s=10)
    
    plt.draw()
    for ii in range(1,len(ego),n):
        points_ego.set_offsets([ego.x.loc[ii],ego.y.loc[ii]])
        # redraw just the points
        if len(bot.x.loc[ii:ii+1])>1:
            points_bot.set_offsets([bot.x.loc[ii],bot.y.loc[ii]])
            
        plt.title('Situation {}: t= {:0.2f} s, i={}'.format(i,ego.t[ii],ii))
        fig.canvas.draw_idle()
        plt.pause(5*(ego.t[ii]-ego.t[ii-1]))
    return


data=pd.read_csv('data/processed_data.csv')

measures=pd.read_csv('data/measures.csv')
Final=measures[['subj_id','RT','is_go_decision','is_collision']]

Final['ego_track']='0'
Final['bot_track']='0'

n_test = 276
test = False

if test:
    ran = range(n_test,n_test+1)
else:
    ran = range(len(Final))
    
for i in ran:   
    print('Situation {}'.format(i))
    local_track = data[data.session==measures.session[i]]
    local_track = local_track[local_track.intersection_no==measures.intersection_no[i]]
    local_track = local_track[local_track.subj_id==measures.subj_id[i]]
    local_track = local_track[local_track.route==measures.route[i]]
    local_track_ego = local_track[['t','ego_x','ego_y','ego_vx','ego_vy','ego_ax','ego_ay']].reset_index(drop = True)
    local_track_bot=local_track[['t','bot_x','bot_y','bot_vx','bot_vy','bot_ax','bot_ay']].reset_index(drop=True)
    # Zero for intersection, and rotate to always have the bot hav vy=0 ad vx<0
    bot_vx = np.array(local_track_bot.bot_vx)
    bot_vy = np.array(local_track_bot.bot_vy)
    test_x = np.abs(bot_vx[1:]-bot_vx[:-1])<0.001
    test_y = np.abs(bot_vy[1:]-bot_vy[:-1])<0.001
    test_v = (np.abs(bot_vx[:-1])>5) |  (np.abs(bot_vy[:-1])>5)
    updated = np.where(test_x & test_y & test_v)[0].tolist()
    bot_spawn_idx = updated[0]
    local_track_bot = local_track_bot.loc[bot_spawn_idx:]
    # Rename to equal columns        
    local_track_ego.columns = ['t','x','y','vx','vy','ax','ay']
    local_track_bot.columns = ['t','x','y','vx','vy','ax','ay']
    
    # Subtract the fucking intersection location
    local_track_ego.x -= local_track.intersection_x.iloc[0]
    local_track_ego.y -= local_track.intersection_y.iloc[0]
    local_track_bot.x -= local_track.intersection_x.iloc[0]
    local_track_bot.y -= local_track.intersection_y.iloc[0]
    
    # Rotate problem so the bot hase vy=0 ad vx>0
    # Determine angle
    theta = np.arctan2(local_track_bot.vx.iloc[0],local_track_bot.vy.iloc[0])
    dtheta = theta-np.pi/2
    rotation = np.array( [[np.cos(dtheta), -np.sin(dtheta)],\
                          [np.sin(dtheta),  np.cos(dtheta)]])
        
    [local_track_bot.x ,local_track_bot.y ] = np.dot(rotation,np.array([local_track_bot.x ,local_track_bot.y ]))
    [local_track_bot.vx,local_track_bot.vy] = np.dot(rotation,np.array([local_track_bot.vx,local_track_bot.vy]))
    [local_track_bot.ax,local_track_bot.ay] = np.dot(rotation,np.array([local_track_bot.ax,local_track_bot.ay]))
    
    [local_track_ego.x ,local_track_ego.y ] = np.dot(rotation,np.array([local_track_ego.x ,local_track_ego.y ]))
    [local_track_ego.vx,local_track_ego.vy] = np.dot(rotation,np.array([local_track_ego.vx,local_track_ego.vy]))
    [local_track_ego.ax,local_track_ego.ay] = np.dot(rotation,np.array([local_track_ego.ax,local_track_ego.ay]))
    # Condition for crash: Bot leaves straight line while crossing the intersection
    Final.ego_track[i] = local_track_ego
    Final.bot_track[i] = local_track_bot
    if test:
        plot_tracks(Final,i)
    
 
if not test:
    # analyisis
    Final.to_pickle("CoR_processed.pkl")