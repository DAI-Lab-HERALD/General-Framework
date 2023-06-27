import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def merge_txt_files(data_path):
    dfs = []
    raw_data_path = os.path.join(data_path, "raw")
    for file in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file)
        if file_path.endswith(".txt"):
            print(file_path)
            dfs.append(pd.read_csv(file_path, sep="\t"))
    df_concat = pd.concat(dfs)
    df_concat.to_csv(os.path.join(data_path, "raw_data_merged.txt"), index=False, sep="\t")


def get_measures(traj):
    """
    This function extracts dependent variables and some other useful measures from an individual trajectory.
    """
    # -1 is assigned as the default value so that if the algorithms below do not trigger, we exclude the trial later on
    idx_bot_spawn = -1
    idx_response = -1
    RT = -1
    go_decision = False
    collision = False
    # See if the bot did spawn (no spawn => no interaction => unneceesary case)
    # Numerically. as v=15 was slowest possible condition, vs close to zero sdo not count and are liekly numerical error
    if (abs(traj.bot_v) > 1).any():
        idx_bot_spawn = np.where(traj.bot_v > 1)[0][0]
        # extract ego position and velocity
        ego = traj[['ego_x','ego_y','ego_vx','ego_vy']].copy()
        ego.columns=['x','y','vx','vy']
        # extract bot position and velocity
        bot = traj[['bot_x','bot_y','bot_vx','bot_vy']].copy()
        bot.columns=['x','y','vx','vy']
        
        # center on intersection        
        ego.x -= traj.intersection_x.iloc[0]
        ego.y -= traj.intersection_y.iloc[0]
        bot.x -= traj.intersection_x.iloc[0]
        bot.y -= traj.intersection_y.iloc[0]
        
        theta=np.arctan2(bot.vx.iloc[idx_bot_spawn],bot.vy.iloc[idx_bot_spawn])
        dtheta=theta-np.pi/2
        # determin rotation matrix necessary so that bot approaches intesection from negative x-direction
        rotation = np.array( [[np.cos(dtheta), -np.sin(dtheta)],\
                              [np.sin(dtheta),  np.cos(dtheta)]])
        # Aply rotation: bot
        [bot.x ,bot.y ]=np.dot(rotation,np.array([bot.x ,bot.y ]))
        [bot.vx,bot.vy]=np.dot(rotation,np.array([bot.vx,bot.vy]))
        
        print('angle = {}°, y = {:0.2f}m'.format(int(theta * 180 / np.pi), - bot.y.iloc[idx_bot_spawn]))
        
        # Aply rotation: ego 
        [ego.x ,ego.y ]=np.dot(rotation,np.array([ego.x ,ego.y ]))
        [ego.vx,ego.vy]=np.dot(rotation,np.array([ego.vx,ego.vy]))        
        # There is the possible case that the driver did not slow down enough to trigger the bot initially
        # but then, after crossing, drove into the stop sign, bringin the ego vehicle to a halt
        # This then would trigger the bot.
        # Therefore, it is necessary to check if at the bot spawning, the ego vehicle already crossed the intersection (y<0)
        if ego.y.iloc[idx_bot_spawn]<bot.y.iloc[idx_bot_spawn]:
            idx_bot_spawn = -1
        else: 
            # If the bot did spawn correctly, one could look at the reaction time
            # by default, the index of the first non-zero value of throttle after the bot spawned is marked as idx_response
            # this accurately determines idx_response if the throttle was not pressed at the time of the bot spawn,             
            throttle = traj.throttle
            # however, if throttle was pressed when the bot was spawned, we need to calculate idx_response differently
            # based on the visual inspection of such trials, we define idx_response as the onset time of the next
            # throttle press after the one already in process at the time the bot spawned
            # to calculate it, we first check if there were any zero throttle values after the bot spawned
            if (throttle.iloc[idx_bot_spawn:] == 0).any():
                # if so, we find the next zero value of throttle (idx_first_zero)
                idx_first_zero = idx_bot_spawn + (throttle.iloc[idx_bot_spawn:] == 0).to_numpy().nonzero()[0][0]
                # and then idx_response is defined as the index of the first non-zero value after that
                idx_response = idx_first_zero + (throttle.iloc[idx_first_zero:] > 0).to_numpy().nonzero()[0][0]

            # If the bot did spawn correctly, one can look for collisions.
            # it can be assumed that the bot never leaves its course parallel to the road, unless there is an collision
            # that means the difference to the initial y-position should never be larger then a certain threshold
            # the threshold 0.1 is choosen to prevent influence from potential machine precision issues.
            if (np.abs(bot.y.iloc[idx_bot_spawn:]-bot.y.iloc[idx_bot_spawn])>0.1).any():
                collision = True
                # Collision can only happen if the vehicle decided to cross
                go_decision = True
            else:
                # Next, the users response can be collected for non-collision cases. 
                # Here, one has to look at the time where the bot is on the same x-position as the ego_vehicle
                # or, alternatively, the closests to the intersection if the trajectory cuts of before the ego_vehicle 
                # reaches the intersection
                idx_bot_crosses = np.argmin(np.abs(bot.x-ego.x))
                # If the ego vehcile as already crossed, its y position should be smaller than that of the bot 
                if ego.y.iloc[idx_bot_crosses]<bot.y.iloc[idx_bot_crosses]:
                    go_decision = True
                    
        # RT = t2 - t1
        if idx_response>idx_bot_spawn:
            RT = traj.t.values[idx_response] - traj.t.values[idx_bot_spawn]

    return pd.Series({"idx_bot_spawn":  idx_bot_spawn,
                      "idx_response":   idx_response,
                      "RT":             RT,
                      "is_go_decision": go_decision,
                      "is_collision":   collision})

    
def filter_traj(traj):
    # Filter bot
    # To prevent to positional jump, which happens when the bot spawns, to ruin the data,
    # only post spawn data will be considered for filtering
    idx_bot_spawn = np.where(((traj.bot_vx)**2 + (traj.bot_vy)**2) > 1)[0]
    
    to_interpolate_ego = np.where( (traj.iloc[1:,1:7] == traj.iloc[:-1,1:7]).all(axis=1))[0]+1
    to_interpolate_bot = np.where( (traj.iloc[1:,7:] == traj.iloc[:-1,7:]).all(axis=1))[0]+1
    traj.iloc[to_interpolate_ego,:7]=np.nan
    traj.iloc[to_interpolate_bot,7:]=np.nan
    if idx_bot_spawn.size>0:        
        traj.iloc[:idx_bot_spawn[0],7:]=0.0
    traj.interpolate(inplace=True)
    return traj


def get_processed_data(data_file="raw_data_merged.txt"):
    data = pd.read_csv(data_file, sep="\t")
    data = data.set_index(["subj_id", "session", "route", "intersection_no"])

    # transforming timestamp so that every trajectory starts at t=0
    data.loc[:, "t"] = data.t.groupby(data.index.names).transform(lambda t: (t - t.min()))

    # we are only interested in left turns
    data = data[data.turn_direction == 1]

    # only consider the data recorded within 20 meters of each intersection
    # I changed it to 20m as there where cases where the car overshot the intersection and left the intersection that way
    data = data[abs(data.ego_distance_to_intersection) < 20]

    # calculate absolute values of speed and acceleration
    data["bot_v"] = np.sqrt(data.bot_vx ** 2 + data.bot_vy ** 2)
    
    # get the DVs and helper variables
    measures = data.groupby(data.index.names).apply(get_measures)
    print(measures.groupby(["subj_id", "session", "route"]).size())
    
    
    # smooth the time series by filtering out noise.
    # apply_filter = lambda traj: savgol_filter(traj, window_length=21, polyorder=2, axis=0)
    cols_to_smooth = ["t","ego_x", "ego_y", "ego_vx", "ego_vy", "ego_ax", "ego_ay",
                     "bot_x", "bot_y", "bot_vx", "bot_vy", "bot_ax", "bot_ay"]
    
    for j in range(len(measures)):
        print(j)
        data.loc[measures.index[j], cols_to_smooth]=filter_traj(data.loc[measures.index[j], cols_to_smooth])
    
    # data.loc[:, cols_to_smooth] = (data.loc[:, cols_to_smooth].groupby(data.index.names).transform(apply_filter))
    
    # calculate smoothed absolute values of speed and acceleration
    data["ego_v"] = np.sqrt(data.ego_vx ** 2 + data.ego_vy ** 2)
    data["bot_v"] = np.sqrt(data.bot_vx ** 2 + data.bot_vy ** 2)
    data["ego_a"] = np.sqrt(data.ego_ax ** 2 + data.ego_ay ** 2)
    data["bot_a"] = np.sqrt(data.bot_ax ** 2 + data.bot_ay ** 2)

    # calculate actual distance between the ego vehicle and the bot, and current tta for each t
    data["d_ego_bot"] = np.sqrt((data.ego_x - data.bot_x) ** 2 + (data.ego_y - data.bot_y) ** 2)
    data["tta"] = data.d_ego_bot / data.bot_v

    # merging the measures into the dynamics dataframe to manipulate the latter more conveniently
    data = data.join(measures)

    # add column "decision" for convenience of visualization
    measures["decision"] = "Stay"
    measures.loc[measures.is_go_decision, ["decision"]] = "Go"

    # add the condition information to the measures dataframe for further analysis
    conditions = data.loc[:, ["tta_condition", "d_condition", "v_condition"]].groupby(data.index.names).first()
    measures = measures.join(conditions)

    # excluded_data_idx = (data.RT <= 0) | ((data.RT > 2.0) & data.is_go_decision)
    # excluded_measures_idx = (measures.RT <= 0) | ((measures.RT > 2.0) & measures.is_go_decision)
    
    # I would not exclude belated positive decisions, as such are the most important to detect, 
    # as they have the greatest risk of injury
    
    excluded_data_idx = data.RT <= 0
    excluded_measures_idx = measures.RT <= 0
    
    # RT is defined as -1 if a driver didn't stop and the bot did not appear at the intersection; we discard these trials
    # WE also discarded the 12 or so trials, where the bot did appear, but the ego vehicle was already accelerating
    print("Number of discarded trials: %i" % (len(measures[excluded_measures_idx])))
    print(measures[excluded_measures_idx].groupby(["subj_id"]).size())

    data[excluded_data_idx].to_csv(os.path.join("processed_data_excluded.csv"), index=True)
    measures[excluded_measures_idx].to_csv(os.path.join("measures_excluded.csv"), index=True)

    data = data[~excluded_data_idx]
    measures = measures[~excluded_measures_idx]

    return data, measures


data_path = "../data"

# merge_txt_files(data_path)
data, measures = get_processed_data("raw_data_merged.txt")

measures.to_csv(os.path.join(data_path, "measures.csv"), index=True)
data.to_csv(os.path.join(data_path, "processed_data.csv"), index=True)