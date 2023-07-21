"""
This tutorial shows you how to create a customized task and use all the
underlying functionalities of CausalWorld as is including reward calculation..etc
"""

import random
import numpy as np
import pandas as pd
from SupportClasses import *
from causal_world.envs.causalworld import CausalWorld
from ReachingSingleFinger import ReachingSingleFinger


def change_stage_colour(env, v):
    R, G, B = STAGE_COLOUR
    R = min(1, 0.5*(v + 0.1)*R)
    G = min(1, 0.5*(v + 0.1)*G)
    B = min(1, 0.5*(v + 0.1)*B)
    colour = np.array([R, G, B])
    env.do_intervention({'stage_color' : colour})
    return np.sum(colour)


def change_floor_colour(env, d):
    R, G, B = FLOOR_COLOUR
    R = np.clip((d + 0.4)*R, 0, 1)
    G = np.clip((d + 0.4)*G, 0, 1)
    B = np.clip((d + 0.4)*B, 0, 1)
    colour = np.array([R, G, B])
    env.do_intervention({'floor_color' : colour})
    return np.sum(colour)


def change_box_colour(env, d, v):
    R, G, B = BOX_COLOUR
    R = np.clip(((0.3*v) + (0.05 * d))*R, 0, 1)
    G = np.clip(((0.3*v) + (0.05 * d))*G, 0, 1)
    B = np.clip(((0.3*v) + (0.05 * d))*B, 0, 1)
    colour = np.array([R, G, B])
    env.do_intervention({'tool_block' : {'color' : colour}})
    return np.sum(colour)


def control_policy(env, obs):
    tip = np.concatenate((obs[-3:], np.array(FIX_POS_120), np.array(FIX_POS_300)))
    return env.get_robot().get_joint_positions_from_tip_positions(tip, obs[1:10])


def cart2cyl(position):
    x = position[0]
    y = position[1]
    z = position[2]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi, z])


def gen_single_goal(env, is_box_goal):
    goal_dict = env.sample_new_goal()
    
    set_goal_box = bool(random.randint(0, 1))
    if set_goal_box and not is_box_goal:
        goal_dict['goal_60']['cylindrical_position'] = cart2cyl(BOX_GOAL_60)
        
    is_box_goal = set_goal_box
    env.do_intervention(goal_dict)
    return is_box_goal

        
def extract_data(data):
    goal_1 = data['desired_goal'][0:3]
    finger_pos_1 = data['achieved_goal'][0:3]
    return goal_1, finger_pos_1


def init(intervention = False):
    robot = Finger(FingerID.F1, HZ, timeout=TIMEOUT)
    task = ReachingSingleFinger()
    env = CausalWorld(task = task,
                      skip_frame = hz_to_skipframe(HZ),
                      enable_visualization = True,
                      action_mode = "joint_positions",
                      normalize_actions = False,
                      normalize_observations = False)
    
    obs = env.reset()
    
    obs, _, _, info = env.step(control_policy(env, obs))
    g1, f1 = extract_data(info)
    robot.set_goal(g1[0], g1[1], g1[2]); robot.set_pos(f1[0], f1[1], f1[2])
    
    obs, _, _, info = env.step(control_policy(env, obs))
    robot.set_goal(g1[0], g1[1], g1[2]); robot.set_pos(f1[0], f1[1], f1[2])
    
    # Sc = change_stage_colour(env, robot.v)
    if not intervention:
        Fc = change_floor_colour(env, robot.dist2D(BOX_GOAL_60))
    else:
        colour = np.array(FLOOR_INT_COLOUR)
        env.do_intervention({'floor_color' : colour})
        Fc = np.sum(colour)
    Bc = change_box_colour(env, robot.dist2D(BOX_GOAL_60), robot.v)
    # df.loc[0] = [Fc, Bc, Sc, robot.dist2D(BOX_GOAL_60), robot.v]
    # df.loc[1] = [Fc, Bc, Sc, robot.dist2D(BOX_GOAL_60), robot.v]
    df.loc[0] = [Fc, Bc, np.sum(STAGE_COLOUR), robot.dist2D(BOX_GOAL_60), robot.v]
    df.loc[1] = [Fc, Bc, np.sum(STAGE_COLOUR), robot.dist2D(BOX_GOAL_60), robot.v]
    
    return env, robot, obs

    
def run(intervention = False):
    global IS_BOX_GOAL
    
    # Init Finger and Environment
    env, F1, obs = init(intervention)
    
    # Loop over time
    for t in range(2, T):
        # Sc = change_stage_colour(env, df.loc[t-1]['v'])
        if not intervention: 
            Fc = change_floor_colour(env, df.loc[t-1]['d_b'])
        else:
            colour = np.array(FLOOR_INT_COLOUR)
            env.do_intervention({'floor_color' : colour})
            Fc = np.sum(colour)        
        Bc = change_box_colour(env, df.loc[t-2]['d_b'], df.loc[t-1]['v'])
        
        obs, _, _, info = env.step(control_policy(env, obs))
        
        g1, f1 = extract_data(info)
        F1.set_goal(g1[0], g1[1], g1[2]); F1.set_pos(f1[0], f1[1], f1[2])
        
            
        if F1.dg <= THRES or F1.timeout < 0: 
            IS_BOX_GOAL = gen_single_goal(env, IS_BOX_GOAL)
            F1.reset_timeout()
            
        F1.dec_timeout()
        # df.loc[t] = [Fc, Bc, Sc, F1.distance(BOX_GOAL_60), F1.v]
        # df.loc[t] += [Fc - df.loc[t]["F_c"], Bc, Sc, F1.dist2D(BOX_GOAL_60), F1.v]
        df.loc[t] += [Fc - df.loc[t]["F_c"], Bc, np.sum(STAGE_COLOUR), F1.dist2D(BOX_GOAL_60), F1.v]
        

    env.close()
    
    
def hz_to_skipframe(hz):
    sec = 1/hz
    return int(250*sec)


if __name__ == '__main__':
    STAGE_COLOUR = np.array([193, 205, 205])/255
    # FLOOR_COLOUR = np.array([139, 115, 85])/255
    FLOOR_COLOUR = np.array([122,122,122])/255
    FLOOR_INT_COLOUR = np.array([128, 128, 128])/255
    BOX_COLOUR = np.array([255, 0, 0])/255
    BOX_GOAL_60 = [0, 0, 0.1]
    FIX_POS_120 = [0.1541, -0.2029, 0.46]
    FIX_POS_300 = [-0.2441, -0.2180,  0.4324]
    HZ = 10
    THRES = 0.03 # [m]
    TIMEOUT = 20
    IS_BOX_GOAL = True
    T = 600
    
    
    columns = ['F_c', 'B_c', 'S_c', 'd_b', 'v']
    data = np.random.uniform(low = -0.03, high = 0.03, size = (T, len(columns)))
    df = pd.DataFrame(data, columns = columns)
    run(intervention = False)
    
    df = df.iloc[2:]
    df.to_csv("ReachingSingleFinger_obs_noise0.03.csv", index = False)
    
    
    columns = ['F_c', 'B_c', 'S_c', 'd_b', 'v']
    data = np.random.uniform(low = -0.03, high = 0.03, size = (T, len(columns)))
    df = pd.DataFrame(data, columns = columns)
    run(intervention = True)
    
    df = df.iloc[2:]
    df.to_csv("ReachingSingleFinger_int_noise0.03.csv", index = False)
    
