import random
import numpy as np
import pandas as pd
from SupportClasses import *
from causal_world.envs.causalworld import CausalWorld
from ReachingSingleFinger import ReachingSingleFinger
import pybullet


def brightness(z, max_height, wz, v, wv, d, wd):
    brightness_depth = 1 - ((max_height - abs(z)) / max_height)
    brightness_motion = 1 - (v / MAX_VEL)
    brightness_distance = 1 - (MAX_DIST - d / MAX_DIST)

    final_brightness = np.clip(wz * brightness_depth + wv * brightness_motion + wd * brightness_distance, 0, 1)
    
    return final_brightness


def change_floor_colour(env, z):
    R, G, B = FLOOR_COLOUR
    b = brightness(z, MAX_HEIGHT, 0.66, 0, 0, 0, 0)*0.1 + 0.6
    R = np.clip(b*R, 0, 1)
    G = np.clip(b*G, 0, 1)
    B = np.clip(b*B, 0, 1)
    colour = np.array([R, G, B])
    env.do_intervention({'floor_color' : colour})
    return np.sum(colour)


def change_box_colour(env, z, v, d):
    R, G, B = BOX_COLOUR
    b = brightness(z - 0.095, MAX_HEIGHT - 0.095, 0.35, v, 0.65, d, 0.10)
    R = np.clip(b*R, 0, 1)
    G = np.clip(b*G, 0, 1)
    B = np.clip(b*B, 0, 1)
    colour = np.array([R, G, B])
    env.do_intervention({'tool_block' : {'color' : colour}})
    return np.sum(colour)


def change_stage_colour(env, n):
    R, G, B = STAGE_COLOUR
    R = min(1, n + R)
    G = min(1, n + G)
    B = min(1, n + B)
    colour = np.array([R, G, B])
    env.do_intervention({'stage_color' : colour})
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
    
    # Set the camera view
    camera_distance = .5  # distance from the camera to the target (object you want to look at)
    camera_yaw = 0.0      # camera yaw angle in degrees (rotation around the vertical axis)
    camera_pitch = -60.0   # camera pitch angle in degrees (rotation around the horizontal axis)
    camera_target_position = [0, 0, 0]  # position of the target object you want to look at

    pybullet.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
    
    obs, _, _, info = env.step(control_policy(env, obs))
    g1, f1 = extract_data(info)
    robot.set_goal(g1[0], g1[1], g1[2]); robot.set_pos(f1[0], f1[1], f1[2])
    
    obs, _, _, info = env.step(control_policy(env, obs))
    robot.set_goal(g1[0], g1[1], g1[2]); robot.set_pos(f1[0], f1[1], f1[2])
    
    Fc = change_floor_colour(env, robot.pos.z)
    if not intervention:
        Bc = change_box_colour(env, robot.pos.z, robot.v, robot.dist2D(BOX_GOAL_60))
    else:
        colour = np.array(BOX_INT_COLOUR)
        env.do_intervention({'tool_block' : {'color' : colour}})
        Bc = np.sum(colour) 
    
    
    df.loc[0] = [Fc, Bc, robot.pos.z, robot.v, robot.dist2D(BOX_GOAL_60)]
    df.loc[1] = [Fc, Bc, robot.pos.z, robot.v, robot.dist2D(BOX_GOAL_60)]
    
    return env, robot, obs

    
def run(intervention = False):
    global IS_BOX_GOAL
    
    # Init Finger and Environment
    env, F1, obs = init(intervention)
    
    # Loop over time
    for t in range(2, T):

        Fc = change_floor_colour(env, df.loc[t-2]['H'])
        if not intervention: 
            Bc = change_box_colour(env, df.loc[t-1]['H'], df.loc[t-1]['v'], df.loc[t-1]['d_b'])
        else:
            colour = np.array(BOX_INT_COLOUR)
            env.do_intervention({'tool_block' : {'color' : colour}})
            Bc = np.sum(colour)   
        
        obs, _, _, info = env.step(control_policy(env, obs))
        
        g1, f1 = extract_data(info)
        F1.set_goal(g1[0], g1[1], g1[2]); F1.set_pos(f1[0], f1[1], f1[2])
        
            
        if F1.dg <= THRES or F1.timeout < 0: 
            IS_BOX_GOAL = gen_single_goal(env, IS_BOX_GOAL)
            F1.reset_timeout()
            
        F1.dec_timeout()
        
        if not intervention: 
            df.loc[t] += [Fc, Bc, F1.pos.z, F1.v, F1.dist2D(BOX_GOAL_60)]
        else:
            df.loc[t] += [Fc, Bc - df.loc[t]["B_c"], F1.pos.z, F1.v, F1.dist2D(BOX_GOAL_60)]
        

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
    BOX_INT_COLOUR = np.array([28,134,238])/255

    BOX_GOAL_60 = [0, 0, 0.1]
    FIX_POS_120 = [0.1541, -0.2029, 0.46]
    FIX_POS_300 = [-0.2441, -0.2180,  0.4324]
    HZ = 10
    THRES = 0.03 # [m]
    TIMEOUT = 20
    IS_BOX_GOAL = True
    T = 600
    MAX_HEIGHT = 0.30
    MAX_DIST = 0.20
    MAX_VEL = 1.50
    
    NOISE = (-0.15, 0.15)

    
    columns = ['F_c', 'B_c', 'H', 'v', 'd_b']
    data = np.random.uniform(low = NOISE[0], high = NOISE[1], size = (T, 2))
    data = np.c_[data, np.random.uniform(low = -0.2, high = 0.2, size = (T, 1)), np.random.uniform(low = NOISE[0], high = NOISE[1], size = (T, 2))]
    df = pd.DataFrame(data, columns = columns)
    run(intervention = False)
    
    df = df.iloc[2:]
    df.to_csv("ReachingSingleFinger_obs_noise" + str(NOISE[1]) + ".csv", index = False)
    
    columns = ['F_c', 'B_c', 'H', 'v', 'd_b']
    data = np.random.uniform(low = NOISE[0], high = NOISE[1], size = (T, len(columns)))
    df = pd.DataFrame(data, columns = columns)
    run(intervention = True)
    
    df = df.iloc[2:]
    df.to_csv("ReachingSingleFinger_int_noise" + str(NOISE[1]) + ".csv", index = False)
    
