import random
import numpy as np
import pandas as pd
from SupportClasses import *
from causal_world.envs.causalworld import CausalWorld
from ReachingSingleFinger import ReachingSingleFinger


def brightness(z, d, v, blending_factor = 0.5):
    brightness_depth = 1 - (((MAX_HEIGHT - z) / MAX_HEIGHT) + ((MAX_DIST - d) / MAX_DIST))
    brightness_motion = 1 - (v / MAX_VEL)

    final_brightness = min(1, blending_factor * brightness_depth + (1 - blending_factor) * brightness_motion)
    
    return final_brightness


# def change_stage_colour(env, d, v):
#     R, G, B = STAGE_COLOUR
#     b = brightness(d, v, blending_factor = .9)
#     R = np.clip(b*R, 0, 1)
#     G = np.clip(b*G, 0, 1)
#     B = np.clip(b*B, 0, 1)
#     colour = np.array([R, G, B])
#     env.do_intervention({'stage_color' : colour})
#     return np.sum(colour)


def change_floor_colour(env, z):
    R, G, B = FLOOR_COLOUR
    b = brightness(z, MAX_DIST, 0, blending_factor = 1)
    R = np.clip(b*R, 0, 1)
    G = np.clip(b*G, 0, 1)
    B = np.clip(b*B, 0, 1)
    colour = np.array([R, G, B])
    env.do_intervention({'floor_color' : colour})
    return np.sum(colour)


def change_box_colour(env, z, d, v):
    R, G, B = BOX_COLOUR
    b = brightness(z, d, v)
    R = np.clip(b*R, 0, 1)
    G = np.clip(b*G, 0, 1)
    B = np.clip(b*B, 0, 1)
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
    
    # Sc = change_stage_colour(env, robot.distance(BOX_GOAL_60), robot.v)
    Fc = change_floor_colour(env, robot.pos.z)
    if not intervention:
        Bc = change_box_colour(env, robot.pos.z, robot.dist2D(BOX_GOAL_60), robot.v)
    else:
        colour = np.array(BOX_INT_COLOUR)
        env.do_intervention({'tool_block' : {'color' : colour}})
        Bc = np.sum(colour)
    df.loc[0] = [robot.pos.z, Fc, robot.dist2D(BOX_GOAL_60), Bc, robot.v]
    df.loc[1] = [robot.pos.z, Fc, robot.dist2D(BOX_GOAL_60), Bc, robot.v]
    
    return env, robot, obs

    
def run(intervention = False):
    global IS_BOX_GOAL
    
    # Init Finger and Environment
    env, F1, obs = init(intervention)
    
    # Loop over time
    for t in range(2, T):
        # Sc = change_stage_colour(env, df.loc[t-2]['d_b'], df.loc[t-1]['v'])
        Fc = change_floor_colour(env, F1.pos.z)  
        if not intervention:
            Bc = change_box_colour(env, F1.pos.z, df.loc[t-1]['d_b'], df.loc[t-1]['v'])
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
        # df.loc[t] = [F1.distance(BOX_GOAL_60), Fc, Bc, F1.v, Sc]
        if not intervention:
            df.loc[t] += [F1.pos.z, Fc, F1.dist2D(BOX_GOAL_60), Bc, F1.v]
        else:
            df.loc[t] += [F1.pos.z, Fc, F1.dist2D(BOX_GOAL_60), Bc - df.loc[t]["B_c"], F1.v]

        # print("PLANAR DISTANCE " + str(F1.dist2D(BOX_GOAL_60)))
        # print("H "+str(F1.pos.z))
    env.close()
    
    
def hz_to_skipframe(hz):
    sec = 1/hz
    return int(250*sec)


if __name__ == '__main__':
    STAGE_COLOUR = np.array([193, 205, 205])/255
    FLOOR_COLOUR = np.array([66,66,66])/255
    FLOOR_INT_COLOUR = np.array([128, 128, 128])/255
    BOX_COLOUR = np.array([255, 0, 0])/255
    BOX_INT_COLOUR = np.array([28,134,238])/255
    
    BOX_GOAL_60 = [0, 0, 0.1]
    MAX_DIST = 0.20
    MAX_HEIGHT = 0.30
    MAX_VEL = 1.50
    
    FIX_POS_120 = [0.1541, -0.2029, 0.46]
    FIX_POS_300 = [-0.2441, -0.2180,  0.4324]
    
    HZ = 10
    THRES = 0.03 # [m]
    TIMEOUT = 20
    IS_BOX_GOAL = True
    T = 600
    NOISE = (-0.03, 0.03)
    
    columns = ['H', 'F_c', 'd_b', 'B_c', 'v']
    data = np.random.uniform(low = NOISE[0], high = NOISE[1], size = (T, len(columns)))
    df = pd.DataFrame(data, columns = columns)
    # df = pd.DataFrame(columns = columns)
    run(intervention = False)
    
    df = df.iloc[2:]
    df.to_csv("ReachingSingleFinger_obs_noise0.03.csv", index = False)
    
    
    columns = ['H', 'F_c', 'd_b', 'B_c', 'v']
    data = np.random.uniform(low = NOISE[0], high = NOISE[1], size = (T, len(columns)))
    df = pd.DataFrame(data, columns = columns)
    run(intervention = True)
    
    df = df.iloc[2:]
    df.to_csv("ReachingSingleFinger_int_noise0.03.csv", index = False)
    
