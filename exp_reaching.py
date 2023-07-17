import pandas as pd
import numpy as np
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from SupportClasses import *

THRES = 0.03


def control_policy(env, obs):
    return env.get_robot().get_joint_positions_from_tip_positions(obs[-9:], obs[1:10])


def gen_single_goal(env, finger:Finger, colour = None):
    goal_dict = env.sample_new_goal()
    goal_dict = {k:v for k,v in goal_dict.items() if k == 'goal_' + finger.value}

    success_signal, obs = env.do_intervention(goal_dict)
    
    if colour is not None: 
        success_signal, obs = env.do_intervention({'goal_' + finger.value : {'color': colour}})
    
    
def change_finger_colour(env, finger:Finger):
    random_colour = np.random.uniform(0, 1, 3,)
    intervention = {'robot_finger_' + finger.value + '_link_0': {'color': random_colour},
                    'robot_finger_' + finger.value + '_link_1': {'color': random_colour},
                    'robot_finger_' + finger.value + '_link_2': {'color': random_colour},
                    'robot_finger_' + finger.value + '_link_3': {'color': random_colour},}
    success_signal, obs = env.do_intervention(intervention)
    return random_colour


def hz_to_skipframe(hz):
    sec = 1/hz
    return int(250*sec)


def extract_data(data):
    goal_1 = data['desired_goal'][0:3]
    goal_2 = data['desired_goal'][3:6]
    goal_3 = data['desired_goal'][6:9]
        
    finger_pos_1 = data['achieved_goal'][0:3]
    finger_pos_2 = data['achieved_goal'][3:6]
    finger_pos_3 = data['achieved_goal'][6:9]
    
    return goal_1, goal_2, goal_3, finger_pos_1, finger_pos_2, finger_pos_3


def end_effector_pos():
    F1 = FingerState(Finger.F1, HZ)
    F2 = FingerState(Finger.F2, HZ)
    F3 = FingerState(Finger.F3, HZ)
    
    
    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      skip_frame = hz_to_skipframe(HZ), # 10Hz
                      enable_visualization=True,
                      action_mode="joint_positions",
                      normalize_actions=False,
                      normalize_observations=False)
    obs = env.reset()
    
    goal_dict = env.sample_new_goal()
    success_signal, obs = env.do_intervention(goal_dict)
    obs, reward, done, info = env.step(control_policy(env, obs))
    
    g1, g2, g3, f1, f2, f3 = extract_data(info)
    F1.set_goal(g1[0], g1[1], g1[2]); F1.set_pos(f1[0], f1[1], f1[2])
    colour = change_finger_colour(env, Finger.F1)
    F1.set_colour(colour[0], colour[1], colour[2])
    
    F2.set_goal(g2[0], g2[1], g2[2]); F2.set_pos(f2[0], f2[1], f2[2])
    colour = change_finger_colour(env, Finger.F2)
    F2.set_colour(colour[0], colour[1], colour[2])
    
    F3.set_goal(g3[0], g3[1], g3[2]); F3.set_pos(f3[0], f3[1], f3[2])
    colour = change_finger_colour(env, Finger.F3)
    F3.set_colour(colour[0], colour[1], colour[2])

    
    for _ in range(1800):
        
        obs, reward, done, info = env.step(control_policy(env, obs))
        
        g1, g2, g3, f1, f2, f3 = extract_data(info)
        F1.set_goal(g1[0], g1[1], g1[2]); F1.set_pos(f1[0], f1[1], f1[2])
        F2.set_goal(g2[0], g2[1], g2[2]); F2.set_pos(f2[0], f2[1], f2[2])
        F3.set_goal(g3[0], g3[1], g3[2]); F3.set_pos(f3[0], f3[1], f3[2])
        
        if F1.dg <= THRES: 
            colour = change_finger_colour(env, Finger.F1)
            gen_single_goal(env, Finger.F1, colour)
            F1.set_colour(colour[0], colour[1], colour[2])
            
        if F2.dg <= THRES: 
            colour = change_finger_colour(env, Finger.F2)
            gen_single_goal(env, Finger.F2, colour)
            F2.set_colour(colour[0], colour[1], colour[2])
            
        if F3.dg <= THRES: 
            colour = change_finger_colour(env, Finger.F3)
            gen_single_goal(env, Finger.F3, colour)
            F3.set_colour(colour[0], colour[1], colour[2])
            
        F1.dec_timeout(); F2.dec_timeout(); F3.dec_timeout()
        
        if F1.timeout <= 0:
            F1.reset_timeout()
            gen_single_goal(env, Finger.F1)
            
        if F2.timeout <= 0:
            F2.reset_timeout()
            gen_single_goal(env, Finger.F2)
            
        if F3.timeout <= 0:
            F3.reset_timeout()
            gen_single_goal(env, Finger.F3)
            
        # Store data in a dataframe
        df.loc[len(df)] = [F1.dg, F2.dg, F3.dg, F1.v, F2.v, F3.v, F1.intensity, F2.intensity, F3.intensity]

    env.close()

            
if __name__ == '__main__':
    HZ = 10
    columns = ['$dg_1$', '$dg_2$', '$dg_3$', '$v_1$', '$v_2$', '$v_3$', '$I_1$', '$I_2$', '$I_3$']
    df = pd.DataFrame(columns = columns)
    
    end_effector_pos()
    
    df.to_csv("exp_reaching_obs.csv")