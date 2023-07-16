"""
This tutorial shows you how to solve reaching task using inverse kinemetics
provided in the package internally.
"""

from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task

def control_policy(env, obs):
    return env.get_robot().get_joint_positions_from_tip_positions(obs[-9:], obs[1:10])


def end_effector_pos():
    task = generate_task(task_generator_id='reaching')
    env = CausalWorld(task=task,
                      enable_visualization=True,
                      action_mode="joint_positions",
                      normalize_actions=False,
                      normalize_observations=False)
    obs = env.reset()
    for _ in range(100):
        goal_dict = env.sample_new_goal()
        success_signal, obs = env.do_intervention(goal_dict)
        obs, reward, done, info = env.step(control_policy(env, obs))
        for _ in range(250):
            
            obs, reward, done, info = env.step(control_policy(env, obs))
            
            desired_goal_1 = info['desired_goal'][0:3]
            desired_goal_2 = info['desired_goal'][3:6]
            desired_goal_3 = info['desired_goal'][6:9]
            
            finger_pos_1 = info['achieved_goal'][0:3]
            finger_pos_2 = info['achieved_goal'][3:6]
            finger_pos_3 = info['achieved_goal'][6:9]
            
            goal_1_distance = (finger_pos_1[0] - desired_goal_1[0])**2 + (finger_pos_1[1] - desired_goal_1[1])**2 + (finger_pos_1[2] - desired_goal_1[2])**2
            goal_2_distance = (finger_pos_2[0] - desired_goal_2[0])**2 + (finger_pos_2[1] - desired_goal_2[1])**2 + (finger_pos_2[2] - desired_goal_2[2])**2
            goal_3_distance = (finger_pos_3[0] - desired_goal_3[0])**2 + (finger_pos_3[1] - desired_goal_3[1])**2 + (finger_pos_3[2] - desired_goal_3[2])**2
            print("Goal 1 distance:" + str(goal_1_distance))
            print("Goal 2 distance:" + str(goal_2_distance))
            print("Goal 3 distance:" + str(goal_3_distance))
    
    env.close()


if __name__ == '__main__':
    end_effector_pos()