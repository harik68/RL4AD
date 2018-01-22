import gym
env = gym.make('NetworkIntrusion-v0')
import random

"""
A dump agent which takes random actions
"""
env.reset()
episode_over = False
sum_rewards = 0.0
while episode_over==False:
    # Take a step and get new state and reward
    action_index = random.randint(0, 1)
    ob, reward, episode_over, details = env.step(action_index)
    sum_rewards += reward
    print(sum_rewards, reward)



