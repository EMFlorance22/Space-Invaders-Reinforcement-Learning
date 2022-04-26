import retro

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import random

env = retro.make(game='SpaceInvaders-Atari2600')

obs = env.reset()
counter = 0
obs_list = []
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    obs_list.append(obs)
    if done:
        obs = env.reset()
    counter += 1
    if counter > 100:
        break
env.close()


def pre_process(obs_stack, m):
    all_obs = np.array(obs_stack)
    y_obs = []
    z_obs = []

    # go through the list of frames and take the max for any two consecutive frames of the frame being encoded and the previous frame
    max_obs = []
    for idx in range(len(obs_stack)):
        if idx > 0:
            max_obs.append(np.maximum(all_obs[idx, ...], all_obs[idx - 1, ...])) # Finds the maximum of all three color channels between the two frames (color channels are in first dimension)
        else:
            max_obs.append(all_obs[idx])

    # Extract the Y channel and resize the frames to 84x84
    for i in range(len(obs_stack)):
        b, g, r = cv2.split(all_obs[i])
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        resized_frame = cv2.resize(luminance, dsize = (84, 84))
        y_obs.append(resized_frame)

    # Stack the most recent m frames as the input to the Q function

    for i in range(len(obs_stack) - m, len(obs_stack)):
        z_obs.append(y_obs[i])

    return z_obs

class MemoryReplay():
    def __init__(self, experience_set, memory_length):
        self.experience_data = experience_set
        self.memory_length = memory_length

    def set_replay(self, exp):
        # Adds an experience tuple to the memory replay up to a maximum of memory_length
        self.add(exp)
        return None 

    def get_replay(self, m):
        # Randomly samples m experience tuples from the memory replay
        random_tups = random.choices(self, k = m)
        
        replay_array = np.array(random_tups) # Converts the randomly sampled tuples into a numpy array to be inputs to the Q NN

        return replay_array     

q_inputs = pre_process(obs_list, 4)   

    
"""# go through the list of frames and take the max for any two consecutive frames of the frame being encoded and the previous frame
    max_obs = []
    for idx in range(len(obs_stack)):
        if idx > 0:
            max_obs.append(np.maximum(all_obs[idx, ...], all_obs[idx - 1, ...]))
        else:
            max_obs.append(all_obs[idx])
"""

    


    



