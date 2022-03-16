import retro

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

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
    for idx in range(0, len(obs_stack) + 1):
        if idx > 0:
            max_obs.append(np.max(all_obs[idx, ...], all_obs[idx - 1, ...]))
        else:
            max_obs.append(all_obs[idx])

    # Extract the Y channel and resize the frames to 84x84
    for i in range(0, len(obs_stack) + 1):
        b, g, r = cv2.split(max_obs[i])
        luminance = 0.2126*r + 0.7152*g + 0.0722*b
        resized_frame = cv2.resize(luminance, dsize = (84, 84))
        y_obs.append(resized_frame)

    # Stack the most recent m frames as the input to the Q function

    for i in range(m,len(obs_stack)+1):
        idx = i
        z_i = []
        while idx >= i - m:
            z_i.append(y_obs[idx])
            idx -= 1

        z_obs.append(z_i)


## pre_process(obs_list, 4)   --> Tests the pre-process function

    


    


    



