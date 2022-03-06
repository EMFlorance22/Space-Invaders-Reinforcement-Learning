import retro

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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


