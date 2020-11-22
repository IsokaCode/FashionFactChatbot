#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import gym

# Create a breakout environment
env = gym.make('SpaceInvaders-v3')

# Reset it, returns the starting frame
frame = env.reset()

# Render
env.render()

is_done = False

while not is_done:
    # Perform a random action, returns the new frame, reward and whether the game is over
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    # Render
    env.render()

    time.sleep(0.01)
    if is_done:
        env.close()
        break

