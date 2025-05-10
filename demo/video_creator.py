# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:26:11 2018

@author: Maria
"""

from environments import CartPole
import numpy as np
import skvideo.io

algorithms = ["seed"]
agents = [27]

for algorithm in algorithms:
  for agent in agents:
    state_action_list = np.load("state_action_list_{}_{}.npy".format(algorithm, agent))

    test = False
    environment = CartPole()
    frames = environment.render(state_action_list, mode='rgb_array')
    print(frames.shape)


    writer = skvideo.io.FFmpegWriter("{}_{}.mp4".format(algorithm, agent),
        outputdict={'-c:v': 'libx264', '-b': '5000k'},  inputdict={'-framerate': str(1 / environment.tau)})
    for frame in frames:
      writer.writeFrame(frame)
    writer.close()
