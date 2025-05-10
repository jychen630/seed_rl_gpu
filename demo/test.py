# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 19:43:32 2018

@author: Maria
"""

import pandas as pd
from math import acos, asin, atan2, cos, sin, pi, atan
import numpy as np
df = pd.read_msgpack('seed_data.msgpack')

def myatan(sin_theta, cos_theta):
  theta = atan(sin_theta / cos_theta)
  if theta < 0: theta += pi
  if sin_theta < 0: theta += pi
  return theta

agents = [75]
agent_types = [b'egreedy']
for agent in agents:
  print(agent)
  for agent_type in agent_types:
    subset = df.loc[(df[b'actor'] == agent) & (df[b'algorithm'] == agent_type)][[b'x', b'x_dot', b'sin_theta', b'cos_theta', b'theta_dot', b'action']]
    tuples = [list(x) for x in subset.values]
    for t in tuples:
      t[2] = myatan(t[2], t[3])
      del t[3]
    tuples = [(tuple(x[0:4]), x[4]) for x in tuples]
    name = "seed" if agent_type == b'rlsvi' else "epsilon"
    np.save("state_action_list_{}_{}.npy".format(name, agent), tuples)

agent_type = b'egreedy'
subset = df.loc[(df[b'algorithm'] == agent_type)].groupby([b't']).mean()[[b'reward']]
rewards_epsilon = subset.values[:, 0]

agent_type = b'rlsvi'
subset = df.loc[(df[b'algorithm'] == agent_type)].groupby([b't']).mean()[[b'reward']]
rewards_seed = subset.values[:, 0]

import matplotlib.pyplot as plt
handles = []
h, = plt.plot(range(len(rewards_epsilon)), rewards_epsilon, color="black", label=r"$\epsilon$-greedy (K=100)")
handles.append(h)
h, = plt.plot(range(len(rewards_seed)), rewards_seed, color=plt.cm.Paired(2), label="seed (K=100)")
handles.append(h)
plt.ylim([None, 1.1])
plt.xticks(np.arange(0, 3001, 500), np.arange(0, 31, 5))
plt.grid(b=True)
plt.legend(handles=handles)
plt.ylabel("Average per agent instantaneous reward")
plt.xlabel("Seconds")
