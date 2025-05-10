# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:34:02 2018

@author: Maria Dimakopoulou
"""

"""
From OpenAI gym, modified for the purpose of MS&E 338
Headless version that doesn't require OpenGL/pyglet
"""

from gym.core import Env
import collections
import math
import numpy as np
import time

Step = collections.namedtuple('Step', ['reward', 'new_obs', 'p_continue'])

class CartPole(Env):
    def __init__(self, verbose=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.01  # seconds between state updates
        self.p_opposite_direction = 0.1
        self.p_no_reward = 0.25
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.9
        self.steps_beyond_done = None
        self.action_space = [0, 1, 2]
        self.move_cost = 0.1
        self.verbose = verbose

    def step(self, action):
        assert action in [0, 1, 2], "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state

        if np.random.random() < self.p_opposite_direction:
            force = -(action - 1) * self.force_mag
        else:
            force = (action - 1) * self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)
        p_continue = x >= -self.x_threshold and x <= self.x_threshold and theta >= -self.theta_threshold_radians and theta <= self.theta_threshold_radians

        if p_continue == 1.0:
            reward = np.random.binomial(n=1, p=1-self.p_no_reward)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = np.random.binomial(n=1, p=1-self.p_no_reward)
        else:
            if self.steps_beyond_done == 0:
                print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        reward -= self.move_cost * np.abs(action - 1)

        step = Step(reward, np.array(self.state), p_continue)
        if self.verbose:
            print(step)

        return step

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def max_per_period_reward(self):
        return 1.0

    def render(self, state_action_list, mode='human'):
        # In headless mode, we'll just print the state and action information
        action_dict = {0: "nothing", 1: "left", 2: "right"}
        for state, action in state_action_list:
            x, x_dot, theta, theta_dot = state
            print(f"State: x={x:.2f}, x_dot={x_dot:.2f}, theta={theta:.2f}, theta_dot={theta_dot:.2f}, Action: {action_dict[action]}")
        return None
    