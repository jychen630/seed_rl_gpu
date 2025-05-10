# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:34:02 2018

@author: Maria Dimakopoulou
"""

"""
From OpenAI gym, modified for the purpose of MS&E 338
"""

from gym.core import Env
from gym.envs.classic_control import rendering

import collections
import math
import numpy as np
import time

Step = collections.namedtuple('Step', ['reward', 'new_obs', 'p_continue'])

class CartPole(Env):

  # Initialize the environment.
  def __init__(self, verbose=False):
    self.gravity = 9.8
    self.masscart = 1.0
    self.masspole = 0.1
    self.total_mass = (self.masspole + self.masscart)
    self.length = 0.5 # actually half the pole's length
    self.polemass_length = (self.masspole * self.length)
    self.force_mag = 10.0
    self.tau = 0.01 # seconds between state updates

    # The probability of failed action, i.e. action 'left' moving cart to the
    # right and action 'right' moving cart to the left.
    self.p_opposite_direction = 0.1
    # Probability of no reward.
    self.p_no_reward = 0.25

    # Angle at which to fail the episode
    self.theta_threshold_radians = 12 * 2 * math.pi / 360
    self.x_threshold = 0.9
    self.steps_beyond_done = None

    # The action space.
    # The better the learning algorithm, the less you’ll have to try to
    # interpret these numbers yourself.
    self.action_space = [0, 1, 2]

    # The actions that move the cart have a cost
    self.move_cost = 0.1

    self.verbose = verbose


  # Let the environment's state evolve based on the chosen action and
  # observe the next state.
  def step(self, action):
    assert action in [0, 1, 2], "%r (%s) invalid"%(action, type(action))

    state = self.state
    x, x_dot, theta, theta_dot = state

    # With some probability an action moves the cart to the opposite
    # direction than the one it should.
    if np.random.random() < self.p_opposite_direction:
      force = -(action - 1) * self.force_mag
    else:
      force = (action - 1) * self.force_mag

    # Compute the next state.
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = \
      (force + self.polemass_length * theta_dot * theta_dot * sintheta) \
      / self.total_mass
    thetaacc = \
      (self.gravity * sintheta - costheta * temp) \
      / (self.length *
         (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
    xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
    x  = x + self.tau * x_dot
    x_dot = x_dot + self.tau * xacc
    theta = theta + self.tau * theta_dot
    theta_dot = theta_dot + self.tau * thetaacc

    self.state = (x, x_dot, theta, theta_dot)

    # Determine the probability of continuing.
    p_continue = x >= -self.x_threshold \
                 and x <= self.x_threshold \
                 and theta >= -self.theta_threshold_radians \
                 and theta <= self.theta_threshold_radians

    # Determine the reward.
    if p_continue == 1.0:
      reward = np.random.binomial(n=1, p=1-self.p_no_reward)
    elif self.steps_beyond_done is None:
      # Pole just fell!
      self.steps_beyond_done = 0
      reward = np.random.binomial(n=1, p=1-self.p_no_reward)
    else:
      if self.steps_beyond_done == 0:
        print("You are calling 'step()' even though this environment has"
              "already returned done = True. You should always call 'reset()'"
              "once you receive 'done = True' -- "
              "any further steps are undefined behavior.")
      self.steps_beyond_done += 1
      reward = 0.0
    reward -= self.move_cost * np.abs(action - 1)

    # Return the observation.
    step = Step(reward, np.array(self.state), p_continue)
    if self.verbose:
      print(step)

    return step


  # Reset the state of the environment, e.g. at the beginning of a learning
  # episode.
  def reset(self):
    self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    self.steps_beyond_done = None
    return np.array(self.state)

  def max_per_period_reward(self):
    return 1.0

  # Utility method to render the cartpole given a list of state action pairs.
  def render(self, state_action_list, mode='human'):

    action_dict = {0: "nothing", 1: "left", 2: "right"}

    screen_width = 600
    screen_height = 600

    carty = screen_height / 2
    cartwidth = screen_width/8#50.0
    cartheight = 0.6 * cartwidth#30.0
    polewidth = cartwidth/5
    polelen = (3.2)*cartheight

    wallwidth = polewidth
    wallheight = screen_height

    viewer = rendering.Viewer(screen_width, screen_height)
    scale = (screen_width/2 - 2*cartwidth/2 - wallwidth)/self.x_threshold

    l,r,t,b = -wallwidth/2, wallwidth/2, wallheight/2, -wallheight/2
    wall = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    gray = 100/255
    wall.set_color(gray, gray, gray)
    walltrans = rendering.Transform()
    walltrans.set_translation(self.x_threshold*scale +wallwidth/2 + cartwidth/2 + screen_width/2.0, carty)
    wall.add_attr(walltrans)
    viewer.add_geom(wall)
    owall = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    owall.set_color(gray, gray, gray)
    owalltrans = rendering.Transform()
    owalltrans.set_translation(-self.x_threshold*scale -wallwidth/2 - cartwidth/2 + screen_width/2.0, carty)
    owall.add_attr(owalltrans)
    viewer.add_geom(owall)


    l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
    axleoffset =cartheight/4.0
    cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    carttrans = rendering.Transform()
    cart.add_attr(carttrans)
    viewer.add_geom(cart)

    l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
    pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
    pole.set_color(.8,.6,.4)
    poletrans = rendering.Transform(translation=(0, axleoffset))
    pole.add_attr(poletrans)
    pole.add_attr(carttrans)
    viewer.add_geom(pole)

    axle = rendering.make_circle(polewidth/2)
    axle.add_attr(poletrans)
    axle.add_attr(carttrans)
    axle.set_color(.5,.5,.8)
    viewer.add_geom(axle)


    right_arrow_points = [(0, 0), (-2, 1), (-2, 0.5), (-6, 0.5),
                       (-6, -0.5), (-2, -0.5), (-2, -1), (0, 0)]
    right_arrow_points = [(screen_width / 2 - cartwidth / 2 + cartwidth/8 * x,
                           carty + cartheight / 2 + cartwidth/8 * y)
                          for (x, y) in right_arrow_points]
    right_arrow = rendering.FilledPolygon(right_arrow_points)
    right_arrow.set_color(0, 0, 0)
    right_arrow_trans = rendering.Transform()
    right_arrow.add_attr(right_arrow_trans)


    left_arrow_points = [(0, 0), (2, 1), (2, 0.5), (6, 0.5),
                        (6, -0.5), (2, -0.5), (2, -1), (0, 0)]
    left_arrow_points = [(screen_width / 2 + cartwidth / 2 + cartwidth/8 * x,
                          carty + cartheight / 2 + cartwidth/8 * y)
                         for (x, y) in left_arrow_points]
    left_arrow = rendering.FilledPolygon(left_arrow_points)
    left_arrow.set_color(0, 0, 0)
    left_arrow_trans = rendering.Transform()
    left_arrow.add_attr(left_arrow_trans)

    track = rendering.Line((-self.x_threshold*scale - cartwidth/2 + screen_width/2.0,carty),
                            (self.x_threshold*scale + cartwidth/2 + screen_width/2.0,carty))
    track.set_color(0,0,0)
    viewer.add_geom(track)
    frames = []

    for state, action in state_action_list:
      x, x_dot, theta, theta_dot = state
      print(x)
      cartx = x*scale + screen_width/2.0 # MIDDLE OF CART
      carttrans.set_translation(cartx, carty)
      poletrans.set_rotation(-theta)
      if action_dict[action] == "right":
        right_arrow_trans.set_translation(x*scale, 0)
        viewer.add_onetime(right_arrow)
      if action_dict[action] == "left":
        left_arrow_trans.set_translation(x*scale, 0)
        viewer.add_onetime(left_arrow)

      frames.append(viewer.render(return_rgb_array = mode=='rgb_array'))

    viewer.close()
    return np.array(frames).astype(np.uint8)

