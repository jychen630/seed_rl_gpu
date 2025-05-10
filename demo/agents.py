# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:28:12 2018
@author: Maria Dimakopoulou
"""

import numpy as np

###############################################################################
class Agent(object):
  """Base class for all agent interface."""

  def __init__(self, **kwargs):
    pass

  def __str__(self):
    pass

  def update_observation(self, obs, action, reward, new_obs, p_continue,
                         **kwargs):
    pass

  def update_policy(self, **kwargs):
    pass

  def pick_action(self, obs, **kwargs):
    pass

  def initialize_episode(self, **kwargs):
    pass

  def _random_argmax(self, vector):
    """Helper function to select argmax at random... not just first one."""
    vector = vector.ravel()
    index = np.random.choice(np.where(vector == vector.max())[0])
    return index

  def _egreedy_action(self, q_vals, epsilon=0):
    """Epsilon-greedy dithering action selection.
    Args:
      q_vals: n_action x 1 - Q value estimates of each action.
      epsilon: float - probability of random action
    Returns:
      action: integer index for action selection
    """
    if np.random.rand() < epsilon:
      return np.random.randint(len(q_vals))
    else:
      return self._random_argmax(q_vals)


  def _boltzmann_action(self, q_vals, beta=0.01):
    """Boltzmann dithering action selection.
    Args:
      q_vals: n_action x 1 - Q value estimates of each action.
      beta: float - temperature for Boltzmann
    Returns:
      action - integer index for action selection
    """
    q_vals = np.exp((q_vals - max(q_vals)) / beta)
    boltzmann_dist = q_vals / np.sum(q_vals)
    return np.random.multinomial(1, boltzmann_dist).argmax()


class RandomAgent(Agent):
  """Take actions completely at random."""
  def __init__(self, num_action=3, **kwargs):
    self.num_action = num_action

  def __str__(self):
    return "RandomAgent(|A|={})".format(self.num_action)

  def pick_action(self, obs, **kwargs):
    action = np.random.randint(self.num_action)
    return action


class ConstantAgent(Agent):
  """Take constant actions."""
  def __init__(self, action=0, **kwargs):
    self.action = action

  def __str__(self):
    return "ConstantAgent(a={})".format(self.action)

  def pick_action(self, obs, **kwargs):
    return self.action


class QLearning(Agent):
  def __init__(self, num_action, feature_extractor, q_init,
               omega=0.75, epsilon=None, beta=None):

    assert(bool(epsilon is not None) != bool(beta is not None)), \
          "Either epsilon (for epsilon-greedy) or beta (for boltzmann)" \
          "should be passed, but not both."

    self.num_action = num_action
    self.num_state = feature_extractor.dimension + 1  # Extra terminal state
    self.feature_extractor = feature_extractor

    self.epsilon = epsilon
    self.beta = beta

    self.omega = omega
    self.state_action_visit_count = np.zeros((self.num_state, self.num_action))

    self.Q = np.full((self.num_state, self.num_action), fill_value=q_init)

  def __str__(self):
    return "QLearning(|S|={}, |A|={}, omega={}, {})" \
           .format(self.num_state, self.num_action, self.omega,
                   "epsilon-greedy(epsilon={})".format(self.epsilon)
                   if self.epsilon is not None else
                   "Boltzmann(beta={})".format(self.beta))

  def update_observation(self, obs, action, reward, new_obs, p_continue,
                         **kwargs):
    raise NotImplementedError

  def pick_action(self, obs, **kwargs):
    state = self.feature_extractor.get_feature(obs)
    q_vals = self.Q[state, :]
    if self.epsilon is not None:
      action = self._egreedy_action(q_vals, self.epsilon)
    elif self.beta is not None:
      action = self._boltzmann_action(q_vals, self.beta)
    else:
      raise ValueError
    return action


class EpisodicQLearning(QLearning):
  def __str__(self):
    return "EpisodicQLearning(|S|={}, |A|={}, omega={}, {})" \
           .format(self.num_state, self.num_action, self.omega,
                   "epsilon-greedy(epsilon={})".format(self.epsilon)
                   if self.epsilon is not None else
                   "Boltzmann(beta={})".format(self.beta))

  def update_observation(self, obs, action, reward, new_obs, p_continue,
                         **kwargs):
    old_state = self.feature_extractor.get_feature(obs)
    if p_continue == 1:
      new_state = self.feature_extractor.get_feature(new_obs)
    else:
      new_state = self.num_state - 1

    self.state_action_visit_count[old_state, action] += 1
    rate = 1.0 / (self.state_action_visit_count[old_state, action] + 1) \
                 ** self.omega

    self.Q[old_state, action] = (1 - rate) * self.Q[old_state, action] \
                              + rate * (reward + np.max(self.Q[new_state, :]))


class SARSA(QLearning):
  def __str__(self):
    return "SARSA(|S|={}, |A|={}, omega={}, {})" \
           .format(self.num_state, self.num_action, self.omega,
                   "epsilon-greedy(epsilon={})".format(self.epsilon)
                   if self.epsilon is not None else
                   "Boltzmann(beta={})".format(self.beta))

  def update_observation(self, obs, action, reward, new_obs, p_continue,
                         new_action=None, **kwargs):
    assert(new_action is not None)

    old_state = self.feature_extractor.get_feature(obs)
    if p_continue == 1:
      new_state = self.feature_extractor.get_feature(new_obs)
    else:
      new_state = self.num_state - 1

    self.state_action_visit_count[old_state, action] += 1
    rate = 1.0 / (self.state_action_visit_count[old_state, action] + 1) \
                 ** self.omega

    self.Q[old_state, action] = (1 - rate) * self.Q[old_state, action] \
                              + rate * (reward + self.Q[new_state, new_action])


###############################################################################
class FeatureExtractor(object):
  """Base feature extractor."""

  def __init__(self, **kwargs):
    pass

  def __str__(self):
    pass

  def get_feature(self, obs):
    pass


class IdentityFeature(FeatureExtractor):
  """For cartpole, pass entire state vector = (x, x_dot, theta, theta_dot)"""
  """For general MDP, pass state s."""

  def __init__(self, dimension=None, **kwargs):
    self.dimension = dimension

  def __str__(self):
    return "IdentityFeature(dimension={})" \
            .format(self.dimension)

  def get_feature(self, obs):
    return obs


class TabularFeatures(FeatureExtractor):

  def __init__(self, num_x, num_x_dot, num_theta, num_theta_dot):
    """Define buckets across each variable."""
    self.num_x = num_x
    self.num_x_dot = num_x_dot
    self.num_theta = num_theta
    self.num_theta_dot = num_theta_dot

    self.x_bins = np.linspace(-3, 3, num_x - 1, endpoint=False)
    self.x_dot_bins = np.linspace(-2, 2, num_x_dot - 1, endpoint=False)
    self.theta_bins = np.linspace(- np.pi / 3, np.pi / 3,
                                  num_theta - 1, endpoint=False)
    self.theta_dot_bins = np.linspace(-4, 4, num_theta_dot - 1, endpoint=False)

    self.dimension = num_x * num_x_dot * num_theta * num_theta_dot

  def __str__(self):
    return "TabularFeatures(num_x={}, num_x_dot={}, " \
                            "num_theta={}, num_theta_dot={})" \
            .format(self.num_x, self.num_x_dot,
                    self.num_theta, self.num_theta_dot)

  def _get_single_ind(self, var, var_bin):
    if len(var_bin) == 0:
      return 0
    else:
      return int(np.digitize(var, var_bin))

  def _get_state_num(self, x_ind, x_dot_ind, theta_ind, theta_dot_ind):
    state_num = \
      (x_ind + x_dot_ind * self.num_x
       + theta_ind * (self.num_x * self.num_x_dot)
       + theta_dot_ind * (self.num_x * self.num_x_dot * self.num_theta_dot))
    return int(state_num)

  def get_feature(self, obs):
    """We get the index using the linear space"""
    x, x_dot, theta, theta_dot = obs
    x_ind = self._get_single_ind(x, self.x_bins)
    x_dot_ind = self._get_single_ind(x_dot, self.x_dot_bins)
    theta_ind = self._get_single_ind(theta, self.theta_bins)
    theta_dot_ind = self._get_single_ind(theta_dot, self.theta_dot_bins)

    state_num = self._get_state_num(x_ind, x_dot_ind, theta_ind, theta_dot_ind)
    return state_num