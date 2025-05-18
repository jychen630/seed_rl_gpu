import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import collections

TIMESTEP = 3000
Step = collections.namedtuple('Step', ['reward', 'new_obs', 'p_continue'])

def glorot_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def featurize(state):
    x, x_dot, theta, theta_dot = state
    return np.array([
        math.cos(theta),
        math.sin(theta),
        theta_dot / 10.0,
        x / 10.0,
        x_dot / 10.0,
        1.0 if abs(x) < 0.1 else 0.0
    ], dtype=np.float32)

class CartPole:
    def __init__(self, verbose=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.01
        self.p_opposite_direction = 0.1
        self.p_no_reward = 0.25
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.9
        self.move_cost = 0.1
        self.verbose = verbose
        self.steps_beyond_done = None

    def step(self, action):
        assert action in [0, 1, 2], f"Invalid action: {action}"
        x, x_dot, theta, theta_dot = self.state
        if np.random.random() < self.p_opposite_direction:
            force = -(action - 1) * self.force_mag
        else:
            force = (action - 1) * self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        p_continue = (
            -self.x_threshold <= x <= self.x_threshold and
            -self.theta_threshold_radians <= theta <= self.theta_threshold_radians
        )
        if (
            math.cos(theta) > 0.95 and
            abs(x) < 0.1 and
            abs(x_dot) < 1.0 and
            abs(theta_dot) < 1.0
        ):
            reward = 1.0
        elif p_continue:
            reward = np.random.binomial(n=1, p=1 - self.p_no_reward)
        else:
            reward = 0.0
        reward -= self.move_cost * abs(action - 1)
        return Step(reward, np.array(self.state), float(p_continue))

    def reset(self):
        theta = np.pi + np.random.uniform(-0.01, 0.01)
        x = np.random.uniform(-0.05, 0.05)
        x_dot = np.random.uniform(-0.01, 0.01)
        theta_dot = np.random.uniform(-0.01, 0.01)
        self.state = (x, x_dot, theta, theta_dot)
        self.steps_beyond_done = None
        return np.array(self.state)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, 3)
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        return self.head(x2 + x1)

class SeedTDNeuralAgent:
    def __init__(self, alpha=0.01, gamma=0.99, lamb=0.01, noise_std=0.01, device="cpu"):
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.noise_std = noise_std
        self.device = device
        self.q = QNetwork().to(device)
        self.q.apply(glorot_init)
        self.q_prior = QNetwork().to(device)
        self.q_prior.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=1e-3)
        self.noise = np.random.normal(0, noise_std, size=100000)
        self.noise_index = 0

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act_and_train(self, env, replay_buffer, timestep_rewards):
        obs = env.reset()
        for step in range(TIMESTEP):
            obs_tensor = torch.tensor(featurize(obs)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.q(obs_tensor) + 3.0 * self.q_prior(obs_tensor)
                action = torch.argmax(q_vals).item()
            step_out = env.step(action)
            noise = self._next_noise()
            replay_buffer.append((obs, action, step_out.reward + noise, step_out.new_obs))
            timestep_rewards[step] += step_out.reward
            obs = step_out.new_obs
            if len(replay_buffer) >= 16:
                batch = random.sample(replay_buffer, 16)
                obs_b, act_b, rew_b, next_obs_b = zip(*batch)
                obs_b = torch.tensor(np.array([featurize(o) for o in obs_b])).float().to(self.device)
                next_obs_b = torch.tensor(np.array([featurize(o) for o in next_obs_b])).float().to(self.device)
                act_b = torch.tensor(act_b).long().to(self.device)
                rew_b = torch.tensor(rew_b).float().to(self.device)
                with torch.no_grad():
                    q_next = self.q(next_obs_b) + 3.0 * self.q_prior(next_obs_b)
                    q_next = q_next.max(1)[0]
                target = rew_b + self.gamma * q_next
                q_pred = self.q(obs_b)
                q_val = q_pred.gather(1, act_b.unsqueeze(1)).squeeze()
                prior_val = self.q_prior(obs_b).gather(1, act_b.unsqueeze(1)).squeeze()
                loss = ((q_val - target)**2).mean() + (1/self.lamb) * ((q_val - prior_val)**2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

def train_with_traj(K=100, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestep_rewards = np.zeros(TIMESTEP)
    shared_buffer = []
    agent = SeedTDNeuralAgent(device=device)
    for _ in tqdm(range(K), desc="Training agents"):
        env = CartPole()
        agent.act_and_train(env, shared_buffer, timestep_rewards)
    return timestep_rewards / K

def run_figure3_combined():
    K_values = [10]
    timestep = np.arange(TIMESTEP) * 0.01
    for K in K_values:
        reward_traj = np.zeros(TIMESTEP)
        for seed in range(5):
            reward_traj += train_with_traj(K, seed=seed)
        reward_traj /= 5
        plt.plot(timestep, reward_traj, label=f'seed TD (K={K})')
    plt.plot(timestep, np.zeros_like(timestep), 'k--', label='DQN ε-greedy')
    plt.xlabel('time elapsed (seconds)')
    plt.ylabel('average instantaneous reward')
    plt.title('Cartpole: Seed TD Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig('figure3_cartpole_learning_curve.png')

if __name__ == '__main__':
    run_figure3_combined()
