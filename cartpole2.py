# Seed TD Ensemble for Cartpole Swing-Up

from dm_control import suite
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

class CartpoleSwingUpEnv:
    def __init__(self):
        self.env = suite.load(domain_name="cartpole", task_name="swingup")
        self.physics = self.env.physics
        self.max_steps = 3000
        self.action_space = [-10.0, 0.0, 10.0]

    def reset(self):
        self.time_step = self.env.reset()
        self.steps = 0
        return self._get_obs()

    def step(self, action):
        torque = np.clip(action, -10.0, 10.0)
        self.time_step = self.env.step([torque])
        self.steps += 1
        return self._get_obs(), self._get_reward(), self.steps >= self.max_steps, {}

    def _get_obs(self):
        phi = self.physics.data.qpos[1]
        phi_dot = self.physics.data.qvel[1]
        x = self.physics.data.qpos[0]
        x_dot = self.physics.data.qvel[0]
        return np.array([
            np.cos(phi),
            np.sin(phi),
            phi_dot / 10.0,
            x / 10.0,
            x_dot / 10.0,
            1.0 if abs(x) < 0.1 else 0.0
        ], dtype=np.float32)

    def _get_reward(self):
        phi = self.physics.data.qpos[1]
        phi_dot = self.physics.data.qvel[1]
        x = self.physics.data.qpos[0]
        x_dot = self.physics.data.qvel[0]
        return float((np.cos(phi) > 0.95) and (abs(x) < 0.1) and (abs(x_dot) < 1.0 and abs(phi_dot) < 1.0))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_):
        self.buffer.append((s, a, r, s_))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_ = map(np.array, zip(*batch))
        return s, a, r, s_

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim=6, z_dim=4, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + z_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, output_dim)
        self.skip = nn.Linear(input_dim + z_dim, output_dim)

    def forward(self, x, z):
        xz = torch.cat([x, z.expand(x.shape[0], -1)], dim=1)
        h = F.relu(self.fc1(xz))
        h = F.relu(self.fc2(h))
        return self.out(h) + 3 * self.skip(xz)

def train_td_ensemble():
    K = 3
    episodes = 10
    max_steps = 3000
    gamma = 0.99
    z_dim = 4
    action_map = [-10.0, 0.0, 10.0]

    models = [QNetwork(z_dim=z_dim) for _ in range(K)]
    z_seeds = [torch.randn(z_dim) for _ in range(K)]
    opts = [optim.Adam(model.parameters(), lr=1e-3) for model in models]
    buffers = [ReplayBuffer() for _ in range(K)]
    all_returns = [[] for _ in range(K)]

    for k in range(K):
        env = CartpoleSwingUpEnv()
        model = models[k]
        opt = opts[k]
        buffer = buffers[k]
        z_k = z_seeds[k]

        for ep in range(episodes):
            s = env.reset()
            total_reward = 0
            for _ in range(max_steps):
                with torch.no_grad():
                    q = model(torch.tensor(s, dtype=torch.float32).unsqueeze(0), z_k.unsqueeze(0))
                    a_idx = q.argmax().item()
                a = action_map[a_idx]
                s_, r, done, _ = env.step(a)
                buffer.push(s, a_idx, r, s_)
                s = s_
                total_reward += r
                if done:
                    break

                if len(buffer) >= 64:
                    bs, ba, br, bs_ = buffer.sample(64)
                    bs = torch.tensor(bs, dtype=torch.float32)
                    ba = torch.tensor(ba, dtype=torch.long)
                    br = torch.tensor(br, dtype=torch.float32)
                    bs_ = torch.tensor(bs_, dtype=torch.float32)
                    with torch.no_grad():
                        q_next = model(bs_, z_k.expand(bs_.shape[0], -1)).max(1)[0]
                    target = br + gamma * q_next
                    pred = model(bs, z_k.expand(bs.shape[0], -1)).gather(1, ba.unsqueeze(1)).squeeze()
                    loss = F.mse_loss(pred, target)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            all_returns[k].append(total_reward)
            print(f"Agent {k}, Episode {ep}, Return: {total_reward:.2f}")

    mean_returns = np.mean(np.array(all_returns), axis=0)
    plt.plot(mean_returns)
    plt.xlabel("Episode")
    plt.ylabel("Mean Episodic Return")
    plt.title("TD-Ensemble on Cartpole Swing-Up")
    plt.grid(True)
    plt.savefig("figure3_td_ensemble.png")
train_td_ensemble()
