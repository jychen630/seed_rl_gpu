import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
from dm_control import suite
import argparse
# Constants from the paper
TIMESTEP = 3000  # 30 seconds at 0.01 timestep
ACTION_MAP = [-10, 0, 10]  # Force values from the paper

def featurize(physics):
    x = physics.named.data.qpos['slider'][0]
    theta = physics.named.data.qpos['hinge_1'][0]
    x_dot = physics.named.data.qvel['slider'][0]
    theta_dot = physics.named.data.qvel['hinge_1'][0]
    return np.array([
        math.cos(theta),
        math.sin(theta),
        theta_dot / 10.0,
        x / 10.0,
        x_dot / 10.0,
        1.0 if abs(x) < 0.1 else 0.0
    ], dtype=np.float32)

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

class SeedEnsemble:
    def __init__(self, ensemble_size=30, device="cpu"):
        self.device = device
        self.ensemble_size = ensemble_size
        self.models = []
        self.priors = []
        self.noise_seeds = []

        for _ in range(ensemble_size):
            model = QNetwork().to(device)
            prior = QNetwork().to(device)

            for net in [model, prior]:
                for layer in net.children():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            noise_seed = np.random.normal(0, 0.01, size=100000)
            self.models.append(model)
            self.priors.append(prior)
            self.noise_seeds.append(noise_seed)

        self.optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in self.models]

    def get_model(self, agent_idx):
        return self.models[agent_idx % self.ensemble_size]

    def get_prior(self, agent_idx):
        return self.priors[agent_idx % self.ensemble_size]

    def get_noise(self, model_idx, exp_idx):
        noise_arr = self.noise_seeds[model_idx % self.ensemble_size]
        return noise_arr[exp_idx % len(noise_arr)]

def create_cartpole_env():
    env = suite.load('cartpole', 'swingup')
    physics = env.physics
    task = env._task
    return env, physics, task

def run_experiment(K=100, ensemble_size=30):
    print("Running experiment with K={}, ensemble_size={}".format(K, ensemble_size))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ensemble = SeedEnsemble(ensemble_size=ensemble_size, device=device)
    replay_buffer = []
    results = np.zeros(TIMESTEP)

    envs, physics_list, tasks = [], [], []
    for _ in range(K):
        env, physics, task = create_cartpole_env()
        env.reset()
        physics.named.data.qpos['slider'] = 0.0
        physics.named.data.qpos['hinge_1'] = np.pi
        physics.named.data.qvel['slider'] = 0.0
        physics.named.data.qvel['hinge_1'] = 0.0
        physics.after_reset()
        envs.append(env)
        physics_list.append(physics)
        tasks.append(task)

    for step in tqdm(range(TIMESTEP)):
        for agent_idx in range(K):
            model = ensemble.get_model(agent_idx)
            prior = ensemble.get_prior(agent_idx)
            optimizer = ensemble.optimizers[agent_idx % ensemble.ensemble_size]

            physics = physics_list[agent_idx]
            obs_tensor = torch.tensor(featurize(physics)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = model(obs_tensor) + 3.0 * prior(obs_tensor)
                action_idx = torch.argmax(q_vals).item()

            action = ACTION_MAP[action_idx]
            envs[agent_idx].step([action])
            reward = tasks[agent_idx].get_reward(physics)
            new_state = physics.state()

            model_idx = agent_idx % ensemble.ensemble_size
            noise = ensemble.get_noise(model_idx, len(replay_buffer))
            replay_buffer.append((physics.state(), action_idx, reward + noise, new_state))

            results[step] += reward

            if len(replay_buffer) % 48 == 0:
                batch = random.sample(replay_buffer, 16)
                obs_b, act_b, rew_b, next_obs_b = zip(*batch)

                obs_b_feats = []
                next_obs_b_feats = []
                for o, o_next in zip(obs_b, next_obs_b):
                    physics_copy = envs[0].physics.copy()
                    physics_copy.set_state(o)
                    obs_b_feats.append(featurize(physics_copy))

                    physics_copy.set_state(o_next)
                    next_obs_b_feats.append(featurize(physics_copy))

                obs_b = torch.tensor(np.array(obs_b_feats)).float().to(device)
                next_obs_b = torch.tensor(np.array(next_obs_b_feats)).float().to(device)

                act_b = torch.tensor(act_b).long().to(device)
                rew_b = torch.tensor(rew_b).float().to(device)

                for model_idx in range(ensemble_size):
                    model = ensemble.models[model_idx]
                    prior = ensemble.priors[model_idx]
                    optimizer = ensemble.optimizers[model_idx]

                    with torch.no_grad():
                        q_next = model(next_obs_b) + 3.0 * prior(next_obs_b)
                        q_next = q_next.max(1)[0]

                    target = rew_b + 0.9 * q_next

                    q_pred = model(obs_b)
                    q_val = q_pred.gather(1, act_b.unsqueeze(1)).squeeze()
                    prior_val = prior(obs_b).gather(1, act_b.unsqueeze(1)).squeeze()

                    loss = (1/0.01) * ((q_val - target)**2).mean() + (1/0.01) * ((q_val - prior_val)**2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        all_results.append(results / K)

    mean_results = np.mean(all_results, axis=0)
    print(mean_results[:5], mean_results[-5:], len(mean_results))
    return mean_results

def plot_results(results):
    timestep = np.arange(TIMESTEP) * 0.01
    window_size = 50
    smoothed_results = np.convolve(results, np.ones(window_size)/window_size, mode='valid')
    smoothed_timestep = timestep[window_size-1:]

    plt.figure(figsize=(10, 6))
    plt.plot(timestep, results, alpha=0.3, label='Raw Reward')
    plt.plot(smoothed_timestep, smoothed_results, label='Smoothed Reward')
    plt.plot(timestep, np.zeros_like(timestep), 'k--', label='DQN ε-greedy')
    plt.xlabel('Time elapsed (seconds)')
    plt.ylabel('Average instantaneous reward')
    plt.title('Cartpole Swing-up: Seed TD Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig('cartpole_torch.png')

if __name__ == '__main__':
    # import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    K = args.K
    ensemble_size = min(K, 30)
    results = run_experiment(K=K, ensemble_size=ensemble_size)
    plot_results(results)
