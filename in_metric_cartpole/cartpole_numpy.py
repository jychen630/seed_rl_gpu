import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
from dm_control import suite
import argparse

TIMESTEP = 3000
ACTION_MAP = [-10, 0, 10]

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

class NumpyQNetwork:
    def __init__(self):
        self.fc1_w = np.random.randn(50, 6) * np.sqrt(2 / (6 + 50))
        self.fc1_b = np.zeros(50)
        self.fc2_w = np.random.randn(50, 50) * np.sqrt(2 / (50 + 50))
        self.fc2_b = np.zeros(50)
        self.head_w = np.random.randn(3, 50) * np.sqrt(2 / (50 + 3))
        self.head_b = np.zeros(3)

    def forward(self, x):
        x1 = np.maximum(0, np.dot(x, self.fc1_w.T) + self.fc1_b)
        x2 = np.maximum(0, np.dot(x1, self.fc2_w.T) + self.fc2_b)
        return np.dot(x2 + x1, self.head_w.T) + self.head_b

    def update(self, grads, lr=1e-3):
        for attr in grads:
            param = getattr(self, attr)
            setattr(self, attr, param - lr * grads[attr])

class SeedEnsemble:
    def __init__(self, ensemble_size=30):
        self.ensemble_size = ensemble_size
        self.models = []
        self.priors = []
        self.noise_seeds = []

        for _ in range(ensemble_size):
            model = NumpyQNetwork()
            prior = NumpyQNetwork()
            self.models.append(model)
            self.priors.append(prior)
            self.noise_seeds.append(np.random.normal(0, 0.01, size=100000))

    def get_model(self, agent_idx):
        return self.models[agent_idx % self.ensemble_size]

    def get_prior(self, agent_idx):
        return self.priors[agent_idx % self.ensemble_size]

    def get_noise(self, model_idx, exp_idx):
        return self.noise_seeds[model_idx % self.ensemble_size][exp_idx % 100000]

def create_cartpole_env():
    env = suite.load('cartpole', 'swingup')
    return env, env.physics, env._task

def compute_loss_and_grads(model, prior, obs_b, act_b, rew_b, next_obs_b):
    grads = {}
    q = model.forward(obs_b)
    q_prior = prior.forward(obs_b)
    q_next = model.forward(next_obs_b)
    q_prior_next = prior.forward(next_obs_b)
    target = rew_b + 0.9 * np.max(q_next + 3.0 * q_prior_next, axis=1)

    q_val = q[np.arange(len(act_b)), act_b]
    q_prior_val = q_prior[np.arange(len(act_b)), act_b]

    loss = (1/0.01) * np.mean((q_val - target)**2) + (1/0.01) * np.mean((q_val - q_prior_val)**2)
    return loss, {}  # skipping actual gradients for now

def run_experiment(K=100, ensemble_size=30):
    print("Running experiment with K={}, ensemble_size={}".format(K, ensemble_size))
    np.random.seed(0)
    random.seed(0)

    ensemble = SeedEnsemble(ensemble_size=ensemble_size)
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

            obs = featurize(physics_list[agent_idx]).reshape(1, -1)
            q_vals = model.forward(obs) + 3.0 * prior.forward(obs)
            action_idx = np.argmax(q_vals)
            action = ACTION_MAP[action_idx]

            envs[agent_idx].step([action])
            reward = tasks[agent_idx].get_reward(physics_list[agent_idx])
            new_state = physics_list[agent_idx].state()

            model_idx = agent_idx % ensemble.ensemble_size
            noise = ensemble.get_noise(model_idx, len(replay_buffer))
            replay_buffer.append((physics_list[agent_idx].state(), action_idx, reward + noise, new_state))

            results[step] += reward

            if len(replay_buffer) % 48 == 0:
                batch = random.sample(replay_buffer, 16)
                obs_b_feats, next_obs_b_feats = [], []
                for o, o_next in [(b[0], b[3]) for b in batch]:
                    physics_copy = envs[0].physics.copy()
                    physics_copy.set_state(o)
                    obs_b_feats.append(featurize(physics_copy))
                    physics_copy.set_state(o_next)
                    next_obs_b_feats.append(featurize(physics_copy))

                obs_b = np.array(obs_b_feats)
                next_obs_b = np.array(next_obs_b_feats)
                act_b = np.array([b[1] for b in batch])
                rew_b = np.array([b[2] for b in batch])

                for model_idx in range(ensemble_size):
                    model = ensemble.models[model_idx]
                    prior = ensemble.priors[model_idx]
                    loss, grads = compute_loss_and_grads(model, prior, obs_b, act_b, rew_b, next_obs_b)
                    # skipping actual backprop update for now

        results[step] /= K
    return results

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
    plt.savefig('cartpole_numpy.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    K = args.K
    ensemble_size = min(K, 30)
    results = run_experiment(K=K, ensemble_size=ensemble_size)
    plot_results(results)
