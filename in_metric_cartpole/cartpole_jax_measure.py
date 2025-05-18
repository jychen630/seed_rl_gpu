import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from dm_control import suite
import argparse
import time
import psutil
import csv
import os

TIMESTEP = 3000
ACTION_MAP = [-10, 0, 10]


def featurize(physics):
    x = physics.named.data.qpos['slider'][0]
    theta = physics.named.data.qpos['hinge_1'][0]
    x_dot = physics.named.data.qvel['slider'][0]
    theta_dot = physics.named.data.qvel['hinge_1'][0]
    return np.array([
        np.cos(theta),
        np.sin(theta),
        theta_dot / 10.0,
        x / 10.0,
        x_dot / 10.0,
        1.0 if abs(x) < 0.1 else 0.0
    ], dtype=np.float32)


class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        x1 = nn.relu(nn.Dense(50)(x))
        x2 = nn.relu(nn.Dense(50)(x1))
        return nn.Dense(3)(x2 + x1)


def create_cartpole_env():
    env = suite.load('cartpole', 'swingup')
    physics = env.physics
    task = env._task
    return env, physics, task


def init_ensemble(rng, size):
    models, priors, opt_states, noises = [], [], [], []
    for i in range(size):
        rng, init_rng = jax.random.split(rng)
        model = QNetwork()
        params = model.init(init_rng, jnp.ones((1, 6)))
        prior_params = model.init(init_rng, jnp.ones((1, 6)))
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        noise = jax.random.normal(init_rng, (100000,)) * 0.01
        models.append((model, params))
        priors.append(prior_params)
        opt_states.append(opt_state)
        noises.append(noise)
    return models, priors, opt_states, noises


def run_experiment(K=100, ensemble_size=30, num_seeds=1):
    # Initialize metrics
    start_time = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024**2  # Convert to MB
    
    all_results = []
    for seed in range(num_seeds):
        np.random.seed(seed)
        random.seed(seed)
        rng = jax.random.PRNGKey(seed)

        models, priors, opt_states, noises = init_ensemble(rng, ensemble_size)
        model = models[0][0]  # Shared model structure

        @jax.jit
        def train_step(params, prior_params, opt_state, obs, act, rew, next_obs):
            def loss_fn(params):
                q_pred = model.apply(params, obs)
                q_val = jnp.take_along_axis(q_pred, act[:, None], axis=1).squeeze()

                q_prior = model.apply(prior_params, obs)
                prior_val = jnp.take_along_axis(q_prior, act[:, None], axis=1).squeeze()

                q_next = model.apply(params, next_obs)
                q_prior_next = model.apply(prior_params, next_obs)
                q_next_val = jnp.max(q_next + 3.0 * q_prior_next, axis=1)
                target = rew + 0.9 * q_next_val

                return (1/0.01) * jnp.mean((q_val - target)**2) + (1/0.01) * jnp.mean((q_val - prior_val)**2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optax.adam(1e-3).update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state

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

        for step in tqdm(range(TIMESTEP), desc=f"Seed {seed+1}/{num_seeds}"):
            for agent_idx in range(K):
                model, params = models[agent_idx % ensemble_size]
                prior_params = priors[agent_idx % ensemble_size]
                opt_state = opt_states[agent_idx % ensemble_size]

                obs = featurize(physics_list[agent_idx])
                obs_tensor = jnp.array(obs).reshape(1, -1)
                q_vals = model.apply(params, obs_tensor) + 3.0 * model.apply(prior_params, obs_tensor)
                action_idx = int(jnp.argmax(q_vals))
                action = ACTION_MAP[action_idx]

                envs[agent_idx].step([action])
                reward = tasks[agent_idx].get_reward(physics_list[agent_idx])
                new_state = physics_list[agent_idx].state()

                model_idx = agent_idx % ensemble_size
                noise = noises[model_idx][len(replay_buffer) % 100000]
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

                obs_b = jnp.array(obs_b_feats)
                next_obs_b = jnp.array(next_obs_b_feats)
                act_b = jnp.array([b[1] for b in batch])
                rew_b = jnp.array([b[2] for b in batch])

                for i in range(ensemble_size):
                    model, params = models[i]
                    prior_params = priors[i]
                    opt_state = opt_states[i]
                    params, opt_state = train_step(params, prior_params, opt_state, obs_b, act_b, rew_b, next_obs_b)
                    models[i] = (model, params)
                    opt_states[i] = opt_state

        all_results.append(results / K)

    end_time = time.time()
    final_memory = psutil.Process().memory_info().rss / 1024**2

    # Calculate metrics
    total_time = end_time - start_time
    time_per_agent = total_time * 1000 / K # per millisecond
    throughput = K / (total_time * 1000)  # agents per millisecond
    memory_per_agent = (final_memory - initial_memory) / K
    # JAX doesn't have direct GPU memory tracking like PyTorch, so we'll set this to 0
    gpu_per_agent = 0

    metrics = {
        'K': K,
        'time_per_agent': time_per_agent,
        'throughput_per_agent': throughput,
        'memory_per_agent': memory_per_agent,
        'gpu_per_agent': gpu_per_agent
    }

    return metrics, np.mean(all_results, axis=0)


def plot_results(results, K):
    timestep = np.arange(TIMESTEP) * 0.01
    window_size = 50
    smoothed_results = np.convolve(results, np.ones(window_size)/window_size, mode='valid')
    smoothed_timestep = timestep[window_size-1:]

    plt.figure(figsize=(10, 6))
    plt.plot(timestep, results, alpha=0.3, label='Raw Reward (K={})'.format(K))
    plt.plot(smoothed_timestep, smoothed_results, label='Smoothed Reward')
    plt.plot(timestep, np.zeros_like(timestep), 'k--', label='DQN ε-greedy')
    plt.xlabel('Time elapsed (seconds)')
    plt.ylabel('Average instantaneous reward')
    plt.title('Cartpole Swing-up: Seed TD Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig('cartpole_jax_k_{}.png'.format(K))


def save_metrics_to_csv(metrics_list, filename='cartpole_jax_measure.csv'):
    fieldnames = ['K', 'time_per_agent', 'throughput_per_agent', 'memory_per_agent', 'gpu_per_agent']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    args = parser.parse_args()
    
    # Run ablation study for different K values
    k_values = [10, 30, 100, 300]
    all_metrics = []
    
    for k in k_values:
        print(f"\nRunning experiment for K={k}")
        ensemble_size = min(k, 30)
        metrics, results = run_experiment(K=k, ensemble_size=ensemble_size)
        all_metrics.append(metrics)
        plot_results(results, K=k)
    
    # Save metrics to CSV
    save_metrics_to_csv(all_metrics)
    print("\nMetrics saved to cartpole_jax_measure.csv")
