# seed_td_lsvi_tabular_jax.py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import psutil
import GPUtil
import time
from collections import defaultdict
import random
import os

# Ensure output directory exists
os.makedirs("bipolar_scale", exist_ok=True)

class BipolarChainEnvJAX:
    def __init__(self, N, theta):
        self.N = N
        self.start = N // 2
        self.theta = theta
        self.revealed = False
        self.revealed_optimal_direction = None

    def get_reward(self, position):
        if position == 0:
            self.revealed = True
            self.revealed_optimal_direction = 'L' if self.theta['L'] > self.theta['R'] else 'R'
            return self.theta['L'], True
        elif position == self.N - 1:
            self.revealed = True
            self.revealed_optimal_direction = 'L' if self.theta['L'] > self.theta['R'] else 'R'
            return self.theta['R'], True
        else:
            return -1, False

class SeedAgentTDJAX:
    def __init__(self, id, N, env, shared_buffer, alpha=0.1, gamma=0.5, lamb=1e-6, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.theta_hat = jax.random.normal(jax.random.PRNGKey(id), (N,))
        self.q = jnp.zeros(N)
        self.noise = jax.random.normal(jax.random.PRNGKey(id + 1000), (10000,))
        self.noise_index = 0

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        q = self.q
        for _ in range(10):
            for (s, r, s_prime) in self.buffer:
                r_perturbed = r + self._next_noise()
                target = r_perturbed + self.gamma * q[s_prime]
                q = q.at[s].add(self.alpha * ((target - q[s]) - (1 / self.lamb) * (q[s] - self.theta_hat[s])))
        self.q = q

        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -jnp.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -jnp.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward

class SeedAgentLSVIJAX:
    def __init__(self, id, N, env, shared_buffer, gamma=0.8, lamb=1e-5, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.gamma = gamma
        self.lamb = lamb
        self.theta_hat = jax.random.normal(jax.random.PRNGKey(id), (N,))
        self.q = jnp.zeros(N)
        self.noise = jax.random.normal(jax.random.PRNGKey(id + 1000), (10000,))
        self.noise_index = 0

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        A = jnp.zeros((self.N, self.N))
        b = jnp.zeros(self.N)

        for (s, r, s_prime) in self.buffer:
            r_perturbed = r + self._next_noise()
            target = r_perturbed + self.gamma * jnp.max(self.q[s_prime])
            A = A.at[s, s].add(1 + (1 / self.lamb))
            b = b.at[s].add(target + (1 / self.lamb) * self.theta_hat[s])

        A += 1e-6 * jnp.eye(self.N)
        self.q = jnp.linalg.solve(A, b)

        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -jnp.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -jnp.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward

def simulate_jax(K, N, H, AgentClass, episodes=30):
    regrets = []
    for _ in range(episodes):
        theta_L = N if random.random() < 0.5 else -N
        theta_R = -theta_L
        r_star = N // 2 if theta_L == N else N // 2 + 1
        theta = {'L': theta_L, 'R': theta_R}
        env = BipolarChainEnvJAX(N, theta)
        buffer = []
        agents = [AgentClass(k, N, env, buffer) for k in range(K)]
        rewards = [agent.act(H) for agent in agents]
        mean_regret = r_star - (sum(rewards) / K)
        regrets.append(mean_regret)
    return np.mean(regrets)

def run_experiments_jax():
    N, H = 50, 100
    Ks = [1, 10, 50, 100]
    results, times, mems, gpus = {}, defaultdict(dict), defaultdict(dict), defaultdict(dict)

    for AgentClass, name in [(SeedAgentTDJAX, 'Seed TD JAX'), (SeedAgentLSVIJAX, 'Seed LSVI JAX')]:
        regrets = []
        for K in Ks:
            print(f"{name}, K={K}")
            p = psutil.Process()
            mem0 = p.memory_info().rss / 1024 / 1024
            gpu0 = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            t0 = time.time()
            regret = simulate_jax(K, N, H, AgentClass)
            t1 = time.time()
            mem1 = p.memory_info().rss / 1024 / 1024
            gpu1 = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            regrets.append(regret)
            times[name][K] = (t1 - t0) / K * 1000
            mems[name][K] = (mem1 - mem0) / K
            gpus[name][K] = (gpu1 - gpu0) / K if GPUtil.getGPUs() else 0
        results[name] = regrets

    def save_csv(name, d):
        with open(f"bipolar_scale/{name}", "w") as f:
            f.write("Algorithm," + ",".join([f"K={k}" for k in Ks]) + "\n")
            for k, v in d.items():
                f.write(k + "," + ",".join([str(v[kk]) for kk in Ks]) + "\n")

    save_csv("bipolar_scale_times_jax.csv", times)
    save_csv("bipolar_scale_memory_jax.csv", mems)
    if GPUtil.getGPUs():
        save_csv("bipolar_scale_gpu_jax.csv", gpus)

    def plot(name, d, ylabel):
        plt.clf()
        for k, v in d.items():
            plt.plot(Ks, [v[kk] for kk in Ks], marker='o', label=k)
        plt.title(name)
        plt.xlabel("K")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid()
        plt.savefig(f"bipolar_scale/{name.replace(' ', '_').lower()}.png")

    plot("JAX Mean Regret", results, "Regret")
    plot("JAX Time per Agent", times, "ms")
    plot("JAX Throughput", {k: {kk: 1 / v[kk] for kk in v} for k, v in times.items()}, "agent/ms")
    plot("JAX Memory", mems, "MB")
    if GPUtil.getGPUs():
        plot("JAX GPU", gpus, "MB")

if __name__ == '__main__':
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    run_experiments_jax()
