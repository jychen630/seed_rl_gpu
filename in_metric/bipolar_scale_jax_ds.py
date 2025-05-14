# seed_td_lsvi_tabular_jax.py
import jax
import jax.numpy as jnp
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import GPUtil
import numpy as np
import time
from collections import defaultdict

jax.config.update('jax_platform_name', 'cpu')  # Change to 'gpu' if GPU is available

class BipolarChainEnv:
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
            return -1.0, False

class ThompsonAgent:
    def __init__(self, agent_id, env, prior_p=0.5):
        self.id = agent_id
        self.env = env
        self.prior_p = prior_p
        self.key = jax.random.PRNGKey(agent_id)

    def act(self, H):
        pos = self.env.start
        total_reward = 0.0

        for t in range(H):
            self.key, subkey = jax.random.split(self.key)
            direction = 'L' if jax.random.uniform(subkey) < self.prior_p else 'R'
            if self.env.revealed:
                direction = self.env.revealed_optimal_direction

            next_pos = pos - 1 if direction == 'L' else pos + 1
            next_pos = max(0, min(self.env.N - 1, next_pos))
            reward, done = self.env.get_reward(next_pos)
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward

class SeedAgentTD:
    def __init__(self, id, N, env, shared_buffer, alpha=0.1, gamma=0.5, lamb=0.000001, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.noise_std = noise_std

        self.key = jax.random.PRNGKey(id)
        self.theta_hat = jax.random.normal(self.key, (N,))
        self.q = jnp.zeros(N)
        self.key = jax.random.split(self.key)[0]

    @staticmethod
    @jax.jit
    def _td_update(q, theta_hat, s, r, s_prime, key, alpha, gamma, lamb, noise_std):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (s.shape[0],)) * noise_std
        r_perturbed = r + noise
        targets = r_perturbed + gamma * q[s_prime]
        updates = alpha * (targets - q[s] - (1/lamb) * (q[s] - theta_hat[s]))
        q = q.at[s].add(updates)
        return q, key

    def act(self, H):
        if len(self.buffer) > 0:
            # Convert buffer contents to JAX arrays
            s_list, r_list, s_prime_list = zip(*self.buffer)
            s_array = jnp.array(s_list, dtype=jnp.int32)
            r_array = jnp.array(r_list)
            s_prime_array = jnp.array(s_prime_list, dtype=jnp.int32)
            
            for _ in range(10):
                self.q, self.key = self._td_update(
                    self.q, self.theta_hat,
                    s_array, r_array, s_prime_array,
                    self.key,
                    self.alpha, self.gamma, self.lamb, self.noise_std
                )

        pos = self.env.start
        total_reward = 0.0

        for t in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -jnp.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -jnp.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward

class SeedAgentLSVI:
    def __init__(self, id, N, env, shared_buffer, gamma=0.8, lamb=0.00001, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.gamma = gamma
        self.lamb = lamb
        self.noise_std = noise_std

        self.key = jax.random.PRNGKey(id)
        self.theta_hat = jax.random.normal(self.key, (N,))
        self.q = jnp.zeros(N)
        self.key = jax.random.split(self.key)[0]

    @staticmethod
    @jax.jit
    def _lsvi_update(q, theta_hat, s, r, s_prime, key, gamma, lamb, noise_std):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (s.shape[0],)) * noise_std
        r_perturbed = r + noise
        targets = r_perturbed + gamma * jnp.max(q[s_prime], axis=1)
        
        A = jnp.zeros((q.shape[0], q.shape[0]))
        b = jnp.zeros(q.shape[0])
        
        A = A.at[s, s].add(1 + 1/lamb)
        b = b.at[s].add(targets + (1/lamb) * theta_hat[s])
        A = A + 1e-6 * jnp.eye(q.shape[0])
        
        return jnp.linalg.solve(A, b), key

    def act(self, H):
        if len(self.buffer) > 0:
            s_list, r_list, s_prime_list = zip(*self.buffer)
            s_array = jnp.array(s_list, dtype=jnp.int32)
            r_array = jnp.array(r_list)
            s_prime_array = jnp.array(s_prime_list, dtype=jnp.int32)
            
            self.q, self.key = self._lsvi_update(
                self.q, self.theta_hat,
                s_array, r_array, s_prime_array,
                self.key,
                self.gamma, self.lamb, self.noise_std
            )

        pos = self.env.start
        total_reward = 0.0
        for t in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -jnp.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -jnp.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward

def simulate(K, N, H, AgentClass, episodes=30):
    regrets = []
    for ep in range(episodes):
        theta_L = N if random.random() < 0.5 else -N
        theta_R = -theta_L
        r_star = N // 2 if theta_L == N else N // 2 + 1
        theta = {'L': theta_L, 'R': theta_R}
        env = BipolarChainEnv(N, theta)

        buffer = []
        if AgentClass in [SeedAgentTD, SeedAgentLSVI]:
            agents = [AgentClass(k, N, env, buffer) for k in range(K)]
        elif AgentClass == ThompsonAgent:
            agents = [ThompsonAgent(k, env) for k in range(K)]
        rewards = [agent.act(H) for agent in agents]
        mean_regret = r_star - (sum(rewards) / K)
        regrets.append(mean_regret)
    return jnp.mean(jnp.array(regrets))

def run_experiments():
    N = 50
    H = 100
    Ks = [1, 10, 50, 100]
    results = {}
    
    times = defaultdict(dict)
    memory_usage = defaultdict(dict)
    gpu_utilization = defaultdict(dict)

    for AgentClass, name in [(SeedAgentTD, 'Seed TD')]:#, (SeedAgentLSVI, 'Seed LSVI')]:
        regrets = []
        for K in Ks:
            print(f"Running {name} with K={K}")
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            initial_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            start_time = time.time()
            regret = simulate(K, N, H, AgentClass)
            end_time = time.time()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            final_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            times[name][K] = (end_time - start_time)/K * 1000
            memory_usage[name][K] = (final_memory - initial_memory)/K
            gpu_utilization[name][K] = (final_gpu - initial_gpu)/K if GPUtil.getGPUs() else 0
            
            regrets.append(regret)
        results[name] = regrets
    # Save time metrics
    with open("bipolar_scale_jax_ds/bipolar_scale_times_jax_ds.csv", "w") as f:
        f.write("Algorithm,")
        for K in Ks:
            f.write(f"K={K},")
        f.write("\n")
        
        for algo, items in times.items():
            f.write(f"{algo},")
            for K in Ks:
                f.write(f"{items[K]},")
            f.write("\n")

    # Save memory metrics
    with open("bipolar_scale_jax_ds/bipolar_scale_memory_jax_ds.csv", "w") as f:
        f.write("Algorithm,")
        for K in Ks:
            f.write(f"K={K},")
        f.write("\n")
        
        for algo, items in memory_usage.items():
            f.write(f"{algo},")
            for K in Ks:
                f.write(f"{items[K]},")
            f.write("\n")

    # Save GPU metrics if available
    if GPUtil.getGPUs():
        with open("bipolar_scale_jax_ds/bipolar_scale_gpu_jax_ds.csv", "w") as f:
            f.write("Algorithm,")
            for K in Ks:
                f.write(f"K={K},")
            f.write("\n")
            
            for algo, items in gpu_utilization.items():
                f.write(f"{algo},")
                for K in Ks:
                    f.write(f"{items[K]},")
                f.write("\n")

    # Original regret plot
    plt.figure(figsize=(8, 5))
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Reference (y=25)')
    plt.axhline(y=55, color='r', linestyle='--', alpha=0.5, label='Reference (y=125)')

    for name, regrets in results.items():
        plt.plot(Ks, regrets, label=name, marker='o')
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Mean Regret per Agent")
    plt.title("Seed TD and Seed LSVI on Bipolar Chain (JAX)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bipolar_scale_jax_ds/bipolar_chain_seed_methods_jax_ds.png")

    # Time per agent plot
    plt.clf()
    for algo, items in times.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Time per Agent (ms)")
    plt.title("Bipolar Chain: Time per Agent (JAX)")
    plt.legend()
    plt.savefig("bipolar_scale_jax_ds/bipolar_scale_time_per_agent_jax_ds.png")

    # Throughput plot
    plt.clf()
    for algo, items in times.items():
        values = np.array(list(items.values()))
        plt.plot(items.keys(), 1/values, marker='o', label=algo)
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Agent per millisecond")
    plt.title("Bipolar Chain: Throughput (agent/ms) (JAX)")
    plt.legend()
    plt.savefig("bipolar_scale_jax_ds/bipolar_scale_throughput_jax_ds.png")

    # Memory usage plot
    plt.clf()
    for algo, items in memory_usage.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Memory Usage per Agent (MB)")
    plt.title("Bipolar Chain: Memory Usage per Agent (JAX)")
    plt.legend()
    plt.savefig("bipolar_scale_jax_ds/bipolar_scale_memory_per_agent_jax_ds.png")

    # GPU utilization plot if available
    if GPUtil.getGPUs():
        plt.clf()
        for algo, items in gpu_utilization.items():
            plt.plot(items.keys(), items.values(), marker='o', label=algo)
        plt.xticks(Ks, labels=[f"{i}" for i in Ks])
        plt.xlabel("Number of Concurrent Agents (K)")
        plt.ylabel("GPU Memory Usage per Agent (MB)")
        plt.title("Bipolar Chain: GPU Memory Usage per Agent (JAX)")
        plt.legend()
        plt.savefig("bipolar_scale_jax_ds/bipolar_scale_gpu_per_agent_jax_ds.png")


if __name__ == '__main__':
    run_experiments()