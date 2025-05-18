import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import GPUtil
import time
from collections import defaultdict
import os

os.makedirs("parallel_scale_numpy", exist_ok=True)
class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars):
        self.C = C
        self.H = H
        self.prior_vars = prior_vars
        self.true_theta = np.random.normal(0, np.sqrt(prior_vars))
        self.chain_data = {c: [] for c in range(C)}
        self.total_observations = 0
        self.optimal_reward = np.max(self.true_theta)

    def observe_chain(self, c):
        reward = np.random.normal(self.true_theta[c], 1)
        self.chain_data[c].append(reward)
        self.total_observations += 1
        return reward

class SeedTDParallelAgent:
    def __init__(self, id, env, buffer, lamb=1.0, noise_std=1.0, alpha=0.1, n_iter=10):
        self.id = id
        self.env = env
        self.buffer = buffer
        self.theta_hat = np.random.normal(0, 1.0, env.C)
        self.noise = np.random.normal(0, noise_std, 10000)
        self.noise_index = 0
        self.lamb = lamb
        self.alpha = alpha
        self.n_iter = n_iter
        self.q = np.zeros(env.C)

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def update(self):
        for _ in range(self.n_iter):
            for (c, r) in self.buffer:
                r_perturbed = r + self._next_noise()
                td_error = r_perturbed - self.q[c]
                self.q[c] += self.alpha * (td_error - (1 / self.lamb) * (self.q[c] - self.theta_hat[c]))

    def act(self):
        self.update()
        return np.argmax(self.q)

class SeedLSVIParallelAgent:
    def __init__(self, id, env, buffer, lamb=1.0, noise_std=1.0):
        self.id = id
        self.env = env
        self.buffer = buffer
        self.theta_hat = np.random.normal(0, 1.0, env.C)
        self.noise = np.random.normal(0, noise_std, 10000)
        self.noise_index = 0
        self.lamb = lamb

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def update(self):
        A = np.zeros((self.env.C, self.env.C))
        b = np.zeros(self.env.C)
        for (c, r) in self.buffer:
            r_perturbed = r + self._next_noise()
            A[c, c] += 1 + (1 / self.lamb)
            b[c] += r_perturbed + (1 / self.lamb) * self.theta_hat[c]
        A += 1e-6 * np.eye(self.env.C)
        self.q = np.linalg.solve(A, b)

    def act(self):
        self.update()
        return np.argmax(self.q)

def simulate_parallel_chains(K, C, H, prior_vars, AgentClass, n_sims=50):
    all_regrets = []
    for _ in tqdm(range(n_sims)):
        env = ParallelChainsEnv(C, H, prior_vars)
        buffer = []
        total_reward = 0
        cumulative_regret = []
        for k in range(K):
            agent = AgentClass(k, env, buffer)
            chosen_chain = agent.act()
            reward = env.observe_chain(chosen_chain)
            buffer.append((chosen_chain, reward))
            total_reward += reward
            regret = (env.optimal_reward * (k + 1)) - total_reward
            cumulative_regret.append(regret / (k + 1))
        all_regrets.append(cumulative_regret)
    mean_regret_per_agent = np.mean([regret[-1] for regret in all_regrets])
    cumulative_regret_avg = np.mean(np.array(all_regrets), axis=0)
    return mean_regret_per_agent, cumulative_regret_avg

def run_experiments():
    C = 4
    H = 4
    prior_vars = np.array([100 + c for c in range(1, C + 1)])
    K_values = [1, 10, 100, 1000]

    algorithms = {
        "Seed TD": SeedTDParallelAgent,
        "Seed LSVI": SeedLSVIParallelAgent,
    }

    # Initialize dictionaries for tracking metrics
    alg_regrets = {name: [] for name in algorithms}
    times = defaultdict(dict)
    memory_usage = defaultdict(dict)
    gpu_utilization = defaultdict(dict)

    for name, cls in algorithms.items():
        print(f"Simulating {name}...")
        for K in tqdm(K_values):
            # Get initial memory and GPU usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            initial_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            start_time = time.time()
            avg_regret, _ = simulate_parallel_chains(K, C, H, prior_vars, cls, n_sims=30)
            end_time = time.time()
            
            # Calculate final memory and GPU usage
            final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            final_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            # Store metrics
            times[name][K] = (end_time - start_time)/K * 1000  # ms per agent
            memory_usage[name][K] = (final_memory - initial_memory)/K  # MB per agent
            gpu_utilization[name][K] = (final_gpu - initial_gpu)/K if GPUtil.getGPUs() else 0  # MB per agent
            
            alg_regrets[name].append(avg_regret)

    # Save time metrics
    with open("parallel_scale_numpy/parallel_scale_numpy_times.csv", "w") as f:
        f.write("Algorithm,")
        for K in K_values:
            f.write(f"K={K},")
        f.write("\n")
        
        for algo, items in times.items():
            f.write(f"{algo},")
            for K in K_values:
                f.write(f"{items[K]},")
            f.write("\n")

    # Save memory metrics
    with open("parallel_scale_numpy/parallel_scale_numpy_memory.csv", "w") as f:
        f.write("Algorithm,")
        for K in K_values:
            f.write(f"K={K},")
        f.write("\n")
        
        for algo, items in memory_usage.items():
            f.write(f"{algo},")
            for K in K_values:
                f.write(f"{items[K]},")
            f.write("\n")

    # Save GPU metrics if available
    if GPUtil.getGPUs():
        with open("parallel_scale_numpy/parallel_scale_numpy_gpu.csv", "w") as f:
            f.write("Algorithm,")
            for K in K_values:
                f.write(f"K={K},")
            f.write("\n")
            
            for algo, items in gpu_utilization.items():
                f.write(f"{algo},")
                for K in K_values:
                    f.write(f"{items[K]},")
                f.write("\n")

    # Original regret plot
    plt.figure(figsize=(10, 6))
    for name, reg in alg_regrets.items():
        plt.plot(K_values, reg, marker='o', label=name)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Mean Regret per Agent")
    plt.title("Parallel Chains: Seed TD & Seed LSVI")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("parallel_scale_numpy/parallel_seed_td_lsvi.png")

    # Time per agent plot
    plt.clf()
    for algo, items in times.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Time per Agent (ms)")
    plt.title("Parallel Chains: Time per Agent")
    plt.legend()
    plt.savefig("parallel_scale_numpy/parallel_scale_numpy_time_per_agent.png")

    # Throughput plot
    plt.clf()
    for algo, items in times.items():
        values = np.array(list(items.values()))
        plt.plot(items.keys(), 1/values, marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Agent per millisecond")
    plt.title("Parallel Chains: Throughput (agent/ms)")
    plt.legend()
    plt.savefig("parallel_scale_numpy/parallel_scale_numpy_throughput.png")

    # Memory usage plot
    plt.clf()
    for algo, items in memory_usage.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Memory Usage per Agent (MB)")
    plt.title("Parallel Chains: Memory Usage per Agent")
    plt.legend()
    plt.savefig("parallel_scale_numpy/parallel_scale_numpy_memory_per_agent.png")

    # GPU utilization plot if available
    if GPUtil.getGPUs():
        plt.clf()
        for algo, items in gpu_utilization.items():
            plt.plot(items.keys(), items.values(), marker='o', label=algo)
        plt.xscale('log')
        plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
        plt.xlabel("Number of Agents (K)")
        plt.ylabel("GPU Memory Usage per Agent (MB)")
        plt.title("Parallel Chains: GPU Memory Usage per Agent")
        plt.legend()
        plt.savefig("parallel_scale_numpy/parallel_scale_numpy_gpu_per_agent.png")

if __name__ == '__main__':
    run_experiments()
