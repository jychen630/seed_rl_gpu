import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import psutil
import GPUtil
import time
import os

# Set up directory structure
sub_dir = "parallel_toy_numpy"
os.makedirs(sub_dir, exist_ok=True)

class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars):
        self.C = C  # Number of chains
        self.H = H  # Horizon (steps per chain)
        self.prior_vars = prior_vars
        self.true_theta = np.array([np.random.normal(0, np.sqrt(var)) for var in prior_vars])
        self.chain_data = {c: [] for c in range(C)}  # Observed rewards per chain
        self.total_observations = 0
        self.post_means = np.zeros(C)  # Prior mean = 0
        self.post_vars = np.array(prior_vars)  # Posterior starts as prior
        self.optimal_reward = np.max(self.true_theta)  # R* = max θ_c

    def observe_chain(self, c):
        """Agent traverses H steps in chain c and observes reward at end."""
        reward = np.random.normal(self.true_theta[c], 1)
        self.chain_data[c].append(reward)
        self.total_observations += 1

        # Bayesian Gaussian update for N(μ, σ²) prior and σ²=1 likelihood
        n = len(self.chain_data[c])
        var_post = 1 / (1 / self.prior_vars[c] + n)
        mean_post = var_post * (np.sum(self.chain_data[c]))

        self.post_vars[c] = var_post
        self.post_means[c] = mean_post

        return reward

class Agent:
    def __init__(self, agent_id, algorithm, env):
        self.algorithm = algorithm
        self.env = env

        if algorithm == "Thompson":
            # Sample from posterior at activation
            self.theta_sample = np.random.normal(env.post_means, np.sqrt(env.post_vars))
            self.chosen_chain = np.argmax(self.theta_sample)

        elif algorithm == "UCRL":
            # Optimistic estimate at activation
            self.ucb = env.post_means + np.sqrt(env.post_vars)
            self.chosen_chain = np.argmax(self.ucb)

        elif algorithm == "SeedSampling":
            self.seed_theta0 = np.random.normal(0, np.sqrt(2 * env.prior_vars))
            self.seed_w = np.random.normal(0, 80, size=env.total_observations + 1000)

    def choose_chain(self):
        if self.algorithm == "Thompson":
            return self.chosen_chain

        elif self.algorithm == "UCRL":
            return self.chosen_chain

        elif self.algorithm == "SeedSampling":
            O = np.zeros((self.env.total_observations, self.env.C))
            R = np.zeros(self.env.total_observations)
            idx = 0
            for c in range(self.env.C):
                for r in self.env.chain_data[c]:
                    if idx >= self.env.total_observations:
                        break
                    O[idx, c] = 1
                    R[idx] = r + self.seed_w[idx]
                    idx += 1
            if idx == 0:
                theta_hat = self.seed_theta0
            else:
                theta_hat = np.linalg.inv(O[:idx].T @ O[:idx] + np.diag(1 / self.env.prior_vars)) @ (O[:idx].T @ R[:idx])
            return np.argmax(theta_hat)

def simulate_parallel_chains(K, C, H, prior_vars, algorithm, n_sims=50):
    all_regrets = []

    for _ in range(n_sims):
        env = ParallelChainsEnv(C, H, prior_vars)
        total_reward = 0
        cumulative_regret = []

        for k in range(K):
            agent = Agent(k, algorithm, env)
            chosen_chain = agent.choose_chain()
            reward = env.observe_chain(chosen_chain)
            total_reward += reward

            regret = (env.optimal_reward * (k + 1)) - total_reward
            cumulative_regret.append(regret / (k + 1))  # mean regret per agent

        all_regrets.append(cumulative_regret)

    mean_regret_per_agent = np.mean([regret[-1] for regret in all_regrets])
    return mean_regret_per_agent

# Parameters
C = 10
H = 5
prior_vars = np.array([100 + c for c in range(1, C + 1)])
K_values = [1, 10, 50, 100, 1000]
algorithms = ["SeedSampling", "Thompson", "UCRL"]

times = defaultdict(dict)
memory_usage = defaultdict(dict)
gpu_utilization = defaultdict(dict)

# Run simulation
results = {}
for algo in algorithms:
    regrets = []
    
    for K in tqdm(K_values):
        print(f"Simulating {algo} for {K} agents...")
        
        # Get initial memory and GPU usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        initial_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
        
        start_time = time.time()
        regret = simulate_parallel_chains(K, C, H, prior_vars, algo, n_sims=30)
        end_time = time.time()
        
        # Calculate final memory and GPU usage
        final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
        final_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
        
        # Store metrics
        times[algo][K] = (end_time - start_time)/K * 1000  # ms per agent
        memory_usage[algo][K] = (final_memory - initial_memory)/K  # MB per agent
        gpu_utilization[algo][K] = (final_gpu - initial_gpu)/K if GPUtil.getGPUs() else 0  # MB per agent
        
        regrets.append(regret)
    results[algo] = regrets

# Save metrics to CSV files
with open(f"{sub_dir}/{sub_dir}_times.csv", "w") as f:
    f.write("Algorithm,")
    for K in K_values:
        f.write(f"K={K},")
    f.write("\n")
    
    for algo, items in times.items():
        f.write(f"{algo},")
        for K in K_values:
            f.write(f"{items[K]},")
        f.write("\n")

with open(f"{sub_dir}/{sub_dir}_memory.csv", "w") as f:
    f.write("Algorithm,")
    for K in K_values:
        f.write(f"K={K},")
    f.write("\n")
    
    for algo, items in memory_usage.items():
        f.write(f"{algo},")
        for K in K_values:
            f.write(f"{items[K]},")
        f.write("\n")

if GPUtil.getGPUs():
    with open(f"{sub_dir}/{sub_dir}_gpu.csv", "w") as f:
        f.write("Algorithm,")
        for K in K_values:
            f.write(f"K={K},")
        f.write("\n")
        
        for algo, items in gpu_utilization.items():
            f.write(f"{algo},")
            for K in K_values:
                f.write(f"{items[K]},")
            f.write("\n")

# Plot results
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(K_values, results[algo], marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.title("Parallel Chains: Mean Regret vs. Number of Agents")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{sub_dir}/{sub_dir}_regret.png")

# Plot time per agent
plt.clf()
for algo, items in times.items():
    plt.plot(items.keys(), items.values(), marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Time per Agent (ms)")
plt.title("Parallel Chains: Time per Agent")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_time_per_agent.png")

# Plot throughput
plt.clf()
for algo, items in times.items():
    values = np.array(list(items.values()))
    plt.plot(items.keys(), 1/values, marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Agent per millisecond")
plt.title("Parallel Chains: Throughput (agent/ms)")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_throughput.png")

# Plot memory usage
plt.clf()
for algo, items in memory_usage.items():
    plt.plot(items.keys(), items.values(), marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Memory Usage per Agent (MB)")
plt.title("Parallel Chains: Memory Usage per Agent")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_memory_per_agent.png")

# Plot GPU utilization if available
if GPUtil.getGPUs():
    plt.clf()
    for algo, items in gpu_utilization.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"{i}" for i in K_values])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("GPU Memory Usage per Agent (MB)")
    plt.title("Parallel Chains: GPU Memory Usage per Agent")
    plt.legend()
    plt.savefig(f"{sub_dir}/{sub_dir}_gpu_per_agent.png")
