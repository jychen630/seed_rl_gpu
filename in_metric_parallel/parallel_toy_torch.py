import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import psutil
import GPUtil
import time
import os

# Set up directory structure
sub_dir = "parallel_toy_torch"
os.makedirs(sub_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars):
        self.C = C  # Number of chains
        self.H = H  # Horizon (steps per chain)
        self.prior_vars = torch.tensor(prior_vars, device=device)
        self.true_theta = torch.normal(0, torch.sqrt(self.prior_vars))
        self.chain_data = torch.zeros((C, 1000), device=device)  # Pre-allocate for observations
        self.chain_counts = torch.zeros(C, device=device)  # Count of observations per chain
        self.total_observations = 0
        self.post_means = torch.zeros(C, device=device)  # Prior mean = 0
        self.post_vars = self.prior_vars.clone()  # Posterior starts as prior
        self.optimal_reward = torch.max(self.true_theta)  # R* = max θ_c

    def observe_chain(self, c):
        """Agent traverses H steps in chain c and observes reward at end."""
        reward = torch.normal(self.true_theta[c], 1.0)
        self.chain_data[c, self.chain_counts[c].long()] = reward
        self.chain_counts[c] += 1
        self.total_observations += 1

        # Bayesian Gaussian update for N(μ, σ²) prior and σ²=1 likelihood
        n = self.chain_counts[c]
        var_post = 1 / (1 / self.prior_vars[c] + n)
        mean_post = var_post * torch.sum(self.chain_data[c, :n.long()])

        self.post_vars[c] = var_post
        self.post_means[c] = mean_post

        return reward

class Agent:
    def __init__(self, agent_id, algorithm, env):
        self.algorithm = algorithm
        self.env = env
        self.key = torch.randint(0, 1000000, (1,), device=device)

        if algorithm == "Thompson":
            # Sample from posterior at activation
            self.theta_sample = torch.normal(self.env.post_means, torch.sqrt(self.env.post_vars))
            self.chosen_chain = torch.argmax(self.theta_sample)

        elif algorithm == "UCRL":
            # Optimistic estimate at activation
            self.ucb = self.env.post_means + torch.sqrt(self.env.post_vars)
            self.chosen_chain = torch.argmax(self.ucb)

        elif algorithm == "SeedSampling":
            self.seed_theta0 = torch.normal(0, torch.sqrt(2 * self.env.prior_vars))
            self.seed_w = torch.normal(0, 80, size=(self.env.total_observations + 1000,), device=device)

    def choose_chain(self):
        if self.algorithm == "Thompson":
            return self.chosen_chain

        elif self.algorithm == "UCRL":
            return self.chosen_chain

        elif self.algorithm == "SeedSampling":
            O = torch.zeros((self.env.total_observations, self.env.C), device=device)
            R = torch.zeros(self.env.total_observations, device=device)
            idx = 0
            for c in range(self.env.C):
                n = self.env.chain_counts[c].long()
                if n > 0:
                    O[idx:idx+n, c] = 1
                    R[idx:idx+n] = self.env.chain_data[c, :n] + self.seed_w[idx:idx+n]
                    idx += n
            if idx == 0:
                theta_hat = self.seed_theta0
            else:
                theta_hat = torch.linalg.inv(O[:idx].T @ O[:idx] + torch.diag(1 / self.env.prior_vars)) @ (O[:idx].T @ R[:idx])
            return torch.argmax(theta_hat)

def simulate_parallel_chains(K, C, H, prior_vars, algorithm, n_sims=50):
    all_regrets = []

    for _ in range(n_sims):
        env = ParallelChainsEnv(C, H, prior_vars)
        total_reward = 0
        cumulative_regret = []

        # Pre-allocate tensors for batch processing
        rewards = torch.zeros(K, device=device)
        chosen_chains = torch.zeros(K, dtype=torch.long, device=device)

        for k in range(K):
            agent = Agent(k, algorithm, env)
            chosen_chain = agent.choose_chain()
            reward = env.observe_chain(chosen_chain)
            total_reward += reward

            regret = (env.optimal_reward * (k + 1)) - total_reward
            cumulative_regret.append(regret / (k + 1))  # mean regret per agent

        all_regrets.append(cumulative_regret)

    mean_regret_per_agent = torch.tensor([regret[-1] for regret in all_regrets]).mean().item()
    return mean_regret_per_agent

# Parameters
C = 10
H = 5
prior_vars = torch.tensor([100 + c for c in range(1, C + 1)], device=device)
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
    values = torch.tensor(list(items.values()))
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

