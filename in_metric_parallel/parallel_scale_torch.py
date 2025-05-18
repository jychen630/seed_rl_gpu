import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import GPUtil
import time
from collections import defaultdict
import os

os.makedirs("parallel_scale_torch", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars):
        self.C = C
        self.H = H
        self.prior_vars = prior_vars.to(device)
        self.true_theta = torch.normal(mean=torch.zeros(C, device=device), 
                                     std=torch.sqrt(self.prior_vars))
        self.chain_data = {c: [] for c in range(C)}
        self.total_observations = 0
        self.optimal_reward = torch.max(self.true_theta).item()

    def observe_chains(self, chains):
        # Vectorized observation for multiple chains
        rewards = torch.normal(mean=self.true_theta[chains], 
                             std=torch.ones(len(chains), device=device))
        for i, c in enumerate(chains):
            self.chain_data[c.item()].append(rewards[i].item())
        self.total_observations += len(chains)
        return rewards

class ParallelAgents:
    def __init__(self, K, C, algorithm, lamb=1.0, noise_std=1.0, alpha=0.1, n_iter=10):
        self.K = K
        self.C = C
        self.algorithm = algorithm
        self.lamb = lamb
        self.alpha = alpha
        self.n_iter = n_iter
        
        # Fix: Use correct torch.normal syntax
        self.theta_hat = torch.normal(mean=torch.zeros((K, C), device=device), 
                                    std=torch.ones((K, C), device=device))
        self.noise = torch.normal(mean=torch.zeros((K, 10000), device=device), 
                                std=noise_std * torch.ones((K, 10000), device=device))
        self.noise_indices = torch.zeros(K, dtype=torch.long, device=device)
        self.q = torch.zeros((K, C), device=device)
        self.buffers = [[] for _ in range(K)]

    def _next_noise(self, agent_ids):
        noise_vals = torch.zeros(len(agent_ids), device=device)
        for i, agent_id in enumerate(agent_ids):
            noise_vals[i] = self.noise[agent_id, self.noise_indices[agent_id] % 10000]
            self.noise_indices[agent_id] += 1
        return noise_vals

    def update_td(self, agent_ids, chains, rewards):
        for i, agent_id in enumerate(agent_ids):
            for _ in range(self.n_iter):
                for c, r in self.buffers[agent_id]:
                    r_perturbed = r + self._next_noise([agent_id])[0].item()
                    td_error = r_perturbed - self.q[agent_id, c]
                    self.q[agent_id, c] += self.alpha * (
                        td_error - (1 / self.lamb) * (self.q[agent_id, c] - self.theta_hat[agent_id, c])
                    )

    def update_lsvi(self, agent_ids, chains, rewards):
        for i, agent_id in enumerate(agent_ids):
            A = torch.zeros((self.C, self.C), device=device)
            b = torch.zeros(self.C, device=device)
            
            if len(self.buffers[agent_id]) > 0:
                prev_chains, prev_rewards = zip(*self.buffers[agent_id])
                prev_chains = torch.tensor(prev_chains, device=device)
                prev_rewards = torch.tensor(prev_rewards, device=device)
                noise = self._next_noise([agent_id] * len(prev_rewards))
                rewards_perturbed = prev_rewards + noise
                
                A[prev_chains, prev_chains] += 1 + (1 / self.lamb)
                b[prev_chains] += rewards_perturbed + (1 / self.lamb) * self.theta_hat[agent_id, prev_chains]
            
            A += 1e-6 * torch.eye(self.C, device=device)
            self.q[agent_id] = torch.linalg.solve(A, b)

    def act(self, agent_ids):
        if self.algorithm == "td":
            self.update_td(agent_ids, None, None)
        else:  # lsvi
            self.update_lsvi(agent_ids, None, None)
        return torch.argmax(self.q[agent_ids], dim=1)

def simulate_parallel_chains(K, C, H, prior_vars, algorithm, n_sims=50, batch_size=32):
    all_regrets = []
    for _ in tqdm(range(n_sims)):
        env = ParallelChainsEnv(C, H, prior_vars)
        agents = ParallelAgents(K, C, algorithm)
        total_reward = 0
        cumulative_regret = []
        
        # Process agents in batches
        for k in range(0, K, batch_size):
            batch_size_actual = min(batch_size, K - k)
            agent_ids = list(range(k, k + batch_size_actual))
            
            # Get actions for all agents in batch
            chosen_chains = agents.act(agent_ids)
            
            # Get rewards for all chosen chains
            rewards = env.observe_chains(chosen_chains)
            
            # Update buffers and total reward
            for i, agent_id in enumerate(agent_ids):
                agents.buffers[agent_id].append((chosen_chains[i].item(), rewards[i].item()))
                total_reward += rewards[i].item()
            
            # Calculate regret
            regret = (env.optimal_reward * (k + batch_size_actual)) - total_reward
            cumulative_regret.append(regret / (k + batch_size_actual))
            
        all_regrets.append(cumulative_regret)
    
    mean_regret_per_agent = torch.tensor([regret[-1] for regret in all_regrets], device=device).mean().item()
    cumulative_regret_avg = torch.tensor(all_regrets, device=device).mean(dim=0)
    return mean_regret_per_agent, cumulative_regret_avg

def run_experiments():
    C = 4
    H = 4
    prior_vars = torch.tensor([100 + c for c in range(1, C + 1)], dtype=torch.float32, device=device)
    K_values = [1, 10, 100, 1000]

    algorithms = {
        "Seed TD": "td",
        "Seed LSVI": "lsvi",
    }

    # Initialize dictionaries for tracking metrics
    alg_regrets = {name: [] for name in algorithms}
    times = defaultdict(dict)
    memory_usage = defaultdict(dict)
    gpu_utilization = defaultdict(dict)

    for name, algo in algorithms.items():
        print(f"Simulating {name}...")
        for K in tqdm(K_values):
            # Get initial memory and GPU usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            initial_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            start_time = time.time()
            avg_regret, _ = simulate_parallel_chains(K, C, H, prior_vars, algo, n_sims=30)
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
    with open("parallel_scale_torch/parallel_scale_torch_times.csv", "w") as f:
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
    with open("parallel_scale_torch/parallel_scale_torch_memory.csv", "w") as f:
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
        with open("parallel_scale_torch/parallel_scale_torch_gpu.csv", "w") as f:
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
    plt.savefig("parallel_scale_torch/parallel_seed_td_lsvi.png")

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
    plt.savefig("parallel_scale_torch/parallel_scale_torch_time_per_agent.png")

    # Throughput plot
    plt.clf()
    for algo, items in times.items():
        values = torch.tensor(list(items.values()))
        plt.plot(items.keys(), 1/values, marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Agent per millisecond")
    plt.title("Parallel Chains: Throughput (agent/ms)")
    plt.legend()
    plt.savefig("parallel_scale_torch/parallel_scale_torch_throughput.png")

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
    plt.savefig("parallel_scale_torch/parallel_scale_torch_memory_per_agent.png")

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
        plt.savefig("parallel_scale_torch/parallel_scale_torch_gpu_per_agent.png")

if __name__ == '__main__':
    run_experiments()