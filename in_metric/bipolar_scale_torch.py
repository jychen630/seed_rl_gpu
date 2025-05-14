# seed_td_lsvi_tabular.py
# PyTorch version of Seed TD and Seed LSVI on Bipolar Chain
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import GPUtil
import time
from collections import defaultdict
import os
sub_dir = "bipolar_scale_torch"
os.makedirs(sub_dir, exist_ok=True)
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
            return -1, False

class ThompsonAgent:
    def __init__(self, agent_id, env, prior_p=0.5):
        self.id = agent_id
        self.env = env
        self.prior_p = prior_p

    def act(self, H):
        pos = self.env.start
        total_reward = 0

        for t in range(H):
            direction = 'L' if torch.rand(1).item() < self.prior_p else 'R'
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

        # Move tensors to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.theta_hat = torch.randn(N, device=device)
        self.q = torch.zeros(N, device=device)
        self.noise = torch.randn(10000, device=device) * noise_std
        self.noise_index = 0

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        # Batch process TD updates
        if len(self.buffer) > 0:
            states = torch.tensor([s for s, _, _ in self.buffer], device=self.q.device)
            rewards = torch.tensor([r for _, r, _ in self.buffer], device=self.q.device)
            next_states = torch.tensor([s_prime for _, _, s_prime in self.buffer], device=self.q.device)
            
            # Add noise to rewards
            noise = self.noise[self.noise_index:self.noise_index + len(rewards)]
            self.noise_index = (self.noise_index + len(rewards)) % len(self.noise)
            rewards = rewards + noise

            # Compute targets and updates in batch
            targets = rewards + self.gamma * self.q[next_states]
            updates = self.alpha * (targets - self.q[states] - (1 / self.lamb) * (self.q[states] - self.theta_hat[states]))
            
            # Update Q-values in batch
            self.q.index_add_(0, states, updates)

        pos = self.env.start
        total_reward = 0

        for t in range(H):
            # Get Q-values for left and r    def act(self, H):
        if len(self.buffer) > 0:
            # Convert buffer data to tensors
            states = torch.tensor([s for s, _, _ in self.buffer], device=self.q.device)
            rewards = torch.tensor([r for _, r, _ in self.buffer], device=self.q.device)
            next_states = torch.tensor([s_prime for _, _, s_prime in self.buffer], device=self.q.device)
            
            # Add noise to rewards
            noise = self.noise[self.noise_index:self.noise_index + len(rewards)]
            self.noise_index = (self.noise_index + len(rewards)) % len(self.noise)
            rewards = rewards + noise

            # Initialize A and b matrices
            A = torch.zeros((self.N, self.N), device=self.q.device)
            b = torch.zeros(self.N, device=self.q.device)

            # Compute targets in batch
            targets = rewards + self.gamma * self.q[next_states]

            # Update A and b matrices efficiently
            # Fix: Use proper indexing for scatter_add_
            for s, target in zip(states, targets):
                A[s, s] += 1 + (1/self.lamb)
                b[s] += target + (1/self.lamb) * self.theta_hat[s]

            # Add regularization
            A += 1e-6 * torch.eye(self.N, device=self.q.device)
            
            # Solve the linear system
            self.q = torch.linalg.solve(A, b)

        pos = self.env.start
        total_reward = 0

        for t in range(H):
            # Get Q-values for left and right actions
            left_q = self.q[pos - 1] if pos > 0 else torch.tensor(float('-inf'), device=self.q.device)
            right_q = self.q[pos + 1] if pos < self.N - 1 else torch.tensor(float('-inf'), device=self.q.device)
            
            # Choose action
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            
            # Get reward and update
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos

            if done:
                break

        return total_rewardelf.N, self.N), device=self.q.device)
            b = torch.zeros(self.N, device=self.q.device)

            # Compute targets in batch
            targets = rewards + self.gamma * self.q[next_states]

            # Create sparse update matrices efficiently
            # Create a sparse matrix for A updates
            indices = torch.stack([states, states])  # 2 x batch_size
            values = torch.ones_like(states, dtype=torch.float) * (1 + 1/self.lamb)
            A = torch.sparse_coo_tensor(indices, values, (self.N, self.N)).to_dense()
            
            # Add regularization
            A += 1e-6 * torch.eye(self.N, device=self.q.device)
            
            # Update b vector efficiently
            b.scatter_add_(0, states, targets + (1/self.lamb) * self.theta_hat[states])
            
            # Solve the linear system
            self.q = torch.linalg.solve(A, b)

        pos = self.env.start
        total_reward = 0

        # Pre-allocate tensors for trajectory
        positions = torch.full((H,), pos, device=self.q.device)
        rewards = torch.zeros(H, device=self.q.device)
        terminated = torch.zeros(H, dtype=torch.bool, device=self.q.device)

        for t in range(H):
            # Get Q-values for left and right actions
            left_q = self.q[pos - 1] if pos > 0 else torch.tensor(float('-inf'), device=self.q.device)
            right_q = self.q[pos + 1] if pos < self.N - 1 else torch.tensor(float('-inf'), device=self.q.device)
            
            # Choose action
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            
            # Get reward and update
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            
            # Update trajectory tensors
            positions[t] = pos
            rewards[t] = reward
            terminated[t] = done

            if done:
                break

        return total_reward

def simulate(K, N, H, AgentClass, episodes=30):
    regrets = []
    for ep in range(episodes):
        theta_L = N if random.random() < 0.5 else -N
        theta_R = -theta_L
        if theta_L == N and theta_R == -N:
            r_star = N // 2
        elif theta_L == -N and theta_R == N:
            r_star = N // 2 + 1
        theta = {'L': theta_L, 'R': theta_R}
        env = BipolarChainEnv(N, theta)

        buffer = []
        if AgentClass == SeedAgentTD or AgentClass == SeedAgentLSVI:
            agents = [AgentClass(k, N, env, buffer) for k in range(K)]
        elif AgentClass == ThompsonAgent:
            agents = [AgentClass(k, env) for k in range(K)]
        rewards = [agent.act(H) for agent in agents]
        mean_regret = r_star - (sum(rewards) / K)
        regrets.append(mean_regret)
    return torch.tensor(regrets).mean().item()


def run_experiments():
    N = 50
    H = 100
    Ks = [1, 10, 50, 100]
    results = {}
    
    # Add dictionaries for tracking metrics
    times = defaultdict(dict)
    memory_usage = defaultdict(dict)
    gpu_utilization = defaultdict(dict)

    for AgentClass, name in [(SeedAgentTD, 'Seed TD'), (SeedAgentLSVI, 'Seed LSVI')]:
        regrets = []
        for K in Ks:
            print(f"Running {name} with K={K}")
            
            # Get initial memory and GPU usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            initial_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            start_time = time.time()
            regret = simulate(K, N, H, AgentClass)
            end_time = time.time()
            
            # Calculate final memory and GPU usage
            final_memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
            final_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            # Store metrics
            times[name][K] = (end_time - start_time)/K * 1000  # ms per agent
            memory_usage[name][K] = (final_memory - initial_memory)/K  # MB per agent
            gpu_utilization[name][K] = (final_gpu - initial_gpu)/K if GPUtil.getGPUs() else 0  # MB per agent
            
            regrets.append(regret)
        results[name] = regrets

    # Save time metrics
    with open(f"{sub_dir}/{sub_dir}_times.csv", "w") as f:
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
    with open(f"{sub_dir}/{sub_dir}_memory.csv", "w") as f:
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
        with open(f"{sub_dir}/{sub_dir}_gpu.csv", "w") as f:
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
    #plt.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Reference (y=200)')

    for name, regrets in results.items():
        plt.plot(Ks, regrets, label=name, marker='o')
    #plt.xscale('log')
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Mean Regret per Agent")
    plt.title("Seed TD and Seed LSVI on Bipolar Chain (PyTorch)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{sub_dir}/bipolar_chain_seed_methods_torch.png")

    # Time per agent plot
    plt.clf()
    for algo, items in times.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Time per Agent (ms)")
    plt.title("Bipolar Chain: Time per Agent")
    plt.legend()
    plt.savefig(f"{sub_dir}/{sub_dir}_time_per_agent.png")

    # Throughput plot
    plt.clf()
    for algo, items in times.items():
        values = torch.tensor(list(items.values()))
        plt.plot(items.keys(), 1/values, marker='o', label=algo)
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Agent per millisecond")
    plt.title("Bipolar Chain: Throughput (agent/ms)")
    plt.legend()
    plt.savefig(f"{sub_dir}/{sub_dir}_throughput.png")

    # Memory usage plot
    plt.clf()
    for algo, items in memory_usage.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xticks(Ks, labels=[f"{i}" for i in Ks])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Memory Usage per Agent (MB)")
    plt.title("Bipolar Chain: Memory Usage per Agent")
    plt.legend()
    plt.savefig(f"{sub_dir}/{sub_dir}_memory_per_agent.png")

    # GPU utilization plot if available
    if GPUtil.getGPUs():
        plt.clf()
        for algo, items in gpu_utilization.items():
            plt.plot(items.keys(), items.values(), marker='o', label=algo)
        plt.xticks(Ks, labels=[f"{i}" for i in Ks])
        plt.xlabel("Number of Concurrent Agents (K)")
        plt.ylabel("GPU Memory Usage per Agent (MB)")
        plt.title("Bipolar Chain: GPU Memory Usage per Agent")
        plt.legend()
        plt.savefig(f"{sub_dir}/{s    performance_indicators = {
        'time_per_agent': '↓',      # Lower is better
        'memory_per_agent': '↓',    # Lower is better
        plt.savefig(f"{sub_dir}/{sub_dir}_gpu_per_agent.png")

    performance_indicators = {
        'throughput_per_agent': '↑' # Higher is better
    }ub_dir}_gpu_per_agent.png")


    }
    if data is not None:
