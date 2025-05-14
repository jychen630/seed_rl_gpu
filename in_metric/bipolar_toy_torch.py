# Reimplementation of the Bipolar Chain experiment from Section 4.1 of Dimakopoulou & Van Roy (2018)
# Implements Thompson Resampling, Seed Sampling, and Concurrent UCRL as described in the paper
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
sub_dir = "bipolar_toy_torch"
os.makedirs(sub_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BipolarChainEnv:
    def __init__(self, N, theta):
        self.N = N
        self.start = N // 2 # v_s, vertex of start; Each one  of the K agents starts from vertext v_s = N/2
        self.theta = theta  # dict with keys 'L' and 'R'
        self.revealed = False
        self.revealed_optimal_direction = None
        
    def get_reward(self, position, k=-1, t=-1):
        if position == 0:
            self.revealed = True
            self.revealed_optimal_direction = 'L' if self.theta['L'] > self.theta['R'] else 'R'
            # logging.info(f"setting revealed to True by agent {k} at time {t}")
            return self.theta['L'], True
        elif position == self.N - 1:
            self.revealed = True
            self.revealed_optimal_direction = 'L' if self.theta['L'] > self.theta['R'] else 'R'
            # logging.info(f"setting revealed to True by agent {k} at time {t}")
            return self.theta['R'], True
        else:
            return -1, False

class Agent:
    def __init__(self, agent_id, algorithm, env, prior_p=0.5):
        self.id = agent_id
        self.algorithm = algorithm
        self.env = env
        self.prior_p = prior_p # only used for Thompson
        self.sampled_theta = None   # only used for SeedSampling
        self.fixed_direction = None # only used for SeedSampling

        if algorithm == 'SeedSampling':
            z_k = torch.rand(1).item()
            direction = 'L' if z_k < 0.5 else 'R'
            theta_L_hat = N if direction == 'L' else -N
            theta_R_hat = -theta_L_hat
            self.sampled_theta = {'L': theta_L_hat, 'R': theta_R_hat}
            self.fixed_direction = 'L' if self.sampled_theta['L'] > self.sampled_theta['R'] else 'R'
       
    def choose_direction(self):
        if self.algorithm == 'Thompson':
            directions = torch.rand(H, device=device) < self.prior_p
            directions = torch.where(directions, torch.tensor(-1, device=device), torch.tensor(1, device=device))
            return directions
        elif self.algorithm == 'SeedSampling':
            return self.fixed_direction
        elif self.algorithm == 'UCRL':
            return 'R'  # Optimistic choice assumed to be right
    def act(self, H):
        pos = self.env.start
        total_reward = 0
        
        # Pre-allocate tensors for batch processing
        positions = torch.full((H,), pos, device=device)
        rewards = torch.zeros(H, device=device)
        terminated = torch.zeros(H, dtype=torch.bool, device=device)
        
        for step in range(H):
            if self.algorithm == 'Thompson':
                direction = self.choose_direction()[step]
            elif self.algorithm == 'SeedSampling':
                direction = self.fixed_direction
            elif self.algorithm == 'UCRL':
                direction = 'R'

            if self.env.revealed:
                direction = self.env.revealed_optimal_direction

            next_pos = positions[step] + direction
            next_pos = torch.clamp(next_pos, 0, self.env.N - 1)
            r, terminal = self.env.get_reward(next_pos)
            total_reward += r
            pos = next_pos
            
            # Update tensors
            positions[step] = pos
            rewards[step] = r
            terminated[step] = terminal

            if terminal:
                # Agent terminates after reaching either endpoint
                break

        return total_reward

def simulate_bipolar_chain_for_K_agents(K, N, H, algorithm, n_episodes=50):
    regrets_all_episodes = []

    for episode in range(n_episodes):
        theta_L = N if random.random() < 0.5 else -N
        theta_R = -theta_L
        if theta_L == N and theta_R == -N:
            r_star = N // 2
        elif theta_L == -N and theta_R == N:
            r_star = N // 2 + 1

        #logging.info("*" * 20 + f" episode {episode} " + "*" * 20)
        theta = {'L': theta_L, 'R': theta_R}
        env = BipolarChainEnv(N, theta)
        if algorithm == 'UCRL' or algorithm == 'SeedSampling':
            # Initialize tensors for batch processing
            agents = [Agent(k, algorithm, env) for k in range(K)]
            positions = torch.full((K,), env.start, device=device)
            directions = torch.tensor([1 if agent.choose_direction()[0] == 1 else -1 for agent in agents], device=device)
            rewards = torch.zeros(K, device=device)
            terminated = torch.zeros(K, dtype=torch.bool, device=device)

            for t in range(H):
                # Update positions for non-terminated agents
                active_mask = ~terminated
                if active_mask.any():
                    if env.revealed:
                        directions[active_mask] = 1 if env.revealed_optimal_direction == 'R' else -1
                    
                    next_positions = positions[active_mask] + directions[active_mask]
                    next_positions = torch.clamp(next_positions, 0, N-1)
                    
                    # Get rewards and update states
                    for i, (pos, next_pos) in enumerate(zip(positions[active_mask], next_positions)):
                        reward, terminal = env.get_reward(next_pos.item(), i, t)
                        rewards[active_mask][i] += reward
                        positions[active_mask][i] = next_pos
                        terminated[active_mask][i] = terminal

        elif algorithm == 'Thompson':
            # Pre-sample all directions for all agents at once
            directions = torch.rand(K, H, device=device) < 0.5
            directions = torch.where(directions, torch.tensor(-1, device=device), torch.tensor(1, device=device))
            
            # Initialize tensors for all agents
            positions = torch.full((K, H), env.start, device=device)
            rewards = torch.zeros(K, H, device=device)
            terminated = torch.zeros(K, H, dtype=torch.bool, device=device)
            
            # Process all agents in parallel
            for t in range(H):
                if env.revealed:
                    directions[:, t:] = -1 if env.revealed_optimal_direction == 'L' else 1
                
                next_positions = positions[:, t] + directions[:, t]
                next_positions = torch.clamp(next_positions, 0, N-1)
                
                # Update positions and check for termination
                positions[:, t] = next_positions
                for k in range(K):
                    r, terminal = env.get_reward(next_positions[k].item(), k, t)
                    rewards[k, t] = r
                    terminated[k, t] = terminal
                    if terminal:
                        directions[k, t+1:] = 0  # Stop moving after termination

        total_rewards_all_agents = rewards.sum().item()
        mean_reward = total_rewards_all_agents / K
        mean_regret = r_star - mean_reward
        regrets_all_episodes.append(mean_regret)

      
    bayes_regret = torch.tensor(regrets_all_episodes).mean().item()
    return bayes_regret

# Parameters from the paper (Figure 3)
N = 50
H = 100
# Wrong (fixed): common prior that should be with probability 0.5, theta_L = N and theta_R = -theta_L
# TODO: if this probability sampled every agent or sampled once for all agent? i think it's the later
# i.e, the agent share the same setting
# When any of the K agents traverses e_L, or e_R for the first time, all K agents learn the true values of theta_L, theta_r


K_values = [1, 10,  50,  100, 1000]#, 10000, 100000]#[1, 10, 100, 1000, 10000, 100000] 
algorithms = [ 'Thompson', 'SeedSampling', "UCRL"]

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
        regret = simulate_bipolar_chain_for_K_agents(K, N, H, algo, n_episodes=30)
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

# Save memory metrics
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

# Save GPU metrics if available
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

# Plot
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(K_values, results[algo], marker='o', label=algo)

plt.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Reference (y=25)')
plt.axhline(y=125, color='r', linestyle='--', alpha=0.5, label='Reference (y=125)')
plt.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Reference (y=200)')

plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.title("Bipolar Chain: Regret vs. Number of Agents")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{sub_dir}/{sub_dir}_regret.png")

plt.clf()
for algo, items in times.items():
    plt.plot(items.keys(), items.values(), marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Time per Agent (ms)")
plt.title("Bipolar Chain: Time per Agent")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_time_per_agent.png")


plt.clf()
for algo, items in times.items():
    values = torch.tensor(list(items.values()))
    plt.plot(items.keys(), 1/values, marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Agent per millisecond")
plt.title("Bipolar Chain: Throughput (agent/ms)")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_throughput.png")

# Add memory usage plot
plt.clf()
for algo, items in memory_usage.items():
    plt.plot(items.keys(), items.values(), marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Memory Usage per Agent (MB)")
plt.title("Bipolar Chain: Memory Usage per Agent")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_memory_per_agent.png")

# Add GPU utilization plot if available
if GPUtil.getGPUs():
    plt.clf()
    for algo, items in gpu_utilization.items():
        plt.plot(items.keys(), items.values(), marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"{i}" for i in K_values])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("GPU Memory Usage per Agent (MB)")
    plt.title("Bipolar Chain: GPU Memory Usage per Agent")
    plt.legend()
    plt.savefig(f"{sub_dir}/{sub_dir}_gpu_per_agent.png")
