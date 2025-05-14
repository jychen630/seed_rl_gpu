# Reimplementation of the Bipolar Chain experiment from Section 4.1 of Dimakopoulou & Van Roy (2018)
# Implements Thompson Resampling, Seed Sampling, and Concurrent UCRL as described in the paper
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import psutil
import GPUtil
import time


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
            z_k = np.random.rand()
            direction = 'L' if z_k < 0.5 else 'R'
            theta_L_hat = N if direction == 'L' else -N
            theta_R_hat = -theta_L_hat
            self.sampled_theta = {'L': theta_L_hat, 'R': theta_R_hat}
            self.fixed_direction = 'L' if self.sampled_theta['L'] > self.sampled_theta['R'] else 'R'
       
    def choose_direction(self):
        if self.algorithm == 'Thompson':
            return 'L' if np.random.rand() < self.prior_p else 'R'
        elif self.algorithm == 'SeedSampling':
            return self.fixed_direction
        elif self.algorithm == 'UCRL':
            return 'R'  # Optimistic choice assumed to be right
    def act(self, H):
        pos = self.env.start
        total_reward = 0
        for step in range(H):
            # Re-sample direction each step if using Thompson
            if self.algorithm == 'Thompson':
                direction = 'L' if np.random.rand() < self.prior_p else 'R'
            elif self.algorithm == 'SeedSampling':
                direction = self.fixed_direction
            elif self.algorithm == 'UCRL':
                direction = 'R'

            # If truth is revealed, override
            if self.env.revealed:
                direction = self.env.revealed_optimal_direction

            next_pos = pos - 1 if direction == 'L' else pos + 1
        
            next_pos = max(0, min(self.env.N - 1, next_pos))
            r, terminal = self.env.get_reward(next_pos)
            total_reward += r
            pos = next_pos

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
            # Initialize agents, states, and logs
            agents = [Agent(k, algorithm, env) for k in range(K)]
            positions = [env.start] * K
            directions = [agent.choose_direction() for agent in agents]
            rewards = [0] * K
            terminated = [False] * K

            for t in range(H):
                for k in range(K):
                    if terminated[k]:
                        continue

                    # If the environment has been revealed, switch direction
                    if env.revealed:
                        directions[k] = env.revealed_optimal_direction

                    next_pos = positions[k] - 1 if directions[k] == 'L' else positions[k] + 1
                    next_pos_adj = max(0, min(N - 1, next_pos))
                    if next_pos_adj != next_pos:
                        raise ValueError(f"algorithm={algorithm}, t={t}, k={k}, next_pos={next_pos}, next_pos_adj={next_pos_adj}")
                        #next_pos = next_pos_adj

                    reward, terminal = env.get_reward(next_pos, k, t)
                    rewards[k] += reward
                    positions[k] = next_pos
                    terminated[k] = terminal
        elif algorithm == 'Thompson':
            agents = [Agent(k, algorithm, env) for k in range(K)]
            rewards = [agent.act(H) for agent in agents]

        
        total_rewards_all_agents = sum(rewards)
        mean_reward = total_rewards_all_agents / K
        mean_regret = r_star - mean_reward
        regrets_all_episodes.append(mean_regret)

      
    bayes_regret = np.mean(regrets_all_episodes)
    return bayes_regret

# Parameters from the paper (Figure 3)
N = 50
H = 100
# Wrong (fixed): common prior that should be with probability 0.5, theta_L = N and theta_R = -theta_L
# TODO: if this probability sampled every agent or sampled once for all agent? i think it's the later
# i.e, the agent share the same setting
# When any of the K agents traverses e_L, or e_R for the first time, all K agents learn the true values of theta_L, theta_r


K_values = [1, 10,  50,  100, 1000, 10000, 100000]#[1, 10, 100, 1000, 10000, 100000] 
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
    
with open("bipolar_toy/bipolar_success_toy_times.csv", "w") as f:
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
with open("bipolar_toy/bipolar_success_toy_memory.csv", "w") as f:
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
    with open("bipolar_toy/bipolar_success_toy_gpu.csv", "w") as f:
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
plt.savefig("bipolar_toy/bipolar_success_toy.png")

plt.clf()
for algo, items in times.items():
    plt.plot(items.keys(), items.values(), marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Time per Agent (ms)")
plt.title("Bipolar Chain: Time per Agent")
plt.legend()
plt.savefig("bipolar_toy/bipolar_success_toy_time_per_agent.png")


plt.clf()
for algo, items in times.items():
    values = np.array(list(items.values()))
    plt.plot(items.keys(), 1/values, marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Agent per millisecond")
plt.title("Bipolar Chain: Throughput (agent/ms)")
plt.legend()
plt.savefig("bipolar_toy/bipolar_success_toy_throughput.png")

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
plt.savefig("bipolar_toy/bipolar_success_toy_memory_per_agent.png")

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
    plt.savefig("bipolar_toy/bipolar_success_toy_gpu_per_agent.png")
