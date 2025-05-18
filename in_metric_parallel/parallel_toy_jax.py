import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import psutil
import GPUtil
import time
import random
import os

# Define a subdirectory for saving results
sub_dir = "parallel_toy_jax"
os.makedirs(sub_dir, exist_ok=True)

class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars):
        self.C = C  # Number of chains
        self.H = H  # Horizon (steps per chain)
        self.prior_vars = prior_vars
        self.true_theta = jnp.array([jax.random.normal(jax.random.PRNGKey(c), shape=()) * jnp.sqrt(var) for c, var in enumerate(prior_vars)])
        self.chain_data = {c: [] for c in range(C)}  # Observed rewards per chain
        self.total_observations = 0
        self.post_means = jnp.zeros(C)  # Prior mean = 0
        self.post_vars = jnp.array(prior_vars)  # Posterior starts as prior
        self.optimal_reward = jnp.max(self.true_theta)  # R* = max θ_c

    def observe_chain(self, c):
        """Agent traverses H steps in chain c and observes reward at end."""
        # Convert c to Python int to ensure it's hashable for dictionary access
        c_int = int(c)
        
        key = jax.random.PRNGKey(self.total_observations)
        reward = jax.random.normal(key) + self.true_theta[c_int]
        # Convert JAX array to float before appending
        reward_float = float(reward)
        self.chain_data[c_int].append(reward_float)
        self.total_observations += 1

        # Bayesian Gaussian update for N(μ, σ²) prior and σ²=1 likelihood
        n = len(self.chain_data[c_int])
        var_post = 1 / (1 / self.prior_vars[c_int] + n)
        mean_post = var_post * (jnp.sum(jnp.array(self.chain_data[c_int])))

        self.post_vars = self.post_vars.at[c_int].set(var_post)
        self.post_means = self.post_means.at[c_int].set(mean_post)

        return reward_float  # Return the float value

class Agent:
    def __init__(self, agent_id, algorithm, env):
        self.algorithm = algorithm
        self.env = env
        self.key = jax.random.PRNGKey(agent_id)

        if algorithm == "Thompson":
            # Sample from posterior at activation
            self.key, subkey = jax.random.split(self.key)
            self.theta_sample = jax.random.normal(subkey, shape=(env.C,)) * jnp.sqrt(env.post_vars) + env.post_means
            self.chosen_chain = jnp.argmax(self.theta_sample)

        elif algorithm == "UCRL":
            # Optimistic estimate at activation
            self.ucb = env.post_means + jnp.sqrt(env.post_vars)
            self.chosen_chain = jnp.argmax(self.ucb)

        elif algorithm == "SeedSampling":
            self.key, subkey = jax.random.split(self.key)
            self.seed_theta0 = jax.random.normal(subkey, shape=(env.C,)) * jnp.sqrt(2 * env.prior_vars)
            self.key, subkey = jax.random.split(self.key)
            self.seed_w = jax.random.normal(subkey, shape=(env.total_observations + 1000,)) * 80

    def choose_chain(self):
        if self.algorithm == "Thompson":
            return int(self.chosen_chain)  # Convert JAX array to Python int

        elif self.algorithm == "UCRL":
            return int(self.chosen_chain)  # Convert JAX array to Python int

        elif self.algorithm == "SeedSampling":
            O = jnp.zeros((self.env.total_observations, self.env.C))
            R = jnp.zeros(self.env.total_observations)
            idx = 0
            for c in range(self.env.C):
                for r in self.env.chain_data[c]:
                    if idx >= self.env.total_observations:
                        break
                    O = O.at[idx, c].set(1)
                    R = R.at[idx].set(r + self.seed_w[idx])
                    idx += 1
            if idx == 0:
                theta_hat = self.seed_theta0
            else:
                theta_hat = jnp.linalg.inv(O[:idx].T @ O[:idx] + jnp.diag(1 / self.env.prior_vars)) @ (O[:idx].T @ R[:idx])
            return int(jnp.argmax(theta_hat))  # Convert JAX array to Python int

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

    mean_regret_per_agent = jnp.mean(jnp.array([regret[-1] for regret in all_regrets]))
    cumulative_regret_avg = jnp.mean(jnp.array(all_regrets), axis=0)
    return mean_regret_per_agent, cumulative_regret_avg

# Parameters
C = 10
H = 5
prior_vars = jnp.array([100 + c for c in range(1, C + 1)])
K_values = [1, 10, 50, 100]#, 1000]
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
        regret, _ = simulate_parallel_chains(K, C, H, prior_vars, algo, n_sims=30)
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
plt.title("Parallel Chains (JAX): Mean Regret vs. Number of Agents")
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
plt.title("Parallel Chains (JAX): Time per Agent")
plt.legend()
plt.savefig(f"{sub_dir}/{sub_dir}_time_per_agent.png")

# Plot throughput
plt.clf()
for algo, items in times.items():
    values = jnp.array(list(items.values()))
    plt.plot(items.keys(), 1/values, marker='o', label=algo)
plt.xscale('log')
plt.xticks(K_values, labels=[f"{i}" for i in K_values])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Agent per millisecond")
plt.title("Parallel Chains (JAX): Throughput (agent/ms)")
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
plt.title("Parallel Chains (JAX): Memory Usage per Agent")
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
    plt.title("Parallel Chains (JAX): GPU Memory Usage per Agent")
    plt.legend()
    plt.savefig(f"{sub_dir}/{sub_dir}_gpu_per_agent.png")
