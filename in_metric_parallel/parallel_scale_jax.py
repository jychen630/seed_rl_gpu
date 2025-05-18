import jax
import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import GPUtil
import time
from collections import defaultdict
import os
import numpy as np

os.makedirs("parallel_scale_jax", exist_ok=True)

# Set device
print(f"Using device: {jax.default_backend()}")

class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars, key):
        self.C = C
        self.H = H
        self.prior_vars = prior_vars
        key, subkey = jax.random.split(key)
        self.true_theta = jax.random.normal(subkey, (C,)) * jnp.sqrt(prior_vars)
        self.optimal_reward = jnp.max(self.true_theta).item()
        self.key = key
        
    def observe_chains(self, chains, key):
        key, subkey = jax.random.split(key)
        rewards = jax.random.normal(subkey, (len(chains),)) + self.true_theta[chains]
        return rewards, key

class ParallelAgents:
    def __init__(self, K, C, algorithm_type, lamb=1.0, noise_std=1.0, alpha=0.1, n_iter=10, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
            
        self.K = K
        self.C = C
        self.algorithm_type = algorithm_type  # 0 for TD, 1 for LSVI
        self.lamb = lamb
        self.alpha = alpha
        self.n_iter = n_iter
        
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.theta_hat = jax.random.normal(subkey1, (K, C))
        self.noise = jax.random.normal(subkey2, (K, 10000)) * noise_std
        self.noise_indices = jnp.zeros(K, dtype=jnp.int32)
        self.q = jnp.zeros((K, C))
        self.key = key
        
        # Use NumPy arrays for buffers (not part of JIT computation)
        self.buffers = [[] for _ in range(K)]
        
    def _next_noise(self, agent_id):
        noise_idx = self.noise_indices[agent_id] % 10000
        noise_val = self.noise[agent_id, noise_idx]
        self.noise_indices = self.noise_indices.at[agent_id].add(1)
        return noise_val
        
    def update_td(self, agent_ids, key):
        for i, agent_id in enumerate(agent_ids):
            agent_id = int(agent_id)  # Convert to int for indexing
            if not self.buffers[agent_id]:
                continue
                
            for _ in range(self.n_iter):
                for c, r in self.buffers[agent_id]:
                    key, subkey = jax.random.split(key)
                    noise = self._next_noise(agent_id)
                    r_perturbed = r + noise
                    td_error = r_perturbed - self.q[agent_id, c]
                    self.q = self.q.at[agent_id, c].add(self.alpha * (
                        td_error - (1 / self.lamb) * (self.q[agent_id, c] - self.theta_hat[agent_id, c])
                    ))
        return key

    def update_lsvi(self, agent_ids, key):
        for i, agent_id in enumerate(agent_ids):
            agent_id = int(agent_id)  # Convert to int for indexing
            if not self.buffers[agent_id]:
                continue
                
            chains, rewards = zip(*self.buffers[agent_id])
            chains = jnp.array(chains)
            rewards = jnp.array(rewards)
            
            noise = jnp.array([self._next_noise(agent_id) for _ in range(len(rewards))])
            rewards_perturbed = rewards + noise
            
            A = jnp.zeros((self.C, self.C))
            b = jnp.zeros(self.C)
            
            for j, (c, r) in enumerate(zip(chains, rewards_perturbed)):
                A = A.at[c, c].add(1 + (1 / self.lamb))
                b = b.at[c].add(r + (1 / self.lamb) * self.theta_hat[agent_id, c])
            
            A = A + 1e-6 * jnp.eye(self.C)
            self.q = self.q.at[agent_id].set(jnp.linalg.solve(A, b))
        return key
        
    def act(self, agent_ids, key):
        # Convert agent_ids to a JAX array to ensure proper indexing
        agent_ids = jnp.array(agent_ids)
        
        # Update Q values based on algorithm type
        if self.algorithm_type == 0:  # TD
            key = self.update_td(agent_ids, key)
        else:  # LSVI
            key = self.update_lsvi(agent_ids, key)
            
        # Get actions using proper indexing
        return jnp.argmax(self.q[agent_ids], axis=1), key

def simulate_parallel_chains(K, C, H, prior_vars, algorithm_type, key, n_sims=50, batch_size=32):
    """Non-JIT version that works with Python objects for buffers"""
    all_regrets = []
    sim_keys = jax.random.split(key, n_sims)
    
    for sim_idx in tqdm(range(n_sims)):
        sim_key = sim_keys[sim_idx]
        env_key, agent_key = jax.random.split(sim_key)
        
        env = ParallelChainsEnv(C, H, prior_vars, env_key)
        agents = ParallelAgents(K, C, algorithm_type, key=agent_key)
        total_reward = 0
        cumulative_regret = []
        action_key = jax.random.PRNGKey(0)
        
        # Process agents in batches
        for k in range(0, K, batch_size):
            batch_size_actual = min(batch_size, K - k)
            agent_ids = jnp.arange(k, k + batch_size_actual)
            
            # Get actions for all agents in batch
            action_key, subkey = jax.random.split(action_key)
            chosen_chains, action_key = agents.act(agent_ids, subkey)
            
            # Get rewards for all chosen chains
            rewards, env.key = env.observe_chains(chosen_chains, env.key)
            
            # Update buffers and total reward
            for i, agent_id in enumerate(agent_ids):
                agent_id = int(agent_id)  # Convert to int for indexing
                agents.buffers[agent_id].append((int(chosen_chains[i]), float(rewards[i])))
                total_reward += float(rewards[i])
            
            # Calculate regret
            regret = (env.optimal_reward * (k + batch_size_actual)) - total_reward
            cumulative_regret.append(regret / (k + batch_size_actual))
            
        all_regrets.append(cumulative_regret[-1])
    
    mean_regret_per_agent = sum(all_regrets) / len(all_regrets)
    return mean_regret_per_agent, all_regrets

def run_experiments():
    C = 4
    H = 4
    prior_vars = jnp.array([100 + c for c in range(1, C + 1)], dtype=jnp.float32)
    K_values = [1, 10, 100, 1000]
    
    algorithms = {
        "Seed TD": 0,  # 0 for TD
        "Seed LSVI": 1,  # 1 for LSVI
    }
    
    # Initialize metrics tracking
    alg_regrets = {name: [] for name in algorithms}
    times = defaultdict(dict)
    memory_usage = defaultdict(dict)
    gpu_utilization = defaultdict(dict)
    
    master_key = jax.random.PRNGKey(0)
    
    for name, algo in algorithms.items():
        print(f"Simulating {name}...")
        for K in tqdm(K_values):
            # Get initial memory and GPU usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            initial_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            start_time = time.time()
            master_key, subkey = jax.random.split(master_key)
            avg_regret, _ = simulate_parallel_chains(K, C, H, prior_vars, algo, subkey, n_sims=30)
            end_time = time.time()
            
            # Calculate final metrics
            final_memory = process.memory_info().rss / 1024 / 1024
            final_gpu = GPUtil.getGPUs()[0].memoryUsed if GPUtil.getGPUs() else 0
            
            times[name][K] = (end_time - start_time)/K * 1000  # ms per agent
            memory_usage[name][K] = (final_memory - initial_memory)/K  # MB per agent
            gpu_utilization[name][K] = (final_gpu - initial_gpu)/K if GPUtil.getGPUs() else 0  # MB per agent
            
            alg_regrets[name].append(avg_regret)

    # Save time metrics
    with open("parallel_scale_jax/parallel_scale_jax_times.csv", "w") as f:
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
    with open("parallel_scale_jax/parallel_scale_jax_memory.csv", "w") as f:
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
        with open("parallel_scale_jax/parallel_scale_jax_gpu.csv", "w") as f:
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
    plt.savefig("parallel_scale_jax/parallel_seed_td_lsvi.png")

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
    plt.savefig("parallel_scale_jax/parallel_scale_jax_time_per_agent.png")

    # Throughput plot
    plt.clf()
    for algo, items in times.items():
        time_values = np.array(list(items.values()))
        plt.plot(items.keys(), 1/time_values, marker='o', label=algo)
    plt.xscale('log')
    plt.xticks(K_values, labels=[f"$10^{i}" for i in range(len(K_values))])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Agent per millisecond")
    plt.title("Parallel Chains: Throughput (agent/ms)")
    plt.legend()
    plt.savefig("parallel_scale_jax/parallel_scale_jax_throughput.png")

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
    plt.savefig("parallel_scale_jax/parallel_scale_jax_memory_per_agent.png")

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
        plt.savefig("parallel_scale_jax/parallel_scale_jax_gpu_per_agent.png")

if __name__ == '__main__':
    run_experiments()