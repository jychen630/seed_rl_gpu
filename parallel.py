import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class ParallelChainsEnv:
    def __init__(self, C, H, prior_vars):
        self.C = C  # Number of chains
        self.H = H  # Horizon (steps per chain)
        self.prior_vars = prior_vars  # Prior variance for each chain: σ_c² = 100 + c
        self.true_theta = np.array([np.random.normal(0, np.sqrt(var)) for var in prior_vars])
        self.post_means = np.zeros(C)  # Shared posterior means (prior mean = 0)
        self.post_vars = np.array(prior_vars)  # Shared posterior variances
        self.chain_data = {c: [] for c in range(C)}  # Observed rewards for each chain
        self.total_observations = 0  # Track global observation count
        self.optimal_reward = np.max(self.true_theta)  # R* = max(θ_c)

    def observe_chain(self, c):
        """Simulate traversing chain c (H steps) and observing reward at the end."""
        # After H steps, observe reward r_c ~ N(θ_c, 1)
        reward = np.random.normal(self.true_theta[c], 1)
        self.chain_data[c].append(reward)
        self.total_observations += 1
        
        # Update posterior for chain c (Bayesian Gaussian update)
        n = len(self.chain_data[c])
        self.post_vars[c] = 1 / (1/self.prior_vars[c] + n/1)  # Likelihood σ² = 1
        self.post_means[c] = self.post_vars[c] * (np.sum(self.chain_data[c]) / 1)
        return reward

class Agent:
    def __init__(self, agent_id, algorithm, env):
        self.algorithm = algorithm
        self.env = env  # Access to environment's shared state
        self.activation_time = agent_id  # Assume agents activate in order
        
        # Seed sampling parameters (Section 3.2.3)
        if algorithm == "SeedSampling":
            self.seed_theta0 = np.random.normal(0, np.sqrt(2 * env.prior_vars))
            # Pre-sample noise for perturbations up to current observations
            self.seed_w = np.random.normal(0, 80, size=self.env.total_observations + 1000)
        
    def choose_chain(self):
        if self.algorithm == "SeedSampling":
            # Martingalean-Gaussian Seed Sampling (perturb historical observations)
            O = np.zeros((self.env.total_observations, self.env.C))
            R = np.zeros(self.env.total_observations)
            idx = 0
            for c in range(self.env.C):
                for r in self.env.chain_data[c]:
                    if idx >= self.env.total_observations:
                        break
                    O[idx, c] = 1
                    R[idx] = r + self.seed_w[idx]  # Perturb reward with seed
                    idx += 1
            if idx == 0:
                theta_hat = self.seed_theta0  # No data: use prior sample
            else:
                theta_hat = np.linalg.inv(O[:idx].T @ O[:idx] + np.diag(1/self.env.prior_vars)) @ (O[:idx].T @ R[:idx])
            return np.argmax(theta_hat)
            
        elif self.algorithm == "Thompson":
            # Thompson Resampling: Sample θ_c ~ N(post_means[c], post_vars[c])
            sampled_theta = np.array([
                np.random.normal(self.env.post_means[c], np.sqrt(self.env.post_vars[c]))
                for c in range(self.env.C)
            ])
            return np.argmax(sampled_theta)
            
        elif self.algorithm == "UCRL":
            # Concurrent UCRL: Optimistic value = post_means[c] + sqrt(post_vars[c])
            return np.argmax(self.env.post_means + np.sqrt(self.env.post_vars))

def simulate_parallel_chains(K, C, H, prior_vars, algorithm, n_sims=50):
    all_regrets = []
    for _ in range(n_sims):
        env = ParallelChainsEnv(C, H, prior_vars)
        cumulative_regret = []
        total_reward = 0
        
        for k in range(K):
            agent = Agent(k, algorithm, env)
            c = agent.choose_chain()
            
            # Simulate agent traversing H steps in chain c
            # (Reward observed only at the end)
            reward = env.observe_chain(c)
            total_reward += reward
            
            # Track cumulative regret (ordered by activation time)
            current_regret = (env.optimal_reward * (k + 1)) - total_reward
            cumulative_regret.append(current_regret / (k + 1))  # Mean regret per agent
        
        all_regrets.append(cumulative_regret)
    
    # Average over simulations
    mean_regret_per_agent = np.mean([regret[-1] for regret in all_regrets])
    cumulative_regret_avg = np.mean(np.array(all_regrets), axis=0)
    return mean_regret_per_agent, cumulative_regret_avg

# Parameters (aligned with Figure 4)
C = 10
H = 5
prior_vars = np.array([100 + c for c in range(1, C + 1)])  # σ_c² = 100 + c (c=1,...,10)

# Simulate Figure 4(a)
K_values = [1, 10, 100, 1000] #,10000]  # 10^0 to 10^4
alg_regrets = {"SeedSampling": []}#, "Thompson": [], "UCRL": []}

for alg in alg_regrets.keys():
    print(f"Simulating {alg}...")
    for K in tqdm(K_values):
        avg_regret, _ = simulate_parallel_chains(K, C, H, prior_vars, alg, n_sims=100)
        alg_regrets[alg].append(avg_regret)

# Plot Figure 4(a)
plt.figure(figsize=(10, 6))
for alg, reg in alg_regrets.items():
    plt.plot(K_values, reg, marker='o', label=alg)
plt.xscale('log')
plt.xticks(K_values, labels=[f"$10^{i}$" for i in range(len(K_values))])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.legend()
plt.title("Parallel Chains: Mean Regret vs. Number of Agents (Figure 4a)")
plt.grid()
plt.savefig("parallela.png")

# # Simulate Figure 4(b) (100,000 agents)
# K_large = 100000
# _, cumulative_regret_seed = simulate_parallel_chains(K_large, C, H, prior_vars, "SeedSampling", n_sims=100)
# _, cumulative_regret_ucrl = simulate_parallel_chains(K_large, C, H, prior_vars, "UCRL", n_sims=100)

# # Plot Figure 4(b)
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(K_large), cumulative_regret_seed, label="Seed Sampling", alpha=0.7)
# plt.plot(np.arange(K_large), cumulative_regret_ucrl, label="Concurrent UCRL", alpha=0.7)
# plt.xlabel("Agent Activation Order (Ascending $t_{k,0}$)")
# plt.ylabel("Cumulative Regret")
# plt.legend()
# plt.title("Parallel Chains: Cumulative Regret for 100,000 Agents (Figure 4b)")
# plt.grid()
# plt.savefig("parallelb.png")