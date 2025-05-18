import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

        elif algorithm == "UCRL":
            # Optimistic estimate at activation
            self.ucb = env.post_means + np.sqrt(env.post_vars)

        elif algorithm == "SeedSampling":
            self.seed_theta0 = np.random.normal(0, np.sqrt(2 * env.prior_vars))
            self.seed_w = np.random.normal(0, 80, size=env.total_observations + 1000)

    def choose_chain(self):
        if self.algorithm == "Thompson":
            return np.argmax(self.theta_sample)

        elif self.algorithm == "UCRL":
            return np.argmax(self.ucb)

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
    cumulative_regret_avg = np.mean(np.array(all_regrets), axis=0)
    return mean_regret_per_agent, cumulative_regret_avg

# Parameters
C = 10
H = 5
prior_vars = np.array([100 + c for c in range(1, C + 1)])
K_values = [1, 10, 100, 1000]#, 10000]
algorithms = ["SeedSampling", "Thompson", "UCRL"]
alg_regrets = {alg: [] for alg in algorithms}

for alg in algorithms:
    print(f"Simulating {alg}...")
    for K in tqdm(K_values):
        avg_regret, _ = simulate_parallel_chains(K, C, H, prior_vars, alg, n_sims=100)
        alg_regrets[alg].append(avg_regret)

# Plot Figure 4a
plt.figure(figsize=(10, 6))
for alg, reg in alg_regrets.items():
    plt.plot(K_values, reg, marker='o', label=alg)
plt.xscale('log')
plt.xticks(K_values, labels=[f"$10^{i}$" for i in range(len(K_values))])
plt.xlabel("Number of Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.title("Parallel Chains: Mean Regret vs. Number of Agents (Correct)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("parallel5.png")
