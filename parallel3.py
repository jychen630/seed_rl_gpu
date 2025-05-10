# Reproducing section 4.2 (Parallel Chains) and figure 4a/4b from Dimakopoulou & Van Roy (2018)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Environment for Parallel Chains
class ParallelChainsEnv:
    def __init__(self, C=10, H=5, mu0=0, sigma0_sq_base=100, obs_noise_sq=1):
        self.C = C
        self.H = H
        self.mu0 = mu0
        self.sigmasq0 = np.array([sigma0_sq_base + c for c in range(1, C+1)])
        self.obs_noise_sq = obs_noise_sq
        self.theta = np.random.normal(mu0, np.sqrt(self.sigmasq0))

    def sample_prior(self, rng):
        return rng.normal(self.mu0, np.sqrt(self.sigmasq0))

    def sample_model_thompson(self, rng, posterior_mean, posterior_var):
        return rng.normal(posterior_mean, np.sqrt(posterior_var))

    def sample_model_standard_gaussian(self, rng, posterior_mean, posterior_cov, z):
        return posterior_mean + posterior_cov @ z

    def sample_model_martingalean_gaussian(self, theta0, O, R, Sigma0, W):
        sigma2 = 1
        A = O.T @ O + sigma2 * np.linalg.inv(Sigma0)
        b = O.T @ (R + W) + sigma2 * np.linalg.inv(Sigma0) @ theta0
        return np.linalg.solve(A, b)

    def observe_reward(self, c):
        return np.random.normal(self.theta[c], np.sqrt(self.obs_noise_sq))


def run_algorithm(K, repeats, algo, mu0, sigma0_sq, Sigma0, O, C, H):
    regrets = []
    for _ in tqdm(range(repeats), desc=f"{algo}"):
        env = ParallelChainsEnv(C=C, H=H)
        theta_star = env.theta.copy()
        agent_rewards = []
        for _ in range(K):
            rng = np.random.default_rng()
            total_reward = 0
            total_reward += 0 * (env.H - 1)

            if algo == 'thompson':
                theta_hat = env.sample_model_thompson(rng, mu0, sigma0_sq)
                chosen_chain = np.argmax(theta_hat)
                final_reward = env.observe_reward(chosen_chain)
            elif algo == 'standard':
                z = rng.normal(size=env.C)
                posterior_cov = np.diag(np.sqrt(sigma0_sq))
                theta_hat = env.sample_model_standard_gaussian(rng, mu0, posterior_cov, z)
                chosen_chain = np.argmax(theta_hat)
                final_reward = env.observe_reward(chosen_chain)
            elif algo == 'martingalean':
                theta0 = rng.normal(mu0, np.sqrt(sigma0_sq))
                chosen_chain = np.argmax(theta0)
                o = O[chosen_chain:chosen_chain+1]
                R = np.array([env.observe_reward(chosen_chain)])
                w = rng.normal(0, np.sqrt(env.obs_noise_sq), size=1)
                theta_hat = env.sample_model_martingalean_gaussian(theta0, o, R, Sigma0, w)
                final_reward = R[0]
            elif algo == 'ucrl':
                chosen_chain = np.argmax(sigma0_sq)
                final_reward = env.observe_reward(chosen_chain)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            total_reward += final_reward
            agent_rewards.append(total_reward)

        regret = np.max(theta_star) - np.mean(agent_rewards)
        regrets.append(regret)
    return np.mean(regrets)


def run_cumulative_regret(K, algo, mu0, sigma0_sq, Sigma0, O, C, H):
    env = ParallelChainsEnv(C=C, H=H)
    theta_star = env.theta.copy()
    agent_rewards = []
    for _ in range(K):
        rng = np.random.default_rng()
        if algo == 'thompson':
            theta_hat = env.sample_model_thompson(rng, mu0, sigma0_sq)
            chosen_chain = np.argmax(theta_hat)
            final_reward = env.observe_reward(chosen_chain)
        elif algo == 'standard':
            z = rng.normal(size=env.C)
            posterior_cov = np.diag(np.sqrt(sigma0_sq))
            theta_hat = env.sample_model_standard_gaussian(rng, mu0, posterior_cov, z)
            chosen_chain = np.argmax(theta_hat)
            final_reward = env.observe_reward(chosen_chain)
        elif algo == 'martingalean':
            theta0 = rng.normal(mu0, np.sqrt(sigma0_sq))
            chosen_chain = np.argmax(theta0)
            o = O[chosen_chain:chosen_chain+1]
            R = np.array([env.observe_reward(chosen_chain)])
            w = rng.normal(0, np.sqrt(env.obs_noise_sq), size=1)
            theta_hat = env.sample_model_martingalean_gaussian(theta0, o, R, Sigma0, w)
            final_reward = R[0]
        elif algo == 'ucrl':
            chosen_chain = np.argmax(sigma0_sq)
            final_reward = env.observe_reward(chosen_chain)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        total_reward = 0 * (H - 1) + final_reward
        regret = np.max(theta_star) - total_reward
        agent_rewards.append(regret)

    return np.cumsum(agent_rewards)


# Simulation setup
C = 10
H = 5
O = np.eye(C)
mu0 = np.zeros(C)
sigma0_sq = np.array([100 + c for c in range(1, C+1)])
Sigma0 = np.diag(sigma0_sq)
num_agents_list = [10**i for i in range(0, 6)]
repeats = 100

results = {algo: [] for algo in ['thompson', 'standard', 'martingalean', 'ucrl']}

for algo in results:
    for K in tqdm(num_agents_list, desc=f"{algo} * num agents: "):
        avg_regret = run_algorithm(K, repeats, algo, mu0, sigma0_sq, Sigma0, O, C, H)
        results[algo].append(avg_regret)

# Plotting Figure 4a progressively
plt.figure(figsize=(8, 5))
for label in results:
    plt.plot(num_agents_list, results[label], label=label)
plt.xscale('log')
plt.xlabel('Number of Agents (log scale)')
plt.ylabel('Mean Regret per Agent')
plt.title('Figure 4a: Mean Regret vs Number of Agents')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("parallel3a.png")

# Plotting Figure 4b (Cumulative Regret over agents)
K_large = 100000
cumulative_rewards = {key: [] for key in results}

for algo in cumulative_rewards:
    cumulative_rewards[algo] = run_cumulative_regret(K_large, algo, mu0, sigma0_sq, Sigma0, O, C, H)

plt.figure(figsize=(8, 5))
for algo in ['thompson', 'standard', 'martingalean', 'ucrl']:
    plt.plot(range(K_large), cumulative_rewards[algo], label=algo)
plt.xlabel('Agent Index (Activation Order)')
plt.ylabel('Cumulative Regret')
plt.title('Figure 4b: Cumulative Regret of 100,000 Agents')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("parallel3b.png")
