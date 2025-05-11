# Reimplementation of the Bipolar Chain experiment from Section 4.1 of Dimakopoulou & Van Roy (2018)
# Implements Thompson Resampling, Seed Sampling, and Concurrent UCRL as described in the paper

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BipolarChainEnv:
    def __init__(self, N, theta):
        self.N = N
        self.mid = N // 2
        self.theta = theta  # dict with keys 'L' and 'R'
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

class Agent:
    def __init__(self, agent_id, algorithm, env, prior_p=0.5):
        self.id = agent_id
        self.algorithm = algorithm
        self.env = env
        self.prior_p = prior_p
        self.fixed_direction = None
        if algorithm == 'SeedSampling':
            self.fixed_direction = 'L' if np.random.rand() < 0.5 else 'R'

    def choose_direction(self):
        if self.algorithm == 'Thompson':
            return 'L' if np.random.rand() < self.prior_p else 'R'
        elif self.algorithm == 'SeedSampling':
            return self.fixed_direction
        elif self.algorithm == 'UCRL':
            return 'R'  # Optimistic choice assumed to be right
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def act(self, H):
        pos = self.env.mid
        total_reward = 0
        direction = self.choose_direction()

        for _ in range(H):
            if self.algorithm in ['SeedSampling', 'UCRL'] and self.env.revealed:
                direction = self.env.revealed_optimal_direction

            next_pos = pos - 1 if direction == 'L' else pos + 1
            next_pos = max(0, min(self.env.N - 1, next_pos))
            reward, terminal = self.env.get_reward(next_pos)
            total_reward += reward
            pos = next_pos
            if terminal:
                break
            else:
                total_reward += -1  # per-step penalty

        return total_reward

def simulate_bipolar_chain(K, N, H, theta_L, theta_R, algorithm, n_sims=50):
    regrets = []
    optimal_reward = max(
        - (N // 2) + theta_L,              # to left endpoint
        - (N - 1 - N // 2) + theta_R       # to right endpoint
    )
    
    for _ in range(n_sims):
        theta = {'L': theta_L, 'R': theta_R}
        env = BipolarChainEnv(N, theta)
        total_reward = 0

        for k in range(K):
            agent = Agent(k, algorithm, env)
            r = agent.act(H)
            total_reward += r

        mean_reward = total_reward / K
        mean_regret = optimal_reward - mean_reward
        regrets.append(mean_regret)

    return np.mean(regrets)

# Parameters from the paper (Figure 3)
N = 100
H = 150
theta_L, theta_R = 100, -100
K_values = [1, 10, 100, 1000, 10000]
algorithms = ['Thompson', 'SeedSampling', 'UCRL']

# Run simulation
results = {}
for algo in algorithms:
    print(f"Simulating {algo}...")
    regrets = []
    for K in K_values:
        r = simulate_bipolar_chain(K, N, H, theta_L, theta_R, algo, n_sims=50)
        regrets.append(r)
    results[algo] = regrets

# Plot
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(K_values, results[algo], marker='o', label=algo)
print(results)
plt.xscale('log')
plt.xticks(K_values, labels=[f"$10^{{{i}}}$" for i in range(5)])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.title("Bipolar Chain: Regret vs. Number of Agents")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig("bipolar5.png")
