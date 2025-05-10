# Reimplementation of the Bipolar Chain experiment from Section 4.1 of Dimakopoulou & Van Roy (2018)
# Implements Thompson Resampling, Seed Sampling, and Concurrent UCRL as described in the paper
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.logger import setup_logging
import logging
# setup_logging()

class BipolarChainEnv:
    def __init__(self, N, theta):
        self.N = N
        self.start = N // 2 # v_s, vertex of start; Each one  of the K agents starts from vertext v_s = N/2
        self.theta = theta  # dict with keys 'L' and 'R'
        self.revealed = False
        self.revealed_optimal_direction = None
        logging.info(f"env {self.N}, {self.theta}, {self.start}, {self.revealed}, {self.revealed_optimal_direction}")

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
        logging.info(f"agent {self.id}, {self.algorithm}, prior_p={self.prior_p}, sampled_theta={self.sampled_theta}, fixed_direction={self.fixed_direction}")
    
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

        # Initialize agents, states, and logs
        agents = [Agent(k, algorithm, env) for k in range(K)]
        positions = [env.start] * K
        directions = [agent.choose_direction() for agent in agents]
        rewards = [0] * K
        terminated = [False] * K

        # for t in range(H):
        #     for k in range(K):
        #         if terminated[k]:
        #             continue

        #         # If the environment has been revealed, switch direction
        #         if env.revealed:
        #             directions[k] = env.revealed_optimal_direction

        #         next_pos = positions[k] - 1 if directions[k] == 'L' else positions[k] + 1
        #         next_pos_adj = max(0, min(N - 1, next_pos))
        #         if next_pos_adj != next_pos:
        #             raise ValueError(f"algorithm={algorithm}, t={t}, k={k}, next_pos={next_pos}, next_pos_adj={next_pos_adj}")
        #             #next_pos = next_pos_adj

        #         reward, terminal = env.get_reward(next_pos, k, t)
        #         rewards[k] += reward
        #         positions[k] = next_pos
        #         terminated[k] = terminal
        agents = [Agent(k, algorithm, env) for k in range(K)]
        rewards = [agent.act(H) for agent in agents]
        total_rewards_all_agents = sum(rewards)
        mean_reward = total_rewards_all_agents / K
        mean_regret = r_star - mean_reward
        regrets_all_episodes.append(mean_regret)

        logging.info(f"mean_reward={mean_reward:.2f}, total_rewards={total_rewards_all_agents:.2f}, mean_regret={mean_regret:.2f}")

    bayes_regret = np.mean(regrets_all_episodes)
    logging.info(f"bayes_regret = {bayes_regret:.2f}, r_star={r_star:.2f}, N={N}, H={H}, K={K}, theta_L={theta_L}, theta_R={theta_R}")
    return bayes_regret

# Parameters from the paper (Figure 3)
N = 100
H = 150
# Wrong (fixed): common prior that should be with probability 0.5, theta_L = N and theta_R = -theta_L
# TODO: if this probability sampled every agent or sampled once for all agent? i think it's the later
# i.e, the agent share the same setting
# When any of the K agents traverses e_L, or e_R for the first time, all K agents learn the true values of theta_L, theta_r


K_values = [1, 10, 100, 1000, 10000] #list(range(1,101, 10))#
algorithms = [ 'Thompson', 'SeedSampling', "UCRL"] #['SeedSampling'] #

# Run simulation
results = {}
for algo in algorithms:
    logging.info(f"Simulating {algo}...")
    regrets = []
    for K in K_values:
        regret = simulate_bipolar_chain_for_K_agents(K, N, H, algo, n_episodes=50)
        regrets.append(regret)
    results[algo] = regrets

# Plot
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(K_values, results[algo], marker='o', label=algo)
logging.info(results)
plt.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Reference (y=25)')
plt.axhline(y=125, color='r', linestyle='--', alpha=0.5, label='Reference (y=125)')
plt.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Reference (y=200)')

plt.xscale('log')
plt.xticks(K_values, labels=[f"$10^{{{i}}}$" for i in range(len(K_values))])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.title("Bipolar Chain: Regret vs. Number of Agents")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()
plt.savefig("bipolar_comment.png")
