# seed_td_lsvi_tabular.py
# Numpy version of Seed TD and Seed LSVI on Bipolar Chain
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
            direction = 'L' if np.random.rand() < self.prior_p else 'R'
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

        self.theta_hat = np.random.normal(0, 1.0, N)
        self.q = np.zeros(N)
        self.noise_index = 0
        self.noise = np.random.normal(0, noise_std, size=10000)

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        for _ in range(10):  # 10 TD steps
            for (s, r, s_prime) in self.buffer:
                r_perturbed = r + self._next_noise()
                target = r_perturbed + self.gamma * self.q[s_prime]
                self.q[s] += self.alpha * ((target - self.q[s]) - (1 / self.lamb) * (self.q[s] - self.theta_hat[s]))

        pos = self.env.start
        total_reward = 0

        for t in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -np.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -np.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break

        return total_reward


class SeedAgentLSVI:
    def __init__(self, id, N, env, shared_buffer, gamma=0.8, lamb=0.00001, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.gamma = gamma
        self.lamb = lamb
        self.noise_std = noise_std

        self.theta_hat = np.random.normal(0, 1.0, N)
        self.noise_index = 0
        self.noise = np.random.normal(0, noise_std, size=10000)
        self.q = np.zeros(N)

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        A = np.zeros((self.N, self.N))
        b = np.zeros(self.N)

        for (s, r, s_prime) in self.buffer:
            r_perturbed = r + self._next_noise()
            target = r_perturbed + self.gamma * np.max(self.q[s_prime])
            A[s, s] += 1 + (1 / self.lamb)
            b[s] += target + (1 / self.lamb) * self.theta_hat[s]

        A += 1e-6 * np.eye(self.N)
        self.q = np.linalg.solve(A, b)

        pos = self.env.start
        total_reward = 0
        for t in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -np.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -np.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
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
    return np.mean(regrets)


def run_experiments():
    N = 50
    H = 100
    Ks = [1, 10, 20,30]#, 1000]
    results = {}

    for AgentClass, name in [(SeedAgentTD, 'Seed TD'), (SeedAgentLSVI, 'Seed LSVI'), (ThompsonAgent, 'Thompson')]:
        regrets = []
        for K in Ks:
            print(f"Running {name} with K={K}")
            regret = simulate(K, N, H, AgentClass)
            regrets.append(regret)
        results[name] = regrets

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
    plt.title("Seed TD and Seed LSVI on Bipolar Chain (Numpy)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bipolar_chain_seed_methods_numpy.png")


if __name__ == '__main__':
    run_experiments()
