# Combined script for Seed TD, Seed LSVI, Thompson Resampling, Seed Sampling, UCRL
# on Bipolar Chain from Dimakopoulou & Van Roy (2018) using JAX for vectorization
# unsuccessful, cant run due to version too old
import jax
import jax.numpy as jnp
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
            return -0.1, False


class ThompsonAgent:
    def __init__(self, id, env):
        self.env = env
        self.prior_p = 0.5

    def act(self, H):
        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            direction = 'L' if np.random.rand() < self.prior_p else 'R'
            if self.env.revealed:
                direction = self.env.revealed_optimal_direction
            pos += -1 if direction == 'L' else 1
            pos = max(0, min(self.env.N - 1, pos))
            r, done = self.env.get_reward(pos)
            total_reward += r
            if done:
                break
        return total_reward


class SeedSamplingAgent:
    def __init__(self, id, env, N):
        self.env = env
        z_k = np.random.rand()
        self.direction = 'L' if z_k < 0.5 else 'R'

    def act(self, H):
        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            direction = self.direction
            if self.env.revealed:
                direction = self.env.revealed_optimal_direction
            pos += -1 if direction == 'L' else 1
            pos = max(0, min(self.env.N - 1, pos))
            r, done = self.env.get_reward(pos)
            total_reward += r
            if done:
                break
        return total_reward


class UCRLAgent:
    def __init__(self, id, env):
        self.env = env

    def act(self, H):
        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            direction = 'R'
            if self.env.revealed:
                direction = self.env.revealed_optimal_direction
            pos += -1 if direction == 'L' else 1
            pos = max(0, min(self.env.N - 1, pos))
            r, done = self.env.get_reward(pos)
            total_reward += r
            if done:
                break
        return total_reward


class SeedAgentTD:
    def __init__(self, id, N, env, shared_buffer, alpha=0.1, gamma=1.0, lamb=1.0, noise_std=0.1):
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.theta_hat = jax.random.normal(jax.random.PRNGKey(id), (N,))
        self.q = jnp.zeros(N)
        self.noise = jax.random.normal(jax.random.PRNGKey(id + 1000), (10000,)) * noise_std
        self.noise_index = 0

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        for _ in range(10):
            for (s, r, s_prime) in self.buffer:
                r_perturbed = r + self._next_noise()
                target = r_perturbed + self.gamma * self.q[s_prime]
                self.q = jax.ops.index_update(
                    self.q, s,
                    self.q[s] + self.alpha * ((target - self.q[s]) - (1 / self.lamb) * (self.q[s] - self.theta_hat[s]))
                )

        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -jnp.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -jnp.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward


class SeedAgentLSVI:
    def __init__(self, id, N, env, shared_buffer, gamma=1.0, lamb=1.0, noise_std=0.1):
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.gamma = gamma
        self.lamb = lamb
        self.theta_hat = jax.random.normal(jax.random.PRNGKey(id), (N,))
        self.q = jnp.zeros(N)
        self.noise = jax.random.normal(jax.random.PRNGKey(id + 2000), (10000,)) * noise_std
        self.noise_index = 0

    def _next_noise(self):
        val = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return val

    def act(self, H):
        A = jnp.zeros((self.N, self.N))
        b = jnp.zeros(self.N)
        for (s, r, s_prime) in self.buffer:
            r_perturbed = r + self._next_noise()
            target = r_perturbed + self.gamma * jnp.max(self.q[s_prime])
            A = jax.ops.index_add(A, (s, s), 1 + (1 / self.lamb))
            b = jax.ops.index_add(b, s, target + (1 / self.lamb) * self.theta_hat[s])

        A += 1e-6 * jnp.eye(self.N)
        self.q = jnp.linalg.solve(A, b)

        pos = self.env.start
        total_reward = 0
        for _ in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -jnp.inf
            right_q = self.q[pos + 1] if pos < self.N - 1 else -jnp.inf
            next_pos = pos - 1 if left_q > right_q else pos + 1
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                break
        return total_reward


def simulate(K, N, H, AgentClass, shared_buffer=False):
    regrets = []
    for _ in tqdm(range(50)):
        theta_L = N if random.random() < 0.5 else -N
        theta_R = -theta_L
        r_star = N // 2 if theta_L > 0 else N // 2 + 1
        env = BipolarChainEnv(N, {'L': theta_L, 'R': theta_R})
        buffer = [] if shared_buffer else None
        agents = [AgentClass(k, N, env, buffer) if shared_buffer else AgentClass(k, env) for k in range(K)]
        rewards = [agent.act(H) for agent in agents]
        regrets.append(r_star - (sum(rewards) / K))
    return np.mean(regrets)


def run_all():
    N = 50
    H = 3 * N // 2
    Ks = [1, 10, 100, 1000]
    results = {}

    agent_configs = [
        (lambda k, N, env, buf: SeedAgentTD(k, N, env, buf), 'Seed TD', True),
        (lambda k, N, env, buf: SeedAgentLSVI(k, N, env, buf), 'Seed LSVI', True),
        (lambda k, env: ThompsonAgent(k, env), 'Thompson', False),
        (lambda k, env: SeedSamplingAgent(k, env, N), 'SeedSampling', False),
        (lambda k, env: UCRLAgent(k, env), 'UCRL', False),
    ]

    for agent_fn, name, shared_buf in agent_configs:
        regrets = []
        for K in Ks:
            regrets.append(simulate(K, N, H, agent_fn, shared_buf))
        results[name] = regrets

    plt.figure(figsize=(10, 6))
    for name, regrets in results.items():
        plt.plot(Ks, regrets, marker='o', label=name)
    plt.xscale('log')
    plt.xticks(Ks, labels=[f"$10^{{{i}}}$" for i in range(len(Ks))])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Mean Regret per Agent")
    plt.title("Bipolar Chain: All Algorithms (JAX)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bipolar_chain_all_algorithms_jax.png")


if __name__ == '__main__':
    run_all()
