# seed_td_lsvi_bipolar.py with PyTorch for scalability
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt


class BipolarChainEnv:
    def __init__(self, N, theta):
        self.N = N
        self.start = N // 2
        self.theta = theta
        self.revealed = False
        self.revealed_optimal_direction = None
        print(f"Initialized environment with N={N}, theta={theta}")

    def get_reward(self, position):
        if position == 0:
            self.revealed = True
            self.revealed_optimal_direction = 'L' if self.theta['L'] > self.theta['R'] else 'R'
            print(f"Reached left end, revealed optimal direction: {self.revealed_optimal_direction}")
            return self.theta['L'], True
        elif position == self.N - 1:
            self.revealed = True
            self.revealed_optimal_direction = 'L' if self.theta['L'] > self.theta['R'] else 'R'
            print(f"Reached right end, revealed optimal direction: {self.revealed_optimal_direction}")
            return self.theta['R'], True
        else:
            return -0.1, False


class SeedTDNetwork(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.q_vals = nn.Parameter(torch.zeros(N))

    def forward(self, s):
        return self.q_vals[s]


class SeedAgentTD:
    def __init__(self, id, N, env, shared_buffer, alpha=0.1, gamma=1.0, lamb=1.0, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.gamma = gamma
        self.lamb = lamb
        print(f"Initialized TD Agent {id} with alpha={alpha}, gamma={gamma}, lambda={lamb}")

        self.noise_std = noise_std
        self.noise = torch.randn(10000) * noise_std
        self.noise_index = 0

        self.model = SeedTDNetwork(N)
        self.theta_hat = torch.randn(N)
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)

    def _next_noise(self):
        z = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return z

    def act(self, H):
        print(f"\nAgent {self.id} starting episode with horizon H={H}")
        # TD learning from shared buffer with noise and prior
        for i in range(10):
            total_loss = 0
            for (s, r, s_prime) in self.buffer:
                r_perturbed = r + self._next_noise().item()
                td_target = r_perturbed + self.gamma * self.model(torch.tensor(s_prime, dtype=torch.long))
                q_val = self.model(torch.tensor(s, dtype=torch.long))
                prior = self.theta_hat[s]
                loss = (td_target - q_val) ** 2 + (1 / self.lamb) * (q_val - prior) ** 2
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % 2 == 0 and i != 0:  # Print every other iteration to avoid too much output
                try:
                    print(f"  TD update iteration {i}, average loss: {total_loss/len(self.buffer):.4f}")
                except:
                    print(f"[0 buff]  TD update iteration {i}, average loss: {total_loss:.4f}")

        pos = self.env.start
        total_reward = 0
        for t in range(H):
            left_q = self.model(torch.tensor(pos - 1, dtype=torch.long)) if pos > 0 else torch.tensor(-float('inf'))
            right_q = self.model(torch.tensor(pos + 1, dtype=torch.long)) if pos < self.N - 1 else torch.tensor(-float('inf'))
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            print(f"  Step {t}: pos={pos}, next_pos={next_pos}, left_q={left_q:.2f}, right_q={right_q:.2f}")
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                print(f"  Episode finished at step {t} with total reward {total_reward}")
                break

        return total_reward


class SeedAgentLSVI:
    def __init__(self, id, N, env, shared_buffer, gamma=1.0, lamb=1.0, noise_std=0.1):
        self.id = id
        self.N = N
        self.env = env
        self.buffer = shared_buffer
        self.gamma = gamma
        self.lamb = lamb
        print(f"Initialized LSVI Agent {id} with gamma={gamma}, lambda={lamb}")

        self.noise = torch.randn(10000) * noise_std
        self.noise_index = 0
        self.theta_hat = torch.randn(N)
        self.q = torch.zeros(N)

    def _next_noise(self):
        z = self.noise[self.noise_index % len(self.noise)]
        self.noise_index += 1
        return z

    def act(self, H):
        print(f"\nAgent {self.id} starting episode with horizon H={H}")
        A = torch.zeros((self.N, self.N))
        b = torch.zeros(self.N)

        for (s, r, s_prime) in self.buffer:
            r_perturbed = r + self._next_noise().item()
            target = r_perturbed + self.gamma * torch.max(self.q[s_prime])
            A[s, s] += 1 + (1 / self.lamb)
            b[s] += target + (1 / self.lamb) * self.theta_hat[s]

        A += 1e-6 * torch.eye(self.N)  # numerical stability
        self.q = torch.linalg.solve(A, b)
        print(f"  LSVI update complete, buffer size: {len(self.buffer)}")

        pos = self.env.start
        total_reward = 0
        for t in range(H):
            left_q = self.q[pos - 1] if pos > 0 else -float('inf')
            right_q = self.q[pos + 1] if pos < self.N - 1 else -float('inf')
            next_pos = pos - 1 if left_q > right_q else pos + 1
            next_pos = max(0, min(self.N - 1, next_pos))
            print(f"  Step {t}: pos={pos}, next_pos={next_pos}, left_q={left_q:.2f}, right_q={right_q:.2f}")
            reward, done = self.env.get_reward(next_pos)
            self.buffer.append((pos, reward, next_pos))
            total_reward += reward
            pos = next_pos
            if done:
                print(f"  Episode finished at step {t} with total reward {total_reward}")
                break

        return total_reward


def simulate(K, N, H, AgentClass, episodes=50):
    regrets = []
    for ep in range(episodes):
        print(f"\nStarting episode {ep+1}/{episodes}")
        theta_L = N if random.random() < 0.5 else -N
        theta_R = -theta_L
        r_star = N // 2 if theta_L > 0 else N // 2 + 1
        theta = {'L': theta_L, 'R': theta_R}
        env = BipolarChainEnv(N, theta)

        buffer = []
        agents = [AgentClass(k, N, env, buffer) for k in range(K)]
        rewards = [agent.act(H) for agent in agents]
        mean_regret = r_star - (sum(rewards) / K)
        regrets.append(mean_regret)
        print(f"Episode {ep+1} complete - Mean regret: {mean_regret:.2f}")
    return np.mean(regrets)


def run_experiments():
    N = 50
    H = 75
    Ks = [1, 10]#, 100]#, 1000]
    results = {}

    for AgentClass, name in [(SeedAgentTD, 'Seed TD'), (SeedAgentLSVI, 'Seed LSVI')]:
        regrets = []
        for K in Ks:
            print(f"Running {name} with K={K}")
            regret = simulate(K, N, H, AgentClass)
            regrets.append(regret)
        results[name] = regrets

    plt.figure(figsize=(8, 5))
    for name, regrets in results.items():
        plt.plot(Ks, regrets, label=name, marker='o')
    plt.xscale('log')
    plt.xticks(Ks, labels=[f"$10^{{{i}}}$" for i in range(len(Ks))])
    plt.xlabel("Number of Concurrent Agents (K)")
    plt.ylabel("Mean Regret per Agent")
    plt.title("Seed TD and Seed LSVI on Bipolar Chain (PyTorch)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bipolar_chain_seed_methods_torch.png")


if __name__ == '__main__':
    run_experiments()
