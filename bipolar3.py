import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BipolarChainEnv:
    def __init__(self, N, true_theta_L, true_theta_R):
        self.N = N
        self.mid = N // 2
        self.true_theta = {'L': true_theta_L, 'R': true_theta_R}
        self.revealed = False

    def traverse(self, agent_path):
        reward = 0
        endpoint_reached = False
        for step in range(len(agent_path) - 1):
            curr = agent_path[step]
            next_v = agent_path[step + 1]
            if next_v == 0:
                reward += self.true_theta['L']
                self.revealed = True
                endpoint_reached = True
            elif next_v == self.N - 1:
                reward += self.true_theta['R']
                self.revealed = True
                endpoint_reached = True
            else:
                reward -= 1
            if endpoint_reached:
                break
        return reward, endpoint_reached

class Agent:
    def __init__(self, agent_id, algorithm, prior_p=0.5):
        self.algorithm = algorithm
        self.prior_p = prior_p
        self.seed = np.random.rand() if algorithm == "SeedSampling" else None
        self.direction = None

    def choose_direction(self):
        if self.algorithm == "SeedSampling":
            self.direction = 'L' if self.seed < 0.5 else 'R'
        elif self.algorithm == "Thompson":
            self.direction = 'L' if np.random.rand() < self.prior_p else 'R'
        elif self.algorithm == "UCRL":
            self.direction = 'R'
        return self.direction

def simulate_bipolar_chain(K, N, H, true_theta_L, true_theta_R, algorithm, n_sims=50):
    optimal_reward = - (N // 2) + max(true_theta_L, true_theta_R)
    regrets = []

    for _ in range(n_sims):
        env = BipolarChainEnv(N, true_theta_L, true_theta_R)
        agents = [Agent(k, algorithm) for k in range(K)]
        total_regret = 0

        for agent in agents:
            path = [env.mid]
            direction = agent.choose_direction()
            endpoint_reached = False

            for step in range(H):
                if algorithm in ["SeedSampling", "UCRL"] and env.revealed:
                    direction = 'L' if env.true_theta['L'] > env.true_theta['R'] else 'R'

                curr = path[-1]
                next_v = curr - 1 if direction == 'L' else curr + 1
                if next_v < 0 or next_v >= N:
                    break
                path.append(next_v)

                if next_v == 0 or next_v == N - 1:
                    break

            agent_reward, _ = env.traverse(path)
            agent_regret = optimal_reward - agent_reward
            total_regret += agent_regret

        mean_regret = total_regret / K
        regrets.append(mean_regret)

    return np.mean(regrets)

# Parameters (aligned with Figure 3)
N = 100
H = 150
true_theta_L, true_theta_R = N, -N
K_values = [1, 10, 100, 1000, 10000]

alg_regrets = {}
for alg in ["SeedSampling", "Thompson", "UCRL"]:
    print(f"Simulating {alg}...")
    regrets = []
    for K in K_values:
        avg_regret = simulate_bipolar_chain(K, N, H, true_theta_L, true_theta_R, alg, n_sims=50)
        regrets.append(avg_regret)
    alg_regrets[alg] = regrets

# Plot
plt.figure(figsize=(10, 6))
for alg, reg in alg_regrets.items():
    print(f"{alg}: {reg}")
    plt.plot(K_values, reg, marker='o', label=alg)

plt.xscale('log')
plt.xticks(K_values, labels=[f"$10^{{{i}}}$" for i in range(5)])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.title("Bipolar Chain: Regret vs. Number of Agents (Corrected)")
plt.savefig("bipolar3.png")
