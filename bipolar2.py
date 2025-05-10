import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BipolarChainEnv:
    def __init__(self, N, true_theta_L, true_theta_R):
        self.N = N  # Total vertices (odd, e.g., N=100)
        self.mid = N // 2  # Start at midpoint (vertex 50)
        self.true_theta = {'L': true_theta_L, 'R': true_theta_R}
        self.revealed = False  # True when any agent reaches L/R endpoint
        
    def traverse(self, agent_path):
        """Calculate reward for an agent's path. 
        Intermediate edges have reward -1; endpoints have theta_L/theta_R."""
        reward = 0
        endpoint_reached = False
        for step in range(len(agent_path) - 1):
            curr = agent_path[step]
            next_v = agent_path[step + 1]
            if next_v == 0:  # Left endpoint
                reward += self.true_theta['L']
                self.revealed = True
                endpoint_reached = True
            elif next_v == self.N - 1:  # Right endpoint
                reward += self.true_theta['R']
                self.revealed = True
                endpoint_reached = True
            else:
                reward -= 1  # Intermediate edge
            if endpoint_reached:
                break
        return reward, endpoint_reached

class Agent:
    def __init__(self, agent_id, algorithm, prior_p=0.5):
        self.algorithm = algorithm
        self.prior_p = prior_p
        self.seed = np.random.rand() if algorithm == "SeedSampling" else None
        self.direction = None  # Fixed direction (L/R) for SeedSampling
        
    def choose_direction(self):
        if self.algorithm == "SeedSampling":
            self.direction = 'L' if self.seed < 0.5 else 'R'
        elif self.algorithm == "Thompson":
            self.direction = 'L' if np.random.rand() < self.prior_p else 'R'
        elif self.algorithm == "UCRL":
            self.direction = 'R'  # Optimistically choose right
        return self.direction

def simulate_bipolar_chain(K, N, H, true_theta_L, true_theta_R, algorithm, n_sims=50):
    # Corrected optimal reward: max endpoint minus number of steps to reach
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
                if env.revealed:
                    # After revelation, switch to optimal direction
                    optimal_dir = 'L' if env.true_theta['L'] > env.true_theta['R'] else 'R'
                    direction = optimal_dir
                
                curr = path[-1]
                next_v = curr - 1 if direction == 'L' else curr + 1

                # Prevent walking off the chain
                if next_v < 0 or next_v >= N:
                    break

                path.append(next_v)

                # Stop early if endpoint reached
                if next_v == 0 or next_v == N - 1:
                    break
            
            # Calculate reward and regret
            agent_reward, _ = env.traverse(path)
            agent_regret = optimal_reward - agent_reward
            total_regret += agent_regret
        
        mean_regret = total_regret / K
        regrets.append(mean_regret)
    
    return np.mean(regrets)

# Parameters (aligned with Figure 3)
N = 100
H = 150
true_theta_L, true_theta_R = N, -N  # Left is optimal (θ_L = +100)
K_values = [1, 10, 100, 1000, 10000]

# Simulate
alg_regrets = {}
for alg in ["SeedSampling", "Thompson", "UCRL"]:
    print(f"Simulating {alg}...")
    regrets = []
    for K in K_values:
        avg_regret = simulate_bipolar_chain(K, N, H, true_theta_L, true_theta_R, alg, n_sims=50)
        regrets.append(avg_regret)
    alg_regrets[alg] = regrets

# Plot (matches paper's Figure 3)
plt.figure(figsize=(10, 6))
for alg, reg in alg_regrets.items():
    print(f"{alg}: {reg}")
    plt.plot(K_values, reg, marker='o', label=alg)

plt.xscale('log')
plt.xticks(K_values, labels=[f"$10^{i}$" for i in range(5)])
plt.xlabel("Number of Concurrent Agents (K)")
plt.ylabel("Mean Regret per Agent")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.title("Bipolar Chain: Regret vs. Number of Agents (Corrected)")
plt.savefig("bipolar2.png")
