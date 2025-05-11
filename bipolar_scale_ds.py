# seed_td_consistent.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class BipolarChainEnv:
    def __init__(self, N):
        self.N = N
        self.start = N // 2
        self.pos = self.start  # Initialize position
        self.theta_L = N if np.random.rand() < 0.5 else -N
        self.theta_R = -self.theta_L
        self.revealed = False
        self.optimal_reward = N//2 if self.theta_L > 0 else N//2 + 1

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        """0=left, 1=right"""
        self.pos = max(0, self.pos-1) if action == 0 else min(self.N-1, self.pos+1)
        reward = -0.1
        done = False
        
        if self.pos == 0:
            reward = self.theta_L
            done = True
        elif self.pos == self.N-1:
            reward = self.theta_R
            done = True
            
        if done: 
            self.revealed = True
        return self.pos, reward, done

class SeedTD:
    def __init__(self, agent_id, state_dim, action_dim=2,
                 alpha=0.1, gamma=0.99, lamb=1.0,
                 prior_std=1.0, noise_std=0.1):
        np.random.seed(agent_id)
        self.theta = np.random.normal(0, prior_std, (state_dim, action_dim))
        self.theta_prior = self.theta.copy()
        self.noise = np.random.normal(0, noise_std, 10000)
        self.noise_ptr = 0
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb

    def act(self, state, epsilon=0.0):
        """Epsilon-greedy action selection"""
        if np.random.rand() < epsilon:
            return np.random.randint(2)
        return np.argmax(self.theta[state])

    def update(self, buffer):
        for s, a, r, s_prime in buffer:
            perturbed_r = r + self.noise[self.noise_ptr]
            self.noise_ptr = (self.noise_ptr + 1) % len(self.noise)
            
            q_next = np.max(self.theta[s_prime]) if s_prime not in [0, self.theta.shape[0]-1] else 0
            target = perturbed_r + self.gamma * q_next
            reg_term = (self.theta[s,a] - self.theta_prior[s,a]) / self.lamb
            self.theta[s,a] += self.alpha * (target - self.theta[s,a] - reg_term)

def simulate(K, N, H, num_episodes=30):
    regrets = []
    for _ in tqdm(range(num_episodes)):
        env = BipolarChainEnv(N)
        buffer = []
        agents = [SeedTD(k, N) for k in range(K)]
        
        episode_regret = 0
        revealed = False
        
        for _ in range(H):
            # Parallel agent execution
            for agent in agents:
                if env.revealed:  # Skip if solution found
                    continue
                    
                s = env.pos
                a = agent.act(s)
                s_prime, r, done = env.step(a)
                buffer.append((s, a, r, s_prime))
                
                if done and not revealed:
                    episode_regret = env.optimal_reward - (r * H)
                    revealed = True
            
            if revealed:
                break
                
            # Update all agents
            for agent in agents:
                agent.update(buffer)
        
        regrets.append(episode_regret/K if revealed else env.optimal_reward)
    return np.mean(regrets)

def run_experiment():
    N = 50
    K_values = [1, 10, 20]#, 30]
    results = {'Seed TD': []}
    
    for K in K_values:
        avg_regret = simulate(K, N, H=100)
        results['Seed TD'].append(avg_regret)
    
    plt.figure(figsize=(10,6))
    plt.plot(K_values, results['Seed TD'], 'o-')
    plt.xscale('log')
    plt.xticks(K_values, [f'10^{int(np.log10(k))}' for k in K_values])
    plt.xlabel("Number of Agents (K)")
    plt.ylabel("Average Regret per Agent")
    plt.title("Bipolar Chain: Seed TD Performance")
    plt.grid(True)
    plt.savefig('seed_td_performance.png')

if __name__ == '__main__':
    run_experiment()