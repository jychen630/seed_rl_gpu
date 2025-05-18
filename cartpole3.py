import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
from dm_control import suite
# Constants from the paper
TIMESTEP = 3000  # 30 seconds at 0.01 timestep
ACTION_MAP = [-10, 0, 10]  # Force values from the paper

def featurize(state):
    """Convert state to features as described in the paper"""
    x, x_dot, theta, theta_dot = state
    return np.array([
        math.cos(theta),
        math.sin(theta),
        theta_dot / 10.0,
        x / 10.0,
        x_dot / 10.0,
        1.0 if abs(x) < 0.1 else 0.0
    ], dtype=np.float32)

class QNetwork(nn.Module):
    """Neural network with skip connection as described in the paper"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, 3)  # 3 actions
        
    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        return self.head(x2 + x1)  # Skip connection

class SeedEnsemble:
    """Implements the seed ensemble approach from section 3.2.3"""
    def __init__(self, ensemble_size=30, device="cpu"):
        self.device = device
        self.ensemble_size = ensemble_size
        self.models = []
        self.priors = []
        self.noise_seeds = []  # Fixed noise seeds for each model
        
        # Initialize ensemble with different seeds
        for _ in range(ensemble_size):
            model = QNetwork().to(device)
            prior = QNetwork().to(device)
            
            # Initialize with different random seeds
            for net in [model, prior]:
                for layer in net.children():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            
            # Store fixed noise seeds for this model
            noise_seed = np.random.normal(0, 0.01, size=100000)  # From paper
            self.models.append(model)
            self.priors.append(prior)
            self.noise_seeds.append(noise_seed)
        
        self.optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in self.models]
    
    def get_model(self, agent_idx):
        """Assign a model to an agent (section 3.2.3)"""
        return self.models[agent_idx % self.ensemble_size]
    
    def get_prior(self, agent_idx):
        """Get the corresponding prior network"""
        return self.priors[agent_idx % self.ensemble_size]
    
    def get_noise(self, model_idx, exp_idx):
        """Get fixed noise for this model and experience"""
        noise_arr = self.noise_seeds[model_idx % self.ensemble_size]
        return noise_arr[exp_idx % len(noise_arr)]

def create_cartpole_env():
    """Create dm_control cartpole swingup environment"""
    env = suite.load('cartpole', 'swingup')
    physics = env.physics
    return env, physics

def is_success(state):
    """Check if pole is balanced as per paper's criteria"""
    x, x_dot, theta, theta_dot = state
    return (math.cos(theta) > 0.95 and abs(theta_dot) < 1.0)
    # return (math.cos(theta) > 0.95 and 
    #         abs(x) < 0.1 and 
    #         abs(x_dot) < 1.0 and 
    #         abs(theta_dot) < 1.0)

def run_agent(agent_idx, ensemble, replay_buffer, results):
    """Run one agent according to the seed TD algorithm"""
    env, physics = create_cartpole_env()
    time_step = env.reset()
    # Set initial state to (x=0, x_dot=0, theta=π, theta_dot=0)
    physics.data.qpos[:] = [np.random.uniform(-0.1, 0.1), np.random.normal(np.pi, 0.1)]       # x, theta
    physics.data.qvel[:] = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]         # x_dot, theta_dot
    physics.after_reset()                    # Propagate changes through physics


    model = ensemble.get_model(agent_idx)
    prior = ensemble.get_prior(agent_idx)
    optimizer = ensemble.optimizers[agent_idx % ensemble.ensemble_size]
    
    total_reward = 0
    state = physics.state()  # [x, x_dot, theta, theta_dot]
    print("state", state)
    for step in range(TIMESTEP):
        # Select action using current model
        print("step", step)
        obs_tensor = torch.tensor(featurize(state)).float().unsqueeze(0).to(ensemble.device)
        with torch.no_grad():
            q_vals = model(obs_tensor) + 3.0 * prior(obs_tensor)  # From paper
            action_idx = torch.argmax(q_vals).item()
        
        # Apply action and get next state
        action = ACTION_MAP[action_idx]
        time_step = env.step([action])
        new_state = physics.state()
        print("new_state", new_state)
        done = is_success(new_state)
        if done:
            reward = 1.0
        else:
            reward = 0.0
            reward -=  abs(action) / 1000.0  # Action cost from paper
        print("reward", reward)
        # Store experience with model-specific noise
        model_idx = agent_idx % ensemble.ensemble_size
        noise = ensemble.get_noise(model_idx, len(replay_buffer))
        replay_buffer.append((state, action_idx, reward + noise, new_state))
        
        # Update results
        results[step] += reward
        total_reward += reward
        state = new_state
        
        # Train on minibatch
        if len(replay_buffer) >= 96:
            batch = random.sample(replay_buffer, 16)
            obs_b, act_b, rew_b, next_obs_b = zip(*batch)
            
            # Convert to tensors
            obs_b = torch.tensor(np.array([featurize(o) for o in obs_b])).float().to(ensemble.device)
            next_obs_b = torch.tensor(np.array([featurize(o) for o in next_obs_b])).float().to(ensemble.device)
            act_b = torch.tensor(act_b).long().to(ensemble.device)
            rew_b = torch.tensor(rew_b).float().to(ensemble.device)
            
            # Compute target with model-specific noise
            with torch.no_grad():
                q_next = model(next_obs_b) + 3.0 * prior(next_obs_b)
                q_next = q_next.max(1)[0]
            
            target = rew_b + 0.99 * q_next  # gamma=0.99
            
            # Compute loss with prior regularization
            q_pred = model(obs_b)
            q_val = q_pred.gather(1, act_b.unsqueeze(1)).squeeze()
            prior_val = prior(obs_b).gather(1, act_b.unsqueeze(1)).squeeze()
            
            loss = ((q_val - target)**2).mean() + (1/0.01) * ((q_val - prior_val)**2).mean()
            loss /= 0.01
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if done:
            # increment the rest of the timesteps a reward of 1
            for i in range(step+1, TIMESTEP):
                results[i] += 1
            break
    return total_reward

def run_experiment(K=100, ensemble_size=30, num_seeds=5):
    """Run the full experiment with multiple random seeds"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []
        
    ensemble = SeedEnsemble(ensemble_size=ensemble_size, device=device)
    replay_buffer = []
    results = [0 for _ in range(TIMESTEP)]
    
    for agent_idx in tqdm(range(K)):
        run_agent(agent_idx, ensemble, replay_buffer, results)
    
    # Average across seeds
    print(results[:5], results[-5:], len(results))
    return np.array(results) / K

def plot_results(results):
    """Plot learning curves as in Figure 3 of the paper"""
    timestep = np.arange(TIMESTEP) * 0.01  # Convert to seconds
    
    # Add smoothing using moving average
    window_size = 50  # Adjust this value to control smoothing amount
    smoothed_results = np.convolve(results, np.ones(window_size)/window_size, mode='valid')
    # Adjust timestep to match smoothed results length
    smoothed_timestep = timestep[window_size-1:]
    
    plt.figure(figsize=(10, 6))
    # Plot both raw and smoothed data
    plt.plot(timestep, results, alpha=0.3, label='Raw Reward')  # Raw data with transparency
    plt.plot(smoothed_timestep, smoothed_results, label='Smoothed Reward')  # Smoothed data
    plt.plot(timestep, np.zeros_like(timestep), 'k--', label='DQN ε-greedy')
    
    plt.xlabel('Time elapsed (seconds)')
    plt.ylabel('Average instantaneous reward')
    plt.title('Cartpole Swing-up: Seed TD Learning Curve')
    plt.legend()
    plt.grid()
    plt.savefig('cartpole3.png')


if __name__ == '__main__':
    # Run with parameters from the paper
    K = 10
    num_seeds = 1
    ensemble_size = min(30, K)
    results = run_experiment(K=K, ensemble_size=ensemble_size, num_seeds=num_seeds)
    plot_results(results)