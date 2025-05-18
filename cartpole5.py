import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
import collections
import math
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def glorot_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# Configuration
NUM_AGENTS = 5
ENSEMBLE_SIZE = min(30, NUM_AGENTS)
TIMESTEPS = 3000
BUFFER_SIZE = 100000
BATCH_SIZE = 32
ACTION_MAP = [-10, 0, 10]

def featurize(state):
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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, 3)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        return self.head(x2 + x1)


class SeedEnsemble:
    def __init__(self):
        self.models = []
        self.priors = []
        self.noise_seeds = []
        
        # Initialize ensemble with different seeds
        for i in range(ENSEMBLE_SIZE):
            # Create model/prior pair
            model = QNetwork()
            prior = QNetwork()
            
            # Initialize with fixed seeds
            torch.manual_seed(i)
            model.apply(glorot_init)
            torch.manual_seed(i)
            prior.apply(glorot_init)
            
            # Freeze prior weights
            for param in prior.parameters():
                param.requires_grad = False
                
            self.models.append(model)
            self.priors.append(prior)
            self.noise_seeds.append(np.random.RandomState(i).normal(0, 0.01, 100000))

class CoordinatedLearner:
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)
        self.ensemble = SeedEnsemble()
        self.optimizers = [optim.Adam(m.parameters(), lr=1e-3) for m in self.ensemble.models]
        
        # Agent tracking
        self.agent_states = [self._init_agent(i) for i in range(NUM_AGENTS)]
        self.total_experiences = 0
        
    def _init_agent(self, agent_id):
        """Initialize agent with persistent seed and environment"""
        env = suite.load('cartpole', 'swingup')
        env.physics.data.qpos[1] = np.pi  # Start with pole down
        return {
            'env': env,
            'physics': env.physics,
            'model_idx': agent_id % ENSEMBLE_SIZE,
            'noise_idx': 0,
            'timestep': 0
        }
        
    def _get_perturbed_reward(self, agent_state, reward):
        """Apply fixed noise perturbation from agent's seed"""
        noise = self.ensemble.noise_seeds[agent_state['model_idx']][
            agent_state['noise_idx'] % len(self.ensemble.noise_seeds[agent_state['model_idx']])
        ]
        agent_state['noise_idx'] += 1
        return reward + noise
        
    def run_epoch(self):
        """Run one timestep for all agents"""
        # Phase 1: Collect experiences from all agents
        new_experiences = []
        
        for agent in self.agent_states:
            if agent['timestep'] >= TIMESTEPS:
                continue
                
            # Get current state
            physics = agent['physics']
            obs = featurize(physics.state())
            
            # Select action using ensemble model
            model = self.ensemble.models[agent['model_idx']]
            prior = self.ensemble.priors[agent['model_idx']]
            with torch.no_grad():
                q_vals = model(torch.tensor(obs)) + 3 * prior(torch.tensor(obs))
                action_idx = torch.argmax(q_vals).item()
                
            # Take action
            action = ACTION_MAP[action_idx]
            physics.set_control([action])
            physics.step()
            new_obs = physics.state()
            
            # Calculate reward
            x, x_dot, theta, theta_dot = new_obs
            success = (math.cos(theta) > 0.75 and 
                      #abs(x) < 0.1 and 
                      #abs(x_dot) < 1.0 and 
                      abs(theta_dot) < 1.0)
            reward = 1.0 if success else 0.0
            reward -= 0.1 * abs(action) / 10.0  # Action cost
            
            # Store experience with perturbation
            perturbed_reward = self._get_perturbed_reward(agent, reward)
            new_experiences.append((obs, action_idx, perturbed_reward, 
                                  featurize(new_obs)))
            
            agent['timestep'] += 1
            
        # Add experiences to shared buffer
        self.buffer.extend(new_experiences)
        self.total_experiences += len(new_experiences)
        
        # Phase 2: Train all ensemble models on shared buffer
        if len(self.buffer) > BATCH_SIZE:
            for model_idx in range(ENSEMBLE_SIZE):
                # Sample batch with model-specific noise
                batch = random.sample(self.buffer, BATCH_SIZE)
                obs_b, act_b, rew_b, next_obs_b = zip(*batch)
                
                # Convert to tensors
                obs_t = torch.tensor(np.array(obs_b)).float()
                next_obs_t = torch.tensor(np.array(next_obs_b)).float()
                act_t = torch.tensor(act_b).long()
                rew_t = torch.tensor(rew_b).float()
                
                # Get Q-values
                model = self.ensemble.models[model_idx]
                prior = self.ensemble.priors[model_idx]
                
                # Compute target with model-specific prior
                with torch.no_grad():
                    q_next = model(next_obs_t) + 3 * prior(next_obs_t)
                    target = rew_t + 0.99 * q_next.max(1)[0]
                
                # Compute loss
                q_pred = model(obs_t).gather(1, act_t.unsqueeze(1)).squeeze()
                prior_val = prior(obs_t).gather(1, act_t.unsqueeze(1)).squeeze()
                loss = ((q_pred - target)**2).mean() + (1/0.01) * ((q_pred - prior_val)**2).mean()
                
                # Update model
                self.optimizers[model_idx].zero_grad()
                loss.backward()
                self.optimizers[model_idx].step()

    def run_training(self):
        rewards = []
        for _ in tqdm(range(TIMESTEPS)):
            self.run_epoch()
            # Calculate average reward for this timestep
            if len(self.buffer) > 0:
                recent_rewards = [exp[2] for exp in 
                                list(self.buffer)[-NUM_AGENTS:]]
                rewards.append(np.mean(recent_rewards))
        return rewards

# Usage
learner = CoordinatedLearner()
rewards = learner.run_training()

def plot_results(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(rewards)) * 0.01, rewards)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Instantaneous Reward')
    plt.title('Coordinated Exploration Learning Curve')
    plt.grid(True)
    plt.savefig('cartpole5.png')
    plt.show()

plot_results(rewards)