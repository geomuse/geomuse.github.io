import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = zip(*batch)
        return s, a, r, s_, d

    def __len__(self):
        return len(self.buffer)

import torch.optim as optim
import numpy as np
import cv2

class DQNAgent:
    def __init__(self, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN(action_dim).to(self.device)
        self.target_net = DQN(action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer()

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def preprocess(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84))
        obs = obs / 255.0
        return torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def select_action(self, obs):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)

        obs = self.preprocess(obs).to(self.device)
        with torch.no_grad():
            return self.q_net(obs).argmax().item()

    def update(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return

        s, a, r, s_, d = self.buffer.sample(batch_size)

        s = torch.cat([self.preprocess(x) for x in s]).to(self.device)
        s_ = torch.cat([self.preprocess(x) for x in s_]).to(self.device)

        a = torch.tensor(a).unsqueeze(1).to(self.device)
        r = torch.tensor(r).float().to(self.device)
        d = torch.tensor(d).float().to(self.device)

        q = self.q_net(s).gather(1, a).squeeze()
        q_next = self.target_net(s_).max(1)[0]
        target = r + self.gamma * q_next * (1 - d)

        loss = F.mse_loss(q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

from gym_snake import SnakeEnv
env = SnakeEnv()
agent = DQNAgent(action_dim=4)

EPISODES = 500

for ep in range(EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)

        agent.buffer.push(obs, action, reward, next_obs, done)
        agent.update()

        obs = next_obs
        total_reward += reward

    agent.target_net.load_state_dict(agent.q_net.state_dict())
    print(f"Episode {ep}, reward = {total_reward}, epsilon = {agent.epsilon:.3f}")
