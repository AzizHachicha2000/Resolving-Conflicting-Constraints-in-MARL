# File: agents/marl_policy.py
import torch
import torch.nn as nn
from models.gnn_actor_critic import GNNActorCritic

class MARLPolicy:
    def __init__(self, obs_dim, act_dim, lr):
        self.model = GNNActorCritic(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = self.model.actor(obs_t)
        return action.numpy()

    def update(self, obs, target):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        target_t = torch.tensor(target, dtype=torch.float32)
        pred = self.model.critic(obs_t)
        loss = self.mse(pred.squeeze(), target_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()