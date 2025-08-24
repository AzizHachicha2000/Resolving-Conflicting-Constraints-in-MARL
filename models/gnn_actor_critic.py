# File: models/gnn_actor_critic.py
import torch
import torch.nn as nn

class GNNActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden=128
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)

# File: safety/reachability_utils.py
import numpy as np

def compute_cbvf_value(pos_i, pos_j, r_safety):
    # Control Barrier-Value Function approx = distance - r_safety
    dist = np.linalg.norm(pos_i - pos_j)
    return dist - r_safety