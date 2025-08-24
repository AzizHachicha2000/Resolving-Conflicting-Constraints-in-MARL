# File: visualize.py
import sys, os
# ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from envs.multi_agent_env import MultiAgentEnv
from agents.marl_policy import MARLPolicy

def main():
    # Load config
    cfg = yaml.safe_load(open('configs/default.yaml'))

    # Initialize environment & policy
    env = MultiAgentEnv(
        cfg['env']['num_agents'],
        cfg['env']['world_size'],
        cfg['env']['r_safety'],
        cfg['env']['r_conflict'],
        cfg['env']['dt']
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = MARLPolicy(obs_dim, act_dim, cfg['training']['lr'])

    # Load the trained weights you saved during training
    policy.model.load_state_dict(torch.load('models/latest_policy.pt'))
    policy.model.eval()

    # Run one evaluation episode and record positions
    trajectories = {i: [] for i in range(env.num_agents)}
    obs = env.reset()
    for _ in range(cfg['evaluation']['steps']):
        for i in range(env.num_agents):
            trajectories[i].append(env.pos[i].copy())
        actions = [policy.act(o) for o in obs]
        obs, _, _, _ = env.step(np.array(actions))

    # Plot trajectories & goals
    plt.figure()
    for i, traj in trajectories.items():
        traj = np.array(traj)
        plt.plot(traj[:,0], traj[:,1], label=f'Agent {i}')
    plt.scatter(env.goals[:,0], env.goals[:,1], marker='x', c='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agent Trajectories')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
