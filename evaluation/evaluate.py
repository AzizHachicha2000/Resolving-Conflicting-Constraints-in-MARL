#evaluate.py
import sys
import os
# Add the project root (one level up) to Python import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from envs.multi_agent_env import MultiAgentEnv
from agents.marl_policy import MARLPolicy
from utils.logger import log

def evaluate_agent(config_path):
    cfg = yaml.safe_load(open(config_path))
    env = MultiAgentEnv(
        cfg['env']['num_agents'], cfg['env']['world_size'],
        cfg['env']['r_safety'], cfg['env']['r_conflict'], cfg['env']['dt']
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = MARLPolicy(obs_dim, act_dim, cfg['training']['lr'])

    obs = env.reset()
    for t in range(cfg['evaluation']['steps']):
        actions = [policy.act(o) for o in obs]
        obs, reward, done, _ = env.step(np.array(actions))

    log(f"Evaluation reward: {np.mean(reward)}")

if __name__ == '__main__':
    # Allow running this file directly
    evaluate_agent('configs/default.yaml')
