# File: training/train.py
import yaml
import numpy as np
from envs.multi_agent_env import MultiAgentEnv
from agents.marl_policy import MARLPolicy
from safety.cbvf_filter import apply_cbvf_filter
from safety.reachability_utils import compute_cbvf_value
from training.curriculum import get_safety_params
from utils.logger import log


def train_agent(config_path):
    cfg = yaml.safe_load(open(config_path))
    env = MultiAgentEnv(
        cfg['env']['num_agents'], cfg['env']['world_size'],
        cfg['env']['r_safety'], cfg['env']['r_conflict'], cfg['env']['dt']
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = MARLPolicy(obs_dim, act_dim, cfg['training']['lr'])
    total = cfg['training']['steps']
    for step in range(total):
        obs = env.reset()
        for t in range(cfg['training']['episode_length']):
            actions = []
            # update safety params
            r_safety, r_conflict = get_safety_params(step, total, cfg['env']['r_safety'], cfg['env']['r_conflict'])
            for i in range(env.num_agents):
                a = policy.act(obs[i])
                # find closest neighbor
                dists = [np.linalg.norm(env.pos[j]-env.pos[i]) for j in range(env.num_agents) if j!=i]
                jn = np.argmin(dists)
                # apply CBVF filter
                a = apply_cbvf_filter(a, env.pos[i], env.vel[i], env.pos[jn], env.vel[jn], r_safety, cfg['env']['gamma'], env.dt)
                actions.append(a)
            actions = np.array(actions)
            next_obs, reward, done, _ = env.step(actions)
            # dummy critic target: negative sum reward
            target = -reward
        # loop over agents:
            for i in range(env.num_agents):
              policy.update(obs[i], target[i])
            obs = next_obs
        if step%100==0:
            log(f"Step {step}/{total}")
    log("Training complete")
    import torch
    torch.save(policy.model.state_dict(), 'models/latest_policy.pt')
    log("Saved policy weights to models/latest_policy.pt")

