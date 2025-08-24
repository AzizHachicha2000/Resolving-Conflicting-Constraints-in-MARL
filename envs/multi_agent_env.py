# File: envs/multi_agent_env.py
import numpy as np
import gym
from gym.spaces import Box

class MultiAgentEnv(gym.Env):
    def __init__(self, num_agents, world_size, r_safety, r_conflict, dt=0.1):
        super().__init__()
        self.num_agents = num_agents
        self.dt = dt
        self.world_size = world_size
        self.r_safety = r_safety
        self.r_conflict = r_conflict
        obs_dim = 4 + 4*(num_agents-1)  # [pos,vel] + neighbors
        act_dim = 2  # acceleration
        self.observation_space = Box(-np.inf, np.inf, (obs_dim,))
        self.action_space = Box(-1, 1, (act_dim,))
        self.reset()

    def reset(self):
        # positions and velocities
        self.pos = np.random.rand(self.num_agents, 2)*self.world_size
        self.vel = np.zeros((self.num_agents,2))
        # set random goals
        self.goals = np.random.rand(self.num_agents,2)*self.world_size
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            own = np.concatenate([self.pos[i], self.vel[i]])
            neigh = []
            for j in range(self.num_agents):
                if j==i: continue
                neigh.extend(self.pos[j]-self.pos[i])
                neigh.extend(self.vel[j])
            obs.append(np.concatenate([own, neigh]))
        return np.array(obs)

    def step(self, actions):
        # apply actions as acceleration
        self.vel += np.clip(actions, -1,1)*self.dt
        self.pos += self.vel*self.dt
        # reward: negative distance to goal + velocity-alignment bonus
        dist_goal = np.linalg.norm(self.pos - self.goals, axis=1)
        # compute unit vector from pos to goal
        to_goal = self.goals - self.pos
        dir_to_goal = to_goal / (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-6)
        vel_alignment = np.sum(self.vel * dir_to_goal, axis=1)
        reward = -dist_goal + 0.1 * vel_alignment   # ↑ added small bonus for moving toward goal
        # conflict penalty
        for i in range(self.num_agents):
            dists = np.linalg.norm(self.pos - self.pos[i], axis=1)
            cnt = np.sum((dists<self.r_conflict)) - 1
            if cnt>1:
                reward[i] -= np.sum(self.r_conflict - dists[dists<self.r_conflict]) * 0.5
        done = False
        return self._get_obs(), reward, done, {}