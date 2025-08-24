# File: pygame_simulation.py

import pygame
import numpy as np
import torch
import os
import sys
import time

# Import your model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.marl_policy import MARLPolicy

# Config
NUM_AGENTS = 4
WORLD_SIZE = 600  # in pixels
SCALE = WORLD_SIZE / 5.0  # 5.0 world units → 600 pixels
RADIUS = 5
FPS = 60
GOAL_THRESHOLD = 0.2
MAX_STEPS = 1000

obs_dim = 4 + 4 * (NUM_AGENTS - 1)
act_dim = 2

# Load policy
policy = MARLPolicy(obs_dim, act_dim, lr=0.0005)
policy.model.load_state_dict(torch.load('models/latest_policy.pt'))
policy.model.eval()

# Init pygame
pygame.init()
screen = pygame.display.set_mode((WORLD_SIZE, WORLD_SIZE))
pygame.display.set_caption("2D Multi-Agent Drone Simulation")
clock = pygame.time.Clock()

# Colors
colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0)]

# Initialize agents
positions = np.random.rand(NUM_AGENTS, 2) * 5.0
velocities = np.zeros((NUM_AGENTS, 2))
goals = np.random.rand(NUM_AGENTS, 2) * 5.0
reached = [False] * NUM_AGENTS

step = 0
running = True

while running and step < MAX_STEPS:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get observations
    obs_all = []
    for i in range(NUM_AGENTS):
        own = np.concatenate([positions[i], velocities[i]])
        neigh = []
        for j in range(NUM_AGENTS):
            if i == j:
                continue
            neigh.extend(positions[j] - positions[i])
            neigh.extend(velocities[j])
        obs = np.concatenate([own, neigh])
        obs_all.append(obs)

    # Get actions using the policy
    with torch.no_grad():
        actions = []
        for obs in obs_all:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # [1, obs_dim]
            action = policy.act(obs_tensor)
            action = np.array(action).flatten()  # ensure it's a flat NumPy array
            actions.append(action)

    # Update positions and velocities
    for i in range(NUM_AGENTS):
        if reached[i]:
            continue  # skip update if goal already reached

        accel = np.clip(actions[i], -1, 1) * 0.1
        velocities[i] += accel
        velocities[i] *= 0.95  # damping
        positions[i] += velocities[i] * (1.0 / FPS)
        positions[i] = np.clip(positions[i], 0, 5.0)  # keep in bounds

        # Check if goal is reached
        if np.linalg.norm(positions[i] - goals[i]) < GOAL_THRESHOLD:
            velocities[i] = np.zeros(2)
            reached[i] = True

    # Check if all agents reached their goals
    if all(reached):
        print(f"All agents reached their goals in {step} steps.")
        time.sleep(2)  # Pause so user can see final state
        running = False

    # Draw everything
    screen.fill((30, 30, 30))
    for i in range(NUM_AGENTS):
        pos_px = (positions[i] * SCALE).astype(int)
        goal_px = (goals[i] * SCALE).astype(int)
        pygame.draw.circle(screen, colors[i], pos_px, RADIUS)
        pygame.draw.circle(screen, (255, 255, 255), goal_px, 4, 1)

    pygame.display.flip()
    clock.tick(FPS)
    step += 1

pygame.quit()
