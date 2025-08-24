# File: safety/reachability_utils.py
import numpy as np

def compute_cbvf_value(pos_i, pos_j, r_safety):
    # Control Barrier-Value Function approx = distance - r_safety
    dist = np.linalg.norm(pos_i - pos_j)
    return dist - r_safety