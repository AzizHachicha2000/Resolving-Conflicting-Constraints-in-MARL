#cbvf_filter.py
import numpy as np

def apply_cbvf_filter(action, pos_i, vel_i, pos_j, vel_j, r_safety, gamma=None, dt=None):
    """
    Simple heuristic safety filter:
    If agent i is within r_safety of agent j, override the MARL action
    by pushing acceleration directly away from agent j at the same magnitude.
    Otherwise, keep the original action.

    Args:
      action   (np.ndarray): original 2-D acceleration [ax, ay]
      pos_i    (np.ndarray): agent i position [x, y]
      vel_i    (np.ndarray): agent i velocity [vx, vy] (unused here)
      pos_j    (np.ndarray): neighbor j position [x, y]
      vel_j    (np.ndarray): neighbor j velocity [vx, vy] (unused here)
      r_safety (float): safety radius threshold
      gamma    (float, optional): unused placeholder
      dt       (float, optional): unused placeholder
    Returns:
      np.ndarray: filtered 2-D acceleration
    """
    delta = pos_j - pos_i
    dist = np.linalg.norm(delta)

    # If inside unsafe zone, steer directly away
    if dist < r_safety and dist > 1e-6:
        # unit vector from j to i
        away_dir = (pos_i - pos_j) / dist
        # preserve original action magnitude
        mag = np.linalg.norm(action)
        return away_dir * mag

    # otherwise, keep the MARL action
    return action
