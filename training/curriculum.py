#curricuulum.py
def get_safety_params(step, total_steps, max_r_safety, max_r_conflict):
    """
    Quadratic schedule: ramps up faster early in training.
    """
    # normalize to [0,1]
    frac = min(1.0, step / (total_steps / 2))
    # quadratic ramp
    schedule = frac ** 2
    return schedule * max_r_safety, schedule * max_r_conflict
