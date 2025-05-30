# scheduler.py

"""
scheduler.py: Implementation of a logit‑normal biased timestep scheduler for diffusion models.
References:
 - Sohl‑Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics”, 2015
 - Ho et al., “Denoising Diffusion Probabilistic Models”, 2020
"""

from typing import List
import numpy as np
from scipy.stats import norm

def get_biased_timesteps(num_steps: int, m: float = 0.0, s: float = 0.5) -> List[int]:
    """
    Generate a logit‑normal–biased sequence of timesteps.

    Args:
        num_steps: Total diffusion steps (T).
        m: Mean shift of the logit‑normal distribution.
        s: Scale of the logit‑normal distribution.

    Returns:
        A monotonic list of integer timesteps in [0, T‑1].
    """
    # 1. Uniform fractions
    u = np.linspace(0, 1, num_steps)
    # 2. Avoid infinite tails
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1 - eps)
    # 3. Apply inverse Gaussian CDF (logit‑normal mapping)
    v = m + s * norm.ppf(u)
    # 4. Rescale to [0, T‑1]
    t = (v - v.min()) / (v.max() - v.min()) * (num_steps - 1)
    # 5. Round and enforce bounds
    timesteps = np.round(t).astype(int)
    timesteps = np.clip(timesteps, 0, num_steps - 1)
    # 6. Ensure monotonicity
    for i in range(1, len(timesteps)):
        if timesteps[i] < timesteps[i-1]:
            timesteps[i] = timesteps[i-1]
    return timesteps.tolist()

