"""Sampling masks and terminal rewards for size and property limits."""

import math

import numpy as np
import torch
from numpy.typing import NDArray
from rdkit import Chem

from rxnflow.envs.features import PROPERTY_NAMES, molecular_properties


def get_size_mask(
    state_size: torch.Tensor,
    synthon_size: torch.Tensor,
    max_atoms: int,
) -> torch.Tensor:
    return state_size + synthon_size <= max_atoms


def get_property_penalty(
    state_prop: torch.Tensor,
    synthon_prop: torch.Tensor,
    property_limits: dict[int, float],
) -> torch.Tensor:
    """Apply additive property limits in raw units during sampling."""
    property_penalty = torch.ones(
        (*state_prop.shape[:-1], len(synthon_prop)),
        dtype=torch.bool,
        device=synthon_prop.device,
    )
    for index, limit in property_limits.items():
        estimate = state_prop[..., index, None] + synthon_prop[:, index]
        # Nonzero bounds allow 1% tolerance; zero bounds remain exact.
        if limit == 0:
            property_penalty &= estimate <= 0
        else:
            property_penalty &= estimate < limit + abs(limit) * 0.01
    return property_penalty


def compute_property_rewards(
    mols: list[Chem.Mol | None],
    property_limits: dict[int, float],
    max_atoms: int,
    sigma_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return terminal penalty multipliers and per-molecule violation flags.

    Each upper bound L has sigma = sigma_ratio * abs(L). A zero sigma makes
    that bound strict. Violations do not change the chemical validity of a route.
    """
    limits = {**property_limits, PROPERTY_NAMES.index("heavy_atoms"): max_atoms}
    rewards = np.zeros(len(mols), dtype=np.float64)
    violations = np.zeros(len(mols), dtype=np.bool_)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        properties = molecular_properties(mol)
        log_reward = 0.0
        for index, limit in limits.items():
            excess = float(properties[index]) - limit
            if excess <= 0:
                continue
            violations[i] = True
            sigma = sigma_ratio * abs(limit)
            if sigma == 0:
                log_reward = -math.inf
                break
            log_reward -= excess / (2 * sigma)
        rewards[i] = math.exp(log_reward)
    return rewards, violations
