"""Chemical prior guidance."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from rxnflow.config import PriorGuidanceConfig


class PriorGuidance:
    """Frozen teacher log probabilities, independent of task rewards."""

    def __init__(
        self,
        config: PriorGuidanceConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        from chemprior import ChemPrior

        self.model = ChemPrior.from_pretrained(config.model_name_or_path, device)
        self.batch_size = config.batch_size

    @torch.no_grad()
    def estimate_log_p_x(self, smiles_list: list[str]) -> NDArray[np.float32]:
        """Return raw sequence log probabilities of shape [batch, 1]."""
        logp = self.model.screening(
            smiles_list, batch_size=self.batch_size, normalize=False
        )
        logp = np.asarray(logp, dtype=np.float32)
        if logp.shape != (len(smiles_list),):
            raise ValueError("Expected one log probability per molecule")
        # Failed scoring receives the same finite penalty as an invalid trajectory.
        logp = np.nan_to_num(logp, nan=-1000.0, neginf=-1000.0, posinf=-1000.0)
        return np.minimum(logp, 0.0)[:, None]

    def __call__(self, smiles_list: list[str | None]) -> NDArray[np.float32]:
        return self.run(smiles_list)

    def run(self, smiles_list: list[str | None]) -> NDArray[np.float32]:
        pending = [i for i, s in enumerate(smiles_list) if s is not None]
        inputs = [smiles_list[i] for i in pending]
        log_p = np.full((len(smiles_list), 1), -1000.0, dtype=np.float32)
        if inputs:
            log_p[pending] = self.estimate_log_p_x(inputs)
        return log_p
