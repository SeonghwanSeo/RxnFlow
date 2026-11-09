"""Chemical prior guidance."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from rxnflow.config import PriorGuidanceConfig
from rxnflow.reward import RewardFunction


class PriorRewardFunction(RewardFunction):
    """Frozen ChemPrior scorer using the shared reward cache and validation."""

    objectives = ("prior",)

    def __init__(
        self,
        config: PriorGuidanceConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.batch_size = config.batch_size
        from chemprior import ChemPrior

        self.teacher = ChemPrior.from_pretrained(
            config.model_name_or_path, device=device
        ).eval()
        self.teacher.requires_grad_(False)

    @torch.no_grad()
    def score(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Return normalized prior scores of shape [batch, 1]."""
        scores = self.teacher.screening(
            smiles_list,
            batch_size=self.batch_size,
            normalize=True,
        )

        return np.array(scores, dtype=np.float32)[:, None]
