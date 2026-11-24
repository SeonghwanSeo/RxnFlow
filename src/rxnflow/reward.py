"""Reward function definition"""

from __future__ import annotations

from functools import cached_property
from math import isfinite
from typing import Any, ClassVar

INVALID_REWARD = -1000.0


class RewardFunction:
    r"""Base class for reward functions.
    Set ``name`` to the objective key. Implement ``score`` for one molecule
    or ``score_list`` for a batch.

    For a single objective with unit prior and property factors,
    p(x) ∝ exp(beta * R(x)).
    """

    name: ClassVar[str] = ""

    def score(
        self,
        smiles: str,
        cond: dict[str, Any] | None = None,
    ) -> float:
        """Return a score for a single molecule."""
        raise NotImplementedError

    def score_list(
        self,
        smiles_list: list[str],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """Return a list of scores for a list of molecules.

        The default implementation applies score to each molecule and its
        optional condition in input order.
        """
        if cond_list is None:
            return [self.score(smi) for smi in smiles_list]
        else:
            return [
                self.score(smi, cond)
                for smi, cond in zip(smiles_list, cond_list, strict=True)
            ]

    @cached_property
    def _cache(self) -> dict[str, float]:
        """Cache scores by SMILES for calls without conditions."""
        return {}

    def __call__(
        self,
        smiles_list: list[str | None],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """Evaluate molecules through ``run``."""
        return self.run(smiles_list, cond_list)

    def run(
        self,
        smiles_list: list[str | None],
        cond_list: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """Score a batch of molecules."""
        if cond_list is not None and len(cond_list) != len(smiles_list):
            raise ValueError("supply one condition per molecule")

        result = [INVALID_REWARD] * len(smiles_list)
        valid = [(i, smi) for i, smi in enumerate(smiles_list) if smi is not None]
        if not valid:
            return result

        if cond_list is not None:
            pending = [smi for _, smi in valid]
            scores = self.score_list(pending, [cond_list[i] for i, _ in valid])
        else:
            pending = list(
                dict.fromkeys(smi for _, smi in valid if smi not in self._cache)
            )
            scores = self.score_list(pending) if pending else []

        if len(scores) != len(pending):
            raise ValueError(
                f"Score must return one float per molecule, "
                f"got {len(scores)} scores for {len(pending)} molecules"
            )
        if not all(isfinite(score) for score in scores):
            raise ValueError(f"Score must return finite floats, got {scores}")

        if cond_list is not None:
            for (i, _), score in zip(valid, scores, strict=True):
                result[i] = float(score)
        else:
            for smi, score in zip(pending, scores, strict=True):
                self._cache[smi] = float(score)
            for i, smi in valid:
                result[i] = self._cache[smi]
        return result
