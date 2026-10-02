"""Action logit matrices, as in RxnFlow/CGFlow, with HSX budget masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from rxnflow.gflownet.types import Action, ActionType


@dataclass
class ActionSubspace:
    """One reaction's sampled block columns and their masked policy logits.

    The library/index mapping defines the subspace; only a selected column is
    converted to an Action. Unary actions have a single column and no library.
    """

    name: str
    action_type: ActionType
    libraries: list[str]
    block_indices: list[Tensor]
    logits: Tensor  # [state, sampled block], or [state, 1] for UniReaction
    log_importance: Tensor  # [sampled block]

    def action_at(self, column: int) -> Action:
        block_type = None
        block_index = None
        for name, indices in zip(self.libraries, self.block_indices, strict=True):
            if column < len(indices):
                block_type, block_index = name, int(indices[column])
                break
            column -= len(indices)
        return Action(
            self.action_type,
            reaction=None if self.action_type == ActionType.FIRST_BLOCK else self.name,
            block_type=block_type,
            block_index=block_index,
        )


@dataclass
class ActionCategorical:
    action_subspaces: list[ActionSubspace]
    graph_emb: Tensor
    logit_scale: Tensor

    def log_partition(self) -> Tensor:
        # RxnFlow estimates the denominator from an independent subsample.
        # Unlike a fixed reference group mask, a budgeted subsample can be
        # entirely masked. A finite floor keeps observed-edge scoring defined;
        # the caller applies the reference nonpositive log P clamp.
        if not self.action_subspaces:
            return self.graph_emb.new_full((len(self.graph_emb),), math.log(1e-38))
        weighted = [p.logits + p.log_importance for p in self.action_subspaces]
        maxima = torch.stack([x.max(1).values for x in weighted]).max(0).values
        maxima = torch.where(torch.isfinite(maxima), maxima, 0).detach()
        totals = sum((x - maxima[:, None]).exp().sum(1) for x in weighted)
        return maxima + totals.clamp_min(1e-38).log()

    def sample(
        self, sampling_temperature: float, random_action_prob: float, importance: float
    ) -> list[Action | None]:
        # Draw on the model device. Only chosen group/column indices cross
        # to Python, not millions of candidate logits.
        if not self.action_subspaces:
            return [None] * len(self.graph_emb)
        random_rows = (
            torch.rand(len(self.graph_emb), device=self.graph_emb.device)
            < random_action_prob
        )
        best_values, best_columns = [], []
        for group in self.action_subspaces:
            values = group.logits + importance * group.log_importance
            if random_action_prob > 0:
                # This is the random policy, not a learned action logit. Balance
                # libraries by their sampled sizes before applying state masks.
                if group.block_indices:
                    random_logits = torch.cat(
                        [
                            values.new_full(
                                (len(indices),),
                                -math.log(len(group.libraries) * len(indices)),
                            )
                            for indices in group.block_indices
                        ]
                    )
                else:
                    random_logits = values.new_zeros(1)
                values = torch.where(random_rows[:, None], random_logits, values)
            values = values.masked_fill(~torch.isfinite(group.logits), -torch.inf)
            # Gumbel-max on retained matrices (RxnFlow/CGFlow categorical).
            noise = torch.rand_like(values).clamp_min(torch.finfo(values.dtype).tiny)
            values = values / sampling_temperature - (-noise.log()).log()
            best, columns = values.max(1)
            best_values.append(best)
            best_columns.append(columns)
        best, group_ids = torch.stack(best_values, 1).max(1)
        columns = torch.stack(best_columns, 1).gather(1, group_ids[:, None]).squeeze(1)
        selected = (
            torch.stack([group_ids, columns, torch.isfinite(best).long()], 1)
            .cpu()
            .tolist()
        )
        return [
            self.action_subspaces[p].action_at(c) if valid else None
            for p, c, valid in selected
        ]
