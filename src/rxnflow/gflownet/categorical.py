"""Protocol logit matrices, as in RxnFlow/CGFlow, with HSX budget masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from rxnflow.gflownet.types import ActionKind, RxnAction


@dataclass
class ProtocolLogits:
    name: str
    kind: ActionKind
    # Columns retain the sampled library order even when their logits are -inf.
    libraries: list[str]
    block_indices: list[Tensor]
    logits: Tensor  # [state, sampled block], or [state, 1] for UniReaction
    log_importance: Tensor  # [sampled block]
    exploration_logits: Tensor  # CGFlow's -log(n_libraries * n_sampled_blocks)

    def action_at(self, column: int) -> RxnAction:
        block_type = None
        block_index = None
        for name, indices in zip(self.libraries, self.block_indices, strict=True):
            if column < len(indices):
                block_type, block_index = name, int(indices[column])
                break
            column -= len(indices)
        return RxnAction(
            self.kind,
            "",
            reaction=None if self.kind == ActionKind.FIRST_BLOCK else self.name,
            block_type=block_type,
            block_index=block_index,
        )


@dataclass
class ActionCategorical:
    protocols: list[ProtocolLogits]
    embeddings: Tensor

    def log_partition(self) -> Tensor:
        # RxnFlow estimates the denominator from an independent subsample.
        # Unlike a fixed reference protocol mask, a budgeted subsample can be
        # entirely masked. A finite floor keeps observed-edge scoring defined;
        # the caller applies the reference nonpositive log P clamp.
        if not self.protocols:
            return self.embeddings.new_full((len(self.embeddings),), math.log(1e-38))
        weighted = [p.logits + p.log_importance for p in self.protocols]
        maxima = torch.stack([x.max(1).values for x in weighted]).max(0).values
        maxima = torch.where(torch.isfinite(maxima), maxima, 0).detach()
        totals = sum((x - maxima[:, None]).exp().sum(1) for x in weighted)
        return maxima + totals.clamp_min(1e-38).log()

    def sample(
        self, temperature: float, random_action_prob: float, importance: float
    ) -> list[RxnAction | None]:
        # Draw on the model device. Only chosen protocol/column indices cross
        # to Python, not millions of candidate logits.
        if not self.protocols:
            return [None] * len(self.embeddings)
        random_rows = (
            torch.rand(len(self.embeddings), device=self.embeddings.device)
            < random_action_prob
        )
        best_values, best_columns = [], []
        for protocol in self.protocols:
            values = protocol.logits + importance * protocol.log_importance
            values = torch.where(
                random_rows[:, None], protocol.exploration_logits, values
            )
            values = values.masked_fill(~torch.isfinite(protocol.logits), -torch.inf)
            # Gumbel-max on retained matrices (RxnFlow/CGFlow categorical).
            noise = torch.rand_like(values).clamp_min(torch.finfo(values.dtype).tiny)
            values = values / temperature - (-noise.log()).log()
            best, columns = values.max(1)
            best_values.append(best)
            best_columns.append(columns)
        best, protocol_ids = torch.stack(best_values, 1).max(1)
        columns = torch.stack(best_columns, 1).gather(1, protocol_ids[:, None]).squeeze(1)
        selected = (
            torch.stack([protocol_ids, columns, torch.isfinite(best).long()], 1)
            .cpu()
            .tolist()
        )
        return [
            self.protocols[p].action_at(c) if valid else None for p, c, valid in selected
        ]
