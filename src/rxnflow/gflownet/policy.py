"""Masked action distributions, shared library subsampling, and trajectory generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from rxnflow.config import Config
from rxnflow.core.errors import InvalidTransition, NoValidActions
from rxnflow.core.types import (
    Action,
    ActionSubspace,
    ActionType,
    BackwardTrajectory,
    InvalidReason,
    State,
    Trajectory,
    Transition,
)
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import molecular_properties
from rxnflow.envs.graph import GraphBatch, GraphData, molecule_to_graph_data
from rxnflow.models import RxnFlowModel


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@dataclass
class ActionLogits:
    subspace: ActionSubspace
    logits: torch.Tensor  # [state, sampled block], or [state, 1] for UniReaction
    log_importance: torch.Tensor  # [sampled block]


class SubsamplingPolicy:
    """Uniform subsampling of a fixed action range, including singleton actions."""

    def __init__(
        self,
        num_actions: int,
        sampling_ratio: float,
        min_sampling: int,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.num_actions = num_actions
        self.num_sampling = min(
            num_actions, max(min_sampling, int(num_actions * sampling_ratio))
        )
        self.log_importance = math.log(num_actions / self.num_sampling)
        self.full_indices = (
            np.arange(num_actions, dtype=np.int64)
            if self.num_sampling == num_actions
            else None
        )

    def sample(self) -> tuple[np.ndarray, float]:
        if self.full_indices is not None:
            # Full ranges, including unary [0], consume no RNG or new allocation.
            return self.full_indices, self.log_importance
        indices = self.rng.choice(self.num_actions, self.num_sampling, replace=False)
        # Keep catalog row order without changing which actions were sampled.
        indices.sort()
        return indices, self.log_importance


@dataclass
class ActionCategorical:
    action_logits: list[ActionLogits]
    graph_emb: torch.Tensor
    logit_scale: torch.Tensor

    def log_partition(self) -> torch.Tensor:
        """Estimate log sum(exp(logit)) with library inclusion weights."""
        # A sampled space may be entirely masked. Floor its total mass so an
        # observed action can still be scored; log_prob caps the result at zero.
        if not self.action_logits:
            return self.graph_emb.new_full((len(self.graph_emb),), math.log(1e-38))
        # Subtract one maximum per state across all subspaces for stability.
        weighted = [p.logits + p.log_importance for p in self.action_logits]
        maxima = torch.stack([x.max(1).values for x in weighted]).max(0).values
        maxima = torch.where(torch.isfinite(maxima), maxima, 0).detach()
        totals = sum((x - maxima[:, None]).exp().sum(1) for x in weighted)
        return maxima + totals.clamp_min(1e-38).log()

    def sample(
        self, sampling_temperature: float, random_action_prob: float, importance: float
    ) -> list[Action | None]:
        """Draw one action per state; return None where every action is masked."""
        # 1. Choose which state rows use the random exploration distribution.
        if not self.action_logits:
            return [None] * len(self.graph_emb)
        random_rows = (
            torch.rand(len(self.graph_emb), device=self.graph_emb.device)
            < random_action_prob
        )
        # 2. Apply exploration/importance weights and sample within each subspace.
        best_values, best_columns = [], []
        for group in self.action_logits:
            values = group.logits + importance * group.log_importance
            if random_action_prob > 0:
                # Give each (reaction, library) subspace unit mass before
                # masking at softmax temperature 1. Unary subspaces have N=1.
                # Surviving mass is reduced by the fraction of masked columns.
                random_logits = values.new_full(
                    (values.shape[1],), -math.log(values.shape[1])
                )
                values = torch.where(random_rows[:, None], random_logits, values)
            values = values.masked_fill(~torch.isfinite(group.logits), -torch.inf)
            # Gumbel-max samples from softmax without materializing probabilities.
            noise = torch.rand_like(values).clamp_min(torch.finfo(values.dtype).tiny)
            values = values / sampling_temperature - (-noise.log()).log()
            best, columns = values.max(1)
            best_values.append(best)
            best_columns.append(columns)
        # 3. Compare subspace winners; transfer only selected indices to Python.
        best, group_ids = torch.stack(best_values, 1).max(1)
        columns = torch.stack(best_columns, 1).gather(1, group_ids[:, None]).squeeze(1)
        selected = (
            torch.stack([group_ids, columns, torch.isfinite(best).long()], 1)
            .cpu()
            .tolist()
        )
        return [
            self.action_logits[p].subspace.action_at(c) if valid else None
            for p, c, valid in selected
        ]


class RxnFlowPolicy:
    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
        rng: np.random.Generator,
    ):
        self.env, self.model, self.config = env, model, config
        self.device, self.rng = device, rng
        # Samplers are shared by library across reaction subspaces. All unary
        # subspaces use the deterministic singleton range under None.
        num_actions = {
            None: 1,
            **{name: len(library) for name, library in env.blocks.items()},
        }
        self.subsampling = {
            name: SubsamplingPolicy(
                count,
                config.subsampling.sampling_ratio,
                config.subsampling.min_sampling,
                rng,
            )
            for name, count in num_actions.items()
        }

    def forward(
        self, states: list[State], beta: torch.Tensor, preferences: torch.Tensor
    ) -> ActionCategorical:
        """Retain sampled columns; mask logits instead of packing valid actions.

        Each library draw is shared across states and reactions. Each reaction
        shares one query/matmul across libraries, then exposes one logit matrix
        per (reaction, library). Observed actions do not affect these draws or
        their importance weights.
        """
        assert states
        # 1. Build state features and collect compatible reaction/library rows.
        graphs: dict[tuple[str, int], GraphData] = {}
        properties: dict[tuple[str, int], NDArray[np.float32]] = {}
        keys = [(state.smiles, state.reaction_count) for state in states]
        # Metadata contains only state rows and library names, never candidate
        # objects or per-state arrays of valid block indices.
        action_rows: dict[str, set[int]] = {}  # reaction -> batch rows
        # reaction -> library -> batch rows
        action_libraries: dict[str, dict[str, list[int]]] = {}
        action_types: dict[str, ActionType] = {}
        library_rows: dict[str | None, set[int]] = {}  # None is the unary range
        for row, (state, key) in enumerate(zip(states, keys, strict=True)):
            if key not in graphs:
                mol_properties = molecular_properties(state.mol)
                properties[key] = mol_properties
                graphs[key] = molecule_to_graph_data(
                    state.mol, self.env.max_atoms, state.reaction_count, mol_properties
                )
            for subspace in self.env.get_action_space(state):
                name, block_type = subspace.name
                action_types[name] = subspace.action_type
                action_rows.setdefault(name, set()).add(row)
                if block_type is not None:
                    action_libraries.setdefault(name, {}).setdefault(
                        block_type, []
                    ).append(row)
                library_rows.setdefault(block_type, set()).add(row)
        # 2. Draw each library once and apply additive budgets to sampled rows.
        descriptors = np.stack([properties[key] for key in keys])
        samples: dict[str | None, NDArray[np.int64]] = {}
        log_importance: dict[str | None, float] = {}
        masks: dict[str, NDArray[np.bool_]] = {}  # library -> [batch, sampled actions]
        # Each entry contains fingerprints, properties and block type indices.
        features: list[
            tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.int64]]
        ] = []
        sizes: list[int] = []
        for library_name, rows in library_rows.items():
            samples[library_name], log_importance[library_name] = self.subsampling[
                library_name
            ].sample()
            if library_name is None:
                continue  # Unary actions share subsampling, but have no block features.
            indices = samples[library_name]
            row_indices = sorted(rows)
            mask = np.zeros((len(states), len(indices)), dtype=np.bool_)
            mask[row_indices] = self.env.get_block_mask(
                descriptors[row_indices], library_name, indices
            )
            masks[library_name] = mask
            library_data = self.env.blocks[library_name]
            features.append(
                (
                    library_data.fingerprints[indices],
                    library_data.properties[indices],
                    np.full(
                        (len(indices),),
                        self.env.block_type_to_index[library_name],
                        dtype=np.int64,
                    ),
                )
            )
            sizes.append(len(indices))

        # 3. Encode conditions, state graphs, and all sampled blocks in batches.
        # Graph/property preprocessing is shared, but identical molecules may
        # have different beta/preferences and need separate neural encodings.
        cond_info = self.model.encode_cond(beta, preferences)
        graph_emb = self.model.graph_embedding(
            GraphBatch.from_graphs([graphs[key] for key in keys]).to(self.device),
            cond_info,
        )
        logit_scale = self.model.logit_scale(cond_info)
        block_embs: dict[str, torch.Tensor] = {}
        if features:
            fps, props, types = zip(*features, strict=True)
            encoded = F.normalize(
                self.model.block_embedding(
                    torch.from_numpy(np.concatenate(fps)).to(
                        self.device, dtype=torch.float32
                    ),
                    torch.from_numpy(np.concatenate(props)).to(self.device),
                    torch.from_numpy(np.concatenate(types)).to(self.device),
                ),
                dim=-1,
            )
            block_embs = dict(zip(masks, encoded.split(sizes), strict=True))

        # 4. Score by reaction, apply masks, and split columns into subspaces.
        action_logits: list[ActionLogits] = []
        for name, rows in action_rows.items():
            row_indices = torch.tensor(sorted(rows), device=self.device)
            state_emb = self.model.forward_mdp(
                graph_emb[row_indices], name, logit_scale[row_indices]
            )
            libraries = list(action_libraries.get(name, {}))
            if libraries:
                # Compute the reaction query and matrix product once, then split
                # columns into library subspaces without recomputing embeddings.
                blocks = torch.cat([block_embs[n] for n in libraries])
                scores = state_emb @ blocks.T
                logits = graph_emb.new_full((len(states), len(blocks)), -torch.inf)
                logits = logits.index_copy(0, row_indices, scores)
                allowed: list[NDArray[np.bool_]] = []
                weight_parts: list[torch.Tensor] = []
                for n in libraries:
                    eligible = np.zeros(len(states), dtype=np.bool_)
                    eligible[action_libraries[name][n]] = True
                    allowed.append(masks[n] & eligible[:, None])
                    count = len(samples[n])
                    weight_parts.append(torch.full((count,), log_importance[n]))
                logits = logits.masked_fill(
                    ~torch.from_numpy(np.concatenate(allowed, axis=1)).to(self.device),
                    -torch.inf,
                )
                weights = torch.cat(weight_parts).to(self.device)
            else:
                logits = graph_emb.new_full((len(states), 1), -torch.inf).index_copy(
                    0, row_indices, state_emb
                )
                weights = graph_emb.new_full((len(samples[None]),), log_importance[None])
            if libraries:
                counts = [len(samples[n]) for n in libraries]
                for library, library_logits, library_weights in zip(
                    libraries,
                    logits.split(counts, dim=1),
                    weights.split(counts),
                    strict=True,
                ):
                    action_logits.append(
                        ActionLogits(
                            ActionSubspace(
                                (name, library),
                                action_types[name],
                                self.subsampling[library].num_actions,
                                samples[library],
                            ),
                            library_logits,
                            library_weights,
                        )
                    )
            else:
                action_logits.append(
                    ActionLogits(
                        ActionSubspace(
                            (name, None), action_types[name], 1, samples[None]
                        ),
                        logits,
                        weights,
                    )
                )
        return ActionCategorical(action_logits, graph_emb, logit_scale)

    def get_action_logits(
        self, graph_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score numerator edges independently of the denominator subsample.

        Group observed actions by reaction and library so their scores use
        batched encodings, including actions absent from the denominator draw.
        """
        # 1. Group observed rows while retaining their original batch positions.
        by_reaction, by_library = {}, {}
        for row, action in enumerate(actions):
            name = (
                "first_block"
                if action.action_type == ActionType.FIRST_BLOCK
                else action.reaction
            )
            by_reaction.setdefault(name, []).append(row)
            if action.block_type is not None:
                by_library.setdefault(action.block_type, []).append(row)
        # 2. Gather and encode only the blocks selected by those actions.
        block_rows, features = [], []
        for name, rows in by_library.items():
            indices = np.array([actions[i].block_index for i in rows], dtype=np.int64)
            library = self.env.blocks[name]
            block_rows.extend(rows)
            features.append(
                (
                    library.fingerprints[indices],
                    library.properties[indices],
                    np.full(
                        (len(rows),), self.env.block_type_to_index[name], dtype=np.int64
                    ),
                )
            )
        blocks = graph_emb.new_zeros((len(actions), self.config.model.num_block_emb))
        if features:
            fp, prop, typ = zip(*features, strict=True)
            values = F.normalize(
                self.model.block_embedding(
                    torch.from_numpy(np.concatenate(fp)).to(
                        self.device, dtype=torch.float32
                    ),
                    torch.from_numpy(np.concatenate(prop)).to(self.device),
                    torch.from_numpy(np.concatenate(typ)).to(self.device),
                ),
                dim=-1,
            )
            blocks = blocks.index_copy(
                0, torch.tensor(block_rows, device=self.device), values
            )
        # 3. Score each reaction and restore the original action order.
        logits = graph_emb.new_zeros(len(actions))
        for name, rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            state_emb = self.model.forward_mdp(
                graph_emb[indices], name, logit_scale[indices]
            )
            values = (
                state_emb.squeeze(1)
                if actions[rows[0]].action_type == ActionType.UNI_REACTION
                else (state_emb * blocks[indices]).sum(1)
            )
            logits = logits.index_copy(0, indices, values)
        return logits

    @torch.no_grad()
    def sample_actions(
        self,
        states: list[State],
        sampling_temperature: float,
        random_action_prob: float,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> list[Action | None]:
        return self.forward(states, beta, preferences).sample(
            sampling_temperature,
            random_action_prob,
            self.config.subsampling.importance_temp,
        )

    def sample_action(
        self,
        state: State,
        sampling_temperature: float,
        random_action_prob: float,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Action:
        action = self.sample_actions(
            [state], sampling_temperature, random_action_prob, beta, preferences
        )[0]
        if action is None:
            raise NoValidActions(
                "the sampled action space has no budget-feasible continuation"
            )
        return action

    def log_prob(
        self,
        states: list[State],
        actions: list[Action],
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> torch.Tensor:
        fwd_cat = self.forward(states, beta, preferences)
        numerator = self.get_action_logits(
            fwd_cat.graph_emb, actions, fwd_cat.logit_scale
        )
        # Independent denominator subsampling can underestimate total mass;
        # cap log probabilities at zero to keep estimated probabilities <= 1.
        return (numerator - fwd_cat.log_partition()).clamp(max=0.0)

    def log_prob_single(
        self, state: State, action: Action, beta: torch.Tensor, preferences: torch.Tensor
    ) -> torch.Tensor:
        return self.log_prob([state], [action], beta, preferences)[0]

    def calc_bck_logprob(
        self,
        action: Action,
        trajectories: list[BackwardTrajectory],
        parent_smiles: str,
    ) -> float | None:
        """Normalize depth-weighted route mass for the observed reverse edge."""
        numerator = 0.0
        denominator = 0.0
        for trajectory in trajectories:
            # Exclude the root edge: the weight measures remaining actions to
            # the empty state. FirstBlock therefore has weight N**0 = 1.
            weight = self.env.num_total_actions ** (-(len(trajectory) - 1))
            denominator += weight
            if trajectory[0] == (action, parent_smiles):
                numerator += weight
        if numerator <= 0 or denominator <= 0:
            return None
        return math.log(numerator) - math.log(denominator)

    def rollout(
        self,
        sampling_temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
        *,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Trajectory:
        return self.rollouts(
            1,
            sampling_temperature,
            random_action_prob,
            analyze_backward,
            beta=beta,
            preferences=preferences,
        )[0]

    def rollouts(
        self,
        count: int,
        sampling_temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
        *,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> list[Trajectory]:
        """Grow a batch with fixed conditions and optional backward analysis."""
        # 1. Initialize trajectory state and keep conditions fixed for every step.
        if count <= 0:
            raise ValueError("rollout count must be positive")
        assert beta.shape == (count,)
        assert preferences.shape == (count, self.model.num_objectives)
        # Transfer once; retain Python metadata for serialization at termination.
        beta_values, preference_values = beta.tolist(), preferences.tolist()
        beta, preferences = beta.to(self.device), preferences.to(self.device)
        states = [self.env.initial_state() for _ in range(count)]
        steps: list[list[Transition]] = [[] for _ in range(count)]
        reasons: list[InvalidReason | None] = [None] * count
        # The empty state has one zero-action path. Every selected forward edge
        # prepends to its parent's paths before submitting the next reverse search.
        backward_trajectories: list[list[BackwardTrajectory]] = [
            [[]] for _ in range(count)
        ]

        def collect_backward() -> None:
            for index, routes in self.env.retro_analyzer.result():
                transition = steps[index][-1]
                value = self.calc_bck_logprob(
                    transition.action, routes, transition.state.smiles
                )
                if value is None:
                    raise RuntimeError("backward analysis lost the generated route")
                transition.log_p_B = value
                backward_trajectories[index] = routes

        # 2. Advance active trajectories: FirstBlock plus max_reactions reactions.
        for _ in range(self.env.max_reactions + 1):
            active = [
                i
                for i, state in enumerate(states)
                if reasons[i] is None and not state.terminated
            ]
            if not active:
                break
            selected = self.sample_actions(
                [states[index] for index in active],
                sampling_temperature,
                random_action_prob,
                beta[active],
                preferences[active],
            )
            # Collect the preceding reverse search after forward sampling so
            # worker processes can overlap it with model computation. Parent
            # routes must be ready before the next transition is submitted.
            if analyze_backward:
                collect_backward()
            for index, action in zip(active, selected, strict=True):
                state = states[index]
                if action is None:
                    reasons[index] = "no_valid_action"
                    continue
                # Preserve the sampled action even when chemistry fails. Its
                # forward probability must receive the invalid-reward TB signal;
                # dropping it would train only the prefix that reached this state.
                try:
                    next_state = self.env.step(state, action)
                except InvalidTransition:
                    steps[index].append(Transition(state, action, ""))
                    reasons[index] = "invalid_transition"
                    continue
                # Unexpected RDKit/model errors still surface to the caller.
                steps[index].append(Transition(state, action, next_state.smiles))
                states[index] = next_state
                if analyze_backward:
                    self.env.retro_analyzer.submit(
                        index,
                        next_state.smiles,
                        next_state.reaction_count,
                        [
                            [(action, state.smiles), *route]
                            for route in backward_trajectories[index]
                        ],
                    )

        # 3. Finish pending analysis, then serialize successes and failed attempts.
        if analyze_backward:
            collect_backward()  # Drain terminal states and the final pending batch.

        trajectories = []
        for index, state in enumerate(states):
            if not state.terminated and reasons[index] is None:
                reasons[index] = "max_reactions"
            trajectories.append(
                Trajectory(
                    steps=steps[index],
                    beta=beta_values[index],
                    preferences=preference_values[index],
                    final_smiles=state.smiles if reasons[index] is None else "",
                    valid=reasons[index] is None,
                    invalid_reason=reasons[index],
                )
            )
        return trajectories
