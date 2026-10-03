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


@dataclass
class ActionLogits:
    subspace: ActionSubspace
    logits: torch.Tensor  # [state, sampled synthon], or [state, 1] for UniReaction
    log_importance: torch.Tensor  # [sampled synthon]


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
        self, softmax_temperature: float, random_action_prob: float, importance: float
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
            values = values / softmax_temperature - (-noise.log()).log()
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
            **{name: len(library) for name, library in env.synthons.items()},
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

    def _get_state_inputs(
        self, states: list[State]
    ) -> tuple[GraphBatch, NDArray[np.float32]]:
        """Reuse molecular preprocessing for repeated states within a batch."""
        graphs: dict[str, GraphData] = {}
        properties: dict[str, NDArray[np.float32]] = {}
        keys = [state.smiles for state in states]
        for state, key in zip(states, keys, strict=True):
            if key not in graphs:
                mol_properties = molecular_properties(state.mol)
                properties[key] = mol_properties
                graphs[key] = molecule_to_graph_data(
                    state.mol, self.env.max_atoms, mol_properties
                )
        batch = GraphBatch.from_graphs([graphs[key] for key in keys]).to(self.device)
        return batch, np.stack([properties[key] for key in keys])

    def _encode_synthons(
        self,
        features: list[tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.int64]]],
    ) -> torch.Tensor:
        """Encode concatenated library rows with the same normalization everywhere."""
        fingerprints, properties, site_indices = zip(*features, strict=True)
        return F.normalize(
            self.model.synthon_embedding(
                torch.from_numpy(np.concatenate(fingerprints)).to(
                    self.device, dtype=torch.float32
                ),
                torch.from_numpy(np.concatenate(properties)).to(self.device),
                torch.from_numpy(np.concatenate(site_indices)).to(self.device),
            ),
            dim=-1,
        )

    def forward(
        self, states: list[State], beta: torch.Tensor, preferences: torch.Tensor
    ) -> ActionCategorical:
        """Retain sampled columns; mask logits instead of packing valid actions.

        Each library draw is shared across states and reactions. Each reaction
        shares one query/matmul across libraries of the same action type, then
        exposes one logit matrix per (reaction, library). Observed actions do not affect these draws or
        their importance weights.
        """
        assert states
        # 1. Build state features and collect compatible reaction/library rows.
        graph_batch, descriptors = self._get_state_inputs(states)
        # Metadata contains only state rows and library names, never candidate
        # objects or per-state arrays of valid synthon indices.
        action_rows: dict[tuple[str, ActionType], set[int]] = {}
        # (reaction, action type) -> library -> batch rows
        action_libraries: dict[tuple[str, ActionType], dict[str, list[int]]] = {}
        library_rows: dict[str | None, set[int]] = {}  # None is the unary range
        for row, state in enumerate(states):
            for subspace in self.env.get_action_space(state):
                name, library_name = subspace.name
                group = (name, subspace.action_type)
                action_rows.setdefault(group, set()).add(row)
                if library_name is not None:
                    action_libraries.setdefault(group, {}).setdefault(
                        library_name, []
                    ).append(row)
                library_rows.setdefault(library_name, set()).add(row)
        # 2. Draw each library once and apply additive budgets to sampled rows.
        samples: dict[str | None, NDArray[np.int64]] = {}
        log_importance: dict[str | None, float] = {}
        masks: dict[str, NDArray[np.bool_]] = {}  # library -> [batch, sampled actions]
        # Each entry contains fingerprints, properties and attachment/remaining type indices.
        features: list[
            tuple[NDArray[np.uint8], NDArray[np.float32], NDArray[np.int64]]
        ] = []
        library_names: list[str] = []
        sizes: list[int] = []
        for library_name, rows in library_rows.items():
            samples[library_name], log_importance[library_name] = self.subsampling[
                library_name
            ].sample()
            if library_name is None:
                continue  # Unary actions share subsampling, but have no synthon features.
            indices = samples[library_name]
            row_indices = sorted(rows)
            mask = np.zeros((len(states), len(indices)), dtype=np.bool_)
            mask[row_indices] = self.env.get_synthon_mask(
                descriptors[row_indices], library_name, indices
            )
            masks[library_name] = mask
            library_data = self.env.synthons[library_name]
            features.append(
                (
                    library_data.fingerprints[indices],
                    library_data.properties[indices],
                    np.full(
                        (len(indices), 2),
                        self.env.library_site_indices[library_name],
                        dtype=np.int64,
                    ),
                )
            )
            library_names.append(library_name)
            sizes.append(len(indices))

        # 3. Encode conditions, state graphs, and all sampled synthons in batches.
        # Graph/property preprocessing is shared, but identical molecules may
        # have different beta/preferences and need separate neural encodings.
        cond_info = self.model.encode_cond(beta, preferences)
        graph_emb = self.model.graph_embedding(
            graph_batch,
            cond_info,
        )
        logit_scale = self.model.logit_scale(cond_info)
        synthon_embs: dict[str, torch.Tensor] = {}
        if features:
            encoded = self._encode_synthons(features)
            synthon_embs = dict(zip(library_names, encoded.split(sizes), strict=True))

        # 4. Score by reaction and action type, apply masks, and split columns into subspaces.
        action_logits: list[ActionLogits] = []
        for group, rows in action_rows.items():
            name, action_type = group
            row_indices = torch.tensor(sorted(rows), device=self.device)
            state_emb = self.model.forward_mdp(
                graph_emb[row_indices], name, logit_scale[row_indices], action_type
            )
            libraries = list(action_libraries.get(group, {}))
            if libraries:
                # Compute the reaction query and matrix product once, then split
                # columns into library subspaces without recomputing embeddings.
                synthons = torch.cat([synthon_embs[n] for n in libraries])
                scores = state_emb @ synthons.T
                logits = graph_emb.new_full((len(states), len(synthons)), -torch.inf)
                logits = logits.index_copy(0, row_indices, scores)
                allowed: list[NDArray[np.bool_]] = []
                weight_parts: list[torch.Tensor] = []
                for n in libraries:
                    eligible = np.zeros(len(states), dtype=np.bool_)
                    eligible[action_libraries[group][n]] = True
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
                                action_type,
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
                        ActionSubspace((name, None), action_type, 1, samples[None]),
                        logits,
                        weights,
                    )
                )
        return ActionCategorical(action_logits, graph_emb, logit_scale)

    def get_action_logits(
        self, graph_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score numerator edges independently of the denominator subsample.

        Group observed actions by reaction, action type and library so their scores use
        batched encodings, including actions absent from the denominator draw.
        """
        # 1. Group observed rows while retaining their original batch positions.
        by_reaction, by_library = {}, {}
        for row, action in enumerate(actions):
            name = (
                "first_synthon"
                if action.action_type == ActionType.FIRST_SYNTHON
                else action.reaction
            )
            by_reaction.setdefault((name, action.action_type), []).append(row)
            if action.library_name is not None:
                by_library.setdefault(action.library_name, []).append(row)
        # 2. Gather and encode only the synthons selected by those actions.
        synthon_rows, features = [], []
        for name, rows in by_library.items():
            indices = np.array([actions[i].synthon_index for i in rows], dtype=np.int64)
            library = self.env.synthons[name]
            synthon_rows.extend(rows)
            features.append(
                (
                    library.fingerprints[indices],
                    library.properties[indices],
                    np.full(
                        (len(rows), 2),
                        self.env.library_site_indices[name],
                        dtype=np.int64,
                    ),
                )
            )
        synthons = graph_emb.new_zeros((len(actions), self.config.model.num_synthon_emb))
        if features:
            values = self._encode_synthons(features)
            synthons = synthons.index_copy(
                0, torch.tensor(synthon_rows, device=self.device), values
            )
        # 3. Score each reaction and restore the original action order.
        logits = graph_emb.new_zeros(len(actions))
        for (name, action_type), rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            state_emb = self.model.forward_mdp(
                graph_emb[indices], name, logit_scale[indices], action_type
            )
            values = (
                state_emb.squeeze(1)
                if action_type.is_unirxn
                else (state_emb * synthons[indices]).sum(1)
            )
            logits = logits.index_copy(0, indices, values)
        return logits

    @torch.no_grad()
    def sample_actions(
        self,
        states: list[State],
        softmax_temperature: float,
        random_action_prob: float,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> list[Action | None]:
        return self.forward(states, beta, preferences).sample(
            softmax_temperature,
            random_action_prob,
            self.config.subsampling.importance_temp,
        )

    def sample_action(
        self,
        state: State,
        softmax_temperature: float,
        random_action_prob: float,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Action:
        action = self.sample_actions(
            [state], softmax_temperature, random_action_prob, beta, preferences
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
        """Normalize synthon-count-weighted route mass for the observed reverse edge."""
        # This is an explicit route preference, independent of catalog size.
        # TODO: Review backward consistency with trajectory limits and search approximation.
        log_penalty = math.log(self.config.training.backward_synthon_penalty)
        numerator = denominator = -math.inf
        for trajectory in trajectories:
            num_synthons = sum(not edge.action_type.is_unirxn for edge, _ in trajectory)
            # Every complete route includes FirstSynthon. Subtract its common
            # count; extra UniReactions leave the route's weight unchanged.
            log_weight = -(num_synthons - 1) * log_penalty
            denominator = np.logaddexp(denominator, log_weight)
            if trajectory[0] == (action, parent_smiles):
                numerator = np.logaddexp(numerator, log_weight)
        if numerator == -math.inf:
            return None
        return float(numerator - denominator)

    def sample_from_model_single(
        self,
        softmax_temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
        *,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Trajectory:
        return self.sample_from_model(
            1,
            softmax_temperature,
            random_action_prob,
            analyze_backward,
            beta=beta,
            preferences=preferences,
        )[0]

    def sample_from_model(
        self,
        count: int,
        softmax_temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
        *,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> list[Trajectory]:
        """Grow a batch with fixed conditions and optional backward analysis."""
        # 1. Initialize trajectory state and keep conditions fixed for every step.
        if count <= 0:
            raise ValueError("sample count must be positive")
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
                # Approximate backward mass includes shorter chemical routes,
                # even when their counters differ from the generated history.
                # Forward action selection still enforces both synthesis budgets.
                value = self.calc_bck_logprob(
                    transition.action, routes, transition.state.smiles
                )
                if value is None:
                    raise RuntimeError("backward analysis lost the generated route")
                transition.log_p_B = value
                backward_trajectories[index] = routes

        # 2. Advance active trajectories: FirstSynthon plus max_reactions reactions.
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
                softmax_temperature,
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
                        self.env.max_reactions,
                        [
                            [(action, state.smiles), *route]
                            for route in backward_trajectories[index]
                            # Allow different lengths, but do not extend an
                            # alternative beyond the configured reaction cap.
                            if len(route) <= self.env.max_reactions
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
