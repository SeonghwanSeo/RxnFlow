"""Masked action distributions, shared library subsampling, and trajectory generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F

from rxnflow.config import Config
from rxnflow.core.errors import InvalidTransition
from rxnflow.core.types import (
    Action,
    ActionKey,
    ActionSpace,
    ActionSubspace,
    ActionType,
    BackwardTrajectory,
    InvalidReason,
    State,
    Trajectory,
    Transition,
)
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import FINGERPRINT_DIM, PROPERTY_DIM
from rxnflow.envs.graph import GraphBatch, GraphData, molecule_to_graph_data
from rxnflow.models import RxnFlowModel


@dataclass
class ActionLogits:
    subspace: ActionSubspace
    logits: torch.Tensor  # [sampled action] for one state.


class SubsamplingPolicy:
    """Uniform subsampling of a synthon library."""

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

    def sample(self) -> NDArray[np.int64]:
        if self.full_indices is not None:
            # Full ranges consume no RNG or new allocation.
            return self.full_indices
        indices = self.rng.choice(self.num_actions, self.num_sampling, replace=False)
        # Keep catalog row order without changing which actions were sampled.
        indices.sort()
        return indices


@dataclass
class ActionCategorical:
    action_logits: list[list[ActionLogits]]  # One list of subspaces per state.
    state_emb: torch.Tensor
    logit_scale: torch.Tensor
    subsampling: dict[str, SubsamplingPolicy]

    def _subspace_metadata(
        self, importance: float = 1.0, exploration: bool = False
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Transfer subspace counts and weights together before state-wise reductions."""
        widths, sizes, weights = [], [], []
        dtype = self.state_emb.dtype
        for state_logits in self.action_logits:
            widths.append(len(state_logits))
            for entry in state_logits:
                size = len(entry.logits)
                library_name = entry.subspace.name[1]
                weight = (
                    0.0
                    if library_name is None
                    else self.subsampling[library_name].log_importance
                )
                sizes.append(size)
                weights.append(
                    (importance * weight, -math.log(size) if exploration else 0.0)
                )
                dtype = entry.logits.dtype
        counts = torch.tensor(sizes, dtype=torch.long, device="cpu").to(
            self.state_emb.device, non_blocking=True
        )
        weights = torch.tensor(weights, dtype=dtype, device="cpu").to(
            self.state_emb.device, non_blocking=True
        )
        return counts.split(widths), weights.split(widths)

    def log_partition(self) -> torch.Tensor:
        """Estimate each state's log partition using library inclusion weights."""
        counts_by_state, weights_by_state = self._subspace_metadata()
        partitions = []
        for state_logits, counts, weights in zip(
            self.action_logits, counts_by_state, weights_by_state, strict=True
        ):
            if not state_logits:
                partitions.append(self.state_emb.new_tensor(math.log(1e-38)))
                continue
            logits = torch.cat([entry.logits for entry in state_logits])
            # Known output sizes avoid device synchronization during expansion.
            offsets = weights[:, 0].repeat_interleave(
                counts, output_size=logits.numel()
            )
            weighted = logits + offsets
            # Floor zero total mass without reviving excluded actions.
            maximum = weighted.max().detach()
            maximum = torch.where(torch.isfinite(maximum), maximum, 0)
            total = (weighted - maximum).exp().sum()
            partitions.append(maximum + total.clamp_min(1e-38).log())
        return torch.stack(partitions)

    def sample(
        self, softmax_temperature: float, random_action_prob: float, importance: float
    ) -> list[Action]:
        """Draw one action per state; fully excluded candidates are sampled uniformly."""
        random_states = (
            torch.rand(len(self.action_logits), device=self.state_emb.device)
            < random_action_prob
        )
        counts_by_state, weights_by_state = self._subspace_metadata(
            importance, exploration=random_action_prob > 0
        )
        columns = []
        for state_i, (state_logits, counts, weights) in enumerate(
            zip(self.action_logits, counts_by_state, weights_by_state, strict=True)
        ):
            assert state_logits
            logits = torch.cat([entry.logits for entry in state_logits])
            # Known output sizes avoid device synchronization during expansion.
            offsets = weights[:, 0].repeat_interleave(
                counts, output_size=logits.numel()
            )
            values = logits + offsets
            if random_action_prob > 0:
                # Each subspace has unit mass before masking at temperature 1.
                random_logits = weights[:, 1].repeat_interleave(
                    counts, output_size=logits.numel()
                )
                values = torch.where(random_states[state_i], random_logits, values)
            values = values.masked_fill(~torch.isfinite(logits), -torch.inf)
            values = values.clamp_min(math.log(1e-38))
            values = values / softmax_temperature
            # Gumbel-max samples from softmax without constructing probabilities.
            uniform = torch.rand_like(values).clamp_min(torch.finfo(values.dtype).tiny)
            gumbels = -(-uniform.log()).log()
            columns.append((values + gumbels).argmax())

        # Transfer all winners together before decoding sampled subspace indices.
        actions: list[Action] = []
        for state_logits, column in zip(
            self.action_logits, torch.stack(columns).tolist(), strict=True
        ):
            for entry in state_logits:
                if column < len(entry.logits):
                    actions.append(entry.subspace.action_at(column))
                    break
                column -= len(entry.logits)
        return actions


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
        # Each library supplies one shared sample per forward call.
        self.subsampling = {
            name: SubsamplingPolicy(
                len(library),
                config.subsampling.sampling_ratio,
                config.subsampling.min_sampling,
                rng,
            )
            for name, library in env.synthons.items()
        }

    def _get_state_inputs(self, states: list[State]) -> GraphBatch:
        """Reuse molecular preprocessing for repeated states in a batch."""
        graphs: dict[str, GraphData] = {}
        for state in states:
            if state.smiles not in graphs:
                cached = state._cache.get("graph")
                graphs[state.smiles] = (
                    molecule_to_graph_data(state.mol) if cached is None else cached
                )
            state._cache["graph"] = graphs[state.smiles]
        return GraphBatch.from_list([graphs[state.smiles] for state in states])

    def _get_synthon_features(
        self, library_name: str, indices: NDArray[np.int64]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        library = self.env.synthons[library_name]
        fp, prop = library.fingerprints, library.properties
        types = self.env.library_site_indices[library_name]
        # Non-blocking device/dtype conversion casts uint8 fingerprints on GPU.
        fp = torch.from_numpy(fp[indices]).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        prop = torch.from_numpy(prop[indices]).to(self.device, non_blocking=True)
        types = torch.tensor(types, dtype=torch.long, device=self.device)
        types = types.expand(len(indices), 2)
        return fp, prop, types

    def _get_size_mask(
        self,
        state_size: torch.Tensor,
        synthon_size: torch.Tensor,
    ) -> torch.Tensor:
        return state_size + synthon_size <= self.env.max_atoms

    def _get_property_penalty(
        self,
        state_prop: torch.Tensor,
        synthon_prop: torch.Tensor,
    ) -> torch.Tensor:
        """Apply additive property limits in raw units."""
        property_penalty = torch.ones(
            (*state_prop.shape[:-1], len(synthon_prop)),
            dtype=torch.bool,
            device=synthon_prop.device,
        )
        for index, limit in self.env.property_limits.items():
            estimate = state_prop[..., index, None] + synthon_prop[:, index]
            # Nonzero bounds allow 1% tolerance; zero bounds remain exact.
            if limit == 0:
                property_penalty &= estimate <= 0
            else:
                property_penalty &= estimate < limit + abs(limit) * 0.01
        return property_penalty

    def _encode_synthons(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Encode synthons with unit-norm embeddings for action scoring."""
        return F.normalize(self.model.encode_synthon(*features), dim=-1)

    def forward(
        self, states: list[State], beta: torch.Tensor, preferences: torch.Tensor
    ) -> ActionCategorical:
        """Score each state's actions using shared library samples and encodings.
        This is not used; forward_batch is preferred for efficiency.
        """
        assert states
        # 1. Encode the states and prepare all needed synthons before scoring.
        batch = self._get_state_inputs(states).to(self.device)
        cond_info = self.model.encode_cond(beta, preferences)
        state_emb = self.model.encode_state(batch, cond_info)
        logit_scale = self.model.logit_scale(cond_info)
        action_spaces = self._prepare_action_space(states)
        synthon_cache = self._get_synthon_cache(action_spaces)

        # 2. Score each state's subspaces and apply property penalties.
        action_logits: list[list[ActionLogits]] = []
        for state_i, action_space in enumerate(action_spaces):
            emb = state_emb[state_i]
            scale = logit_scale[state_i]
            state_prop, state_size = batch.mol_features[state_i], batch.size[state_i]

            state_logits: list[ActionLogits] = []
            for subspace in action_space:
                action_name, library_name = subspace.name
                action_type = subspace.action_type
                if action_type.is_unirxn:
                    logits = self._unirxn_logits(action_type, action_name, emb) * scale
                elif action_type.is_first:
                    synthon_emb, synthon_prop, synthon_size = synthon_cache[library_name]
                    logits = self._first_synthon_logits(emb, synthon_emb) * scale
                    size_mask = self._get_size_mask(state_size, synthon_size)
                    property_penalty = self._get_property_penalty(
                        state_prop, synthon_prop
                    )
                    logits = logits.masked_fill(
                        ~(size_mask & property_penalty), -torch.inf
                    )
                else:
                    synthon_emb, synthon_prop, synthon_size = synthon_cache[library_name]
                    logits = (
                        self._birxn_logits(action_type, action_name, emb, synthon_emb)
                        * scale
                    )
                    size_mask = self._get_size_mask(state_size, synthon_size)
                    property_penalty = self._get_property_penalty(
                        state_prop, synthon_prop
                    )
                    logits = logits.masked_fill(
                        ~(size_mask & property_penalty), -torch.inf
                    )
                state_logits.append(ActionLogits(subspace, logits))

            action_logits.append(state_logits)
        return ActionCategorical(action_logits, state_emb, logit_scale, self.subsampling)

    def forward_batch(
        self, states: list[State], beta: torch.Tensor, preferences: torch.Tensor
    ) -> ActionCategorical:
        """Batch states by attachment and score each head's libraries together."""
        assert states
        # 1. Encode states and all sampled synthons before scoring.
        batch = self._get_state_inputs(states).to(self.device)
        cond_info = self.model.encode_cond(beta, preferences)
        state_emb = self.model.encode_state(batch, cond_info)
        logit_scale = self.model.logit_scale(cond_info)
        action_spaces = self._prepare_action_space(states)
        synthon_cache = self._get_synthon_cache(action_spaces)

        # 2. Group states by attachment type
        state_groups: dict[int | None, list[int]] = {}
        for state_i, state in enumerate(states):
            assert action_spaces[state_i]
            state_groups.setdefault(state.attachment_type, []).append(state_i)

        action_logits: list[list[ActionLogits]] = [[] for _ in states]
        for rows in state_groups.values():
            indices = torch.tensor(rows, device=self.device)
            emb = state_emb[indices]
            scale = logit_scale[indices, None]
            subspaces = {
                subspace.name: subspace
                for state_i in rows
                for subspace in action_spaces[state_i]
            }
            spaces_by_action: dict[tuple[ActionType, str], ActionSpace] = {}
            for subspace in subspaces.values():
                key = (subspace.action_type, subspace.name[0])
                spaces_by_action.setdefault(key, []).append(subspace)

            # 3. Apply size/property constraints to all libraries in the group.
            library_names = list(
                dict.fromkeys(
                    space.name[1]
                    for space in subspaces.values()
                    if space.name[1] is not None
                )
            )
            library_masks: dict[str, torch.Tensor] = {}
            if library_names:
                prop = torch.cat([synthon_cache[name][1] for name in library_names])
                size = torch.cat([synthon_cache[name][2] for name in library_names])
                size_mask = self._get_size_mask(batch.size[indices, None], size)
                property_penalty = self._get_property_penalty(
                    batch.mol_features[indices], prop
                )
                widths = [len(synthon_cache[name][0]) for name in library_names]
                masks = (size_mask & property_penalty).split(widths, dim=1)
                library_masks = dict(zip(library_names, masks, strict=True))

            # 4. One head call and matrix product for all of its library candidates.
            group_logits: dict[ActionKey, tuple[torch.Tensor, ...]] = {}
            for (action_type, action_name), spaces in spaces_by_action.items():
                if action_type.is_unirxn:
                    logits = self._unirxn_logits(action_type, action_name, emb) * scale
                    group_logits[spaces[0].name] = logits.unbind(0)
                    continue

                libraries = [space.name[1] for space in spaces]
                synthons = torch.cat([synthon_cache[name][0] for name in libraries])
                if action_type.is_first:
                    logits = self._first_synthon_logits(emb, synthons) * scale
                else:
                    logits = (
                        self._birxn_logits(action_type, action_name, emb, synthons)
                        * scale
                    )
                mask = torch.cat([library_masks[name] for name in libraries], dim=1)
                logits = logits.masked_fill(~mask, -torch.inf)
                widths = [len(synthon_cache[name][0]) for name in libraries]
                for subspace, values in zip(
                    spaces, logits.split(widths, dim=1), strict=True
                ):
                    group_logits[subspace.name] = values.unbind(0)

            # 5. Preserve each state's budget eligibility and original subspace order.
            for local_row, state_i in enumerate(rows):
                action_logits[state_i] = [
                    ActionLogits(subspace, group_logits[subspace.name][local_row])
                    for subspace in action_spaces[state_i]
                ]

        return ActionCategorical(action_logits, state_emb, logit_scale, self.subsampling)

    def _prepare_action_space(self, states: list[State]) -> list[ActionSpace]:
        """Resolve eligible subspaces with one shared sample per library."""
        samples: dict[str, NDArray[np.int64]] = {}
        sampled_spaces: list[ActionSpace] = []
        for state in states:
            sampled_space: ActionSpace = []
            for subspace in self.env.get_action_space(state):
                library_name = subspace.name[1]
                if library_name is None:  # Unimolecular reactions.
                    sampled_space.append(subspace)
                    continue
                if library_name not in samples:
                    samples[library_name] = self.subsampling[library_name].sample()
                indices = samples[library_name]
                sampled_space.append(
                    ActionSubspace(
                        subspace.name, subspace.action_type, subspace.num_actions, indices
                    )
                )
            assert sampled_space, f"state {state.smiles} has no valid actions"
            sampled_spaces.append(sampled_space)
        return sampled_spaces

    def _get_synthon_cache(
        self, action_spaces: list[ActionSpace]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode all sampled libraries together, then share their embedding views."""
        samples: dict[str, NDArray[np.int64]] = {}
        for space in action_spaces:
            for subspace in space:
                library_name = subspace.name[1]
                if library_name is not None:
                    samples.setdefault(library_name, subspace.sample_indices)
        if not samples:
            return {}

        widths = [len(indices) for indices in samples.values()]
        count = sum(widths)
        fp_arr = np.empty((count, FINGERPRINT_DIM), dtype=np.uint8)
        prop_arr = np.empty((count, PROPERTY_DIM), dtype=np.float32)
        type_arr = np.empty((count, 2), dtype=np.int64)
        size_arr = np.empty(count, dtype=np.uint8)
        offset = 0
        for library_name, indices in samples.items():
            library = self.env.synthons[library_name]
            rows = slice(offset, offset + len(indices))
            # Subsampling supplies valid indices. Clip mode lets take write directly
            # into the final buffer without the temporary used by raise mode.
            np.take(library.fingerprints, indices, axis=0, out=fp_arr[rows], mode="clip")
            np.take(library.properties, indices, axis=0, out=prop_arr[rows], mode="clip")
            np.take(library.heavy_atoms, indices, out=size_arr[rows], mode="clip")
            type_arr[rows] = self.env.library_site_indices[library_name]
            offset += len(indices)

        # Transfer to device.
        fp = torch.from_numpy(fp_arr).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        prop = torch.from_numpy(prop_arr).to(self.device, non_blocking=True)
        types = torch.from_numpy(type_arr).to(
            self.device, dtype=torch.long, non_blocking=True
        )
        size = torch.from_numpy(size_arr).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        embeddings = self._encode_synthons((fp, prop, types))
        return {
            library_name: (emb, prop, size)
            for library_name, emb, prop, size in zip(
                samples,
                embeddings.split(widths),
                prop.split(widths),
                size.split(widths),
                strict=True,
            )
        }

    def _first_synthon_logits(
        self,
        state_emb: torch.Tensor,
        synthons: torch.Tensor,
    ) -> torch.Tensor:
        """Score initial synthons with a state-only query."""
        query = self.model.forward_first_synthon(state_emb)
        return query @ synthons.T

    def _unirxn_logits(
        self,
        action_type: ActionType,
        action_name: str,
        state_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score the single action of a unimolecular reaction."""
        return self.model.forward_unirxn(state_emb, action_name, action_type)

    def _birxn_logits(
        self,
        action_type: ActionType,
        action_name: str,
        state_emb: torch.Tensor,
        synthons: torch.Tensor,
    ) -> torch.Tensor:
        """Score incoming synthons with a state/reaction query."""
        query = self.model.forward_birxn(state_emb, action_name, action_type)
        return query @ synthons.T

    def get_action_logits(
        self, state_emb: torch.Tensor, actions: list[Action], logit_scale: torch.Tensor
    ) -> torch.Tensor:
        """Score numerator edges independently of the denominator subsample.

        Group heads by (action type, name) and synthon features by library.
        Observed actions are scored even if absent from the denominator sample.
        """
        # 1. Group observed rows while retaining their original batch positions.
        by_reaction, by_library = {}, {}
        for row, action in enumerate(actions):
            name = "" if action.action_type.is_first else action.reaction
            by_reaction.setdefault((action.action_type, name), []).append(row)
            if action.library_name is not None:
                by_library.setdefault(action.library_name, []).append(row)
        # 2. Gather and encode only the synthons selected by those actions.
        synthon_rows, features = [], []
        for name, rows in by_library.items():
            indices = np.array([actions[i].synthon_index for i in rows], dtype=np.int64)
            synthon_rows.extend(rows)
            features.append(self._get_synthon_features(name, indices))
        synthons = state_emb.new_zeros((len(actions), self.config.model.synthon_dim))
        if features:
            fp, prop, types = zip(*features, strict=True)
            values = self._encode_synthons(
                (torch.cat(fp), torch.cat(prop), torch.cat(types))
            )
            synthons = synthons.index_copy(
                0, torch.tensor(synthon_rows, device=self.device), values
            )
        # 3. Score each action group and restore the original batch order.
        logits = state_emb.new_zeros(len(actions))
        for (action_type, action_name), rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            if action_type.is_unirxn:
                values = self._unirxn_logits(
                    action_type, action_name, state_emb[indices]
                ).squeeze(1)
            elif action_type.is_first:
                query = self.model.forward_first_synthon(state_emb[indices])
                values = (query * synthons[indices]).sum(1)
            else:
                query = self.model.forward_birxn(
                    state_emb[indices], action_name, action_type
                )
                values = (query * synthons[indices]).sum(1)
            logits = logits.index_copy(0, indices, values)
        return logits * logit_scale

    @torch.no_grad()
    def sample_actions(
        self,
        states: list[State],
        softmax_temperature: float,
        random_action_prob: float,
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> list[Action]:
        return self.forward_batch(states, beta, preferences).sample(
            softmax_temperature,
            random_action_prob,
            self.config.subsampling.importance_temp,
        )

    def log_prob(
        self,
        states: list[State],
        actions: list[Action],
        beta: torch.Tensor,
        preferences: torch.Tensor,
    ) -> torch.Tensor:
        fwd_cat = self.forward_batch(states, beta, preferences)
        numerator = self.get_action_logits(
            fwd_cat.state_emb, actions, fwd_cat.logit_scale
        )
        # Independent denominator subsampling can underestimate total mass;
        # cap log probabilities at zero to keep estimated probabilities <= 1.
        return (numerator - fwd_cat.log_partition()).clamp(max=0.0)

    def calc_bck_logprob(
        self,
        action: Action,
        trajectories: list[BackwardTrajectory],
        parent_smiles: str,
    ) -> float | None:
        """Normalize synthon-count-weighted route mass for the observed reverse edge."""
        # This is an explicit route preference, independent of catalog size.
        # TODO: Review backward consistency with trajectory limits and search
        # approximation.
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
        # Retain condition values for serialization at termination.
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
