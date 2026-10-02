"""Reference-style masked policy matrices over the synthon environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from rxnflow.config import Config, SubsamplingConfig
from rxnflow.core.errors import InvalidTransition, NoValidActions
from rxnflow.core.types import (
    Action,
    ActionSubspace,
    ActionType,
    State,
    Trajectory,
    Transition,
)
from rxnflow.envs.env import SynthesisEnv
from rxnflow.envs.features import molecular_properties
from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data
from rxnflow.envs.retrosynthesis import RetrosynthesisTree
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
    logits: Tensor  # [state, sampled block], or [state, 1] for UniReaction
    log_importance: Tensor  # [sampled block]


class SubsamplingPolicy:
    """Reference uniform draws: one shared sample per needed library per batch."""

    def __init__(
        self, env: SynthesisEnv, config: SubsamplingConfig, rng: np.random.Generator
    ):
        self.rng = rng
        self.num_blocks = {name: len(library) for name, library in env.blocks.items()}
        self.num_sampling = {
            name: min(size, max(config.min_sampling, int(size * config.sampling_ratio)))
            for name, size in self.num_blocks.items()
        }
        self.log_importance = {
            name: math.log(size / self.num_sampling[name])
            for name, size in self.num_blocks.items()
        }
        self.full_indices = {
            name: np.arange(size, dtype=np.int64)
            for name, size in self.num_blocks.items()
            if self.num_sampling[name] == size
        }

    def sample(self, block_type: str) -> tuple[np.ndarray, float]:
        if block_type in self.full_indices:
            # No randomness or repeated allocation for full-library draws.
            return self.full_indices[block_type], 0.0
        indices = self.rng.choice(
            self.num_blocks[block_type], self.num_sampling[block_type], replace=False
        )
        # HSX sorts selected rows. Sorting does not alter uniform inclusion.
        indices.sort()
        return indices, self.log_importance[block_type]


@dataclass
class ActionCategorical:
    action_logits: list[ActionLogits]
    graph_emb: Tensor
    logit_scale: Tensor

    def log_partition(self) -> Tensor:
        # RxnFlow estimates the denominator from an independent subsample.
        # Unlike a fixed reference group mask, a budgeted subsample can be
        # entirely masked. A finite floor keeps observed-edge scoring defined;
        # the caller applies the reference nonpositive log P clamp.
        if not self.action_logits:
            return self.graph_emb.new_full((len(self.graph_emb),), math.log(1e-38))
        weighted = [p.logits + p.log_importance for p in self.action_logits]
        maxima = torch.stack([x.max(1).values for x in weighted]).max(0).values
        maxima = torch.where(torch.isfinite(maxima), maxima, 0).detach()
        totals = sum((x - maxima[:, None]).exp().sum(1) for x in weighted)
        return maxima + totals.clamp_min(1e-38).log()

    def sample(
        self, sampling_temperature: float, random_action_prob: float, importance: float
    ) -> list[Action | None]:
        # Draw on the model device. Only chosen group/column indices cross
        # to Python, not millions of candidate logits.
        if not self.action_logits:
            return [None] * len(self.graph_emb)
        random_rows = (
            torch.rand(len(self.graph_emb), device=self.graph_emb.device)
            < random_action_prob
        )
        best_values, best_columns = [], []
        for group in self.action_logits:
            values = group.logits + importance * group.log_importance
            if random_action_prob > 0:
                # This is the random policy, not a learned action logit. Balance
                # libraries by their sampled sizes before applying state masks.
                if group.subspace.libraries:
                    counts = (
                        group.subspace.library_sizes
                        if group.subspace.sample_indices is None
                        else [len(indices) for indices in group.subspace.sample_indices]
                    )
                    random_logits = torch.cat(
                        [
                            values.new_full(
                                (count,),
                                -math.log(len(group.subspace.libraries) * count),
                            )
                            for count in counts
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
        self.subsampling = SubsamplingPolicy(env, config.subsampling, rng)

    def forward(
        self, states: list[State], beta: Tensor, preferences: Tensor
    ) -> ActionCategorical:
        """Retain sampled columns; mask logits instead of packing valid actions.

        One common draw per library is the agreed adaptation of CGFlow. Each
        reaction concatenates its compatible libraries into one score matrix.
        No observed action enters these draws or their importance weights.
        """
        assert states
        graphs, properties = {}, {}
        keys = [(state.smiles, state.reaction_count) for state in states]
        # Metadata contains only state rows and library names, never candidate
        # objects or per-state arrays of valid block indices.
        action_rows, action_libraries, action_types, library_rows = {}, {}, {}, {}
        for row, (state, key) in enumerate(zip(states, keys, strict=True)):
            if key not in graphs:
                descriptors = molecular_properties(state.mol)
                properties[key] = torch.from_numpy(descriptors)
                graphs[key] = molecule_to_graph_data(
                    state.mol, self.env.max_atoms, state.reaction_count, descriptors
                )
            for subspace in self.env.get_action_space(state):
                name = subspace.name
                action_types[name] = subspace.action_type
                action_rows.setdefault(name, set()).add(row)
                for block_type in subspace.libraries:
                    action_libraries.setdefault(name, {}).setdefault(
                        block_type, []
                    ).append(row)
                    library_rows.setdefault(block_type, set()).add(row)
        descriptors = torch.stack([properties[key] for key in keys])
        samples, log_importance, masks, features, sizes = {}, {}, {}, [], []
        for name, rows in library_rows.items():
            samples[name], log_importance[name] = self.subsampling.sample(name)
            indices = torch.from_numpy(samples[name])
            rows = torch.tensor(sorted(rows))
            mask = torch.zeros((len(states), len(indices)), dtype=torch.bool)
            mask[rows] = self.env.get_block_mask(descriptors[rows], name, indices)
            masks[name] = mask
            library = self.env.blocks[name]
            features.append(
                (
                    library.fingerprints[indices],
                    library.properties[indices],
                    torch.full(
                        (len(indices),),
                        self.env.block_type_to_index[name],
                        dtype=torch.long,
                    ),
                )
            )
            sizes.append(len(indices))

        # Graph/property preprocessing is shared, but identical molecules may
        # have different beta/preferences and need separate neural encodings.
        cond_info = self.model.encode_cond(beta, preferences)
        graph_emb = self.model.graph_embedding(
            GraphBatch.from_graphs([graphs[key] for key in keys]).to(self.device),
            cond_info,
        )
        logit_scale = self.model.logit_scale(cond_info)
        block_embs = {}
        if features:
            fps, props, types = zip(*features, strict=True)
            encoded = F.normalize(
                self.model.block_embedding(
                    torch.cat(fps).to(self.device, dtype=torch.float32),
                    torch.cat(props).to(self.device),
                    torch.cat(types).to(self.device),
                ),
                dim=-1,
            )
            block_embs = dict(zip(samples, encoded.split(sizes), strict=True))

        action_logits = []
        for name, rows in action_rows.items():
            rows = torch.tensor(sorted(rows), device=self.device)
            state_emb = self.model.forward_mdp(graph_emb[rows], name, logit_scale[rows])
            libraries = list(action_libraries.get(name, {}))
            if libraries:
                # CGFlow: a single group matrix over concatenated libraries.
                blocks = torch.cat([block_embs[n] for n in libraries])
                scores = state_emb @ blocks.T
                logits = graph_emb.new_full((len(states), len(blocks)), -torch.inf)
                logits = logits.index_copy(0, rows, scores)
                allowed, weights = [], []
                for n in libraries:
                    eligible = torch.zeros(len(states), dtype=torch.bool)
                    eligible[action_libraries[name][n]] = True
                    allowed.append(masks[n] & eligible[:, None])
                    count = len(samples[n])
                    weights.append(torch.full((count,), log_importance[n]))
                logits = logits.masked_fill(
                    ~torch.cat(allowed, 1).to(self.device), -torch.inf
                )
                weights = torch.cat(weights).to(self.device)
            else:
                logits = graph_emb.new_full((len(states), 1), -torch.inf).index_copy(
                    0, rows, state_emb
                )
                weights = graph_emb.new_zeros(1)
            action_logits.append(
                ActionLogits(
                    ActionSubspace(
                        name,
                        action_types[name],
                        libraries,
                        [self.subsampling.num_blocks[n] for n in libraries],
                        [samples[n] for n in libraries],
                    ),
                    logits,
                    weights,
                )
            )
        return ActionCategorical(action_logits, graph_emb, logit_scale)

    def get_action_logits(
        self, graph_emb: Tensor, actions: list[Action], logit_scale: Tensor
    ) -> Tensor:
        """Score numerator edges independently of the denominator subsample.

        This is RxnFlow's _cal_action_logits, batched by reaction/library to
        preserve its values without one model invocation per training state.
        """
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
        block_rows, features = [], []
        for name, rows in by_library.items():
            indices = torch.tensor([actions[i].block_index for i in rows])
            library = self.env.blocks[name]
            block_rows.extend(rows)
            features.append(
                (
                    library.fingerprints[indices],
                    library.properties[indices],
                    torch.full(
                        (len(rows),), self.env.block_type_to_index[name], dtype=torch.long
                    ),
                )
            )
        blocks = graph_emb.new_zeros((len(actions), self.config.model.num_block_emb))
        if features:
            fp, prop, typ = zip(*features, strict=True)
            values = F.normalize(
                self.model.block_embedding(
                    torch.cat(fp).to(self.device, dtype=torch.float32),
                    torch.cat(prop).to(self.device),
                    torch.cat(typ).to(self.device),
                ),
                dim=-1,
            )
            blocks = blocks.index_copy(
                0, torch.tensor(block_rows, device=self.device), values
            )
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
        beta: Tensor,
        preferences: Tensor,
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
        beta: Tensor,
        preferences: Tensor,
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
        beta: Tensor,
        preferences: Tensor,
    ) -> Tensor:
        fwd_cat = self.forward(states, beta, preferences)
        numerator = self.get_action_logits(
            fwd_cat.graph_emb, actions, fwd_cat.logit_scale
        )
        return (numerator - fwd_cat.log_partition()).clamp(max=0.0)

    def log_prob_single(
        self, state: State, action: Action, beta: Tensor, preferences: Tensor
    ) -> Tensor:
        return self.log_prob([state], [action], beta, preferences)[0]

    def rollout(
        self,
        sampling_temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
        *,
        beta: Tensor,
        preferences: Tensor,
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
        beta: Tensor,
        preferences: Tensor,
    ) -> list[Trajectory]:
        if count <= 0:
            raise ValueError("rollout count must be positive")
        assert beta.shape == (count,)
        assert preferences.shape == (count, self.model.num_objectives)
        # Transfer once; retain Python metadata for serialization at termination.
        beta_values, preference_values = beta.tolist(), preferences.tolist()
        beta, preferences = beta.to(self.device), preferences.to(self.device)
        states = [self.env.initial_state() for _ in range(count)]
        steps: list[list[Transition]] = [[] for _ in range(count)]
        reasons: list[str | None] = [None] * count
        retro_trees = [RetrosynthesisTree("") for _ in range(count)]

        def collect_backward() -> None:
            for index, tree in self.env.retro_analyzer.result():
                transition = steps[index][-1]
                value = self.env.retro_analyzer.tree_log_probability(
                    tree,
                    transition.action,
                    self.env.num_total_actions,
                    transition.state.smiles,
                )
                if value is None or tree is None:
                    raise RuntimeError("backward analysis lost the generated route")
                transition.log_p_B = value
                retro_trees[index] = tree

        # FirstBlock + at most max_reactions chemical transformations.
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
            # RxnFlow pipeline: the previous reverse search runs while this
            # iteration encodes states and samples actions on the model device.
            if analyze_backward:
                collect_backward()
            for index, action in zip(active, selected, strict=True):
                state = states[index]
                if action is None:
                    reasons[index] = (
                        "the sampled action space has no budget-feasible continuation"
                    )
                    continue
                # Preserve the sampled action even when chemistry fails. Its
                # forward probability must receive the invalid-reward TB signal;
                # dropping it would train only the prefix that reached this state.
                try:
                    next_state = self.env.step(state, action)
                except InvalidTransition as error:
                    steps[index].append(Transition(state, action, ""))
                    reasons[index] = str(error)
                    continue
                # Unexpected RDKit/model errors still surface to the caller.
                steps[index].append(Transition(state, action, next_state.smiles))
                states[index] = next_state
                if analyze_backward:
                    self.env.retro_analyzer.submit(
                        index,
                        next_state.smiles,
                        next_state.reaction_count,
                        [(action, retro_trees[index])],
                    )

        if analyze_backward:
            collect_backward()  # Drain terminal states and the final pending batch.

        trajectories = []
        for index, state in enumerate(states):
            if not state.terminated and reasons[index] is None:
                reasons[index] = "reaction limit reached with an open handle"
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
