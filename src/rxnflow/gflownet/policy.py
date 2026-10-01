"""Shared policy execution over exact, site-specific synthon transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from rxnflow.config import Config
from rxnflow.envs.chemistry.features import molecular_properties
from rxnflow.envs.env import ActionGroup, InvalidTransition, SynthesisEnv
from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data
from rxnflow.envs.retrosynthesis import RetrosynthesisTree
from rxnflow.gflownet.categorical import sample_position
from rxnflow.gflownet.subsampling import BlockSubsampler
from rxnflow.gflownet.types import (
    ActionKind,
    MoleculeState,
    RxnAction,
    Sample,
    Trajectory,
    TrajectoryStep,
)
from rxnflow.models import RxnFlowModel


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class NoValidActions(ValueError):
    """The sampled action space has no budget-feasible continuation."""


@dataclass
class ActionLogits:
    """Budget-masked action choices, logits, and subsampling corrections."""

    groups: list[tuple[ActionGroup, Tensor]]
    logits: Tensor
    log_importance: Tensor
    required_position: int | None = None

    def action_at(self, position: int) -> RxnAction:
        """Materialize only the selected row; candidates remain index tensors."""
        for group, indices in self.groups:
            count = len(indices)
            if position < count:
                return RxnAction(
                    group.kind,
                    "",
                    reaction=None if group.kind == ActionKind.FIRST_BLOCK else group.name,
                    block_type=group.block_type,
                    block_index=int(indices[position])
                    if group.block_type is not None
                    else None,
                )
            position -= count
        raise IndexError("action position outside candidate space")


class SynthesisPolicy:
    """Forward policy P_F: score actions and sample synthesis trajectories.

    Trainer and sampler share this code; the trainable parameters belong to
    RxnFlowModel. Backward probabilities come from the environment's search.
    """

    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
        generator: torch.Generator,
    ):
        self.env = env
        self.model = model
        self.config = config
        self.device = device
        self.generator = generator
        self.block_subsamplers = {
            name: BlockSubsampler(len(library), config.subsampling)
            for name, library in env.blocks.items()
        }

    def candidate_batch(
        self,
        states: list[MoleculeState],
        required: list[RxnAction] | None = None,
    ) -> list[ActionLogits]:
        """Share library draws and score query/block matrices, then gather masks.

        Encode each sampled block once. A matrix product per library avoids
        expanding two hidden vectors for every state/candidate pair. Metadata
        and masks stay on CPU; only packed indices/features cross to the model.
        """
        assert states and (required is None or len(required) == len(states))
        graphs, properties = {}, {}
        required_rows: dict[str, set[int]] = {}
        for action in required or []:
            if action.block_type is not None:
                required_rows.setdefault(action.block_type, set()).add(action.block_index)
        shared_samples, masked_samples = {}, {}
        action_groups, required_positions, counts, importance = [], [], [], []
        query_states, query_actions, query_kinds = [], [], []
        # Each library records (query row, sampled columns, output offset).
        block_groups = {}
        unary_queries, unary_outputs = [], []
        keys = [(state.smiles, state.reaction_count) for state in states]
        available = [self.env.available_groups(state) for state in states]
        library_states = {}
        for state, key, groups in zip(states, keys, available, strict=True):
            if key not in graphs:
                descriptors = molecular_properties(state.mol)
                properties[key] = torch.from_numpy(descriptors)
                graphs[key] = molecule_to_graph_data(
                    state.mol, self.env.max_atoms, state.reaction_count, descriptors
                )
            for group in groups:
                if group.block_type is not None:
                    library_states.setdefault(group.block_type, {}).setdefault(key, None)
        for name, state_keys in library_states.items():
            sample = self.block_subsamplers[name].sample(
                self.generator, tuple(required_rows.get(name, ()))
            )
            shared_samples[name] = sample
            masks = self.env.block_mask(
                torch.stack([properties[key] for key in state_keys]), name, sample.indices
            )
            for key, allowed in zip(state_keys, masks, strict=True):
                columns = allowed.nonzero().flatten()
                # Preserve the draw's importance weights; masks never refill it.
                masked_samples[key, name] = (
                    sample.indices[columns],
                    sample.log_importance[columns],
                    columns,
                )
        output_count = 0
        for state_index, key in enumerate(keys):
            observed = None if required is None else required[state_index]
            state_groups, queries = [], {}
            state_count = 0
            required_position = None
            for group in available[state_index]:
                name = group.block_type
                if name is not None:
                    indices, weights, columns = masked_samples[key, name]
                    if not len(indices):
                        continue
                else:
                    indices = torch.tensor([-1], dtype=torch.long)
                    weights = torch.zeros(1)
                if group.name not in queries:
                    queries[group.name] = len(query_states)
                    query_states.append(state_index)
                    query_actions.append(self.env.action_to_index[group.name])
                    query_kinds.append(group.kind)
                query = queries[group.name]
                if name is None:
                    unary_queries.append(query)
                    unary_outputs.append(output_count + state_count)
                else:
                    block_groups.setdefault(name, []).append(
                        (query, columns, output_count + state_count)
                    )
                if observed is not None and (
                    observed.kind == group.kind
                    and observed.block_type == name
                    and observed.reaction
                    == (None if group.kind == ActionKind.FIRST_BLOCK else group.name)
                ):
                    position = (
                        0
                        if name is None
                        else int(torch.searchsorted(indices, observed.block_index))
                    )
                    assert position < len(indices) and (
                        name is None or indices[position] == observed.block_index
                    ), "observed block is masked out"
                    required_position = state_count + position
                state_groups.append((group, indices))
                importance.append(weights)
                state_count += len(indices)
            if observed is not None:
                assert required_position is not None, "observed action is not available"
            action_groups.append(state_groups)
            required_positions.append(required_position)
            counts.append(state_count)
            output_count += state_count

        if not output_count:
            empty = torch.empty(0, device=self.device)
            return [ActionLogits([], empty, empty) for _ in states]
        if self.model.training:
            # Preserve independent dropout for each occurrence during training.
            batch = GraphBatch.from_graphs([graphs[key] for key in keys]).to(self.device)
            embeddings = self.model.encode_graphs(batch)
        else:
            # All initial states are identical during rollout. Encode unique
            # states once, then gather; nothing is cached beyond this call.
            batch = GraphBatch.from_graphs(list(graphs.values())).to(self.device)
            graph_indices = {key: index for index, key in enumerate(graphs)}
            embeddings = self.model.encode_graphs(batch)[
                torch.tensor([graph_indices[key] for key in keys], device=self.device)
            ]
        kinds = torch.tensor(query_kinds, dtype=torch.long)
        queries, unary = self.model.action_queries(
            embeddings[torch.tensor(query_states, device=self.device)],
            torch.tensor(query_actions, device=self.device),
            [(kinds == kind).nonzero().flatten().to(self.device) for kind in ActionKind],
        )
        logits = embeddings.new_zeros(output_count)
        if block_groups:
            fingerprints, descriptors, types = [], [], []
            matrix_queries, matrix_indices, output_indices = [], [], []
            sizes = []
            for name, entries in block_groups.items():
                library = self.env.blocks[name]
                indices = shared_samples[name].indices
                fingerprints.append(library.fingerprints[indices])
                descriptors.append(library.properties[indices])
                types.append(
                    torch.full(
                        (len(indices),),
                        self.env.block_type_to_index[name],
                        dtype=torch.long,
                    )
                )
                matrix_queries.extend(query for query, _, _ in entries)
                selected_count = 0
                for row, (_, columns, offset) in enumerate(entries):
                    matrix_indices.append(row * len(indices) + columns)
                    output_indices.append(torch.arange(offset, offset + len(columns)))
                    selected_count += len(columns)
                sizes.append((len(indices), len(entries), selected_count))
            blocks = torch.nn.functional.normalize(
                self.model.encode_block_features(
                    torch.cat(fingerprints).to(self.device, dtype=torch.float32),
                    torch.cat(descriptors).to(self.device),
                    torch.cat(types).to(self.device),
                ),
                dim=-1,
            )
            matrix_queries = torch.tensor(matrix_queries, device=self.device)
            matrix_indices = torch.cat(matrix_indices).to(self.device)
            output_indices = torch.cat(output_indices).to(self.device)
            values = []
            block_offset = query_offset = selected_offset = 0
            for n_blocks, n_queries, n_selected in sizes:
                # Same matrix scoring as RxnFlow master/CGFlow. Masked rows
                # are gathered afterwards; no [candidate_count, hidden] copies.
                scores = (
                    queries[matrix_queries[query_offset : query_offset + n_queries]]
                    @ blocks[block_offset : block_offset + n_blocks].T
                )
                values.append(
                    scores.flatten()[
                        matrix_indices[selected_offset : selected_offset + n_selected]
                    ]
                )
                block_offset += n_blocks
                query_offset += n_queries
                selected_offset += n_selected
            logits = logits.index_copy(0, output_indices, torch.cat(values))
        if unary_queries:
            logits = logits.index_copy(
                0,
                torch.tensor(unary_outputs, device=self.device),
                unary[torch.tensor(unary_queries, device=self.device)],
            )
        weights = torch.cat(importance).to(self.device)
        return [
            ActionLogits(groups, scores, correction, position)
            for groups, scores, correction, position in zip(
                action_groups,
                logits.split(counts),
                weights.split(counts),
                required_positions,
                strict=True,
            )
        ]

    def candidates(
        self, state: MoleculeState, required: RxnAction | None = None
    ) -> ActionLogits:
        result = self.candidate_batch([state], None if required is None else [required])[
            0
        ]
        if not result.logits.numel():
            raise NoValidActions(
                "the sampled action space has no budget-feasible continuation"
            )
        return result

    @torch.no_grad()
    def choose_actions(
        self, states: list[MoleculeState], temperature: float, random_action_prob: float
    ) -> list[RxnAction | None]:
        candidates = self.candidate_batch(states)
        # Copy all sampling logits in one transfer, rather than synchronizing
        # CUDA once for each molecule. Exploration still uses the saved CPU RNG.
        counts = [value.logits.numel() for value in candidates]
        logits = (
            torch.cat(
                [
                    value.logits
                    + self.config.subsampling.importance_temp * value.log_importance
                    for value in candidates
                ]
            )
            .cpu()
            .split(counts)
        )
        return [
            value.action_at(
                sample_position(scores, self.generator, temperature, random_action_prob)
            )
            if value.logits.numel()
            else None
            for value, scores in zip(candidates, logits, strict=True)
        ]

    def choose_action(
        self, state: MoleculeState, temperature: float, random_action_prob: float
    ) -> RxnAction:
        action = self.choose_actions([state], temperature, random_action_prob)[0]
        if action is None:
            raise NoValidActions(
                "the sampled action space has no budget-feasible continuation"
            )
        return action

    def action_log_probabilities(
        self, states: list[MoleculeState], actions: list[RxnAction]
    ) -> Tensor:
        candidates = self.candidate_batch(states, required=actions)
        logits = torch.cat([value.logits for value in candidates])
        corrected = logits + torch.cat([value.log_importance for value in candidates])
        counts = [value.logits.numel() for value in candidates]
        lengths = torch.tensor(counts, device=self.device)
        positions = []
        offset = 0
        for value, count in zip(candidates, counts, strict=True):
            positions.append(offset + value.required_position)
            offset += count
        # Segmented logsumexp avoids one small CUDA reduction per state, while
        # keeping ragged candidates instead of padding to the largest library.
        maxima = torch.segment_reduce(corrected.detach(), "max", lengths=lengths)
        shifted = corrected - torch.repeat_interleave(
            maxima, lengths, output_size=len(logits)
        )
        totals = torch.segment_reduce(shifted.exp(), "sum", lengths=lengths)
        return logits[torch.tensor(positions, device=self.device)] - maxima - totals.log()

    def action_log_probability(self, state: MoleculeState, action: RxnAction) -> Tensor:
        return self.action_log_probabilities([state], [action])[0]

    def rollout(
        self,
        temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
    ) -> Trajectory:
        return self.rollouts(1, temperature, random_action_prob, analyze_backward)[0]

    def rollouts(
        self,
        count: int,
        temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
    ) -> list[Trajectory]:
        if count <= 0:
            raise ValueError("rollout count must be positive")
        states = [self.env.initial_state() for _ in range(count)]
        steps: list[list[TrajectoryStep]] = [[] for _ in range(count)]
        reasons: list[str | None] = [None] * count
        retro_trees = [RetrosynthesisTree("") for _ in range(count)]

        # FirstBlock + at most max_reactions chemical transformations.
        for _ in range(self.env.max_reactions + 1):
            active = [
                i
                for i, state in enumerate(states)
                if reasons[i] is None and not state.terminated
            ]
            if not active:
                break
            selected = self.choose_actions(
                [states[index] for index in active], temperature, random_action_prob
            )
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
                    steps[index].append(TrajectoryStep(state, action, ""))
                    reasons[index] = str(error)
                    continue
                # Unexpected RDKit/model errors still surface to the caller.
                action = replace(action, product_smiles=next_state.smiles)
                steps[index].append(TrajectoryStep(state, action, next_state.smiles))
                states[index] = next_state
                if analyze_backward:
                    self.env.retro_analyzer.submit(
                        index,
                        next_state.smiles,
                        next_state.reaction_count,
                        [(action, retro_trees[index])],
                    )
            if analyze_backward:
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
                    transition.log_backward = value
                    retro_trees[index] = tree

        trajectories = []
        for index, state in enumerate(states):
            if not state.terminated and reasons[index] is None:
                reasons[index] = "reaction limit reached with an open handle"
            trajectories.append(
                Trajectory(
                    steps=steps[index],
                    final_smiles=state.smiles if reasons[index] is None else "",
                    valid=reasons[index] is None,
                    invalid_reason=reasons[index],
                )
            )
        return trajectories


def trajectory_sample(trajectory: Trajectory) -> Sample | None:
    return Sample.from_smiles(trajectory.final_smiles) if trajectory.valid else None
