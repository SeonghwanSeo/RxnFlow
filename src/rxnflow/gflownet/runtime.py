"""Shared policy execution over exact, site-specific synthon transitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from rxnflow.config import Config
from rxnflow.envs import SynthesisEnv
from rxnflow.envs.chemistry.features import block_feature_row
from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data
from rxnflow.envs.retrosynthesis import RetroSynthesisTree
from rxnflow.gflownet.categorical import corrected_log_probability, sample_position
from rxnflow.gflownet.subsampling import UniformActionSpace
from rxnflow.models import RxnFlowModel
from rxnflow.sample import Sample
from rxnflow.types import ActionKind, MoleculeState, RxnAction, Trajectory, TrajectoryStep


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class NoValidActions(ValueError):
    """The sampled action space has no feasible continuation."""


@dataclass
class CandidateSet:
    actions: list[RxnAction]
    logits: Tensor
    log_importance: Tensor


class PolicyRuntime:
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
        self.action_spaces = {
            name: UniformActionSpace(len(library), config.subsampling)
            for name, library in env.blocks.items()
        }
        self._outcome_features = lru_cache(maxsize=8192)(block_feature_row)

    def encode(self, state: MoleculeState) -> Tensor:
        graph = molecule_to_graph_data(
            state.smiles, self.env.max_atoms, state.reaction_count
        )
        batch = GraphBatch.from_graphs([graph]).to(self.device)
        return self.model.encode_graphs(batch)

    def candidates(
        self, state: MoleculeState, required: RxnAction | None = None
    ) -> CandidateSet:
        embedding = self.encode(state)
        actions: list[RxnAction] = []
        logits: list[Tensor] = []
        importance: list[Tensor] = []
        samples = {}
        for group in self.env.available_groups(state):
            group_actions: list[RxnAction] = []
            if group.block_type is None:
                group_actions = self.env.outcomes(state, group)
                if not group_actions:
                    continue
                scores = self.model.score_scalar(embedding, group.name).expand(
                    len(group_actions)
                )
                weights = torch.zeros(len(group_actions), device=self.device)
            else:
                name = group.block_type
                if name not in samples:
                    required_index = (
                        required.block_index
                        if required is not None and required.block_type == name
                        else None
                    )
                    samples[name] = self.action_spaces[name].sample(
                        self.generator, required_index
                    )
                sample = samples[name]
                positions = []
                for position, index in enumerate(sample.indices.tolist()):
                    outcomes = self.env.outcomes(state, group, index)
                    group_actions.extend(outcomes)
                    positions.extend([position] * len(outcomes))
                if not group_actions:
                    continue
                # All positional outcomes of a sampled block have the same
                # inclusion probability. Chemistry masks run before scoring.
                indices = sample.indices[positions]
                weights = sample.log_importance[positions].to(self.device)
                scores = self.model.score_blocks(embedding, group.name, name, indices)
            if group.kind != ActionKind.FIRST_BLOCK:
                features = [
                    self._outcome_features(action.product_smiles)
                    for action in group_actions
                ]
                scores = scores + self.model.score_outcomes(
                    embedding,
                    group.name,
                    torch.from_numpy(np.stack([row[0] for row in features])),
                    torch.from_numpy(np.stack([row[1] for row in features])),
                )
            actions.extend(group_actions)
            logits.append(scores)
            importance.append(weights)
        if not actions:
            raise NoValidActions("the sampled action space has no feasible continuation")
        return CandidateSet(actions, torch.cat(logits), torch.cat(importance))

    def choose_action(
        self, state: MoleculeState, temperature: float, random_action_prob: float
    ) -> RxnAction:
        with torch.no_grad():
            candidates = self.candidates(state)
            sampling_logits = (
                candidates.logits
                + self.config.subsampling.importance_temp * candidates.log_importance
            )
            position = sample_position(
                sampling_logits, self.generator, temperature, random_action_prob
            )
        return candidates.actions[position]

    def action_log_probability(self, state: MoleculeState, action: RxnAction) -> Tensor:
        # Keep the observed block in the sampled denominator. The other blocks
        # retain their conditional inclusion correction; no log-probability clamp
        # or retry is needed when a fresh subsample misses the observed action.
        candidates = self.candidates(state, required=action)
        selected = candidates.actions.index(action)
        return corrected_log_probability(
            candidates.logits[selected], candidates.logits, candidates.log_importance
        )

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
        retro_trees = [RetroSynthesisTree("") for _ in range(count)]

        # FirstBlock + at most max_reactions chemical transformations.
        for _ in range(self.env.max_reactions + 1):
            active = [
                i
                for i, state in enumerate(states)
                if reasons[i] is None and not state.terminated
            ]
            if not active:
                break
            for index in active:
                state = states[index]
                try:
                    action = self.choose_action(state, temperature, random_action_prob)
                except NoValidActions as error:
                    reasons[index] = str(error)
                    continue
                # Unexpected RDKit/model/programming errors are not invalid
                # chemistry. Let them surface instead of silently training on them.
                next_state = self.env.step(state, action)
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
