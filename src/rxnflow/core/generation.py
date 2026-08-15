"""Shared local policy execution and trajectory log-probabilities."""

from __future__ import annotations

import torch
from torch import Tensor

from rxnflow.chemistry import (
    PROPERTY_NAMES,
    heavy_atom_count,
    molecular_properties,
    parse_molecule,
)
from rxnflow.config import Config
from rxnflow.data.graph import GraphBatch
from rxnflow.envs import SynthesisEnv
from rxnflow.models import RxnFlowModel
from rxnflow.policy import (
    TieredActionSpace,
    block_penalty,
    corrected_log_probability,
    sample_position,
)
from rxnflow.sample import Sample
from rxnflow.types import ActionKind, MoleculeState, RxnAction, Trajectory, TrajectoryStep


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
            block_type: TieredActionSpace(library.tiers, config.subsampling)
            for block_type, library in env.blocks.items()
        }

    def encode(self, state: MoleculeState) -> Tensor:
        graph = self.env.graph_data(state)
        batch = GraphBatch.from_graphs([graph]).to(self.device)
        return self.model.encode_graphs(batch)

    def _sampled_blocks(self, state: MoleculeState) -> tuple[str, Tensor, Tensor]:
        protocol = self.env.current_protocol(state)
        assert protocol.block_type is not None
        library = self.env.blocks[protocol.block_type]
        sample = self.action_spaces[protocol.block_type].sample(self.generator)
        current_mol = parse_molecule(state.smiles)
        current_atoms = (
            heavy_atom_count(current_mol)
            if protocol.kind == ActionKind.BI_REACTION
            else 0
        )
        current_mol_features = molecular_properties(current_mol)
        mol_feature_limits = {
            PROPERTY_NAMES.index(name): value
            for name, value in self.config.property_penalty.items()
        }
        penalty = block_penalty(
            library,
            sample.indices,
            current_atoms,
            self.env.max_atoms,
            current_mol_features=current_mol_features,
            mol_feature_limits=mol_feature_limits,
        )
        feasible = torch.isfinite(penalty)
        indices = sample.indices[feasible]
        importance = sample.log_importance[feasible]
        if indices.numel() == 0:
            raise ValueError(f"no valid sampled building blocks for {protocol.name}")
        return protocol.block_type, indices, importance

    def choose_action(
        self,
        state: MoleculeState,
        temperature: float,
        random_action_prob: float,
    ) -> RxnAction:
        kind = self.env.next_action_kind(state)
        with torch.no_grad():
            embedding = self.encode(state)
            if kind == ActionKind.SET_WORKFLOW:
                logits = self.model.score_workflows(embedding)[0]
                workflow = sample_position(
                    logits, self.generator, temperature, random_action_prob
                )
                return RxnAction(kind=kind, workflow_index=workflow)
            protocol = self.env.current_protocol(state)
            if kind == ActionKind.UNI_REACTION:
                return RxnAction(kind, state.workflow_index, state.protocol_order)
            block_type, indices, importance = self._sampled_blocks(state)
            logits = self.model.score_blocks(
                embedding, protocol.name, block_type, indices
            )
            position = sample_position(
                logits
                + self.config.subsampling.importance_temp * importance.to(logits.device),
                self.generator,
                temperature,
                random_action_prob,
            )
            return RxnAction(
                kind=kind,
                workflow_index=state.workflow_index,
                protocol_order=state.protocol_order,
                block_type=block_type,
                block_index=int(indices[position]),
            )

    def action_log_probability(self, state: MoleculeState, action: RxnAction) -> Tensor:
        kind = self.env.next_action_kind(state)
        assert kind == action.kind
        embedding = self.encode(state)
        if kind == ActionKind.SET_WORKFLOW:
            logits = self.model.score_workflows(embedding)[0]
            return torch.log_softmax(logits, dim=0)[action.workflow_index]
        if kind == ActionKind.UNI_REACTION:
            return embedding.sum() * 0.0
        protocol = self.env.current_protocol(state)
        assert action.block_type is not None and action.block_index is not None
        block_type, indices, importance = self._sampled_blocks(state)
        assert action.block_type == block_type
        sampled_logits = self.model.score_blocks(
            embedding, protocol.name, block_type, indices
        )
        selected_logit = self.model.score_one_block(
            embedding, protocol.name, block_type, action.block_index
        )
        return corrected_log_probability(selected_logit, sampled_logits, importance)

    def rollout(
        self, temperature: float = 1.0, random_action_prob: float = 0.0
    ) -> Trajectory:
        state = self.env.initial_state()
        steps: list[TrajectoryStep] = []
        valid = True
        maximum_steps = 1 + max(
            len(workflow.protocols) for workflow in self.env.workflows
        )
        for _ in range(maximum_steps):
            if self.env.is_terminal(state):
                break
            try:
                action = self.choose_action(state, temperature, random_action_prob)
                next_state = self.env.step(state, action)
            except (IndexError, RuntimeError, ValueError):
                valid = False
                break
            steps.append(
                TrajectoryStep(
                    state=state, action=action, product_smiles=next_state.smiles
                )
            )
            state = next_state
        if not self.env.is_terminal(state):
            valid = False
        final_smiles = state.smiles if valid else ""
        return Trajectory(steps=steps, final_smiles=final_smiles, valid=valid)


def trajectory_sample(trajectory: Trajectory) -> Sample | None:
    if not trajectory.valid:
        return None
    return Sample.from_smiles(trajectory.final_smiles)
