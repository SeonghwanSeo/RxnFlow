"""Reference-style masked policy matrices over the synthon environment."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from rxnflow.config import Config
from rxnflow.envs.chemistry.features import molecular_properties
from rxnflow.envs.env import InvalidTransition, SynthesisEnv
from rxnflow.envs.graph import GraphBatch, molecule_to_graph_data
from rxnflow.envs.retrosynthesis import RetrosynthesisTree
from rxnflow.gflownet.categorical import ActionCategorical, ActionLogits
from rxnflow.gflownet.subsampling import BlockSubsampler
from rxnflow.gflownet.types import (
    Action,
    ActionKind,
    MoleculeState,
    Trajectory,
    Transition,
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
    """The sampled space has no budget-feasible continuation."""


class SynthesisPolicy:
    def __init__(
        self,
        env: SynthesisEnv,
        model: RxnFlowModel,
        config: Config,
        device: torch.device,
        generator: torch.Generator,
    ):
        self.env, self.model, self.config = env, model, config
        self.device, self.generator = device, generator
        self.block_subsamplers = {
            name: BlockSubsampler(len(library), config.subsampling)
            for name, library in env.blocks.items()
        }

    def candidate_batch(
        self, states: list[MoleculeState], beta: Tensor, preferences: Tensor
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
        action_rows, action_libraries, kinds, library_rows = {}, {}, {}, {}
        for row, (state, key) in enumerate(zip(states, keys, strict=True)):
            if key not in graphs:
                descriptors = molecular_properties(state.mol)
                properties[key] = torch.from_numpy(descriptors)
                graphs[key] = molecule_to_graph_data(
                    state.mol, self.env.max_atoms, state.reaction_count, descriptors
                )
            for group in self.env.available_groups(state):
                kinds[group.name] = group.kind
                action_rows.setdefault(group.name, set()).add(row)
                if group.block_type is not None:
                    action_libraries.setdefault(group.name, {}).setdefault(
                        group.block_type, []
                    ).append(row)
                    library_rows.setdefault(group.block_type, set()).add(row)
        descriptors = torch.stack([properties[key] for key in keys])
        samples, masks, features, sizes = {}, {}, [], []
        for name, rows in library_rows.items():
            sample = self.block_subsamplers[name].sample(self.generator)
            samples[name] = sample
            rows = torch.tensor(sorted(rows))
            mask = torch.zeros((len(states), len(sample.indices)), dtype=torch.bool)
            mask[rows] = self.env.block_mask(descriptors[rows], name, sample.indices)
            masks[name] = mask
            library = self.env.blocks[name]
            features.append(
                (
                    library.fingerprints[sample.indices],
                    library.properties[sample.indices],
                    torch.full(
                        (len(sample.indices),),
                        self.env.block_type_to_index[name],
                        dtype=torch.long,
                    ),
                )
            )
            sizes.append(len(sample.indices))

        # Graph/property preprocessing is shared, but identical molecules may
        # have different beta/preferences and need separate neural encodings.
        condition = self.model.encode_condition(beta, preferences)
        embeddings = self.model.encode_graphs(
            GraphBatch.from_graphs([graphs[key] for key in keys]).to(self.device),
            condition,
        )
        temperatures = self.model.temperature(condition)
        block_embeddings = {}
        if features:
            fps, props, types = zip(*features, strict=True)
            encoded = F.normalize(
                self.model.encode_block_features(
                    torch.cat(fps).to(self.device, dtype=torch.float32),
                    torch.cat(props).to(self.device),
                    torch.cat(types).to(self.device),
                ),
                dim=-1,
            )
            block_embeddings = dict(zip(samples, encoded.split(sizes), strict=True))

        action_groups = []
        for name, rows in action_rows.items():
            rows = torch.tensor(sorted(rows), device=self.device)
            query = self.model.action_query(embeddings[rows], name, temperatures[rows])
            libraries = list(action_libraries.get(name, {}))
            if libraries:
                # CGFlow: a single group matrix over concatenated libraries.
                blocks = torch.cat([block_embeddings[n] for n in libraries])
                scores = query @ blocks.T
                logits = embeddings.new_full((len(states), len(blocks)), -torch.inf)
                logits = logits.index_copy(0, rows, scores)
                allowed, weights, exploration = [], [], []
                for n in libraries:
                    eligible = torch.zeros(len(states), dtype=torch.bool)
                    eligible[action_libraries[name][n]] = True
                    allowed.append(masks[n] & eligible[:, None])
                    count = len(samples[n].indices)
                    weights.append(torch.full((count,), samples[n].log_importance))
                    # Exactly CGFlow's library-size correction, before masking.
                    exploration.append(
                        torch.full((count,), -math.log(len(libraries) * count))
                    )
                logits = logits.masked_fill(
                    ~torch.cat(allowed, 1).to(self.device), -torch.inf
                )
                weights = torch.cat(weights).to(self.device)
                exploration = torch.cat(exploration).to(self.device)
            else:
                logits = embeddings.new_full((len(states), 1), -torch.inf).index_copy(
                    0, rows, query
                )
                weights = embeddings.new_zeros(1)
                exploration = embeddings.new_zeros(1)
            action_groups.append(
                ActionLogits(
                    name,
                    kinds[name],
                    libraries,
                    [samples[n].indices for n in libraries],
                    logits,
                    weights,
                    exploration,
                )
            )
        return ActionCategorical(action_groups, embeddings, temperatures)

    def observed_logits(
        self, embeddings: Tensor, actions: list[Action], temperatures: Tensor
    ) -> Tensor:
        """Score numerator edges independently of the denominator subsample.

        This is RxnFlow's _cal_action_logits, batched by reaction/library to
        preserve its values without one model invocation per training state.
        """
        by_reaction, by_library = {}, {}
        for row, action in enumerate(actions):
            name = (
                "first_block"
                if action.kind == ActionKind.FIRST_BLOCK
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
        blocks = embeddings.new_zeros((len(actions), self.config.model.block_dim))
        if features:
            fp, prop, typ = zip(*features, strict=True)
            values = F.normalize(
                self.model.encode_block_features(
                    torch.cat(fp).to(self.device, dtype=torch.float32),
                    torch.cat(prop).to(self.device),
                    torch.cat(typ).to(self.device),
                ),
                dim=-1,
            )
            blocks = blocks.index_copy(
                0, torch.tensor(block_rows, device=self.device), values
            )
        logits = embeddings.new_zeros(len(actions))
        for name, rows in by_reaction.items():
            indices = torch.tensor(rows, device=self.device)
            query = self.model.action_query(
                embeddings[indices], name, temperatures[indices]
            )
            values = (
                query.squeeze(1)
                if actions[rows[0]].kind == ActionKind.UNI_REACTION
                else (query * blocks[indices]).sum(1)
            )
            logits = logits.index_copy(0, indices, values)
        return logits

    @torch.no_grad()
    def choose_actions(
        self,
        states: list[MoleculeState],
        temperature: float,
        random_action_prob: float,
        beta: Tensor,
        preferences: Tensor,
    ) -> list[Action | None]:
        return self.candidate_batch(states, beta, preferences).sample(
            temperature, random_action_prob, self.config.subsampling.importance_temp
        )

    def choose_action(
        self,
        state: MoleculeState,
        temperature: float,
        random_action_prob: float,
        beta: Tensor,
        preferences: Tensor,
    ) -> Action:
        action = self.choose_actions(
            [state], temperature, random_action_prob, beta, preferences
        )[0]
        if action is None:
            raise NoValidActions(
                "the sampled action space has no budget-feasible continuation"
            )
        return action

    def action_log_probabilities(
        self,
        states: list[MoleculeState],
        actions: list[Action],
        beta: Tensor,
        preferences: Tensor,
    ) -> Tensor:
        categorical = self.candidate_batch(states, beta, preferences)
        numerator = self.observed_logits(
            categorical.embeddings, actions, categorical.temperatures
        )
        return (numerator - categorical.log_partition()).clamp(max=0.0)

    def action_log_probability(
        self, state: MoleculeState, action: Action, beta: Tensor, preferences: Tensor
    ) -> Tensor:
        return self.action_log_probabilities([state], [action], beta, preferences)[0]

    def rollout(
        self,
        temperature: float = 1.0,
        random_action_prob: float = 0.0,
        analyze_backward: bool = True,
        *,
        beta: Tensor,
        preferences: Tensor,
    ) -> Trajectory:
        return self.rollouts(
            1,
            temperature,
            random_action_prob,
            analyze_backward,
            beta=beta,
            preferences=preferences,
        )[0]

    def rollouts(
        self,
        count: int,
        temperature: float = 1.0,
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
                transition.log_pb = value
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
            selected = self.choose_actions(
                [states[index] for index in active],
                temperature,
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
