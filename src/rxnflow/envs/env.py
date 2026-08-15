"""eMolecules eXplore/Synple synthesis environment."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml
from rdkit import Chem

from rxnflow.chemistry import heavy_atom_count, parse_molecule
from rxnflow.data.graph import molecule_to_graph_data
from rxnflow.types import ActionKind, MoleculeState, RxnAction

from .building_block import load_block_libraries
from .workflow import Protocol, Workflow

PROTOCOL_KINDS = {
    "FirstBlock": ActionKind.FIRST_BLOCK,
    "UniRxn": ActionKind.UNI_REACTION,
    "BiRxn": ActionKind.BI_REACTION,
}


class SynthesisEnv:
    """A local, fixed-workflow synthesis environment.

    The first decision chooses a workflow. Every subsequent decision follows
    that workflow's protocol sequence, so backward probabilities are one.
    """

    def __init__(self, env_dir: str | Path, max_atoms: int = 50):
        self.env_dir = Path(env_dir)
        self.max_atoms = max_atoms
        assert max_atoms > 0
        self.blocks = load_block_libraries(self.env_dir)
        self.protocols = self._load_protocols(self.env_dir / "protocol.yaml")
        self.workflows = self._load_workflows(self.env_dir / "workflow_map.csv")
        self._validate_references()
        self.protocol_names = sorted(self.protocols)
        self.protocol_to_index = {
            name: index for index, name in enumerate(self.protocol_names)
        }
        self.block_types = sorted(self.blocks)
        self.block_type_to_index = {
            name: index for index, name in enumerate(self.block_types)
        }
        self.signature = self._build_signature()

    def _build_signature(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for workflow in self.workflows:
            digest.update(workflow.identifier.encode())
            digest.update(workflow.name.encode())
            for protocol in workflow.protocols:
                digest.update(protocol.name.encode())
                digest.update(str(int(protocol.kind)).encode())
                digest.update((protocol.block_type or "").encode())
                digest.update((protocol.forward or "").encode())
        for block_type in self.block_types:
            library = self.blocks[block_type]
            digest.update(block_type.encode())
            for smiles, identifier, tier, atom_count in zip(
                library.smiles,
                library.identifiers,
                library.tiers.tolist(),
                library.heavy_atoms.tolist(),
                strict=True,
            ):
                digest.update(f"{smiles}\t{identifier}\t{tier}\t{atom_count}\n".encode())
        return {
            "workflow_ids": [workflow.identifier for workflow in self.workflows],
            "block_counts": {name: len(library) for name, library in self.blocks.items()},
            "protocol_names": self.protocol_names,
            "content_sha256": digest.hexdigest(),
        }

    @staticmethod
    def _load_protocols(path: Path) -> dict[str, Protocol]:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError("protocol.yaml must contain a mapping")
        protocols: dict[str, Protocol] = {}
        for section, kind in PROTOCOL_KINDS.items():
            entries = data.get(section, {})
            if not isinstance(entries, dict):
                raise ValueError(f"protocol section {section} must be a mapping")
            for name, raw in entries.items():
                if name in protocols:
                    raise ValueError(f"duplicate protocol name {name}")
                if not isinstance(raw, dict):
                    raise ValueError(f"protocol {name} must be a mapping")
                unknown = set(raw) - {"block_type", "forward"}
                if unknown:
                    raise ValueError(
                        f"unknown fields in protocol {name}: {sorted(unknown)}"
                    )
                protocols[name] = Protocol(name=name, kind=kind, **raw)
        if not protocols:
            raise ValueError("protocol.yaml does not define any protocols")
        return protocols

    def _load_workflows(self, path: Path) -> list[Workflow]:
        if not path.is_file():
            raise FileNotFoundError(path)
        workflows: list[Workflow] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("workflow_map.csv has no header")
            protocol_columns = [
                name for name in reader.fieldnames if name.lower().startswith("protocol ")
            ]
            required_columns = {"workflow id", "workflow name", *protocol_columns}
            if not protocol_columns or set(reader.fieldnames) != required_columns:
                raise ValueError(
                    "workflow_map.csv requires only workflow id, workflow name, and protocol N columns"
                )
            for line_number, row in enumerate(reader, start=2):
                identifier = (row.get("workflow id") or "").strip()
                name = (row.get("workflow name") or "").strip()
                names = [(row.get(column) or "").strip() for column in protocol_columns]
                names = [value for value in names if value and value.lower() != "nan"]
                if not identifier or not name or not names:
                    raise ValueError(
                        f"workflow_map.csv:{line_number}: incomplete workflow"
                    )
                try:
                    protocols = tuple(self.protocols[value] for value in names)
                except KeyError as error:
                    raise ValueError(
                        f"workflow_map.csv:{line_number}: unknown protocol {error.args[0]}"
                    ) from error
                if protocols[0].kind != ActionKind.FIRST_BLOCK:
                    raise ValueError(f"workflow {identifier} must begin with FirstBlock")
                workflows.append(
                    Workflow(identifier=identifier, name=name, protocols=protocols)
                )
        if not workflows:
            raise ValueError("workflow_map.csv contains no workflows")
        identifiers = [workflow.identifier for workflow in workflows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("workflow identifiers must be unique")
        return workflows

    def _validate_references(self) -> None:
        referenced_types = {
            protocol.block_type
            for workflow in self.workflows
            for protocol in workflow.protocols
            if protocol.block_type is not None
        }
        missing = referenced_types - set(self.blocks)
        extra = set(self.blocks) - referenced_types
        if missing:
            raise ValueError(
                f"missing SMILES/features for block types: {sorted(missing)}"
            )
        if extra:
            raise ValueError(f"unreferenced block types in environment: {sorted(extra)}")

    @staticmethod
    def initial_state() -> MoleculeState:
        return MoleculeState()

    def is_terminal(self, state: MoleculeState) -> bool:
        return state.workflow_index >= 0 and state.protocol_order >= len(
            self.workflows[state.workflow_index].protocols
        )

    def next_action_kind(self, state: MoleculeState) -> ActionKind:
        if state.workflow_index < 0:
            return ActionKind.SET_WORKFLOW
        if self.is_terminal(state):
            raise ValueError("terminal states do not have actions")
        return self.workflows[state.workflow_index].protocols[state.protocol_order].kind

    def current_protocol(self, state: MoleculeState) -> Protocol:
        if state.workflow_index < 0 or self.is_terminal(state):
            raise ValueError("state has no current protocol")
        return self.workflows[state.workflow_index].protocols[state.protocol_order]

    def graph_data(self, state: MoleculeState):
        kind = self.next_action_kind(state)
        return molecule_to_graph_data(
            state.smiles,
            self.max_atoms,
            state.workflow_index,
            state.protocol_order,
            int(kind),
        )

    def step(self, state: MoleculeState, action: RxnAction) -> MoleculeState:
        expected = self.next_action_kind(state)
        assert action.kind == expected
        if expected == ActionKind.SET_WORKFLOW:
            assert 0 <= action.workflow_index < len(self.workflows)
            return MoleculeState("", action.workflow_index, 0)
        assert action.workflow_index == state.workflow_index
        assert action.protocol_order == state.protocol_order

        protocol = self.current_protocol(state)
        if expected in (ActionKind.FIRST_BLOCK, ActionKind.BI_REACTION):
            assert action.block_type == protocol.block_type
            assert action.block_index is not None
            library = self.blocks[protocol.block_type]
            block_smiles = library.smiles[action.block_index]
        else:
            block_smiles = ""

        if expected == ActionKind.FIRST_BLOCK:
            product_smiles = block_smiles
        else:
            current = parse_molecule(state.smiles)
            if current is None or protocol.reaction is None:
                raise ValueError("reaction state is missing a valid molecule")
            if expected == ActionKind.UNI_REACTION:
                product_smiles = protocol.reaction.run(current)
            else:
                block = parse_molecule(block_smiles)
                if block is None:
                    raise ValueError("invalid building block")
                product_smiles = protocol.reaction.run(current, block)

        product = Chem.MolFromSmiles(product_smiles)
        atom_count = heavy_atom_count(product)
        if atom_count > self.max_atoms:
            raise ValueError(
                f"reaction product has {atom_count} heavy atoms; limit is {self.max_atoms}"
            )
        canonical = Chem.MolToSmiles(product) if product is not None else ""
        if not canonical:
            raise ValueError("reaction produced an invalid molecule")
        return MoleculeState(canonical, state.workflow_index, state.protocol_order + 1)

    def workflow_label(self, index: int) -> str:
        workflow = self.workflows[index]
        return f"{workflow.identifier}: {workflow.name}"

    def action_to_dict(self, action: RxnAction) -> dict[str, object]:
        workflow = self.workflows[action.workflow_index]
        result: dict[str, object] = {
            "type": action.kind.name,
            "workflow_id": workflow.identifier,
            "workflow_name": workflow.name,
            "protocol_order": action.protocol_order,
        }
        if action.protocol_order >= 0:
            protocol = workflow.protocols[action.protocol_order]
            result["protocol"] = protocol.name
        if action.block_index is not None and action.block_type is not None:
            library = self.blocks[action.block_type]
            result.update(
                {
                    "block_type": action.block_type,
                    "block_index": action.block_index,
                    "block_smiles": library.smiles[action.block_index],
                    "block_identifier": library.identifiers[action.block_index],
                    "tier": int(library.tiers[action.block_index]),
                }
            )
        return result
