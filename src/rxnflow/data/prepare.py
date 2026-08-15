"""Resumable eMolecules eXplore/Synple data preparation stages."""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import torch
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem.SaltRemover import SaltRemover

from rxnflow.chemistry import block_feature_row
from rxnflow.envs.building_block import read_smiles_file

ALLOWED_ATOMIC_NUMBERS = {5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 85}
MANIFEST_NAME = "prepare_manifest.json"


@dataclass(frozen=True)
class ConversionSpec:
    name: str
    smi_file: str
    original: str
    convert: str


class Conversion:
    def __init__(self, spec: ConversionSpec):
        self.spec = spec
        self.template = f"{spec.original}>>{spec.convert}"
        reaction = ReactionFromSmarts(self.template)
        if reaction is None:
            raise ValueError(f"invalid synthon conversion {spec.name}: {self.template}")
        reaction.Initialize()
        self.reaction = reaction

    def run(self, smiles: str) -> list[str]:
        mol = Chem.MolFromSmiles(
            smiles,
            replacements={
                "[CH]": "C",
                "[C]": "C",
                "[CH2]": "C",
                "[c]": "c",
                "[N]": "[1N]",
            },
        )
        if mol is None:
            return []
        products: set[str] = set()
        for product_tuple in self.reaction.RunReactants((mol,), 10):
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    result = Chem.MolToSmiles(product)
                    if "[1N]" in result:
                        restored = Chem.MolFromSmiles(result, replacements={"[1N]": "N"})
                        if restored is None:
                            continue
                        result = Chem.MolToSmiles(restored)
                except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
                    continue
                products.add(result)
        return sorted(products)


def _load_manifest(env_dir: Path) -> dict[str, object]:
    path = env_dir / MANIFEST_NAME
    if not path.exists():
        return {"format": "rxnflow-prepare-v1", "stages": {}}
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "rxnflow-prepare-v1" or not isinstance(
        manifest.get("stages"), dict
    ):
        raise ValueError(f"invalid preparation manifest: {path}")
    return manifest


def _complete_stage(env_dir: Path, stage: str, details: dict[str, object]) -> None:
    manifest = _load_manifest(env_dir)
    stages = cast(dict[str, object], manifest["stages"])
    stages[stage] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    path = env_dir / MANIFEST_NAME
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _stage_done(env_dir: Path, stage: str) -> bool:
    stages = cast(dict[str, object], _load_manifest(env_dir)["stages"])
    return stage in stages


def _load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _merge_templates(template_dirs: Iterable[Path], env_dir: Path) -> list[Conversion]:
    directories = [Path(path) for path in template_dirs]
    if not directories:
        raise ValueError("at least one template directory is required")

    protocol: dict[str, dict[str, object]] = {"FirstBlock": {}, "UniRxn": {}, "BiRxn": {}}
    specs: list[ConversionSpec] = []
    workflow_header: list[str] | None = None
    workflow_rows: list[dict[str, str]] = []
    workflow_ids: set[str] = set()
    for directory in directories:
        for required in ("protocol.yaml", "workflow_map.csv", "synthon.yaml"):
            if not (directory / required).is_file():
                raise FileNotFoundError(directory / required)
        raw_protocol = _load_yaml(directory / "protocol.yaml")
        if not isinstance(raw_protocol, dict):
            raise ValueError(f"{directory / 'protocol.yaml'} must be a mapping")
        for section in protocol:
            entries = raw_protocol.get(section, {})
            if not isinstance(entries, dict):
                raise ValueError(f"{directory}: {section} must be a mapping")
            duplicate = set(protocol[section]) & set(entries)
            if duplicate:
                raise ValueError(f"duplicate protocol names: {sorted(duplicate)}")
            protocol[section].update(entries)

        raw_specs = _load_yaml(directory / "synthon.yaml")
        if not isinstance(raw_specs, list):
            raise ValueError(f"{directory / 'synthon.yaml'} must be a list")
        for raw in raw_specs:
            if not isinstance(raw, dict):
                raise ValueError("each synthon conversion must be a mapping")
            specs.append(ConversionSpec(**raw))

        with (directory / "workflow_map.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"empty workflow map: {directory}")
            if workflow_header is None:
                workflow_header = reader.fieldnames
            elif reader.fieldnames != workflow_header:
                raise ValueError("template workflow maps must use identical columns")
            id_column = reader.fieldnames[0]
            for row in reader:
                identifier = row[id_column]
                if identifier in workflow_ids:
                    raise ValueError(f"duplicate workflow identifier {identifier}")
                workflow_ids.add(identifier)
                workflow_rows.append(row)

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("synthon conversion names must be unique")
    if workflow_header is None:
        raise ValueError("no workflows found")

    with (env_dir / "protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)
    with (env_dir / "workflow_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=workflow_header)
        writer.writeheader()
        writer.writerows(workflow_rows)
    return [Conversion(spec) for spec in specs]


def _parse_tier(value: str) -> int:
    match = re.fullmatch(r"Tier\s+(\d+)", value.strip(), re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid eMolecules price tier: {value!r}")
    tier = int(match.group(1))
    if tier <= 0:
        raise ValueError(f"tier must be positive: {value!r}")
    return tier


def _clean_smiles(smiles: str, salt_remover: SaltRemover) -> str | None:
    if "[2H]" in smiles or "[13C" in smiles or "[13c" in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    stripped = salt_remover.StripMol(mol)
    if stripped is None or not stripped.GetNumAtoms():
        return None
    if {atom.GetAtomicNum() for atom in stripped.GetAtoms()} - ALLOWED_ATOMIC_NUMBERS:
        return None
    canonical = Chem.MolToSmiles(stripped)
    return None if "." in canonical else canonical


def _read_raw_records(path: Path) -> list[tuple[str, str, int]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"raw eMolecules file has no header: {path}")
        columns = {name.lower(): name for name in reader.fieldnames}
        required = {"smiles", "price tier", "emolecules id"}
        if not required <= columns.keys():
            raise ValueError(f"{path} requires SMILES, price tier, and eMolecules ID")
        smiles_column = columns["smiles"]
        tier_column = columns["price tier"]
        identifier_column = columns["emolecules id"]
        records: list[tuple[str, str, int]] = []
        for line_number, row in enumerate(reader, start=2):
            if (
                not row.get(smiles_column)
                or not row.get(identifier_column)
                or not row.get(tier_column)
            ):
                raise ValueError(
                    f"{path}:{line_number}: missing SMILES, identifier, or tier"
                )
            records.append(
                (
                    row[smiles_column].strip(),
                    row[identifier_column].strip(),
                    _parse_tier(row[tier_column]),
                )
            )
    return records


def convert_stage(
    raw_data_dir: str | Path,
    env_dir: str | Path,
    template_dirs: Iterable[str | Path],
    force: bool = False,
) -> None:
    env_path = Path(env_dir)
    raw_path = Path(raw_data_dir)
    if _stage_done(env_path, "convert") and not force:
        return
    if not raw_path.is_dir():
        raise FileNotFoundError(raw_path)
    env_path.mkdir(parents=True, exist_ok=True)
    smiles_dir = env_path / "smiles"
    smiles_dir.mkdir(exist_ok=True)
    conversions = _merge_templates([Path(path) for path in template_dirs], env_path)
    salt_remover = SaltRemover()
    counts: dict[str, int] = {}
    for conversion in conversions:
        source_path = raw_path / conversion.spec.smi_file
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        synthons: OrderedDict[str, list[str]] = OrderedDict()
        tiers: dict[str, int] = {}
        for raw_smiles, identifier, tier in _read_raw_records(source_path):
            cleaned = _clean_smiles(raw_smiles, salt_remover)
            if cleaned is None:
                continue
            products = conversion.run(cleaned)
            if not products:
                continue
            for product in products:
                synthons.setdefault(product, []).append(identifier)
                tiers[product] = min(tier, tiers.get(product, tier))
        output = smiles_dir / f"{conversion.spec.name}.smi"
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for synthon, identifiers in synthons.items():
                normalized = ";".join(dict.fromkeys(identifiers))
                handle.write(f"{synthon}\tTier {tiers[synthon]}\t{normalized}\n")
        temporary.replace(output)
        counts[conversion.spec.name] = len(synthons)
    if not counts or any(count == 0 for count in counts.values()):
        empty = [name for name, count in counts.items() if count == 0]
        raise ValueError(f"conversion produced empty block types: {empty}")
    _complete_stage(env_path, "convert", {"block_counts": counts})


def features_stage(env_dir: str | Path, force: bool = False) -> None:
    env_path = Path(env_dir)
    if _stage_done(env_path, "features") and not force:
        return
    smiles_dir = env_path / "smiles"
    if not smiles_dir.is_dir():
        raise FileNotFoundError(smiles_dir)
    blocks: dict[str, dict[str, torch.Tensor]] = {}
    for path in sorted(smiles_dir.glob("*.smi")):
        smiles, tiers, _ = read_smiles_file(path)
        if not smiles:
            raise ValueError(f"empty building-block file: {path}")
        properties = []
        fingerprints = []
        heavy_atoms = []
        for value in smiles:
            prop, fingerprint, atom_count = block_feature_row(value)
            properties.append(prop)
            fingerprints.append(fingerprint)
            heavy_atoms.append(atom_count)
        blocks[path.stem] = {
            "properties": torch.stack(properties),
            "fingerprints": torch.stack(fingerprints),
            "heavy_atoms": torch.tensor(heavy_atoms, dtype=torch.long),
            "tiers": torch.tensor(tiers, dtype=torch.long),
        }
    if not blocks:
        raise ValueError(f"no .smi files in {smiles_dir}")
    output = env_path / "bb_feature.pt"
    temporary = output.with_suffix(".tmp")
    torch.save({"format": "rxnflow-bb-feature-v1", "blocks": blocks}, temporary)
    temporary.replace(output)
    _complete_stage(env_path, "features", {"block_types": sorted(blocks)})


def _raw_identifier_weights(raw_data_dir: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in sorted(
        value
        for value in raw_data_dir.iterdir()
        if value.suffix.lower() in {".csv", ".smi"}
    ):
        for smiles, identifier, _ in _read_raw_records(path):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            weight = float(Descriptors.ExactMolWt(mol))
            # A vendor identifier can occur in several reaction-specific files
            # with salt/stereo variants. Use a deterministic minimum for the
            # identifier-ordering hint; identity and feature alignment remain
            # defined by the converted synthon row.
            result[identifier] = min(weight, result.get(identifier, weight))
    return result


def reorder_stage(
    raw_data_dir: str | Path,
    env_dir: str | Path,
    force: bool = False,
) -> None:
    env_path = Path(env_dir)
    if _stage_done(env_path, "reorder") and not force:
        return
    raw_weights = _raw_identifier_weights(Path(raw_data_dir))
    changed = 0
    for path in sorted((env_path / "smiles").glob("*.smi")):
        rows: list[tuple[str, str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 3:
                    raise ValueError(f"{path}:{line_number}: expected three fields")
                identifiers = fields[2].split(";")
                try:
                    ranked = sorted(
                        identifiers,
                        key=lambda value: raw_weights[value],
                    )
                except KeyError as error:
                    raise ValueError(
                        f"identifier {error.args[0]} is absent from raw records"
                    ) from error
                normalized = ";".join(ranked)
                changed += normalized != fields[2]
                rows.append((fields[0], fields[1], normalized))
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write("\t".join(row) + "\n")
        temporary.replace(path)
    _complete_stage(env_path, "reorder", {"changed_rows": changed})


def prepare_all(
    raw_data_dir: str | Path,
    env_dir: str | Path,
    template_dirs: Iterable[str | Path],
    force: bool = False,
) -> None:
    convert_stage(
        raw_data_dir,
        env_dir,
        template_dirs,
        force=force,
    )
    features_stage(env_dir, force=force)
    reorder_stage(raw_data_dir, env_dir, force=force)
