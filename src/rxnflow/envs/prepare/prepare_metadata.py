"""Validate and snapshot synthesis definitions before preparing building blocks."""

import shutil
from pathlib import Path

import yaml
from rdkit import Chem

from rxnflow.core.reaction import load_reactions
from rxnflow.core.synthon import (
    SynthonConversion,
    load_synthon_templates,
)


def prepare_metadata(config_path: str | Path, env_dir: str | Path) -> None:
    """Validate template contracts before processing any building blocks."""
    config_path, env_path = Path(config_path), Path(env_dir)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or not {"synthon", "reaction"} <= config.keys():
        raise ValueError("Template config requires synthon and reaction paths")
    paths = {name: config_path.parent / config[name] for name in ("synthon", "reaction")}
    if config.get("exclude_smarts"):
        paths["exclude_smarts"] = config_path.parent / config["exclude_smarts"]

    # Each conversion produces one typed site. Other existing sites are checked
    # on actual products during linker conversion, where their context is known.
    templates = load_synthon_templates(paths["synthon"])
    for synthon_type, patterns in templates.items():
        conversion = SynthonConversion(synthon_type, *patterns)
        reaction = conversion.reaction
        if (
            reaction.GetNumReactantTemplates() != 1
            or reaction.GetNumProductTemplates() != 1
        ):
            raise ValueError(
                f"Synthon {synthon_type}: expected one reactant and one product"
            )
        _, errors = reaction.Validate(silent=True)
        if errors:
            raise ValueError(f"Synthon {synthon_type}: invalid reaction SMARTS")
        # A mapped, unlabelled wildcard (e.g. [*:2]) preserves a reactant atom;
        # it does not introduce a dummy. Count new dummies and explicit labels.
        sites = [
            atom.GetIsotope()
            for atom in conversion.product_pattern.GetAtoms()
            if atom.GetAtomicNum() == 0
            and (atom.GetIsotope() > 0 or atom.GetAtomMapNum() == 0)
        ]
        if sites != [synthon_type]:
            raise ValueError(
                f"Synthon {synthon_type}: expected exactly one declared dummy site"
            )

    # Reaction constructors check SMARTS shape and site labels. Here we also
    # ensure every referenced type is defined by the synthon templates.
    uni, bi = load_reactions(paths["reaction"])
    for reaction in uni.values():
        types = {reaction.input_type}
        if reaction.output_type is not None:
            types.add(reaction.output_type)
        if not types <= templates.keys():
            raise ValueError(f"Unknown synthon type in {reaction.name}")
    for reaction in bi.values():
        if not set(reaction.synthon_types) <= templates.keys():
            raise ValueError(f"Unknown synthon type in {reaction.name}")
    names = ["first_synthon", *uni, *bi]
    if len(names) != len(set(names)):
        raise ValueError("Reaction action names must be unique")
    for reaction in (*uni.values(), *bi.values()):
        for compiled in (reaction.forward_reaction, reaction.reverse_reaction):
            _, errors = compiled.Validate(silent=True)
            if errors:
                raise ValueError(f"Invalid reaction SMARTS in {reaction.name}")

    if "exclude_smarts" in paths:
        patterns = yaml.safe_load(paths["exclude_smarts"].read_text())
        if not isinstance(patterns, list):
            raise ValueError("The exclude_smarts file must contain a YAML list")
        for pattern in patterns:
            if not isinstance(pattern, str) or Chem.MolFromSmarts(pattern) is None:
                raise ValueError(f"Invalid exclusion SMARTS: {pattern}")

    # Snapshot templates only after all definitions pass.
    env_path.mkdir(parents=True, exist_ok=True)
    if "exclude_smarts" not in paths:
        (env_path / "exclude_smarts.yaml").unlink(missing_ok=True)
    for name, source in paths.items():
        shutil.copyfile(source, env_path / f"{name}.yaml")
    (env_path / "config.yaml").write_text(
        yaml.safe_dump({name: f"{name}.yaml" for name in paths}, sort_keys=False),
    )
