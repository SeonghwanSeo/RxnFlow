# Preparing a synthesis environment

## Prepare your building block catalog

Prepare a building-block library, for example from [eMolecules](https://www.emolecules.com/data-downloads) or [Enamine](https://enamine.net/building-blocks/building-blocks-catalog). Convert your library to a headerless file with one `SMILES<TAB>ID` record per line:

```text
CC#N	BB001
CC(=O)O	BB002
CCOC(=O)CC(=O)Cl	BB003
```

IDs identify the original building blocks in generated routes. A building block can provide several synthons depending on which reaction sites are selected: bricks have one site, while linkers have two and can attach in either direction.

## Build the environment

```bash
python scripts/prepare.py \
  --block-smi /path/to/building_blocks.smi \
  --env-dir /path/to/prepared/environment \
  --num-workers 16
```

- `--config`: preparation settings; defaults to the supplied `data/templates/explore/config.yaml`.
- `--max-atoms`: retain synthons with at most this many heavy atoms, excluding dummy handles. Default: `30`.
- `--min-library-size`: retain libraries with at least this many distinct oriented synthons. Default: `10`; use `1` for small trial catalogs.
- `--druglikeness-threshold`: retain building blocks with a [DeepDL druglikeness](https://pypi.org/project/druglikeness/) score at or above this value. Screening is disabled by default. Try `60` as a starting threshold; use `--druglikeness-device cuda` for GPU scoring (default: `cpu`).

Use a new output directory. Once preparation finishes, set the training configuration to:

```yaml
env_dir: /path/to/prepared/environment
```

## Choose a template set

- `data/templates/explore/config.yaml` is the default: 26 bimolecular and 4 unimolecular templates.
- `data/templates/cgflow/config.yaml` supplies the CGFlow set: 38 bimolecular and 4 unimolecular templates.

## Customize the templates

To change the supported reactions or synthon definitions, copy `data/templates/explore/` to a new directory and edit its YAML files. The `config.yaml` file references the reaction and synthon definitions and an optional exclusion list:

```yaml
# Paths are relative to this configuration's directory.
reaction: reaction.yaml
synthon: synthon.yaml
exclude_smarts: exclude_smarts.yaml
```

- To add or remove a supported reaction, edit `reaction.yaml`. Define forward and reverse SMARTS; input and output types are inferred from labels such as `[2*]`. Set `ordered: true` for a bimolecular reaction to allow both state/incoming orientations.
- To change which building-block functional groups become reaction sites, edit `synthon.yaml`. Each entry gives a numeric `type`, an `original` SMARTS pattern and a `convert` template. Use those same types in the reaction definitions.
- To change which unused groups may remain in a synthon, edit `exclude_smarts.yaml`. Remove the `exclude_smarts` setting entirely to disable this filter.

Pass your configuration to preparation with `--config /path/to/custom/config.yaml`. Rebuild the environment after changing the catalog, templates or filter settings.

### Exclude functional groups

Use `exclude_smarts.yaml` to specify protecting groups and functional groups that are allowed during synthesis but should not remain in final products. Write one quoted SMARTS pattern per YAML list entry:

```yaml
# Acyl halides
- '[CX3](=O)[F,Cl,Br,I]'

# Sulfonyl chloride/bromide/iodide
- '[SX4](=O)(=O)[Cl,Br,I]'
```

See the supplied [exclusion list](../data/templates/explore/exclude_smarts.yaml) for the default patterns.
