# RxnFlow

RxnFlow trains a generative flow network over an Enamine-derived synthon reaction space. Molecules are assembled through dynamically selected FirstBlock, UniReaction, and BiReaction actions using a local PyTorch and RDKit runtime.

The research project is described in [Generative Flows on Synthetic Pathway for Drug Design](https://arxiv.org/abs/2410.04542).

The synthon environment owns its chemistry and catalog preparation under `envs/`; model code and GFlowNet execution live in `models/` and `gflownet/`. See [project layout](docs/project-layout.md) for the source map and dependency boundaries.

## Installation

RxnFlow is under development toward its initial `v1.0.0` release and requires Python 3.10. There is one current environment, config, and checkpoint format; regenerate older artifacts.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The implementation uses native PyTorch tensors and does not require PyTorch Geometric or compiled scatter extensions.

## Prepare the Enamine synthon environment

Supply an Enamine stock file with one `SMILES<TAB>ID` record per line. Vendor records are local inputs and are not part of this repository.

```bash
rxnflow-prepare \
  --building-blocks /path/to/enamine_stock.smi \
  --template-dir data/templates \
  --env-dir /path/to/prepared/enamine
```

Preparation applies the 35 typed synthon conversions in `synthon.yaml`. One-site products become bricks and two-site products become linkers, including pairs of the same type. Protected handles count as sites. Both conversion orders are considered for linkers. Identical canonical synthons aggregate their Enamine IDs; different representations of the same source molecule remain distinct blocks.

The prepared environment contains:

```text
prepared/enamine/
├── blocks/
│   ├── 1.smi
│   ├── 1-1.smi
│   └── ...
├── building_blocks.json
├── bb_feature.npz
├── synthon.yaml
├── reaction.yaml
└── prepare_manifest.json
```

Each `blocks/*.smi` row is `synthon SMILES<TAB>JSON array of source IDs`. `building_blocks.json` maps each ID to its standardized source BB SMILES; it is provenance, not a parallel molecular state. The NPZ feature file uses flat `<type>/smiles`, `<type>/properties`, `<type>/fingerprints`, and `<type>/heavy_atoms` arrays, plus a format marker. Feature rows include their aligned SMILES. The CLI performs conversion followed by feature generation and requires a new output directory. There are no resume/force options or `all` stage selector. In Python, `convert_stage` and `features_stage` can be imported from `rxnflow.envs.prepare` and called directly. Re-conversion invalidates prior features. Rebuild after changing source files or templates.

## Train

Set the prepared environment and output directory in a config:

```yaml
data:
  env_dir: /path/to/prepared/enamine
  max_atoms: 50

generation:
  min_reactions: 1
  max_reactions: 3

run:
  output_dir: runs/qed
  device: auto
  seed: 0
```

Then start local QED optimization:

```bash
rxnflow-train --config configs/qed.yaml
```

A trajectory starts with a one-site brick and grows one intermediate. Each BiReaction consumes one site from the intermediate and one from a catalog block. A linker leaves one site; a brick leaves none and terminates the trajectory. UniReaction acts directly on the marked site and its required neighboring substructure, producing either one site (continue) or none (terminate). Termination is immediate; terminal molecules cannot reactivate. There is no Stop or terminal restoration.

`min_reactions` and `max_reactions` count UniReaction and BiReaction, excluding FirstBlock. Before the minimum, terminating actions are masked. At the final allowed reaction, only terminating actions remain. No feasible sampled action means an invalid trajectory, not automatic capping. For example, three reactions allow brick → linker coupling → Boc deprotection → brick coupling. Distinct products from different reaction sites are separate actions identified by canonical product SMILES; symmetry-equivalent matches are merged.

Building blocks are uniformly subsampled per library with inclusion-probability correction. All distinct products of those sampled blocks are enumerated and checked before scoring. `data.max_atoms` and the optional top-level `property_penalty` bounds apply to the resulting synthon at every step, including the dummy-free terminal product. Reactant descriptors are not summed. Dummy isotope labels do not contribute fictitious mass; intermediate descriptors still describe an abstraction, not a restored real molecule.

Graph tensors reserve `max_atoms` RDKit heavy-atom slots plus one dummy slot. Molecules are never truncated. The policy scores state/reaction/block features plus outcome fingerprints and properties, allowing it to choose among positional products.

Training uses trajectory balance, replay, an EMA sampling model, and restartable checkpoints. Backward analysis preserves the generated route and adds forward-verified candidates from a depth-pruned reverse search with at most two canonical precursor sets per rule. Its depth-weighted probabilities are a bounded heuristic, not exhaustive route enumeration. TB/replay always includes the observed block in the subsampled denominator and corrects the remaining population. Invalid trajectories receive zero raw reward and the configured training reward floor.

## Custom rewards

Rewards are explicit local Python objects implementing `RewardFunction`:

```python
from rxnflow import Config, RewardFunction, RxnFlowTrainer, Sample


class CarbonReward(RewardFunction):
    def score(self, samples: list[Sample]) -> list[float]:
        return [
            sum(atom.GetAtomicNum() == 6 for atom in sample.mol.GetAtoms()) / 50
            for sample in samples
        ]


config = Config.from_file("config.yaml")
trainer = RxnFlowTrainer(config, CarbonReward())
trainer.run()
```

`reward.settings` is passed to the selected reward constructor. YAML does not import or choose a reward class.

## Sample

```bash
rxnflow-sample \
  --checkpoint runs/qed/checkpoint_latest.pt \
  --num-samples 100 \
  --output samples.json
```

Structured results contain dummy-free `smiles`, `trajectory`, `intermediates`, optional `reward`, and `metadata`. Each trajectory action includes its selected `product_smiles`. Block actions additionally record brick/linker role, catalog index, synthon SMILES, structured `block_ids`, and the corresponding source BB structures. `intermediates` contains every action product, including FirstBlock and the terminal product. There is no separate terminal `synthon_smiles`: the final internal state is already the output molecule.

## Reaction templates

`reaction.yaml` contains 38 bimolecular rules and four synthon-level unary rules. Each unary definition declares `input_type`, `output_type` (`null` for terminal), and forward/reverse SMARTS. Reverse rules enumerate candidate precursors, not experimental reverse protocols.

| UniReaction | Input representation | Output representation | Terminal |
| --- | --- | --- | --- |
| Boc deprotection | `R-N-[33*]` | `R-N-[1*]` | No |
| Methyl ester hydrolysis | `R-[34*]` | `R-[3*]` | No |
| Ethyl ester hydrolysis | `R-[35*]` | `R-[3*]` | No |
| Nitrile → tetrazole | `R-[11*]` | `R-c1nnn[nH]1` | Yes |

For types 3, 11, 34, and 35, the marker represents the entire acid, nitrile, or ester handle, respectively. For 33, nitrogen is retained and the Boc group is abstracted. No whole-molecule site recognition runs after a reaction: the remaining marked handle is carried by the product SMARTS. Additional production unary chemistry and halogen exchange remain separate curation work. See [the implementation review guide](docs/linear-synthesis.md) for the state/action contract and current type inventory.

`real.txt` and `real_raw.txt` are retained as provenance and curation references. They are not runtime inputs.

## Validation

```bash
./test.sh quick
```

The heavy suite accepts either an existing prepared environment or a local Enamine stock file:

```bash
RXNFLOW_ENV_DIR=/path/to/prepared/enamine ./test.sh heavy
RXNFLOW_ENAMINE_STOCK=/path/to/enamine_stock.smi ./test.sh heavy
```

Set `RXNFLOW_FULL_PREPARE=1` to process the complete stock file instead of the representative prefix used by the heavy test.

## License

See [LICENSE](LICENSE).
