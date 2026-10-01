# RxnFlow

RxnFlow trains a generative flow network over an Enamine-derived synthon reaction space. Molecules are assembled through dynamically selected FirstBlock, UniReaction, and BiReaction actions using a local PyTorch and RDKit runtime.

The research project is described in [Generative Flows on Synthetic Pathway for Drug Design](https://arxiv.org/abs/2410.04542).

The synthon environment owns its chemistry and catalog preparation under `envs/`; model code and GFlowNet execution live in `models/` and `gflownet/`. See [project layout](codex/docs/project-layout.md) for the source map and dependency boundaries.

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
  --env-dir /path/to/prepared/enamine \
  --num-workers 4
```

Preparation applies the 35 typed synthon conversions in `synthon.yaml`. One-site products become bricks and two-site products become linkers, including pairs of the same type. Protected handles count as sites. Both conversion orders are considered for linkers. Each linker is stored in both attachment orientations. The incoming site is marked `[*]` (isotope 0); its chemical type is the first component of the ordered library key `A-B`, and the other site retains isotope B. Bricks use the same marker with a single-type library key; FirstBlock restores that type in the state. Symmetry-equivalent orientations are deduplicated. Identical canonical synthons aggregate their Enamine IDs; different representations of the same source molecule remain distinct blocks.

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

Each `blocks/*.smi` row is `synthon SMILES<TAB>JSON array of source IDs`. `building_blocks.json` maps each ID to its standardized source BB SMILES; it is provenance, not a parallel molecular state. The NPZ feature file uses flat `<type>/smiles`, `<type>/properties`, `<type>/fingerprints`, and `<type>/heavy_atoms` arrays, plus a format marker. Feature rows include their aligned SMILES. Fingerprints concatenate 512 Morgan counts (radius 2, dummy-isotope invariants, saturated at 255) and 166 MACCS bits as uint8. Libraries retain uint8; selected model inputs convert to float32. Properties remain float32. The CLI performs conversion followed by feature generation and requires a new output directory. There are no resume/force options or `all` stage selector. In Python, `convert_stage` and `features_stage` can be imported from `rxnflow.envs.prepare` and called directly. Re-conversion invalidates prior features. Rebuild after changing source files or templates. `--num-workers N` parallelizes synthon conversion and fingerprint/property calculation with N processes (default 1, serial). Both Python stage functions accept `num_workers=N`. Source cleaning and final file writing remain serial; parallel execution preserves library row order and feature alignment.

For the local random 10,000-record development environment, see [subset preparation and evaluation](codex/docs/development-subset.md).

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

`min_reactions` and `max_reactions` count UniReaction and BiReaction, excluding FirstBlock. Before the minimum, terminating actions are masked. At the final allowed reaction, only terminating actions remain. No feasible sampled action means an invalid trajectory, not automatic capping. For example, three reactions allow brick → linker coupling → Boc deprotection → brick coupling. The linker attachment site is fixed by its catalog row, so selecting a block also selects its orientation. Actions record canonical product SMILES. Each reaction/block choice must yield a unique connected product; symmetry-equivalent matches merge, and ambiguous templates raise an error.

Following HSX, building-block masks compare current state properties plus precomputed block properties against the configured upper bounds. Following `explore_250509`, select compatible reaction/library types, uniformly subsample each library, then apply the state + block budget mask to those sampled rows. Inclusion probabilities use the full library size and the original draw count, not the number surviving the mask. Rejected rows are not refilled; a sampled action space with no feasible continuation fails the trajectory. `property_penalty` therefore constrains an additive estimate, not exact product descriptors. Comparisons use raw units: nonzero bounds allow 0.1% relative tolerance, while zero bounds and graph capacity remain strict. Enamine dummy labels have zero mass; HSX's Synple/eXplore At-isotope subtraction and `+29.0` linker MW correction do not apply.

Graph tensors have a fixed capacity of `max_atoms + 1` (heavy atoms plus a dummy slot). Rollout batches all active states at each synthesis step; trajectory-balance training batches all observed transitions. Candidate counts remain variable: sampled rows are encoded once per library, then scored against compatible state/reaction queries by matrix multiplication. Budget masks are broadcast over each library’s applicable states; only allowed entries enter the categorical distribution. Each policy call samples each needed library once and shares those rows across all states, then applies state-specific budget masks. Repeated states also reuse CPU graph construction and budget masks. In evaluation mode they share graph encoding within the call; training dropout remains per graph row. Candidate metadata consists of reaction/library groups and row-index tensors. Only a selected row becomes an action object; TB/replay directly locates its observed row without materializing or searching a list of candidate actions. Embeddings are never cached across optimizer updates. Sampling draws only the needed library indices without shuffling a full library. Checkpoints retain the RNG state needed for reproducible continuation under this execution path.

The policy scores state/reaction embeddings and prepared block features without executing candidate reactions. Only the chosen action executes RDKit chemistry, checking the resulting site signature and actual `data.max_atoms` capacity. An invalid selected transition fails the trajectory without retry or unmasking. Unary actions use typed-handle and trajectory-length eligibility, without an additive block budget or exact product-property filter. The state holds the resulting RDKit `Mol`, shared with graph encoding; canonical SMILES identify and serialize states. Graph tensors reserve `max_atoms` RDKit heavy-atom slots plus one dummy slot, and molecules are never truncated.

The block encoder projects fingerprints and properties separately, then combines them with the block type. Additive reaction-conditioned heads use action-normalized dot scores and learned temperatures, while unary actions use a scalar head on the same scale. The graph readout concatenates the molecular mean and virtual node.

Training uses MSE trajectory balance, uniform FIFO replay, an EMA sampling model, and restartable checkpoints. Failed selected reactions remain in the trajectory so their forward probability receives the invalid-reward signal. `training.log_z_learning_rate` controls logZ separately, and `training.lr_decay_steps` is the learning-rate half-life. Scheduler and RNG states are restored with the optimizer. Backward analysis preserves the generated route and adds forward-verified candidates from a depth-pruned reverse search with at most two canonical precursor sets per rule. Its depth-weighted probabilities are a bounded heuristic, not exhaustive route enumeration. TB/replay includes the union of observed rows per library in the shared draw (inclusion probability 1), then uniformly samples the remaining population with conditional inclusion correction. Invalid trajectories receive zero raw reward and the configured training reward floor.

## Custom rewards

Rewards are explicit local Python objects implementing `RewardFunction`:

```python
from rxnflow.config import Config
from rxnflow.gflownet.types import Sample
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


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

For types 3, 11, 34, and 35, the marker represents the entire acid, nitrile, or ester handle, respectively. For 33, nitrogen is retained and the Boc group is abstracted. No whole-molecule site recognition runs after a reaction: the remaining marked handle is carried by the product SMARTS. Additional production unary chemistry and halogen exchange remain separate curation work. See [the implementation review guide](codex/docs/linear-synthesis.md) for the state/action contract and current type inventory.

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

The selected HSX components, intentional differences, and backward approximation are documented in [the HSX port review](codex/docs/hsx-port.md).

Training metrics include `step_seconds` (rollout through optimizer/EMA, excluding checkpoint I/O) and `rollout_seconds` to distinguish inference/chemistry from the optimization phase.

Prepared environment loading trusts chemistry and values checked during preparation. It checks array schema/shape but does not reparse all SMILES, scan feature values, or decompress duplicate SMILES just to compare rows. Checkpoint environment identity still uses its content digest.

For Python sampling, use `RxnFlowSampler(checkpoint, reward=..., device=...)`. Configuration comes from the checkpoint; it is loaded once on CPU and only model weights are moved to the requested device. Reverse-search indices are created only when backward analysis is requested.
