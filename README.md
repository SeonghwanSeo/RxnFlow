# RxnFlow

RxnFlow trains a generative flow network over an Enamine-derived synthon reaction space. Molecules are assembled through dynamically selected FirstBlock, UniReaction, and BiReaction actions using a local PyTorch and RDKit runtime.

The research project is described in [Generative Flows on Synthetic Pathway for Drug Design](https://arxiv.org/abs/2410.04542).

The synthon environment owns its chemistry and catalog preparation under `envs/`; model code and GFlowNet execution live in `models/` and `gflownet/`. See [project layout](codex/docs/project-layout.md) for the source map and dependency boundaries.

The [section-by-section reference comparison](codex/docs/reference-comparison.md) records which parts follow RxnFlow master, CGFlow-RxnFlow, HSX main, or HSX explore_250509, including remaining differences in features, model parameters, masking, and training defaults.

## Installation

RxnFlow is under development toward its initial `v1.0.0` release and requires Python 3.10. There is one current environment, config, and checkpoint format; regenerate older artifacts.

```bash
pip install -e '.[dev]'
```

The implementation uses native PyTorch tensors and does not require PyTorch Geometric or compiled scatter extensions.

## Prepare the Enamine synthon environment

Supply an Enamine stock file with one `SMILES<TAB>ID` record per line. Vendor records are local inputs and are not part of this repository.

```bash
rxnflow-prepare \
  --building-blocks /path/to/enamine_stock.smi \
  --template-dir data/templates \
  --env-dir /path/to/prepared/enamine \
  --num-workers 4 \
  --min-library-size 10
```

Preparation applies the 35 typed synthon conversions in `synthon.yaml`. One-site products become bricks and two-site products become linkers, including pairs of the same type. Protected handles count as sites. Both conversion orders are considered for linkers. Each linker is stored in both attachment orientations. The incoming site is marked `[*]` (isotope 0); its chemical type is the first component of the ordered library key `A-B`, and the other site retains isotope B. Bricks use the same marker with a single-type library key; FirstBlock restores that type in the state. Symmetry-equivalent orientations are deduplicated. Identical canonical synthons aggregate their Enamine IDs; different representations of the same source molecule remain distinct blocks.

`--min-library-size N` keeps libraries with at least N unique oriented synthon rows after all conversion batches are merged and deduplicated, before feature generation. The default is 1; use 10 for the full catalog. Multiple source IDs for the same synthon count as one row. The cutoff applies to both bricks and linkers. `convert_stage(..., min_library_size=N)` exposes the same option in Python. The conversion manifest records `min_library_size`, retained `block_counts`, and `excluded_block_counts`. Filtering changes the action space and environment signature; checkpoints trained on the unfiltered catalog must not be reused with the filtered environment.

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

Following HSX, building-block masks compare current state properties plus precomputed block properties against the configured upper bounds. Following `explore_250509`, select compatible reaction/library types, uniformly subsample each library, then apply the state + block budget mask to those sampled rows. Inclusion probabilities use the full library size and the original draw count, not the number surviving the mask. Rejected rows are not refilled; a sampled action space with no feasible continuation fails the trajectory. `property_penalty` therefore constrains an additive estimate, not exact product descriptors. Comparisons use raw units: nonzero bounds allow HSX main's 1% relative tolerance, while zero bounds and graph capacity remain strict. State and block MW use `Descriptors.ExactMolWt`. Enamine dummy labels have zero mass; HSX's Synple/eXplore At-isotope subtraction and `+29.0` linker MW correction do not apply.

Graph tensors have a fixed capacity of `max_atoms + 1` (heavy atoms plus a dummy slot). Rollout batches all active states; trajectory-balance training batches all observed transitions. Each policy call draws each needed library once and shares its sampled rows across states and reactions. Sampled block features are encoded together, and each reaction concatenates its compatible libraries for one score matrix over eligible states. Boolean budget masks set excluded logits to `-inf` while preserving sampled columns. Only selected indices become action objects. Repeated states reuse CPU graph construction and share graph encoding in evaluation mode; training keeps separate graph rows. Embeddings are not cached across updates. Full-library draws use their existing indices without randomness; partial draws sample the requested count without shuffling the full library.

Training follows RxnFlow master: the subsample estimates the denominator with `log(library_size / draw_count)` weights, and observed actions are scored separately even when absent from the draw. Observed actions do not alter the subsample or its inclusion weights. The resulting log probability is clamped at zero, as in the reference. Sampling uses device-side Gumbel draws and copies only selected indices to Python. Random exploration follows CGFlow's `-log(number_of_libraries * sampled_library_size)` offsets, with property masks retained; this balances libraries before masking rather than drawing uniformly over all surviving blocks.

The policy scores state/reaction embeddings and prepared block features without executing candidate reactions. Only the chosen action executes RDKit chemistry, checking the resulting site signature and actual `data.max_atoms` capacity. An invalid selected transition fails the trajectory without retry or unmasking. Unary actions use typed-handle and trajectory-length eligibility, without an additive block budget or exact product-property filter. The state holds the resulting RDKit `Mol`, shared with graph encoding; canonical SMILES identify and serialize states. Graph tensors reserve `max_atoms` RDKit heavy-atom slots plus one dummy slot, and molecules are never truncated.

The graph encoder uses residual GINE message passing in native PyTorch, with graph-mode normalization, conditional scale/shift and bidirectional virtual-node edges. Each layer sums ReLU(source + bond), adds the normalized target once (fixed epsilon=0), and applies a two-Linear H→2H→H MLP before the conditioned residual update. Explicit self loops and attention are absent. Atom features include chirality. The 13 bond features follow HSX main: four bond types, seven stereo categories (NONE/ANY/Z/E/CIS/TRANS/unknown), conjugation and ring membership. Molecular mean and virtual-node pooling give a `2 * hidden_dim` readout followed by LayerNorm. The block encoder projects fingerprints and properties with separate Linear/LayerNorm branches using HSX main's Xavier initialization, then combines them with a type embedding through the fusion MLP. HSX explore supplies the fusion/policy normalization order and post-GNN reaction conditioning; HSX main supplies the block-normalized dot score and bounded reaction temperature, initialized at 0.2. Graph/fusion/policy MLPs use Kaiming hidden weights and Xavier output weights, with zero biases; reaction and block-type embeddings use uniform[-0.1,0.1]. UniReaction uses a scalar head. Defaults are `hidden_dim=128`, `num_layers=4`, `block_dim=128`, `mlp_layers=2`, and `block_mlp_layers=1`. The backbone is an explicitly selected GINE variant; the reference 2H readout and late conditioning remain. Anchor readout is not added. The obsolete `num_heads` option has been removed. Checkpoints with the previous architecture or dimensions cannot be loaded into the new default model; existing MolWt-based prepared libraries need their feature stage rerun before use; block conversion is unchanged.

Training uses MSE trajectory balance, uniform FIFO replay, an EMA sampling model, and restartable checkpoints. Replay and checkpoints store trajectories as plain dictionaries with SMILES; only sampled replay trajectories reconstruct RDKit molecules. Failed selected reactions retain their forward probability and receive zero raw reward with the configured training reward floor. Policy gradients use global norm clipping at 100, excluding logZ. Default random action probability is 0.1, reward floor is 1e-4, and weight decay remains 1e-8. `training.log_z_learning_rate` controls logZ separately; `training.lr_decay_steps` is the learning-rate half-life. Optimizer, scheduler, library RNG and model-device RNG states are restored. Backward analysis preserves the generated route and adds forward-verified precursor routes within the reaction-depth bound. Its depth-weighted probabilities remain an approximation to the backward distribution. Reverse workers run alongside the next forward-policy computation; pending results are collected before extending their parent trees, including after the final reaction.

## Custom rewards

Rewards are explicit local Python objects implementing `RewardFunction`. Both `score` and `sample_filter` receive RDKit molecules directly; use `Chem.MolToSmiles(mol)` when strings are needed:

```python
from rxnflow.config import Config
from rdkit import Chem
from rxnflow.reward import RewardFunction
from rxnflow.trainer import RxnFlowTrainer


class CarbonReward(RewardFunction):
    def score(self, molecules: list[Chem.Mol]) -> list[float]:
        return [
            sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()) / 50
            for mol in molecules
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

Structured results contain dummy-free `smiles`, `trajectory`, `intermediates`, optional `reward`, and `metadata`. Each serialized trajectory step includes `product_smiles` from the transition result; the internal `Action` stores only the selected reaction and block. Block actions additionally record brick/linker role, catalog index, synthon SMILES, structured `block_ids`, and the corresponding source BB structures. `intermediates` contains every action product, including FirstBlock and the terminal product. There is no separate terminal `synthon_smiles`: the final internal state is already the output molecule.

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
