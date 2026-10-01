# 10,000-record development environment

Production preparation remains deferred. This environment uses 10,000 rows drawn uniformly without replacement from all 297,705 rows of the local Enamine stock, with seed 0. Sampling is over source records, before chemistry filtering or synthon expansion.

## Reproduction

Run from the repository root on `gnode7`; both output paths must be new. The source catalog and generated environment remain local and are excluded from Git.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python codex/scripts/prepare_random_subset.py \
  --source data/building_blocks/enamine_stock.smi \
  --sample data/building_blocks/enamine_stock_random_10000_seed0.smi \
  --env-dir data/envs/enamine_random_10000_seed0 \
  --count 10000 --seed 0
```

`data/envs/enamine_random_10000_seed0/subset.json` records the source/sample/template hashes, original line numbers, seed, timing, and preparation memory. `prepare_manifest.json` records each library's row count.

| Measurement | Result |
| --- | ---: |
| Sampled input rows / unique source IDs | 10,000 / 10,000 |
| IDs retained after source cleaning | 9,992 |
| Source IDs represented by at least one synthon | 9,314 |
| Cleaned source IDs without a synthon | 678 |
| Brick rows | 26,924 |
| Oriented linker rows | 48,066 |
| Libraries | 853 |
| Sampling | 0.51 s |
| Synthon conversion | 98.13 s |
| Feature generation | 81.46 s |
| Peak preparation RSS | 183.18 MiB |
| Feature arrays before compression, including SMILES | 71.83 MiB |
| Compressed `bb_feature.npz` | 6.45 MiB |
| Whole prepared environment, including provenance | 10.35 MiB |

One source can yield multiple bricks and linkers, and duplicate synthons merge their source IDs. Catalog row counts therefore differ from the number of source BBs. Preparation retains converted bricks and linkers before runtime size/property masking; `data.max_atoms` and property bounds determine which actions are available during generation.

The 678 cleaned but unrepresented sources produced no accepted brick under the current 35 conversions; they remain in `building_blocks.json` for provenance. No catalog ID refers to a missing source. Conversion coverage is distinct from runtime action feasibility and experimental reaction success.

## Technical review

The environment/library, MDP/masking, model/features, reward/backward, training/restart, and output provenance paths were inspected in that order. Source/block reparsing during preparation was removed; synthetic catalog rows, source mappings and NPZ arrays matched the reference. Zero and negative property upper bounds are now accepted where meaningful. Checkpoint restoration keeps RNG states on CPU and restores dropout RNG as well as the explicit action generator.

State/product Mol ownership, terminal and last-step rules, direction-specific catalog lookup, exact product masks, fixed graph capacity, native PyTorch attention masks, conditional subsampling corrections, and known backward branches were retained. Bounded backward probabilities remain an intentional approximation. User code-review checkboxes in PLAN.md have not been marked complete by this technical review.

The existing graph features do not explicitly encode atom chirality or E/Z identity, and block fingerprints can collide. RDKit still preserves molecular stereochemistry. This feature limitation is recorded for subsequent research decisions; no new feature representation was introduced in this pass.

## Integration and pilot evaluation

The integration command reuses this environment and performs 10 training steps, one resumed update, and four valid samples:

```bash
OMP_NUM_THREADS=1 RXNFLOW_ENV_DIR=/home/shwan/Project/HSX/data/envs/enamine_random_10000_seed0 ./test.sh heavy
```

`codex/scripts/evaluate_subset.py` performs a small QED pilot and saves unfiltered rollout records before/after training. Its validity fraction counts failed attempts; evaluation does not retry until a requested number of valid molecules is reached. Morgan fingerprints are computed only for the offline diversity metric, not for policy scoring. Evaluation uses its own fixed seed and restores the training action RNG afterward.

The integration passed on gnode7 in 375.84 seconds. Synthetic quick verification passed 30 tests, including exact continuation of a dropout-enabled update after restart. The pilot command was:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python codex/scripts/evaluate_subset.py \
  --env-dir data/envs/enamine_random_10000_seed0 \
  --output-dir runs/enamine_random_10000_seed0_pilot \
  --steps 30 --count 64
```

The pilot used seed 0 for training and seed 11 for each evaluation, QED exponent 8, MW ≤ 500, max 50 heavy atoms, 1–3 reactions, uniform library subsampling at ratio 0.002/minimum 10, batch size 4 plus replay 4, and a 64-dimensional two-layer model. `device=auto` selected **CPU** on gnode7. The CUDA-specific RNG restore branch was not exercised by this run; the checkpoint code fix was reviewed and CPU/dropout continuation was verified.

| Metric over 64 raw rollouts | Before | After 30 updates |
| --- | ---: | ---: |
| Valid outputs | 35 | 60 |
| Valid fraction | 54.69% | 93.75% |
| Unique valid outputs | 35 | 60 |
| Mean QED among valid outputs | 0.5790 | 0.5887 |
| Mean QED over all attempts, failures = 0 | 0.3166 | 0.5519 |
| Mean pairwise Morgan Tanimoto distance | 0.8932 | 0.8845 |
| Valid 1 / 2 / 3-reaction routes | 24 / 3 / 8 | 50 / 10 / 0 |
| Rollout evaluation time | 86.11 s | 91.72 s |

Environment loading plus model initialization took 10.09 seconds; training took 462.68 seconds. CPU library tensors occupied 51.63 MiB. Peak RSS for the main pilot process was 673.41 MiB, excluding the reverse workers' process memory.

The increase in overall reward largely accompanies improved validity; QED conditional on validity changed little. Routes shifted toward shorter synthesis and no three-reaction route appeared among the 64 post-training attempts. This single-seed, 30-update pilot establishes execution and initial behavior, not convergence or reliable improvement in molecular quality. Every invalid rollout ended because the sampled action space had no feasible continuation; this does not prove that the complete unsampled action space had none.

Within 114 libraries, distinct block SMILES contained **531 excess duplicate feature rows**: sum of `number of rows − number of distinct (fingerprint, property) rows` within each library. These block choices receive identical embeddings in the current model. This count is not the number of all rows participating in duplicate groups, and the audit does not attribute every duplicate to stereochemistry or hashing. Catalog identity/source mapping remains separate and intact.

Full results are in `runs/enamine_random_10000_seed0_pilot/evaluation.json`; `before_trajectories.json` and `after_trajectories.json` preserve all valid and invalid trajectories, and `checkpoint_00000030.pt` retains the trained state. Production/full-stock preparation, longer multi-seed training, the explicit stereochemistry feature decision, and chemical curation remain outside this pilot.


## Subsequent masking revision

The pilot above predates the HSX budget-mask migration. Current policy now follows the verified HSX explore ordering: sample the compatible block library, apply state + block property masks to those sampled rows, then execute only the selected reaction. The intermediate mask-first implementation was superseded on 2026-10-02. These earlier validity and timing results must not be interpreted as measurements of the revised policy. Prepared descriptors remain usable: they already omit dummy-isotope mass and contain no Synple-specific +29 MW correction.

최신 HSX 선택적 이식의 baseline/revised 비교는 [HSX 이식 기록](hsx-port.md)에 별도로 기록한다. 기존 pilot 결과와 변경 후 결과를 혼용하지 않는다.
