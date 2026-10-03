# Configuration, training and sampling

Set `data.env_dir` in [qed.yaml](../configs/qed.yaml) or [qed_sa.yaml](../configs/qed_sa.yaml). Pass execution settings to the training script:

```bash
python examples/qed.py --config configs/qed.yaml \
  --output-dir runs/qed --steps 1000 --seed 1
```

- Omit `--device` to use CUDA when available, otherwise CPU; choose explicitly with `--device cpu` or `--device cuda`.
- `--steps` is required and specifies the number of updates for this invocation. In Python, use `trainer.run(steps)`; step count is not stored in the configuration.
- In Python, pass `output_dir`, `device` and `seed` to `RxnFlowTrainer`. The trainer and sampler accept a string or `torch.device` and default to CPU.
- See [template.yaml](../configs/template.yaml) for all model and training settings.

## Configuration

Defaults below apply when a setting is omitted. Values explicitly set in your YAML, including the complete template, override them.

| Setting | Meaning |
| --- | --- |
| `generation.min_synthons` / `max_synthons` | Selected synthons, including the initial brick; defaults 2 / 3. |
| `generation.min_reactions` / `max_reactions` | Unary + binary reactions, excluding initial selection; defaults 1 / 3. |
| `data.max_atoms` | Heavy-atom capacity of generated states; default 50. Independent of preparation’s `--max-atoms` (default 30 per completed synthon, excluding dummies). |
| `property_penalty` | Optional upper bounds on additive state-plus-synthon estimates, e.g. `{mw: 500, hba: 10, hbd: 5}`. These do not guarantee exact final-product bounds. |
| `subsampling.sampling_ratio` / `min_sampling` | Defaults to 5% of each library, with at least 50 candidates when available. Larger values cover more candidates at higher computation cost. |
| `training.num_online` / `num_replay` | New trajectories and previously generated trajectories reused per update; defaults 64 / 64. Set `num_replay: 0` to use only new trajectories. |
| `training.learning_rate` / `learning_rate_logZ` | Policy and logZ learning rates; defaults `1e-4` / `1e-3`. |
| `training.retrosynthesis_workers` | Reverse-search processes; default 4, or 0 for synchronous execution. |

## Resume training

To resume an interrupted run, use its saved configuration and checkpoint with the same reward example, and choose a new output directory. Keep the prepared environment and package version unchanged. The checkpoint restores RNG state; the supplied seed only initializes new runs. `--steps` specifies additional updates. In Python, call `trainer.run(steps, resume_from_checkpoint=path)`:

```bash
python examples/qed.py \
  --config runs/qed/config.yaml \
  --output-dir runs/qed_resumed \
  --resume-from-checkpoint runs/qed/checkpoints/latest.ckpt \
  --steps 1000
```

## Outputs

Files are saved under the trainer’s `output_dir`, which must not already exist. Resumed runs write only their new updates into the new directory. Invalid attempts, such as trajectories with no feasible continuation, are also recorded in the sample files.

| Path | Contents |
| --- | --- |
| `config.yaml` | Resolved configuration. |
| `checkpoints/latest.ckpt` | Model, optimizer, replay and RNG state for restart. Numbered checkpoints are saved every 500 updates by default. |
| `training.jsonl` | Metrics for every update. `log_every` controls console output only. |
| `samples/step_XXXXXX.jsonl` | All online attempts for each update, including invalid ones. |

- `reward` is before beta; `objective_rewards` contains the individual scores.
- `preferences` contains effective weights: they sum to the number of objectives for `mul`, or to one for `sum`.
- `traj_lens` includes initial synthon selection. The compact `traj` omits that selection and records pre-reaction state, reaction name and added synthon.
- `batch_entropy` is mean negative trajectory log forward probability over online + replay data, not categorical action entropy.

## Sampling

Use your trained checkpoint; omitted beta and preferences reuse their training settings. This example uses the QED/SA run:

```bash
python scripts/sample.py \
  --checkpoint runs/qed_sa/checkpoints/latest.ckpt \
  --num-samples 100 \
  --output samples.csv
```

- Sampling prints the training reward configuration and the requested sampling settings before generating molecules.
- Omitted `--beta` uses the training value or distribution. To select a fixed exponent, pass `--beta 32`.
- Fixed-beta training requires the same fixed beta at sampling. For uniform-beta training, use a fixed value or uniform subrange within the training range.
- Fixed-preference training requires the same relative weights; varying preferences require a model trained with varying preferences.
- `--env-dir` selects another prepared building-block catalog. Its reaction, synthon and exclusion definitions must match training; BB identities and library sizes may differ. Types defined in the templates but unused during training may have untrained embeddings.
- `--batch-size` controls trajectories generated per batch (default 64), independently of the training batch size. Duplicate molecules are retained.
- Formats: `.smi` for SMILES, `.csv` for molecules and paths, `.json` for structured trajectories and provenance.
- Omitted preferences use the checkpoint setting. For a preference-conditioned model, add `--preferences "fixed(0.3,0.7)"`.
- `--softmax-temperature` is an additional softmax temperature (default 1), separate from beta.

JSON records contain `smiles`, `traj` and `metadata`; metadata holds `beta` and `preferences`. CSV columns are `smiles`, `traj`, `beta` and `preferences`; `traj` and `preferences` are JSON strings. Sampling trajectories include the initial synthon selection and each action’s product. SMILES output contains only SMILES.

Sampling does not evaluate rewards. Evaluate generated molecules separately when needed:

```python
from rdkit import Chem

from examples.qed_sa import QEDSAReward
from rxnflow.sampler import RxnFlowSampler

sampler = RxnFlowSampler("runs/qed_sa/checkpoints/latest.ckpt")
results = sampler.sample(100, beta=("fixed", [32.0]), seed=0)
mols: list[Chem.Mol] = [Chem.MolFromSmiles(result.smiles) for result in results]
reward = QEDSAReward()
objective_rewards = reward(mols)
```

## Extract a sampling model

Extraction is optional: `--checkpoint` accepts either a full training checkpoint or an extracted model. Extract once to avoid loading optimizer and replay data on each sampling run:

```bash
python scripts/extract_model.py \
  --checkpoint runs/qed_sa/checkpoints/latest.ckpt \
  --output runs/qed_sa/model.pt
```

Use `--checkpoint runs/qed_sa/model.pt` in the sampling command. The file contains the configuration, EMA model weights and metadata needed for sampling. Use the configured environment path or select a compatible catalog with `--env-dir`. Extracted models cannot resume training.
