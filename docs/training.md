# Configuration, training and sampling

Choose [qed.yaml](../configs/qed.yaml) or [qed_sa.yaml](../configs/qed_sa.yaml), then set:

- `data.env_dir`: your prepared environment directory.
- `run.output_dir`: where training results will be saved.
- `run.device`: the examples use `cuda`; use `cpu` without a GPU, or `auto` to select automatically.

Run the corresponding example from the [README](../README.md). See [template.yaml](../configs/template.yaml) for all settings.

## Configuration

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

## Training and restart

To resume an interrupted run, use its saved configuration and checkpoint with the same reward example. Keep the prepared environment and package version unchanged. `--steps` specifies additional updates:

```bash
python examples/qed.py \
  --config runs/qed/config.yaml \
  --restart runs/qed/checkpoints/latest.ckpt \
  --steps 1000
```

## Outputs

Files are saved under `run.output_dir`. Invalid attempts, such as trajectories with no feasible continuation, are also recorded in the sample files.

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

Use your trained checkpoint and a beta value from its training range. This example uses the QED/SA run:

```bash
python scripts/sample.py \
  --checkpoint runs/qed_sa/checkpoints/latest.ckpt \
  --num-samples 100 \
  --beta 32 \
  --output samples.json
```

- Formats: `.smi` for SMILES, `.csv` for molecules and paths, `.json` for structured trajectories and provenance.
- Omitted preferences use the checkpoint setting. For a preference-conditioned model, add `--preferences "fixed(0.3,0.7)"`.
- `--sampling-temperature` is an additional softmax temperature (default 1), separate from beta.

The script does not evaluate rewards. To score samples, pass the reward explicitly:

```python
from examples.qed_sa import QEDSAReward
from rxnflow.sampler import RxnFlowSampler

reward = QEDSAReward()
sampler = RxnFlowSampler("runs/qed_sa/checkpoints/latest.ckpt", reward=reward)
results = sampler.sample(100, beta=("fixed", [32.0]), seed=0)
sampler.write(results, "scored_samples.json")
```
