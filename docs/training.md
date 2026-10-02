# Configuration, training and sampling

Set `data.env_dir` and `run.output_dir` in [qed.yaml](../configs/qed.yaml) or [qed_sa.yaml](../configs/qed_sa.yaml), then run the corresponding example from the [README](../README.md). All settings are listed in [template.yaml](../configs/template.yaml).

## Configuration

| Setting | Meaning |
| --- | --- |
| `generation.min_synthons` / `max_synthons` | Selected synthons, including the initial brick; defaults 2 / 3. |
| `generation.min_reactions` / `max_reactions` | Unary + binary reactions, excluding initial selection; defaults 1 / 3. |
| `data.max_atoms` | Strict heavy-atom capacity; default 50. |
| `property_penalty` | Optional upper bounds on additive state-plus-synthon estimates, e.g. `{mw: 500, hba: 10, hbd: 5}`. These do not guarantee exact final-product bounds. |
| `subsampling.sampling_ratio` / `min_sampling` | Library draw size: `min(N, max(min_sampling, floor(N * ratio)))`; defaults 0.05 / 50. Draws are shared across compatible states. |
| `training.retrosynthesis_workers` | Reverse-search processes; default 4, or 0 for synchronous execution. |
| `training.backward_synthon_penalty` | Each additional synthon divides reverse-path weight by this factor; default 100. Unary reactions do not add this penalty. |

There is no Stop action: a brick addition or terminal unary reaction consumes the final site. Reverse probabilities use bounded, approximate search. Subsampling and property penalties can leave no feasible continuation, producing an invalid attempt.

## Training and restart

Use the same configuration, reward implementation, package version and prepared environment. `--steps` specifies additional updates:

```bash
python examples/qed.py \
  --config configs/qed.yaml \
  --restart runs/qed/checkpoints/latest.ckpt \
  --steps 1000
```

## Outputs

| Path | Contents |
| --- | --- |
| `config.yaml` | Resolved configuration. |
| `checkpoints/latest.ckpt` | Model, optimizer, replay and RNG state for restart. Numbered checkpoints are saved every 500 updates by default. |
| `training.jsonl` | Metrics for every update. `log_every` controls console output only. |
| `samples/step_XXXXXX.jsonl` | All online attempts for each update, including invalid ones. |

- `reward` is before beta; `objective_rewards` contains the individual scores.
- `preferences` contains effective weights: sum N for `mul`, sum 1 for `sum`.
- `traj_lens` includes initial synthon selection. The compact `traj` omits that selection and records pre-reaction state, reaction name and added synthon.
- `batch_entropy` is mean negative trajectory log forward probability over online + replay data, not categorical action entropy.

## Sampling

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

sampler = RxnFlowSampler("runs/qed_sa/checkpoints/latest.ckpt", reward=QEDSAReward())
results = sampler.sample(100, beta=("fixed", [32.0]), seed=0)
sampler.write(results, "scored_samples.json")
```
