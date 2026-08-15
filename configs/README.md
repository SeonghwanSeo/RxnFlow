# Configuration guide

RxnFlow merges each YAML file with its built-in defaults, so omitted fields do not need to be copied into an application configuration.

## Which file to use

- `qed.yaml` is the minimal application example. Normally, set `data.env_dir`, choose `run.output_dir`, and start training.
- `template.yaml` exposes every supported option. Copy it only when tuning the model, replay, optimizer, reward transformation, or sampling policy.

The bundled command `rxnflow-train` always uses `QEDReward`. A custom reward is selected explicitly in Python by passing a different `RewardFunction`. The top-level `reward` section contains both trajectory-balance scaling and the arguments for that selected reward:

```yaml
reward:
  exponent: 32.0
  floor: 0.0001
  settings: {}
```

`reward.exponent` and `reward.floor` control how numerical rewards enter the trajectory-balance loss. `reward.settings` is expanded as keyword arguments to the reward constructor. YAML never imports or selects a custom reward class.

## Settings to choose first

1. `data.env_dir` must point to a prepared eMolecules environment.
2. `run.output_dir` is where logs, resolved configuration, and checkpoints are written.
3. `reward.exponent` strongly affects objective optimization. The bundled QED example uses 32.
4. `data.max_atoms` is the maximum RDKit heavy-atom count. The default is 50.
5. `training.steps` and `training.batch_size` determine the main training budget. `training.replay_batch_size` adds replayed trajectories to each optimization step; reduce both batch sizes when compute or memory is limited.
6. `subsampling.sampling_ratio` controls the fraction of the block library considered per decision. Lower values reduce compute and increase estimator variance.
7. Leave `run.device: auto` unless a specific CPU or CUDA device is required.

## Optional molecular limits

The top-level `property_penalty` mapping applies hard upper bounds before block scoring. For example:

```yaml
property_penalty:
  mw: 500
  tpsa: 140
```

Supported properties are `mw`, `tpsa`, `hbd`, `hba`, `logp`, `rotatable_bonds`, `rings`, `aromatic_rings`, and `heavy_atoms`. These additive pre-reaction estimates are followed by exact product validation.

## Advanced settings

- `model` changes graph-transformer capacity.
- `subsampling.min_sampling` protects small price tiers; `importance_temp` controls the online sampling correction.
- `training.learning_rate` and `training.weight_decay` configure AdamW.
- `training.replay_batch_size` and `training.replay_capacity` configure replay. Set `replay_capacity: 0` to disable replay entirely.
- `reward.exponent` and `reward.floor` configure reward scaling.
- `reward.settings` contains keyword arguments for the explicitly selected reward constructor.
- `training.sampling_temperature` and `training.random_action_prob` configure trajectory exploration.
- `training.ema_decay` configures the sampling model update.
- `training.checkpoint_every` and `training.log_every` set output intervals.

At startup, RxnFlow writes a complete resolved configuration to `run.output_dir/config.yaml`. Use that file when restarting so the checkpoint configuration remains exact.

For a custom reward, keep its implementation and selection in Python:

```python
config = Config.from_file("config.yaml")
reward = CustomReward(**config.reward.settings)
trainer = RxnFlowTrainer(config, reward)
```
